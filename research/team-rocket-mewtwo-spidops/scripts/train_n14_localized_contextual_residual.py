#!/usr/bin/env python3
"""Train the single preregistered N14 localized contextual residual."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
ZERO_PARITY = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_parity.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/training_v1"
PROTOCOL = "N14_LOCALIZED_CONTEXTUAL_RESIDUAL_PROTOCOL.md"

LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_MODEL_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
ZERO_NET_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
ZERO_PARITY_SHA = "c8638bd6d5a879d32eb88bba30f7ddeb334d222e9b295c64f0ca2619096611d5"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
SEED = 814001
STEPS = 2_000
BATCH = 256
THRESHOLDS = (12, 24, 36, 48)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ZERO))

import train_n12_counterfactual_margin_adapter as support  # noqa: E402
from train_live_outcome_awr import (  # noqa: E402
    collate_like_live_agent,
    deployment_logits,
)

# The shared support module prepends the live candidate for its legacy imports.
# N14 must load the residual-capable package when ``soldier`` is imported in
# ``main``; put it back at the front after all support imports are complete.
sys.path = [entry for entry in sys.path if Path(entry).resolve() != ZERO.resolve()]
sys.path.insert(0, str(ZERO))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_adapter(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    if model.logit_adapter is None or int(model.adapter_rank) != 16:
        raise ValueError("N14 residual architecture changed")
    parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("logit_adapter."))
        if parameter.requires_grad:
            parameters.append(parameter)
    if sum(parameter.numel() for parameter in parameters) != 6_928:
        raise ValueError("N14 adapter parameter count changed")
    return parameters


@torch.no_grad()
def attach_contextual_reference(
    model: torch.nn.Module,
    examples: Sequence[support.N12Example],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    mismatches = 0
    final_error = 0.0
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start:start + batch_size])
        batch, availability = collate_like_live_agent(selected, torch.device("cpu"))
        captured: list[torch.Tensor] = []
        hook = model.logit_head.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            logits, _ = model.forward_batch(batch)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError(f"N14 contextual hook count changed: {len(captured)}")
        logits = deployment_logits(logits, availability, selected)
        predictions = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)
        embeddings = captured[0]
        for row, (example, prediction) in enumerate(zip(selected, predictions.tolist())):
            width = int(example.feat["opt_feats"].shape[0])
            example.reference_top = int(prediction)
            example.reference_probabilities = (
                probabilities[row, :width].cpu().numpy().astype(np.float32)
            )
            legal = logits[row, :width].clone()
            legal[example.reference_top] = -torch.inf
            example.reference_margin = max(
                0.0, float(logits[row, example.reference_top] - legal.max())
            )
            example.contextual_options = embeddings[row, :width].cpu().numpy().astype(np.float32)
            example.reference_logits = logits[row, :width].cpu().numpy().astype(np.float32)
            mismatches += int(prediction != example.action)
            residual = model.logit_adapter(embeddings[row, :width]).squeeze(-1)
            final_error = max(final_error, float(residual.abs().max()))
    return {
        "examples": len(examples),
        "reference_action_mismatches": mismatches,
        "maximum_initial_residual": final_error,
    }


def cached_logits(
    adapter: torch.nn.Module,
    examples: Sequence[support.N12Example],
    device: torch.device,
) -> torch.Tensor:
    if not examples:
        raise ValueError("N14 cached adapter batch is empty")
    width = max(int(example.feat["opt_feats"].shape[0]) for example in examples)
    embeddings = torch.zeros(
        len(examples), width, 384, dtype=torch.float32, device=device
    )
    baseline = torch.full(
        (len(examples), width), -1.0e9, dtype=torch.float32, device=device
    )
    for row, example in enumerate(examples):
        if not hasattr(example, "contextual_options") or not hasattr(example, "reference_logits"):
            raise ValueError("N14 contextual cache is absent")
        count = int(example.feat["opt_feats"].shape[0])
        embeddings[row, :count] = torch.as_tensor(
            example.contextual_options, dtype=torch.float32, device=device
        )
        baseline[row, :count] = torch.as_tensor(
            example.reference_logits, dtype=torch.float32, device=device
        )
    return baseline + adapter(embeddings).squeeze(-1)


def relative_displacement(
    adapter: torch.nn.Module,
    initial: dict[str, torch.Tensor],
) -> torch.Tensor:
    numerator = None
    denominator = 0.0
    for name, parameter in adapter.named_parameters():
        delta = parameter - initial[name].to(parameter.device)
        term = (delta * delta).sum()
        numerator = term if numerator is None else numerator + term
        denominator += float((initial[name] ** 2).sum())
    if numerator is None or denominator <= 0.0:
        raise ValueError("N14 adapter displacement is undefined")
    return numerator / denominator


@torch.no_grad()
def correction_fit(
    adapter: torch.nn.Module,
    examples: Sequence[support.N12Example],
) -> dict[str, Any]:
    adapter.eval()
    logits = cached_logits(adapter, examples, torch.device("cpu"))
    predictions = logits.argmax(dim=1).tolist()
    cells: dict[str, int] = {}
    for example, prediction in zip(examples, predictions):
        cells.setdefault(example.cell, 0)
        cells[example.cell] += int(prediction == example.teacher_best)
    return {
        "examples": len(examples),
        "fitted": sum(cells.values()),
        "by_cell": dict(sorted(cells.items())),
    }


@torch.no_grad()
def preservation_metrics(
    adapter: torch.nn.Module,
    examples: Sequence[support.N12Example],
    *,
    batch_size: int = 256,
) -> dict[str, Any]:
    adapter.eval()
    total = retained = 0
    kl_total = 0.0
    grouped: dict[str, dict[str, int]] = {}
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start:start + batch_size])
        logits = cached_logits(adapter, selected, torch.device("cpu"))
        reference = support.pad_reference(selected, logits.shape[1], logits.device)
        row_kl = (
            reference
            * (torch.log(reference.clamp_min(1.0e-12)) - torch.log_softmax(logits, dim=1))
        ).sum(dim=1)
        predictions = logits.argmax(dim=1).tolist()
        for row, (example, prediction) in enumerate(zip(selected, predictions)):
            keep = int(prediction == example.reference_top)
            total += 1
            retained += keep
            kl_total += float(row_kl[row])
            group = grouped.setdefault(example.group, {"examples": 0, "retained": 0})
            group["examples"] += 1
            group["retained"] += keep
    return {
        "examples": total,
        "retained": retained,
        "top1_retention": retained / max(1, total),
        "mean_forward_kl": kl_total / max(1, total),
        "by_group": {
            name: {
                **row,
                "top1_retention": row["retained"] / max(1, row["examples"]),
            }
            for name, row in sorted(grouped.items())
        },
    }


@torch.no_grad()
def full_cached_parity(
    model: torch.nn.Module,
    examples: Sequence[support.N12Example],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    mismatches = 0
    maximum = 0.0
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start:start + batch_size])
        batch, availability = collate_like_live_agent(selected, torch.device("cpu"))
        full, _ = model.forward_batch(batch)
        full = deployment_logits(full, availability, selected)
        cached = cached_logits(model.logit_adapter, selected, torch.device("cpu"))
        mismatches += int((full.argmax(dim=1) != cached.argmax(dim=1)).sum())
        for row, example in enumerate(selected):
            width = int(example.feat["opt_feats"].shape[0])
            anchor = int(full[row, :width].argmax())
            left = full[row, :width] - full[row, anchor]
            right = cached[row, :width] - cached[row, anchor]
            finite = (left > -1.0e8) & (right > -1.0e8)
            if bool(finite.any()):
                maximum = max(maximum, float((left[finite] - right[finite]).abs().max()))
    return {
        "examples": len(examples),
        "top_action_mismatches": mismatches,
        "max_finite_relative_logit_error": maximum,
    }


def changed_keys(reference: torch.nn.Module, candidate: torch.nn.Module) -> list[str]:
    before = reference.state_dict()
    return sorted(
        name for name, value in candidate.state_dict().items()
        if not torch.equal(value, before[name])
    )


def admission(
    threshold: int,
    fit: dict[str, Any],
    n6: dict[str, Any],
    n9: dict[str, Any],
    n6_parity: dict[str, Any],
    n9_parity: dict[str, Any],
    keys: list[str],
) -> dict[str, Any]:
    gates = {
        "named_fit_threshold": int(fit["fitted"]) >= threshold,
        "adapter_keys_only": bool(keys) and all(name.startswith("logit_adapter.") for name in keys),
        "n6_retention_ge_0_98": float(n6["top1_retention"]) >= 0.98,
        "n6_kl_le_0_05": float(n6["mean_forward_kl"]) <= 0.05,
        "n9_retention_ge_0_99": float(n9["top1_retention"]) >= 0.99,
        "n9_each_group_ge_0_985": all(
            float(row["top1_retention"]) >= 0.985 for row in n9["by_group"].values()
        ),
        "n9_kl_le_0_02": float(n9["mean_forward_kl"]) <= 0.02,
        "n6_runtime_cache_parity": (
            int(n6_parity["top_action_mismatches"]) == 0
            and float(n6_parity["max_finite_relative_logit_error"]) <= 2.0e-5
        ),
        "n9_runtime_cache_parity": (
            int(n9_parity["top_action_mismatches"]) == 0
            and float(n9_parity["max_finite_relative_logit_error"]) <= 2.0e-5
        ),
    }
    return {"passes": all(gates.values()), "gates": gates}


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N14 trainer accepts no command-line overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite N14 training: {OUTPUT}")
    checks = {
        ZERO / "model.pt": ZERO_MODEL_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": ZERO_NET_SHA,
        ZERO_PARITY: ZERO_PARITY_SHA,
        LIVE / "model.pt": LIVE_MODEL_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N14 frozen input changed: {path}")
    parity = json.loads(ZERO_PARITY.read_text(encoding="utf-8"))
    for name in ("n6", "n9"):
        if parity[name]["live_zero_top_mismatch"] or parity[name]["max_finite_relative_logit_error"]:
            raise ValueError("N14 zero-base parity is not exact")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cpu")
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    deck = [int(value) for value in (ZERO / "deck.csv").read_text().replace(",", "\n").split()]
    library = json.loads((ZERO / "deck_library.json").read_text(encoding="utf-8"))
    model = load_policy(ZERO / "model.pt", map_location=device).to(device).eval()
    zero_reference = load_policy(ZERO / "model.pt", map_location=device).to(device).eval()
    trainable = configure_adapter(model)
    for parameter in zero_reference.parameters():
        parameter.requires_grad_(False)

    n6_all, n6_provenance = support.load_n6_examples(support.CORPUS, support.AUDIT)
    n9_all, n9_provenance = support.load_n9_examples(
        LIVE.resolve(), support.REPLAYS, support.PARITY, deck, library, Belief, features
    )
    n6_reference = attach_contextual_reference(model, n6_all)
    n9_reference = attach_contextual_reference(model, n9_all)
    for name, row in (("n6", n6_reference), ("n9", n9_reference)):
        if row["reference_action_mismatches"] or row["maximum_initial_residual"] != 0.0:
            raise ValueError(f"N14 initial contextual parity failed: {name}={row}")
    corrections = [example for example in n6_all if example.correction]
    n6_anchors = [example for example in n6_all if not example.correction]
    if len(corrections) != 48 or len(n6_anchors) != 591 or len(n9_all) != 6_753:
        raise ValueError("N14 support counts changed")
    weights = support.correction_cell_weights(corrections).to(device)
    initial_adapter = {
        name: value.detach().clone() for name, value in model.logit_adapter.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        trainable, lr=0.001, weight_decay=0.0001
    )
    order_size = STEPS * BATCH
    n6_order = support.repeated_order(len(n6_anchors), order_size, SEED + 101)
    n9_order = support.repeated_order(len(n9_all), order_size, SEED + 202)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    checkpoint_steps: dict[int, int] = {}
    checkpoint_paths: dict[int, Path] = {}
    history = []
    started = monotonic()
    for step in range(1, STEPS + 1):
        begin = (step - 1) * BATCH
        end = step * BATCH
        n6_batch = [n6_anchors[index] for index in n6_order[begin:end]]
        n9_batch = [n9_all[index] for index in n9_order[begin:end]]
        model.logit_adapter.train()
        correction_logits = cached_logits(model.logit_adapter, corrections, device)
        n6_logits = cached_logits(model.logit_adapter, n6_batch, device)
        n9_logits = cached_logits(model.logit_adapter, n9_batch, device)
        components = {
            "correction": support.correction_loss(
                correction_logits,
                corrections,
                margin=0.10,
                temperature=0.25,
                weights=weights,
            ),
            "n6_kl": support.forward_kl(n6_logits, n6_batch),
            "n9_kl": support.forward_kl(n9_logits, n9_batch),
            "n6_retention": support.retention_loss(
                n6_logits, n6_batch, cap=0.25, temperature=0.25
            ),
            "n9_retention": support.retention_loss(
                n9_logits, n9_batch, cap=0.25, temperature=0.25
            ),
            "displacement": relative_displacement(model.logit_adapter, initial_adapter),
        }
        loss = (
            components["correction"]
            + 2.0 * components["n6_kl"]
            + 2.0 * components["n9_kl"]
            + components["n6_retention"]
            + components["n9_retention"]
            + 0.001 * components["displacement"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        fit = correction_fit(model.logit_adapter, corrections)
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "correction_fit": fit,
            "components": {name: float(value.detach()) for name, value in components.items()},
        }
        history.append(record)
        for threshold in THRESHOLDS:
            if threshold not in checkpoint_steps and int(fit["fitted"]) >= threshold:
                path = OUTPUT / f"fit{threshold:02d}.pt"
                save_policy(model, path)
                checkpoint_steps[threshold] = step
                checkpoint_paths[threshold] = path
        if step == 1 or step % 100 == 0 or any(value == step for value in checkpoint_steps.values()):
            print(json.dumps(record, sort_keys=True), flush=True)

    evaluations: dict[str, Any] = {}
    seen_hashes: dict[str, str] = {}
    for threshold in THRESHOLDS:
        if threshold not in checkpoint_paths:
            continue
        path = checkpoint_paths[threshold]
        artifact_hash = digest(path)
        duplicate_of = seen_hashes.get(artifact_hash)
        if duplicate_of is None:
            seen_hashes[artifact_hash] = f"fit{threshold:02d}"
        candidate = load_policy(path, map_location=device).to(device).eval()
        keys = changed_keys(zero_reference, candidate)
        fit = correction_fit(candidate.logit_adapter, corrections)
        n6_metrics = preservation_metrics(candidate.logit_adapter, n6_anchors)
        n9_metrics = preservation_metrics(candidate.logit_adapter, n9_all)
        n6_runtime = full_cached_parity(candidate, n6_all)
        n9_runtime = full_cached_parity(candidate, n9_all)
        gate = admission(
            threshold, fit, n6_metrics, n9_metrics, n6_runtime, n9_runtime, keys
        )
        if duplicate_of is not None:
            gate["passes"] = False
            gate["gates"]["distinct_checkpoint"] = False
        else:
            gate["gates"]["distinct_checkpoint"] = True
        evaluations[f"fit{threshold:02d}"] = {
            "threshold": threshold,
            "step": checkpoint_steps[threshold],
            "path": str(path),
            "sha256": artifact_hash,
            "duplicate_of": duplicate_of,
            "changed_state_keys": keys,
            "correction_fit": fit,
            "n6_preservation": n6_metrics,
            "n9_preservation": n9_metrics,
            "n6_runtime_cache_parity": n6_runtime,
            "n9_runtime_cache_parity": n9_runtime,
            "admission": gate,
        }
    admitted = [name for name, row in evaluations.items() if row["admission"]["passes"]]
    report = {
        "schema": "rocket-n14-localized-contextual-residual-training/v1",
        "protocol": PROTOCOL,
        "protocol_sha256": digest(ROOT / PROTOCOL),
        "live_model_sha256": LIVE_MODEL_SHA,
        "zero_model_sha256": ZERO_MODEL_SHA,
        "zero_net_v10_sha256": ZERO_NET_SHA,
        "zero_parity_sha256": ZERO_PARITY_SHA,
        "deck_sha256": DECK_SHA,
        "hyperparameters": {
            "rank": 16,
            "scale": 1.0,
            "steps": STEPS,
            "anchor_batch_size": BATCH,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_cap": 1.0,
            "seed": SEED,
            "threads": 4,
            "correction_margin": 0.10,
            "softplus_temperature": 0.25,
            "n6_kl_coefficient": 2.0,
            "n9_kl_coefficient": 2.0,
            "n6_retention_coefficient": 1.0,
            "n9_retention_coefficient": 1.0,
            "displacement_coefficient": 0.001,
            "thresholds": list(THRESHOLDS),
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "n6_provenance": n6_provenance,
        "n9_provenance": n9_provenance,
        "reference_parity": {"n6": n6_reference, "n9": n9_reference},
        "history": history,
        "evaluations": evaluations,
        "admitted": admitted,
        "elapsed_seconds": monotonic() - started,
        "claims": {
            "development_panel_authority": bool(admitted),
            "deployment_authority": False,
            "strength_improvement": False,
        },
    }
    report_path = OUTPUT / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "admitted": admitted}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
