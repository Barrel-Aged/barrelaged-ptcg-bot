#!/usr/bin/env python3
"""Train and audit the sole terminal N17 replicated-target residual."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from time import monotonic
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
ZERO = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/zero_base"
ZERO_PARITY = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/zero_parity.json"
TARGET_AUDIT = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/target_audit.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/training_v1"
PROTOCOL = ROOT / "N17_REPLICATED_TARGET_RESIDUAL_PROTOCOL.md"

LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_MODEL_SHA = "de4679c136af57af9dc6818b3100c1e10d74ac5c32d615ae02019c897ec07a6c"
ZERO_NET_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
ZERO_PARITY_SHA = "6268b01da89e73ee3e1fbf21d9141100b86bd7a503a09eeeaf1f4ecdcea692f2"
TARGET_AUDIT_SHA = "c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c"
PROTOCOL_SHA = "0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"

TRAIN_SEED = 817002
STEPS = 4_000
ANCHOR_BATCH = 384
STRICT_MARGIN = 0.10
EXPANDED_MARGIN = 0.05
TEMPERATURE = 0.25

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ZERO))

import train_n12_counterfactual_margin_adapter as support  # noqa: E402
from train_live_outcome_awr import (  # noqa: E402
    collate_like_live_agent,
    deployment_logits,
)

# The shared support module prepends the live bundle for legacy imports. N17
# must import the residual-capable package in main, so restore it to the front.
sys.path = [entry for entry in sys.path if Path(entry).resolve() != ZERO.resolve()]
sys.path.insert(0, str(ZERO))


EXPECTED_STRICT_CELLS = {
    f"{support.GROUPS[0]}:turn_2": 11,
    f"{support.GROUPS[0]}:turn_4": 17,
    f"{support.GROUPS[1]}:turn_2": 11,
    f"{support.GROUPS[1]}:turn_4": 9,
}
EXPECTED_EXPANDED_CELLS = {
    f"{support.GROUPS[0]}:turn_2": 34,
    f"{support.GROUPS[0]}:turn_4": 21,
    f"{support.GROUPS[1]}:turn_2": 35,
    f"{support.GROUPS[1]}:turn_4": 32,
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_adapter(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    if model.logit_adapter is None or int(model.adapter_rank) != 32:
        raise ValueError("N17 residual architecture changed")
    parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("logit_adapter."))
        if parameter.requires_grad:
            parameters.append(parameter)
    if sum(parameter.numel() for parameter in parameters) != 13_088:
        raise ValueError("N17 adapter parameter count changed")
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
            raise ValueError(f"N17 contextual hook count changed: {len(captured)}")
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
        raise ValueError("N17 cached adapter batch is empty")
    width = max(int(example.feat["opt_feats"].shape[0]) for example in examples)
    embeddings = torch.zeros(
        len(examples), width, 384, dtype=torch.float32, device=device
    )
    baseline = torch.full(
        (len(examples), width), -1.0e9, dtype=torch.float32, device=device
    )
    for row, example in enumerate(examples):
        if not hasattr(example, "contextual_options") or not hasattr(example, "reference_logits"):
            raise ValueError("N17 contextual cache is absent")
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
    initial: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    numerator = None
    denominator = 0.0
    for name, parameter in adapter.named_parameters():
        delta = parameter - initial[name].to(parameter.device)
        term = (delta * delta).sum()
        numerator = term if numerator is None else numerator + term
        denominator += float((initial[name] ** 2).sum())
    if numerator is None or denominator <= 0.0:
        raise ValueError("N17 adapter displacement is undefined")
    return numerator / denominator


def _fingerprint(example: support.N12Example) -> str:
    prefix = "n6r:"
    if not example.game.startswith(prefix):
        raise ValueError(f"N17 unexpected N6R example id: {example.game}")
    return example.game[len(prefix):]


def partition_targets(
    examples: Sequence[support.N12Example],
    audit: Mapping[str, Any],
) -> tuple[list[support.N12Example], list[support.N12Example], list[support.N12Example]]:
    if audit.get("schema") != "rocket-n17-replicated-counterfactual-target-audit/v1":
        raise ValueError("N17 target-audit schema changed")
    if int(audit.get("source_roots", -1)) != 639 or int(audit.get("union_targets", -1)) != 170:
        raise ValueError("N17 target-audit support counts changed")
    records = {str(row["observation_fingerprint"]): row for row in audit["targets"]}
    if len(records) != 170:
        raise ValueError("N17 target-audit fingerprints are not unique")
    strict: list[support.N12Example] = []
    expanded: list[support.N12Example] = []
    anchors: list[support.N12Example] = []
    seen: set[str] = set()
    for example in examples:
        fingerprint = _fingerprint(example)
        target = records.get(fingerprint)
        if target is None:
            if example.correction:
                raise ValueError("N17 strict correction missing from union target audit")
            anchors.append(example)
            continue
        seen.add(fingerprint)
        expected_cell = f"{target['opponent']}:turn_{int(target['turn'])}"
        if (
            int(target["baseline"]) != int(example.action)
            or str(target["opponent"]) != example.group
            or expected_cell != example.cell
        ):
            raise ValueError(f"N17 target/example identity changed: {fingerprint}")
        teacher = int(target["teacher_best"])
        if teacher == int(example.action):
            raise ValueError(f"N17 target does not change the baseline: {fingerprint}")
        example.teacher_best = teacher
        is_strict = bool(target["strict_admissible"])
        if is_strict != bool(example.correction):
            raise ValueError(f"N17 strict-target flag changed: {fingerprint}")
        if is_strict:
            strict.append(example)
        else:
            if not bool(target["replicated_half_best"]):
                raise ValueError(f"N17 expanded target lost replicated support: {fingerprint}")
            expanded.append(example)
    if seen != set(records):
        raise ValueError("N17 target audit includes an absent N6R fingerprint")
    if len(strict) != 48 or len(expanded) != 122 or len(anchors) != 469:
        raise ValueError(
            f"N17 partition changed: strict={len(strict)} expanded={len(expanded)} "
            f"anchors={len(anchors)}"
        )
    return strict, expanded, anchors


def balanced_cell_weights(
    examples: Sequence[support.N12Example],
    expected_counts: Mapping[str, int],
) -> torch.Tensor:
    counts = Counter(example.cell for example in examples)
    if dict(counts) != dict(expected_counts):
        raise ValueError(f"N17 target cells changed: {dict(counts)}")
    weights = np.asarray(
        [1.0 / (len(counts) * counts[example.cell]) for example in examples],
        dtype=np.float32,
    )
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-7):
        raise AssertionError("N17 target weights do not sum to one")
    for cell in counts:
        mask = np.asarray([example.cell == cell for example in examples])
        if not np.isclose(weights[mask].sum(), 0.25, atol=1.0e-7):
            raise AssertionError(f"N17 cell loss mass changed: {cell}")
    return torch.from_numpy(weights)


@torch.no_grad()
def fit_metrics(
    adapter: torch.nn.Module,
    examples: Sequence[support.N12Example],
) -> dict[str, Any]:
    adapter.eval()
    logits = cached_logits(adapter, examples, torch.device("cpu"))
    predictions = logits.argmax(dim=1).tolist()
    cells: dict[str, Counter[str]] = {}
    total_fitted = 0
    for example, prediction in zip(examples, predictions):
        fitted = int(prediction == example.teacher_best)
        total_fitted += fitted
        row = cells.setdefault(example.cell, Counter())
        row["examples"] += 1
        row["fitted"] += fitted
    return {
        "examples": len(examples),
        "fitted": total_fitted,
        "fit_rate": total_fitted / max(1, len(examples)),
        "by_cell": {
            name: {
                "examples": int(row["examples"]),
                "fitted": int(row["fitted"]),
                "fit_rate": row["fitted"] / max(1, row["examples"]),
            }
            for name, row in sorted(cells.items())
        },
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
    grouped: dict[str, Counter[str]] = {}
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
            group = grouped.setdefault(example.group, Counter())
            group["examples"] += 1
            group["retained"] += keep
    return {
        "examples": total,
        "retained": retained,
        "top1_retention": retained / max(1, total),
        "mean_forward_kl": kl_total / max(1, total),
        "by_group": {
            name: {
                "examples": int(row["examples"]),
                "retained": int(row["retained"]),
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
    after = candidate.state_dict()
    if set(before) != set(after):
        raise ValueError("N17 serialized model state keys changed")
    return sorted(name for name, value in after.items() if not torch.equal(value, before[name]))


def serialization_audit(
    in_memory: torch.nn.Module,
    reloaded: torch.nn.Module,
) -> dict[str, Any]:
    left = in_memory.state_dict()
    right = reloaded.state_dict()
    exact = set(left) == set(right) and all(torch.equal(value, right[name]) for name, value in left.items())
    return {"state_keys": len(left), "tensor_exact": bool(exact)}


def runtime_asset_hashes(root: Path, *, output: bool = False) -> dict[str, str]:
    excluded = {
        "model.pt",
        "N17_ZERO_BASE_PROVENANCE.json",
        "N17_TRAINING_PROVENANCE.json",
        "N17_TRAINING_REPORT.json",
    }
    hashes = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        hashes[path.relative_to(root).as_posix()] = digest(path)
    if not hashes:
        raise ValueError(f"N17 runtime asset set is empty: {root}")
    return hashes


def admission(
    strict_fit: Mapping[str, Any],
    expanded_fit: Mapping[str, Any],
    n6: Mapping[str, Any],
    n9: Mapping[str, Any],
    n6_parity: Mapping[str, Any],
    n9_parity: Mapping[str, Any],
    keys: Sequence[str],
    serialization: Mapping[str, Any],
    assets_exact: bool,
) -> dict[str, Any]:
    gates = {
        "strict_fit_48_of_48": (
            int(strict_fit["examples"]) == 48 and int(strict_fit["fitted"]) == 48
        ),
        "expanded_fit_ge_110_of_122": (
            int(expanded_fit["examples"]) == 122 and int(expanded_fit["fitted"]) >= 110
        ),
        "expanded_each_cell_fit_ge_0_85": all(
            float(row["fit_rate"]) >= 0.85 for row in expanded_fit["by_cell"].values()
        ) and set(expanded_fit["by_cell"]) == set(EXPECTED_EXPANDED_CELLS),
        "adapter_keys_only": bool(keys) and all(name.startswith("logit_adapter.") for name in keys),
        "n6_retention_ge_0_98": float(n6["top1_retention"]) >= 0.98,
        "n6_kl_le_0_05": float(n6["mean_forward_kl"]) <= 0.05,
        "n9_retention_ge_0_99": float(n9["top1_retention"]) >= 0.99,
        "n9_each_group_ge_0_985": all(
            float(row["top1_retention"]) >= 0.985 for row in n9["by_group"].values()
        ) and set(n9["by_group"]) == set(support.GROUPS),
        "n9_kl_le_0_02": float(n9["mean_forward_kl"]) <= 0.02,
        "n6_runtime_cache_parity": (
            int(n6_parity["top_action_mismatches"]) == 0
            and float(n6_parity["max_finite_relative_logit_error"]) <= 2.0e-5
        ),
        "n9_runtime_cache_parity": (
            int(n9_parity["top_action_mismatches"]) == 0
            and float(n9_parity["max_finite_relative_logit_error"]) <= 2.0e-5
        ),
        "serialization_tensor_exact": bool(serialization["tensor_exact"]),
        "deck_and_runtime_assets_exact": bool(assets_exact),
    }
    return {"passes": all(gates.values()), "gates": gates}


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N17 trainer accepts no command-line overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite N17 training: {OUTPUT}")
    checks = {
        LIVE / "model.pt": LIVE_MODEL_SHA,
        ZERO / "model.pt": ZERO_MODEL_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": ZERO_NET_SHA,
        ZERO_PARITY: ZERO_PARITY_SHA,
        TARGET_AUDIT: TARGET_AUDIT_SHA,
        PROTOCOL: PROTOCOL_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N17 frozen training input changed: {path}")
    parity = json.loads(ZERO_PARITY.read_text(encoding="utf-8"))
    for name in ("n6", "n9"):
        if parity[name]["live_zero_top_mismatch"] or parity[name]["max_finite_relative_logit_error"]:
            raise ValueError("N17 zero-base parity is not exact")

    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    torch.manual_seed(TRAIN_SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
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
            raise ValueError(f"N17 initial contextual parity failed: {name}={row}")

    target_audit = json.loads(TARGET_AUDIT.read_text(encoding="utf-8"))
    strict, expanded, n6_anchors = partition_targets(n6_all, target_audit)
    if len(n9_all) != 6_753:
        raise ValueError("N17 N9 support count changed")
    strict_weights = balanced_cell_weights(strict, EXPECTED_STRICT_CELLS).to(device)
    expanded_weights = balanced_cell_weights(expanded, EXPECTED_EXPANDED_CELLS).to(device)

    initial_adapter = {
        name: value.detach().clone() for name, value in model.logit_adapter.named_parameters()
    }
    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.0001)
    order_size = STEPS * ANCHOR_BATCH
    n6_order = support.repeated_order(len(n6_anchors), order_size, TRAIN_SEED + 101)
    n9_order = support.repeated_order(len(n9_all), order_size, TRAIN_SEED + 202)
    history = []
    started = monotonic()
    for step in range(1, STEPS + 1):
        begin = (step - 1) * ANCHOR_BATCH
        end = step * ANCHOR_BATCH
        n6_batch = [n6_anchors[index] for index in n6_order[begin:end]]
        n9_batch = [n9_all[index] for index in n9_order[begin:end]]
        model.logit_adapter.train()
        strict_logits = cached_logits(model.logit_adapter, strict, device)
        expanded_logits = cached_logits(model.logit_adapter, expanded, device)
        n6_logits = cached_logits(model.logit_adapter, n6_batch, device)
        n9_logits = cached_logits(model.logit_adapter, n9_batch, device)
        components = {
            "strict_correction": support.correction_loss(
                strict_logits,
                strict,
                margin=STRICT_MARGIN,
                temperature=TEMPERATURE,
                weights=strict_weights,
            ),
            "expanded_correction": support.correction_loss(
                expanded_logits,
                expanded,
                margin=EXPANDED_MARGIN,
                temperature=TEMPERATURE,
                weights=expanded_weights,
            ),
            "n6_kl": support.forward_kl(n6_logits, n6_batch),
            "n9_kl": support.forward_kl(n9_logits, n9_batch),
            "n6_retention": support.retention_loss(
                n6_logits, n6_batch, cap=0.25, temperature=TEMPERATURE
            ),
            "n9_retention": support.retention_loss(
                n9_logits, n9_batch, cap=0.25, temperature=TEMPERATURE
            ),
            "displacement": relative_displacement(model.logit_adapter, initial_adapter),
        }
        loss = (
            components["strict_correction"]
            + components["expanded_correction"]
            + 3.0 * components["n6_kl"]
            + 3.0 * components["n9_kl"]
            + 1.5 * components["n6_retention"]
            + 1.5 * components["n9_retention"]
            + 0.001 * components["displacement"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == STEPS:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "strict_fit": fit_metrics(model.logit_adapter, strict),
                "expanded_fit": fit_metrics(model.logit_adapter, expanded),
                "components": {name: float(value.detach()) for name, value in components.items()},
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    terminal_strict = fit_metrics(model.logit_adapter, strict)
    terminal_expanded = fit_metrics(model.logit_adapter, expanded)
    terminal_n6 = preservation_metrics(model.logit_adapter, n6_anchors)
    terminal_n9 = preservation_metrics(model.logit_adapter, n9_all)

    shutil.copytree(
        ZERO,
        OUTPUT,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", "N17_ZERO_BASE_PROVENANCE.json"
        ),
    )
    save_policy(model, OUTPUT / "model.pt")
    candidate = load_policy(OUTPUT / "model.pt", map_location=device).to(device).eval()
    serialization = serialization_audit(model, candidate)
    keys = changed_keys(zero_reference, candidate)
    source_assets = runtime_asset_hashes(ZERO)
    candidate_assets = runtime_asset_hashes(OUTPUT, output=True)
    assets_exact = source_assets == candidate_assets
    n6_runtime = full_cached_parity(candidate, n6_all)
    n9_runtime = full_cached_parity(candidate, n9_all)
    gate = admission(
        terminal_strict,
        terminal_expanded,
        terminal_n6,
        terminal_n9,
        n6_runtime,
        n9_runtime,
        keys,
        serialization,
        assets_exact,
    )
    model_sha = digest(OUTPUT / "model.pt")
    report = {
        "schema": "rocket-n17-replicated-target-residual-training/v1",
        "protocol": "rocket-n17-replicated-target-residual/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "target_audit_sha256": TARGET_AUDIT_SHA,
        "live_model_sha256": LIVE_MODEL_SHA,
        "zero_model_sha256": ZERO_MODEL_SHA,
        "zero_net_v10_sha256": ZERO_NET_SHA,
        "zero_parity_sha256": ZERO_PARITY_SHA,
        "deck_sha256": DECK_SHA,
        "terminal_model_sha256": model_sha,
        "hyperparameters": {
            "rank": 32,
            "scale": 1.0,
            "steps": STEPS,
            "anchor_batch_size": ANCHOR_BATCH,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_cap": 1.0,
            "training_seed": TRAIN_SEED,
            "n6_order_seed": TRAIN_SEED + 101,
            "n9_order_seed": TRAIN_SEED + 202,
            "threads": 4,
            "strict_margin": STRICT_MARGIN,
            "expanded_margin": EXPANDED_MARGIN,
            "softplus_temperature": TEMPERATURE,
            "strict_correction_coefficient": 1.0,
            "expanded_correction_coefficient": 1.0,
            "n6_kl_coefficient": 3.0,
            "n9_kl_coefficient": 3.0,
            "n6_retention_coefficient": 1.5,
            "n9_retention_coefficient": 1.5,
            "retention_cap": 0.25,
            "displacement_coefficient": 0.001,
            "terminal_only": True,
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "support": {
            "strict_targets": len(strict),
            "expanded_only_targets": len(expanded),
            "n6_non_target_anchors": len(n6_anchors),
            "n9_anchors": len(n9_all),
            "strict_cells": EXPECTED_STRICT_CELLS,
            "expanded_cells": EXPECTED_EXPANDED_CELLS,
        },
        "n6_provenance": n6_provenance,
        "n9_provenance": n9_provenance,
        "reference_parity": {"n6": n6_reference, "n9": n9_reference},
        "history": history,
        "terminal_evaluation": {
            "strict_fit": terminal_strict,
            "expanded_fit": terminal_expanded,
            "n6_preservation": terminal_n6,
            "n9_preservation": terminal_n9,
            "n6_runtime_cache_parity": n6_runtime,
            "n9_runtime_cache_parity": n9_runtime,
            "changed_state_keys": keys,
            "serialization": serialization,
            "runtime_assets": {
                "exact": assets_exact,
                "files": len(source_assets),
            },
            "admission": gate,
        },
        "elapsed_seconds": monotonic() - started,
        "claims": {
            "development_panel_authority": bool(gate["passes"]),
            "deployment_authority": False,
            "strength_improvement": False,
        },
    }
    report_path = OUTPUT / "N17_TRAINING_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "schema": "rocket-n17-terminal-candidate/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "target_audit_sha256": TARGET_AUDIT_SHA,
        "source_zero_model_sha256": ZERO_MODEL_SHA,
        "model_sha256": model_sha,
        "deck_sha256": DECK_SHA,
        "net_v10_sha256": digest(OUTPUT / "soldier/net_v10.py"),
        "training_report_sha256": digest(report_path),
        "terminal_step": STEPS,
        "offline_admitted": bool(gate["passes"]),
        "claims": {
            "development_panel_authority": bool(gate["passes"]),
            "deployment_authority": False,
            "strength_improvement": False,
        },
    }
    (OUTPUT / "N17_TRAINING_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "model_sha256": model_sha,
                "offline_admitted": bool(gate["passes"]),
                "failed_gates": [name for name, passed in gate["gates"].items() if not passed],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
