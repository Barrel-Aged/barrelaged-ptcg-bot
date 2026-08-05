#!/usr/bin/env python3
"""Replay N14 exactly and evaluate its sole prospective step-2000 terminal state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
N14_OUTPUT = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/training_v1"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/training_v1"
PROTOCOL = ROOT / "N15_TERMINAL_CONSOLIDATION_PROTOCOL.md"
N14_PROTOCOL = ROOT / "N14_LOCALIZED_CONTEXTUAL_RESIDUAL_PROTOCOL.md"
N14_RESULT = ROOT / "N14_LOCALIZED_CONTEXTUAL_RESIDUAL_RESULT.md"

DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_MODEL_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
ZERO_NET_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
ZERO_PARITY_SHA = "c8638bd6d5a879d32eb88bba30f7ddeb334d222e9b295c64f0ca2619096611d5"
N14_PROTOCOL_SHA = "2cbf0206af920072705421a349b7d13476766a1eff61dbd0494844bd0ed6821d"
N14_REPORT_SHA = "84ce98182af34919ba753a72be3d71242b31b0d290f16c29c4b77f7729511581"
N14_RESULT_SHA = "20efbba7c2775dc5c1d0a767bbb64cd0d950d4cd28e864e70f91e3ba01ea9627"

SEED = 814001
STEPS = 2_000
BATCH = 256
PREFIXES = {
    62: (
        "fit12.pt",
        "e498bac097594901e35ca5f4b57b83ea4b343bb21561abc91cd08fe267ffab0b",
        12,
    ),
    90: (
        "fit24.pt",
        "56a62c22a59f6f9710fee04173172eb9080ea0003b57e71b8d8f2fc5a4ac8c63",
        24,
    ),
    149: (
        "fit36.pt",
        "bea9104a72154c5a609ab0e06d193e3c48468781ca22af8cd219c6eb43571c49",
        37,
    ),
    286: (
        "fit48.pt",
        "0a11204beb06645334647211ca17af978023c864ac74367649e4d7d8451a361f",
        48,
    ),
}

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train_n14_localized_contextual_residual as n14  # noqa: E402

# N14's imports intentionally manipulate ``sys.path``. Restore the residual
# package as the leading soldier implementation before importing it in main.
sys.path = [entry for entry in sys.path if Path(entry).resolve() != ZERO.resolve()]
sys.path.insert(0, str(ZERO))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_state_dicts(
    candidate: torch.nn.Module,
    reference: torch.nn.Module,
) -> dict[str, Any]:
    left = candidate.state_dict()
    right = reference.state_dict()
    missing = sorted(set(right) - set(left))
    unexpected = sorted(set(left) - set(right))
    unequal = sorted(
        name for name in set(left).intersection(right)
        if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
    )
    return {
        "passes": not missing and not unexpected and not unequal,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "unequal_keys": unequal,
        "compared_keys": len(set(left).intersection(right)),
    }


def terminal_admission(
    fit: dict[str, Any],
    n6: dict[str, Any],
    n9: dict[str, Any],
    n6_parity: dict[str, Any],
    n9_parity: dict[str, Any],
    keys: list[str],
    prefix_proofs: dict[str, Any],
    serialization: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "all_prefixes_exact": (
            len(prefix_proofs) == len(PREFIXES)
            and all(row["state_equality"]["passes"] for row in prefix_proofs.values())
        ),
        "fit_48_of_48": int(fit["fitted"]) == 48,
        "each_correction_cell_nonempty": (
            len(fit["by_cell"]) == 4 and all(int(value) > 0 for value in fit["by_cell"].values())
        ),
        "adapter_keys_only": bool(keys) and all(
            name.startswith("logit_adapter.") for name in keys
        ),
        "n6_retention_ge_0_98": float(n6["top1_retention"]) >= 0.98,
        "n6_kl_le_0_05": float(n6["mean_forward_kl"]) <= 0.05,
        "n9_retention_ge_0_99": float(n9["top1_retention"]) >= 0.99,
        "n9_each_group_ge_0_985": all(
            float(row["top1_retention"]) >= 0.985
            for row in n9["by_group"].values()
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
        "serialization_roundtrip_exact": bool(serialization["passes"]),
    }
    return {"passes": all(gates.values()), "gates": gates}


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N15 trainer accepts no command-line overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite N15 training: {OUTPUT}")

    checks = {
        ZERO / "model.pt": ZERO_MODEL_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": ZERO_NET_SHA,
        n14.ZERO_PARITY: ZERO_PARITY_SHA,
        n14.LIVE / "model.pt": LIVE_MODEL_SHA,
        N14_PROTOCOL: N14_PROTOCOL_SHA,
        N14_OUTPUT / "training_report.json": N14_REPORT_SHA,
        N14_RESULT: N14_RESULT_SHA,
    }
    for step, (filename, expected, _fit) in PREFIXES.items():
        del step
        checks[N14_OUTPUT / filename] = expected
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N15 frozen input changed: {path}")

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
    trainable = n14.configure_adapter(model)
    for parameter in zero_reference.parameters():
        parameter.requires_grad_(False)

    n6_all, n6_provenance = n14.support.load_n6_examples(
        n14.support.CORPUS, n14.support.AUDIT
    )
    n9_all, n9_provenance = n14.support.load_n9_examples(
        n14.LIVE.resolve(),
        n14.support.REPLAYS,
        n14.support.PARITY,
        deck,
        library,
        Belief,
        features,
    )
    n6_reference = n14.attach_contextual_reference(model, n6_all)
    n9_reference = n14.attach_contextual_reference(model, n9_all)
    for name, row in (("n6", n6_reference), ("n9", n9_reference)):
        if row["reference_action_mismatches"] or row["maximum_initial_residual"] != 0.0:
            raise ValueError(f"N15 initial contextual parity failed: {name}={row}")

    corrections = [example for example in n6_all if example.correction]
    n6_anchors = [example for example in n6_all if not example.correction]
    if len(corrections) != 48 or len(n6_anchors) != 591 or len(n9_all) != 6_753:
        raise ValueError("N15 support counts changed")
    weights = n14.support.correction_cell_weights(corrections).to(device)
    initial_adapter = {
        name: value.detach().clone()
        for name, value in model.logit_adapter.named_parameters()
    }
    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.0001)
    order_size = STEPS * BATCH
    n6_order = n14.support.repeated_order(len(n6_anchors), order_size, SEED + 101)
    n9_order = n14.support.repeated_order(len(n9_all), order_size, SEED + 202)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    prefix_proofs: dict[str, Any] = {}
    history = []
    started = monotonic()
    for step in range(1, STEPS + 1):
        begin = (step - 1) * BATCH
        end = step * BATCH
        n6_batch = [n6_anchors[index] for index in n6_order[begin:end]]
        n9_batch = [n9_all[index] for index in n9_order[begin:end]]
        model.logit_adapter.train()
        correction_logits = n14.cached_logits(model.logit_adapter, corrections, device)
        n6_logits = n14.cached_logits(model.logit_adapter, n6_batch, device)
        n9_logits = n14.cached_logits(model.logit_adapter, n9_batch, device)
        components = {
            "correction": n14.support.correction_loss(
                correction_logits,
                corrections,
                margin=0.10,
                temperature=0.25,
                weights=weights,
            ),
            "n6_kl": n14.support.forward_kl(n6_logits, n6_batch),
            "n9_kl": n14.support.forward_kl(n9_logits, n9_batch),
            "n6_retention": n14.support.retention_loss(
                n6_logits, n6_batch, cap=0.25, temperature=0.25
            ),
            "n9_retention": n14.support.retention_loss(
                n9_logits, n9_batch, cap=0.25, temperature=0.25
            ),
            "displacement": n14.relative_displacement(
                model.logit_adapter, initial_adapter
            ),
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
        fit = n14.correction_fit(model.logit_adapter, corrections)
        record = {
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "correction_fit": fit,
            "components": {
                name: float(value.detach()) for name, value in components.items()
            },
        }
        history.append(record)

        if step in PREFIXES:
            filename, expected_hash, expected_fit = PREFIXES[step]
            archived = load_policy(
                N14_OUTPUT / filename, map_location=device
            ).to(device).eval()
            equality = compare_state_dicts(model, archived)
            proof = {
                "step": step,
                "archive": str(N14_OUTPUT / filename),
                "archive_sha256": expected_hash,
                "expected_correction_fit": expected_fit,
                "observed_correction_fit": int(fit["fitted"]),
                "state_equality": equality,
            }
            prefix_proofs[str(step)] = proof
            if not equality["passes"] or int(fit["fitted"]) != expected_fit:
                failure = {
                    "schema": "rocket-n15-prefix-failure/v1",
                    "protocol_sha256": digest(PROTOCOL),
                    "prefix_proofs": prefix_proofs,
                    "claims": {
                        "offline_admission": False,
                        "development_panel_authority": False,
                        "deployment_authority": False,
                        "strength_improvement": False,
                    },
                }
                (OUTPUT / "prefix_failure.json").write_text(
                    json.dumps(failure, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(f"N15 exact replay failed at step {step}: {proof}")

        if step == 1 or step % 100 == 0 or step in PREFIXES:
            print(json.dumps(record, sort_keys=True), flush=True)

    terminal_path = OUTPUT / "terminal2000.pt"
    save_policy(model, terminal_path)
    candidate = load_policy(terminal_path, map_location=device).to(device).eval()
    serialization = compare_state_dicts(model, candidate)
    keys = n14.changed_keys(zero_reference, candidate)
    fit = n14.correction_fit(candidate.logit_adapter, corrections)
    n6_metrics = n14.preservation_metrics(candidate.logit_adapter, n6_anchors)
    n9_metrics = n14.preservation_metrics(candidate.logit_adapter, n9_all)
    n6_runtime = n14.full_cached_parity(candidate, n6_all)
    n9_runtime = n14.full_cached_parity(candidate, n9_all)
    gate = terminal_admission(
        fit,
        n6_metrics,
        n9_metrics,
        n6_runtime,
        n9_runtime,
        keys,
        prefix_proofs,
        serialization,
    )
    report = {
        "schema": "rocket-n15-prospective-terminal-consolidation-training/v1",
        "protocol": PROTOCOL.name,
        "protocol_sha256": digest(PROTOCOL),
        "deck_sha256": DECK_SHA,
        "live_model_sha256": LIVE_MODEL_SHA,
        "zero_model_sha256": ZERO_MODEL_SHA,
        "n14_protocol_sha256": N14_PROTOCOL_SHA,
        "n14_training_report_sha256": N14_REPORT_SHA,
        "n14_result_sha256": N14_RESULT_SHA,
        "hyperparameters": {
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
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "n6_provenance": n6_provenance,
        "n9_provenance": n9_provenance,
        "reference_parity": {"n6": n6_reference, "n9": n9_reference},
        "prefix_proofs": prefix_proofs,
        "history": history,
        "terminal": {
            "step": STEPS,
            "path": str(terminal_path),
            "sha256": digest(terminal_path),
            "serialization_roundtrip": serialization,
            "changed_state_keys": keys,
            "correction_fit": fit,
            "n6_preservation": n6_metrics,
            "n9_preservation": n9_metrics,
            "n6_runtime_cache_parity": n6_runtime,
            "n9_runtime_cache_parity": n9_runtime,
            "admission": gate,
        },
        "elapsed_seconds": monotonic() - started,
        "claims": {
            "offline_admission": bool(gate["passes"]),
            "development_panel_authority": bool(gate["passes"]),
            "deployment_authority": False,
            "strength_improvement": False,
        },
    }
    report_path = OUTPUT / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "terminal_sha256": report["terminal"]["sha256"],
                "offline_admission": gate["passes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
