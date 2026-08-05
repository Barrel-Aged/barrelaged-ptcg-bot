#!/usr/bin/env python3
"""Construct and audit the sole N25 complementary parameter composition."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
N15 = (
    ROOT
    / "artifacts/team_rocket_recovery/n15_terminal_consolidation/training_v1/terminal2000.pt"
)
N15_REPORT = (
    ROOT
    / "artifacts/team_rocket_recovery/n15_terminal_consolidation/training_v1/training_report.json"
)
N8B = (
    ROOT
    / "artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_arms/delta_100.pt"
)
N8B_CONTROL = (
    ROOT
    / "artifacts/team_rocket_recovery/candidates/scale_value_calibrated_v1/model.pt"
)
N8B_ARMS = (
    ROOT
    / "artifacts/team_rocket_recovery/n8b_recovery_logit_delta/development_arms/arms.json"
)
PROTOCOL = ROOT / "N25_COMPLEMENTARY_SAFE_COMPOSITION_PROTOCOL.md"
SOURCE_CORRECTION = ROOT / "N25_SOURCE_BASE_CORRECTION.md"
OUTPUT = (
    ROOT / "artifacts/team_rocket_recovery/n25_complementary_safe_composition/build_v1"
)

DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
LIVE_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
N15_SHA = "354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7"
N15_REPORT_SHA = "b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1"
N8B_SHA = "1ef2a93c982072d8fca105c0b80686885338b94077a98d5227163e6a4d62fede"
N8B_CONTROL_SHA = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
N8B_ARMS_SHA = "3eccbc2b4d10138dac66391e5ac83e18a4e3d951c920adef0fbfb7d14aebccd2"
PROTOCOL_SHA = "82ba2a57736e396ecda5c2fdcda1a9577d68be25de88f4c2ddb4f098d86e78a0"
SOURCE_CORRECTION_SHA = "aadd14c675e44a641b9cb0022c111aa578e516206b4f1b556fc918ff5a3d9e33"
HEAD_KEYS = ("logit_head.bias", "logit_head.weight")


sys.path.insert(0, str(ROOT / "scripts"))
import build_n16_minimum_scale_projection as n16  # noqa: E402

n14 = n16.n14


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or "state_dict" not in value or "config" not in value:
        raise ValueError(f"N25 unsupported model artifact: {path}")
    return value


def changed_keys(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> list[str]:
    if set(left) != set(right):
        raise ValueError("N25 source state dictionaries differ")
    return sorted(key for key in left if not torch.equal(left[key], right[key]))


def composition_audit(
    live: Mapping[str, torch.Tensor],
    control: Mapping[str, torch.Tensor],
    n8b: Mapping[str, torch.Tensor],
    n15: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    control_live_keys = changed_keys(live, control)
    source_delta_keys = changed_keys(control, n8b)
    live_control_head_exact = {
        key: torch.equal(live[key], control[key]) for key in HEAD_KEYS
    }
    non_head_changes = []
    head_errors = {}
    for key in n15:
        if key in HEAD_KEYS:
            expected = (
                n15[key].to(torch.float32)
                + n8b[key].to(torch.float32)
                - control[key].to(torch.float32)
            ).to(n15[key].dtype)
            head_errors[key] = float((candidate[key] - expected).abs().max())
        elif not torch.equal(candidate[key], n15[key]):
            non_head_changes.append(key)
    passes = (
        bool(control_live_keys)
        and all(key.startswith("value.") for key in control_live_keys)
        and
        tuple(source_delta_keys) == HEAD_KEYS
        and all(live_control_head_exact.values())
        and not non_head_changes
        and all(value == 0.0 for value in head_errors.values())
    )
    return {
        "passes": passes,
        "control_live_changed_keys": control_live_keys,
        "control_live_value_only": bool(control_live_keys)
        and all(key.startswith("value.") for key in control_live_keys),
        "live_control_head_exact": live_control_head_exact,
        "source_delta_keys": source_delta_keys,
        "non_head_changes_vs_n15": non_head_changes,
        "head_max_abs_error": head_errors,
    }


@torch.no_grad()
def full_metrics(
    model: torch.nn.Module,
    examples: Sequence[Any],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    total = retained = 0
    kl_total = 0.0
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"examples": 0, "retained": 0}
    )
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start : start + batch_size])
        batch, availability = n14.collate_like_live_agent(
            selected, torch.device("cpu")
        )
        logits, _ = model.forward_batch(batch)
        logits = n14.deployment_logits(logits, availability, selected)
        reference = n14.support.pad_reference(
            selected, logits.shape[1], torch.device("cpu")
        )
        row_kl = (
            reference
            * (
                torch.log(reference.clamp_min(1.0e-12))
                - torch.log_softmax(logits, dim=1)
            )
        ).sum(dim=1)
        predictions = logits.argmax(dim=1).tolist()
        for row, (example, prediction) in enumerate(
            zip(selected, predictions, strict=True)
        ):
            keep = int(prediction == int(example.reference_top))
            total += 1
            retained += keep
            kl_total += float(row_kl[row])
            group = grouped[str(example.group)]
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
def teacher_metrics(
    model: torch.nn.Module,
    examples: Sequence[Any],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    fitted = 0
    minimum = float("inf")
    margins = []
    cells: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"examples": 0, "fitted": 0, "minimum_margin": float("inf")}
    )
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start : start + batch_size])
        batch, availability = n14.collate_like_live_agent(
            selected, torch.device("cpu")
        )
        logits, _ = model.forward_batch(batch)
        logits = n14.deployment_logits(logits, availability, selected)
        for row, example in enumerate(selected):
            width = int(example.feat["opt_feats"].shape[0])
            values = logits[row, :width].clone()
            teacher = int(example.teacher_best)
            teacher_value = values[teacher].clone()
            values[teacher] = -torch.inf
            margin = float(teacher_value - values.max())
            fit = int(int(logits[row, :width].argmax()) == teacher)
            fitted += fit
            minimum = min(minimum, margin)
            margins.append(margin)
            cell = cells[str(example.cell)]
            cell["examples"] = int(cell["examples"]) + 1
            cell["fitted"] = int(cell["fitted"]) + fit
            cell["minimum_margin"] = min(float(cell["minimum_margin"]), margin)
    return {
        "examples": len(examples),
        "fitted": fitted,
        "minimum_margin": minimum,
        "mean_margin": sum(margins) / len(margins),
        "by_cell": dict(sorted(cells.items())),
    }


def roundtrip_audit(
    before: torch.nn.Module, after: torch.nn.Module
) -> dict[str, Any]:
    left = before.state_dict()
    right = after.state_dict()
    missing = sorted(set(left) - set(right))
    unexpected = sorted(set(right) - set(left))
    unequal = sorted(
        key
        for key in set(left).intersection(right)
        if not torch.equal(left[key], right[key])
    )
    return {
        "passes": not missing and not unexpected and not unequal,
        "missing": missing,
        "unexpected": unexpected,
        "unequal": unequal,
    }


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N25 builder accepts no overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N25 build: {OUTPUT}")
    checks = {
        ZERO / "model.pt": ZERO_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": RUNTIME_SHA,
        LIVE / "model.pt": LIVE_SHA,
        LIVE / "deck.csv": DECK_SHA,
        N15: N15_SHA,
        N15_REPORT: N15_REPORT_SHA,
        N8B_CONTROL: N8B_CONTROL_SHA,
        N8B: N8B_SHA,
        N8B_ARMS: N8B_ARMS_SHA,
        PROTOCOL: PROTOCOL_SHA,
        SOURCE_CORRECTION: SOURCE_CORRECTION_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N25 frozen input changed: {path}")
    n15_report = json.loads(N15_REPORT.read_text(encoding="utf-8"))
    if not n15_report["terminal"]["admission"]["passes"]:
        raise ValueError("N25 N15 source lacked offline admission")

    torch.set_num_threads(4)
    device = torch.device("cpu")
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    live_raw = load_raw(LIVE / "model.pt")
    control_raw = load_raw(N8B_CONTROL)
    n8b_raw = load_raw(N8B)
    n15_raw = load_raw(N15)
    if not live_raw["config"] == control_raw["config"] == n8b_raw["config"]:
        raise ValueError("N25 live/N8B source model configurations differ")
    expected_n15_config = dict(live_raw["config"])
    expected_n15_config.update(adapter_rank=16, adapter_scale=1.0)
    if n15_raw["config"] != expected_n15_config:
        raise ValueError("N25 N15 configuration is not the exact adapter extension")
    candidate = load_policy(N15, map_location=device).to(device).eval()
    with torch.no_grad():
        parameters = dict(candidate.named_parameters())
        for key in HEAD_KEYS:
            parameters[key].copy_(
                n15_raw["state_dict"][key].to(torch.float32)
                + n8b_raw["state_dict"][key].to(torch.float32)
                - control_raw["state_dict"][key].to(torch.float32)
            )
    composition = composition_audit(
        live_raw["state_dict"],
        control_raw["state_dict"],
        n8b_raw["state_dict"],
        n15_raw["state_dict"],
        candidate.state_dict(),
    )
    if not composition["passes"]:
        raise AssertionError(f"N25 composition failed: {composition}")

    deck = [
        int(value)
        for value in (ZERO / "deck.csv").read_text().replace(",", "\n").split()
    ]
    library = json.loads((ZERO / "deck_library.json").read_text(encoding="utf-8"))
    zero = load_policy(ZERO / "model.pt", map_location=device).to(device).eval()
    n6_all, n6_provenance = n14.support.load_n6_examples(
        n14.support.CORPUS, n14.support.AUDIT
    )
    n9_all, n9_provenance = n14.support.load_n9_examples(
        LIVE.resolve(),
        n14.support.REPLAYS,
        n14.support.PARITY,
        deck,
        library,
        Belief,
        features,
    )
    n6_reference = n14.attach_contextual_reference(zero, n6_all)
    n9_reference = n14.attach_contextual_reference(zero, n9_all)
    for name, row in (("n6", n6_reference), ("n9", n9_reference)):
        if row["reference_action_mismatches"] or row["maximum_initial_residual"]:
            raise ValueError(f"N25 reference parity failed: {name}={row}")
    corrections = [example for example in n6_all if example.correction]
    n6_anchors = [example for example in n6_all if not example.correction]
    if len(corrections) != 48 or len(n6_anchors) != 591 or len(n9_all) != 6_753:
        raise ValueError("N25 support counts changed")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    model_path = OUTPUT / "model.pt"
    save_policy(candidate, model_path)
    reloaded = load_policy(model_path, map_location=device).to(device).eval()
    serialization = roundtrip_audit(candidate, reloaded)
    composition_reload = composition_audit(
        live_raw["state_dict"],
        control_raw["state_dict"],
        n8b_raw["state_dict"],
        n15_raw["state_dict"],
        reloaded.state_dict(),
    )
    changed = n14.changed_keys(zero, reloaded)
    teacher = teacher_metrics(reloaded, corrections)
    n6 = full_metrics(reloaded, n6_anchors)
    n9 = full_metrics(reloaded, n9_all)
    allowed = set(HEAD_KEYS)
    gates = {
        "composition_exact": bool(composition_reload["passes"]),
        "serialization_exact": bool(serialization["passes"]),
        "changed_keys_allowed": bool(changed)
        and all(key.startswith("logit_adapter.") or key in allowed for key in changed),
        "teacher_fit_48": int(teacher["fitted"]) == 48,
        "teacher_minimum_margin_ge_0_25": float(teacher["minimum_margin"]) >= 0.25,
        "n6_retention_ge_0_99": float(n6["top1_retention"]) >= 0.99,
        "n6_each_group_ge_0_985": all(
            float(row["top1_retention"]) >= 0.985 for row in n6["by_group"].values()
        ),
        "n6_kl_le_0_02": float(n6["mean_forward_kl"]) <= 0.02,
        "n9_retention_ge_0_98": float(n9["top1_retention"]) >= 0.98,
        "n9_each_group_ge_0_98": all(
            float(row["top1_retention"]) >= 0.98 for row in n9["by_group"].values()
        ),
        "n9_kl_le_0_01": float(n9["mean_forward_kl"]) <= 0.01,
    }
    admitted = all(gates.values())
    report = {
        "schema": "rocket-n25-complementary-safe-composition-build/v1",
        "protocol": "rocket-n25-complementary-safe-composition/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "source_base_correction_sha256": SOURCE_CORRECTION_SHA,
        "source": {
            "deck_sha256": DECK_SHA,
            "live_model_sha256": LIVE_SHA,
            "zero_model_sha256": ZERO_SHA,
            "n15_model_sha256": N15_SHA,
            "n15_report_sha256": N15_REPORT_SHA,
            "n8b_control_sha256": N8B_CONTROL_SHA,
            "n8b_delta_100_sha256": N8B_SHA,
            "n8b_arms_sha256": N8B_ARMS_SHA,
        },
        "composition": composition_reload,
        "model": str(model_path),
        "model_sha256": digest(model_path),
        "changed_keys_vs_zero": changed,
        "serialization": serialization,
        "teacher": teacher,
        "n6_preservation": n6,
        "n9_preservation": n9,
        "n6_reference": n6_reference,
        "n9_reference": n9_reference,
        "n6_provenance": n6_provenance,
        "n9_provenance": n9_provenance,
        "gates": gates,
        "offline_admission": admitted,
        "official_engine_candidate_execution": False,
        "development_authority": admitted,
        "deployment_authority": False,
        "strength_improvement_claim": False,
        "builder_sha256": digest(Path(__file__).resolve()),
    }
    report_path = OUTPUT / "build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "model": str(model_path),
                "model_sha256": report["model_sha256"],
                "report": str(report_path),
                "offline_admission": admitted,
                "gates": gates,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
