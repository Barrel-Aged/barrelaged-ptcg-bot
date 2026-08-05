#!/usr/bin/env python3
"""Construct and audit the sole analytic N16 residual projection."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
TERMINAL = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/training_v1/terminal2000.pt"
N15_REPORT = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/training_v1/training_report.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n16_minimum_scale_projection/build_v1"
PROTOCOL = ROOT / "N16_MINIMUM_SCALE_RESIDUAL_PROTOCOL.md"

DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_MODEL_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
ZERO_NET_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
TERMINAL_SHA = "354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7"
N15_REPORT_SHA = "b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1"
PROTOCOL_SHA = "f7b74e00ba0f7250550bca21d4454bcf903169343e82d0dee60cab326d0ba190"
TARGET_MARGIN = 0.02
ROUNDING_GUARD = 1.0e-6
ROUNDING_GRID = 1_000_000
FINAL_KEY = "logit_adapter.3.weight"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_n14_localized_contextual_residual as n14  # noqa: E402

sys.path = [entry for entry in sys.path if Path(entry).resolve() != ZERO.resolve()]
sys.path.insert(0, str(ZERO))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def constraint_interval(
    baselines: Sequence[np.ndarray],
    residuals: Sequence[np.ndarray],
    teachers: Sequence[int],
    *,
    margin: float,
) -> dict[str, Any]:
    if not (len(baselines) == len(residuals) == len(teachers)) or not baselines:
        raise ValueError("N16 constraints require aligned nonempty inputs")
    lower = 0.0
    upper = 1.0
    lower_witness: dict[str, Any] | None = None
    upper_witness: dict[str, Any] | None = None
    constraints = 0
    infeasible = []
    for example_index, (base_raw, residual_raw, teacher_raw) in enumerate(
        zip(baselines, residuals, teachers)
    ):
        base = np.asarray(base_raw, dtype=np.float64)
        residual = np.asarray(residual_raw, dtype=np.float64)
        teacher = int(teacher_raw)
        if base.shape != residual.shape or base.ndim != 1:
            raise ValueError("N16 logit vectors changed shape")
        legal = np.flatnonzero(np.isfinite(base) & (base > -1.0e8))
        if teacher not in legal:
            raise ValueError("N16 teacher is not legal")
        for competitor in legal:
            competitor = int(competitor)
            if competitor == teacher:
                continue
            constraints += 1
            baseline_margin = float(base[teacher] - base[competitor])
            residual_delta = float(residual[teacher] - residual[competitor])
            if residual_delta > 0.0:
                bound = (margin - baseline_margin) / residual_delta
                if bound > lower:
                    lower = bound
                    lower_witness = {
                        "example": example_index,
                        "teacher": teacher,
                        "competitor": competitor,
                        "baseline_margin": baseline_margin,
                        "residual_delta": residual_delta,
                        "bound": bound,
                    }
            elif residual_delta < 0.0:
                bound = (baseline_margin - margin) / (-residual_delta)
                if bound < upper:
                    upper = bound
                    upper_witness = {
                        "example": example_index,
                        "teacher": teacher,
                        "competitor": competitor,
                        "baseline_margin": baseline_margin,
                        "residual_delta": residual_delta,
                        "bound": bound,
                    }
            elif baseline_margin < margin:
                infeasible.append(
                    {
                        "example": example_index,
                        "teacher": teacher,
                        "competitor": competitor,
                        "baseline_margin": baseline_margin,
                    }
                )
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return {
        "constraints": constraints,
        "lower": lower,
        "upper": upper,
        "lower_witness": lower_witness,
        "upper_witness": upper_witness,
        "zero_slope_infeasible": infeasible,
        "feasible": not infeasible and lower <= upper,
    }


def rounded_scale(lower: float) -> float:
    # Decimal arithmetic prevents binary-float representation from adding an
    # unintended second grid unit when the lower bound is already on-grid.
    value = (Decimal(str(float(lower))) + Decimal("0.000001")) * Decimal(ROUNDING_GRID)
    units = value.to_integral_value(rounding=ROUND_CEILING)
    return float(units / Decimal(ROUNDING_GRID))


@torch.no_grad()
def residual_vectors(
    adapter: torch.nn.Module,
    examples: Sequence[n14.support.N12Example],
) -> list[np.ndarray]:
    adapter.eval()
    output = []
    for example in examples:
        embedding = torch.as_tensor(example.contextual_options, dtype=torch.float32)
        output.append(adapter(embedding).squeeze(-1).cpu().numpy().astype(np.float32))
    return output


@torch.no_grad()
def correction_margin_metrics(
    adapter: torch.nn.Module,
    examples: Sequence[n14.support.N12Example],
) -> dict[str, Any]:
    adapter.eval()
    logits = n14.cached_logits(adapter, examples, torch.device("cpu"))
    fitted = 0
    minimum = float("inf")
    by_cell: dict[str, dict[str, Any]] = {}
    margins = []
    for row, example in enumerate(examples):
        width = int(example.feat["opt_feats"].shape[0])
        legal = logits[row, :width].clone()
        teacher = int(example.teacher_best)
        teacher_logit = legal[teacher].clone()
        legal[teacher] = -torch.inf
        margin = float(teacher_logit - legal.max())
        prediction = int(logits[row, :width].argmax())
        fit = int(prediction == teacher)
        fitted += fit
        minimum = min(minimum, margin)
        margins.append(margin)
        cell = by_cell.setdefault(example.cell, {"examples": 0, "fitted": 0, "minimum_margin": float("inf")})
        cell["examples"] += 1
        cell["fitted"] += fit
        cell["minimum_margin"] = min(float(cell["minimum_margin"]), margin)
    return {
        "examples": len(examples),
        "fitted": fitted,
        "minimum_margin": minimum,
        "mean_margin": float(np.mean(margins)),
        "by_cell": dict(sorted(by_cell.items())),
    }


def compare_roundtrip(left: torch.nn.Module, right: torch.nn.Module) -> dict[str, Any]:
    before = left.state_dict()
    after = right.state_dict()
    unequal = sorted(
        name for name in set(before).intersection(after)
        if not torch.equal(before[name].detach().cpu(), after[name].detach().cpu())
    )
    missing = sorted(set(before) - set(after))
    unexpected = sorted(set(after) - set(before))
    return {
        "passes": not unequal and not missing and not unexpected,
        "unequal_keys": unequal,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def direction_audit(
    terminal: torch.nn.Module,
    candidate: torch.nn.Module,
    scale: float,
) -> dict[str, Any]:
    source = terminal.state_dict()
    projected = candidate.state_dict()
    unequal_nonfinal = sorted(
        name for name in source
        if name != FINAL_KEY and not torch.equal(source[name], projected[name])
    )
    expected = source[FINAL_KEY] * float(scale)
    error = float((projected[FINAL_KEY] - expected).abs().max())
    return {
        "passes": not unequal_nonfinal and error == 0.0,
        "unequal_nonfinal_keys": unequal_nonfinal,
        "final_projection_max_abs_error": error,
        "terminal_config": dict(terminal.cfg),
        "candidate_config": dict(candidate.cfg),
        "config_exact": terminal.cfg == candidate.cfg,
    }


def admission(
    interval: dict[str, Any],
    scale: float,
    fit: dict[str, Any],
    direction: dict[str, Any],
    changed: list[str],
    n6: dict[str, Any],
    n9: dict[str, Any],
    n6_parity: dict[str, Any],
    n9_parity: dict[str, Any],
    serialization: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "constraint_interval_feasible": bool(interval["feasible"]),
        "scale_strictly_between_zero_and_one": 0.0 < scale < 1.0,
        "scale_within_upper_bound": scale <= float(interval["upper"]),
        "fit_48_of_48": int(fit["fitted"]) == 48,
        "minimum_margin_ge_0_019999": float(fit["minimum_margin"]) >= 0.019999,
        "direction_exact": bool(direction["passes"] and direction["config_exact"]),
        "adapter_keys_only_vs_zero": bool(changed) and all(
            name.startswith("logit_adapter.") for name in changed
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
        raise ValueError("N16 builder accepts no command-line overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N16 output: {OUTPUT}")
    checks = {
        ZERO / "model.pt": ZERO_MODEL_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": ZERO_NET_SHA,
        n14.LIVE / "model.pt": LIVE_MODEL_SHA,
        TERMINAL: TERMINAL_SHA,
        N15_REPORT: N15_REPORT_SHA,
        PROTOCOL: PROTOCOL_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N16 frozen input changed: {path}")
    n15_report = json.loads(N15_REPORT.read_text(encoding="utf-8"))
    if not n15_report["terminal"]["admission"]["passes"]:
        raise ValueError("N16 source did not pass N15 offline admission")

    torch.set_num_threads(4)
    device = torch.device("cpu")
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    deck = [int(value) for value in (ZERO / "deck.csv").read_text().replace(",", "\n").split()]
    library = json.loads((ZERO / "deck_library.json").read_text(encoding="utf-8"))
    zero = load_policy(ZERO / "model.pt", map_location=device).to(device).eval()
    terminal = load_policy(TERMINAL, map_location=device).to(device).eval()
    candidate = load_policy(TERMINAL, map_location=device).to(device).eval()
    if float(terminal.adapter_scale) != 1.0 or int(terminal.adapter_rank) != 16:
        raise ValueError("N16 terminal residual config changed")

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
    n6_reference = n14.attach_contextual_reference(zero, n6_all)
    n9_reference = n14.attach_contextual_reference(zero, n9_all)
    for name, row in (("n6", n6_reference), ("n9", n9_reference)):
        if row["reference_action_mismatches"] or row["maximum_initial_residual"] != 0.0:
            raise ValueError(f"N16 reference parity failed: {name}={row}")
    corrections = [example for example in n6_all if example.correction]
    n6_anchors = [example for example in n6_all if not example.correction]
    if len(corrections) != 48 or len(n6_anchors) != 591 or len(n9_all) != 6_753:
        raise ValueError("N16 support counts changed")

    residuals = residual_vectors(terminal.logit_adapter, corrections)
    baselines = [np.asarray(example.reference_logits, dtype=np.float32) for example in corrections]
    teachers = [int(example.teacher_best) for example in corrections]
    interval = constraint_interval(
        baselines, residuals, teachers, margin=TARGET_MARGIN
    )
    scale = rounded_scale(float(interval["lower"]))
    if (
        not interval["feasible"]
        or not (0.0 < scale < 1.0)
        or scale > float(interval["upper"])
    ):
        raise ValueError(f"N16 analytic projection is infeasible: interval={interval}, scale={scale}")

    with torch.no_grad():
        candidate.logit_adapter[-1].weight.copy_(
            terminal.logit_adapter[-1].weight * float(scale)
        )
    direction = direction_audit(terminal, candidate, scale)
    if not direction["passes"] or not direction["config_exact"]:
        raise AssertionError(f"N16 direction construction failed: {direction}")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    model_path = OUTPUT / "model.pt"
    save_policy(candidate, model_path)
    reloaded = load_policy(model_path, map_location=device).to(device).eval()
    serialization = compare_roundtrip(candidate, reloaded)
    direction_reload = direction_audit(terminal, reloaded, scale)
    fit = correction_margin_metrics(reloaded.logit_adapter, corrections)
    changed = n14.changed_keys(zero, reloaded)
    n6_metrics = n14.preservation_metrics(reloaded.logit_adapter, n6_anchors)
    n9_metrics = n14.preservation_metrics(reloaded.logit_adapter, n9_all)
    n6_runtime = n14.full_cached_parity(reloaded, n6_all)
    n9_runtime = n14.full_cached_parity(reloaded, n9_all)
    gate = admission(
        interval,
        scale,
        fit,
        direction_reload,
        changed,
        n6_metrics,
        n9_metrics,
        n6_runtime,
        n9_runtime,
        serialization,
    )
    report = {
        "schema": "rocket-n16-minimum-scale-residual-projection/v1",
        "protocol": PROTOCOL.name,
        "protocol_sha256": PROTOCOL_SHA,
        "deck_sha256": DECK_SHA,
        "live_model_sha256": LIVE_MODEL_SHA,
        "zero_model_sha256": ZERO_MODEL_SHA,
        "terminal_model_sha256": TERMINAL_SHA,
        "n15_training_report_sha256": N15_REPORT_SHA,
        "target_margin": TARGET_MARGIN,
        "rounding_guard": ROUNDING_GUARD,
        "rounding_grid": ROUNDING_GRID,
        "constraint_interval": interval,
        "selected_scale": scale,
        "model_path": str(model_path),
        "model_sha256": digest(model_path),
        "direction_audit": direction_reload,
        "serialization_roundtrip": serialization,
        "changed_state_keys_vs_zero": changed,
        "correction_fit": fit,
        "n6_preservation": n6_metrics,
        "n9_preservation": n9_metrics,
        "n6_runtime_cache_parity": n6_runtime,
        "n9_runtime_cache_parity": n9_runtime,
        "n6_provenance": n6_provenance,
        "n9_provenance": n9_provenance,
        "reference_parity": {"n6": n6_reference, "n9": n9_reference},
        "admission": gate,
        "claims": {
            "offline_admission": bool(gate["passes"]),
            "development_panel_authority": bool(gate["passes"]),
            "deployment_authority": False,
            "strength_improvement": False,
        },
    }
    report_path = OUTPUT / "projection_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "model_sha256": report["model_sha256"],
                "selected_scale": scale,
                "offline_admission": gate["passes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
