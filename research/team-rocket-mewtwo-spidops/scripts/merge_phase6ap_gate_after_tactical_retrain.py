"""Carry the admitted gate across an action-only tactical retrain."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.integrated_planner import FrozenStrategicPlanner  # noqa: E402
from rocketbot.integrated_training import _canonical, _fingerprint  # noqa: E402


PLANNER_LAYERS = (
    "planner_hidden_1",
    "planner_hidden_2",
    "archetype_head",
    "intent_head",
    "resource_head",
    "plan_policy_head",
    "belief_plan_bridge",
    "plan_value_head",
    "strategic_head",
    "meta_head",
    "uncertainty_head",
    "search_gate_head",
)


def main() -> None:
    directory = PROJECT_ROOT / "artifacts" / "integrated_controller"
    admitted_path = directory / "integrated_planner_phase6ap_pre_action_retrain.json"
    output_path = directory / "integrated_planner.json"
    admitted = json.loads(admitted_path.read_text(encoding="utf-8"))
    tactical = json.loads(output_path.read_text(encoding="utf-8"))
    invariants = {
        "planner_layers_identical": all(
            _canonical(admitted["layers"][name]) == _canonical(tactical["layers"][name])
            for name in PLANNER_LAYERS
        ),
        "planner_normalizer_identical": _canonical(admitted["normalizers"]["planner"])
        == _canonical(tactical["normalizers"]["planner"]),
        "counterfactual_calibration_identical": all(
            admitted["calibration"].get(name) == tactical["calibration"].get(name)
            for name in (
                "archetype_temperature",
                "intent_temperature",
                "plan_temperature",
                "behavior_evidence_weight",
                "counterfactual_bounds_admitted",
                "counterfactual_advantage_radius",
                "safe_non_regression_tolerance",
                "safe_improvement_margin",
                "maximum_resolve_mass",
                "plan_persistence",
            )
        ),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"strategic checkpoint changed during tactical retrain: {invariants}")
    merged = dict(tactical)
    merged.pop("fingerprint", None)
    merged["layers"] = dict(merged["layers"])
    merged["layers"]["search_gate_direct_head"] = admitted["layers"]["search_gate_direct_head"]
    merged["normalizers"] = dict(merged["normalizers"])
    merged["normalizers"]["search_gate"] = admitted["normalizers"]["search_gate"]
    merged["calibration"] = dict(merged["calibration"])
    for name in ("search_gate_threshold", "selective_search_gate_admitted"):
        merged["calibration"][name] = admitted["calibration"][name]
    merged["evidence"] = dict(merged["evidence"])
    merged["evidence"]["selective_search_gate"] = admitted["evidence"]["selective_search_gate"]
    merged["evidence"]["gate_tactical_retrain_invariance"] = invariants
    merged["fingerprint"] = _fingerprint(merged)
    output_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    FrozenStrategicPlanner(json.loads(output_path.read_text(encoding="utf-8")))
    report = {
        "schema_version": 1,
        "admitted_gate_source_fingerprint": admitted["fingerprint"],
        "tactical_retrain_source_fingerprint": tactical["fingerprint"],
        "merged_model_fingerprint": merged["fingerprint"],
        "merged_model_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "invariants": invariants,
    }
    report["fingerprint"] = _fingerprint(report)
    (directory / "gate_tactical_retrain_merge_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
