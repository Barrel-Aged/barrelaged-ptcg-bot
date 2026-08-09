"""Freeze, resume, evaluate, and write Phase 6AJ artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_branch_conditioned_values import BranchConditionedRoot  # noqa: E402
from rocketbot.phase6_semiparametric_branch_hybrid import (  # noqa: E402
    BranchHybridCalibration,
    BranchHybridModel,
    BranchHybridValidationDataset,
    _load_phase6ai_artifacts,
    collect_branch_hybrid_validation_dataset,
    prepare_branch_hybrid,
    run_phase6_semiparametric_branch_hybrid,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_shards(directory: Path) -> tuple[BranchConditionedRoot, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        BranchConditionedRoot.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in sorted(directory.glob("*.json"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_semiparametric_branch_hybrid",
    )
    parser.add_argument(
        "--shard-directory",
        type=Path,
        default=PROJECT_ROOT / "work" / "phase6aj_root_shards",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-validation", action="store_true")
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    model_path = target / "branch_hybrid_model.json"
    calibration_path = target / "branch_hybrid_calibration.json"
    validation_path = target / "branch_hybrid_validation_dataset.json"

    if model_path.is_file() or calibration_path.is_file():
        if not (model_path.is_file() and calibration_path.is_file()):
            raise RuntimeError("Phase 6AJ frozen model/calibration pair is incomplete")
        model = BranchHybridModel.from_dict(
            json.loads(model_path.read_text(encoding="utf-8"))
        )
        calibration = BranchHybridCalibration.from_dict(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
    else:
        source = _load_phase6ai_artifacts(PROJECT_ROOT)
        model, calibration = prepare_branch_hybrid(*source)
        _write(model_path, model.to_json() + "\n")
        _write(calibration_path, calibration.to_json() + "\n")

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "phase6aj_model_and_bounds_frozen_before_validation",
                    "model_fingerprint": model.fingerprint,
                    "calibration_fingerprint": calibration.fingerprint,
                    "selected_mean_blend": model.selected_mean_blend,
                    "selected_advantage_scale": model.selected_advantage_scale,
                    "conformal_radius": calibration.conformal_radius,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.reuse_validation:
        if not validation_path.is_file():
            raise FileNotFoundError(
                "--reuse-validation requires branch_hybrid_validation_dataset.json"
            )
        dataset = BranchHybridValidationDataset.from_dict(
            json.loads(validation_path.read_text(encoding="utf-8"))
        )
    else:
        dataset = collect_branch_hybrid_validation_dataset(
            project_root=PROJECT_ROOT,
            existing_roots=_load_shards(args.shard_directory.resolve()),
        )
        _write(validation_path, dataset.to_json() + "\n")

    artifacts = run_phase6_semiparametric_branch_hybrid(
        project_root=PROJECT_ROOT,
        validation_dataset=dataset,
        model=model,
        calibration=calibration,
    )
    _write(
        target / "branch_hybrid_evaluation.json",
        artifacts.evaluation.to_json() + "\n",
    )
    _write(target / "phase6aj_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6aj_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "validation_dataset_fingerprint": artifacts.validation_dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "evaluation_fingerprint": artifacts.evaluation.fingerprint,
                "validation_mse": artifacts.evidence.validation_mse,
                "validation_zero_mse": artifacts.evidence.validation_zero_mse,
                "validation_root_heuristic_mse": artifacts.evidence.validation_root_heuristic_mse,
                "validation_branch_heuristic_mse": artifacts.evidence.validation_branch_heuristic_mse,
                "validation_action_contrast_mse": artifacts.evidence.validation_action_contrast_mse,
                "validation_zero_action_contrast_mse": artifacts.evidence.validation_zero_action_contrast_mse,
                "zero_contrast_bootstrap_lower_bound": artifacts.evidence.zero_contrast_bootstrap_lower_bound,
                "validation_coverage": artifacts.evidence.validation_coverage,
                "validation_mean_width": artifacts.evidence.validation_mean_width,
                "minimum_validation_slice_coverage": artifacts.evidence.minimum_validation_slice_coverage,
                "maximum_target_standard_error": artifacts.evidence.maximum_target_standard_error,
                "sampling_precision_gate_passed": artifacts.evidence.sampling_precision_gate_passed,
                "point_mse_gate_passed": artifacts.evidence.point_mse_gate_passed,
                "contrast_relative_gate_passed": artifacts.evidence.contrast_relative_gate_passed,
                "contrast_bootstrap_gate_passed": artifacts.evidence.contrast_bootstrap_gate_passed,
                "interval_gate_passed": artifacts.evidence.interval_gate_passed,
                "implementation_holdout_gate_passed": artifacts.evidence.implementation_holdout_gate_passed,
                "action_sensitive_point_model_validated": artifacts.evidence.action_sensitive_point_model_validated,
                "conditional_expected_mean_model_admitted": artifacts.evidence.conditional_expected_mean_model_admitted,
                "counterfactual_value_compatible": False,
                "equilibrium_convergence_validated": False,
                "safe_resolver_enabled": False,
                "opponent_exploitation_enabled": False,
                "official_strategy_control_enabled": False,
                "artifact_directory": str(target),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
