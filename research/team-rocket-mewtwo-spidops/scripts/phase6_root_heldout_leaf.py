"""Generate Phase 6W collection-heldout public leaf evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_root_heldout_leaf import (  # noqa: E402
    run_phase6_root_heldout_leaf,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_leaf_model"
            / "native_low_noise_leaf_dataset.json"
        ),
    )
    parser.add_argument(
        "--calibration-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_public_leaf"
            / "fresh_native_leaf_dataset.json"
        ),
    )
    parser.add_argument(
        "--phase6v-evidence",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_public_leaf"
            / "phase6v_evidence.json"
        ),
    )
    parser.add_argument("--validation-dataset", type=Path)
    parser.add_argument("--seed", type=int, default=6_029)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_root_heldout_leaf",
    )
    args = parser.parse_args()
    artifacts = run_phase6_root_heldout_leaf(
        project_root=PROJECT_ROOT,
        training_dataset_path=args.training_dataset,
        calibration_dataset_path=args.calibration_dataset,
        source_phase6v_evidence_path=args.phase6v_evidence,
        validation_dataset_path=args.validation_dataset,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "fresh_validation_leaf_dataset.json",
        artifacts.validation_dataset.to_json() + "\n",
    )
    _write(target / "root_heldout_public_leaf.json", artifacts.model.to_json() + "\n")
    _write(
        target / "root_heldout_leaf_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6w_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6w_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "validation_dataset_fingerprint": (
                    artifacts.validation_dataset.fingerprint
                ),
                "selected_regularization": artifacts.model.selected_regularization,
                "conformal_radius": artifacts.calibration.conformal_radius,
                "validation_coverage": artifacts.calibration.validation_coverage,
                "validation_root_mean_square_error": (
                    artifacts.calibration.validation_root_mean_square_error
                ),
                "best_reference_model": artifacts.calibration.best_reference_model,
                "best_reference_root_mean_square_error": (
                    artifacts.calibration.best_reference_root_mean_square_error
                ),
                "outperformed_best_reference": (
                    artifacts.calibration.outperformed_best_reference
                ),
                "mean_interval_width": artifacts.calibration.mean_interval_width,
                "empirical_fixed_profile_leaf_gate_passed": (
                    artifacts.calibration.empirical_fixed_profile_leaf_gate_passed
                ),
                "counterfactual_value_gate_passed": (
                    artifacts.calibration.counterfactual_value_gate_passed
                ),
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
