"""Generate Phase 6V public-only ridge leaf evidence on a fresh holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_public_leaf import (  # noqa: E402
    run_phase6_native_public_leaf,
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
    parser.add_argument("--fresh-holdout-dataset", type=Path)
    parser.add_argument("--seed", type=int, default=6_028)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_public_leaf",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_public_leaf(
        project_root=PROJECT_ROOT,
        training_dataset_path=args.training_dataset,
        fresh_holdout_dataset_path=args.fresh_holdout_dataset,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "fresh_native_leaf_dataset.json",
        artifacts.fresh_dataset.to_json() + "\n",
    )
    _write(target / "public_ridge_leaf.json", artifacts.model.to_json() + "\n")
    _write(
        target / "public_leaf_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6v_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6v_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "training_dataset_fingerprint": (
                    artifacts.evidence.training_dataset_fingerprint
                ),
                "fresh_holdout_dataset_fingerprint": (
                    artifacts.fresh_dataset.fingerprint
                ),
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "selected_regularization": artifacts.model.selected_regularization,
                "best_reference_model": artifacts.calibration.best_reference_model,
                "outperformed_best_reference": (
                    artifacts.calibration.outperformed_best_reference
                ),
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
