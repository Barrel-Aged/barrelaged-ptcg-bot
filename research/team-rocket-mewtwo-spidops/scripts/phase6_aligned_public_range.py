"""Generate Phase 6AA aligned public-archetype range evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_aligned_public_range import (  # noqa: E402
    run_phase6_aligned_public_range,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-phase6z-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_aligned_counterfactual_values"
            / "public_root_aligned_value_dataset.json"
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_aligned_public_range",
    )
    args = parser.parse_args()
    artifacts = run_phase6_aligned_public_range(
        project_root=PROJECT_ROOT,
        source_phase6z_dataset_path=args.source_phase6z_dataset,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "aligned_public_range_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(
        target / "prototype_public_range_model.json",
        artifacts.model.to_json() + "\n",
    )
    _write(
        target / "aligned_public_range_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6aa_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6aa_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "selected_temperature": artifacts.model.selected_temperature,
                "selected_uniform_mix": artifacts.model.selected_uniform_mix,
                "calibration_nll": artifacts.calibration.calibration.nll,
                "validation_nll": artifacts.calibration.validation.nll,
                "uniform_nll": artifacts.calibration.uniform_nll,
                "validation_brier": artifacts.calibration.validation.brier,
                "uniform_brier": artifacts.calibration.uniform_brier,
                "validation_top1_accuracy": (
                    artifacts.calibration.validation.top1_accuracy
                ),
                "validation_expected_calibration_error": (
                    artifacts.calibration.validation.expected_calibration_error
                ),
                "heldout_archetype_signal_gate_passed": (
                    artifacts.calibration.heldout_archetype_signal_gate_passed
                ),
                "official_public_range_calibrated": False,
                "hidden_hand_or_resource_posterior": False,
                "safe_resolver_enabled": False,
                "official_strategy_control_enabled": False,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
