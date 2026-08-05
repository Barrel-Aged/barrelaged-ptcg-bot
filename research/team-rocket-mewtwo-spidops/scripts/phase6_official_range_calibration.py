"""Collect and write Phase 6AD observer-oriented official range evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_official_range_calibration import (  # noqa: E402
    OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE,
    OFFICIAL_RANGE_ROOTS_PER_CANDIDATE_PER_SPLIT,
    run_phase6_official_range_calibration,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_official_range_calibration",
    )
    parser.add_argument(
        "--roots-per-candidate-per-split",
        type=int,
        default=OFFICIAL_RANGE_ROOTS_PER_CANDIDATE_PER_SPLIT,
    )
    parser.add_argument(
        "--particles-per-candidate",
        type=int,
        default=OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE,
    )
    parser.add_argument("--maximum-live-selections", type=int, default=900)
    parser.add_argument("--maximum-capture-attempts", type=int, default=8)
    args = parser.parse_args()
    artifacts = run_phase6_official_range_calibration(
        project_root=PROJECT_ROOT,
        roots_per_candidate_per_split=args.roots_per_candidate_per_split,
        particles_per_candidate=args.particles_per_candidate,
        maximum_live_selections=args.maximum_live_selections,
        maximum_capture_attempts=args.maximum_capture_attempts,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "official_range_truth_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(
        target / "official_public_range_model.json",
        artifacts.model.to_json() + "\n",
    )
    _write(
        target / "official_range_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6ad_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6ad_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "root_count": artifacts.evidence.root_count,
                "roots_per_split": artifacts.evidence.roots_per_split,
                "live_selection_count": artifacts.evidence.live_selection_count,
                "total_public_consistent_particles": (
                    artifacts.evidence.total_public_consistent_particles
                ),
                "validation_candidate_nll": (
                    artifacts.evidence.validation_candidate_nll
                ),
                "validation_candidate_ece": (
                    artifacts.evidence.validation_candidate_ece
                ),
                "validation_resource_presence_brier": (
                    artifacts.evidence.validation_resource_presence_brier
                ),
                "validation_resource_ece": (
                    artifacts.evidence.validation_resource_ece
                ),
                "in_scope_official_hidden_card_range_calibrated": (
                    artifacts.evidence.in_scope_official_hidden_card_range_calibrated
                ),
                "full_official_public_belief_calibrated": False,
                "counterfactual_value_compatible": False,
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
