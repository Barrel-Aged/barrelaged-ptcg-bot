"""Collect and write Phase 6AF implementation-heldout intent evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_official_policy_generalization import (  # noqa: E402
    OFFICIAL_POLICY_PARTICLES_PER_CANDIDATE,
    OFFICIAL_POLICY_ROOTS_PER_CANDIDATE_INTENT_SPLIT,
    run_phase6_official_policy_generalization,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "phase6_official_policy_generalization"
        ),
    )
    parser.add_argument(
        "--source-phase6ad-artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_official_range_calibration",
    )
    parser.add_argument(
        "--source-phase6ae-artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_official_intent_calibration",
    )
    parser.add_argument(
        "--roots-per-candidate-intent-split",
        type=int,
        default=OFFICIAL_POLICY_ROOTS_PER_CANDIDATE_INTENT_SPLIT,
    )
    parser.add_argument(
        "--particles-per-candidate",
        type=int,
        default=OFFICIAL_POLICY_PARTICLES_PER_CANDIDATE,
    )
    parser.add_argument("--maximum-live-selections", type=int, default=900)
    parser.add_argument("--maximum-capture-attempts", type=int, default=8)
    args = parser.parse_args()
    artifacts = run_phase6_official_policy_generalization(
        project_root=PROJECT_ROOT,
        source_phase6ad_artifact_directory=(
            args.source_phase6ad_artifact_directory
        ),
        source_phase6ae_artifact_directory=(
            args.source_phase6ae_artifact_directory
        ),
        roots_per_candidate_intent_split=(
            args.roots_per_candidate_intent_split
        ),
        particles_per_candidate=args.particles_per_candidate,
        maximum_live_selections=args.maximum_live_selections,
        maximum_capture_attempts=args.maximum_capture_attempts,
    )
    target = args.artifact_directory.resolve()
    _write(target / "official_policy_variant_dataset.json", artifacts.dataset.to_json() + "\n")
    _write(target / "official_policy_heldout_model.json", artifacts.model.to_json() + "\n")
    _write(
        target / "official_policy_heldout_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6af_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6af_evidence.md", artifacts.evidence.to_markdown())
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
                "opponent_main_action_count": (
                    artifacts.evidence.opponent_main_action_count
                ),
                "total_public_consistent_particles": (
                    artifacts.evidence.total_public_consistent_particles
                ),
                "validation_intent_top1_accuracy": (
                    artifacts.evidence.validation_intent_top1_accuracy
                ),
                "validation_intent_ece": (
                    artifacts.evidence.validation_intent_ece
                ),
                "validation_joint_ece": artifacts.evidence.validation_joint_ece,
                "minimum_validation_intent_slice_top1": (
                    artifacts.evidence.minimum_validation_intent_slice_top1
                ),
                "implementation_heldout_joint_range_calibrated": (
                    artifacts.evidence.implementation_heldout_joint_range_calibrated
                ),
                "human_latent_intent_calibrated": False,
                "full_official_public_belief_calibrated": False,
                "counterfactual_value_compatible": False,
                "safe_resolver_enabled": False,
                "opponent_exploitation_enabled": False,
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
