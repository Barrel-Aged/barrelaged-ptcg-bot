"""Generate Phase 6Z public-root-aligned conditional-value evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_aligned_counterfactual_values import (  # noqa: E402
    run_phase6_aligned_counterfactual_values,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particle-variants", type=int, default=3)
    parser.add_argument("--rollouts-per-action", type=int, default=3)
    parser.add_argument("--maximum-live-selections", type=int, default=900)
    parser.add_argument("--maximum-capture-attempts", type=int, default=8)
    parser.add_argument("--maximum-rollout-transitions", type=int, default=512)
    parser.add_argument("--maximum-tactical-transitions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=6_032)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "phase6_aligned_counterfactual_values"
        ),
    )
    args = parser.parse_args()
    artifacts = run_phase6_aligned_counterfactual_values(
        project_root=PROJECT_ROOT,
        particle_variants=args.particle_variants,
        rollouts_per_action=args.rollouts_per_action,
        maximum_live_selections=args.maximum_live_selections,
        maximum_capture_attempts=args.maximum_capture_attempts,
        maximum_rollout_transitions=args.maximum_rollout_transitions,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "public_root_aligned_value_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(
        target / "hypothesis_conditioned_value_model.json",
        artifacts.model.to_json() + "\n",
    )
    _write(
        target / "aligned_value_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6z_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6z_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "root_count": artifacts.evidence.root_count,
                "surface_count": artifacts.evidence.surface_count,
                "action_target_count": artifacts.evidence.action_target_count,
                "terminal_rollout_count": artifacts.evidence.terminal_rollout_count,
                "native_transition_count": artifacts.evidence.native_transition_count,
                "validation_coverage": artifacts.evidence.validation_coverage,
                "validation_rmse": artifacts.evidence.validation_rmse,
                "validation_zero_baseline_rmse": (
                    artifacts.evidence.validation_zero_baseline_rmse
                ),
                "maximum_rollout_standard_error": (
                    artifacts.evidence.maximum_rollout_standard_error
                ),
                "public_root_alignment_gate_passed": (
                    artifacts.evidence.public_root_alignment_gate_passed
                ),
                "collection_disjoint_gate_passed": (
                    artifacts.evidence.collection_disjoint_gate_passed
                ),
                "empirical_conditional_model_gate_passed": (
                    artifacts.evidence.empirical_conditional_model_gate_passed
                ),
                "native_counterfactual_values_calibrated": False,
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
