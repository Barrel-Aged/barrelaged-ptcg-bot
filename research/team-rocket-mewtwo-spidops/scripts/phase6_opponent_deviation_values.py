"""Generate Phase 6X native opponent plan-deviation value evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_opponent_deviation_values import (  # noqa: E402
    run_phase6_opponent_deviation_values,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particle-variants", type=int, default=2)
    parser.add_argument("--rollouts-per-action", type=int, default=10)
    parser.add_argument("--maximum-reach-transitions", type=int, default=16)
    parser.add_argument("--maximum-rollout-transitions", type=int, default=512)
    parser.add_argument("--maximum-tactical-transitions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=6_030)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_opponent_deviation_values",
    )
    args = parser.parse_args()
    artifacts = run_phase6_opponent_deviation_values(
        project_root=PROJECT_ROOT,
        particle_variants=args.particle_variants,
        rollouts_per_action=args.rollouts_per_action,
        maximum_reach_transitions=args.maximum_reach_transitions,
        maximum_rollout_transitions=args.maximum_rollout_transitions,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "opponent_deviation_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(
        target / "opponent_deviation_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6x_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6x_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "surface_count": len(artifacts.dataset.surfaces),
                "action_target_count": artifacts.evidence.action_target_count,
                "terminal_rollout_count": artifacts.dataset.terminal_rollout_count,
                "native_transition_count": artifacts.dataset.native_transition_count,
                "validation_nontrivial_action_surface_fraction": (
                    artifacts.calibration.validation.nontrivial_action_surface_fraction
                ),
                "validation_mean_action_span": (
                    artifacts.calibration.validation.mean_action_span
                ),
                "validation_mean_empirical_deviation_gain": (
                    artifacts.calibration.validation.mean_empirical_deviation_gain
                ),
                "validation_maximum_rollout_standard_error": (
                    artifacts.calibration.validation.maximum_rollout_standard_error
                ),
                "conditional_deviation_substrate_gate_passed": (
                    artifacts.calibration.conditional_deviation_substrate_gate_passed
                ),
                "opponent_counterfactual_value_gate_passed": (
                    artifacts.calibration.opponent_counterfactual_value_gate_passed
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
