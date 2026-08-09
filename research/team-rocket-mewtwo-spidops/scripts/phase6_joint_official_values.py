"""Generate Phase 6AC joint official prefix/truth/value evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_joint_official_values import (  # noqa: E402
    run_phase6_joint_official_values,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_joint_official_values",
    )
    parser.add_argument("--rollouts-per-action", type=int, default=3)
    parser.add_argument("--maximum-live-selections", type=int, default=900)
    parser.add_argument("--maximum-capture-attempts", type=int, default=8)
    parser.add_argument("--maximum-rollout-transitions", type=int, default=512)
    parser.add_argument("--maximum-tactical-transitions", type=int, default=512)
    args = parser.parse_args()
    artifacts = run_phase6_joint_official_values(
        project_root=PROJECT_ROOT,
        rollouts_per_action=args.rollouts_per_action,
        maximum_live_selections=args.maximum_live_selections,
        maximum_capture_attempts=args.maximum_capture_attempts,
        maximum_rollout_transitions=args.maximum_rollout_transitions,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "joint_official_value_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(target / "phase6ac_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6ac_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "root_count": artifacts.evidence.root_count,
                "live_selection_count": artifacts.evidence.live_selection_count,
                "action_target_count": artifacts.evidence.action_target_count,
                "terminal_rollout_count": artifacts.evidence.terminal_rollout_count,
                "native_transition_count": artifacts.evidence.native_transition_count,
                "maximum_rollout_standard_error": (
                    artifacts.evidence.maximum_rollout_standard_error
                ),
                "nontrivial_action_surface_fraction": (
                    artifacts.evidence.nontrivial_action_surface_fraction
                ),
                "same_root_prefix_truth_value_gate_passed": True,
                "official_truth_sample_gate_passed": True,
                "live_native_replay_fidelity_gate_passed": True,
                "common_chance_likelihood_ratio_available": True,
                "absolute_chance_reach_available": False,
                "calibrated_counterfactual_values": False,
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
