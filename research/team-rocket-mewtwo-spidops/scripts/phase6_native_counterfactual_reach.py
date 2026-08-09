"""Generate Phase 6Y exact and native counterfactual-reach evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_counterfactual_reach import (  # noqa: E402
    run_phase6_native_counterfactual_reach,
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
    parser.add_argument("--exact-error-threshold", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=6_031)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_counterfactual_reach",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_counterfactual_reach(
        project_root=PROJECT_ROOT,
        particle_variants=args.particle_variants,
        rollouts_per_action=args.rollouts_per_action,
        maximum_reach_transitions=args.maximum_reach_transitions,
        maximum_rollout_transitions=args.maximum_rollout_transitions,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
        exact_error_threshold=args.exact_error_threshold,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "source_opponent_deviation_dataset.json",
        artifacts.source_dataset.to_json() + "\n",
    )
    _write(
        target / "source_opponent_deviation_calibration.json",
        artifacts.source_calibration.to_json() + "\n",
    )
    _write(target / "exact_reach_audit.json", artifacts.exact_audit.to_json() + "\n")
    _write(
        target / "native_counterfactual_reach_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(
        target / "native_counterfactual_reach_admission.json",
        artifacts.admission.to_json() + "\n",
    )
    _write(target / "phase6y_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6y_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "exact_reach_audit_fingerprint": artifacts.exact_audit.fingerprint,
                "source_dataset_fingerprint": artifacts.source_dataset.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "admission_fingerprint": artifacts.admission.fingerprint,
                "surface_count": artifacts.evidence.surface_count,
                "action_target_count": artifacts.evidence.action_target_count,
                "terminal_rollout_count": artifacts.evidence.terminal_rollout_count,
                "native_transition_count": artifacts.evidence.native_transition_count,
                "exact_maximum_absolute_error": (
                    artifacts.evidence.exact_maximum_absolute_error
                ),
                "native_importance_weight_range": [
                    artifacts.evidence.native_minimum_importance_weight,
                    artifacts.evidence.native_maximum_importance_weight,
                ],
                "exact_reach_semantics_gate_passed": (
                    artifacts.evidence.exact_reach_semantics_gate_passed
                ),
                "native_reach_ledger_gate_passed": (
                    artifacts.evidence.native_reach_ledger_gate_passed
                ),
                "native_counterfactual_values_calibrated": (
                    artifacts.evidence.native_counterfactual_values_calibrated
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
