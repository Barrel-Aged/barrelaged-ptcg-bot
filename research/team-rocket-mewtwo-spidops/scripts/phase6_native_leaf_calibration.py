"""Generate Phase 6S held-out native leaf-continuation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_leaf_calibration import (  # noqa: E402
    run_phase6_native_leaf_calibration,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maximum-source-depth-delta", type=int, default=2)
    parser.add_argument("--particle-variants", type=int, default=3)
    parser.add_argument("--rollouts-per-leaf", type=int, default=3)
    parser.add_argument("--maximum-macro-transitions", type=int, default=512)
    parser.add_argument("--maximum-tactical-transitions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=6_025)
    parser.add_argument(
        "--source-tactical-blueprint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_blueprint"
            / "native_regret_blueprint.json"
        ),
    )
    parser.add_argument(
        "--source-plan-blueprint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_plan_iterations"
            / "iterative_native_plan_blueprint.json"
        ),
    )
    parser.add_argument(
        "--source-phase6p-evidence",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_plan_iterations"
            / "phase6p_evidence.json"
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_leaf_calibration",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_leaf_calibration(
        project_root=PROJECT_ROOT,
        source_tactical_blueprint_path=args.source_tactical_blueprint,
        source_plan_blueprint_path=args.source_plan_blueprint,
        source_phase6p_evidence_path=args.source_phase6p_evidence,
        maximum_source_depth_delta=args.maximum_source_depth_delta,
        particle_variants=args.particle_variants,
        rollouts_per_leaf=args.rollouts_per_leaf,
        maximum_macro_transitions=args.maximum_macro_transitions,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "native_leaf_continuation_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(
        target / "native_leaf_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6s_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6s_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "empirical_rollout_leaf_gate_passed": (
                    artifacts.calibration.empirical_rollout_leaf_gate_passed
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
