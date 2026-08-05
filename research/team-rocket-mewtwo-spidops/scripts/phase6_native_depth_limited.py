"""Generate deterministic Phase 6R depth-limited native-plan evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_depth_limited import (  # noqa: E402
    run_phase6_native_depth_limited,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--maximum-source-depth-delta", type=int, default=6)
    parser.add_argument("--reservoir-capacity", type=int, default=4_096)
    parser.add_argument("--maximum-tactical-transitions", type=int, default=512)
    parser.add_argument("--seed", type=int, default=6_024)
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
        default=PROJECT_ROOT / "artifacts" / "phase6_native_depth_limited",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_depth_limited(
        project_root=PROJECT_ROOT,
        source_tactical_blueprint_path=args.source_tactical_blueprint,
        source_plan_blueprint_path=args.source_plan_blueprint,
        source_phase6p_evidence_path=args.source_phase6p_evidence,
        iterations=args.iterations,
        exploration=args.exploration,
        maximum_source_depth_delta=args.maximum_source_depth_delta,
        reservoir_capacity=args.reservoir_capacity,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "depth_limited_native_plan_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(target / "phase6r_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6r_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "artifact_directory": str(target),
                "raw_regret_p95_absolute": (
                    artifacts.evidence.raw_regret_p95_absolute
                ),
                "raw_regret_maximum_absolute": (
                    artifacts.evidence.raw_regret_maximum_absolute
                ),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
