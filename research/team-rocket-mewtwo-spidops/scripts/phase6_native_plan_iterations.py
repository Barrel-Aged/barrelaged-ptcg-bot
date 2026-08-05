"""Generate durable Phase 6P iterative native turn-plan evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_plan_iterations import (  # noqa: E402
    run_phase6_native_plan_iterations,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--reservoir-capacity", type=int, default=12_288)
    parser.add_argument("--maximum-macro-transitions", type=int, default=256)
    parser.add_argument("--maximum-tactical-transitions", type=int, default=512)
    parser.add_argument("--epochs-per-round", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=6_023)
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
        "--initial-plan-blueprint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_plan_blueprint"
            / "native_plan_blueprint.json"
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_plan_iterations",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_plan_iterations(
        project_root=PROJECT_ROOT,
        source_tactical_blueprint_path=args.source_tactical_blueprint,
        initial_plan_blueprint_path=args.initial_plan_blueprint,
        rounds=args.rounds,
        exploration=args.exploration,
        reservoir_capacity=args.reservoir_capacity,
        maximum_macro_transitions=args.maximum_macro_transitions,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
        epochs_per_round=args.epochs_per_round,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(target / "iterative_native_plan_dataset.json", artifacts.dataset.to_json() + "\n")
    _write(target / "iterative_native_plan_blueprint.json", artifacts.model.to_json() + "\n")
    _write(
        target / "iterative_native_plan_training_report.json",
        json.dumps(asdict(artifacts.report), indent=2, sort_keys=True) + "\n",
    )
    _write(target / "phase6p_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6p_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
