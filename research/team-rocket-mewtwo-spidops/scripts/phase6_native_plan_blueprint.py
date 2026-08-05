"""Generate durable Phase 6O native turn-plan blueprint evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_plan_blueprint import (  # noqa: E402
    run_phase6_native_plan_blueprint,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-iterations", type=int, default=3)
    parser.add_argument("--exploration", type=float, default=0.2)
    parser.add_argument("--reservoir-capacity", type=int, default=8_192)
    parser.add_argument("--maximum-macro-transitions", type=int, default=256)
    parser.add_argument("--maximum-tactical-transitions", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=6_022)
    parser.add_argument(
        "--source-blueprint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_blueprint"
            / "native_regret_blueprint.json"
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_plan_blueprint",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_plan_blueprint(
        project_root=PROJECT_ROOT,
        source_blueprint_path=args.source_blueprint,
        sampling_iterations=args.sampling_iterations,
        exploration=args.exploration,
        reservoir_capacity=args.reservoir_capacity,
        maximum_macro_transitions=args.maximum_macro_transitions,
        maximum_tactical_transitions=args.maximum_tactical_transitions,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(target / "native_plan_outcome_dataset.json", artifacts.dataset.to_json() + "\n")
    _write(target / "native_plan_blueprint.json", artifacts.model.to_json() + "\n")
    _write(
        target / "native_plan_training_report.json",
        json.dumps(asdict(artifacts.report), indent=2, sort_keys=True) + "\n",
    )
    _write(target / "phase6o_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6o_evidence.md", artifacts.evidence.to_markdown())
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
