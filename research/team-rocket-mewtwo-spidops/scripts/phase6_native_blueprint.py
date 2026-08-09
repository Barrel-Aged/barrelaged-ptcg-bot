"""Train and persist Phase 6K native regret-blueprint evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_blueprint import (  # noqa: E402
    run_phase6_native_regret_blueprint,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-rounds", type=int, default=2)
    parser.add_argument("--epochs-per-round", type=int, default=40)
    parser.add_argument("--maximum-branch-actions", type=int, default=6)
    parser.add_argument("--rollouts-per-action", type=int, default=1)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_blueprint",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_regret_blueprint(
        project_root=PROJECT_ROOT,
        training_rounds=args.training_rounds,
        epochs_per_round=args.epochs_per_round,
        maximum_branch_actions=args.maximum_branch_actions,
        rollouts_per_action=args.rollouts_per_action,
    )
    target = args.artifact_directory.resolve()
    _write(target / "native_regret_dataset.json", artifacts.dataset.to_json() + "\n")
    _write(
        target / "native_regret_blueprint.json",
        artifacts.training.artifact.to_json() + "\n",
    )
    _write(target / "phase6k_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6k_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.training.artifact.fingerprint,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

