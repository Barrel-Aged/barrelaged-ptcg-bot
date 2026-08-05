"""Train and persist Phase 6I logged-trajectory neural-bootstrap evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_trajectory_neural import (  # noqa: E402
    run_phase6_trajectory_neural_bootstrap,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_trajectory_neural",
    )
    args = parser.parse_args()
    artifacts = run_phase6_trajectory_neural_bootstrap(
        project_root=PROJECT_ROOT,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "logged_trajectory_bootstrap.json",
        artifacts.artifact.to_json() + "\n",
    )
    _write(
        target / "phase6i_evidence.json",
        artifacts.evidence.to_json() + "\n",
    )
    _write(target / "phase6i_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "artifact_fingerprint": artifacts.artifact.fingerprint,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

