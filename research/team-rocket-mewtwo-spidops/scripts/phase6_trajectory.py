"""Generate durable Phase 6H logged official-trajectory evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_trajectory import run_phase6_official_trajectory  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_official_trajectory",
    )
    args = parser.parse_args()
    artifacts = run_phase6_official_trajectory(project_root=PROJECT_ROOT)
    target = args.artifact_directory.resolve()
    _write(
        target / "official_trajectory_dataset.json",
        artifacts.dataset.to_json() + "\n",
    )
    _write(
        target / "phase6h_evidence.json",
        artifacts.evidence.to_json() + "\n",
    )
    _write(target / "phase6h_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "trajectory_dataset_fingerprint": artifacts.dataset.fingerprint,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
