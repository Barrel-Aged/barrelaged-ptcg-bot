"""Generate durable Phase 6A-6B public-replay and exact-game evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_foundation import run_phase6_foundation  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfr-iterations", type=int, default=20_000)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_game_theory_foundation",
    )
    args = parser.parse_args()
    artifacts = run_phase6_foundation(
        project_root=PROJECT_ROOT,
        cfr_iterations=args.cfr_iterations,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "public_replay_dataset.json",
        artifacts.public_replays.to_json() + "\n",
    )
    _write(
        target / "phase6ab_evidence.json",
        artifacts.evidence.to_json() + "\n",
    )
    _write(target / "phase6ab_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "public_replay_dataset_fingerprint": (
                    artifacts.public_replays.fingerprint
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
