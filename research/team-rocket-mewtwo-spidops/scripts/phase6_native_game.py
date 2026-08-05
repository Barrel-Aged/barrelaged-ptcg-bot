"""Run and persist Phase 6J branchable native-search-game evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_game import run_phase6_native_search_game  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_search_game",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_search_game(project_root=PROJECT_ROOT)
    target = args.artifact_directory.resolve()
    _write(target / "phase6j_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6j_evidence.md", artifacts.evidence.to_markdown())
    _write(
        target / "hidden_completion_manifests.json",
        json.dumps(
            {
                "schema_version": 1,
                "game_fingerprint": artifacts.evidence.game_fingerprint,
                "raw_private_card_ids_persisted": 0,
                "manifests": [
                    dict(item) for item in artifacts.evidence.completion_manifests
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "game_fingerprint": artifacts.evidence.game_fingerprint,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

