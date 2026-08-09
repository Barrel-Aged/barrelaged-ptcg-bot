"""Run and persist Phase 6E interruption-safe exact resolving evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_resolving import run_phase6_safe_resolving  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_safe_resolving",
    )
    args = parser.parse_args()
    evidence = run_phase6_safe_resolving(project_root=PROJECT_ROOT)
    target = args.artifact_directory.resolve()
    _write(target / "phase6e_evidence.json", evidence.to_json() + "\n")
    _write(target / "phase6e_evidence.md", evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": evidence.status,
                "evidence_fingerprint": evidence.fingerprint,
                "total_exact_actor_value_gain": evidence.total_exact_actor_value_gain,
                "endgame_changed": evidence.endgame.changed_strategies,
                "continual_changed": evidence.continual_root.changed_strategies,
                "artifact_directory": str(target),
                "claims": dict(evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
