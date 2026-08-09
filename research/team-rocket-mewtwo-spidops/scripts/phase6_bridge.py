"""Run and persist the Phase 6G live official shadow-bridge audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_bridge import run_phase6_official_bridge_audit  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-directory",
        type=Path,
        default=PROJECT_ROOT / "work" / "official_kaggle" / "sample_submission",
    )
    parser.add_argument("--match-limit", type=int)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_official_bridge",
    )
    args = parser.parse_args()
    artifacts = run_phase6_official_bridge_audit(
        project_root=PROJECT_ROOT,
        runtime_directory=args.runtime_directory,
        match_limit=args.match_limit,
    )
    target = args.artifact_directory.resolve()
    _write(target / "phase6g_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6g_evidence.md", artifacts.evidence.to_markdown())
    _write(
        target / "bridge_decisions.json",
        json.dumps(
            {
                "schema_version": 1,
                "decisions_fingerprint": artifacts.evidence.decisions_fingerprint,
                "records": [asdict(item) for item in artifacts.decisions],
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
                "matches": len(artifacts.evidence.matches),
                "target_decisions": artifacts.evidence.target_decisions,
                "bridge_errors": artifacts.evidence.bridge_errors,
                "exact_selection_roundtrips": (
                    artifacts.evidence.exact_selection_roundtrips
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
