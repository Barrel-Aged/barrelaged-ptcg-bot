"""Calibrate, audit, and persist Phase 6D exact-game value bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_bounds import run_phase6_counterfactual_bounds  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_counterfactual_bounds",
    )
    args = parser.parse_args()
    artifacts = run_phase6_counterfactual_bounds(project_root=PROJECT_ROOT)
    target = args.artifact_directory.resolve()
    _write(target / "phase6d_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6d_evidence.md", artifacts.evidence.to_markdown())
    _write(
        target / "counterfactual_calibration.json",
        json.dumps(artifacts.calibration.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "validation_coverage": artifacts.evidence.validation_coverage,
                "false_safe_certificates": (
                    artifacts.evidence.audit.false_safe_certificates
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
