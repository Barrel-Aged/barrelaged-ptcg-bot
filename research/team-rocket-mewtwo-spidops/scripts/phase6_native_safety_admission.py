"""Generate deterministic Phase 6Q native safety-admission evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_safety_admission import (  # noqa: E402
    run_phase6_native_safety_admission,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    artifacts = run_phase6_native_safety_admission(project_root=PROJECT_ROOT)
    target = PROJECT_ROOT / "artifacts" / "phase6_native_safety_admission"
    _write(target / "native_plan_safety_admission.json", artifacts.admission.to_json() + "\n")
    _write(target / "phase6q_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6q_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "admission_fingerprint": artifacts.admission.fingerprint,
                "admitted": artifacts.admission.admitted,
                "failed_check_ids": [
                    item.check_id for item in artifacts.admission.failed_checks
                ],
                "artifact_directory": str(target.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
