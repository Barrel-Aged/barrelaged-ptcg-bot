"""Generate durable Phase 6L MCCFR-reference evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_mccfr_reference import run_phase6_mccfr_reference  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=6_019)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_mccfr_reference",
    )
    args = parser.parse_args()
    artifacts = run_phase6_mccfr_reference(
        training_iterations=args.training_iterations,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(target / "expectation_audit.json", json.dumps(
        artifacts.audit.to_dict(), indent=2, sort_keys=True
    ) + "\n")
    _write(target / "training_result.json", json.dumps(
        artifacts.training.to_dict(), indent=2, sort_keys=True
    ) + "\n")
    _write(target / "phase6l_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6l_evidence.md", artifacts.evidence.to_markdown())
    print(json.dumps({
        "status": artifacts.evidence.status,
        "evidence_fingerprint": artifacts.evidence.fingerprint,
        "audit_fingerprint": artifacts.audit.fingerprint,
        "solver_fingerprint": artifacts.training.solver_fingerprint,
        "artifact_directory": str(target),
        "claims": dict(artifacts.evidence.claims),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
