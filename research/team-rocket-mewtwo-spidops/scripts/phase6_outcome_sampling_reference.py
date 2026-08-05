"""Generate durable Phase 6N outcome-sampling reference evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_outcome_sampling_reference import (  # noqa: E402
    run_phase6_outcome_sampling_reference,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-iterations", type=int, default=20_000)
    parser.add_argument("--exploration", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=6_021)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_outcome_sampling_reference",
    )
    args = parser.parse_args()
    artifacts = run_phase6_outcome_sampling_reference(
        training_iterations=args.training_iterations,
        exploration=args.exploration,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(
        target / "expectation_audit.json",
        json.dumps(artifacts.audit.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    _write(
        target / "training_result.json",
        json.dumps(artifacts.training.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    _write(target / "phase6n_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6n_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "audit_fingerprint": artifacts.audit.fingerprint,
                "solver_fingerprint": artifacts.training.solver_fingerprint,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
