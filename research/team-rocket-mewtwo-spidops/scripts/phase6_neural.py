"""Train, compare, persist, and verify Phase 6C exact-game neural blueprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_neural import run_phase6_neural_blueprint  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfr-iterations", type=int, default=20_000)
    parser.add_argument("--policy-epochs", type=int, default=2_000)
    parser.add_argument("--regret-epochs", type=int, default=4_000)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_neural_blueprint",
    )
    args = parser.parse_args()
    artifacts = run_phase6_neural_blueprint(
        project_root=PROJECT_ROOT,
        cfr_iterations=args.cfr_iterations,
        policy_epochs=args.policy_epochs,
        regret_epochs=args.regret_epochs,
    )
    target = args.artifact_directory.resolve()
    _write(target / "phase6c_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6c_evidence.md", artifacts.evidence.to_markdown())
    _write(
        target / "policy_candidate.json",
        json.dumps(
            artifacts.policy_run.artifact.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write(
        target / "regret_candidate.json",
        json.dumps(
            artifacts.regret_run.artifact.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write(
        target / "selected_neural_blueprint.json",
        json.dumps(
            artifacts.selected_artifact.to_dict(),
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
                "selected_strategy_mode": artifacts.evidence.selected_strategy_mode,
                "selected_artifact_fingerprint": (
                    artifacts.evidence.selected_artifact_fingerprint
                ),
                "selected_nash_conv": artifacts.evidence.selected_nash_conv,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
