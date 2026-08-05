"""Generate durable Phase 6M native endgame external-sampling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_external_sampling import (  # noqa: E402
    run_phase6_native_external_sampling,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reservoir-capacity", type=int, default=1_024)
    parser.add_argument("--maximum-root-actions", type=int, default=32)
    parser.add_argument("--maximum-transitions", type=int, default=1_000)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=6_020)
    parser.add_argument(
        "--source-blueprint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_blueprint"
            / "native_regret_blueprint.json"
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_external_sampling",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_external_sampling(
        project_root=PROJECT_ROOT,
        source_blueprint_path=args.source_blueprint,
        reservoir_capacity=args.reservoir_capacity,
        maximum_root_actions=args.maximum_root_actions,
        maximum_transitions=args.maximum_transitions,
        maximum_attempts=args.maximum_attempts,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(target / "native_external_sampling_dataset.json", artifacts.dataset.to_json() + "\n")
    _write(target / "phase6m_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6m_evidence.md", artifacts.evidence.to_markdown())
    print(json.dumps({
        "status": artifacts.evidence.status,
        "evidence_fingerprint": artifacts.evidence.fingerprint,
        "dataset_fingerprint": artifacts.dataset.fingerprint,
        "artifact_directory": str(target),
        "claims": dict(artifacts.evidence.claims),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
