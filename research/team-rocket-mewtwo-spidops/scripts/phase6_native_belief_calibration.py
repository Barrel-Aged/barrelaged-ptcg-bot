"""Generate Phase 6T held-out allocation/public-range evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_native_belief_calibration import (  # noqa: E402
    run_phase6_native_belief_calibration,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent-roots", type=int, default=2)
    parser.add_argument("--particle-variants", type=int, default=3)
    parser.add_argument("--observations-per-particle", type=int, default=6)
    parser.add_argument("--maximum-transitions-per-particle", type=int, default=256)
    parser.add_argument("--seed", type=int, default=6_026)
    parser.add_argument(
        "--source-replay-dataset",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_official_trajectory"
            / "official_trajectory_dataset.json"
        ),
    )
    parser.add_argument(
        "--source-range-model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_trajectory_neural"
            / "logged_trajectory_bootstrap.json"
        ),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_native_belief_calibration",
    )
    args = parser.parse_args()
    artifacts = run_phase6_native_belief_calibration(
        project_root=PROJECT_ROOT,
        source_replay_dataset_path=args.source_replay_dataset,
        source_range_model_path=args.source_range_model,
        independent_roots=args.independent_roots,
        particle_variants=args.particle_variants,
        observations_per_particle=args.observations_per_particle,
        maximum_transitions_per_particle=args.maximum_transitions_per_particle,
        seed=args.seed,
    )
    target = args.artifact_directory.resolve()
    _write(target / "native_allocation_range_dataset.json", artifacts.dataset.to_json() + "\n")
    _write(target / "native_belief_calibration.json", artifacts.calibration.to_json() + "\n")
    _write(target / "phase6t_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6t_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "allocation_generalization_gate_passed": (
                    artifacts.calibration.allocation_generalization_gate_passed
                ),
                "public_belief_calibrated": artifacts.calibration.public_belief_calibrated,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
