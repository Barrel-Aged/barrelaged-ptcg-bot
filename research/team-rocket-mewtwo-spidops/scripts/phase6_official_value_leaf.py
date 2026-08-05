"""Collect, train, calibrate, and write Phase 6AG value-leaf evidence."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_official_value_leaf import (  # noqa: E402
    PHASE6_OFFICIAL_VALUE_LEAF_EPOCHS,
    PHASE6_OFFICIAL_VALUE_LEAF_LEARNING_RATE,
    collect_official_value_leaf_dataset,
    rebalance_official_value_leaf_manifest,
    run_phase6_official_value_leaf,
)
from rocketbot.official_trajectory import OfficialTrajectoryDataset  # noqa: E402


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_official_value_leaf",
    )
    parser.add_argument("--reuse-source", action="store_true")
    parser.add_argument("--rebalance-source", action="store_true")
    parser.add_argument("--epochs", type=int, default=PHASE6_OFFICIAL_VALUE_LEAF_EPOCHS)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=PHASE6_OFFICIAL_VALUE_LEAF_LEARNING_RATE,
    )
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    dataset_path = target / "implementation_heldout_trajectory_dataset.json"
    manifest_path = target / "official_value_leaf_manifest.json"
    if args.reuse_source:
        if not dataset_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "--reuse-source requires the Phase 6AG dataset and manifest"
            )
    else:
        dataset, manifest = collect_official_value_leaf_dataset(
            project_root=PROJECT_ROOT
        )
        _write(dataset_path, dataset.to_json() + "\n")
        _write(manifest_path, manifest.to_json() + "\n")
    if args.rebalance_source:
        dataset = OfficialTrajectoryDataset.from_dict(
            json.loads(dataset_path.read_text(encoding="utf-8"))
        )
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = rebalance_official_value_leaf_manifest(raw_manifest, dataset)
        _write(manifest_path, manifest.to_json() + "\n")
        del dataset, raw_manifest, manifest
        gc.collect()
    artifacts = run_phase6_official_value_leaf(
        project_root=PROJECT_ROOT,
        source_dataset_path=dataset_path,
        source_manifest_path=manifest_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    _write(target / "official_value_leaf_model.json", artifacts.model.to_json() + "\n")
    _write(
        target / "official_value_leaf_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6ag_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6ag_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "manifest_fingerprint": artifacts.manifest.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "match_count": artifacts.evidence.match_count,
                "episode_count": artifacts.evidence.episode_count,
                "trajectory_step_count": artifacts.evidence.trajectory_step_count,
                "live_selection_count": artifacts.evidence.live_selection_count,
                "validation_episode_mse": (
                    artifacts.evidence.validation_episode_mse
                ),
                "validation_zero_episode_mse": (
                    artifacts.evidence.validation_zero_episode_mse
                ),
                "validation_heuristic_episode_mse": (
                    artifacts.evidence.validation_heuristic_episode_mse
                ),
                "validation_coverage": artifacts.evidence.validation_coverage,
                "validation_mean_width": artifacts.evidence.validation_mean_width,
                "validation_endgame_coverage": (
                    artifacts.evidence.validation_endgame_coverage
                ),
                "validation_endgame_mean_width": (
                    artifacts.evidence.validation_endgame_mean_width
                ),
                "implementation_heldout_point_leaf_validated": (
                    artifacts.evidence.implementation_heldout_point_leaf_validated
                ),
                "implementation_heldout_value_leaf_calibrated": (
                    artifacts.evidence.implementation_heldout_value_leaf_calibrated
                ),
                "counterfactual_value_compatible": False,
                "safe_resolver_enabled": False,
                "opponent_exploitation_enabled": False,
                "official_strategy_control_enabled": False,
                "artifact_directory": str(target),
                "claims": dict(artifacts.evidence.claims),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
