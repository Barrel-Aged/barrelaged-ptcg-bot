"""Collect, resume, train, calibrate, and write Phase 6AH evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_expected_value_surfaces import (  # noqa: E402
    ExpectedValueDataset,
    ExpectedValueRoot,
    collect_expected_value_dataset,
    run_phase6_expected_value_surfaces,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_shards(directory: Path) -> tuple[ExpectedValueRoot, ...]:
    if not directory.is_dir():
        return ()
    roots = []
    for path in sorted(directory.glob("*.json")):
        roots.append(
            ExpectedValueRoot.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        )
    return tuple(roots)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_expected_value_surfaces",
    )
    parser.add_argument(
        "--shard-directory",
        type=Path,
        default=PROJECT_ROOT / "work" / "phase6ah_root_shards",
    )
    parser.add_argument("--reuse-source", action="store_true")
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    shards = args.shard_directory.resolve()
    dataset_path = target / "expected_value_surface_dataset.json"

    if args.reuse_source:
        if not dataset_path.is_file():
            raise FileNotFoundError(
                "--reuse-source requires expected_value_surface_dataset.json"
            )
        dataset = ExpectedValueDataset.from_dict(
            json.loads(dataset_path.read_text(encoding="utf-8"))
        )
    else:
        existing = _load_shards(shards)

        def persist_root(root: ExpectedValueRoot, complete: int, total: int) -> None:
            _write(shards / f"{root.match_id}.json", root.to_json() + "\n")
            print(
                json.dumps(
                    {
                        "phase": "phase6ah_collection",
                        "complete_roots": complete,
                        "total_roots": total,
                        "match_id": root.match_id,
                        "root_id": root.root_id,
                        "terminal_rollouts": root.terminal_rollout_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        dataset = collect_expected_value_dataset(
            project_root=PROJECT_ROOT,
            existing_roots=existing,
            root_observer=persist_root,
        )
        _write(dataset_path, dataset.to_json() + "\n")

    artifacts = run_phase6_expected_value_surfaces(
        project_root=PROJECT_ROOT,
        source_dataset=dataset,
    )
    _write(target / "expected_value_model.json", artifacts.model.to_json() + "\n")
    _write(
        target / "expected_value_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6ah_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6ah_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "root_count": artifacts.evidence.root_count,
                "surface_count": artifacts.evidence.surface_count,
                "action_target_count": artifacts.evidence.action_target_count,
                "terminal_rollout_count": artifacts.evidence.terminal_rollout_count,
                "maximum_target_standard_error": (
                    artifacts.evidence.maximum_target_standard_error
                ),
                "validation_mse": artifacts.evidence.validation_mse,
                "validation_zero_mse": artifacts.evidence.validation_zero_mse,
                "validation_heuristic_mse": (
                    artifacts.evidence.validation_heuristic_mse
                ),
                "validation_action_contrast_mse": (
                    artifacts.evidence.validation_action_contrast_mse
                ),
                "validation_zero_action_contrast_mse": (
                    artifacts.evidence.validation_zero_action_contrast_mse
                ),
                "validation_coverage": artifacts.evidence.validation_coverage,
                "validation_mean_width": artifacts.evidence.validation_mean_width,
                "minimum_validation_slice_coverage": (
                    artifacts.evidence.minimum_validation_slice_coverage
                ),
                "sampling_precision_gate_passed": (
                    artifacts.evidence.sampling_precision_gate_passed
                ),
                "implementation_heldout_point_model_validated": (
                    artifacts.evidence.implementation_heldout_point_model_validated
                ),
                "implementation_heldout_conditional_expected_mean_admitted": (
                    artifacts.evidence.implementation_heldout_conditional_expected_mean_admitted
                ),
                "counterfactual_value_compatible": False,
                "equilibrium_convergence_validated": False,
                "safe_resolver_enabled": False,
                "opponent_exploitation_enabled": False,
                "official_strategy_control_enabled": False,
                "artifact_directory": str(target),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
