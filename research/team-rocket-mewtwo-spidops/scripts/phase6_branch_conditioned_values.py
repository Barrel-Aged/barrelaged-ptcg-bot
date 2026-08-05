"""Collect, resume, train, calibrate, and write Phase 6AI evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.phase6_branch_conditioned_values import (  # noqa: E402
    BranchConditionedDataset,
    BranchConditionedRoot,
    collect_branch_conditioned_dataset,
    run_phase6_branch_conditioned_values,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_shards(directory: Path) -> tuple[BranchConditionedRoot, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        BranchConditionedRoot.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in sorted(directory.glob("*.json"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_branch_conditioned_values",
    )
    parser.add_argument(
        "--shard-directory",
        type=Path,
        default=PROJECT_ROOT / "work" / "phase6ai_root_shards",
    )
    parser.add_argument("--reuse-source", action="store_true")
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    shards = args.shard_directory.resolve()
    dataset_path = target / "branch_conditioned_dataset.json"

    if args.reuse_source:
        if not dataset_path.is_file():
            raise FileNotFoundError(
                "--reuse-source requires branch_conditioned_dataset.json"
            )
        dataset = BranchConditionedDataset.from_dict(
            json.loads(dataset_path.read_text(encoding="utf-8"))
        )
    else:
        existing = _load_shards(shards)

        def persist_root(
            root: BranchConditionedRoot,
            complete: int,
            total: int,
        ) -> None:
            _write(shards / f"{root.match_id}.json", root.to_json() + "\n")
            print(
                json.dumps(
                    {
                        "phase": "phase6ai_collection",
                        "complete_roots": complete,
                        "total_roots": total,
                        "match_id": root.match_id,
                        "root_id": root.root_id,
                        "branch_observations": len(root.branch_observations),
                        "terminal_rollouts": root.value_root.terminal_rollout_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        dataset = collect_branch_conditioned_dataset(
            project_root=PROJECT_ROOT,
            existing_roots=existing,
            root_observer=persist_root,
        )
        _write(dataset_path, dataset.to_json() + "\n")

    artifacts = run_phase6_branch_conditioned_values(
        project_root=PROJECT_ROOT,
        source_dataset=dataset,
    )
    _write(target / "branch_value_model.json", artifacts.model.to_json() + "\n")
    _write(
        target / "branch_value_calibration.json",
        artifacts.calibration.to_json() + "\n",
    )
    _write(target / "phase6ai_evidence.json", artifacts.evidence.to_json() + "\n")
    _write(target / "phase6ai_evidence.md", artifacts.evidence.to_markdown())
    print(
        json.dumps(
            {
                "status": artifacts.evidence.status,
                "evidence_fingerprint": artifacts.evidence.fingerprint,
                "dataset_fingerprint": artifacts.dataset.fingerprint,
                "model_fingerprint": artifacts.model.fingerprint,
                "calibration_fingerprint": artifacts.calibration.fingerprint,
                "root_count": artifacts.evidence.root_count,
                "branch_observation_count": (
                    artifacts.evidence.branch_observation_count
                ),
                "terminal_rollout_count": (
                    artifacts.evidence.terminal_rollout_count
                ),
                "validation_mse": artifacts.evidence.validation_mse,
                "validation_zero_mse": artifacts.evidence.validation_zero_mse,
                "validation_root_heuristic_mse": (
                    artifacts.evidence.validation_root_heuristic_mse
                ),
                "validation_branch_heuristic_mse": (
                    artifacts.evidence.validation_branch_heuristic_mse
                ),
                "validation_action_contrast_mse": (
                    artifacts.evidence.validation_action_contrast_mse
                ),
                "validation_zero_action_contrast_mse": (
                    artifacts.evidence.validation_zero_action_contrast_mse
                ),
                "validation_branch_action_contrast_mse": (
                    artifacts.evidence.validation_branch_action_contrast_mse
                ),
                "minimum_contrast_bootstrap_lower_bound": (
                    artifacts.evidence.minimum_contrast_bootstrap_lower_bound
                ),
                "validation_coverage": artifacts.evidence.validation_coverage,
                "validation_mean_width": artifacts.evidence.validation_mean_width,
                "minimum_validation_slice_coverage": (
                    artifacts.evidence.minimum_validation_slice_coverage
                ),
                "maximum_target_standard_error": (
                    artifacts.evidence.maximum_target_standard_error
                ),
                "sampling_precision_gate_passed": (
                    artifacts.evidence.sampling_precision_gate_passed
                ),
                "action_sensitive_point_model_validated": (
                    artifacts.evidence.action_sensitive_point_model_validated
                ),
                "conditional_expected_mean_model_admitted": (
                    artifacts.evidence.conditional_expected_mean_model_admitted
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
