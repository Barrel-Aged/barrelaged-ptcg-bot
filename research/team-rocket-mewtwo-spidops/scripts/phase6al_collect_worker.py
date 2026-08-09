"""Collect one disjoint worker slice of a Phase 6AL panel."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.beliefs import OPPONENT_INTENTS  # noqa: E402
from rocketbot.phase6_branch_conditioned_values import (  # noqa: E402
    _build_branch_conditioned_root,
)
from rocketbot.phase6_expected_value_surfaces import (  # noqa: E402
    PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    _capture_complete_match_late_root,
    _load_expected_value_sources,
)
from rocketbot.phase6_high_precision_confirmation import (  # noqa: E402
    HighPrecisionCalibration,
    HighPrecisionModelReference,
    HighPrecisionPanel,
    HighPrecisionRoot,
    _batch_seeds,
    _load_source,
    _role_namespace,
)


def _atomic_write(path: Path, text: str, worker_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.worker-{worker_index}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("calibration", "validation"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--shard-directory", type=Path)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "phase6_high_precision_confirmation",
    )
    args = parser.parse_args()
    if args.worker_count <= 0 or args.worker_index not in range(args.worker_count):
        raise ValueError("worker index/count are invalid")
    shard_directory = (
        args.shard_directory
        or PROJECT_ROOT / "work" / f"phase6al_{args.role}_shards"
    ).resolve()
    artifact_directory = args.artifact_directory.resolve()
    reference_path = artifact_directory / "high_precision_model_reference.json"
    if not reference_path.is_file():
        raise FileNotFoundError("Phase 6AL worker requires the frozen model reference")
    reference = HighPrecisionModelReference.from_dict(
        json.loads(reference_path.read_text(encoding="utf-8"))
    )
    model, _, _, _ = _load_source(PROJECT_ROOT)
    if reference.source_model_fingerprint != model.fingerprint:
        raise ValueError("Phase 6AL worker reference/model binding failed")
    if args.role == "validation":
        panel_path = artifact_directory / "calibration_panel.json"
        calibration_path = artifact_directory / "high_precision_calibration.json"
        if not panel_path.is_file() or not calibration_path.is_file():
            raise FileNotFoundError(
                "Phase 6AL validation cannot start before frozen calibration"
            )
        panel = HighPrecisionPanel.from_dict(
            json.loads(panel_path.read_text(encoding="utf-8"))
        )
        calibration = HighPrecisionCalibration.from_dict(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
        if (
            panel.role != "calibration"
            or panel.source_reference_fingerprint != reference.fingerprint
            or calibration.source_reference_fingerprint != reference.fingerprint
            or calibration.calibration_panel_fingerprint != panel.fingerprint
        ):
            raise ValueError("Phase 6AL validation prerequisite binding failed")

    sources = _load_expected_value_sources(PROJECT_ROOT)
    runtime_text = str(sources.runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

    keys = tuple(
        (candidate.archetype, intent.value)
        for candidate in sources.candidates
        for intent in OPPONENT_INTENTS
    )
    candidates = {item.archetype: item for item in sources.candidates}
    intents = {item.value: item for item in OPPONENT_INTENTS}
    assigned = [
        (ordinal, key)
        for ordinal, key in enumerate(keys)
        if ordinal % args.worker_count == args.worker_index
    ]
    completed = 0
    prefix = f"{_role_namespace(args.role)}-action_first-"
    for ordinal, (candidate_id, intent_id) in assigned:
        match_id = f"{prefix}{candidate_id}-{intent_id}-r0"
        path = shard_directory / f"{match_id}.json"
        if path.is_file():
            completed += 1
            continue
        last_error: Exception | None = None
        for attempt in range(1, PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS + 1):
            try:
                capture = _capture_complete_match_late_root(
                    battle_start=battle_start,
                    battle_select=battle_select,
                    battle_finish=battle_finish,
                    target=sources.target,
                    candidate=candidates[candidate_id],
                    intent=intents[intent_id],
                    policy_variant="action_first",
                    replica=0,
                    capture_attempt=attempt,
                    maximum_live_selections=PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
                    match_id=match_id,
                )
                batches = tuple(
                    _build_branch_conditioned_root(
                        deepcopy(capture),
                        sources=sources,
                        candidate=candidates[candidate_id],
                        intent=intents[intent_id],
                        policy_variant="action_first",
                        split="validation",
                        replica=0,
                        match_id=match_id,
                        seed=batch_seed,
                        maximum_rollout_transitions=(
                            PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS
                        ),
                        maximum_tactical_transitions=(
                            PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS
                        ),
                        experiment_label="Phase 6AL",
                    )
                    for batch_seed in _batch_seeds(args.role)
                )
                root = HighPrecisionRoot(
                    role=args.role,
                    batch_seeds=_batch_seeds(args.role),
                    batches=batches,
                )
                _atomic_write(path, root.to_json() + "\n", args.worker_index)
                completed += 1
                print(
                    json.dumps(
                        {
                            "role": args.role,
                            "worker": args.worker_index,
                            "assigned_complete": completed,
                            "assigned_total": len(assigned),
                            "global_ordinal": ordinal + 1,
                            "match_id": match_id,
                            "root_id": root.root_id,
                            "capture_attempt": (
                                root.batches[0].value_root.capture_attempt
                            ),
                            "replay_batches": len(root.batches),
                            "terminal_rollouts": root.terminal_rollout_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"worker {args.worker_index} exhausted attempts for {match_id}"
            ) from last_error
    print(
        json.dumps(
            {
                "role": args.role,
                "worker": args.worker_index,
                "status": "complete",
                "assigned_complete": completed,
                "assigned_total": len(assigned),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
