"""Collect a deterministic disjoint subset of Phase 6AI root shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.beliefs import OPPONENT_INTENTS  # noqa: E402
from rocketbot.phase6_branch_conditioned_values import (  # noqa: E402
    PHASE6_BRANCH_VALUE_SEED,
    _build_branch_conditioned_root,
)
from rocketbot.phase6_expected_value_surfaces import (  # noqa: E402
    PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL,
    _capture_complete_match_late_root,
    _load_expected_value_sources,
    _protocol_by_variant,
)


def _atomic_write(path: Path, text: str, worker_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".json.worker-{worker_index}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument(
        "--shard-directory",
        type=Path,
        default=PROJECT_ROOT / "work" / "phase6ai_root_shards",
    )
    args = parser.parse_args()
    if args.worker_count <= 0 or args.worker_index not in range(args.worker_count):
        raise ValueError("worker index/count are invalid")

    sources = _load_expected_value_sources(PROJECT_ROOT)
    runtime_text = str(sources.runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

    keys = tuple(
        (variant, candidate.archetype, intent.value, replica)
        for variant, _, replicas in PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL
        for candidate in sources.candidates
        for intent in OPPONENT_INTENTS
        for replica in range(replicas)
    )
    candidate_by_id = {item.archetype: item for item in sources.candidates}
    intent_by_id = {item.value: item for item in OPPONENT_INTENTS}
    assigned = [
        (ordinal, key)
        for ordinal, key in enumerate(keys)
        if ordinal % args.worker_count == args.worker_index
    ]
    completed = 0
    for ordinal, key in assigned:
        variant, candidate_id, intent_id, replica = key
        split, _ = _protocol_by_variant()[variant]
        match_id = f"phase6ai-{split}-{variant}-{candidate_id}-{intent_id}-r{replica}"
        path = args.shard_directory.resolve() / f"{match_id}.json"
        if path.is_file():
            completed += 1
            continue
        candidate = candidate_by_id[candidate_id]
        intent = intent_by_id[intent_id]
        last_error: Exception | None = None
        for attempt in range(1, PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS + 1):
            try:
                capture = _capture_complete_match_late_root(
                    battle_start=battle_start,
                    battle_select=battle_select,
                    battle_finish=battle_finish,
                    target=sources.target,
                    candidate=candidate,
                    intent=intent,
                    policy_variant=variant,
                    replica=replica,
                    capture_attempt=attempt,
                    maximum_live_selections=(
                        PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS
                    ),
                    match_id=match_id,
                )
                root = _build_branch_conditioned_root(
                    capture,
                    sources=sources,
                    candidate=candidate,
                    intent=intent,
                    policy_variant=variant,
                    split=split,
                    replica=replica,
                    match_id=match_id,
                    seed=PHASE6_BRANCH_VALUE_SEED,
                    maximum_rollout_transitions=(
                        PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS
                    ),
                    maximum_tactical_transitions=(
                        PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS
                    ),
                )
                _atomic_write(path, root.to_json() + "\n", args.worker_index)
                completed += 1
                print(
                    json.dumps(
                        {
                            "worker": args.worker_index,
                            "assigned_complete": completed,
                            "assigned_total": len(assigned),
                            "global_ordinal": ordinal + 1,
                            "match_id": match_id,
                            "root_id": root.root_id,
                            "capture_attempt": root.value_root.capture_attempt,
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
