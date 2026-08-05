"""Collect one fixed worker slice of the Phase 6AM validation panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.beliefs import OPPONENT_INTENTS  # noqa: E402
from rocketbot.phase6_expected_value_surfaces import (  # noqa: E402
    PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    _capture_complete_match_late_root,
    _load_expected_value_sources,
)
from rocketbot.phase6_public_range_counterfactual_advantages import (  # noqa: E402
    PublicRangeCounterfactualReference,
    _load_range_model,
    build_public_range_counterfactual_root,
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
        default=PROJECT_ROOT / "work" / "phase6am_validation_shards",
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "phase6_public_range_counterfactual_advantages"
        ),
    )
    args = parser.parse_args()
    if args.worker_count <= 0 or args.worker_index not in range(args.worker_count):
        raise ValueError("Phase 6AM worker index/count is invalid")
    reference_path = args.artifact_directory / "counterfactual_advantage_reference.json"
    if not reference_path.is_file():
        raise FileNotFoundError("Phase 6AM worker requires a frozen reference")
    if (args.artifact_directory / "validation_panel.json").exists():
        raise RuntimeError("Phase 6AM validation panel is already sealed")
    reference = PublicRangeCounterfactualReference.from_dict(
        json.loads(reference_path.read_text(encoding="utf-8"))
    )
    sources = _load_expected_value_sources(PROJECT_ROOT)
    range_model = _load_range_model(PROJECT_ROOT)
    runtime_text = str(sources.runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

    candidates = {item.archetype: item for item in sources.candidates}
    intents = {item.value: item for item in OPPONENT_INTENTS}
    keys = tuple(
        (candidate.archetype, intent.value)
        for candidate in sources.candidates
        for intent in OPPONENT_INTENTS
    )
    assigned = [
        (ordinal, key)
        for ordinal, key in enumerate(keys)
        if ordinal % args.worker_count == args.worker_index
    ]
    completed = 0
    for ordinal, (candidate_id, intent_id) in assigned:
        match_id = f"phase6am-validation-action_first-{candidate_id}-{intent_id}-r0"
        path = args.shard_directory / f"{match_id}.json"
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
                root = build_public_range_counterfactual_root(
                    capture,
                    sources=sources,
                    range_model=range_model,
                    true_candidate=candidates[candidate_id],
                    intent=intents[intent_id],
                    match_id=match_id,
                    reference=reference,
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
                            "capture_attempt": root.batches[0].value_root.capture_attempt,
                            "true_candidate": candidate_id,
                            "top_candidate": root.top_candidate_id,
                            "true_candidate_probability": root.true_candidate_probability,
                            "terminal_rollouts": sum(
                                batch.value_root.terminal_rollout_count
                                for batch in root.batches
                            ),
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
                f"Phase 6AM worker {args.worker_index} exhausted attempts for {match_id}"
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
