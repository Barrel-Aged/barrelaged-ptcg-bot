#!/usr/bin/env python3
"""Collect N6R paired rollout targets with exact policy-feature snapshots."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys
from time import time
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from collect_belief_rollout_teacher_n6 import (  # noqa: E402
    BeliefRolloutRuntime,
    DECK_SHA256,
    MODEL_SHA256,
)
from collect_live_truth_causal_action_values import sha256, utility  # noqa: E402
from rocketbot.policy_feature_snapshot import serialize_policy_feature  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "team_rocket_recovery"
        / "candidates"
        / "scale_neural_resolver_safe_tie_v3",
    )
    parser.add_argument(
        "--own-candidate",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "team_rocket_recovery"
        / "candidates"
        / "scale_value_calibrated_v1",
    )
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", choices=("first", "second", "balanced", "any"), default="second")
    parser.add_argument("--roots-per-game", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument("--coin-rollouts", type=int, default=2)
    parser.add_argument("--particle-offset", type=int, default=0)
    parser.add_argument("--earliest-turn", type=int, default=1)
    parser.add_argument("--latest-turn", type=int, default=8)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=600)
    parser.add_argument("--root-budget-seconds", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "games",
        "roots_per_game",
        "candidates",
        "particles",
        "coin_rollouts",
        "max_live_decisions",
        "max_rollout_decisions",
        "threads",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.particle_offset < 0 or args.root_budget_seconds <= 0:
        raise ValueError("invalid N6R particle offset or root budget")
    if args.earliest_turn < 0 or args.latest_turn < args.earliest_turn:
        raise ValueError("invalid N6R turn window")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite N6R shard: {args.output}")


@torch.no_grad()
def capture_policy_feature(
    runtime: BeliefRolloutRuntime,
    root: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture and verify the feature state used by the frozen live actor.

    ``capture_match`` snapshots the live agent's belief immediately after it
    chooses the root action. Updating a deep copy with the same observation is
    idempotent, so this recreates the exact actor-visible feature tensor while
    leaving the live policy and source match untouched.
    """

    from soldier import features  # type: ignore
    from soldier.net import attack_safe_logits, feat_to_tensors  # type: ignore

    if getattr(runtime.own_model, "mem", 0):
        raise ValueError("N6R feature capture does not support recurrent policy memory")
    belief = copy.deepcopy(root["_beliefs"][0])
    feature = features.extract(
        root["observation"],
        belief=belief,
        slot_link=bool(getattr(runtime.own_model, "slot_link", False)),
    )
    tensors = feat_to_tensors(feature, "cpu")
    logits, _ = runtime.own_model(tensors)
    options = (root["observation"].get("select") or {}).get("option") or ()
    if any(option.get("type") == 13 for option in options):
        logits = attack_safe_logits(logits, options)
    reproduced = int(torch.argmax(logits))
    baseline = int(root["baseline_action"][0])
    if reproduced != baseline:
        raise RuntimeError(
            f"N6R policy-feature snapshot failed baseline parity: {reproduced} != {baseline}"
        )
    return serialize_policy_feature(feature)


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    own = args.own_candidate.resolve()
    if sha256(own / "deck.csv") != DECK_SHA256:
        raise ValueError("N6R frozen Team Rocket deck changed")
    if sha256(own / "model.pt") != MODEL_SHA256:
        raise ValueError("N6R frozen initial model changed")
    runtime = BeliefRolloutRuntime(
        args.runtime,
        args.own_candidate,
        args.opponent,
        args.candidates,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    accepted_games = attempts = 0
    started = time()
    with output.open("x", encoding="utf-8") as sink:
        while accepted_games < args.games:
            attempts += 1
            capture = runtime.capture_match(
                seat_mode=args.seat,
                accepted_game_index=accepted_games + 1,
                roots_per_game=args.roots_per_game,
                earliest_turn=args.earliest_turn,
                latest_turn=args.latest_turn,
                max_decisions=args.max_live_decisions,
                common_random_coins=True,
                capture_truth=False,
            )
            if capture.get("seat_skipped"):
                counters["seat_skips"] += 1
                continue
            accepted_games += 1
            baseline_utility = utility(int(capture["result"]), int(capture["own_seat"]))
            counters["games"] += 1
            counters["baseline_wins"] += int(baseline_utility > 0)
            counters["baseline_losses"] += int(baseline_utility < 0)
            counters["games_without_roots"] += int(not capture["roots"])
            for root_index, root in enumerate(capture["roots"]):
                policy_feature = capture_policy_feature(runtime, root)
                counters["policy_feature_baseline_reproductions"] += 1
                evaluation = runtime.evaluate_belief_root(
                    root,
                    particles=args.particles,
                    particle_offset=args.particle_offset,
                    coin_rollouts=args.coin_rollouts,
                    max_rollout_decisions=args.max_rollout_decisions,
                    budget_seconds=args.root_budget_seconds,
                    seed=args.seed + 100_003 * accepted_games + 3_001 * root_index,
                )
                counters["roots"] += 1
                counters["fully_complete_roots"] += int(evaluation["fully_complete"])
                counters["complete_pairs"] += int(evaluation["complete_pairs"])
                counters["attempted_pairs"] += int(evaluation["attempted_pairs"])
                counters["rollout_errors"] += len(evaluation["errors"])
                row = {
                    "schema": "rocket-n6r-belief-rollout-root/v2",
                    "game_id": accepted_games,
                    "capture_attempt": attempts,
                    "root_index": root_index,
                    "opponent": args.opponent.resolve().name,
                    "own_seat": capture["own_seat"],
                    "first_player": capture["first_player"],
                    "started_first": capture["started_first"],
                    "baseline_game_result": capture["result"],
                    "baseline_game_utility": baseline_utility,
                    "baseline_game_decisions": capture["decisions"],
                    "step": root["step"],
                    "turn": root["turn"],
                    "turn_action_count": root["turn_action_count"],
                    "baseline_action": root["baseline_action"],
                    "candidate_indices": root["candidate_indices"],
                    "candidate_options": root["candidate_options"],
                    "observation_fingerprint": root["observation_fingerprint"],
                    "observation": root["observation"],
                    "policy_feature": policy_feature,
                    "policy_feature_baseline_reproduced": True,
                    "policy_feature_actor_information": "normal observation and persistent public/own-visible belief only",
                    **evaluation,
                }
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                sink.flush()
            print(
                json.dumps(
                    {
                        "game": accepted_games,
                        "of": args.games,
                        "total_roots": counters["roots"],
                        "complete_roots": counters["fully_complete_roots"],
                        "feature_reproductions": counters[
                            "policy_feature_baseline_reproductions"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    manifest = {
        "schema": "rocket-n6r-belief-rollout-manifest/v2",
        "protocol": "N6R_EXACT_FEATURE_BELIEF_ROLLOUT_PROTOCOL.md",
        "method": "paired sampled-public-belief terminal rollout teacher with exact actor-feature snapshots",
        "genuine_official_engine_source_matches": True,
        "genuine_official_engine_descendant_transitions": True,
        "visualizer_or_realized_hidden_truth_read": False,
        "hidden_completion_distribution_official": False,
        "raw_hidden_allocations_serialized": False,
        "policy_features_from_actor_visible_information_only": True,
        "policy_feature_baseline_reproduction_required": True,
        "actor_information": "normal agent observation plus persistent public/own-visible belief only",
        "public_information_set_expected_value": False,
        "equilibrium_or_safety_certificate": False,
        "common_random_coin_sequences": True,
        "leaf_value_substitution": False,
        "deployment_authority": False,
        "own_candidate": str(own),
        "opponent": str(args.opponent.resolve()),
        "own_model_sha256": sha256(own / "model.pt"),
        "opponent_model_sha256": sha256(args.opponent.resolve() / "model.pt"),
        "own_deck_sha256": sha256(own / "deck.csv"),
        "opponent_deck_sha256": sha256(args.opponent.resolve() / "deck.csv"),
        "output": str(output),
        "output_sha256": sha256(output),
        "attempted_matches": attempts,
        "counts": dict(counters),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "elapsed_seconds": time() - started,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"complete": True, "counts": manifest["counts"], "manifest": str(manifest_path)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
