#!/usr/bin/env python3
"""Collect randomized public-action outcomes from complete official matches.

Team Rocket is placed in seat 1 and accepted only when the official observation
declares seat 0 the first player. At the first eligible Team Rocket turn-2 main
root, one of up to four semantically distinct blueprint actions is assigned
uniformly. No hidden truth or search branch is read or used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys
from time import perf_counter, time
from typing import Any, Mapping, Sequence

import torch

from collect_live_truth_causal_action_values import (
    MAIN_SELECT,
    OfficialCausalRuntime,
    _valid_choice,
    fingerprint,
    option_summary,
    sha256,
    utility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_neural_resolver_safe_tie_v3"
        ),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_value_calibrated_v1"
        ),
    )
    parser.add_argument(
        "--opponent",
        type=Path,
        default=(PROJECT_ROOT / "work" / "arena_bundles" / "AshMarnieGeneralV110_extracted"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test", "smoke"), required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--protocol",
        default="RANDOMIZED_FULL_MATCH_CAUSAL_RECOVERY_PROTOCOL.md",
    )
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--eligible-turn", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--maximum-attempts", type=int, default=0)
    parser.add_argument(
        "--include-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.games <= 0 or args.candidates < 2 or args.max_decisions <= 0:
        raise ValueError("games, candidates, and max-decisions must be positive")
    if args.eligible_turn < 0:
        raise ValueError("eligible-turn must be non-negative")
    if args.maximum_attempts <= 0:
        args.maximum_attempts = max(args.games * 3, args.games + 50)
    if args.maximum_attempts < args.games:
        raise ValueError("maximum-attempts cannot be smaller than games")


def public_training_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Retain only fields available to an ordinary acting agent."""
    result = {
        "current": copy.deepcopy(observation.get("current")),
        "select": copy.deepcopy(observation.get("select")),
    }
    current = result.get("current") or {}
    players = current.get("players") or ()
    me = int(current.get("yourIndex") or 0)
    if len(players) != 2 or me not in (0, 1):
        raise ValueError("public observation does not contain two valid players")
    opponent_hand = (players[1 - me] or {}).get("hand")
    if isinstance(opponent_hand, list) and any(
        isinstance(card, Mapping) and card.get("id") is not None
        for card in opponent_hand
    ):
        raise RuntimeError("redaction audit found a visible opponent hand card")
    return result


def semantic_signature(teacher: Any, observation: Mapping[str, Any], index: int) -> Any:
    option = (observation.get("select") or {}).get("option")[index]
    return teacher._semantic_signature(observation, option)


def capture_randomized_match(
    runtime: OfficialCausalRuntime,
    *,
    accepted_game_index: int,
    attempt: int,
    seed: int,
    eligible_turn: int,
    max_decisions: int,
    include_observation: bool,
) -> dict[str, Any]:
    own_seat = 1
    observation, start = runtime.battle_start(runtime.opponent_deck, runtime.own_deck)
    if observation is None:
        raise RuntimeError(
            f"battle start failed: player={start.errorPlayer} type={start.errorType}"
        )
    own_agent, opponent_agent = runtime.new_agents()
    agents = (opponent_agent, own_agent)
    randomized_root: dict[str, Any] | None = None
    first_player: int | None = None
    started = perf_counter()
    try:
        for step in range(max_decisions):
            current = observation.get("current") or {}
            observed_first = current.get("firstPlayer")
            if first_player is None and observed_first in (0, 1):
                first_player = int(observed_first)
                if first_player == own_seat:
                    return {"seat_skipped": True, "decisions": step}
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                if first_player is None:
                    raise RuntimeError("match terminated before first player was declared")
                return {
                    "seat_skipped": False,
                    "eligible": randomized_root is not None,
                    "result": result,
                    "utility": utility(result, own_seat),
                    "own_seat": own_seat,
                    "first_player": first_player,
                    "started_first": first_player == own_seat,
                    "decisions": step,
                    "duration_seconds": perf_counter() - started,
                    "root": randomized_root,
                }
            actor = int(current.get("yourIndex") or 0)
            baseline = agents[actor](observation)
            if not _valid_choice(observation, baseline):
                raise RuntimeError(f"blueprint returned illegal action at step {step}: {baseline}")
            chosen = list(baseline)
            select = observation.get("select") or {}
            options = select.get("option") or ()
            turn = int(current.get("turn") or 0)
            eligible = bool(
                randomized_root is None
                and actor == own_seat
                and turn == eligible_turn
                and select.get("type") == MAIN_SELECT
                and select.get("minCount") == 1
                and select.get("maxCount") == 1
                and len(options) >= 2
                and len(baseline) == 1
            )
            if eligible:
                teacher = runtime.teacher_type(
                    own_agent,
                    own_model=runtime.own_model,
                    opponent_model=runtime.opponent_model,
                    own_deck=runtime.own_deck,
                    deck_library=runtime.deck_library,
                    candidates=runtime.candidates,
                )
                shortlist = tuple(
                    int(index)
                    for index in teacher._candidate_indices(observation, int(baseline[0]))
                )
                if len(shortlist) >= 2:
                    assignment_rng = random.Random(
                        int(seed) + 1_000_003 * int(attempt) + 97 * int(accepted_game_index)
                    )
                    assigned_position = assignment_rng.randrange(len(shortlist))
                    assigned_index = shortlist[assigned_position]
                    chosen = [assigned_index]
                    if not _valid_choice(observation, chosen):
                        raise RuntimeError("randomized shortlist action is not legal")
                    public_observation = public_training_observation(observation)
                    randomized_root = {
                        "step": step,
                        "turn": turn,
                        "turn_action_count": current.get("turnActionCount"),
                        "baseline_action_index": int(baseline[0]),
                        "candidate_indices": list(shortlist),
                        "candidate_options": [
                            option_summary(options[index], observation) for index in shortlist
                        ],
                        "candidate_signatures": [
                            semantic_signature(teacher, observation, index) for index in shortlist
                        ],
                        "assigned_position": assigned_position,
                        "assigned_action_index": assigned_index,
                        "propensity": 1.0 / len(shortlist),
                        "changed_from_blueprint": assigned_position != 0,
                        "observation_fingerprint": fingerprint(public_observation),
                    }
                    if include_observation:
                        randomized_root["observation"] = public_observation
            observation = runtime.battle_select(chosen)
        raise RuntimeError(f"official match exceeded {max_decisions} decisions")
    finally:
        runtime.battle_finish()


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    runtime = OfficialCausalRuntime(
        args.runtime,
        args.candidate,
        args.opponent,
        args.candidates,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    assigned_positions: Counter[str] = Counter()
    accepted = 0
    attempts = 0
    started = time()
    with output.open("w", encoding="utf-8") as sink:
        while accepted < args.games and attempts < args.maximum_attempts:
            attempts += 1
            capture = capture_randomized_match(
                runtime,
                accepted_game_index=accepted + 1,
                attempt=attempts,
                seed=args.seed,
                eligible_turn=args.eligible_turn,
                max_decisions=args.max_decisions,
                include_observation=args.include_observation,
            )
            if capture.get("seat_skipped"):
                counters["seat_skips"] += 1
                continue
            counters["official_complete_matches"] += 1
            if not capture.get("eligible"):
                counters["matches_without_eligible_root"] += 1
                continue
            accepted += 1
            root = capture["root"]
            position = int(root["assigned_position"])
            assigned_positions[str(position)] += 1
            counters["accepted_games"] += 1
            counters["wins"] += int(capture["utility"] > 0)
            counters["losses"] += int(capture["utility"] < 0)
            counters["draws"] += int(capture["utility"] == 0)
            counters["changed_assignments"] += int(root["changed_from_blueprint"])
            row = {
                "schema": "rocket-randomized-full-match-action/v1",
                "split": args.split,
                "game_id": accepted,
                "capture_attempt": attempts,
                "seed": args.seed,
                "opponent": args.opponent.resolve().name,
                "own_seat": capture["own_seat"],
                "first_player": capture["first_player"],
                "started_first": capture["started_first"],
                "terminal_result": capture["result"],
                "terminal_utility": capture["utility"],
                "match_decisions": capture["decisions"],
                "match_duration_seconds": capture["duration_seconds"],
                **root,
            }
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")
            sink.flush()
            if accepted == 1 or accepted % 10 == 0 or accepted == args.games:
                print(
                    json.dumps(
                        {
                            "accepted": accepted,
                            "of": args.games,
                            "attempts": attempts,
                            "wins": counters["wins"],
                            "assigned_positions": dict(assigned_positions),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    if accepted != args.games:
        raise RuntimeError(
            f"collected only {accepted} eligible games in {attempts} attempts"
        )
    manifest = {
        "schema": "rocket-randomized-full-match-manifest/v1",
        "protocol": args.protocol,
        "split": args.split,
        "genuine_official_engine_complete_matches": True,
        "randomized_public_action": True,
        "uniform_assignment": True,
        "visualizer_truth_used": False,
        "hidden_truth_serialized": False,
        "search_branch_used": False,
        "equilibrium_or_safety_certificate": False,
        "own_candidate": str(args.candidate.resolve()),
        "opponent": str(args.opponent.resolve()),
        "own_model_sha256": sha256(args.candidate.resolve() / "model.pt"),
        "own_deck_sha256": sha256(args.candidate.resolve() / "deck.csv"),
        "opponent_model_sha256": sha256(args.opponent.resolve() / "model.pt"),
        "opponent_deck_sha256": sha256(args.opponent.resolve() / "deck.csv"),
        "output": str(output),
        "output_sha256": sha256(output),
        "attempted_matches": attempts,
        "counts": dict(counters),
        "assigned_positions": dict(assigned_positions),
        "parameters": {
            key: str(value.resolve()) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "randomized action outcomes from complete local official-engine matches; "
            "not an information-set equilibrium or arena-strength claim"
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"complete": True, "manifest": str(manifest_path), "counts": dict(counters)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
