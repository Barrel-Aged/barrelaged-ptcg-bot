#!/usr/bin/env python3
"""Collect randomized whole-game persistent-plan outcomes in the official engine."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from time import perf_counter, time
from typing import Any, Mapping, Sequence

import torch

from collect_live_truth_causal_action_values import (
    OfficialCausalRuntime,
    _valid_choice,
    fingerprint,
    sha256,
    utility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "rocket-persistent-plan-causal/v1"
FROZEN_DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
FROZEN_BLUEPRINT_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
PLAN_AXES = (
    "board_setup",
    "mewtwo_access",
    "spidops_engine",
    "mewtwo_charge",
    "second_attacker",
    "charging_fuel",
    "draw_engine",
    "tool_assignment",
    "attack_tempo",
    "bench_discipline",
    "fuel_preservation",
)


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
    parser.add_argument(
        "--plans",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "persistent_plan_axis_screen_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("smoke", "discovery", "development", "validation", "test"),
        required=True,
    )
    parser.add_argument("--games-per-plan", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--maximum-attempts", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_plans(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "rocket-persistent-plan-library/v1":
        raise ValueError("unexpected persistent-plan library schema")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("plan library belongs to a different protocol")
    arms = payload.get("arms") or []
    if len(arms) < 2:
        raise ValueError("plan library must contain at least two arms")
    names = [str(arm.get("name") or "") for arm in arms]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("plan names must be nonempty and unique")
    if "zero_blueprint" not in names:
        raise ValueError("plan library must contain zero_blueprint")
    if tuple(payload.get("axes") or ()) != PLAN_AXES:
        raise ValueError("plan library axes do not match persistent-plan v1")
    normalized = []
    for arm in arms:
        weights = arm.get("weights") or {}
        unknown = set(weights) - set(PLAN_AXES)
        if unknown:
            raise ValueError(f"unknown persistent-plan axes: {sorted(unknown)}")
        vector = [float(weights.get(name, 0.0)) for name in PLAN_AXES]
        if any(not (-3.0 <= value <= 3.0) for value in vector):
            raise ValueError("persistent-plan weights are bounded to [-3, 3]")
        normalized.append(
            {
                "name": str(arm["name"]),
                "weights": {name: float(vector[index]) for index, name in enumerate(PLAN_AXES)},
            }
        )
    return payload, normalized


def _own_public_metrics(
    observation: Mapping[str, Any],
    own_seat: int,
) -> dict[str, Any]:
    from persistent_plan_runtime import ATTACKERS, MEWTWO, ROCKET_POKEMON, _energy_card_ids, _ready

    current = observation.get("current") or {}
    players = current.get("players") or ()
    if len(players) != 2:
        return {
            "mewtwo_on_board": False,
            "ready_attacker_count": 0,
            "second_ready_attacker": False,
        }
    player = players[own_seat] or {}
    board = [
        pokemon
        for zone in ("active", "bench")
        for pokemon in player.get(zone) or ()
        if isinstance(pokemon, Mapping)
    ]
    rocket_count = sum(int(pokemon.get("id") or 0) in ROCKET_POKEMON for pokemon in board)
    ready_count = sum(
        int(pokemon.get("id") or 0) in ATTACKERS
        and _ready(
            int(pokemon.get("id") or 0),
            _energy_card_ids(pokemon),
            rocket_count,
        )
        for pokemon in board
    )
    return {
        "mewtwo_on_board": any(int(pokemon.get("id") or 0) == MEWTWO for pokemon in board),
        "ready_attacker_count": int(ready_count),
        "second_ready_attacker": ready_count >= 2,
    }


def capture_match(
    runtime: OfficialCausalRuntime,
    arm: Mapping[str, Any],
    *,
    accepted_game_index: int,
    attempt: int,
    max_decisions: int,
) -> dict[str, Any]:
    from persistent_plan_runtime import MEWTWO, PersistentPlanAgent

    own_seat = 1
    observation, start = runtime.battle_start(runtime.opponent_deck, runtime.own_deck)
    if observation is None:
        raise RuntimeError(
            f"battle start failed: player={start.errorPlayer} type={start.errorType}"
        )
    own_agent = PersistentPlanAgent(
        runtime.own_model,
        arm["weights"],
        plan_name=str(arm["name"]),
        deck_ids=runtime.own_deck,
        opponent_deck_priors=runtime.deck_library,
        forbid_end_when_attack=True,
    )
    opponent_agent = runtime.agent_type(
        runtime.opponent_model,
        greedy=True,
        deck_ids=runtime.opponent_deck,
        forbid_end_when_attack=True,
        opponent_deck_priors=runtime.deck_library,
    )
    agents = (opponent_agent, own_agent)
    first_player: int | None = None
    ever_mewtwo = False
    ever_second_ready = False
    max_ready_attackers = 0
    mewtwo_attack_legal_roots = 0
    mewtwo_attacks = 0
    started = perf_counter()
    try:
        for step in range(max_decisions):
            current = observation.get("current") or {}
            observed_first = current.get("firstPlayer")
            if first_player is None and observed_first in (0, 1):
                first_player = int(observed_first)
                if first_player == own_seat:
                    return {
                        "seat_skipped": True,
                        "attempt": attempt,
                        "decisions": step,
                        "plan": str(arm["name"]),
                    }
            public_metrics = _own_public_metrics(observation, own_seat)
            ever_mewtwo = ever_mewtwo or bool(public_metrics["mewtwo_on_board"])
            ever_second_ready = ever_second_ready or bool(public_metrics["second_ready_attacker"])
            max_ready_attackers = max(max_ready_attackers, int(public_metrics["ready_attacker_count"]))
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                if first_player is None:
                    raise RuntimeError("official match ended before firstPlayer was declared")
                telemetry = own_agent.telemetry_snapshot()
                return {
                    "schema": "rocket-randomized-persistent-plan-match/v1",
                    "protocol": PROTOCOL,
                    "seat_skipped": False,
                    "accepted_game_index": accepted_game_index,
                    "attempt": attempt,
                    "plan": str(arm["name"]),
                    "propensity": None,
                    "result": result,
                    "utility": utility(result, own_seat),
                    "win": int(result == own_seat),
                    "own_seat": own_seat,
                    "first_player": first_player,
                    "started_second": first_player != own_seat,
                    "decisions": step,
                    "duration_seconds": perf_counter() - started,
                    "plan_decisions": int(telemetry["decisions"]),
                    "plan_interventions": int(telemetry["interventions"]),
                    "games_with_intervention": int(telemetry["interventions"] > 0),
                    "exception_fallbacks": int(telemetry["exception_fallbacks"]),
                    "mean_abs_chosen_bias": float(telemetry["mean_abs_chosen_bias"]),
                    "ever_mewtwo_on_board": bool(ever_mewtwo),
                    "ever_second_ready_attacker": bool(ever_second_ready),
                    "max_ready_attacker_count": int(max_ready_attackers),
                    "mewtwo_attack_legal_roots": int(mewtwo_attack_legal_roots),
                    "mewtwo_attacks": int(mewtwo_attacks),
                }
            actor = int(current.get("yourIndex") or 0)
            select = observation.get("select") or {}
            options = select.get("option") or ()
            if actor == own_seat:
                legal_mewtwo_attack = any(
                    int(option.get("type") or 0) == 13
                    and int(option.get("attackId") or 0) == 608
                    for option in options
                    if isinstance(option, Mapping)
                )
                mewtwo_attack_legal_roots += int(legal_mewtwo_attack)
            action = agents[actor](observation)
            if not _valid_choice(observation, action):
                raise RuntimeError(
                    f"agent returned illegal action at step {step}: actor={actor} action={action}"
                )
            if actor == own_seat:
                mewtwo_attacks += sum(
                    int(options[index].get("type") or 0) == 13
                    and int(options[index].get("attackId") or 0) == 608
                    for index in action
                )
            observation = runtime.battle_select(action)
        raise RuntimeError(f"official match exceeded {max_decisions} decisions")
    finally:
        runtime.battle_finish()


def main() -> int:
    args = parse_args()
    if args.games_per_plan <= 0 or args.max_decisions <= 0:
        raise ValueError("games-per-plan and max-decisions must be positive")
    plan_payload, plans = load_plans(args.plans.resolve())
    target_games = args.games_per_plan * len(plans)
    if args.maximum_attempts <= 0:
        args.maximum_attempts = target_games * 3
    if args.maximum_attempts < target_games:
        raise ValueError("maximum-attempts cannot be smaller than accepted target")
    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if not args.overwrite and (output.exists() or manifest_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing persistent-plan shard: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if str(PROJECT_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    runtime = OfficialCausalRuntime(
        args.runtime.resolve(),
        args.candidate.resolve(),
        args.opponent.resolve(),
        candidates=2,
    )
    deck_hash = sha256(args.candidate.resolve() / "deck.csv")
    model_hash = sha256(args.candidate.resolve() / "model.pt")
    if deck_hash != FROZEN_DECK_SHA256:
        raise RuntimeError(f"frozen deck hash mismatch: {deck_hash}")
    if model_hash != FROZEN_BLUEPRINT_SHA256:
        raise RuntimeError(f"frozen blueprint hash mismatch: {model_hash}")

    schedule_rng = random.Random(args.seed)
    schedule: list[dict[str, Any]] = []
    for _ in range(args.games_per_plan):
        block = list(plans)
        schedule_rng.shuffle(block)
        schedule.extend(block)
    counters: Counter[str] = Counter()
    plan_counts: Counter[str] = Counter()
    plan_wins: Counter[str] = Counter()
    plan_intervention_games: Counter[str] = Counter()
    attempted = 0
    accepted = 0
    started = time()
    with output.open("w", encoding="utf-8") as sink:
        while accepted < len(schedule) and attempted < args.maximum_attempts:
            attempted += 1
            arm = schedule[accepted]
            capture = capture_match(
                runtime,
                arm,
                accepted_game_index=accepted + 1,
                attempt=attempted,
                max_decisions=args.max_decisions,
            )
            if capture.get("seat_skipped"):
                counters["seat_skips"] += 1
                continue
            accepted += 1
            capture["split"] = args.split
            capture["seed"] = args.seed
            capture["propensity"] = 1.0 / len(plans)
            capture["opponent"] = args.opponent.resolve().name
            capture["plan_library_fingerprint"] = fingerprint(plan_payload)
            sink.write(json.dumps(capture, sort_keys=True, separators=(",", ":")) + "\n")
            sink.flush()
            plan_name = str(capture["plan"])
            plan_counts[plan_name] += 1
            plan_wins[plan_name] += int(capture["win"])
            plan_intervention_games[plan_name] += int(capture["games_with_intervention"])
            counters["accepted_games"] += 1
            counters["wins"] += int(capture["win"])
            counters["losses"] += int(not capture["win"])
            counters["exception_fallbacks"] += int(capture["exception_fallbacks"])
            if accepted == 1 or accepted % max(1, len(plans)) == 0 or accepted == target_games:
                progress = {
                    "accepted": accepted,
                    "target": target_games,
                    "attempted": attempted,
                    "complete_blocks": accepted // len(plans),
                    "split": args.split,
                }
                if args.split != "test":
                    progress["latest_plan"] = plan_name
                print(json.dumps(progress, sort_keys=True), flush=True)
    if accepted != target_games:
        raise RuntimeError(f"collected only {accepted} of {target_games} accepted games")
    expected = {arm["name"]: args.games_per_plan for arm in plans}
    if dict(plan_counts) != expected:
        raise RuntimeError(f"blocked randomization imbalance: {dict(plan_counts)} != {expected}")

    manifest: dict[str, Any] = {
        "schema": "rocket-randomized-persistent-plan-shard/v1",
        "protocol": PROTOCOL,
        "protocol_document": str((PROJECT_ROOT / "PERSISTENT_PLAN_CAUSAL_SEARCH_PROTOCOL.md").resolve()),
        "protocol_document_sha256": sha256(PROJECT_ROOT / "PERSISTENT_PLAN_CAUSAL_SEARCH_PROTOCOL.md"),
        "split": args.split,
        "sealed": args.split == "test",
        "seed": args.seed,
        "runtime": str(args.runtime.resolve()),
        "candidate": str(args.candidate.resolve()),
        "opponent": str(args.opponent.resolve()),
        "plan_library": str(args.plans.resolve()),
        "plan_library_sha256": sha256(args.plans.resolve()),
        "plan_library_fingerprint": fingerprint(plan_payload),
        "deck_sha256": deck_hash,
        "blueprint_model_sha256": model_hash,
        "opponent_deck_sha256": sha256(args.opponent.resolve() / "deck.csv"),
        "opponent_model_sha256": sha256(args.opponent.resolve() / "model.pt"),
        "runtime_source_sha256": sha256(PROJECT_ROOT / "scripts" / "persistent_plan_runtime.py"),
        "output": str(output),
        "output_sha256": sha256(output),
        "accepted_games": accepted,
        "attempted_games": attempted,
        "seat_skips": counters["seat_skips"],
        "games_per_plan": args.games_per_plan,
        "plan_counts": dict(sorted(plan_counts.items())),
        "propensity": 1.0 / len(plans),
        "exception_fallbacks": counters["exception_fallbacks"],
        "elapsed_seconds": time() - started,
        "official_execution": "complete local official-engine matches using ordinary redacted observations",
        "abstraction_boundary": "persistent semantic plan features and bounded neural-logit overlay are research abstractions",
        "hidden_information_used_by_policy": False,
        "visualizer_truth_used": False,
        "search_branches_used": False,
    }
    if args.split != "test":
        manifest["wins"] = counters["wins"]
        manifest["losses"] = counters["losses"]
        manifest["plan_wins"] = dict(sorted(plan_wins.items()))
        manifest["plan_intervention_games"] = dict(sorted(plan_intervention_games.items()))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    printable = {
        "manifest": str(manifest_path),
        "accepted_games": accepted,
        "attempted_games": attempted,
        "split": args.split,
        "sealed": args.split == "test",
        "exception_fallbacks": counters["exception_fallbacks"],
    }
    if args.split != "test":
        printable["wins"] = counters["wins"]
        printable["losses"] = counters["losses"]
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
