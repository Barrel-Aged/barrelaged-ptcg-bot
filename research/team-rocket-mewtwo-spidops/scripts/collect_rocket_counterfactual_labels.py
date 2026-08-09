#!/usr/bin/env python3
"""Collect belief-sampled official-engine counterfactual action labels.

This is an offline Phase-5 teacher, not a claim of exact hidden-state search.
For selected public Marnie states it samples legal hidden completions from an
explicit deck prior, rolls the top blueprint actions through ``cg.search_*``,
and records paired values.  The actor and eventual student still receive only
their normal observation; hidden completions never enter policy features.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter, time
from typing import Any

import torch

from train_live_outcome_awr import family, read_game


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=PROJECT_ROOT / "work" / "arena_scale_v2_609")
    parser.add_argument("--recorded-bot", default="HelbrosRocketSpidopsScaleBCV2")
    parser.add_argument(
        "--runtime",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_neural_resolver_safe_tie_v3",
        help="Audited runtime supplying official-search determinization helpers.",
    )
    parser.add_argument(
        "--own-candidate",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1",
    )
    parser.add_argument(
        "--opponent",
        type=Path,
        default=PROJECT_ROOT / "work" / "arena_bundles" / "HelbrosMarnieGeneralRecoveryV4_extracted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "counterfactual_labels_v1.jsonl",
    )
    parser.add_argument("--max-games", type=int, default=48)
    parser.add_argument(
        "--game-offset",
        type=int,
        default=0,
        help="Skip this many eligible games after deterministic outcome/name ordering.",
    )
    parser.add_argument(
        "--outcome",
        choices=("all", "win", "loss"),
        default="all",
        help="Restrict source trajectories by the recorded bot's result.",
    )
    parser.add_argument("--states-per-game", type=int, default=2)
    parser.add_argument("--latest-turn", type=int, default=8)
    parser.add_argument("--particles", type=int, default=3)
    parser.add_argument(
        "--particle-offset",
        type=int,
        default=0,
        help="Start hidden-state particle IDs here; use a nonzero offset for replication.",
    )
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=80)
    parser.add_argument("--state-budget-seconds", type=float, default=10.0)
    parser.add_argument("--minimum-mean-gain", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [int(value) for value in path.read_text(encoding="utf-8").replace(",", "\n").split()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(deck)}")
    return deck


def option_summary(option: Any, observation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(option, dict):
        return {"raw_type": type(option).__name__}
    card = option.get("card") or {}
    pokemon = option.get("pokemon") or {}
    summary = {
        "type": option.get("type"),
        "index": option.get("index"),
        "area": option.get("area"),
        "in_play_area": option.get("inPlayArea"),
        "in_play_index": option.get("inPlayIndex"),
        "card_id": card.get("id") if isinstance(card, dict) else None,
        "pokemon_id": pokemon.get("id") if isinstance(pokemon, dict) else None,
        "attack_id": option.get("attackId"),
    }
    # For normal main-phase choices, the official observation represents a
    # playable/attachable hand card by its stable hand index instead of nesting
    # the card in the option.  Resolve that public pointer for auditability.
    current = observation.get("current") or {}
    players = current.get("players") or []
    your_index = current.get("yourIndex")
    index = option.get("index")
    if isinstance(your_index, int) and 0 <= your_index < len(players) and isinstance(index, int):
        hand = (players[your_index] or {}).get("hand") or []
        if 0 <= index < len(hand) and isinstance(hand[index], dict):
            summary["hand_card_id"] = hand[index].get("id")
            summary["hand_card_serial"] = hand[index].get("serial")
    return summary


def evaluate_state(
    teacher,
    observation: dict[str, Any],
    baseline: int,
    candidate_indices: tuple[int, ...],
    opponent_deck: list[int],
    particles: int,
    particle_offset: int,
    budget_seconds: float,
) -> dict[str, Any]:
    from cg.api import search_begin, search_end, search_step, to_observation_class

    current = observation.get("current") or {}
    root_player = int(current.get("yourIndex"))
    deadline = perf_counter() + budget_seconds
    values: dict[int, list[float]] = {index: [] for index in candidate_indices}
    terminal: dict[int, int] = {index: 0 for index in candidate_indices}
    errors: list[str] = []
    for particle in range(particle_offset, particle_offset + particles):
        if perf_counter() >= deadline:
            break
        predictions = teacher._hidden_predictions(observation, opponent_deck, particle=particle)
        root = None
        try:
            root = search_begin(to_observation_class(observation), **predictions)
            for index in candidate_indices:
                if perf_counter() >= deadline:
                    break
                branch = search_step(root.searchId, [index])
                value, reached_terminal = teacher._rollout(
                    branch,
                    root_player=root_player,
                    opponent_deck=opponent_deck,
                    deadline=deadline,
                    search_step=search_step,
                )
                values[index].append(float(value))
                terminal[index] += int(reached_terminal)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {str(error)[:160]}")
        finally:
            if root is not None:
                try:
                    search_end()
                except Exception:
                    pass

    complete_count = min((len(samples) for samples in values.values()), default=0)
    comparable = {
        index: samples[:complete_count]
        for index, samples in values.items()
        if complete_count > 0 and len(samples) >= complete_count
    }
    means = {
        index: sum(samples) / len(samples)
        for index, samples in comparable.items()
        if samples
    }
    best = max(means, key=lambda index: (means[index], -index)) if baseline in means else baseline
    deltas = [
        candidate - base
        for candidate, base in zip(comparable.get(best, []), comparable.get(baseline, []))
    ] if best != baseline else []
    mean_gain = sum(deltas) / len(deltas) if deltas else 0.0
    robust = bool(
        deltas
        and complete_count >= particles
        and mean_gain >= teacher.minimum_gain
        and sum(delta > 0.0 for delta in deltas) >= max(2, particles - 1)
        and min(deltas) >= -0.10
    )
    return {
        "baseline": baseline,
        "best": best,
        "robust_best": best if robust else baseline,
        "changed": robust,
        "values": {str(index): samples for index, samples in values.items()},
        "means": {str(index): value for index, value in means.items()},
        "terminal": {str(index): value for index, value in terminal.items()},
        "paired_deltas": deltas,
        "mean_gain": mean_gain,
        "complete_particles": complete_count,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    runtime = args.runtime.resolve()
    own = args.own_candidate.resolve()
    opponent = args.opponent.resolve()
    sys.path.insert(0, str(runtime))

    from soldier.net import NNAgent, load_policy
    from soldier.neural_search import NeuralRolloutAgent

    own_deck = read_deck(own / "deck.csv")
    opponent_deck = read_deck(opponent / "deck.csv")
    own_model = load_policy(own / "model.pt").eval()
    opponent_model = load_policy(opponent / "model.pt").eval()
    library_path = own / "deck_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else {}
    library["marnie"] = opponent_deck

    eligible = []
    for path in sorted(args.replays.resolve().glob("*.jsonl")):
        try:
            game, decisions = read_game(path, args.recorded_bot)
        except Exception:
            continue
        if family(game.opponent) != "marnie" or game.started_first:
            continue
        if args.outcome == "win" and not game.won:
            continue
        if args.outcome == "loss" and game.won:
            continue
        eligible.append((game, decisions))
    random.Random(args.seed).shuffle(eligible)
    # Keep outcomes represented while prioritizing losses where alternatives
    # can generate a genuine policy-improvement target.
    eligible.sort(key=lambda item: (item[0].won, item[0].path.name))
    eligible = eligible[args.game_offset : args.game_offset + args.max_games]

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    started = time()
    with args.output.resolve().open("w", encoding="utf-8") as sink:
        for game_index, (game, decisions) in enumerate(eligible, start=1):
            blueprint = NNAgent(
                own_model,
                greedy=True,
                deck_ids=own_deck,
                forbid_end_when_attack=True,
                opponent_deck_priors=library,
            )
            teacher = NeuralRolloutAgent(
                blueprint,
                own_model=own_model,
                opponent_model=opponent_model,
                own_deck=own_deck,
                deck_library=library,
                particles=args.particles,
                candidates=args.candidates,
                maximum_steps=args.rollout_steps,
                budget_seconds=args.state_budget_seconds,
                minimum_gain=args.minimum_mean_gain,
                maximum_searches=args.states_per_game,
                latest_search_turn=args.latest_turn,
            )
            searched_turns: set[int] = set()
            states = 0
            for record in decisions:
                observation = record.get("obs") or {}
                baseline_action = blueprint(observation)
                select = observation.get("select") or {}
                current = observation.get("current") or {}
                options = select.get("option") or []
                turn = int(current.get("turn") or 0)
                if (
                    states >= args.states_per_game
                    or turn in searched_turns
                    or turn > args.latest_turn
                    or current.get("yourIndex") == current.get("firstPlayer")
                    or select.get("type") != 0
                    or select.get("minCount") != 1
                    or select.get("maxCount") != 1
                    or len(options) < 2
                    or len(baseline_action) != 1
                ):
                    continue
                try:
                    candidates = teacher._candidate_indices(observation, int(baseline_action[0]))
                    if len(candidates) < 2:
                        continue
                    result = evaluate_state(
                        teacher,
                        observation,
                        int(baseline_action[0]),
                        candidates,
                        opponent_deck,
                        args.particles,
                        args.particle_offset,
                        args.state_budget_seconds,
                    )
                except Exception as error:
                    counts[f"state_error:{type(error).__name__}"] += 1
                    continue
                searched_turns.add(turn)
                states += 1
                counts["states"] += 1
                counts["changed"] += int(result["changed"])
                counts["complete"] += int(result["complete_particles"] >= args.particles)
                row = {
                    "schema": "rocket-counterfactual-label/v1",
                    "game": game.path.name,
                    "opponent": game.opponent,
                    "won": game.won,
                    "started_first": game.started_first,
                    "turn": turn,
                    "turn_action_count": current.get("turnActionCount"),
                    "recorded_action": record.get("action"),
                    "candidate_indices": list(candidates),
                    "candidate_options": {
                        str(index): option_summary(options[index], observation) for index in candidates
                    },
                    **result,
                }
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                sink.flush()
            counts["games"] += 1
            counts["winning_games"] += int(game.won)
            print(
                json.dumps(
                    {
                        "game": game_index,
                        "of": len(eligible),
                        "states": states,
                        "total_states": counts["states"],
                        "changed": counts["changed"],
                    }
                ),
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "method": "public-belief sampled official-engine rollout teacher",
        "genuine_official_engine_execution": True,
        "exact_hidden_state_used": False,
        "hidden_completion_source": "public zones plus explicit Marnie deck prior",
        "actor_information": "normal policy observation only",
        "replays": str(args.replays.resolve()),
        "own_model": str((own / "model.pt").resolve()),
        "own_model_sha256": sha256(own / "model.pt"),
        "opponent_model": str((opponent / "model.pt").resolve()),
        "opponent_model_sha256": sha256(opponent / "model.pt"),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output.resolve()),
        "counts": dict(counts),
        "parameters": {
            key: value for key, value in vars(args).items() if isinstance(value, (int, float, str, bool))
        },
        "elapsed_seconds": time() - started,
    }
    manifest_path = args.output.resolve().with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, **manifest["counts"], "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
