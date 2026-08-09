#!/usr/bin/env python3
"""Collect an interleaved official-engine control/candidate match panel.

Both Wobbuffet-free Rocket arms face the same fixed opponent and seat within a
randomized schedule.  Games use the bundled official engine and packaged
``main.agent`` entry points in isolated workers with strict legality checks.
The default requires a deck-matched policy comparison; an explicit structural
mode permits two different Wobbuffet-free decks under otherwise frozen policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter, time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_rank1_second_seat_ppo import (  # noqa: E402
    OpponentWorker,
    legal,
    read_deck,
    sha256,
)


DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs"
    r"\ptcg_agent_lab\tools\evaluate.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--games-per-arm", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", choices=("first", "second"), required=True)
    parser.add_argument(
        "--structural-deck-comparison",
        action="store_true",
        help="Allow two different exact Wobbuffet-free deck sequences.",
    )
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix()
        if (
            "__pycache__" in item.parts
            or item.suffix in {".pyc", ".pyo"}
            or item.name == ".src_mtime"
        ):
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def bundle_fingerprint(path: Path) -> dict[str, Any]:
    model_path = path / "model.pt"
    return {
        "path": str(path),
        "tree_sha256": tree_sha256(path),
        "deck_sha256": sha256(path / "deck.csv"),
        "model_sha256": sha256(model_path) if model_path.is_file() else None,
        "main_sha256": sha256(path / "main.py"),
    }


def play_game(
    engine: tuple[Any, Any, Any],
    own: OpponentWorker,
    opponent: OpponentWorker,
    own_seat: int,
    max_decisions: int,
) -> dict[str, Any]:
    battle_start, battle_select, battle_finish = engine
    own.reset()
    opponent.reset()
    decks = (own.deck, opponent.deck) if own_seat == 0 else (opponent.deck, own.deck)
    observation, start = battle_start(*decks)
    if observation is None or int(start.errorPlayer) >= 0:
        raise RuntimeError(
            f"official engine start failed: {start.errorPlayer}/{start.errorType}"
        )
    first_player: int | None = None
    own_decisions = 0
    opponent_decisions = 0
    mechanism = Counter()
    mechanism_flags: set[str] = set()

    def energy_units(card: dict[str, Any]) -> tuple[int, int]:
        units = card.get("energies") or []
        total = len(units)
        lightning = sum(
            value == 4
            or (isinstance(value, dict) and value.get("type") == 4)
            for value in units
        )
        if lightning == 0:
            lightning = sum(
                isinstance(value, dict) and value.get("id") == 4
                for value in (card.get("energyCards") or [])
            )
        return total, lightning

    def option_card_id(
        option: dict[str, Any],
        selection: dict[str, Any],
        current: dict[str, Any],
        actor: int,
    ) -> int:
        if option.get("cardId"):
            return int(option["cardId"])
        players = current.get("players") or []
        mine = players[actor] if len(players) == 2 else {}
        if option.get("type") == 7:
            index = option.get("index")
            hand = (mine or {}).get("hand") or []
            if isinstance(index, int) and 0 <= index < len(hand):
                card = hand[index]
                return int(card.get("id") or 0) if isinstance(card, dict) else 0
        area = option.get("area")
        index = option.get("index")
        if area == 1:
            deck = selection.get("deck") or []
            if isinstance(index, int) and 0 <= index < len(deck):
                card = deck[index]
                return int(card.get("id") or 0) if isinstance(card, dict) else 0
        owner = option.get("playerIndex", actor)
        if owner in (0, 1) and len(players) == 2 and isinstance(index, int):
            zones = {2: "hand", 3: "discard", 4: "active", 5: "bench"}
            zone = zones.get(area)
            cards = (players[owner] or {}).get(zone) or [] if zone else []
            if 0 <= index < len(cards):
                card = cards[index]
                return int(card.get("id") or 0) if isinstance(card, dict) else 0
        return 0

    def option_target_card_id(
        option: dict[str, Any], current: dict[str, Any], actor: int
    ) -> int:
        players = current.get("players") or []
        if len(players) != 2:
            return 0
        area = option.get("inPlayArea", option.get("area"))
        index = option.get("inPlayIndex", option.get("index"))
        owner = option.get("playerIndex", actor)
        if owner not in (0, 1) or not isinstance(index, int):
            return 0
        zone = "active" if area == 4 else "bench" if area == 5 else None
        cards = (players[owner] or {}).get(zone) or [] if zone else []
        if 0 <= index < len(cards):
            card = cards[index]
            return int(card.get("id") or 0) if isinstance(card, dict) else 0
        return 0

    started = perf_counter()
    try:
        for decision in range(max_decisions):
            current = observation.get("current") or {}
            players = current.get("players") or []
            mine = players[own_seat] if len(players) == 2 else {}
            active = (mine or {}).get("active") or []
            bench = (mine or {}).get("bench") or []
            board = [
                card
                for card in (*active, *bench)
                if isinstance(card, dict)
            ]
            spidops_count = sum(card.get("id") == 401 for card in board)
            ready_spidops_count = sum(
                card.get("id") == 401
                and len(card.get("energies") or []) >= 2
                and any(
                    energy == 1
                    or (isinstance(energy, dict) and energy.get("type") == 1)
                    for energy in (card.get("energies") or [])
                )
                for card in board
            )
            mechanism["max_spidops_in_play"] = max(
                mechanism["max_spidops_in_play"], spidops_count
            )
            mechanism["max_ready_spidops"] = max(
                mechanism["max_ready_spidops"], ready_spidops_count
            )
            for name, observed in (
                ("two_spidops_seen_games", spidops_count >= 2),
                ("three_spidops_seen_games", spidops_count >= 3),
                ("two_ready_spidops_seen_games", ready_spidops_count >= 2),
            ):
                if observed and name not in mechanism_flags:
                    mechanism[name] += 1
                    mechanism_flags.add(name)
            stadium = current.get("stadium") or {}
            public_mechanisms = {
                "dwebble_seen_games": any(card.get("id") == 344 for card in board),
                "crustle_seen_games": any(card.get("id") == 345 for card in board),
                "crustle_active_games": bool(
                    active
                    and isinstance(active[0], dict)
                    and active[0].get("id") == 345
                ),
                "neutralization_zone_seen_games": bool(
                    isinstance(stadium, dict) and stadium.get("id") == 1247
                ),
                "miraidon_seen_games": any(card.get("id") == 957 for card in board),
                "miraidon_active_games": bool(
                    active
                    and isinstance(active[0], dict)
                    and active[0].get("id") == 957
                ),
                "miraidon_ready_games": any(
                    card.get("id") == 957
                    and energy_units(card)[0] >= 3
                    and energy_units(card)[1] >= 2
                    for card in board
                ),
                "koko_seen_games": any(card.get("id") == 329 for card in board),
                "koko_active_games": bool(
                    active
                    and isinstance(active[0], dict)
                    and active[0].get("id") == 329
                ),
                "koko_ready_games": any(
                    card.get("id") == 329
                    and energy_units(card)[0] >= 2
                    and energy_units(card)[1] >= 1
                    for card in board
                ),
                "koko_belt_games": any(
                    card.get("id") == 329
                    and any(
                        isinstance(tool, dict) and tool.get("id") == 1158
                        for tool in (card.get("tools") or [])
                    )
                    for card in board
                ),
            }
            for name, observed in public_mechanisms.items():
                if observed and name not in mechanism_flags:
                    mechanism[name] += 1
                    mechanism_flags.add(name)
            observed_first = current.get("firstPlayer")
            if observed_first in (0, 1):
                first_player = int(observed_first)
                if first_player != 0:
                    raise RuntimeError("official engine changed first-player convention")
            raw_result = current.get("result")
            result = int(raw_result) if raw_result is not None else -1
            if result != -1:
                if result not in (0, 1, 2):
                    raise RuntimeError(f"invalid terminal result {result}")
                return {
                    "result": result,
                    "win": int(result == own_seat),
                    "draw": int(result == 2),
                    "own_seat": own_seat,
                    "started_first": own_seat == 0,
                    "first_player": first_player,
                    "decisions": decision,
                    "own_decisions": own_decisions,
                    "opponent_decisions": opponent_decisions,
                    "duration_seconds": perf_counter() - started,
                    "illegal_actions": 0,
                    "exception_fallbacks": 0,
                    "mechanism": dict(mechanism),
                }
            selection = observation.get("select")
            if not isinstance(selection, dict):
                raise RuntimeError("official engine emitted no live selection")
            actor = int(current.get("yourIndex") if current.get("yourIndex") is not None else -1)
            if actor == own_seat:
                action = own.act(dict(observation))
                own_decisions += 1
                options = selection.get("option") or []
                chosen = [
                    options[index]
                    for index in action
                    if isinstance(index, int) and 0 <= index < len(options)
                ]
                theirs = players[1 - own_seat] if len(players) == 2 else {}
                opponent_active = (theirs or {}).get("active") or []
                chosen_card_ids = [
                    option_card_id(option, selection, current, actor)
                    for option in chosen
                ]
                chosen_target_ids = [
                    option_target_card_id(option, current, actor)
                    for option in chosen
                ]
                mechanism["crispin_plays"] += sum(
                    option.get("type") == 7 and card_id == 1198
                    for option, card_id in zip(chosen, chosen_card_ids)
                )
                mechanism["ultra_ball_miraidon_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1121
                    and card_id == 957
                    for card_id in chosen_card_ids
                )
                mechanism["crispin_lightning_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1198
                    and card_id == 4
                    for card_id in chosen_card_ids
                )
                mechanism["crispin_attach_to_miraidon"] += sum(
                    (selection.get("effect") or {}).get("id") == 1198
                    and target_id == 957
                    for target_id in chosen_target_ids
                )
                mechanism["crispin_attach_to_koko"] += sum(
                    (selection.get("effect") or {}).get("id") == 1198
                    and target_id == 329
                    for target_id in chosen_target_ids
                )
                mechanism["manual_lightning_to_miraidon"] += sum(
                    option.get("type") == 8
                    and card_id == 4
                    and target_id == 957
                    for option, card_id, target_id in zip(
                        chosen, chosen_card_ids, chosen_target_ids
                    )
                )
                mechanism["manual_lightning_to_koko"] += sum(
                    option.get("type") == 8
                    and card_id == 4
                    and target_id == 329
                    for option, card_id, target_id in zip(
                        chosen, chosen_card_ids, chosen_target_ids
                    )
                )
                mechanism["manual_energy_to_koko"] += sum(
                    option.get("type") == 8
                    and card_id in (1, 4, 5, 15)
                    and target_id == 329
                    for option, card_id, target_id in zip(
                        chosen, chosen_card_ids, chosen_target_ids
                    )
                )
                mechanism["ultra_ball_koko_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1121
                    and card_id == 329
                    for card_id in chosen_card_ids
                )
                mechanism["poke_pad_koko_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1152
                    and card_id == 329
                    for card_id in chosen_card_ids
                )
                mechanism["petrel_plays"] += sum(
                    option.get("type") == 7 and card_id == 1219
                    for option, card_id in zip(chosen, chosen_card_ids)
                )
                mechanism["transceiver_crispin_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1134
                    and card_id == 1198
                    for card_id in chosen_card_ids
                )
                mechanism["transceiver_petrel_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1134
                    and card_id == 1219
                    for card_id in chosen_card_ids
                )
                mechanism["petrel_maxbelt_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1219
                    and card_id == 1158
                    for card_id in chosen_card_ids
                )
                mechanism["maxbelt_to_koko"] += sum(
                    option.get("type") == 8
                    and card_id == 1158
                    and target_id == 329
                    for option, card_id, target_id in zip(
                        chosen, chosen_card_ids, chosen_target_ids
                    )
                )
                mechanism["miraidon_promotions"] += sum(
                    selection.get("context") in (3, 4) and target_id == 957
                    for target_id in chosen_target_ids
                )
                mechanism["miraidon_hadron_attacks"] += sum(
                    option.get("type") == 13 and option.get("attackId") == 1377
                    for option in chosen
                )
                mechanism["hadron_attacks_vs_starmie"] += sum(
                    option.get("type") == 13
                    and option.get("attackId") == 1377
                    and bool(
                        opponent_active
                        and isinstance(opponent_active[0], dict)
                        and opponent_active[0].get("id") in (1030, 1031)
                    )
                    for option in chosen
                )
                linked_lightning = sum(
                    option.get("type") == 13 and option.get("attackId") == 458
                    for option in chosen
                )
                mechanism["linked_lightning_attacks"] += linked_lightning
                if linked_lightning:
                    turn = int(current.get("turn") or 0)
                    if "first_linked_lightning_recorded" not in mechanism_flags:
                        mechanism["first_linked_lightning_turn"] = turn
                        mechanism_flags.add("first_linked_lightning_recorded")
                    if "linked_lightning_game" not in mechanism_flags:
                        mechanism["linked_lightning_games"] += 1
                        mechanism_flags.add("linked_lightning_game")
                    target = (
                        opponent_active[0]
                        if opponent_active and isinstance(opponent_active[0], dict)
                        else {}
                    )
                    target_id = int(target.get("id") or 0)
                    active_koko = bool(
                        active
                        and isinstance(active[0], dict)
                        and active[0].get("id") == 329
                    )
                    has_belt = bool(
                        active_koko
                        and any(
                            isinstance(tool, dict) and tool.get("id") == 1158
                            for tool in (active[0].get("tools") or [])
                        )
                    )
                    estimated_damage = (
                        60 + 20 * len(bench) + (50 if has_belt else 0)
                    )
                    if target_id in (1030, 1031):
                        estimated_damage *= 2
                    mechanism["linked_lightning_estimated_damage"] += (
                        linked_lightning * estimated_damage
                    )
                    if target_id == 1030:
                        mechanism["linked_lightning_attacks_vs_staryu"] += linked_lightning
                    if target_id == 1031:
                        mechanism["linked_lightning_attacks_vs_mega_starmie"] += linked_lightning
                        if has_belt:
                            mechanism["linked_belt_attacks_vs_mega_starmie"] += linked_lightning
                        if estimated_damage >= int(target.get("hp") or 0):
                            mechanism["linked_lethal_attacks_vs_mega_starmie"] += linked_lightning
                        if (
                            int(target.get("hp") or 0) >= 330
                            and estimated_damage >= int(target.get("hp") or 0)
                        ):
                            mechanism["linked_full_hp_lethals_vs_mega_starmie"] += linked_lightning
                    if len(bench) >= 3:
                        mechanism["linked_attacks_with_three_plus_bench"] += linked_lightning
                    if has_belt and "linked_belt_game" not in mechanism_flags:
                        mechanism["linked_belt_games"] += 1
                        mechanism_flags.add("linked_belt_game")
                mechanism["dwebble_ascension_attacks"] += sum(
                    option.get("type") == 13 and option.get("attackId") == 478
                    for option in chosen
                )
                mechanism["crustle_attacks"] += sum(
                    option.get("type") == 13 and option.get("attackId") == 479
                    for option in chosen
                )
                mechanism["proton_plays"] += sum(
                    option.get("type") == 7 and card_id == 1220
                    for option, card_id in zip(chosen, chosen_card_ids)
                )
                mechanism["transceiver_proton_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1134
                    and card_id == 1220
                    for card_id in chosen_card_ids
                )
                mechanism["spidops_evolutions"] += sum(
                    option.get("type") == 9 and card_id == 401
                    for option, card_id in zip(chosen, chosen_card_ids)
                )
                mechanism["charging_up_uses"] += sum(
                    option.get("type") == 10 and target_id == 401
                    for option, target_id in zip(chosen, chosen_target_ids)
                )
                mechanism["manual_energy_to_spidops"] += sum(
                    option.get("type") == 8
                    and card_id in (1, 5, 15)
                    and target_id == 401
                    for option, card_id, target_id in zip(
                        chosen, chosen_card_ids, chosen_target_ids
                    )
                )
                mechanism["ultra_ball_grass_discards"] += sum(
                    (selection.get("effect") or {}).get("id") == 1121
                    and selection.get("context") == 8
                    and card_id == 1
                    for card_id in chosen_card_ids
                )
                mechanism["ultra_ball_spidops_searches"] += sum(
                    (selection.get("effect") or {}).get("id") == 1121
                    and selection.get("context") == 7
                    and card_id == 401
                    for card_id in chosen_card_ids
                )
                rocket_rushes = sum(
                    option.get("type") == 13 and option.get("attackId") == 560
                    for option in chosen
                )
                mechanism["rocket_rush_attacks"] += rocket_rushes
                if rocket_rushes:
                    first_key = "first_rocket_rush_recorded"
                    if first_key not in mechanism_flags:
                        mechanism["first_rocket_rush_turn"] = int(
                            current.get("turn") or 0
                        )
                        mechanism_flags.add(first_key)
                    opening_attack_turn = 3 if own_seat == 0 else 4
                    if int(current.get("turn") or 0) <= opening_attack_turn:
                        mechanism["opening_window_rocket_rush_attacks"] += rocket_rushes
                        if "opening_rocket_rush_game" not in mechanism_flags:
                            mechanism["opening_rocket_rush_games"] += 1
                            mechanism_flags.add("opening_rocket_rush_game")
                    if bool(
                        opponent_active
                        and isinstance(opponent_active[0], dict)
                        and opponent_active[0].get("id") == 1030
                    ):
                        mechanism["rocket_rush_attacks_vs_staryu"] += rocket_rushes
                target_is_starmie = bool(
                    opponent_active
                    and isinstance(opponent_active[0], dict)
                    and opponent_active[0].get("id") == 1031
                )
                if target_is_starmie and any(
                    option.get("type") == 13 and option.get("attackId") == 608
                    for option in chosen
                ):
                    mechanism["mewtwo_erasure_attacks_vs_mega_starmie"] += 1
                if (
                    target_is_starmie
                    and active
                    and isinstance(active[0], dict)
                    and active[0].get("id") == 431
                    and 280 < int(opponent_active[0].get("hp") or 0) <= 330
                    and sum(
                        len(card.get("energyCards") or [])
                        for card in ((mine or {}).get("bench") or [])
                        if isinstance(card, dict)
                    ) >= 2
                    and any(
                        option.get("type") == 13 and option.get("attackId") == 608
                        for option in options
                    )
                ):
                    hand = (mine or {}).get("hand") or []
                    for option in chosen:
                        hand_index = option.get("index")
                        if (
                            option.get("type") == 8
                            and option.get("inPlayArea") == 4
                            and option.get("inPlayIndex") == 0
                            and isinstance(hand_index, int)
                            and 0 <= hand_index < len(hand)
                            and isinstance(hand[hand_index], dict)
                            and hand[hand_index].get("id") == 1158
                        ):
                            mechanism["maxbelt_starmie_lethal_attachments"] += 1
                if (
                    selection.get("context") == 26
                    and (selection.get("effect") or {}).get("id") == 431
                    and target_is_starmie
                    and active
                    and isinstance(active[0], dict)
                    and any(
                        isinstance(tool, dict) and tool.get("id") == 1158
                        for tool in (active[0].get("tools") or [])
                    )
                ):
                    mechanism["maxbelt_starmie_discard_prompts"] += 1
                    mechanism["maxbelt_starmie_energy_cards_discarded"] += len(action)
            elif actor == 1 - own_seat:
                action = opponent.act(dict(observation))
                opponent_decisions += 1
            else:
                raise RuntimeError(f"official engine emitted invalid actor {actor}")
            if not legal(action, selection):
                raise RuntimeError(f"packaged worker returned illegal action: {action}")
            observation = battle_select(list(action))
        raise RuntimeError(f"game exceeded {max_decisions} decisions")
    finally:
        battle_finish()


def main() -> int:
    args = parse_args()
    if args.games_per_arm <= 0 or args.max_decisions <= 0:
        raise ValueError("game counts and decision limits must be positive")
    control = args.control.resolve()
    candidate = args.candidate.resolve()
    opponent_path = args.opponent.resolve()
    evaluator = args.evaluator.resolve()
    output = args.output.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing panel output: {output}")

    control_deck = read_deck(control)
    candidate_deck = read_deck(candidate)
    if control_deck != candidate_deck and not args.structural_deck_comparison:
        raise ValueError("control and candidate deck sequences differ")
    if args.structural_deck_comparison and control_deck == candidate_deck:
        raise ValueError("structural deck comparison requires different deck sequences")
    if any(len(deck) != 60 or deck.count(432) != 0 for deck in (control_deck, candidate_deck)):
        raise ValueError("panel requires two exact 60-card Wobbuffet-free decks")
    if tree_sha256(control) == tree_sha256(candidate):
        raise ValueError("candidate bundle is byte-identical to control")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    sys.path.insert(0, str(control))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore  # noqa: E402

    engine = (battle_start, battle_select, battle_finish)
    own_seat = 0 if args.seat == "first" else 1
    arm_paths = {"control": control, "candidate": candidate}
    schedule = [name for name in arm_paths for _ in range(args.games_per_arm)]
    random.Random(args.seed).shuffle(schedule)
    workers: dict[str, OpponentWorker] = {}
    opponent = OpponentWorker(opponent_path, evaluator, args.action_timeout_seconds)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    draws: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time()
    try:
        workers = {
            name: OpponentWorker(path, evaluator, args.action_timeout_seconds)
            for name, path in arm_paths.items()
        }
        with output.open("x", encoding="utf-8") as sink:
            for game_index, arm in enumerate(schedule, start=1):
                row = play_game(
                    engine,
                    workers[arm],
                    opponent,
                    own_seat,
                    args.max_decisions,
                )
                row.update(
                    {
                        "schema": "target80-deck-matched-bc-panel-game/v1",
                        "game_index": game_index,
                        "schedule_seed": args.seed,
                        "arm": arm,
                        "opponent": opponent_path.name,
                    }
                )
                sink.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                sink.flush()
                counts[arm] += 1
                wins[arm] += int(row["win"])
                draws[arm] += int(row["draw"])
                for name, value in row["mechanism"].items():
                    counts[f"{arm}:{name}"] += int(value)
                if game_index == 1 or game_index % 20 == 0 or game_index == len(schedule):
                    print(
                        json.dumps(
                            {
                                "completed": game_index,
                                "target": len(schedule),
                                "seat": args.seat,
                                "counts": dict(counts),
                                "wins": dict(wins),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        opponent.close()
        for worker in workers.values():
            worker.close()

    fingerprints = {
        "control": bundle_fingerprint(control),
        "candidate": bundle_fingerprint(candidate),
        "opponent": bundle_fingerprint(opponent_path),
    }
    manifest = {
        "schema": "target80-deck-matched-bc-panel/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_engine_execution": True,
        "packaged_worker_execution": True,
        "strict_legality": True,
        "hidden_information_used_by_actor": False,
        "visualizer_hidden_truth_used": False,
        "arms": fingerprints,
        "comparison_mode": (
            "structural_deck"
            if args.structural_deck_comparison
            else "deck_matched_policy"
        ),
        "shared_deck_sha256": (
            fingerprints["control"]["deck_sha256"]
            if control_deck == candidate_deck
            else None
        ),
        "deck_sha256": {
            name: fingerprints[name]["deck_sha256"]
            for name in ("control", "candidate")
        },
        "wobbuffet_count": 0,
        "opponent": str(opponent_path),
        "seed": args.seed,
        "seat": args.seat,
        "games_per_arm": args.games_per_arm,
        "rows": len(schedule),
        "counts": dict(counts),
        "wins": dict(wins),
        "draws": dict(draws),
        "output": str(output),
        "output_sha256": sha256(output),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "evaluator_sha256": sha256(evaluator),
        "elapsed_seconds": time() - started,
        "claim_boundary": "development panel only; no arena strength or deployment claim",
    }
    sidecar.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
