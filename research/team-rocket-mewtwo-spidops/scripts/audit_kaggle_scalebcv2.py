#!/usr/bin/env python3
"""Audit every public/validation replay for the frozen ScaleBCV2 Kaggle bot.

The replay is the source of truth.  This script does not contact Kaggle, alter
the submitted archive, or use hidden state to recommend live actions.  Kaggle's
visualizer frames do contain both decks and full zones after a game; those are
used only for retrospective integrity and outcome analysis.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


FOCAL_DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
ROCKET_POKEMON = {400, 401, 414, 431, 432, 463}
BASIC_POKEMON = {400, 414, 431, 432, 463}
SUPPORTERS = {1216, 1217, 1218, 1219, 1220, 1227}

TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
MEWTWO = 431
WOBBUFFET = 432
MURKROW = 463
HONCHKROW = 891

ATTACK_TO_CARD = {
    559: TAROUNTULA,
    560: SPIDOPS,
    608: MEWTWO,
    609: WOBBUFFET,  # Rocket Mirror
    610: WOBBUFFET,  # Headbutt Bounce
    652: MURKROW,    # Deceit
    653: MURKROW,    # Torment
}
ATTACK_NAMES = {
    559: "Bug Bite",
    560: "Rocket Rush",
    608: "Erasure Ball",
    609: "Rocket Mirror",
    610: "Headbutt Bounce",
    652: "Deceit",
    653: "Torment",
}

OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_EVOLVE = 9
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14
AREA_ACTIVE = 4
AREA_BENCH = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_card_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("Card ID") or "").strip()
            if not raw:
                continue
            names.setdefault(int(raw), row.get("Card Name") or raw)
    return names


def canonical_deck_bytes(deck: list[int]) -> bytes:
    return "".join(f"{card_id}\n" for card_id in deck).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_frames(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the single chronological visualizer stream embedded in a replay."""
    for step in data.get("steps") or []:
        entries = step if isinstance(step, list) else [step]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            frames = entry.get("visualize")
            if isinstance(frames, list) and frames:
                return [frame for frame in frames if isinstance(frame, dict)]
    raise ValueError("replay contains no visualizer frames")


def initial_decks(frames: list[dict[str, Any]]) -> list[list[int]]:
    for frame in frames:
        action = frame.get("action")
        if (
            isinstance(action, list)
            and len(action) == 2
            and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
        ):
            return [[int(card_id) for card_id in deck] for deck in action]
    raise ValueError("visualizer stream contains no two-deck declaration")


def detect_focal_seat(decks: list[list[int]]) -> tuple[int, bool]:
    matches = [sha256(canonical_deck_bytes(deck)).hexdigest() == FOCAL_DECK_SHA256 for deck in decks]
    if not any(matches):
        raise ValueError("neither replay deck matches the frozen ScaleBCV2 deck")
    # The validation episode is a mirror.  Analyze one actor, not both.
    return matches.index(True), all(matches)


def cards(zone: Any) -> list[dict[str, Any]]:
    return [card for card in (zone or []) if isinstance(card, dict)]


def in_play(player: dict[str, Any]) -> list[dict[str, Any]]:
    return cards(player.get("active")) + cards(player.get("bench"))


def energy_cards(pokemon: dict[str, Any]) -> list[dict[str, Any]]:
    return cards(pokemon.get("energyCards"))


def energy_units(pokemon: dict[str, Any]) -> int:
    attached = energy_cards(pokemon)
    if attached:
        # Team Rocket's Energy provides two Energy in any combination of
        # Psychic and Darkness while attached to a Team Rocket Pokémon.
        return sum(2 if int(card.get("id") or 0) == 15 else 1 for card in attached)
    return len(pokemon.get("energies") or [])


def typed_energy_units(pokemon: dict[str, Any], card_ids: set[int]) -> int:
    attached = energy_cards(pokemon)
    if not attached:
        # Older visualizer versions expose already-expanded type IDs.  Count
        # them directly; current Kaggle replays always include energyCards.
        return sum(int(energy_type) in card_ids for energy_type in pokemon.get("energies") or [])
    units = 0
    for card in attached:
        card_id = int(card.get("id") or 0)
        if card_id in card_ids:
            units += 1
        elif card_id == 15 and 5 in card_ids:
            units += 2
    return units


def ready_attacker_ids(player: dict[str, Any]) -> list[int]:
    pokes = in_play(player)
    rocket_count = sum(int(pokemon.get("id") in ROCKET_POKEMON) for pokemon in pokes)
    ready: list[int] = []
    for pokemon in pokes:
        card_id = int(pokemon.get("id") or 0)
        units = energy_units(pokemon)
        if card_id == SPIDOPS and units >= 2 and typed_energy_units(pokemon, {1}) >= 1:
            ready.append(card_id)
        elif (
            card_id == MEWTWO
            and units >= 3
            and rocket_count >= 4
            and typed_energy_units(pokemon, {5}) >= 2
        ):
            ready.append(card_id)
        elif card_id == ARTICUNO and units >= 3 and typed_energy_units(pokemon, {3}) >= 1:
            ready.append(card_id)
        elif card_id == WOBBUFFET and units >= 2 and typed_energy_units(pokemon, {5}) >= 1:
            ready.append(card_id)
        elif card_id == MURKROW and units >= 1:
            ready.append(card_id)
        elif card_id == TAROUNTULA and units >= 1 and typed_energy_units(pokemon, {1}) >= 1:
            ready.append(card_id)
    return ready


def selected_indices(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value if isinstance(item, int)]
    if isinstance(value, int):
        return [value]
    return []


def selected_options(obs: dict[str, Any], selected: Any) -> list[dict[str, Any]]:
    options = ((obs.get("select") or {}).get("option") or [])
    return [options[index] for index in selected_indices(selected) if 0 <= index < len(options)]


def option_card_id(option: dict[str, Any], state: dict[str, Any], actor: int) -> int:
    raw = option.get("cardId")
    if isinstance(raw, int):
        return raw
    index = option.get("index")
    area = option.get("area")
    if not isinstance(index, int):
        return 0
    if area == 1:
        zone = []
    else:
        players = state.get("players") or []
        if actor >= len(players):
            return 0
        zone = (players[actor] or {}).get("hand") or []
    if 0 <= index < len(zone) and isinstance(zone[index], dict):
        return int(zone[index].get("id") or 0)
    return 0


def option_target_id(option: dict[str, Any], state: dict[str, Any], actor: int) -> int:
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if area not in (AREA_ACTIVE, AREA_BENCH) or not isinstance(index, int):
        return 0
    players = state.get("players") or []
    if actor >= len(players):
        return 0
    zone_name = "active" if area == AREA_ACTIVE else "bench"
    zone = (players[actor] or {}).get(zone_name) or []
    if 0 <= index < len(zone) and isinstance(zone[index], dict):
        return int(zone[index].get("id") or 0)
    return 0


def setup_card_id(option: dict[str, Any], state: dict[str, Any], actor: int) -> int:
    index = option.get("index")
    players = state.get("players") or []
    if not isinstance(index, int) or actor >= len(players):
        return 0
    hand = (players[actor] or {}).get("hand") or []
    if 0 <= index < len(hand) and isinstance(hand[index], dict):
        return int(hand[index].get("id") or 0)
    return 0


def opponent_archetype(deck: list[int], names: dict[int, str]) -> str:
    pokemon = {names.get(card_id, str(card_id)).lower() for card_id in deck}
    if "marnie's grimmsnarl ex" in pokemon and "froslass" in pokemon:
        return "marnie_grimmsnarl_froslass"
    if any("archaludon" in name for name in pokemon) and "cinderace" in pokemon:
        return "archaludon_cinderace"
    if any("archaludon" in name for name in pokemon):
        return "archaludon"
    if "mega kangaskhan ex" in pokemon and "crustle" in pokemon:
        return "kangaskhan_crustle"
    if "great tusk" in pokemon and "crustle" in pokemon:
        return "great_tusk_crustle"
    if "crustle" in pokemon:
        return "crustle"
    if "mega lucario ex" in pokemon:
        return "mega_lucario"
    if "alakazam" in pokemon:
        return "alakazam"
    if "dragapult ex" in pokemon:
        return "dragapult"
    if SPIDOPS in deck and MEWTWO in deck:
        return "rocket_mirror"
    return "other"


def pokemon_summary(deck: list[int], names: dict[int, str]) -> str:
    counts = Counter(deck)
    pokemon_names = []
    # A compact set of known Pokémon IDs is enough for replay matchup labels;
    # names absent from this list remain visible in the machine-readable deck.
    trainer_floor = 1000
    for card_id, count in counts.items():
        name = names.get(card_id, str(card_id))
        if card_id < trainer_floor and not name.startswith("Basic {") and "Energy" not in name:
            pokemon_names.append(f"{name} x{count}")
    return ", ".join(pokemon_names)


@dataclass
class DecisionCounters:
    attachment_targets: Counter[int]
    play_cards: Counter[int]
    evolution_targets: Counter[int]
    selected_types: Counter[int]
    main_ends: int = 0
    end_with_attack: int = 0
    end_with_attachment: int = 0
    end_with_supporter: int = 0
    end_with_evolution: int = 0
    wob_setup_forced: bool = False
    wob_voluntary_plays: int = 0
    wob_attachments: int = 0
    wob_attack_over_ready_spidops: int = 0


def analyze_replay(path: Path, names: dict[int, str]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = iter_frames(data)
    decks = initial_decks(frames)
    seat, validation_mirror = detect_focal_seat(decks)
    opponent_seat = 1 - seat
    rewards = data.get("rewards") or [None, None]
    teams = (data.get("info") or {}).get("TeamNames") or ["player_0", "player_1"]
    first_player: int | None = None
    setup_active: int | None = None
    setup_legal: list[int] = []
    attack_ids: Counter[int] = Counter()
    attacker_ids: Counter[int] = Counter()
    counters = DecisionCounters(Counter(), Counter(), Counter(), Counter())
    max_ready = 0
    max_rocket = 0
    max_spidops = 0
    max_board = 0
    first_attack_turn: int | None = None
    first_prize_turn: int | None = None
    min_prizes = 6
    opponent_min_prizes = 6
    first_second_ready_turn: int | None = None
    mewtwo_seen = False
    mewtwo_attacked = False
    wobbuffet_seen_in_play = False
    active_decision_frames: Counter[int] = Counter()
    decisions = 0

    for frame in frames:
        full_current = frame.get("current") or {}
        if first_player is None and full_current.get("firstPlayer") in (0, 1):
            first_player = int(full_current["firstPlayer"])
        full_players = full_current.get("players") or []
        turn = full_current.get("turn")
        if isinstance(turn, int) and turn > 0 and len(full_players) == 2:
            mine_full = full_players[seat] or {}
            opp_full = full_players[opponent_seat] or {}
            own_prizes = mine_full.get("prize")
            opp_prizes = opp_full.get("prize")
            if isinstance(own_prizes, list) and own_prizes:
                remaining = len(own_prizes)
                min_prizes = min(min_prizes, remaining)
                if remaining < 6 and first_prize_turn is None:
                    first_prize_turn = turn
            if isinstance(opp_prizes, list) and opp_prizes:
                opponent_min_prizes = min(opponent_min_prizes, len(opp_prizes))

        for log in frame.get("logs") or []:
            if not isinstance(log, dict) or log.get("playerIndex") != seat:
                continue
            if log.get("type") == "Attack":
                attack_id = int(log.get("attackId") or 0)
                attacker = int(log.get("cardId") or ATTACK_TO_CARD.get(attack_id, 0))
                attack_ids[attack_id] += 1
                attacker_ids[attacker] += 1
                if first_attack_turn is None and isinstance(turn, int):
                    first_attack_turn = turn
                mewtwo_attacked = mewtwo_attacked or attacker == MEWTWO

        obs = frame.get("obs") or {}
        state = obs.get("current") or {}
        selection = obs.get("select") or {}
        if state.get("yourIndex") != seat or not selection:
            continue
        decisions += 1
        players = state.get("players") or []
        if len(players) != 2:
            continue
        mine = players[seat] or {}
        pokes = in_play(mine)
        ids = [int(pokemon.get("id") or 0) for pokemon in pokes]
        active_id = int(((mine.get("active") or [{}])[0] or {}).get("id") or 0)
        active_decision_frames[active_id] += 1
        ready = ready_attacker_ids(mine)
        rocket = sum(card_id in ROCKET_POKEMON for card_id in ids)
        max_ready = max(max_ready, len(ready))
        max_rocket = max(max_rocket, rocket)
        max_spidops = max(max_spidops, ids.count(SPIDOPS))
        max_board = max(max_board, len(ids))
        if len(ready) >= 2 and first_second_ready_turn is None:
            first_second_ready_turn = int(state.get("turn") or 0)
        mewtwo_seen = mewtwo_seen or MEWTWO in ids
        wobbuffet_seen_in_play = wobbuffet_seen_in_play or WOBBUFFET in ids

        options = selection.get("option") or []
        chosen = selected_options(obs, frame.get("selected"))
        context = int(selection.get("context") or 0)
        if context == 1 and chosen:
            setup_legal = [setup_card_id(option, state, seat) for option in options]
            setup_active = setup_card_id(chosen[0], state, seat)
            counters.wob_setup_forced = setup_active == WOBBUFFET and len(options) == 1

        for option in chosen:
            option_type = int(option.get("type") if option.get("type") is not None else -1)
            counters.selected_types[option_type] += 1
            if option_type == OPTION_PLAY:
                card_id = option_card_id(option, state, seat)
                counters.play_cards[card_id] += 1
                if card_id == WOBBUFFET:
                    counters.wob_voluntary_plays += 1
            elif option_type == OPTION_ATTACH:
                target = option_target_id(option, state, seat)
                counters.attachment_targets[target] += 1
                if target == WOBBUFFET:
                    counters.wob_attachments += 1
            elif option_type == OPTION_EVOLVE:
                counters.evolution_targets[option_target_id(option, state, seat)] += 1
            elif option_type == OPTION_ATTACK:
                if active_id == WOBBUFFET and SPIDOPS in ready:
                    available_types = {int(item.get("type") or -1) for item in options}
                    if OPTION_RETREAT in available_types:
                        counters.wob_attack_over_ready_spidops += 1

        if context == 0 and any(int(option.get("type") or -1) == OPTION_END for option in chosen):
            counters.main_ends += 1
            available_types = {int(option.get("type") or -1) for option in options}
            counters.end_with_attack += int(OPTION_ATTACK in available_types)
            counters.end_with_attachment += int(
                OPTION_ATTACH in available_types and not bool(state.get("energyAttached"))
            )
            counters.end_with_evolution += int(OPTION_EVOLVE in available_types)
            counters.end_with_supporter += int(
                not bool(state.get("supporterPlayed"))
                and any(
                    int(option.get("type") or -1) == OPTION_PLAY
                    and option_card_id(option, state, seat) in SUPPORTERS
                    for option in options
                )
            )

    focal_deck = decks[seat]
    focal_hash = sha256(canonical_deck_bytes(focal_deck)).hexdigest()
    final_reward = rewards[seat] if seat < len(rewards) else None
    won = final_reward is not None and final_reward > (rewards[opponent_seat] if opponent_seat < len(rewards) else final_reward)
    lost = final_reward is not None and final_reward < (rewards[opponent_seat] if opponent_seat < len(rewards) else final_reward)
    return {
        "episode_id": int((data.get("info") or {}).get("EpisodeId") or 0),
        "file": path.name,
        "file_sha256": file_sha256(path),
        "seat": seat,
        "validation_mirror": validation_mirror,
        "team": teams[seat],
        "opponent": teams[opponent_seat],
        "opponent_archetype": opponent_archetype(decks[opponent_seat], names),
        "opponent_pokemon": pokemon_summary(decks[opponent_seat], names),
        "opponent_deck": decks[opponent_seat],
        "focal_deck": focal_deck,
        "focal_deck_sha256": focal_hash,
        "deck_integrity": focal_hash == FOCAL_DECK_SHA256,
        "first_player": first_player,
        "started_first": first_player == seat,
        "rewards": rewards,
        "won": won,
        "lost": lost,
        "steps": len(data.get("steps") or []),
        "decisions": decisions,
        "setup_active": setup_active,
        "setup_active_name": names.get(setup_active or 0, str(setup_active)),
        "setup_legal": setup_legal,
        "setup_forced": len(setup_legal) == 1,
        "attack_ids": dict(sorted(attack_ids.items())),
        "attack_names": {
            ATTACK_NAMES.get(attack_id, str(attack_id)): count for attack_id, count in sorted(attack_ids.items())
        },
        "attacker_ids": dict(sorted(attacker_ids.items())),
        "attacker_names": {
            names.get(card_id, str(card_id)): count for card_id, count in sorted(attacker_ids.items())
        },
        "attack_count": sum(attack_ids.values()),
        "first_attack_turn": first_attack_turn,
        "first_prize_turn": first_prize_turn,
        "prizes_taken": 6 - min_prizes,
        "opponent_prizes_taken": 6 - opponent_min_prizes,
        "maximum_ready_attackers": max_ready,
        "first_second_ready_turn": first_second_ready_turn,
        "maximum_rocket_in_play": max_rocket,
        "maximum_spidops_in_play": max_spidops,
        "maximum_board_size": max_board,
        "mewtwo_seen": mewtwo_seen,
        "mewtwo_attacked": mewtwo_attacked,
        "wobbuffet_seen_in_play": wobbuffet_seen_in_play,
        "wobbuffet_attacks": attack_ids[609] + attack_ids[610],
        "wobbuffet_setup_forced": counters.wob_setup_forced,
        "wobbuffet_voluntary_plays": counters.wob_voluntary_plays,
        "wobbuffet_attachments": counters.wob_attachments,
        "wobbuffet_attack_over_ready_spidops": counters.wob_attack_over_ready_spidops,
        "attachment_targets": dict(sorted(counters.attachment_targets.items())),
        "attachment_target_names": {
            names.get(card_id, str(card_id)): count
            for card_id, count in sorted(counters.attachment_targets.items())
        },
        "played_cards": dict(sorted(counters.play_cards.items())),
        "evolution_targets": dict(sorted(counters.evolution_targets.items())),
        "main_ends": counters.main_ends,
        "end_with_attack": counters.end_with_attack,
        "end_with_attachment": counters.end_with_attachment,
        "end_with_supporter": counters.end_with_supporter,
        "end_with_evolution": counters.end_with_evolution,
        "active_decision_frames": {
            names.get(card_id, str(card_id)): count for card_id, count in sorted(active_decision_frames.items())
        },
    }


def outcome(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(row["won"]) for row in rows)
    losses = sum(int(row["lost"]) for row in rows)
    return {
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": len(rows) - wins - losses,
        "win_rate": wins / len(rows) if rows else None,
    }


def aggregate(games: list[dict[str, Any]]) -> dict[str, Any]:
    public = [game for game in games if not game["validation_mirror"]]
    by_matchup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attacks: Counter[str] = Counter()
    attackers: Counter[str] = Counter()
    attachments: Counter[str] = Counter()
    for game in games:
        by_matchup[game["opponent_archetype"]].append(game)
        by_setup[game["setup_active_name"]].append(game)
        attacks.update(game["attack_names"])
        attackers.update(game["attacker_names"])
        attachments.update(game["attachment_target_names"])
    losses = [game for game in public if game["lost"]]
    total_attacks = sum(attacks.values())
    return {
        "all_available": outcome(games),
        "public_only": outcome(public),
        "started_first_public": outcome([game for game in public if game["started_first"]]),
        "started_second_public": outcome([game for game in public if not game["started_first"]]),
        "deck_integrity_all": all(game["deck_integrity"] for game in games),
        "matchups": {name: outcome(rows) for name, rows in sorted(by_matchup.items())},
        "setup_actives": {name: outcome(rows) for name, rows in sorted(by_setup.items())},
        "attack_names": dict(attacks.most_common()),
        "attacker_names": dict(attackers.most_common()),
        "attachment_target_names": dict(attachments.most_common()),
        "total_attacks": total_attacks,
        "wobbuffet": {
            "deck_copies": Counter(games[0]["focal_deck"])[WOBBUFFET] if games else 0,
            "setup_games": sum(game["setup_active"] == WOBBUFFET for game in games),
            "forced_setup_games": sum(game["wobbuffet_setup_forced"] for game in games),
            "seen_in_play_games": sum(game["wobbuffet_seen_in_play"] for game in games),
            "voluntary_plays": sum(game["wobbuffet_voluntary_plays"] for game in games),
            "attachments": sum(game["wobbuffet_attachments"] for game in games),
            "attacks": sum(game["wobbuffet_attacks"] for game in games),
            "attack_share": (
                sum(game["wobbuffet_attacks"] for game in games) / total_attacks if total_attacks else 0.0
            ),
            "attack_games": sum(game["wobbuffet_attacks"] > 0 for game in games),
            "attack_over_ready_spidops": sum(
                game["wobbuffet_attack_over_ready_spidops"] for game in games
            ),
        },
        "loss_buckets_public": {
            "mewtwo_never_attacked": sum(not game["mewtwo_attacked"] for game in losses),
            "no_second_ready_attacker": sum(game["maximum_ready_attackers"] < 2 for game in losses),
            "never_reached_four_rocket": sum(game["maximum_rocket_in_play"] < 4 for game in losses),
            "took_zero_prizes": sum(game["prizes_taken"] == 0 for game in losses),
            "ended_with_legal_supporter": sum(game["end_with_supporter"] > 0 for game in losses),
            "ended_with_legal_attachment": sum(game["end_with_attachment"] > 0 for game in losses),
            "ended_with_legal_evolution": sum(game["end_with_evolution"] > 0 for game in losses),
            "ended_with_legal_attack": sum(game["end_with_attack"] > 0 for game in losses),
        },
    }


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    games = payload["games"]
    public = summary["public_only"]
    wob = summary["wobbuffet"]
    lines = [
        "# ScaleBCV2 Kaggle replay audit",
        "",
        f"Submission: `{payload['submission_id']}` (`HelbrosRocketSpidopsScaleBCV2.tar.gz`)",
        f"Replay files audited: **{len(games)}** ({public['games']} public plus "
        f"{len(games) - public['games']} validation mirror)",
        f"Frozen deck SHA-256: `{FOCAL_DECK_SHA256}`; integrity across every replay: "
        f"**{'pass' if summary['deck_integrity_all'] else 'FAIL'}**.",
        "",
        "## Outcome",
        "",
        "| Slice | Games | W-L-D | Win rate |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("all_available", "All available"),
        ("public_only", "Public only"),
        ("started_first_public", "Public, started first"),
        ("started_second_public", "Public, started second"),
    ):
        row = summary[key]
        lines.append(
            f"| {label} | {row['games']} | {row['wins']}-{row['losses']}-{row['draws']} | "
            f"{percentage(row['win_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Wobbuffet observation",
            "",
            f"The deck contains {wob['deck_copies']} Wobbuffet. It opened Active in "
            f"{wob['setup_games']}/{len(games)} games, was forced by a one-Basic opening in "
            f"{wob['forced_setup_games']}, attacked {wob['attacks']} times in {wob['attack_games']} game, "
            f"and made {percentage(wob['attack_share'])} of all focal attacks.",
            "",
            "| Measure | Count |",
            "|---|---:|",
            f"| Games seen in play | {wob['seen_in_play_games']} |",
            f"| Voluntary bench plays | {wob['voluntary_plays']} |",
            f"| Attachments targeted to Wobbuffet | {wob['attachments']} |",
            f"| Wobbuffet attacks with a ready Spidops and legal retreat | {wob['attack_over_ready_spidops']} |",
            "",
            "## Matchups",
            "",
            "| Opponent archetype | Games | W-L | Win rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, row in summary["matchups"].items():
        lines.append(
            f"| {name} | {row['games']} | {row['wins']}-{row['losses']} | {percentage(row['win_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Attackers",
            "",
            "| Attacker | Attacks |",
            "|---|---:|",
        ]
    )
    for name, count in summary["attacker_names"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## Public losses",
            "",
            "| Episode | Opponent | Setup | Attacks | Prizes | Max ready | Mewtwo attacked | Flags |",
            "|---:|---|---|---|---:|---:|---|---|",
        ]
    )
    for game in games:
        if not game["lost"] or game["validation_mirror"]:
            continue
        flags = []
        if game["maximum_ready_attackers"] < 2:
            flags.append("no second attacker")
        if not game["mewtwo_attacked"]:
            flags.append("no Mewtwo attack")
        if game["maximum_rocket_in_play"] < 4:
            flags.append("under 4 Rocket")
        if game["wobbuffet_attacks"]:
            flags.append(f"Wobbuffet attacks={game['wobbuffet_attacks']}")
        lines.append(
            f"| {game['episode_id']} | {game['opponent_archetype']} | {game['setup_active_name']} | "
            f"{', '.join(f'{name}×{count}' for name, count in game['attacker_names'].items()) or 'none'} | "
            f"{game['prizes_taken']} | {game['maximum_ready_attackers']} | "
            f"{'yes' if game['mewtwo_attacked'] else 'no'} | {', '.join(flags) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Per-game ledger",
            "",
            "| Episode | Type | Result | Seat/order | Opponent | Setup | First attack | First prize | Prizes | Attackers |",
            "|---:|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for game in games:
        result = "W" if game["won"] else "L" if game["lost"] else "D"
        kind = "validation" if game["validation_mirror"] else "public"
        order = "first" if game["started_first"] else "second"
        attackers = ", ".join(f"{name}×{count}" for name, count in game["attacker_names"].items()) or "none"
        lines.append(
            f"| {game['episode_id']} | {kind} | {result} | seat {game['seat']}, {order} | "
            f"{game['opponent_archetype']} | {game['setup_active_name']} | "
            f"{game['first_attack_turn'] if game['first_attack_turn'] is not None else '—'} | "
            f"{game['first_prize_turn'] if game['first_prize_turn'] is not None else '—'} | "
            f"{game['prizes_taken']} | {attackers} |"
        )
    lines.extend(
        [
            "",
            "The replay associations are descriptive, not causal. Deck substitutions and policy guards need a "
            "separate preregistered official-engine A/B before deployment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    names = load_card_names(args.cards)
    replay_paths = sorted(args.replays.glob("episode-*-replay.json"))
    if not replay_paths:
        raise FileNotFoundError(f"no Kaggle replay files under {args.replays}")
    games = [analyze_replay(path, names) for path in replay_paths]
    payload = {
        "schema_version": 1,
        "submission_id": 55172263,
        "submission_file": "HelbrosRocketSpidopsScaleBCV2.tar.gz",
        "focal_deck_sha256": FOCAL_DECK_SHA256,
        "replay_directory": str(args.replays.resolve()),
        "summary": aggregate(games),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
