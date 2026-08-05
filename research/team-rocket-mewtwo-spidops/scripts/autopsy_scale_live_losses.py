#!/usr/bin/env python3
"""Quantify recurring policy leaks in pulled Team Rocket arena games.

The analysis is intentionally based on real outcomes and public observations.
It does not use the policy's own value estimate to decide what was a mistake.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


BOT = "HelbrosRocketSpidopsScaleBCV2"
ROCKET_POKEMON = {400, 401, 414, 431, 432, 463}
SUPPORTERS = {1216, 1217, 1218, 1219, 1220, 1227}
NAMES = {
    400: "Tarountula",
    401: "Spidops",
    414: "Articuno",
    431: "Mewtwo",
    432: "Wobbuffet",
    463: "Murkrow",
}
BASIC_POKEMON = {400, 414, 431, 432, 463}
ACTIVE = 4
BENCH = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bot",
        default=BOT,
        help=f"Exact focal bot identity (default: {BOT})",
    )
    parser.add_argument(
        "--opponent-contains",
        action="append",
        default=[],
        help=(
            "Only analyze games whose opponent name contains this text; may be "
            "repeated and matching is case-insensitive"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-game snapshots from autopsy.json after aggregation",
    )
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def replay_players(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        first = handle.readline()
    return [str(value) for value in json.loads(first)["meta"]["players"]]


def card_ids(value: Any) -> Iterable[int]:
    if isinstance(value, list):
        for item in value:
            yield from card_ids(item)
    elif isinstance(value, dict):
        if isinstance(value.get("id"), int):
            yield int(value["id"])
        for key in ("preEvolution", "tools", "energyCards"):
            yield from card_ids(value.get(key) or [])


def option_card_id(
    option: dict[str, Any], state: dict[str, Any], actor: int
) -> int:
    value = option.get("cardId")
    if isinstance(value, int):
        return value
    if (
        option.get("type") in (7, 8, 9) or option.get("area") == 2
    ) and isinstance(option.get("index"), int):
        hand = ((state.get("players") or [{}, {}])[actor] or {}).get("hand") or []
        index = int(option["index"])
        if 0 <= index < len(hand) and isinstance(hand[index], dict):
            return int(hand[index].get("id") or 0)
    return 0


def option_target_card_id(
    option: dict[str, Any], state: dict[str, Any], actor: int
) -> int:
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if not isinstance(index, int) or area not in (ACTIVE, BENCH):
        return 0
    player = ((state.get("players") or [{}, {}])[actor] or {})
    zone = player.get("active" if area == ACTIVE else "bench") or []
    if 0 <= index < len(zone) and isinstance(zone[index], dict):
        return int(zone[index].get("id") or 0)
    return 0


def in_play(player: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        pokemon
        for pokemon in (player.get("active") or []) + (player.get("bench") or [])
        if isinstance(pokemon, dict)
    ]


def energy_units(pokemon: dict[str, Any]) -> int:
    return len(pokemon.get("energies") or [])


def ready_attackers(player: dict[str, Any]) -> int:
    pokes = in_play(player)
    rocket_count = sum(int(pokemon.get("id") in ROCKET_POKEMON) for pokemon in pokes)
    ready = 0
    for pokemon in pokes:
        card_id = int(pokemon.get("id") or 0)
        units = energy_units(pokemon)
        if card_id == 401 and units >= 2:
            ready += 1
        elif card_id == 431 and units >= 3 and rocket_count >= 4:
            ready += 1
        elif card_id == 414 and units >= 3:
            ready += 1
        elif card_id == 432 and units >= 2:
            ready += 1
        elif card_id == 463 and units >= 1:
            ready += 1
        elif card_id == 400 and units >= 1:
            ready += 1
    return ready


def family(name: str) -> str:
    lowered = name.lower()
    for token in (
        "marnie",
        "alakazam",
        "soldier",
        "starmie",
        "rocket",
        "dragapult",
        "lucario",
        "archaludon",
        "trevenant",
        "garchomp",
        "crustle",
        "abomasnow",
        "gardevoir",
        "ragingbolt",
        "hydrapple",
        "comfey",
    ):
        if token in lowered:
            return token
    return "other"


def analyze_game(path: Path, bot: str = BOT) -> dict[str, Any]:
    records = rows(path)
    meta = records[0]["meta"]
    players = list(meta["players"])
    if bot not in players:
        raise ValueError(f"{path.name} does not contain {bot}")
    seat = players.index(bot)
    opponent_name = players[1 - seat]
    terminal = records[-1]["result"]
    winner_value = terminal.get("winner")
    winner = int(winner_value) if winner_value is not None else None
    won = winner == seat
    lost = winner == 1 - seat
    draw = winner is None
    decisions = [record for record in records[1:-1] if "obs" in record]
    focal = [record for record in decisions if int(record.get("p", -1)) == seat]
    first_player = None
    setup_active = None
    attacks = Counter()
    played = Counter()
    selected_types = Counter()
    attachment_targets = Counter()
    evolution_targets = Counter()
    early_basic_plays = Counter()
    joint_spidops_mewtwo_attachment_choices = Counter()
    multi_basic_play_choices = Counter()
    card_selection_picks = Counter()
    main_ends = 0
    early_ends = 0
    end_with_attack = 0
    end_with_supporter = 0
    end_with_attach = 0
    end_with_evolve = 0
    maximum_ready = 0
    maximum_rocket = 0
    maximum_spidops = 0
    maximum_mewtwo_energy = 0
    mewtwo_seen = False
    mewtwo_attacked = False
    first_attack_turn = None
    first_second_ready_turn = None
    first_two_spidops_turn = None
    maximum_board_size = 0
    snapshots: list[dict[str, Any]] = []

    for record in focal:
        obs = record["obs"]
        state = obs.get("current") or {}
        selection = obs.get("select") or {}
        action = list(record.get("action") or [])
        actor = int(state.get("yourIndex", seat) or seat)
        if first_player is None and state.get("firstPlayer") in (0, 1):
            first_player = int(state["firstPlayer"])
        players_state = state.get("players") or [{}, {}]
        mine = players_state[seat] or {}
        pokes = in_play(mine)
        ids = [int(pokemon.get("id") or 0) for pokemon in pokes]
        ready = ready_attackers(mine)
        rocket = sum(int(card_id in ROCKET_POKEMON) for card_id in ids)
        maximum_ready = max(maximum_ready, ready)
        maximum_rocket = max(maximum_rocket, rocket)
        maximum_spidops = max(maximum_spidops, ids.count(401))
        maximum_board_size = max(maximum_board_size, len(ids))
        if ready >= 2 and first_second_ready_turn is None:
            first_second_ready_turn = int(state.get("turn") or 0)
        if ids.count(401) >= 2 and first_two_spidops_turn is None:
            first_two_spidops_turn = int(state.get("turn") or 0)
        mewtwo_seen = mewtwo_seen or 431 in ids
        for pokemon in pokes:
            if int(pokemon.get("id") or 0) == 431:
                maximum_mewtwo_energy = max(
                    maximum_mewtwo_energy, energy_units(pokemon)
                )
        snapshots.append(
            {
                "turn": int(state.get("turn") or 0),
                "ready": ready,
                "rocket": rocket,
                "spidops": ids.count(401),
                "mewtwo": ids.count(431),
                "prizes": len(mine.get("prize") or []),
                "deck": int(mine.get("deckCount") or 0),
            }
        )

        if selection.get("context") == 1 and action:
            options = selection.get("option") or []
            index = action[0]
            if 0 <= index < len(options):
                option = options[index]
                area_index = option.get("index")
                hand = mine.get("hand") or []
                if isinstance(area_index, int) and 0 <= area_index < len(hand):
                    setup_active = int((hand[area_index] or {}).get("id") or 0)

        options = selection.get("option") or []
        selected_options = [
            options[index]
            for index in action
            if isinstance(index, int) and 0 <= index < len(options)
        ]
        available_attachment_targets = {
            option_target_card_id(option, state, actor)
            for option in options
            if option.get("type") == 8
        }
        if {401, 431}.issubset(available_attachment_targets):
            selected_attachment_targets = [
                option_target_card_id(option, state, actor)
                for option in selected_options
                if option.get("type") == 8
            ]
            label = (
                str(selected_attachment_targets[0])
                if selected_attachment_targets
                else "deferred"
            )
            joint_spidops_mewtwo_attachment_choices[label] += 1
        available_basic_plays = {
            option_card_id(option, state, actor)
            for option in options
            if option.get("type") == 7
            and option_card_id(option, state, actor) in BASIC_POKEMON
        }
        if len(available_basic_plays) >= 2:
            selected_basics = [
                option_card_id(option, state, actor)
                for option in selected_options
                if option.get("type") == 7
                and option_card_id(option, state, actor) in BASIC_POKEMON
            ]
            label = str(selected_basics[0]) if selected_basics else "deferred"
            multi_basic_play_choices[label] += 1
        for option in selected_options:
            option_type_value = option.get("type")
            option_type = int(option_type_value) if option_type_value is not None else -1
            selected_types[option_type] += 1
            if option_type == 13:
                attack_id = int(option.get("attackId") or 0)
                attacks[attack_id] += 1
                if first_attack_turn is None:
                    first_attack_turn = int(state.get("turn") or 0)
                if attack_id == 608:
                    mewtwo_attacked = True
            elif option_type == 7:
                card_id = option_card_id(option, state, actor)
                played[card_id] += 1
                if card_id in BASIC_POKEMON and int(state.get("turn") or 0) <= 5:
                    early_basic_plays[card_id] += 1
            elif option_type == 8:
                attachment_targets[option_target_card_id(option, state, actor)] += 1
            elif option_type == 9:
                evolution_targets[option_target_card_id(option, state, actor)] += 1
            elif option_type == 3:
                selection_context = int(selection.get("context") or 0)
                card_selection_picks[
                    f"context_{selection_context}:card_{option_card_id(option, state, actor)}"
                ] += 1

        if selection.get("context") == 0 and any(
            int(option.get("type") or -1) == 14 for option in selected_options
        ):
            main_ends += 1
            turn = int(state.get("turn") or 0)
            if turn <= 6:
                early_ends += 1
            available_types = {int(option.get("type") or -1) for option in options}
            if 13 in available_types:
                end_with_attack += 1
            if 8 in available_types and not bool(state.get("energyAttached")):
                end_with_attach += 1
            if 9 in available_types:
                end_with_evolve += 1
            if not bool(state.get("supporterPlayed")) and any(
                int(option.get("type") or -1) == 7
                and option_card_id(option, state, actor) in SUPPORTERS
                for option in options
            ):
                end_with_supporter += 1

    last_state = (decisions[-1].get("obs") or {}).get("current") or {}
    last_players = last_state.get("players") or [{}, {}]
    final_mine = last_players[seat] or {}
    final_opp = last_players[1 - seat] or {}
    return {
        "file": path.name,
        "opponent": opponent_name,
        "family": family(opponent_name),
        "seat": seat,
        "first_player": first_player,
        "started_first": first_player == seat,
        "won": won,
        "lost": lost,
        "draw": draw,
        "steps": int(terminal.get("steps") or 0),
        "setup_active": setup_active,
        "setup_active_name": NAMES.get(setup_active, str(setup_active)),
        "attacks": dict(attacks),
        "attack_count": sum(attacks.values()),
        "first_attack_turn": first_attack_turn,
        "mewtwo_attacked": mewtwo_attacked,
        "played": dict(played),
        "selected_types": dict(selected_types),
        "attachment_targets": dict(attachment_targets),
        "evolution_targets": dict(evolution_targets),
        "early_basic_plays": dict(early_basic_plays),
        "joint_spidops_mewtwo_attachment_choices": dict(
            joint_spidops_mewtwo_attachment_choices
        ),
        "multi_basic_play_choices": dict(multi_basic_play_choices),
        "card_selection_picks": dict(card_selection_picks),
        "main_ends": main_ends,
        "early_ends": early_ends,
        "end_with_attack": end_with_attack,
        "end_with_supporter": end_with_supporter,
        "end_with_attach": end_with_attach,
        "end_with_evolve": end_with_evolve,
        "maximum_ready_attackers": maximum_ready,
        "maximum_rocket_in_play": maximum_rocket,
        "maximum_spidops_in_play": maximum_spidops,
        "maximum_board_size": maximum_board_size,
        "first_second_ready_turn": first_second_ready_turn,
        "first_two_spidops_turn": first_two_spidops_turn,
        "mewtwo_seen": mewtwo_seen,
        "maximum_mewtwo_energy": maximum_mewtwo_energy,
        "final_prizes": len(final_mine.get("prize") or []),
        "opponent_final_prizes": len(final_opp.get("prize") or []),
        "final_deck": int(final_mine.get("deckCount") or 0),
        "snapshots": snapshots,
    }


def rate(rows_: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows_)
    wins = sum(int(row["won"]) for row in rows_)
    draws = sum(int(row.get("draw", False)) for row in rows_)
    return {
        "games": games,
        "wins": wins,
        "losses": games - wins - draws,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
    }


def mean(rows_: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows_) / max(1, len(rows_))


def aggregate(games: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        by_family[game["family"]].append(game)
        by_setup[game["setup_active_name"]].append(game)
    wins = [game for game in games if game["won"]]
    losses = [game for game in games if game["lost"]]

    def normalized_counter(
        selected: list[dict[str, Any]], key: str
    ) -> dict[str, float]:
        total: Counter[str] = Counter()
        for row in selected:
            total.update({str(k): int(v) for k, v in row[key].items()})
        return {
            name: value / max(1, len(selected))
            for name, value in sorted(total.items(), key=lambda item: (-item[1], item[0]))
        }

    def outcome_metrics(selected: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mean_attack_count": mean(selected, "attack_count"),
            "mean_first_attack_turn": mean(
                [row for row in selected if row["first_attack_turn"] is not None],
                "first_attack_turn",
            ),
            "mean_max_ready_attackers": mean(selected, "maximum_ready_attackers"),
            "mean_max_rocket_in_play": mean(selected, "maximum_rocket_in_play"),
            "mean_max_spidops_in_play": mean(selected, "maximum_spidops_in_play"),
            "mean_max_board_size": mean(selected, "maximum_board_size"),
            "mean_final_prizes": mean(selected, "final_prizes"),
            "mewtwo_attack_game_rate": sum(
                int(row["mewtwo_attacked"]) for row in selected
            )
            / max(1, len(selected)),
            "no_second_ready_attacker_rate": sum(
                int(row["maximum_ready_attackers"] < 2) for row in selected
            )
            / max(1, len(selected)),
            "never_reached_four_rocket_rate": sum(
                int(row["maximum_rocket_in_play"] < 4) for row in selected
            )
            / max(1, len(selected)),
            "end_with_supporter_game_rate": sum(
                int(row["end_with_supporter"] > 0) for row in selected
            )
            / max(1, len(selected)),
            "end_with_attach_game_rate": sum(
                int(row["end_with_attach"] > 0) for row in selected
            )
            / max(1, len(selected)),
            "end_with_evolve_game_rate": sum(
                int(row["end_with_evolve"] > 0) for row in selected
            )
            / max(1, len(selected)),
            "end_with_attack_count": sum(row["end_with_attack"] for row in selected),
            "selected_option_types_per_game": normalized_counter(
                selected, "selected_types"
            ),
            "played_cards_per_game": normalized_counter(selected, "played"),
            "early_basic_plays_per_game": normalized_counter(
                selected, "early_basic_plays"
            ),
            "attachment_targets_per_game": normalized_counter(
                selected, "attachment_targets"
            ),
            "evolution_targets_per_game": normalized_counter(
                selected, "evolution_targets"
            ),
            "attack_ids_per_game": normalized_counter(selected, "attacks"),
            "joint_spidops_mewtwo_attachment_choices_per_game": normalized_counter(
                selected, "joint_spidops_mewtwo_attachment_choices"
            ),
            "multi_basic_play_choices_per_game": normalized_counter(
                selected, "multi_basic_play_choices"
            ),
            "card_selection_picks_per_game": normalized_counter(
                selected, "card_selection_picks"
            ),
        }

    return {
        "overall": rate(games),
        "started_first": rate([game for game in games if game["started_first"]]),
        "started_second": rate([game for game in games if not game["started_first"]]),
        "families": {
            name: rate(selected)
            for name, selected in sorted(
                by_family.items(), key=lambda item: (-len(item[1]), item[0])
            )
        },
        "setup_actives": {
            name: rate(selected)
            for name, selected in sorted(
                by_setup.items(), key=lambda item: (-len(item[1]), item[0])
            )
        },
        "wins": outcome_metrics(wins),
        "losses": outcome_metrics(losses),
        "loss_buckets": {
            "no_second_ready_attacker": sum(
                int(game["maximum_ready_attackers"] < 2) for game in losses
            ),
            "never_reached_four_rocket": sum(
                int(game["maximum_rocket_in_play"] < 4) for game in losses
            ),
            "mewtwo_never_attacked": sum(
                int(not game["mewtwo_attacked"]) for game in losses
            ),
            "ended_with_legal_supporter": sum(
                int(game["end_with_supporter"] > 0) for game in losses
            ),
            "ended_with_legal_attachment": sum(
                int(game["end_with_attach"] > 0) for game in losses
            ),
            "ended_with_legal_evolution": sum(
                int(game["end_with_evolve"] > 0) for game in losses
            ),
            "ended_with_legal_attack": sum(
                int(game["end_with_attack"] > 0) for game in losses
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# {report['bot']} live-game autopsy",
        "",
        f"Analyzed {summary['overall']['games']} pulled official-arena games.",
        "",
        "## Outcome",
        "",
        "| Slice | Games | W-L | Win rate |",
        "|---|---:|---:|---:|",
    ]
    for label in ("overall", "started_first", "started_second"):
        value = summary[label]
        lines.append(
            f"| {label.replace('_', ' ')} | {value['games']} | "
            f"{value['wins']}-{value['losses']} | {100 * value['win_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Largest loss buckets",
            "",
            "| Pattern | Loss games |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(
        summary["loss_buckets"].items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines.extend(
        [
            "",
            "## By opponent family",
            "",
            "| Family | Games | W-L | Win rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, value in summary["families"].items():
        lines.append(
            f"| {name} | {value['games']} | {value['wins']}-{value['losses']} | "
            f"{100 * value['win_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The buckets are associations for lever discovery, not causal claims. "
            "Any proposed post-filter still requires a controlled A/B and broad "
            "matchup regression gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    paths = sorted(args.replays.resolve().rglob("*.jsonl"))
    filters = [value.casefold() for value in args.opponent_contains]
    if filters:
        selected_paths = []
        for path in paths:
            players = replay_players(path)
            if args.bot not in players:
                continue
            opponent = players[1 - players.index(args.bot)]
            if any(value in opponent.casefold() for value in filters):
                selected_paths.append(path)
        paths = selected_paths
    games = [analyze_game(path, args.bot) for path in paths]
    if not games:
        raise ValueError("no games matched the focal bot and opponent filters")
    report = {
        "schema_version": 1,
        "bot": args.bot,
        "source": str(args.replays.resolve()),
        "opponent_contains": args.opponent_contains,
        "summary": aggregate(games),
        "games": [] if args.summary_only else games,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "autopsy.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "autopsy.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
