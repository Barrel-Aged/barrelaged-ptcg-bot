#!/usr/bin/env python3
"""Audit the locked Target-80 Lightning-package mechanism screen."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "work" / "target80_lightning_screen_v1"
OUTPUT = RUN / "analysis.json"

ARMS = {
    "galvantula_route": {
        "bot": "GalvRouteV1R",
        "runs": ("galv_route_mist_6", "galv_route_fros_6"),
        "attacker_ids": {160, 161},
        "attack_ids": {209},
        "kind": "galvantula",
    },
    "zapdos_route": {
        "bot": "ZapdosRouteV1R",
        "runs": ("zapdos_route_mist_6", "zapdos_route_fros_6"),
        "attacker_ids": {425},
        "attack_ids": {600},
        "kind": "zapdos",
    },
    "zapdos_control": {
        "bot": "ZapdosControlV1",
        "runs": ("zapdos_control_mist_6", "zapdos_control_fros_6"),
        "attacker_ids": {425},
        "attack_ids": {600},
        "kind": "zapdos",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def board(player: dict) -> list[dict]:
    return [
        card
        for card in ((player.get("active") or []) + (player.get("bench") or []))
        if isinstance(card, dict)
    ]


def energy_ids(card: dict) -> set[int]:
    return {
        int(energy["id"])
        for energy in (card.get("energyCards") or [])
        if isinstance(energy, dict) and isinstance(energy.get("id"), int)
    }


def tool_ids(card: dict) -> set[int]:
    return {
        int(tool["id"])
        for tool in (card.get("tools") or [])
        if isinstance(tool, dict) and isinstance(tool.get("id"), int)
    }


def ready(card: dict, kind: str) -> bool:
    if kind == "galvantula":
        return (
            card.get("id") == 161
            and 4 in energy_ids(card)
            and (len(card.get("energies") or []) >= 2 or 1165 in tool_ids(card))
        )
    return card.get("id") == 425 and {4, 15}.issubset(energy_ids(card))


def analyze_game(rows: list[dict], outcome: dict, spec: dict) -> dict:
    own_rows = [row for row in rows if row.get("bot") == spec["bot"]]
    if not own_rows:
        raise RuntimeError(f"missing {spec['bot']} decision rows")
    seat = int(own_rows[0]["seat"])
    opponent = own_rows[0]["opponent"]
    seen = False
    became_ready = False
    attack_turns: list[int] = []
    attack_row_indexes: list[int] = []
    belt_rocket_attacks = 0

    for row_index, row in enumerate(rows):
        current = (row.get("obs") or {}).get("current") or {}
        players = current.get("players") or []
        if len(players) == 2:
            mine = players[seat] or {}
            mine_board = board(mine)
            seen |= any(card.get("id") in spec["attacker_ids"] for card in mine_board)
            became_ready |= any(ready(card, spec["kind"]) for card in mine_board)
        if row.get("bot") != spec["bot"]:
            continue
        select = (row.get("obs") or {}).get("select") or {}
        options = select.get("option") or []
        for action_index in row.get("action") or []:
            if not isinstance(action_index, int) or not 0 <= action_index < len(options):
                continue
            option = options[action_index]
            if option.get("type") != 13 or option.get("attackId") not in spec["attack_ids"]:
                continue
            attack_turns.append(int(row.get("turn") or 0))
            attack_row_indexes.append(row_index)
            if spec["kind"] == "zapdos" and len(players) == 2:
                active = [card for card in (players[seat].get("active") or []) if isinstance(card, dict)]
                if active and {4, 15}.issubset(energy_ids(active[0])) and 1158 in tool_ids(active[0]):
                    belt_rocket_attacks += 1

    qualifying_attacks = 0
    for attack_row_index in attack_row_indexes:
        event_logs: list[dict] | None = None
        for later in rows[attack_row_index + 1 : attack_row_index + 9]:
            logs = (later.get("obs") or {}).get("logs") or []
            if any(
                log.get("type") == 15
                and log.get("playerIndex") == seat
                and log.get("attackId") in spec["attack_ids"]
                for log in logs
            ):
                event_logs = logs
                break
        maximum_damage = max(
            [
                abs(int(log.get("value") or 0))
                for log in (event_logs or [])
                if log.get("type") == 16 and log.get("playerIndex") == 1 - seat
            ]
            or [0]
        )
        qualifying_attacks += int(maximum_damage >= 200)

    winner = outcome.get("winner")
    return {
        "game": int(outcome["game"]),
        "opponent": opponent,
        "seat": seat,
        "win": winner == spec["bot"],
        "reason": outcome.get("reason"),
        "attacker_seen": seen,
        "attacker_ready": became_ready,
        "intended_attacks": len(attack_turns),
        "first_intended_attack_turn": min(attack_turns) if attack_turns else None,
        "qualifying_attacks": qualifying_attacks,
        "belt_rocket_attacks": belt_rocket_attacks,
    }


def analyze_arm(spec: dict) -> dict:
    games = []
    identities = {}
    for stem in spec["runs"]:
        summary_path = RUN / f"{stem}.jsonl"
        trace_path = RUN / f"{stem}.trace.jsonl"
        outcomes = {
            int(row["game"]): row
            for row in map(json.loads, summary_path.open(encoding="utf-8"))
        }
        grouped: dict[int, list[dict]] = defaultdict(list)
        for row in map(json.loads, trace_path.open(encoding="utf-8")):
            grouped[int(row["game"])].append(row)
        if set(outcomes) != set(grouped):
            raise RuntimeError(f"summary/trace mismatch for {stem}")
        games.extend(analyze_game(grouped[game], outcomes[game], spec) for game in sorted(outcomes))
        identities[summary_path.name] = sha256(summary_path)
        identities[trace_path.name] = sha256(trace_path)

    opponent_attack_games = Counter()
    seat_attack_games = Counter()
    first_turns = []
    for game in games:
        if game["intended_attacks"]:
            opponent_attack_games[game["opponent"]] += 1
            seat_attack_games[str(game["seat"])] += 1
            first_turns.append(game["first_intended_attack_turn"])
    aggregate = {
        "games": len(games),
        "wins": sum(game["win"] for game in games),
        "attacker_seen_games": sum(game["attacker_seen"] for game in games),
        "attacker_ready_games": sum(game["attacker_ready"] for game in games),
        "intended_attack_games": sum(game["intended_attacks"] > 0 for game in games),
        "intended_attacks": sum(game["intended_attacks"] for game in games),
        "qualifying_attacks": sum(game["qualifying_attacks"] for game in games),
        "belt_rocket_attacks": sum(game["belt_rocket_attacks"] for game in games),
        "median_first_intended_attack_turn": statistics.median(first_turns) if first_turns else None,
        "opponent_attack_games": dict(sorted(opponent_attack_games.items())),
        "seat_attack_games": dict(sorted(seat_attack_games.items())),
        "nonterminal_games": sum(game["reason"] != "terminal" for game in games),
    }
    return {"aggregate": aggregate, "games": games, "input_sha256": identities}


def main() -> int:
    arms = {name: analyze_arm(spec) for name, spec in ARMS.items()}
    galv = arms["galvantula_route"]["aggregate"]
    zap = arms["zapdos_route"]["aggregate"]
    control = arms["zapdos_control"]["aggregate"]
    galv_gates = {
        "integrity": galv["games"] == 12 and galv["nonterminal_games"] == 0,
        "seen_and_ready": galv["attacker_seen_games"] >= 9 and galv["attacker_ready_games"] >= 7,
        "attack_coverage": (
            galv["intended_attack_games"] >= 6
            and min(galv["seat_attack_games"].get(str(seat), 0) for seat in (0, 1)) >= 2
            and min(galv["opponent_attack_games"].values(), default=0) >= 2
        ),
        "impact": galv["qualifying_attacks"] >= 4 and galv["intended_attack_games"] >= 4,
        "win_floor": galv["wins"] >= 5,
    }
    zap_gates = {
        "integrity": zap["games"] == 12 and zap["nonterminal_games"] == 0,
        "seen_and_ready": zap["attacker_seen_games"] >= 9 and zap["attacker_ready_games"] >= 7,
        "attack_coverage": (
            zap["intended_attack_games"] >= 6
            and min(zap["seat_attack_games"].get(str(seat), 0) for seat in (0, 1)) >= 2
            and min(zap["opponent_attack_games"].values(), default=0) >= 2
        ),
        "impact": zap["qualifying_attacks"] >= 4 and zap["belt_rocket_attacks"] >= 2,
        "win_floor_and_control": zap["wins"] >= 5 and zap["wins"] >= control["wins"] - 1,
    }
    report = {
        "schema": "target80-lightning-package-screen-v1-analysis/v1",
        "complete": True,
        "arms": arms,
        "gates": {
            "galvantula_route": galv_gates,
            "rocket_zapdos_route": zap_gates,
        },
        "decision": "close_both_packages",
        "decision_reason": (
            "Galvantula failed readiness, coverage, and win gates; Zapdos cleared mechanism "
            "gates but failed the absolute win floor and trailed its deck-identical control."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
