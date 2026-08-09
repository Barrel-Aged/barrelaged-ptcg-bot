#!/usr/bin/env python3
"""Audit ScaleBCv2 Erasure Ball discard decisions in Kaggle replays.

The audit reads only observations available to the acting agent.  It groups
successive context-26 prompts in the same turn as one Erasure Ball resolution,
computes the minimum number of 60-damage discards needed for the current Active
Pokemon, and records which Benched Pokemon each chosen Energy came from.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any


MEWTWO = 431
SPIDOPS = 401
ERASURE_BALL = 608
DISCARD_CONTEXT = 26
PSYCHIC = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _target_damage(card: Any, discarded: int) -> int:
    damage = 160 + 60 * discarded
    if int(getattr(card, "weakness", 0) or 0) == PSYCHIC:
        damage *= 2
    if int(getattr(card, "resistance", 0) or 0) == PSYCHIC:
        damage -= 30
    return max(0, damage)


def _minimum_discards(card: Any, hp: int) -> int:
    for count in range(3):
        if _target_damage(card, count) >= hp:
            return count
    return 2


def _card_id(card: Any) -> int:
    return int(card.get("id") or 0) if isinstance(card, dict) else 0


def audit(replays: Path, package: Path) -> dict[str, Any]:
    sys.path.insert(0, str(package.resolve()))
    from soldier.cards import CARDS  # type: ignore[import-not-found]

    rows: list[dict[str, Any]] = []
    games = sorted(replays.resolve().glob("episode-*-replay.json"))
    for path in games:
        replay = json.loads(path.read_text(encoding="utf-8"))
        episode = int(
            ((replay.get("info") or {}).get("EpisodeId"))
            or path.stem.split("-")[1]
        )
        discarded_by_turn: Counter[tuple[int, int, int]] = Counter()
        for step_index, pair in enumerate(replay.get("steps") or []):
            for seat, entry in enumerate(pair or []):
                observation = entry.get("observation") or {}
                selection = observation.get("select") or {}
                effect = selection.get("effect") or {}
                if (
                    int(selection.get("context") or -1) != DISCARD_CONTEXT
                    or int(effect.get("id") or 0) != MEWTWO
                ):
                    continue
                current = observation.get("current") or {}
                your_index = current.get("yourIndex")
                players = current.get("players") or []
                if your_index not in (0, 1) or len(players) != 2:
                    continue
                mine = players[your_index] or {}
                theirs = players[1 - your_index] or {}
                active = (mine.get("active") or [{}])[0] or {}
                target = (theirs.get("active") or [{}])[0] or {}
                if _card_id(active) != MEWTWO or _card_id(target) not in CARDS:
                    continue
                turn = int(current.get("turn") or 0)
                chain_key = (
                    turn,
                    int(active.get("serial") or 0),
                    int(target.get("serial") or 0),
                )
                already = discarded_by_turn[chain_key]
                total_required = _minimum_discards(
                    CARDS[_card_id(target)], int(target.get("hp") or 0)
                )
                remaining_required = max(0, total_required - already)
                options = selection.get("option") or []
                action = [
                    int(index)
                    for index in (entry.get("action") or [])
                    if isinstance(index, int) and 0 <= index < len(options)
                ]
                chosen_sources = []
                for index in action:
                    option = options[index]
                    bench_index = int(option.get("index") or 0)
                    bench = mine.get("bench") or []
                    source = bench[bench_index] if 0 <= bench_index < len(bench) else {}
                    energy_cards = source.get("energyCards") or [] if isinstance(source, dict) else []
                    chosen_sources.append(
                        {
                            "option_index": index,
                            "bench_index": bench_index,
                            "pokemon_id": _card_id(source),
                            "energy_cards_before": len(energy_cards),
                            "expanded_energy_before": len(source.get("energies") or [])
                            if isinstance(source, dict)
                            else 0,
                        }
                    )
                discarded_by_turn[chain_key] += len(action)
                rows.append(
                    {
                        "episode_id": episode,
                        "step": step_index,
                        "seat": seat,
                        "turn": turn,
                        "target_id": _card_id(target),
                        "target_hp": int(target.get("hp") or 0),
                        "base_damage": _target_damage(CARDS[_card_id(target)], 0),
                        "total_required_discards": total_required,
                        "already_discarded": already,
                        "remaining_required": remaining_required,
                        "action": action,
                        "chosen_sources": chosen_sources,
                        "redundant_discard": bool(action and remaining_required == 0),
                        "chosen_spidops_energy": any(
                            source["pokemon_id"] == SPIDOPS for source in chosen_sources
                        ),
                    }
                )

    return {
        "schema": "kaggle-scalebcv2-erasure-ball-audit/v1",
        "replay_games": len(games),
        "discard_prompts": len(rows),
        "energy_cards_discarded": sum(len(row["action"]) for row in rows),
        "redundant_discard_prompts": sum(row["redundant_discard"] for row in rows),
        "spidops_energy_discard_prompts": sum(row["chosen_spidops_energy"] for row in rows),
        "rows": rows,
        "claim_boundary": (
            "Replay-observed action audit only; minimum discard is computed for "
            "the immediate Active knockout and is not a terminal outcome estimate."
        ),
    }


def main() -> int:
    args = parse_args()
    result = audit(args.replays, args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
