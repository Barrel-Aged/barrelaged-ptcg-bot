#!/usr/bin/env python3
"""Inventory Team Rocket Mewtwo/Spidops decks in public replay summaries.

This is a read-only scouting utility.  It validates every downloaded summary,
then records exact deck composition, outcome, seat, opponent, and ordered deck
fingerprints for any replay containing the Mewtwo/Spidops core.  Ordered hashes
are extracted from the raw replay only for matching episodes, keeping a large
daily scan reasonably cheap while preventing accidental policy/deck mixing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
MEWTWO = 431
WOBBUFFET = 432
MIMIKYU = 434
MURKROW = 463
HONCHKROW = 891
ROCKET_CORE = {TAROUNTULA, SPIDOPS, MEWTWO}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def canonical_sequence_bytes(deck: Iterable[int]) -> bytes:
    return "".join(f"{int(card_id)}\n" for card_id in deck).encode("utf-8")


def canonical_multiset_bytes(counts: dict[int, int]) -> bytes:
    return "".join(
        f"{card_id}:{counts[card_id]}\n" for card_id in sorted(counts)
    ).encode("utf-8")


def iter_frames(data: dict[str, Any]):
    for step in data.get("steps") or []:
        entries = step if isinstance(step, list) else [step]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for frame in entry.get("visualize") or []:
                if isinstance(frame, dict):
                    yield frame


def initial_decks(data: dict[str, Any]) -> list[list[int]]:
    for frame in iter_frames(data):
        action = frame.get("action")
        if (
            isinstance(action, list)
            and len(action) == 2
            and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
        ):
            return [[int(card_id) for card_id in deck] for deck in action]
    raise ValueError("raw replay has no two-deck declaration")


def card_counts(player: dict[str, Any]) -> tuple[dict[int, int], dict[int, str]]:
    counts: dict[int, int] = {}
    names: dict[int, str] = {}
    for card in ((player.get("deck") or {}).get("cards") or []):
        card_id = int(card["card_id"])
        count = int(card["count"])
        if count <= 0 or card_id in counts:
            raise ValueError(f"invalid or duplicate card entry {card_id}")
        counts[card_id] = count
        names[card_id] = str(card.get("name") or card_id)
    return counts, names


def count(counts: dict[int, int], card_id: int) -> int:
    return int(counts.get(card_id, 0))


def result_label(reward: Any) -> str:
    if reward == 1:
        return "win"
    if reward == -1:
        return "loss"
    return "draw"


def summarize_deck(cards: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{card['count']}x {card['name']} ({card['card_id']})" for card in cards
    )


def main() -> int:
    args = parse_args()
    summaries = sorted(args.episodes.glob("*.summary.json"))
    if not summaries:
        raise SystemExit(f"no *.summary.json files under {args.episodes}")

    errors: list[str] = []
    seen_episode_ids: set[int] = set()
    rocket_rows: list[dict[str, Any]] = []
    validated_players = 0

    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            episode_id = int(summary["episode_id"])
            if episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode id {episode_id}")
            seen_episode_ids.add(episode_id)
            players = summary.get("players") or []
            if len(players) != 2:
                raise ValueError(f"expected two players, got {len(players)}")
            first_player = summary.get("first_player")
            if first_player not in (0, 1, None):
                raise ValueError(f"invalid first_player {first_player!r}")

            parsed_players: list[tuple[dict[int, int], dict[int, str]]] = []
            for player in players:
                counts, names = card_counts(player)
                if sum(counts.values()) != 60:
                    raise ValueError(
                        f"player {player.get('index')} has {sum(counts.values())} cards"
                    )
                parsed_players.append((counts, names))
                validated_players += 1

            source_value = summary.get("source")
            raw_path = Path(source_value) if source_value else args.episodes / f"{episode_id}.json"
            if not raw_path.is_absolute():
                raw_path = args.episodes / raw_path
            matching_seats = [
                seat
                for seat, (counts, _) in enumerate(parsed_players)
                if ROCKET_CORE.issubset(counts)
            ]
            if matching_seats and not raw_path.is_file():
                raise ValueError(f"missing raw replay {raw_path.name}")
            raw_decks = (
                initial_decks(json.loads(raw_path.read_text(encoding="utf-8")))
                if matching_seats
                else None
            )

            for seat in matching_seats:
                player = players[seat]
                opponent = players[1 - seat]
                counts, names = parsed_players[seat]
                deck = raw_decks[seat] if raw_decks is not None else []
                if Counter(deck) != Counter(counts):
                    raise ValueError(
                        f"raw/summary deck mismatch for episode {episode_id} seat {seat}"
                    )
                cards = [
                    {
                        "card_id": card_id,
                        "count": counts[card_id],
                        "name": names[card_id],
                    }
                    for card_id in sorted(counts)
                ]
                reward = player.get("reward")
                rocket_rows.append(
                    {
                        "episode_id": episode_id,
                        "team": str(player.get("team") or f"player_{seat}"),
                        "seat": seat,
                        "reward": reward,
                        "result": result_label(reward),
                        "first_player": first_player,
                        "went_first": first_player == seat if first_player in (0, 1) else None,
                        "opponent": str(opponent.get("team") or f"player_{1 - seat}"),
                        "opponent_archetype": opponent.get("archetype"),
                        "steps": int(summary.get("steps") or 0),
                        "deck_sequence_sha256": sha256(
                            canonical_sequence_bytes(deck)
                        ).hexdigest(),
                        "deck_multiset_sha256": sha256(
                            canonical_multiset_bytes(counts)
                        ).hexdigest(),
                        "wobbuffet_free": count(counts, WOBBUFFET) == 0,
                        "key_counts": {
                            "tarountula": count(counts, TAROUNTULA),
                            "spidops": count(counts, SPIDOPS),
                            "articuno": count(counts, ARTICUNO),
                            "mewtwo": count(counts, MEWTWO),
                            "wobbuffet": count(counts, WOBBUFFET),
                            "mimikyu": count(counts, MIMIKYU),
                            "murkrow": count(counts, MURKROW),
                            "honchkrow": count(counts, HONCHKROW),
                        },
                        "cards": cards,
                    }
                )
        except Exception as exc:  # keep a complete audit trail before failing
            errors.append(f"{summary_path.name}: {exc}")

    if errors:
        raise SystemExit("summary validation failed:\n" + "\n".join(errors))

    by_deck: dict[str, dict[str, Any]] = {}
    for row in rocket_rows:
        digest = row["deck_sequence_sha256"]
        aggregate = by_deck.setdefault(
            digest,
            {
                "deck_sequence_sha256": digest,
                "deck_multiset_sha256": row["deck_multiset_sha256"],
                "wobbuffet_free": row["wobbuffet_free"],
                "key_counts": row["key_counts"],
                "cards": row["cards"],
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "went_first_games": 0,
                "went_second_games": 0,
                "teams": defaultdict(int),
                "opponents": defaultdict(int),
                "episode_ids": [],
            },
        )
        aggregate["games"] += 1
        result_bucket = {"win": "wins", "loss": "losses", "draw": "draws"}[
            row["result"]
        ]
        aggregate[result_bucket] += 1
        if row["went_first"] is True:
            aggregate["went_first_games"] += 1
        elif row["went_first"] is False:
            aggregate["went_second_games"] += 1
        aggregate["teams"][row["team"]] += 1
        aggregate["opponents"][row["opponent"]] += 1
        aggregate["episode_ids"].append(row["episode_id"])

    deck_rows = []
    for aggregate in by_deck.values():
        aggregate["teams"] = dict(sorted(aggregate["teams"].items()))
        aggregate["opponents"] = dict(sorted(aggregate["opponents"].items()))
        aggregate["episode_ids"].sort()
        aggregate["win_rate"] = (
            aggregate["wins"] / aggregate["games"] if aggregate["games"] else None
        )
        deck_rows.append(aggregate)
    deck_rows.sort(key=lambda item: (-item["games"], item["deck_sequence_sha256"]))
    rocket_rows.sort(key=lambda item: item["episode_id"])

    payload = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(args.episodes.resolve()),
        "integrity": {
            "summary_files": len(summaries),
            "unique_episode_ids": len(seen_episode_ids),
            "validated_players": validated_players,
            "validation_errors": 0,
        },
        "rocket_definition": {
            "required_card_ids": sorted(ROCKET_CORE),
            "description": "contains Team Rocket's Tarountula, Spidops, and Mewtwo ex",
        },
        "rocket_appearances": len(rocket_rows),
        "wobbuffet_free_appearances": sum(
            int(row["wobbuffet_free"]) for row in rocket_rows
        ),
        "decks": deck_rows,
        "episodes": rocket_rows,
        "interpretation_guard": (
            "This is a top-episode occurrence audit, not an unbiased win-rate sample. "
            "Do not infer deck or policy strength from these rates alone."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Wobbuffet-free public Rocket replay scout",
        "",
        f"- Validated `{len(summaries)}` episode summaries / `{validated_players}` player decks with zero errors.",
        f"- Found `{len(rocket_rows)}` Team Rocket Mewtwo/Spidops appearances across `{len(deck_rows)}` exact ordered deck identities.",
        f"- Wobbuffet-free appearances: `{payload['wobbuffet_free_appearances']}`.",
        "- Sampling warning: these are selected top episodes, not an unbiased performance panel.",
        "",
        "## Exact deck identities",
        "",
    ]
    for index, deck_row in enumerate(deck_rows, start=1):
        teams = ", ".join(
            f"{team} ({games})" for team, games in deck_row["teams"].items()
        )
        lines.extend(
            [
                f"### Deck {index}",
                "",
                f"- Ordered SHA-256: `{deck_row['deck_sequence_sha256']}`",
                f"- Multiset SHA-256: `{deck_row['deck_multiset_sha256']}`",
                f"- Wobbuffet-free: `{deck_row['wobbuffet_free']}`",
                f"- Observed record: `{deck_row['wins']}-{deck_row['losses']}-{deck_row['draws']}` in `{deck_row['games']}` selected episodes",
                f"- Teams: {teams}",
                f"- Episode IDs: {', '.join(str(value) for value in deck_row['episode_ids'])}",
                f"- Cards: {summarize_deck(deck_row['cards'])}",
                "",
            ]
        )
    lines.extend(["## Episode inventory", ""])
    for row in rocket_rows:
        order = "first" if row["went_first"] else "second"
        lines.append(
            f"- `{row['episode_id']}` — {row['team']} {row['result']} as seat "
            f"{row['seat']} going {order} vs {row['opponent']} "
            f"({row['opponent_archetype']})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation_guard"],
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["integrity"] | {
        "rocket_appearances": payload["rocket_appearances"],
        "wobbuffet_free_appearances": payload["wobbuffet_free_appearances"],
        "exact_decks": len(deck_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
