#!/usr/bin/env python3
"""Audit one public Kaggle submission's replay set and exact deck identity."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_deck_sha256(deck: Iterable[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(card)) for card in deck).encode("ascii")
    ).hexdigest()


def iter_frames(payload: dict[str, Any]):
    for step in payload.get("steps") or ():
        entries = step if isinstance(step, list) else (step,)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for frame in entry.get("visualize") or ():
                if isinstance(frame, dict):
                    yield frame


def initial_decks(payload: dict[str, Any]) -> list[list[int]]:
    for frame in iter_frames(payload):
        action = frame.get("action")
        if (
            isinstance(action, list)
            and len(action) == 2
            and all(isinstance(deck, list) and len(deck) == 60 for deck in action)
        ):
            return [[int(card) for card in deck] for deck in action]
    raise ValueError("replay contains no two-deck declaration")


def first_player(payload: dict[str, Any]) -> int:
    for frame in iter_frames(payload):
        current = frame.get("current") or {}
        value = current.get("firstPlayer")
        if value in (0, 1):
            return int(value)
    raise ValueError("replay contains no first-player marker")


def main() -> int:
    args = parse_args()
    paths = sorted(args.replays.resolve().glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no JSON replays under {args.replays}")
    rows: list[dict[str, Any]] = []
    deck_hashes: Counter[str] = Counter()
    deck_by_hash: dict[str, list[int]] = {}
    outcomes: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    starts: Counter[str] = Counter()
    seen_episode_ids: set[int] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        info = payload.get("info") or {}
        episode_id = int(info.get("EpisodeId") or 0)
        if not episode_id or episode_id in seen_episode_ids:
            raise ValueError(f"invalid or duplicate episode id in {path.name}")
        seen_episode_ids.add(episode_id)
        teams = list(info.get("TeamNames") or ())
        if len(teams) != 2 or teams.count(args.team) != 1:
            raise ValueError(f"team identity mismatch in {path.name}: {teams}")
        if list(payload.get("statuses") or ()) != ["DONE", "DONE"]:
            raise ValueError(f"nonterminal replay {path.name}")
        rewards = list(payload.get("rewards") or ())
        if len(rewards) != 2 or any(value not in (-1, 0, 1) for value in rewards):
            raise ValueError(f"invalid rewards in {path.name}: {rewards}")
        seat = teams.index(args.team)
        opponent = str(teams[1 - seat])
        deck = initial_decks(payload)[seat]
        deck_hash = canonical_deck_sha256(deck)
        deck_hashes[deck_hash] += 1
        deck_by_hash.setdefault(deck_hash, deck)
        first = first_player(payload)
        result = "win" if rewards[seat] == 1 else "loss" if rewards[seat] == -1 else "draw"
        outcomes[result] += 1
        opponents[opponent] += 1
        starts["first" if first == seat else "second"] += 1
        rows.append(
            {
                "episode_id": episode_id,
                "file": path.name,
                "file_sha256": sha256(path),
                "seat": seat,
                "first_player": first,
                "started_first": first == seat,
                "result": result,
                "opponent": opponent,
                "deck_sha256": deck_hash,
            }
        )
    if len(deck_hashes) != 1:
        raise ValueError(f"submission used {len(deck_hashes)} deck identities")
    deck_hash = next(iter(deck_hashes))
    deck = deck_by_hash[deck_hash]
    tree = hashlib.sha256()
    for row in rows:
        tree.update(str(row["episode_id"]).encode("ascii"))
        tree.update(b"\0")
        tree.update(str(row["file_sha256"]).encode("ascii"))
        tree.update(b"\0")
    output = {
        "schema": "target80-public-submission-audit/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "public Kaggle competition replays",
        "team": args.team,
        "submission_id": args.submission_id,
        "replay_directory": str(args.replays.resolve()),
        "games": len(rows),
        "outcomes": dict(outcomes),
        "starts": dict(starts),
        "opponents": dict(sorted(opponents.items())),
        "exact_deck_identity_count": len(deck_hashes),
        "deck_sha256": deck_hash,
        "deck": deck,
        "wobbuffet_count": deck.count(432),
        "replay_tree_sha256": tree.hexdigest(),
        "rows": rows,
        "interpretation_guard": (
            "Public submission record and policy demonstrations; not an unbiased "
            "estimate for a different policy or the internal arena."
        ),
    }
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "games": len(rows),
                "outcomes": dict(outcomes),
                "starts": dict(starts),
                "deck_sha256": deck_hash,
                "wobbuffet_count": deck.count(432),
                "replay_tree_sha256": tree.hexdigest(),
                "output": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
