#!/usr/bin/env python3
"""Build an exact-deck, focal-winner behavior-cloning corpus.

Public Kaggle replays contain both players.  A generic "train on the winner"
pipeline is unsafe when the focal bot loses because it silently imitates the
opponent under the focal deck prior.  This utility admits only games where one
named team used one exact ordered deck and won, then hard-links those raw
replays into the filename convention expected by the V110 trainer.

Deck lists and terminal outcomes are used retrospectively for corpus
integrity/filtering only; no hidden information is introduced into features.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--deck-sha256", required=True)
    parser.add_argument("--require-min-games", type=int, default=3)
    parser.add_argument("--require-min-wins", type=int, default=1)
    return parser.parse_args()


def canonical_deck_bytes(deck: Iterable[int]) -> bytes:
    return "".join(f"{int(card_id)}\n" for card_id in deck).encode("utf-8")


def digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    raise ValueError("replay has no two-deck declaration")


def first_player(data: dict[str, Any]) -> int | None:
    for frame in iter_frames(data):
        value = (frame.get("current") or {}).get("firstPlayer")
        if value in (0, 1):
            return int(value)
    return None


def team_names(data: dict[str, Any]) -> list[str]:
    names = (data.get("info") or {}).get("TeamNames")
    if isinstance(names, list) and len(names) >= 2:
        return [str(names[0]), str(names[1])]
    agents = (data.get("info") or {}).get("Agents") or []
    if len(agents) >= 2:
        return [str((agents[i] or {}).get("Name") or f"player_{i}") for i in range(2)]
    raise ValueError("replay has no two team names")


def result(rewards: Any, seat: int) -> str:
    if not isinstance(rewards, list) or len(rewards) < 2:
        return "unknown"
    if rewards[seat] == 1 and rewards[1 - seat] == -1:
        return "win"
    if rewards[seat] == -1 and rewards[1 - seat] == 1:
        return "loss"
    if rewards[seat] == rewards[1 - seat]:
        return "draw"
    return "unknown"


def link_or_verify(source: Path, target: Path) -> str:
    if target.exists():
        if source.stat().st_size != target.stat().st_size:
            raise ValueError(f"existing corpus file size mismatch: {target}")
        if digest_file(source) != digest_file(target):
            raise ValueError(f"existing corpus file hash mismatch: {target}")
        return "verified_existing"
    try:
        os.link(source, target)
        return "hard_link"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def main() -> int:
    args = parse_args()
    sources = sorted(args.replays.resolve().glob("episode-*-replay.json"))
    if len(sources) < args.require_min_games:
        raise SystemExit(
            f"need at least {args.require_min_games} replay files, found {len(sources)}"
        )
    if len(args.deck_sha256) != 64:
        raise ValueError("--deck-sha256 must be a 64-character digest")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    observed_decks: Counter[str] = Counter()
    seen_episode_ids: set[int] = set()
    for path in sources:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            episode_id = int((data.get("info") or {}).get("EpisodeId"))
            if episode_id in seen_episode_ids:
                raise ValueError(f"duplicate episode id {episode_id}")
            seen_episode_ids.add(episode_id)
            names = team_names(data)
            decks = initial_decks(data)
            deck_hashes = [
                sha256(canonical_deck_bytes(deck)).hexdigest() for deck in decks
            ]
            for digest in deck_hashes:
                observed_decks[digest] += 1
            matching = [
                seat
                for seat in (0, 1)
                if names[seat] == args.team
                and deck_hashes[seat] == args.deck_sha256
            ]
            if len(matching) != 1:
                raise ValueError(
                    "expected exactly one focal team/deck seat; "
                    f"names={names!r}, hashes={deck_hashes!r}"
                )
            seat = matching[0]
            outcome = result(data.get("rewards"), seat)
            if outcome == "unknown":
                raise ValueError(f"unsupported rewards {data.get('rewards')!r}")
            starter = first_player(data)
            rows.append(
                {
                    "episode_id": episode_id,
                    "source": str(path),
                    "source_sha256": digest_file(path),
                    "seat": seat,
                    "opponent": names[1 - seat],
                    "result": outcome,
                    "first_player": starter,
                    "went_first": starter == seat if starter in (0, 1) else None,
                    "steps": len(data.get("steps") or []),
                    "deck_sha256": deck_hashes[seat],
                }
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise SystemExit("corpus integrity failed:\n" + "\n".join(errors))
    wins = [row for row in rows if row["result"] == "win"]
    if len(wins) < args.require_min_wins:
        raise SystemExit(
            f"need at least {args.require_min_wins} focal wins, found {len(wins)}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    link_methods: Counter[str] = Counter()
    for row in wins:
        source = Path(row["source"])
        target = args.output / f"episode_{row['episode_id']}.json"
        method = link_or_verify(source, target)
        link_methods[method] += 1
        row["corpus_file"] = target.name
        row["link_method"] = method

    result_counts = Counter(row["result"] for row in rows)
    seat_counts = Counter(str(row["seat"]) for row in rows)
    order_counts = Counter(
        "first" if row["went_first"] else "second"
        for row in rows
        if row["went_first"] is not None
    )
    payload = {
        "schema": "target80-deck-matched-public-bc-corpus/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(args.replays.resolve()),
        "output_directory": str(args.output.resolve()),
        "focal_team": args.team,
        "focal_deck_sequence_sha256": args.deck_sha256,
        "source_games": len(rows),
        "source_record": dict(sorted(result_counts.items())),
        "source_win_rate": len(wins) / len(rows),
        "source_seats": dict(sorted(seat_counts.items())),
        "source_turn_order": dict(sorted(order_counts.items())),
        "admitted_winning_games": len(wins),
        "link_methods": dict(sorted(link_methods.items())),
        "observed_deck_seat_counts": dict(sorted(observed_decks.items())),
        "integrity": {
            "unique_episode_ids": len(seen_episode_ids),
            "errors": 0,
            "all_focal_decks_exact": all(
                row["deck_sha256"] == args.deck_sha256 for row in rows
            ),
            "all_admitted_games_are_focal_wins": all(
                row["result"] == "win" for row in wins
            ),
        },
        "games": rows,
        "claim_boundary": (
            "Retrospective exact-deck and winner filtering only. This manifest "
            "does not establish policy strength or authorize deployment."
        ),
    }
    manifest = args.output / "corpus_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_games": len(rows),
                "record": dict(sorted(result_counts.items())),
                "win_rate": payload["source_win_rate"],
                "admitted_wins": len(wins),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
