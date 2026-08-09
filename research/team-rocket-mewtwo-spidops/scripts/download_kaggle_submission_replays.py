#!/usr/bin/env python3
"""Download a deterministic, evenly spaced sample of public Kaggle replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def evenly_spaced(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    episodes = api.competition_list_episodes(args.submission)
    public = [
        episode
        for episode in episodes
        if str(episode.type).endswith("PUBLIC") and str(episode.state).endswith("COMPLETED")
    ]
    public.sort(key=lambda episode: int(episode.id))
    selected = evenly_spaced(public, args.limit)

    rows = []
    for episode in selected:
        episode_id = int(episode.id)
        replay_path = output / f"episode-{episode_id}-replay.json"
        if not replay_path.exists():
            api.competition_episode_replay(episode_id, path=str(output), quiet=True)
        agents = []
        for agent in episode.agents or []:
            agents.append(
                {
                    "submission_id": int(agent.submission_id),
                    "index": int(agent.index),
                    "reward": float(agent.reward),
                    "team_name": str(agent.team_name),
                    "team_id": int(agent.team_id),
                }
            )
        rows.append(
            {
                "episode_id": episode_id,
                "create_time": str(episode.create_time),
                "replay": replay_path.name,
                "sha256": sha256(replay_path),
                "agents": agents,
            }
        )

    payload = {
        "schema": "kaggle-public-replay-sample/v1",
        "submission_id": args.submission,
        "available_public_episodes": len(public),
        "selection": "evenly-spaced-by-episode-id",
        "sample_size": len(rows),
        "episodes": rows,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "submission_id": args.submission,
                "available_public_episodes": len(public),
                "downloaded_sample": len(rows),
                "manifest": str(manifest),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

