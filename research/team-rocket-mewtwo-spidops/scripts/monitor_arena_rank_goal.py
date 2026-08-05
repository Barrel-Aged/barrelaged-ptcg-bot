#!/usr/bin/env python3
"""Read-only monitor for immutable Team Rocket 300-game rank-one candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ARENA_LAB = Path(r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab")
ARENA_CLI = ARENA_LAB / "tools/arena.py"
OUTPUT = ROOT / "work/arena_rank_goal_monitor"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
TARGETS = (
    {
        "name": "HelbrosRocketSpidopsScaleBCV2",
        "bundle": ROOT
        / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2",
        "model_sha256": "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b",
    },
    {
        "name": "HelbrosRocketSpidopsScaleStopV3",
        "bundle": ROOT / "work/scale_stop_archive_smoke_v3",
        "model_sha256": "c6b808e3e50b5fffcfd44b59ad2a541c6661e8d22915b1cbb5ade5a57464d8f2",
    },
)
TARGET_GAMES = 300
ROW = re.compile(
    r"^\s*(?P<rank>\d+)\s+(?P<name>.+?)\s+"
    r"(?P<rating>-?\d+(?:\.\d+)?)\s+"
    r"\((?P<w>\d+)-(?P<l>\d+)-(?P<d>\d+)\)\s*$"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratings() -> list[dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(ARENA_CLI), "ratings"],
        cwd=ARENA_LAB,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    parsed = []
    for line in completed.stdout.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        values = match.groupdict()
        wins = int(values["w"])
        losses = int(values["l"])
        draws = int(values["d"])
        parsed.append(
            {
                "rank": int(values["rank"]),
                "name": values["name"].strip(),
                "rating": float(values["rating"]),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": wins + losses + draws,
            }
        )
    if not parsed:
        raise RuntimeError("arena ratings response contained no parsable rows")
    return parsed


def snapshot() -> dict[str, object]:
    verified_targets = []
    for target in TARGETS:
        bundle = Path(target["bundle"])
        if digest(bundle / "deck.csv") != DECK_SHA:
            raise ValueError(f"monitored Team Rocket deck changed: {target['name']}")
        if digest(bundle / "model.pt") != target["model_sha256"]:
            raise ValueError(f"monitored model changed: {target['name']}")
        verified_targets.append(
            {
                "name": target["name"],
                "bundle": str(bundle.resolve()),
                "deck_sha256": DECK_SHA,
                "model_sha256": target["model_sha256"],
            }
        )
    table = ratings()
    rows_by_name = {str(row["name"]): row for row in table}
    bots = []
    for target in verified_targets:
        row = rows_by_name.get(str(target["name"]))
        bots.append({**target, "rating_row": row, "present": row is not None})
    present = [item for item in bots if item["rating_row"] is not None]
    present.sort(key=lambda item: int(item["rating_row"]["rank"]))
    focal = present[0]["rating_row"] if present else None
    goal_bot = next(
        (
            item
            for item in present
            if int(item["rating_row"]["games"]) >= TARGET_GAMES
            and int(item["rating_row"]["rank"]) == 1
        ),
        None,
    )
    return {
        "schema": "rocket-arena-rank-goal-monitor/v2",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "bot": focal,
        "bots": bots,
        "bot_present": bool(present),
        "goal_bot": goal_bot,
        "target_games": TARGET_GAMES,
        "goal_met": goal_bot is not None,
        "qualifying_deck_sha256": DECK_SHA,
        "top": table[:10],
        "read_only": True,
    }


def write_snapshot(value: dict[str, object]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status = OUTPUT / "status.json"
    temporary = OUTPUT / "status.json.tmp"
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(status)
    with (OUTPUT / "history.jsonl").open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 60:
        raise ValueError("monitor interval must be at least 60 seconds")
    consecutive_errors = 0
    while True:
        try:
            value = snapshot()
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            value = {
                "schema": "rocket-arena-rank-goal-monitor/v2",
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "bot": None,
                "bot_present": False,
                "target_games": TARGET_GAMES,
                "goal_met": False,
                "qualifying_deck_sha256": DECK_SHA,
                "targets": [
                    {
                        "name": target["name"],
                        "model_sha256": target["model_sha256"],
                    }
                    for target in TARGETS
                ],
                "read_only": True,
                "observation_error": f"{type(exc).__name__}: {exc}",
                "consecutive_errors": consecutive_errors,
            }
        write_snapshot(value)
        print(json.dumps(value, sort_keys=True), flush=True)
        if args.once:
            return 0 if "observation_error" not in value else 2
        if value["goal_met"]:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
