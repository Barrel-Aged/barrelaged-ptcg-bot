#!/usr/bin/env python3
"""Audit V13 against both frozen blueprints on identical official observations."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ADAPTED_IDS = frozenset({119, 120, 121, 646, 647, 648, 1030, 1031})


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("ptcg_local_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_route(obs: dict[str, Any], seat: int) -> bool:
    players = (obs.get("current") or {}).get("players") or []
    if len(players) != 2:
        return False
    theirs = players[1 - seat] or {}
    cards = (
        (theirs.get("active") or [])
        + (theirs.get("bench") or [])
        + (theirs.get("discard") or [])
    )
    return any(
        isinstance(card, dict) and int(card.get("id") or 0) in ADAPTED_IDS
        for card in cards
    )


def legal(action: list[int], obs: dict[str, Any]) -> bool:
    selection = obs.get("select") or {}
    options = selection.get("option") or []
    lower = int(selection.get("minCount") or 0)
    upper = min(int(selection.get("maxCount") or 0), len(options))
    return (
        len(action) == len(set(action))
        and lower <= len(action) <= upper
        and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--safe", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--bot", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evaluator = load_evaluator(args.evaluator.resolve())
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    log_dir = output.with_suffix(".logs")
    log_dir.mkdir(parents=True, exist_ok=False)
    paths = [
        path
        for path in sorted(args.replays.resolve().glob("*.jsonl"))
        if args.bot in path.name and "Soldier" in path.name
    ][: args.limit]
    if not paths:
        raise ValueError("no matching official replays")

    workers = [
        evaluator.AgentWorker("router", args.router.resolve(), {}, log_dir),
        evaluator.AgentWorker("safe", args.safe.resolve(), {}, log_dir),
        evaluator.AgentWorker("adapted", args.adapted.resolve(), {}, log_dir),
    ]
    games = decisions = legal_count = safe_count = adapted_count = mismatches = 0
    mismatch_rows = []
    try:
        for path in paths:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            players = list(records[0]["meta"]["players"])
            seat = players.index(args.bot)
            decks = [worker.act({"select": None}, 30.0) for worker in workers]
            if not (decks[0] == decks[1] == decks[2] and len(decks[0]) == 60):
                raise ValueError("blueprint decks differ")
            latched = False
            for row in records[1:-1]:
                if int(row.get("p", -1)) != seat or "obs" not in row:
                    continue
                obs = row["obs"]
                router_action, safe_action, adapted_action = [
                    worker.act(obs, 30.0) for worker in workers
                ]
                latched = latched or public_route(obs, seat)
                expected = adapted_action if latched else safe_action
                decisions += 1
                legal_count += int(legal(router_action, obs))
                safe_count += int(not latched)
                adapted_count += int(latched)
                if router_action != expected:
                    mismatches += 1
                    if len(mismatch_rows) < 20:
                        mismatch_rows.append(
                            {
                                "file": path.name,
                                "decision": decisions,
                                "latched": latched,
                                "router": router_action,
                                "expected": expected,
                            }
                        )
            games += 1
    finally:
        for worker in workers:
            worker.close()

    report = {
        "schema": "public-range-safe-router-v13-parity/v1",
        "official_observations": True,
        "games": games,
        "decisions": decisions,
        "legal": legal_count,
        "safe_route_decisions": safe_count,
        "adapted_route_decisions": adapted_count,
        "mismatches": mismatches,
        "mismatch_rows": mismatch_rows,
        "passes": decisions >= 500 and legal_count == decisions and mismatches == 0,
        "paths": [path.name for path in paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
