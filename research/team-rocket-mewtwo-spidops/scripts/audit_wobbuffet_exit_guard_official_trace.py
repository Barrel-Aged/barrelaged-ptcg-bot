#!/usr/bin/env python3
"""Replay an official-engine trace through the frozen blueprint and exit guard."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--bot-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location(
        "wobbuffet_exit_guard_official_trace", path / "main.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection_equivalent(observation: dict[str, Any], left: list[int], right: list[int]) -> bool:
    selection = observation.get("select") or {}
    options = selection.get("option") or []
    upper = min(int(selection.get("maxCount") or 0), len(options))
    return upper == len(options) and len(left) == upper and sorted(left) == sorted(right)


def main() -> int:
    args = parse_args()
    candidate = load_candidate(args.candidate.resolve())
    candidate._GUARD_ENABLED = True

    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with args.trace.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("bot") == args.bot_name:
                games[int(row["game"])].append(row)

    totals: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for game, rows in sorted(games.items()):
        blueprint = candidate._fresh_agent()
        for row in sorted(rows, key=lambda item: int(item["step"])):
            observation = row["obs"]
            recorded = [int(index) for index in row.get("action") or []]
            baseline = list(blueprint(observation))
            guarded = candidate._wobbuffet_exit_guard_action(observation, baseline)
            totals["decisions"] += 1
            if guarded == recorded:
                totals["guarded_exact"] += 1
            elif selection_equivalent(observation, guarded, recorded):
                totals["guarded_set_equivalent"] += 1
            else:
                totals["mismatches"] += 1
                mismatches.append(
                    {
                        "game": game,
                        "step": row["step"],
                        "turn": row.get("turn"),
                        "recorded": recorded,
                        "baseline": baseline,
                        "guarded": guarded,
                    }
                )
            if guarded == baseline:
                totals["unchanged"] += 1
                continue
            selection = observation.get("select") or {}
            options = selection.get("option") or []
            totals["changed"] += 1
            changes.append(
                {
                    "game": game,
                    "step": row["step"],
                    "turn": row.get("turn"),
                    "recorded": recorded,
                    "baseline": baseline,
                    "guarded": guarded,
                    "source_option": options[baseline[0]],
                    "target_option": options[guarded[0]],
                }
            )
        totals["games"] += 1

    payload = {
        "schema": "wobbuffet-exit-guard-official-trace-audit/v1",
        "candidate": str(args.candidate.resolve()),
        "trace": str(args.trace.resolve()),
        "bot_name": args.bot_name,
        "totals": dict(totals),
        "changes": changes,
        "mismatches": mismatches[:20],
        "evidence_boundary": (
            "This reproduces the candidate's actions on official-engine observations and "
            "locates guard deltas. Match outcomes remain observational rather than a paired "
            "counterfactual estimate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
