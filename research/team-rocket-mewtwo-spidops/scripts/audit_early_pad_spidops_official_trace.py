#!/usr/bin/env python3
"""Replay an early-Pad-to-Spidops candidate trace and locate its action deltas."""
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
        "early_pad_spidops_official_trace", path / "main.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equivalent(observation: dict[str, Any], left: list[int], right: list[int]) -> bool:
    selection = observation.get("select") or {}
    options = selection.get("option") or []
    upper = min(int(selection.get("maxCount") or 0), len(options))
    return upper == len(options) and len(left) == upper and sorted(left) == sorted(right)


def main() -> int:
    args = parse_args()
    candidate = load_candidate(args.candidate.resolve())
    games: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with args.trace.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("bot") == args.bot_name:
                games[int(row["game"])].append(row)

    totals: Counter[str] = Counter()
    direct_overrides: list[dict[str, Any]] = []
    downstream_deltas: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    per_game: dict[str, dict[str, int]] = {}
    for game, rows in sorted(games.items()):
        pure_blueprint = candidate._fresh_agent()
        guarded_blueprint = candidate._fresh_agent()
        game_counts: Counter[str] = Counter()
        for row in sorted(rows, key=lambda item: int(item["step"])):
            observation = row["obs"]
            recorded = [int(index) for index in row.get("action") or []]
            pure_action = list(pure_blueprint(observation))
            override = candidate._early_spidops_search_override(observation)
            if override is None:
                actual = list(guarded_blueprint(observation))
            else:
                # Match the packaged candidate exactly: its guard returns before
                # calling the neural actor on this observation.
                actual = list(override)
                totals["direct_overrides"] += 1
                game_counts["direct_overrides"] += 1
                selection = observation.get("select") or {}
                options = selection.get("option") or []
                direct_overrides.append(
                    {
                        "game": game,
                        "step": row["step"],
                        "turn": row.get("turn"),
                        "recorded": recorded,
                        "pure_blueprint": pure_action,
                        "override": actual,
                        "source_option": options[pure_action[0]] if len(pure_action) == 1 else None,
                        "target_option": options[actual[0]],
                    }
                )

            totals["decisions"] += 1
            if actual == recorded:
                totals["candidate_exact"] += 1
            elif equivalent(observation, actual, recorded):
                totals["candidate_set_equivalent"] += 1
            else:
                totals["mismatches"] += 1
                mismatches.append(
                    {
                        "game": game,
                        "step": row["step"],
                        "turn": row.get("turn"),
                        "recorded": recorded,
                        "replayed": actual,
                        "pure_blueprint": pure_action,
                    }
                )
            if actual != pure_action and not equivalent(observation, actual, pure_action):
                totals["all_action_deltas"] += 1
                game_counts["all_action_deltas"] += 1
                if override is None:
                    totals["downstream_deltas"] += 1
                    game_counts["downstream_deltas"] += 1
                    downstream_deltas.append(
                        {
                            "game": game,
                            "step": row["step"],
                            "turn": row.get("turn"),
                            "pure_blueprint": pure_action,
                            "candidate": actual,
                        }
                    )
        totals["games"] += 1
        per_game[str(game)] = dict(game_counts)

    payload = {
        "schema": "early-pad-spidops-official-trace-audit/v1",
        "candidate": str(args.candidate.resolve()),
        "trace": str(args.trace.resolve()),
        "bot_name": args.bot_name,
        "totals": dict(totals),
        "per_game": per_game,
        "direct_overrides": direct_overrides,
        "downstream_deltas": downstream_deltas[:100],
        "mismatches": mismatches[:20],
        "evidence_boundary": (
            "This reproduces the packaged candidate and compares it with an always-updated "
            "frozen blueprint on the same official observations. Match scores remain an "
            "independent-panel estimate, not paired counterfactual outcomes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "direct_overrides": direct_overrides[:20]}, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
