#!/usr/bin/env python3
"""Summarize opened development-only causal action evidence.

Do not pass an untouched admission shard to this script.  It intentionally
reports labels and is therefore suitable only after evidence has been opened
for model development or a selection gate has rejected.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--minimum-pattern-count", type=int, default=4)
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def action(row: dict[str, Any], index: int) -> tuple[int, int, int, int]:
    option = (row.get("candidate_options") or {}).get(str(index)) or {}
    return (
        int(option.get("type") or 0),
        int(option.get("hand_card_id") or 0),
        int(option.get("in_play_area") or 0),
        int(option.get("attack_id") or 0),
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.data:
        source = path.resolve().name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "rocket-live-truth-causal-action/v1" or not row.get("fully_complete"):
                continue
            row["_source"] = source
            rows.append(row)

    root_deltas = []
    oracle_positive = Counter()
    pair_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    pair_source_groups: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    alt_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    base_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    by_turn: dict[int, list[float]] = defaultdict(list)
    pair_counts = Counter()
    for row in rows:
        baseline = int(row["baseline"])
        means = {int(key): float(value) for key, value in row["means"].items()}
        base_value = means[baseline]
        alternatives = [index for index in map(int, row["candidate_indices"]) if index != baseline]
        if not alternatives:
            continue
        best_alt = max(alternatives, key=lambda index: means[index])
        best_delta = means[best_alt] - base_value
        root_deltas.append(best_delta)
        source = str(row["_source"])
        by_source[source].append(best_delta)
        turn = int(row.get("turn") or 0)
        by_turn[turn].append(best_delta)
        oracle_positive["positive"] += int(best_delta > 0)
        oracle_positive["tie"] += int(best_delta == 0)
        oracle_positive["negative"] += int(best_delta < 0)
        base_action = action(row, baseline)
        for alternative in alternatives:
            delta = means[alternative] - base_value
            alt_action = action(row, alternative)
            pair_groups[(base_action, alt_action)].append(delta)
            pair_source_groups[(base_action, alt_action)][source].append(delta)
            alt_groups[alt_action].append(delta)
            base_groups[base_action].append(delta)
            if abs(delta) >= 0.5:
                pair_counts["non_tied"] += 1
                pair_counts["positive"] += int(delta > 0)
                pair_counts["negative"] += int(delta < 0)
            else:
                pair_counts["tied"] += 1

    def ranked(groups: dict[tuple[Any, ...], list[float]]) -> list[dict[str, Any]]:
        result = []
        for key, values in groups.items():
            if len(values) < args.minimum_pattern_count:
                continue
            positives = sum(value > 0 for value in values)
            negatives = sum(value < 0 for value in values)
            result.append(
                {
                    "pattern": repr(key),
                    "count": len(values),
                    "positive": positives,
                    "negative": negatives,
                    "tie": len(values) - positives - negatives,
                    "precision_positive": positives / max(1, positives + negatives),
                    "mean_delta": mean(values),
                }
            )
        return sorted(result, key=lambda item: (item["mean_delta"], item["count"]), reverse=True)[: args.top]

    report = {
        "opened_development_only": True,
        "sources": [path.resolve().name for path in args.data],
        "roots": len(rows),
        "root_oracle": {
            **dict(oracle_positive),
            "mean_best_alternative_delta": mean(root_deltas),
        },
        "pairs": dict(pair_counts),
        "by_source": {
            source: {
                "roots": len(values),
                "positive_roots": sum(value > 0 for value in values),
                "mean_best_alternative_delta": mean(values),
            }
            for source, values in sorted(by_source.items())
        },
        "by_turn": {
            str(turn): {
                "roots": len(values),
                "positive_roots": sum(value > 0 for value in values),
                "mean_best_alternative_delta": mean(values),
            }
            for turn, values in sorted(by_turn.items())
        },
        "best_action_pairs": ranked(pair_groups),
        "stable_action_pairs": sorted(
            (
                {
                    "pattern": repr(pattern),
                    "count": sum(len(values) for values in sources.values()),
                    "non_tied_sources": sum(
                        any(value != 0 for value in values)
                        for values in sources.values()
                    ),
                    "pooled_mean_delta": mean(
                        value for values in sources.values() for value in values
                    ),
                    "sources": {
                        source: {
                            "count": len(values),
                            "positive": sum(value > 0 for value in values),
                            "negative": sum(value < 0 for value in values),
                            "tie": sum(value == 0 for value in values),
                            "mean_delta": mean(values),
                        }
                        for source, values in sorted(sources.items())
                    },
                }
                for pattern, sources in pair_source_groups.items()
                if sum(len(values) for values in sources.values())
                >= args.minimum_pattern_count
            ),
            key=lambda item: (
                item["non_tied_sources"],
                item["pooled_mean_delta"],
                item["count"],
            ),
            reverse=True,
        )[: args.top],
        "best_alternative_actions": ranked(alt_groups),
        "baseline_action_contexts": ranked(base_groups),
        "action_tuple": ["type", "hand_card_id", "in_play_area", "attack_id"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
