#!/usr/bin/env python3
"""Development-only leave-one-shard-out exploration for semantic causal rules.

This script must never receive untouched admission evidence.  It searches a
small declared grid on opened selection shards and reports cross-shard metrics;
the chosen configuration must be frozen in a protocol before a separate
admission tool is allowed to inspect sealed data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


Action = tuple[int, int, int, int]
Pattern = tuple[Action, Action]


@dataclass(frozen=True)
class Root:
    source: str
    game: str
    baseline: int
    actions: Mapping[int, Action]
    deltas: Mapping[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=615_701)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def signature(option: Mapping[str, Any]) -> Action:
    return (
        int(option.get("type") or 0),
        int(option.get("hand_card_id") or 0),
        int(option.get("in_play_area") or 0),
        int(option.get("attack_id") or 0),
    )


def read_roots(paths: Sequence[Path]) -> list[Root]:
    result = []
    for path in paths:
        source = path.resolve().name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "rocket-live-truth-causal-action/v1" or not row.get("fully_complete"):
                continue
            baseline = int(row["baseline"])
            means = {int(key): float(value) for key, value in row["means"].items()}
            actions = {
                int(index): signature(option)
                for index, option in (row.get("candidate_options") or {}).items()
            }
            candidates = [int(index) for index in row["candidate_indices"]]
            if baseline not in actions or any(index not in actions for index in candidates):
                continue
            result.append(
                Root(
                    source=source,
                    game=f"{source}:game={int(row['game_id'])}",
                    baseline=baseline,
                    actions={index: actions[index] for index in candidates},
                    deltas={index: means[index] - means[baseline] for index in candidates},
                )
            )
    return result


def fit_rules(
    roots: Sequence[Root],
    *,
    min_count: int,
    min_sources: int,
    min_positive: int,
    min_mean: float,
    max_harm_rate: float,
    min_decisive_precision: float,
    require_source_nonnegative: bool,
) -> dict[Pattern, dict[str, Any]]:
    values: dict[Pattern, list[tuple[str, float]]] = defaultdict(list)
    for root in roots:
        base = root.actions[root.baseline]
        for index, delta in root.deltas.items():
            if index != root.baseline:
                values[(base, root.actions[index])].append((root.source, delta))
    rules = {}
    for pattern, observations in values.items():
        deltas = [delta for _, delta in observations]
        positives = sum(delta > 0 for delta in deltas)
        negatives = sum(delta < 0 for delta in deltas)
        seen_sources = {source for source, _ in observations}
        source_means = {
            source: statistics.fmean(delta for item_source, delta in observations if item_source == source)
            for source in seen_sources
        }
        decisive = positives + negatives
        if (
            len(deltas) < min_count
            or len(seen_sources) < min_sources
            or positives < min_positive
            or statistics.fmean(deltas) < min_mean
            or negatives / len(deltas) > max_harm_rate
            or positives / max(1, decisive) < min_decisive_precision
            or (require_source_nonnegative and any(value < 0 for value in source_means.values()))
        ):
            continue
        rules[pattern] = {
            "count": len(deltas),
            "positive": positives,
            "negative": negatives,
            "mean": statistics.fmean(deltas),
            "sources": len(seen_sources),
            "source_means": source_means,
        }
    return rules


def predict(roots: Sequence[Root], rules: Mapping[Pattern, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for root in roots:
        base = root.actions[root.baseline]
        choices = []
        for index, delta in root.deltas.items():
            if index == root.baseline:
                continue
            pattern = (base, root.actions[index])
            if pattern in rules:
                rule = rules[pattern]
                choices.append(
                    (
                        float(rule["mean"]),
                        -float(rule["negative"]) / float(rule["count"]),
                        int(rule["count"]),
                        index,
                        pattern,
                        delta,
                    )
                )
        if choices:
            choice = max(choices)
            result.append(
                {
                    "game": root.game,
                    "intervened": True,
                    "delta": float(choice[-1]),
                    "pattern": repr(choice[-2]),
                }
            )
        else:
            result.append({"game": root.game, "intervened": False, "delta": 0.0, "pattern": None})
    return result


def interval(rows: Sequence[Mapping[str, Any]], samples: int, seed: int) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["game"])].append(float(row["delta"]))
    games = sorted(grouped)
    observed = statistics.fmean(value for values in grouped.values() for value in values)
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        chosen = [games[rng.randrange(len(games))] for _ in games]
        boot.append(statistics.fmean(value for game in chosen for value in grouped[game]))
    boot.sort()
    return observed, boot[max(0, int(0.025 * len(boot)) - 1)], boot[min(len(boot) - 1, int(0.975 * len(boot)))]


def metrics(rows: Sequence[Mapping[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    interventions = [row for row in rows if row["intervened"]]
    positive = sum(float(row["delta"]) > 0 for row in interventions)
    negative = sum(float(row["delta"]) < 0 for row in interventions)
    tied = len(interventions) - positive - negative
    observed, lower, upper = interval(rows, samples, seed)
    return {
        "roots": len(rows),
        "interventions": len(interventions),
        "intervention_rate": len(interventions) / max(1, len(rows)),
        "positive": positive,
        "negative": negative,
        "tie": tied,
        "non_harm_rate": (positive + tied) / max(1, len(interventions)),
        "decisive_precision": positive / max(1, positive + negative),
        "mean_paired_improvement": observed,
        "game_clustered_95_interval": [lower, upper],
    }


def main() -> int:
    args = parse_args()
    roots = read_roots(args.data)
    sources = sorted({root.source for root in roots})
    if len(sources) < 3:
        raise ValueError("at least three independent development shards are required")
    grid = itertools.product(
        (2, 3, 4, 5),
        (1, 2),
        (1, 2, 3),
        (0.10, 0.25, 0.40),
        (0.0, 0.10, 0.20),
        (0.65, 0.75),
        (False, True),
    )
    table = []
    for config_index, values in enumerate(grid):
        config = dict(
            zip(
                (
                    "min_count",
                    "min_sources",
                    "min_positive",
                    "min_mean",
                    "max_harm_rate",
                    "min_decisive_precision",
                    "require_source_nonnegative",
                ),
                values,
            )
        )
        held_rows = []
        held_rule_counts = {}
        for held_source in sources:
            training = [root for root in roots if root.source != held_source]
            held = [root for root in roots if root.source == held_source]
            rules = fit_rules(training, **config)
            held_rule_counts[held_source] = len(rules)
            held_rows.extend(predict(held, rules))
        result = metrics(
            held_rows,
            args.bootstrap_samples,
            args.seed + config_index,
        )
        qualifies = bool(
            8 <= result["interventions"] <= int(0.35 * result["roots"])
            and result["non_harm_rate"] >= 0.85
            and result["decisive_precision"] >= 0.70
            and result["mean_paired_improvement"] > 0
            and result["game_clustered_95_interval"][0] > 0
        )
        table.append(
            {
                "config": config,
                "held_rule_counts": held_rule_counts,
                "metrics": result,
                "qualifies": qualifies,
            }
        )
    table.sort(
        key=lambda row: (
            row["qualifies"],
            row["metrics"]["game_clustered_95_interval"][0],
            row["metrics"]["mean_paired_improvement"],
            row["metrics"]["non_harm_rate"],
            -row["metrics"]["intervention_rate"],
        ),
        reverse=True,
    )
    print(
        json.dumps(
            {
                "opened_development_only": True,
                "sources": sources,
                "roots": len(roots),
                "grid_configurations": len(table),
                "qualifying_configurations": sum(row["qualifies"] for row in table),
                "top": table[: args.top],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
