#!/usr/bin/env python3
"""Analyze non-test randomized persistent-plan shards with stratified bootstrap."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--zero-plan", default="zero_blueprint")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=910_771)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in paths:
        path = path.resolve()
        manifest_file = manifest_path(path)
        if not manifest_file.exists():
            raise FileNotFoundError(f"missing shard manifest: {manifest_file}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("sealed") or manifest.get("split") == "test":
            raise RuntimeError(f"refusing to open sealed persistent-plan test shard: {path}")
        if manifest.get("schema") != "rocket-randomized-persistent-plan-shard/v1":
            raise ValueError(f"unexpected manifest schema: {manifest_file}")
        manifests.append(manifest)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "rocket-randomized-persistent-plan-match/v1":
                    raise ValueError(f"unexpected row schema at {path}:{line_number}")
                if row.get("split") == "test":
                    raise RuntimeError(f"refusing test row at {path}:{line_number}")
                rows.append(row)
    return rows, manifests


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def bootstrap_difference(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    opponents: list[str],
    plan: str,
    zero: str,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        differences = []
        for opponent in opponents:
            candidate = np.asarray(
                [float(row["win"]) for row in grouped[(opponent, plan)]],
                dtype=np.float64,
            )
            baseline = np.asarray(
                [float(row["win"]) for row in grouped[(opponent, zero)]],
                dtype=np.float64,
            )
            c_draw = candidate[rng.integers(0, candidate.size, candidate.size)]
            b_draw = baseline[rng.integers(0, baseline.size, baseline.size)]
            differences.append(float(c_draw.mean() - b_draw.mean()))
        draws[replicate] = float(np.mean(differences))
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.5)),
        float(np.quantile(draws, 0.975)),
    )


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1_000:
        raise ValueError("bootstrap must be at least 1000")
    rows, manifests = load_rows(args.inputs)
    if not rows:
        raise ValueError("no persistent-plan rows loaded")
    protocols = {row.get("protocol") for row in rows}
    libraries = {row.get("plan_library_fingerprint") for row in rows}
    if protocols != {"rocket-persistent-plan-causal/v1"}:
        raise ValueError(f"mixed protocols: {protocols}")
    if len(libraries) != 1:
        raise ValueError("analysis inputs mix different plan libraries")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["plan"]))].append(row)
    opponents = sorted({opponent for opponent, _ in grouped})
    plans = sorted({plan for _, plan in grouped})
    if args.zero_plan not in plans:
        raise ValueError(f"zero plan {args.zero_plan!r} is absent")
    for opponent in opponents:
        missing = [plan for plan in plans if not grouped[(opponent, plan)]]
        if missing:
            raise ValueError(f"opponent {opponent} lacks plans {missing}")
    rng = np.random.default_rng(args.seed)
    results = []
    for plan in plans:
        by_opponent = {}
        differences = []
        for opponent in opponents:
            plan_rows = grouped[(opponent, plan)]
            zero_rows = grouped[(opponent, args.zero_plan)]
            plan_rate = mean([float(row["win"]) for row in plan_rows])
            zero_rate = mean([float(row["win"]) for row in zero_rows])
            difference = plan_rate - zero_rate
            differences.append(difference)
            by_opponent[opponent] = {
                "games": len(plan_rows),
                "wins": sum(int(row["win"]) for row in plan_rows),
                "win_rate": plan_rate,
                "zero_games": len(zero_rows),
                "zero_wins": sum(int(row["win"]) for row in zero_rows),
                "zero_win_rate": zero_rate,
                "difference": difference,
            }
        plan_rows_all = [row for opponent in opponents for row in grouped[(opponent, plan)]]
        if plan == args.zero_plan:
            interval = (0.0, 0.0, 0.0)
        else:
            interval = bootstrap_difference(
                grouped,
                opponents,
                plan,
                args.zero_plan,
                replicates=args.bootstrap,
                rng=rng,
            )
        result = {
            "plan": plan,
            "games": len(plan_rows_all),
            "wins": sum(int(row["win"]) for row in plan_rows_all),
            "equal_opponent_win_rate": mean(
                [mean([float(row["win"]) for row in grouped[(opponent, plan)]]) for opponent in opponents]
            ),
            "stratified_difference_vs_zero": mean(differences),
            "bootstrap_95": {
                "lower": interval[0],
                "median": interval[1],
                "upper": interval[2],
            },
            "intervention_game_rate": mean(
                [float(row["games_with_intervention"]) for row in plan_rows_all]
            ),
            "mean_interventions": mean(
                [float(row["plan_interventions"]) for row in plan_rows_all]
            ),
            "mewtwo_board_rate": mean(
                [float(row["ever_mewtwo_on_board"]) for row in plan_rows_all]
            ),
            "mewtwo_attack_rate": mean(
                [float(int(row["mewtwo_attacks"]) > 0) for row in plan_rows_all]
            ),
            "second_ready_attacker_rate": mean(
                [float(row["ever_second_ready_attacker"]) for row in plan_rows_all]
            ),
            "exception_fallbacks": sum(int(row["exception_fallbacks"]) for row in plan_rows_all),
            "by_opponent": by_opponent,
            "d1_advances_point_gate": bool(
                plan != args.zero_plan
                and mean(differences) > 0.0
                and min(differences) >= -0.05
            ),
        }
        results.append(result)
    results.sort(
        key=lambda row: (
            float(row["stratified_difference_vs_zero"]),
            float(row["bootstrap_95"]["lower"]),
        ),
        reverse=True,
    )
    report = {
        "schema": "rocket-randomized-persistent-plan-analysis/v1",
        "protocol": "rocket-persistent-plan-causal/v1",
        "split": sorted({str(row["split"]) for row in rows}),
        "inputs": [str(path.resolve()) for path in args.inputs],
        "manifests": [str(manifest_path(path.resolve())) for path in args.inputs],
        "plan_library_fingerprint": next(iter(libraries)),
        "rows": len(rows),
        "opponents": opponents,
        "plans": plans,
        "zero_plan": args.zero_plan,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "results": results,
        "correctness": {
            "exception_fallbacks": sum(int(row["exception_fallbacks"]) for row in rows),
            "all_started_second": all(bool(row["started_second"]) for row in rows),
            "all_official_complete": all(int(row["result"]) in (0, 1, 2) for row in rows),
        },
        "claim_boundary": "development evidence from complete local official-engine games; not an arena strength claim",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    concise = {
        "output": str(output),
        "rows": len(rows),
        "opponents": opponents,
        "top": [
            {
                "plan": row["plan"],
                "difference": row["stratified_difference_vs_zero"],
                "bootstrap_95": row["bootstrap_95"],
                "intervention_game_rate": row["intervention_game_rate"],
                "d1_advances_point_gate": row["d1_advances_point_gate"],
            }
            for row in results[:8]
        ],
        "correctness": report["correctness"],
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
