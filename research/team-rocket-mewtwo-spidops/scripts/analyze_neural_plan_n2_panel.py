#!/usr/bin/env python3
"""Analyze N2 ensemble panels with locked opponent-stratified gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


LOCKED_ARMS = (
    "zero_control",
    "mean_logits_025",
    "mean_logits_050",
    "mean_logits_100",
    "consensus_3of4",
    "consensus_4of4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--control", default="zero_control")
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=785901)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sidecar(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def load(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests = []
    arm_hashes = set()
    for raw in inputs:
        path = raw.resolve()
        manifest = json.loads(sidecar(path).read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-neural-plan-ensemble-panel-shard/v1":
            raise ValueError(f"unexpected N2 manifest: {sidecar(path)}")
        if manifest.get("sealed") or manifest.get("split") == "test":
            raise RuntimeError(f"refusing sealed/test N2 panel: {path}")
        if not manifest.get("official_engine_execution"):
            raise ValueError(f"N2 panel is not official execution: {path}")
        arm_hashes.add(str(manifest.get("arms_manifest_sha256")))
        manifests.append(manifest)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "rocket-neural-plan-ensemble-panel-match/v1":
                    raise ValueError(f"unexpected N2 row at {path}:{line_number}")
                rows.append(row)
    if len(arm_hashes) != 1:
        raise ValueError("N2 shards use different arm manifests")
    return rows, manifests


def bootstrap(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    opponents: list[str],
    arm: str,
    control: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    draws = np.empty(replicates, dtype=np.float64)
    arrays = {
        (opponent, label): np.asarray(
            [float(row["win"]) for row in grouped[(opponent, label)]], dtype=np.float64
        )
        for opponent in opponents
        for label in (arm, control)
    }
    for index in range(replicates):
        differences = []
        for opponent in opponents:
            candidate = arrays[(opponent, arm)]
            baseline = arrays[(opponent, control)]
            differences.append(
                float(
                    candidate[rng.integers(0, candidate.size, candidate.size)].mean()
                    - baseline[rng.integers(0, baseline.size, baseline.size)].mean()
                )
            )
        draws[index] = float(np.mean(differences))
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.5, 0.975)))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1_000:
        raise ValueError("bootstrap must be at least 1000")
    rows, manifests = load(args.inputs)
    if not rows:
        raise ValueError("no N2 rows")
    splits = {str(row["split"]) for row in rows}
    if {str(row["protocol"]) for row in rows} != {"rocket-neural-plan-ensemble/v1"}:
        raise ValueError("mixed N2 protocols")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({opponent for opponent, _ in grouped})
    arms = sorted({arm for _, arm in grouped})
    if args.control not in arms:
        raise ValueError("N2 control absent")
    if splits == {"development"} and tuple(sorted(arms)) != tuple(sorted(LOCKED_ARMS)):
        raise ValueError(f"N2 development arms differ from lock: {arms}")
    for opponent in opponents:
        for arm in arms:
            count = len(grouped[(opponent, arm)])
            if count == 0:
                raise ValueError(f"missing N2 cell: {opponent}/{arm}")
            if splits == {"development"} and count != 240:
                raise ValueError(f"N2 development cell is not 240 games: {opponent}/{arm}={count}")
    rng = np.random.default_rng(args.seed)
    results = []
    for arm in arms:
        arm_rows = [row for opponent in opponents for row in grouped[(opponent, arm)]]
        by_opponent = {}
        differences = []
        for opponent in opponents:
            candidate_rows = grouped[(opponent, arm)]
            control_rows = grouped[(opponent, args.control)]
            rate = mean([float(row["win"]) for row in candidate_rows])
            control_rate = mean([float(row["win"]) for row in control_rows])
            difference = rate - control_rate
            differences.append(difference)
            by_opponent[opponent] = {
                "games": len(candidate_rows),
                "wins": sum(int(row["win"]) for row in candidate_rows),
                "win_rate": rate,
                "control_games": len(control_rows),
                "control_wins": sum(int(row["win"]) for row in control_rows),
                "control_win_rate": control_rate,
                "difference": difference,
            }
        interval = (
            (0.0, 0.0, 0.0)
            if arm == args.control
            else bootstrap(grouped, opponents, arm, args.control, args.bootstrap, rng)
        )
        difference = mean(differences)
        intervention = mean([float(row["games_with_intervention"]) for row in arm_rows])
        correctness = bool(
            all(bool(row["started_second"]) and int(row["result"]) in (0, 1, 2) for row in arm_rows)
            and sum(int(row["exception_fallbacks"]) for row in arm_rows) == 0
        )
        passes = bool(
            arm != args.control
            and difference >= 0.03
            and interval[0] > 0.0
            and min(differences) >= -0.02
            and intervention >= 0.05
            and correctness
        )
        results.append(
            {
                "arm": arm,
                "games": len(arm_rows),
                "wins": sum(int(row["win"]) for row in arm_rows),
                "equal_opponent_win_rate": mean(
                    [by_opponent[opponent]["win_rate"] for opponent in opponents]
                ),
                "stratified_difference_vs_control": difference,
                "bootstrap_95": {"lower": interval[0], "median": interval[1], "upper": interval[2]},
                "intervention_game_rate": intervention,
                "mean_intervention_decisions": mean(
                    [float(row["intervention_decisions"]) for row in arm_rows]
                ),
                "exception_fallbacks": sum(int(row["exception_fallbacks"]) for row in arm_rows),
                "correctness_pass": correctness,
                "development_gate_pass": passes,
                "by_opponent": by_opponent,
            }
        )
    eligible = sorted(
        [row for row in results if row["development_gate_pass"]],
        key=lambda row: (
            -float(row["stratified_difference_vs_control"]),
            -float(row["bootstrap_95"]["lower"]),
            str(row["arm"]),
        ),
    )
    advances = [str(eligible[0]["arm"])] if eligible else []
    report = {
        "schema": "rocket-neural-plan-ensemble-analysis/v1",
        "protocol": "rocket-neural-plan-ensemble/v1",
        "splits": sorted(splits),
        "inputs": [str(path.resolve()) for path in args.inputs],
        "manifests": [str(sidecar(path.resolve())) for path in args.inputs],
        "rows": len(rows),
        "opponents": opponents,
        "arms": arms,
        "control": args.control,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "results": sorted(results, key=lambda row: str(row["arm"])),
        "advances_at_most_one": advances,
        "correctness": {
            "all_started_second": all(bool(row["started_second"]) for row in rows),
            "all_official_complete": all(int(row["result"]) in (0, 1, 2) for row in rows),
            "exception_fallbacks": sum(int(row["exception_fallbacks"]) for row in rows),
        },
        "claim_boundary": "N2 development selection evidence; not an arena strength claim",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "advances_at_most_one": advances, "correctness": report["correctness"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
