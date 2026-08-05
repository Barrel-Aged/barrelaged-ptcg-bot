#!/usr/bin/env python3
"""Analyze non-test neural-plan panels with opponent-stratified bootstrap."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np


LOCKED_ARMS = (
    "zero_control",
    "seed_781101",
    "seed_781102",
    "seed_781103",
    "seed_781104",
    "mean_all4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--control", default="zero_control")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=783001)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def load(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    arm_manifest_hashes: set[str] = set()
    for raw_path in inputs:
        path = raw_path.resolve()
        sidecar = manifest_path(path)
        if not sidecar.exists():
            raise FileNotFoundError(sidecar)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-neural-plan-panel-shard/v1":
            raise ValueError(f"unexpected panel manifest: {sidecar}")
        if manifest.get("sealed") or manifest.get("split") == "test":
            raise RuntimeError(f"refusing sealed/test panel: {path}")
        if not manifest.get("official_engine_execution"):
            raise ValueError(f"panel is not official execution: {path}")
        arm_manifest_hashes.add(str(manifest.get("arms_manifest_sha256")))
        manifests.append(manifest)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "rocket-neural-plan-panel-match/v1":
                    raise ValueError(f"unexpected row at {path}:{line_number}")
                if row.get("split") == "test":
                    raise RuntimeError(f"refusing test row at {path}:{line_number}")
                rows.append(row)
    if len(arm_manifest_hashes) != 1:
        raise ValueError("panel shards use different arm manifests")
    return rows, manifests


def bootstrap_difference(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    opponents: list[str],
    arm: str,
    control: str,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        differences = []
        for opponent in opponents:
            candidate = np.asarray(
                [float(row["win"]) for row in grouped[(opponent, arm)]],
                dtype=np.float64,
            )
            baseline = np.asarray(
                [float(row["win"]) for row in grouped[(opponent, control)]],
                dtype=np.float64,
            )
            candidate_draw = candidate[
                rng.integers(0, candidate.size, candidate.size)
            ]
            baseline_draw = baseline[
                rng.integers(0, baseline.size, baseline.size)
            ]
            differences.append(float(candidate_draw.mean() - baseline_draw.mean()))
        samples[index] = float(np.mean(differences))
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.5, 0.975)))


def average(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1_000:
        raise ValueError("bootstrap must be at least 1000")
    rows, manifests = load(args.inputs)
    if not rows:
        raise ValueError("no neural-plan panel rows")
    splits = {str(row["split"]) for row in rows}
    protocols = {str(row["protocol"]) for row in rows}
    if protocols != {"rocket-neural-plan-residual/v1"}:
        raise ValueError(f"mixed protocols: {protocols}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    arms = sorted({key[1] for key in grouped})
    if args.control not in arms:
        raise ValueError("control arm is absent")
    if splits == {"development"} and tuple(sorted(arms)) != tuple(sorted(LOCKED_ARMS)):
        raise ValueError(f"development arm set differs from lock: {arms}")
    for opponent in opponents:
        missing = [arm for arm in arms if not grouped[(opponent, arm)]]
        if missing:
            raise ValueError(f"{opponent} lacks arms: {missing}")
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    for arm in arms:
        differences = []
        by_opponent = {}
        arm_rows = [row for opponent in opponents for row in grouped[(opponent, arm)]]
        for opponent in opponents:
            candidate_rows = grouped[(opponent, arm)]
            control_rows = grouped[(opponent, args.control)]
            rate = average([float(row["win"]) for row in candidate_rows])
            control_rate = average([float(row["win"]) for row in control_rows])
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
            else bootstrap_difference(
                grouped,
                opponents,
                arm,
                args.control,
                replicates=args.bootstrap,
                rng=rng,
            )
        )
        improvement = average(differences)
        intervention_rate = average(
            [float(row["games_with_intervention"]) for row in arm_rows]
        )
        correctness = bool(
            all(bool(row["started_second"]) for row in arm_rows)
            and all(int(row["result"]) in (0, 1, 2) for row in arm_rows)
            and sum(int(row["exception_fallbacks"]) for row in arm_rows) == 0
        )
        eligible = bool(
            arm != args.control
            and improvement > 0.0
            and intervention_rate >= 0.05
            and min(differences) >= -0.02
            and correctness
        )
        results.append(
            {
                "arm": arm,
                "games": len(arm_rows),
                "wins": sum(int(row["win"]) for row in arm_rows),
                "equal_opponent_win_rate": average(
                    [by_opponent[opponent]["win_rate"] for opponent in opponents]
                ),
                "stratified_difference_vs_control": improvement,
                "bootstrap_95": {
                    "lower": interval[0],
                    "median": interval[1],
                    "upper": interval[2],
                },
                "intervention_game_rate": intervention_rate,
                "mean_intervention_decisions": average(
                    [float(row["intervention_decisions"]) for row in arm_rows]
                ),
                "mewtwo_board_rate": average(
                    [float(row["ever_mewtwo_on_board"]) for row in arm_rows]
                ),
                "second_ready_rate": average(
                    [float(row["ever_second_ready_attacker"]) for row in arm_rows]
                ),
                "exception_fallbacks": sum(
                    int(row["exception_fallbacks"]) for row in arm_rows
                ),
                "correctness_pass": correctness,
                "development_gate_pass": eligible,
                "by_opponent": by_opponent,
            }
        )
    ranked = sorted(
        [row for row in results if row["development_gate_pass"]],
        key=lambda row: (
            -float(row["stratified_difference_vs_control"]),
            -float(row["bootstrap_95"]["lower"]),
            str(row["arm"]),
        ),
    )
    advances = [str(row["arm"]) for row in ranked[:2]]
    report = {
        "schema": "rocket-neural-plan-panel-analysis/v1",
        "protocol": "rocket-neural-plan-residual/v1",
        "splits": sorted(splits),
        "inputs": [str(path.resolve()) for path in args.inputs],
        "manifests": [str(manifest_path(path.resolve())) for path in args.inputs],
        "rows": len(rows),
        "opponents": opponents,
        "arms": arms,
        "control": args.control,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "results": sorted(results, key=lambda row: str(row["arm"])),
        "advances_at_most_two": advances,
        "correctness": {
            "all_started_second": all(bool(row["started_second"]) for row in rows),
            "all_official_complete": all(int(row["result"]) in (0, 1, 2) for row in rows),
            "exception_fallbacks": sum(int(row["exception_fallbacks"]) for row in rows),
        },
        "claim_boundary": "development selection evidence; not an arena strength claim",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": len(rows),
                "advances_at_most_two": advances,
                "correctness": report["correctness"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
