#!/usr/bin/env python3
"""Apply the locked advancement gates to four N6 development shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


LOCKED_ARMS = ("control", "arm_a", "arm_b", "arm_c", "arm_d")
LOCKED_SEEDS = {791101, 791102, 791201, 791202}
LOCKED_OPPONENTS = {"AshMarnieGeneralV110_extracted", "HelbrosMarnieGeneralRecoveryV4_extracted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=791001)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    manifests = []
    for raw in inputs:
        path = raw.resolve()
        sidecar = path.with_suffix(".manifest.json")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-n6-development-panel-shard/v1":
            raise ValueError(f"unexpected N6 panel manifest: {sidecar}")
        if not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
            raise ValueError(f"N6 shard lacks strict official execution: {path}")
        if sha256(path) != manifest.get("output_sha256"):
            raise ValueError(f"N6 shard changed after collection: {path}")
        manifests.append(manifest)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("schema") != "rocket-n6-development-panel-match/v1":
                raise ValueError(f"unexpected N6 row {path}:{line_number}")
            rows.append(row)
    return rows, manifests


def bootstrap_difference(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    opponents: list[str],
    arm: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    samples = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        differences = []
        for opponent in opponents:
            candidate = np.asarray([float(row["win"]) for row in grouped[(opponent, arm)]])
            control = np.asarray([float(row["win"]) for row in grouped[(opponent, "control")]])
            candidate_draw = candidate[rng.integers(0, candidate.size, candidate.size)]
            control_draw = control[rng.integers(0, control.size, control.size)]
            differences.append(float(candidate_draw.mean() - control_draw.mean()))
        samples[draw] = float(np.mean(differences))
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.5, 0.975)))


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N6 panel analysis: {output}")
    if len(args.inputs) != 4 or args.bootstrap != 20_000 or args.seed != 791001:
        raise ValueError("N6 analysis inputs/bootstrap differ from lock")
    rows, manifests = load(args.inputs)
    if {int(item["schedule_seed"]) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N6 schedule seeds differ from lock")
    if {Path(str(item["opponent"])).name for item in manifests} != LOCKED_OPPONENTS:
        raise ValueError("N6 opponents differ from lock")
    if any(int(item["games_per_arm"]) != 120 or int(item["rows"]) != 600 for item in manifests):
        raise ValueError("N6 shard sizes differ from lock")
    if len({item["arms_manifest_sha256"] for item in manifests}) != 1:
        raise ValueError("N6 arm manifest changed across shards")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    if tuple(sorted({key[1] for key in grouped}, key=LOCKED_ARMS.index)) != LOCKED_ARMS:
        raise ValueError("N6 arms differ from lock")
    for opponent in opponents:
        if any(len(grouped[(opponent, arm)]) != 240 for arm in LOCKED_ARMS):
            raise ValueError(f"N6 aggregate count mismatch for {opponent}")
    rng = np.random.default_rng(args.seed)
    results = []
    for arm in LOCKED_ARMS:
        by_opponent = {}
        differences = []
        total_wins = total_games = 0
        for opponent in opponents:
            candidate = grouped[(opponent, arm)]
            control = grouped[(opponent, "control")]
            rate = float(np.mean([row["win"] for row in candidate]))
            control_rate = float(np.mean([row["win"] for row in control]))
            difference = rate - control_rate
            differences.append(difference)
            total_wins += sum(int(row["win"]) for row in candidate)
            total_games += len(candidate)
            by_opponent[opponent] = {
                "games": len(candidate),
                "wins": sum(int(row["win"]) for row in candidate),
                "win_rate": rate,
                "control_win_rate": control_rate,
                "difference": difference,
            }
        aggregate_difference = float(np.mean(differences))
        interval = (0.0, 0.0, 0.0) if arm == "control" else bootstrap_difference(
            grouped, opponents, arm, args.bootstrap, rng
        )
        arm_rows = [row for opponent in opponents for row in grouped[(opponent, arm)]]
        correctness = bool(
            len(arm_rows) == 480
            and all(row.get("started_second") for row in arm_rows)
            and all(int(row["result"]) in (0, 1, 2) for row in arm_rows)
            and sum(int(row.get("exception_fallbacks", 0)) for row in arm_rows) == 0
        )
        gate = bool(
            arm != "control"
            and aggregate_difference >= 0.015
            and min(differences) > 0.0
            and interval[0] > -0.005
            and correctness
        )
        results.append(
            {
                "arm": arm,
                "games": total_games,
                "wins": total_wins,
                "stratified_difference_vs_control": aggregate_difference,
                "bootstrap_95": {"lower": interval[0], "median": interval[1], "upper": interval[2]},
                "worst_opponent_difference": min(differences),
                "correctness_pass": correctness,
                "development_gate_pass": gate,
                "by_opponent": by_opponent,
            }
        )
    eligible = sorted(
        (row for row in results if row["development_gate_pass"]),
        key=lambda row: (
            -float(row["stratified_difference_vs_control"]),
            -float(row["bootstrap_95"]["lower"]),
            str(row["arm"]),
        ),
    )
    advances = [str(eligible[0]["arm"])] if eligible else []
    report = {
        "schema": "rocket-n6-development-panel-analysis/v1",
        "protocol": "N6_BELIEF_ROLLOUT_DISTILLATION_PROTOCOL.md",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "arms_manifest_sha256": manifests[0]["arms_manifest_sha256"],
        "rows": len(rows),
        "opponents": opponents,
        "arms": list(LOCKED_ARMS),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "locked_gates": {
            "minimum_games_per_arm": 480,
            "minimum_stratified_improvement": 0.015,
            "minimum_worst_opponent_difference_strict": 0.0,
            "bootstrap_95_lower_strict": -0.005,
            "strict_correctness": True,
        },
        "results": results,
        "advances_exactly_one_or_none": advances,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "advances": advances}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

