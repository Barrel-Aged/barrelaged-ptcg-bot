#!/usr/bin/env python3
"""Apply the locked N10 confirmation gate to four untouched shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ORDER = ("control", "candidate")
LOCKED_SEEDS = {799101, 799102, 799201, 799202}
LOCKED_OPPONENTS = {"AshMarnieGeneralV110_extracted", "HelbrosMarnieGeneralRecoveryV4_extracted"}
ANALYSIS_SEED = 799001
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests = []
    for raw in paths:
        path = raw.resolve()
        sidecar = path.with_suffix(".manifest.json")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-n10-confirmation-shard/v1":
            raise ValueError(f"unexpected N10 shard manifest: {sidecar}")
        if not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
            raise ValueError(f"N10 shard is not strict official execution: {path}")
        if digest(path) != manifest.get("output_sha256"):
            raise ValueError(f"N10 shard changed: {path}")
        manifests.append(manifest)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("schema") != "rocket-n10-confirmation-match/v1":
                raise ValueError(f"unexpected N10 row {path}:{line_number}")
            rows.append(row)
    return rows, manifests


def bootstrap(grouped: dict[tuple[str, str], list[dict[str, Any]]], opponents: list[str], replicates: int, rng: np.random.Generator) -> tuple[dict[str, tuple[float, float, float]], tuple[float, float, float]]:
    aggregate = np.empty(replicates, dtype=np.float64)
    by_opponent: dict[str, np.ndarray] = {}
    for opponent in opponents:
        candidate = np.asarray([float(row["win"]) for row in grouped[(opponent, "candidate")]])
        control = np.asarray([float(row["win"]) for row in grouped[(opponent, "control")]])
        values = np.empty(replicates, dtype=np.float64)
        for draw in range(replicates):
            values[draw] = candidate[rng.integers(0, candidate.size, candidate.size)].mean() - control[rng.integers(0, control.size, control.size)].mean()
        by_opponent[opponent] = values
    aggregate[:] = np.mean(np.stack([by_opponent[name] for name in opponents], axis=0), axis=0)
    quantiles = lambda values: tuple(float(v) for v in np.quantile(values, (0.025, 0.5, 0.975)))
    return {name: quantiles(values) for name, values in by_opponent.items()}, quantiles(aggregate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N10 analysis: {output}")
    if len(args.inputs) != 4 or args.bootstrap != 50_000 or args.seed != ANALYSIS_SEED:
        raise ValueError("N10 analysis arguments differ from lock")
    rows, manifests = load(args.inputs)
    if {int(item["schedule_seed"]) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N10 shard seeds differ from lock")
    if {Path(str(item["opponent"])).name for item in manifests} != LOCKED_OPPONENTS:
        raise ValueError("N10 opponents differ from lock")
    if any(int(item["games_per_arm"]) != 400 or int(item["rows"]) != 800 for item in manifests):
        raise ValueError("N10 shard size differs from lock")
    if len({item["arms_manifest_sha256"] for item in manifests}) != 1 or len({item["collector_sha256"] for item in manifests}) != 1:
        raise ValueError("N10 arm/collector hashes changed across shards")
    if any(item.get("initial_model_sha256") != CONTROL_SHA or item.get("candidate_model_sha256") != CANDIDATE_SHA for item in manifests):
        raise ValueError("N10 model hashes differ from lock")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    if tuple(sorted({key[1] for key in grouped}, key=ORDER.index)) != ORDER:
        raise ValueError("N10 arms differ from lock")
    for opponent in opponents:
        if any(len(grouped[(opponent, arm)]) != 800 for arm in ORDER):
            raise ValueError(f"N10 cell count mismatch: {opponent}")
    arm_hash = {"control": CONTROL_SHA, "candidate": CANDIDATE_SHA}
    correctness = bool(
        len(rows) == 3_200
        and all(row.get("started_second") for row in rows)
        and all(int(row["result"]) in (0, 1, 2) for row in rows)
        and all(str(row.get("arm_sha256")) == arm_hash[str(row["arm"])] for row in rows)
        and sum(int(row.get("exception_fallbacks", 0)) for row in rows) == 0
    )
    rng = np.random.default_rng(args.seed)
    opponent_intervals, aggregate_interval = bootstrap(grouped, opponents, args.bootstrap, rng)
    by_opponent = {}
    differences = []
    for opponent in opponents:
        candidate = grouped[(opponent, "candidate")]
        control = grouped[(opponent, "control")]
        candidate_rate = float(np.mean([row["win"] for row in candidate]))
        control_rate = float(np.mean([row["win"] for row in control]))
        difference = candidate_rate - control_rate
        differences.append(difference)
        interval = opponent_intervals[opponent]
        by_opponent[opponent] = {
            "games_per_arm": 800,
            "candidate_wins": sum(int(row["win"]) for row in candidate),
            "candidate_win_rate": candidate_rate,
            "control_wins": sum(int(row["win"]) for row in control),
            "control_win_rate": control_rate,
            "difference": difference,
            "bootstrap_95": {"lower": interval[0], "median": interval[1], "upper": interval[2]},
        }
    aggregate = float(np.mean(differences))
    passed = bool(correctness and aggregate >= 0.020 and aggregate_interval[0] > 0.0 and min(differences) > 0.0)
    report = {
        "schema": "rocket-n10-confirmation-analysis/v1",
        "protocol": "rocket-n10-independent-epoch4-confirmation/v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "input_sha256": {str(path.resolve()): digest(path.resolve()) for path in args.inputs},
        "arms_manifest_sha256": manifests[0]["arms_manifest_sha256"],
        "collector_sha256": manifests[0]["collector_sha256"],
        "analyzer_sha256": digest(Path(__file__).resolve()),
        "rows": len(rows),
        "games_per_arm": 1_600,
        "opponents": opponents,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "correctness_pass": correctness,
        "by_opponent": by_opponent,
        "stratified_difference_vs_control": aggregate,
        "bootstrap_95": {"lower": aggregate_interval[0], "median": aggregate_interval[1], "upper": aggregate_interval[2]},
        "worst_opponent_difference": min(differences),
        "locked_gates": {
            "minimum_total_rows": 3_200,
            "minimum_games_per_arm_opponent_cell": 800,
            "minimum_stratified_improvement": 0.020,
            "minimum_worst_opponent_difference_strict": 0.0,
            "bootstrap_95_lower_strict": 0.0,
            "strict_correctness": True,
        },
        "confirmation_pass": passed,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "confirmation_pass": passed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

