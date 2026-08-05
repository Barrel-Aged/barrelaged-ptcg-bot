#!/usr/bin/env python3
"""Apply the locked N9 advancement gate to four development shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_belief_rollout_n6_panel import bootstrap_difference


ORDER = ("control", "epoch_01", "epoch_02", "epoch_03", "epoch_04")
EPOCH = {name: index for index, name in enumerate(ORDER)}
LOCKED_SEEDS = {798101, 798102, 798201, 798202}
LOCKED_OPPONENTS = {
    "AshMarnieGeneralV110_extracted",
    "HelbrosMarnieGeneralRecoveryV4_extracted",
}
ANALYSIS_SEED = 798001


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    manifests = []
    for raw in paths:
        path = raw.resolve()
        sidecar = path.with_suffix(".manifest.json")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-n9-development-panel-shard/v1":
            raise ValueError(f"unexpected N9 shard manifest: {sidecar}")
        if not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
            raise ValueError(f"N9 shard is not strict official execution: {path}")
        if digest(path) != manifest.get("output_sha256"):
            raise ValueError(f"N9 shard changed: {path}")
        manifests.append(manifest)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("schema") != "rocket-n9-development-panel-match/v1":
                raise ValueError(f"unexpected N9 row {path}:{line_number}")
            rows.append(row)
    return rows, manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N9 analysis: {output}")
    if len(args.inputs) != 4 or args.bootstrap != 20_000 or args.seed != ANALYSIS_SEED:
        raise ValueError("N9 analysis arguments differ from lock")
    rows, manifests = load(args.inputs)
    if {int(item["schedule_seed"]) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N9 shard seeds differ from lock")
    if {Path(str(item["opponent"])).name for item in manifests} != LOCKED_OPPONENTS:
        raise ValueError("N9 opponents differ from lock")
    if any(int(item["games_per_arm"]) != 120 or int(item["rows"]) != 600 for item in manifests):
        raise ValueError("N9 shard size differs from lock")
    if len({item["arms_manifest_sha256"] for item in manifests}) != 1 or len({item["collector_sha256"] for item in manifests}) != 1:
        raise ValueError("N9 arm/collector hashes changed across shards")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    if tuple(sorted({key[1] for key in grouped}, key=ORDER.index)) != ORDER:
        raise ValueError("N9 arms differ from lock")
    for opponent in opponents:
        if any(len(grouped[(opponent, arm)]) != 240 for arm in ORDER):
            raise ValueError(f"N9 cell count mismatch: {opponent}")
    rng = np.random.default_rng(args.seed)
    results = []
    for arm in ORDER:
        cells = {}
        differences = []
        total_wins = 0
        for opponent in opponents:
            candidate = grouped[(opponent, arm)]
            control = grouped[(opponent, "control")]
            rate = float(np.mean([row["win"] for row in candidate]))
            control_rate = float(np.mean([row["win"] for row in control]))
            difference = rate - control_rate
            differences.append(difference)
            wins = sum(int(row["win"]) for row in candidate)
            total_wins += wins
            cells[opponent] = {
                "games": len(candidate), "wins": wins, "win_rate": rate,
                "control_win_rate": control_rate, "difference": difference,
            }
        aggregate = float(np.mean(differences))
        interval = (
            (0.0, 0.0, 0.0)
            if arm == "control"
            else bootstrap_difference(grouped, opponents, arm, args.bootstrap, rng)
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
            and aggregate >= 0.020
            and min(differences) > 0.0
            and interval[0] > -0.005
            and correctness
        )
        results.append(
            {
                "arm": arm,
                "epoch": EPOCH[arm],
                "games": 480,
                "wins": total_wins,
                "stratified_difference_vs_control": aggregate,
                "bootstrap_95": {"lower": interval[0], "median": interval[1], "upper": interval[2]},
                "worst_opponent_difference": min(differences),
                "correctness_pass": correctness,
                "development_gate_pass": gate,
                "by_opponent": cells,
            }
        )
    eligible = sorted(
        (row for row in results if row["development_gate_pass"]),
        key=lambda row: (
            -float(row["stratified_difference_vs_control"]),
            -float(row["worst_opponent_difference"]),
            int(row["epoch"]),
        ),
    )
    advances = [str(eligible[0]["arm"])] if eligible else []
    report = {
        "schema": "rocket-n9-development-panel-analysis/v1",
        "protocol": "rocket-balanced-exact-pcgrad-awr/v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "input_sha256": {str(path.resolve()): digest(path.resolve()) for path in args.inputs},
        "arms_manifest_sha256": manifests[0]["arms_manifest_sha256"],
        "collector_sha256": manifests[0]["collector_sha256"],
        "analyzer_sha256": digest(Path(__file__).resolve()),
        "rows": len(rows),
        "opponents": opponents,
        "arms": list(ORDER),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "locked_gates": {
            "minimum_games_per_arm": 480,
            "minimum_stratified_improvement": 0.020,
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
