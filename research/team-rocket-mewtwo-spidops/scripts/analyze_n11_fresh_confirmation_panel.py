#!/usr/bin/env python3
"""Apply the locked gate to four complete, fresh N11 shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

import analyze_n10_confirmation_panel as n10


ORDER = ("control", "candidate")
LOCKED_SEEDS = {800101, 800102, 800201, 800202}
LOCKED_OPPONENTS = {"AshMarnieGeneralV110_extracted", "HelbrosMarnieGeneralRecoveryV4_extracted"}
ANALYSIS_SEED = 800001
N10_ANALYZER_SHA = "db11727af97580a7c67a06118fc7cbc3e223faa95bf7b1718701f16523fc94da"
ARMS_SHA = "d60ab7cbd53a49cc9a924f8a9b8d7445937f7db4b9c80d90ef7a04c007bdf443"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020"


def load(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests = []
    for raw in paths:
        path = raw.resolve()
        sidecar = path.with_suffix(".manifest.json")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-n11-confirmation-shard/v1":
            raise ValueError(f"unexpected N11 shard manifest: {sidecar}")
        if not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
            raise ValueError(f"N11 shard is not strict official execution: {path}")
        if n10.digest(path) != manifest.get("output_sha256"):
            raise ValueError(f"N11 shard changed: {path}")
        manifests.append(manifest)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("schema") != "rocket-n11-confirmation-match/v1":
                raise ValueError(f"unexpected N11 row {path}:{line_number}")
            rows.append(row)
    return rows, manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N11 analysis: {output}")
    if n10.digest(Path(n10.__file__).resolve()) != N10_ANALYZER_SHA:
        raise ValueError("frozen bootstrap dependency changed")
    if len(args.inputs) != 4 or args.bootstrap != 50_000 or args.seed != ANALYSIS_SEED:
        raise ValueError("N11 analysis arguments differ from lock")
    rows, manifests = load(args.inputs)
    if {int(item["schedule_seed"]) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N11 shard seeds differ from lock")
    if {Path(str(item["opponent"])).name for item in manifests} != LOCKED_OPPONENTS:
        raise ValueError("N11 opponents differ from lock")
    if any(int(item["games_per_arm"]) != 400 or int(item["rows"]) != 800 for item in manifests):
        raise ValueError("N11 shard size differs from lock")
    if {item["arms_manifest_sha256"] for item in manifests} != {ARMS_SHA}:
        raise ValueError("N11 arm manifest differs from lock")
    if len({item["collector_sha256"] for item in manifests}) != 1:
        raise ValueError("N11 collector changed across shards")
    if any(item.get("initial_model_sha256") != CONTROL_SHA or item.get("candidate_model_sha256") != CANDIDATE_SHA for item in manifests):
        raise ValueError("N11 model hashes differ from lock")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    if tuple(sorted({key[1] for key in grouped}, key=ORDER.index)) != ORDER:
        raise ValueError("N11 arms differ from lock")
    for opponent in opponents:
        if any(len(grouped[(opponent, arm)]) != 800 for arm in ORDER):
            raise ValueError(f"N11 cell count mismatch: {opponent}")
    arm_hash = {"control": CONTROL_SHA, "candidate": CANDIDATE_SHA}
    correctness = bool(
        len(rows) == 3_200
        and all(row.get("started_second") for row in rows)
        and all(int(row["result"]) in (0, 1, 2) for row in rows)
        and all(str(row.get("arm_sha256")) == arm_hash[str(row["arm"])] for row in rows)
        and sum(int(row.get("exception_fallbacks", 0)) for row in rows) == 0
    )
    rng = np.random.default_rng(args.seed)
    opponent_intervals, aggregate_interval = n10.bootstrap(grouped, opponents, args.bootstrap, rng)
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
        "schema": "rocket-n11-confirmation-analysis/v1",
        "protocol": "rocket-n11-fresh-epoch4-confirmation/v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "input_sha256": {str(path.resolve()): n10.digest(path.resolve()) for path in args.inputs},
        "arms_manifest_sha256": ARMS_SHA,
        "collector_sha256": manifests[0]["collector_sha256"],
        "analyzer_sha256": n10.digest(Path(__file__).resolve()),
        "analyzer_dependency_sha256": N10_ANALYZER_SHA,
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

