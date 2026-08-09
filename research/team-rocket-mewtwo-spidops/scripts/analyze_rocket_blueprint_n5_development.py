#!/usr/bin/env python3
"""Analyze only unsealed N5 development panels under the locked advancement gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


LOCKED_ARMS = (
    "control",
    "seed_788101",
    "seed_788102",
    "seed_788103",
    "seed_788104",
    "mean_all4",
)
LOCKED_SEEDS = {789101, 789102, 789201, 789202}
LOCKED_OPPONENTS = {"AshMarnieGeneralV110_extracted", "HelbrosMarnieGeneralRecoveryV4_extracted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=789001)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(inputs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    arm_hashes: set[str] = set()
    for raw in inputs:
        path = raw.resolve()
        sidecar = manifest_path(path)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-blueprint-n5-panel-shard/v1":
            raise ValueError(f"unexpected N5 manifest: {sidecar}")
        if manifest.get("sealed") or manifest.get("split") != "development":
            raise RuntimeError(f"development analyzer refuses sealed/non-development shard: {path}")
        if not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
            raise ValueError(f"N5 shard lacks official strict execution: {path}")
        if sha256(path) != manifest.get("output_sha256"):
            raise ValueError(f"N5 development shard changed after collection: {path}")
        arm_hashes.add(str(manifest.get("arms_manifest_sha256")))
        manifests.append(manifest)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "rocket-blueprint-n5-panel-match/v1":
                    raise ValueError(f"unexpected N5 row at {path}:{line_number}")
                if row.get("split") != "development":
                    raise RuntimeError(f"development analyzer refuses row at {path}:{line_number}")
                rows.append(row)
    if len(arm_hashes) != 1:
        raise ValueError("N5 development shards use different arm manifests")
    if len({str(item.get("candidate_bundle_sha256")) for item in manifests}) != 1:
        raise ValueError("N5 development candidate runtime changed across shards")
    return rows, manifests


def bootstrap_difference(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    opponents: list[str],
    arm: str,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        differences = []
        for opponent in opponents:
            candidate = np.asarray([float(row["win"]) for row in grouped[(opponent, arm)]])
            control = np.asarray([float(row["win"]) for row in grouped[(opponent, "control")]])
            candidate_draw = candidate[rng.integers(0, candidate.size, candidate.size)]
            control_draw = control[rng.integers(0, control.size, control.size)]
            differences.append(float(candidate_draw.mean() - control_draw.mean()))
        samples[index] = float(np.mean(differences))
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.5, 0.975)))


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N5 development analysis: {output}")
    if len(args.inputs) != 4:
        raise ValueError("N5 development requires exactly four shards")
    if args.bootstrap != 20_000 or args.seed != 789001:
        raise ValueError("N5 development bootstrap count/seed differs from lock")
    rows, manifests = load(args.inputs)
    if not rows:
        raise ValueError("no N5 development rows")
    if {int(item.get("schedule_seed", -1)) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N5 development schedule seeds differ from lock")
    if any(int(item.get("games_per_arm", -1)) != 120 or int(item.get("rows", -1)) != 720 for item in manifests):
        raise ValueError("N5 development shard size differs from lock")
    if any(int(item.get("own_seat", -1)) != 1 for item in manifests):
        raise ValueError("N5 development did not keep Rocket in the second seat")
    if {Path(str(item["opponent"])).name for item in manifests} != LOCKED_OPPONENTS:
        raise ValueError("N5 development opponents differ from lock")
    for opponent in LOCKED_OPPONENTS:
        fingerprints = {
            (
                str(item.get("opponent_deck_sha256")),
                str(item.get("opponent_model_sha256")),
                str(item.get("opponent_main_sha256")),
                str(item.get("opponent_bundle_sha256")),
            )
            for item in manifests
            if Path(str(item["opponent"])).name == opponent
        }
        if len(fingerprints) != 1:
            raise ValueError(f"N5 development opponent changed across shards: {opponent}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    arms = tuple(sorted({key[1] for key in grouped}, key=LOCKED_ARMS.index))
    if arms != LOCKED_ARMS:
        raise ValueError(f"N5 development arms differ from lock: {arms}")
    if len(opponents) != 2:
        raise ValueError(f"N5 requires exactly two development opponents: {opponents}")
    for opponent in opponents:
        counts = {arm: len(grouped[(opponent, arm)]) for arm in arms}
        if set(counts.values()) != {240}:
            raise ValueError(f"N5 requires 240 games/arm/opponent, got {opponent}: {counts}")

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    for arm in arms:
        by_opponent: dict[str, Any] = {}
        differences: list[float] = []
        total_games = total_wins = 0
        for opponent in opponents:
            candidate = grouped[(opponent, arm)]
            control = grouped[(opponent, "control")]
            rate = float(np.mean([float(row["win"]) for row in candidate]))
            control_rate = float(np.mean([float(row["win"]) for row in control]))
            difference = rate - control_rate
            differences.append(difference)
            total_games += len(candidate)
            total_wins += sum(int(row["win"]) for row in candidate)
            by_opponent[opponent] = {
                "games": len(candidate),
                "wins": sum(int(row["win"]) for row in candidate),
                "win_rate": rate,
                "control_games": len(control),
                "control_wins": sum(int(row["win"]) for row in control),
                "control_win_rate": control_rate,
                "difference": difference,
            }
        difference = float(np.mean(differences))
        interval = (0.0, 0.0, 0.0) if arm == "control" else bootstrap_difference(grouped, opponents, arm, args.bootstrap, rng)
        arm_rows = [row for opponent in opponents for row in grouped[(opponent, arm)]]
        correctness = bool(
            all(bool(row["started_second"]) for row in arm_rows)
            and all(int(row["result"]) in (0, 1, 2) for row in arm_rows)
            and sum(int(row["exception_fallbacks"]) for row in arm_rows) == 0
        )
        gate = bool(
            arm != "control"
            and total_games >= 480
            and difference >= 0.03
            and interval[0] > 0.0
            and min(differences) >= -0.02
            and correctness
        )
        results.append(
            {
                "arm": arm,
                "games": total_games,
                "wins": total_wins,
                "equal_opponent_win_rate": float(np.mean([by_opponent[name]["win_rate"] for name in opponents])),
                "stratified_difference_vs_control": difference,
                "bootstrap_95": {"lower": interval[0], "median": interval[1], "upper": interval[2]},
                "worst_opponent_difference": min(differences),
                "correctness_pass": correctness,
                "development_gate_pass": gate,
                "by_opponent": by_opponent,
            }
        )
    eligible = sorted(
        (row for row in results if row["development_gate_pass"]),
        key=lambda row: (-float(row["stratified_difference_vs_control"]), -float(row["bootstrap_95"]["lower"]), str(row["arm"])),
    )
    advances = [str(eligible[0]["arm"])] if eligible else []
    report = {
        "schema": "rocket-blueprint-n5-development-analysis/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "manifests": [str(manifest_path(path.resolve())) for path in args.inputs],
        "arms_manifest_sha256": str(manifests[0]["arms_manifest_sha256"]),
        "rows": len(rows),
        "opponents": opponents,
        "arms": list(arms),
        "control": "control",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "locked_gates": {
            "minimum_games_per_arm": 480,
            "minimum_stratified_improvement": 0.03,
            "bootstrap_95_lower_strictly_positive": True,
            "minimum_worst_opponent_difference": -0.02,
            "strict_correctness": True,
        },
        "results": results,
        "advances_exactly_one_or_none": advances,
        "claim_boundary": "development selection evidence; not validation or arena strength evidence",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "advances": advances}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
