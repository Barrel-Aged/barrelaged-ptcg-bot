#!/usr/bin/env python3
"""Analyze the preregistered N5 two-seat, three-opponent regression panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


LOCKED_SEEDS = {791101, 791102, 791201, 791202, 791301, 791302}
LOCKED_SCHEDULE = {
    ("AshMarnieGeneralV110_extracted", 0, 791101),
    ("AshMarnieGeneralV110_extracted", 1, 791102),
    ("HelbrosMarnieGeneralRecoveryV4_extracted", 0, 791201),
    ("HelbrosMarnieGeneralRecoveryV4_extracted", 1, 791202),
    ("SparkRocketSpidopsV1_extracted", 0, 791301),
    ("SparkRocketSpidopsV1_extracted", 1, 791302),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--validation-analysis", type=Path, required=True)
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


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N5 regression analysis: {output}")
    validation = json.loads(args.validation_analysis.resolve().read_text(encoding="utf-8"))
    if validation.get("schema") != "rocket-blueprint-n5-validation-analysis/v1" or not validation.get("locked_validation_pass"):
        raise RuntimeError("N5 broader regression requires a passing locked validation")
    if len(args.inputs) != 6:
        raise ValueError("N5 broader regression requires exactly six shards")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    arms_hashes: set[str] = set()
    for raw in args.inputs:
        path = raw.resolve()
        manifest = json.loads(manifest_path(path).read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-blueprint-n5-panel-shard/v1" or manifest.get("split") != "regression":
            raise ValueError(f"unexpected N5 regression shard: {path}")
        if manifest.get("sealed") or not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
            raise ValueError(f"N5 regression shard lacks strict official execution: {path}")
        if int(manifest.get("games_per_arm", -1)) != 100 or int(manifest.get("rows", -1)) != 200:
            raise ValueError(f"N5 regression shard size changed: {path}")
        if sha256(path) != manifest.get("output_sha256"):
            raise ValueError(f"N5 regression shard changed after collection: {path}")
        arms_hashes.add(str(manifest.get("arms_manifest_sha256")))
        manifests.append(manifest)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "rocket-blueprint-n5-panel-match/v1" or row.get("split") != "regression":
                    raise ValueError(f"unexpected N5 regression row at {path}:{line_number}")
                rows.append(row)
    if len(arms_hashes) != 1 or {int(item["schedule_seed"]) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N5 regression arms or schedule seeds differ from lock")
    if len({str(item.get("candidate_bundle_sha256")) for item in manifests}) != 1:
        raise ValueError("N5 regression candidate runtime changed across shards")
    if validation.get("arms_manifest_sha256") != next(iter(arms_hashes)):
        raise ValueError("N5 regression frozen arms differ from locked validation")
    observed_schedule = {
        (Path(str(item["opponent"])).name, int(item["own_seat"]), int(item["schedule_seed"]))
        for item in manifests
    }
    if observed_schedule != LOCKED_SCHEDULE:
        raise ValueError(f"N5 regression opponent/seat/seed schedule differs from lock: {observed_schedule}")
    for opponent in {str(item["opponent"]) for item in manifests}:
        fingerprints = {
            (
                str(item.get("opponent_deck_sha256")),
                str(item.get("opponent_model_sha256")),
                str(item.get("opponent_main_sha256")),
                str(item.get("opponent_bundle_sha256")),
            )
            for item in manifests
            if str(item["opponent"]) == opponent
        }
        if len(fingerprints) != 1:
            raise ValueError(f"N5 regression opponent changed across seats: {opponent}")
    if len(rows) != 1200:
        raise ValueError(f"N5 regression requires 1200 rows, got {len(rows)}")

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), int(row["own_seat"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    seats = sorted({key[1] for key in grouped})
    arms = sorted({key[2] for key in grouped})
    if len(opponents) != 3 or seats != [0, 1] or arms != ["candidate", "control"]:
        raise ValueError(f"N5 regression shape changed: opponents={opponents}, seats={seats}, arms={arms}")

    cells: list[dict[str, Any]] = []
    for opponent in opponents:
        for seat in seats:
            candidate = grouped[(opponent, seat, "candidate")]
            control = grouped[(opponent, seat, "control")]
            if len(candidate) != 100 or len(control) != 100:
                raise ValueError(f"N5 regression cell size changed: {opponent}/seat{seat}")
            candidate_rate = float(np.mean([float(row["win"]) for row in candidate]))
            control_rate = float(np.mean([float(row["win"]) for row in control]))
            cells.append(
                {
                    "opponent": opponent,
                    "own_seat": seat,
                    "candidate_games": len(candidate),
                    "candidate_wins": sum(int(row["win"]) for row in candidate),
                    "candidate_win_rate": candidate_rate,
                    "control_games": len(control),
                    "control_wins": sum(int(row["win"]) for row in control),
                    "control_win_rate": control_rate,
                    "difference": candidate_rate - control_rate,
                }
            )
    differences = [float(cell["difference"]) for cell in cells]
    first_differences = [float(cell["difference"]) for cell in cells if int(cell["own_seat"]) == 0]
    correctness = bool(
        all(int(row["result"]) in (0, 1, 2) for row in rows)
        and all(int(row["first_player"]) == 0 for row in rows)
        and sum(int(row["exception_fallbacks"]) for row in rows) == 0
    )
    equal_cell_difference = float(np.mean(differences))
    first_seat_difference = float(np.mean(first_differences))
    passed = bool(
        equal_cell_difference >= 0.0
        and first_seat_difference >= -0.02
        and min(differences) >= -0.10
        and correctness
    )
    report = {
        "schema": "rocket-blueprint-n5-regression-analysis/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "validation_analysis": str(args.validation_analysis.resolve()),
        "validation_analysis_sha256": sha256(args.validation_analysis.resolve()),
        "validation_pass_confirmed": True,
        "arms_manifest_sha256": next(iter(arms_hashes)),
        "frozen_arms": manifests[0]["arms"],
        "inputs": [str(path.resolve()) for path in args.inputs],
        "rows": len(rows),
        "cells": cells,
        "equal_cell_difference_vs_control": equal_cell_difference,
        "first_seat_difference_vs_control": first_seat_difference,
        "worst_cell_difference_vs_control": min(differences),
        "correctness_pass": correctness,
        "broader_regression_pass": passed,
        "packaging_authorized_by_this_report": passed,
        "claim_boundary": "official local-engine regression evidence; not an arena rank claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "equal_cell_difference": equal_cell_difference, "first_seat_difference": first_seat_difference, "worst_cell_difference": min(differences), "regression_pass": passed}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
