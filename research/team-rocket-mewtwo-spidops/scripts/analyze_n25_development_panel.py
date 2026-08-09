#!/usr/bin/env python3
"""Apply the frozen N25 gate after all four shards are sealed."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

import analyze_belief_rollout_n6_panel as n6


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "artifacts/team_rocket_recovery/n25_complementary_safe_composition/development_panel"
INPUTS = (
    PANEL / "v110_01.jsonl",
    PANEL / "v110_02.jsonl",
    PANEL / "recovery_01.jsonl",
    PANEL / "recovery_02.jsonl",
)
OUTPUT = PANEL / "analysis.json"
ORDER = ("control", "candidate")
SEEDS = {825101, 825102, 825201, 825202}
OPPONENTS = {
    "AshMarnieGeneralV110_extracted",
    "HelbrosMarnieGeneralRecoveryV4_extracted",
}
ANALYSIS_SEED = 825001
BOOTSTRAP = 20_000

COLLECTOR_SHA = "e73612c3dda5fd784218ddfad365dd37daef57fd1eefdf2b16e4c7b01684401c"
BOOTSTRAP_SHA = "cabf1026ab21429ec11fb37b8e98760be3ef354a3303316c9f09d72340281df8"
ARMS_SHA = "defd9b9a02bbb221a8ef6d22b496d9dd37931870c8bb93001c32fdeaefe176a1"
BUNDLE_SHA = "38ce5eaed5074544bf89fb723789ca39258dd9d734bf714f9803f5235e60dcbd"
PARITY_SHA = "43556e15d5e783a0e2a1f14c2eb033b13ce0e7943281ecbd7222855d4341a602"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "e1d6567dfefee9c7b408b9438153e75e75f613864d7c0a2da021f7f5c63ebf04"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests = []
    for path in INPUTS:
        sidecar = path.with_suffix(".manifest.json")
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(f"N25 shard is not sealed: {path}")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "rocket-n25-development-panel-shard/v1"
            or manifest.get("protocol")
            != "rocket-n25-complementary-safe-composition/v1"
            or not manifest.get("official_engine_execution")
            or not manifest.get("strict_no_fallback_execution")
            or manifest.get("hidden_information_used_by_actor") is not False
        ):
            raise ValueError(f"invalid N25 manifest: {sidecar}")
        if digest(path) != manifest.get("output_sha256"):
            raise ValueError(f"N25 shard changed: {path}")
        shard_rows = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            if (
                row.get("schema") != "rocket-n25-development-panel-match/v1"
                or row.get("protocol")
                != "rocket-n25-complementary-safe-composition/v1"
            ):
                raise ValueError(f"invalid N25 row {path}:{line_number}")
            shard_rows.append(row)
        if [int(row["game_index"]) for row in shard_rows] != list(range(1, 241)):
            raise ValueError(f"N25 game indices changed: {path}")
        if any(
            int(row["schedule_seed"]) != int(manifest["schedule_seed"])
            for row in shard_rows
        ):
            raise ValueError(f"N25 row seed changed: {path}")
        rows.extend(shard_rows)
        manifests.append(manifest)
    return rows, manifests


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N25 analyzer accepts no overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N25 analysis: {OUTPUT}")
    if digest(Path(n6.__file__).resolve()) != BOOTSTRAP_SHA:
        raise ValueError("N25 bootstrap dependency changed")
    rows, manifests = load()
    if {int(manifest["schedule_seed"]) for manifest in manifests} != SEEDS:
        raise ValueError("N25 seeds differ from lock")
    if {Path(str(manifest["opponent"])).name for manifest in manifests} != OPPONENTS:
        raise ValueError("N25 opponents differ from lock")
    if any(
        int(manifest["games_per_arm"]) != 120
        or int(manifest["rows"]) != 240
        or manifest.get("counts") != {"candidate": 120, "control": 120}
        or manifest.get("collector_sha256") != COLLECTOR_SHA
        or manifest.get("arms_manifest_sha256") != ARMS_SHA
        or manifest.get("candidate_bundle_sha256") != BUNDLE_SHA
        or manifest.get("runtime_parity_sha256") != PARITY_SHA
        or manifest.get("deck_sha256") != DECK_SHA
        or manifest.get("control_model_sha256") != CONTROL_SHA
        or manifest.get("candidate_model_sha256") != CANDIDATE_SHA
        for manifest in manifests
    ):
        raise ValueError("N25 manifest provenance/counts differ from lock")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    if set(opponents) != OPPONENTS:
        raise ValueError("N25 row opponents differ from lock")
    if tuple(sorted({key[1] for key in grouped}, key=ORDER.index)) != ORDER:
        raise ValueError("N25 arms differ from lock")
    if any(
        len(grouped[(opponent, arm)]) != 240
        for opponent in opponents
        for arm in ORDER
    ):
        raise ValueError("N25 aggregate cell count changed")
    hashes = {"control": CONTROL_SHA, "candidate": CANDIDATE_SHA}
    correctness = bool(
        len(rows) == 960
        and all(row.get("started_second") for row in rows)
        and all(int(row["result"]) in (0, 1, 2) for row in rows)
        and all(
            str(row["arm_sha256"]) == hashes[str(row["arm"])] for row in rows
        )
        and sum(int(row.get("exception_fallbacks", 0)) for row in rows) == 0
    )
    cells = {}
    differences = []
    for opponent in opponents:
        candidate = grouped[(opponent, "candidate")]
        control = grouped[(opponent, "control")]
        candidate_rate = float(np.mean([float(row["win"]) for row in candidate]))
        control_rate = float(np.mean([float(row["win"]) for row in control]))
        difference = candidate_rate - control_rate
        differences.append(difference)
        cells[opponent] = {
            "games_per_arm": 240,
            "candidate_wins": sum(int(row["win"]) for row in candidate),
            "candidate_win_rate": candidate_rate,
            "control_wins": sum(int(row["win"]) for row in control),
            "control_win_rate": control_rate,
            "difference": difference,
        }
    interval = n6.bootstrap_difference(
        grouped,
        opponents,
        "candidate",
        BOOTSTRAP,
        np.random.default_rng(ANALYSIS_SEED),
    )
    aggregate = float(np.mean(differences))
    passed = bool(
        correctness
        and aggregate >= 0.030
        and min(differences) > 0.0
        and interval[0] > -0.005
    )
    report = {
        "schema": "rocket-n25-development-panel-analysis/v1",
        "protocol": "rocket-n25-complementary-safe-composition/v1",
        "inputs": [str(path.resolve()) for path in INPUTS],
        "input_sha256": {
            str(path.resolve()): digest(path.resolve()) for path in INPUTS
        },
        "arms_manifest_sha256": ARMS_SHA,
        "candidate_bundle_sha256": BUNDLE_SHA,
        "runtime_parity_sha256": PARITY_SHA,
        "collector_sha256": COLLECTOR_SHA,
        "analyzer_sha256": digest(Path(__file__).resolve()),
        "bootstrap_dependency_sha256": BOOTSTRAP_SHA,
        "rows": len(rows),
        "games_per_arm": 480,
        "opponents": opponents,
        "bootstrap_replicates": BOOTSTRAP,
        "bootstrap_seed": ANALYSIS_SEED,
        "correctness_pass": correctness,
        "by_opponent": cells,
        "stratified_difference_vs_control": aggregate,
        "bootstrap_95": {
            "lower": interval[0],
            "median": interval[1],
            "upper": interval[2],
        },
        "worst_opponent_difference": min(differences),
        "locked_gates": {
            "minimum_total_rows": 960,
            "minimum_games_per_arm_opponent_cell": 240,
            "minimum_stratified_improvement": 0.030,
            "minimum_worst_opponent_difference_strict": 0.0,
            "bootstrap_95_lower_strict": -0.005,
            "strict_correctness": True,
        },
        "development_gate_pass": passed,
        "advances_to_fresh_confirmation": ["candidate"] if passed else [],
        "deployment_authority": False,
        "strength_improvement_claim": False,
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "rows": len(rows),
                "development_gate_pass": passed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
