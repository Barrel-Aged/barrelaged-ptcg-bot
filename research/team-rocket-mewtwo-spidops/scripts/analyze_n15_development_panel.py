#!/usr/bin/env python3
"""Apply the locked N15 gate after all four development shards are sealed."""

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
PANEL = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/development_panel"
INPUTS = (
    PANEL / "v110_01.jsonl",
    PANEL / "v110_02.jsonl",
    PANEL / "recovery_01.jsonl",
    PANEL / "recovery_02.jsonl",
)
OUTPUT = PANEL / "analysis.json"
ORDER = ("control", "candidate")
LOCKED_SEEDS = {815101, 815102, 815201, 815202}
LOCKED_OPPONENTS = {
    "AshMarnieGeneralV110_extracted",
    "HelbrosMarnieGeneralRecoveryV4_extracted",
}
ANALYSIS_SEED = 815002
BOOTSTRAP_REPLICATES = 20_000

COLLECTOR_SHA = "e90626b37c892301dc4d283afa8335c05a96c8c7610ce46a0e1c41cbf608d7bb"
BOOTSTRAP_DEPENDENCY_SHA = "cabf1026ab21429ec11fb37b8e98760be3ef354a3303316c9f09d72340281df8"
ARMS_SHA = "e46197852d7e88d069b54d9ecc5cc978b5d0d5b0c731cb34409a9d88c1fa34ef"
BUNDLE_SHA = "982fa183a4ed963bda201c5dd01b55cf316e37848797441ef34d4848e0c7ff74"
PARITY_SHA = "95f6d3f74883baf73395a2504772ac014607659bf5a9fe0a9060b37b57cf8ca8"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in INPUTS:
        sidecar = path.with_suffix(".manifest.json")
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(f"N15 shard is not sealed: {path}")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "rocket-n15-development-panel-shard/v1"
            or manifest.get("protocol")
            != "rocket-n15-prospective-terminal-consolidation/v1"
        ):
            raise ValueError(f"unexpected N15 manifest: {sidecar}")
        if (
            not manifest.get("official_engine_execution")
            or not manifest.get("strict_no_fallback_execution")
            or manifest.get("hidden_information_used_by_actor") is not False
        ):
            raise ValueError(f"N15 shard is not strict official execution: {path}")
        if digest(path) != manifest.get("output_sha256"):
            raise ValueError(f"N15 shard changed after collection: {path}")
        manifests.append(manifest)
        shard_rows = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            row = json.loads(line)
            if (
                row.get("schema") != "rocket-n15-development-panel-match/v1"
                or row.get("protocol")
                != "rocket-n15-prospective-terminal-consolidation/v1"
            ):
                raise ValueError(f"unexpected N15 row {path}:{line_number}")
            shard_rows.append(row)
        if [int(row["game_index"]) for row in shard_rows] != list(range(1, 241)):
            raise ValueError(f"N15 game indices changed: {path}")
        if any(int(row["schedule_seed"]) != int(manifest["schedule_seed"]) for row in shard_rows):
            raise ValueError(f"N15 row seed differs from manifest: {path}")
        rows.extend(shard_rows)
    return rows, manifests


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N15 analyzer accepts no command-line overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N15 analysis: {OUTPUT}")
    if digest(Path(n6.__file__).resolve()) != BOOTSTRAP_DEPENDENCY_SHA:
        raise ValueError("N15 bootstrap dependency changed")
    rows, manifests = load()
    if {int(item["schedule_seed"]) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N15 shard seeds differ from lock")
    if {Path(str(item["opponent"])).name for item in manifests} != LOCKED_OPPONENTS:
        raise ValueError("N15 opponents differ from lock")
    if any(
        int(item["games_per_arm"]) != 120
        or int(item["rows"]) != 240
        or item.get("counts") != {"candidate": 120, "control": 120}
        for item in manifests
    ):
        raise ValueError("N15 shard sizes differ from lock")
    if any(item.get("collector_sha256") != COLLECTOR_SHA for item in manifests):
        raise ValueError("N15 collector hash differs from lock")
    if any(item.get("arms_manifest_sha256") != ARMS_SHA for item in manifests):
        raise ValueError("N15 arm manifest differs from lock")
    if any(item.get("candidate_bundle_sha256") != BUNDLE_SHA for item in manifests):
        raise ValueError("N15 candidate bundle differs from lock")
    if any(item.get("runtime_parity_sha256") != PARITY_SHA for item in manifests):
        raise ValueError("N15 runtime parity differs from lock")
    if any(
        item.get("deck_sha256") != DECK_SHA
        or item.get("control_model_sha256") != CONTROL_SHA
        or item.get("candidate_model_sha256") != CANDIDATE_SHA
        for item in manifests
    ):
        raise ValueError("N15 model or deck provenance differs from lock")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    if set(opponents) != LOCKED_OPPONENTS:
        raise ValueError("N15 row opponents differ from lock")
    if tuple(sorted({key[1] for key in grouped}, key=ORDER.index)) != ORDER:
        raise ValueError("N15 row arms differ from lock")
    for opponent in opponents:
        if any(len(grouped[(opponent, arm)]) != 240 for arm in ORDER):
            raise ValueError(f"N15 aggregate cell count mismatch: {opponent}")

    arm_hashes = {"control": CONTROL_SHA, "candidate": CANDIDATE_SHA}
    correctness = bool(
        len(rows) == 960
        and all(row.get("started_second") for row in rows)
        and all(int(row["result"]) in (0, 1, 2) for row in rows)
        and all(str(row.get("arm_sha256")) == arm_hashes[str(row["arm"])] for row in rows)
        and sum(int(row.get("exception_fallbacks", 0)) for row in rows) == 0
    )
    by_opponent: dict[str, Any] = {}
    differences = []
    for opponent in opponents:
        candidate = grouped[(opponent, "candidate")]
        control = grouped[(opponent, "control")]
        candidate_rate = float(np.mean([float(row["win"]) for row in candidate]))
        control_rate = float(np.mean([float(row["win"]) for row in control]))
        difference = candidate_rate - control_rate
        differences.append(difference)
        by_opponent[opponent] = {
            "games_per_arm": 240,
            "candidate_wins": sum(int(row["win"]) for row in candidate),
            "candidate_win_rate": candidate_rate,
            "control_wins": sum(int(row["win"]) for row in control),
            "control_win_rate": control_rate,
            "difference": difference,
        }
    rng = np.random.default_rng(ANALYSIS_SEED)
    interval = n6.bootstrap_difference(
        grouped, opponents, "candidate", BOOTSTRAP_REPLICATES, rng
    )
    aggregate = float(np.mean(differences))
    passed = bool(
        correctness
        and aggregate >= 0.020
        and min(differences) > 0.0
        and interval[0] > -0.005
    )
    report = {
        "schema": "rocket-n15-development-panel-analysis/v1",
        "protocol": "rocket-n15-prospective-terminal-consolidation/v1",
        "inputs": [str(path.resolve()) for path in INPUTS],
        "input_sha256": {str(path.resolve()): digest(path.resolve()) for path in INPUTS},
        "arms_manifest_sha256": ARMS_SHA,
        "candidate_bundle_sha256": BUNDLE_SHA,
        "runtime_parity_sha256": PARITY_SHA,
        "collector_sha256": COLLECTOR_SHA,
        "analyzer_sha256": digest(Path(__file__).resolve()),
        "bootstrap_dependency_sha256": BOOTSTRAP_DEPENDENCY_SHA,
        "rows": len(rows),
        "games_per_arm": 480,
        "opponents": opponents,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": ANALYSIS_SEED,
        "correctness_pass": correctness,
        "by_opponent": by_opponent,
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
            "minimum_stratified_improvement": 0.020,
            "minimum_worst_opponent_difference_strict": 0.0,
            "bootstrap_95_lower_strict": -0.005,
            "strict_correctness": True,
        },
        "development_gate_pass": passed,
        "advances_to_fresh_confirmation": ["candidate"] if passed else [],
        "deployment_authority": False,
        "strength_improvement_claim": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
