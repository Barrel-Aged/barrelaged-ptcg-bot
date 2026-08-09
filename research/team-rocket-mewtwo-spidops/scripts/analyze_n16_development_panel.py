#!/usr/bin/env python3
"""Apply the frozen N16 gate after all four shards are sealed."""

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
PANEL = ROOT / "artifacts/team_rocket_recovery/n16_minimum_scale_projection/development_panel"
INPUTS = (
    PANEL / "v110_01.jsonl",
    PANEL / "v110_02.jsonl",
    PANEL / "recovery_01.jsonl",
    PANEL / "recovery_02.jsonl",
)
OUTPUT = PANEL / "analysis.json"
ORDER = ("control", "candidate")
SEEDS = {816101, 816102, 816201, 816202}
OPPONENTS = {"AshMarnieGeneralV110_extracted", "HelbrosMarnieGeneralRecoveryV4_extracted"}
ANALYSIS_SEED = 816001
BOOTSTRAP = 20_000

COLLECTOR_SHA = "6895b48af08c825354efae65c826e0b8c96e0f493e4bd89e602990dfc9068dd7"
BOOTSTRAP_SHA = "cabf1026ab21429ec11fb37b8e98760be3ef354a3303316c9f09d72340281df8"
ARMS_SHA = "3cb430f6e281004b1671d0df3941e6a14386278d789c2e153b51221fe6e4bf72"
BUNDLE_SHA = "d4732ea468b9046abffd808937935963cba4091f180ae72704d03e66a6bfcf02"
PARITY_SHA = "0a8bad5cc2e6f231ecd5eba66d8b9b3880c43ab22680cebade2f7630f1068a34"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "b4f35c7816097800b0c7265691f5f650bc9b467a4258612930d0fef07afcbcf7"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    manifests = []
    for path in INPUTS:
        sidecar = path.with_suffix(".manifest.json")
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(f"N16 shard is not sealed: {path}")
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "rocket-n16-development-panel-shard/v1"
            or manifest.get("protocol") != "rocket-n16-minimum-scale-residual-projection/v1"
            or not manifest.get("official_engine_execution")
            or not manifest.get("strict_no_fallback_execution")
            or manifest.get("hidden_information_used_by_actor") is not False
        ):
            raise ValueError(f"invalid N16 manifest: {sidecar}")
        if digest(path) != manifest.get("output_sha256"):
            raise ValueError(f"N16 shard changed: {path}")
        shard_rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if (
                row.get("schema") != "rocket-n16-development-panel-match/v1"
                or row.get("protocol") != "rocket-n16-minimum-scale-residual-projection/v1"
            ):
                raise ValueError(f"invalid N16 row {path}:{line_number}")
            shard_rows.append(row)
        if [int(row["game_index"]) for row in shard_rows] != list(range(1, 241)):
            raise ValueError(f"N16 game indices changed: {path}")
        if any(int(row["schedule_seed"]) != int(manifest["schedule_seed"]) for row in shard_rows):
            raise ValueError(f"N16 row seed changed: {path}")
        rows.extend(shard_rows)
        manifests.append(manifest)
    return rows, manifests


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N16 analyzer accepts no overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N16 analysis: {OUTPUT}")
    if digest(Path(n6.__file__).resolve()) != BOOTSTRAP_SHA:
        raise ValueError("N16 bootstrap dependency changed")
    rows, manifests = load()
    if {int(m["schedule_seed"]) for m in manifests} != SEEDS:
        raise ValueError("N16 seeds differ from lock")
    if {Path(str(m["opponent"])).name for m in manifests} != OPPONENTS:
        raise ValueError("N16 opponents differ from lock")
    if any(
        int(m["games_per_arm"]) != 120
        or int(m["rows"]) != 240
        or m.get("counts") != {"candidate": 120, "control": 120}
        or m.get("collector_sha256") != COLLECTOR_SHA
        or m.get("arms_manifest_sha256") != ARMS_SHA
        or m.get("candidate_bundle_sha256") != BUNDLE_SHA
        or m.get("runtime_parity_sha256") != PARITY_SHA
        or m.get("deck_sha256") != DECK_SHA
        or m.get("control_model_sha256") != CONTROL_SHA
        or m.get("candidate_model_sha256") != CANDIDATE_SHA
        for m in manifests
    ):
        raise ValueError("N16 manifest provenance/counts differ from lock")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    if set(opponents) != OPPONENTS:
        raise ValueError("N16 row opponents differ from lock")
    if tuple(sorted({key[1] for key in grouped}, key=ORDER.index)) != ORDER:
        raise ValueError("N16 arms differ from lock")
    if any(len(grouped[(opponent, arm)]) != 240 for opponent in opponents for arm in ORDER):
        raise ValueError("N16 aggregate cell count changed")
    hashes = {"control": CONTROL_SHA, "candidate": CANDIDATE_SHA}
    correctness = bool(
        len(rows) == 960
        and all(row.get("started_second") for row in rows)
        and all(int(row["result"]) in (0, 1, 2) for row in rows)
        and all(str(row["arm_sha256"]) == hashes[str(row["arm"])] for row in rows)
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
        grouped, opponents, "candidate", BOOTSTRAP, np.random.default_rng(ANALYSIS_SEED)
    )
    aggregate = float(np.mean(differences))
    passed = bool(
        correctness and aggregate >= 0.020 and min(differences) > 0.0 and interval[0] > -0.005
    )
    report = {
        "schema": "rocket-n16-development-panel-analysis/v1",
        "protocol": "rocket-n16-minimum-scale-residual-projection/v1",
        "inputs": [str(path.resolve()) for path in INPUTS],
        "input_sha256": {str(path.resolve()): digest(path.resolve()) for path in INPUTS},
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
        "bootstrap_95": {"lower": interval[0], "median": interval[1], "upper": interval[2]},
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
    print(json.dumps({"output": str(OUTPUT), "rows": len(rows), "development_gate_pass": passed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
