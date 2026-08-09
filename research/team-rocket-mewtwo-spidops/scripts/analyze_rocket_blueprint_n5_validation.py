#!/usr/bin/env python3
"""Open and analyze exactly the preregistered 3,000-game N5 validation panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


UNLOCK_TOKEN = "N5_LOCKED_VALIDATION_3000"
LOCKED_SEEDS = {790101, 790102, 790201, 790202}
LOCKED_SCHEDULE = {
    ("AshMarnieGeneralV110_extracted", 790101),
    ("AshMarnieGeneralV110_extracted", 790102),
    ("HelbrosMarnieGeneralRecoveryV4_extracted", 790201),
    ("HelbrosMarnieGeneralRecoveryV4_extracted", 790202),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--unlock-token", required=True)
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=790901)
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


def bootstrap(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    opponents: list[str],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        differences = []
        for opponent in opponents:
            candidate = np.asarray([float(row["win"]) for row in grouped[(opponent, "candidate")]])
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
        raise FileExistsError(f"refusing existing N5 validation analysis: {output}")
    if args.unlock_token != UNLOCK_TOKEN:
        raise RuntimeError("incorrect N5 validation unlock token")
    if len(args.inputs) != 4 or args.bootstrap != 50_000 or args.seed != 790901:
        raise ValueError("N5 validation shard/bootstrap/seed configuration differs from lock")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    arms_hashes: set[str] = set()
    for raw in args.inputs:
        path = raw.resolve()
        sidecar = manifest_path(path)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if manifest.get("schema") != "rocket-blueprint-n5-panel-shard/v1":
            raise ValueError(f"unexpected N5 validation manifest: {sidecar}")
        if manifest.get("split") != "validation" or not manifest.get("sealed"):
            raise RuntimeError(f"N5 validation shard is not sealed: {path}")
        if not manifest.get("official_engine_execution") or not manifest.get("strict_no_fallback_execution"):
            raise ValueError(f"N5 validation was not strict official execution: {path}")
        if int(manifest.get("games_per_arm", -1)) != 375 or int(manifest.get("rows", -1)) != 750:
            raise ValueError(f"N5 validation shard size changed: {path}")
        if int(manifest.get("own_seat", -1)) != 1:
            raise ValueError(f"N5 validation did not keep Rocket second: {path}")
        if sha256(path) != manifest.get("output_sha256"):
            raise ValueError(f"N5 validation shard changed after collection: {path}")
        arms_hashes.add(str(manifest.get("arms_manifest_sha256")))
        manifests.append(manifest)
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != "rocket-blueprint-n5-panel-match/v1" or row.get("split") != "validation":
                    raise ValueError(f"unexpected N5 validation row at {path}:{line_number}")
                rows.append(row)
    if len(arms_hashes) != 1:
        raise ValueError("N5 validation shards use different frozen arms")
    if len({str(manifest.get("candidate_bundle_sha256")) for manifest in manifests}) != 1:
        raise ValueError("N5 validation candidate runtime changed across shards")
    if {int(manifest["schedule_seed"]) for manifest in manifests} != LOCKED_SEEDS:
        raise ValueError("N5 validation schedule seeds differ from lock")
    if {(Path(str(manifest["opponent"])).name, int(manifest["schedule_seed"])) for manifest in manifests} != LOCKED_SCHEDULE:
        raise ValueError("N5 validation opponent/seed pairing differs from lock")
    for opponent in {item[0] for item in LOCKED_SCHEDULE}:
        fingerprints = {
            (
                str(manifest.get("opponent_deck_sha256")),
                str(manifest.get("opponent_model_sha256")),
                str(manifest.get("opponent_main_sha256")),
                str(manifest.get("opponent_bundle_sha256")),
            )
            for manifest in manifests
            if Path(str(manifest["opponent"])).name == opponent
        }
        if len(fingerprints) != 1:
            raise ValueError(f"N5 validation opponent changed across shards: {opponent}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["opponent"]), str(row["arm"]))].append(row)
    opponents = sorted({key[0] for key in grouped})
    arms = sorted({key[1] for key in grouped})
    if len(rows) != 3000 or len(opponents) != 2 or arms != ["candidate", "control"]:
        raise ValueError(f"N5 validation panel shape changed: rows={len(rows)}, opponents={opponents}, arms={arms}")
    for opponent in opponents:
        for arm in arms:
            if len(grouped[(opponent, arm)]) != 750:
                raise ValueError(f"N5 validation requires 750 games for {opponent}/{arm}")

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
            "candidate_games": len(candidate),
            "candidate_wins": sum(int(row["win"]) for row in candidate),
            "candidate_win_rate": candidate_rate,
            "control_games": len(control),
            "control_wins": sum(int(row["win"]) for row in control),
            "control_win_rate": control_rate,
            "difference": difference,
        }
    rng = np.random.default_rng(args.seed)
    interval = bootstrap(grouped, opponents, args.bootstrap, rng)
    difference = float(np.mean(differences))
    correctness = bool(
        all(bool(row["started_second"]) for row in rows)
        and all(int(row["result"]) in (0, 1, 2) for row in rows)
        and sum(int(row["exception_fallbacks"]) for row in rows) == 0
    )
    passed = bool(
        difference >= 0.03
        and interval[0] > 0.0
        and min(differences) >= -0.02
        and correctness
    )
    candidate_rows = [row for opponent in opponents for row in grouped[(opponent, "candidate")]]
    control_rows = [row for opponent in opponents for row in grouped[(opponent, "control")]]
    report = {
        "schema": "rocket-blueprint-n5-validation-analysis/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "validation_opened_with_locked_analyzer": True,
        "inputs": [str(path.resolve()) for path in args.inputs],
        "manifests": [str(manifest_path(path.resolve())) for path in args.inputs],
        "arms_manifest_sha256": next(iter(arms_hashes)),
        "frozen_arms": manifests[0]["arms"],
        "rows": len(rows),
        "opponents": opponents,
        "candidate": {
            "games": len(candidate_rows),
            "wins": sum(int(row["win"]) for row in candidate_rows),
            "equal_opponent_win_rate": float(np.mean([by_opponent[name]["candidate_win_rate"] for name in opponents])),
        },
        "control": {
            "games": len(control_rows),
            "wins": sum(int(row["win"]) for row in control_rows),
            "equal_opponent_win_rate": float(np.mean([by_opponent[name]["control_win_rate"] for name in opponents])),
        },
        "by_opponent": by_opponent,
        "stratified_difference_vs_control": difference,
        "worst_opponent_difference": min(differences),
        "bootstrap_95": {"lower": interval[0], "median": interval[1], "upper": interval[2]},
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "correctness_pass": correctness,
        "locked_validation_pass": passed,
        "submission_authorized_by_this_report": passed,
        "claim_boundary": "official local-engine validation against two frozen opponents; not an arena rank claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "difference": difference, "bootstrap_95": report["bootstrap_95"], "validation_pass": passed}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
