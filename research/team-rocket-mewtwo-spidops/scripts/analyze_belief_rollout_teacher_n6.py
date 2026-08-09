#!/usr/bin/env python3
"""Analyze frozen N6 paired rollout shards with particle-cluster uncertainty."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.n6_teacher import estimate_root  # noqa: E402
from rocketbot.policy_feature_snapshot import deserialize_policy_feature  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=790401)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite N6 analysis: {args.output}")
    records: list[tuple[Path, dict[str, Any]]] = []
    input_hashes = {}
    manifest_hashes = {}
    for raw_path in args.input:
        path = raw_path.resolve()
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") not in {
            "rocket-n6-belief-rollout-manifest/v1",
            "rocket-n6r-belief-rollout-manifest/v2",
        }:
            raise ValueError(f"{manifest_path}: unexpected N6 manifest schema")
        if not manifest.get("genuine_official_engine_descendant_transitions"):
            raise ValueError(f"{manifest_path}: descendant transitions are not official")
        if manifest.get("visualizer_or_realized_hidden_truth_read"):
            raise ValueError(f"{manifest_path}: realized hidden truth was read")
        if manifest.get("raw_hidden_allocations_serialized"):
            raise ValueError(f"{manifest_path}: raw hidden allocations were serialized")
        path_hash = sha256(path)
        if manifest.get("output_sha256") != path_hash:
            raise ValueError(f"{path}: content hash differs from its N6 manifest")
        input_hashes[str(path)] = path_hash
        manifest_hashes[str(manifest_path)] = sha256(manifest_path)
        file_records = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("schema") not in {
                "rocket-n6-belief-rollout-root/v1",
                "rocket-n6r-belief-rollout-root/v2",
            }:
                raise ValueError(f"{path}:{line_number}: unexpected schema")
            if row.get("schema") == "rocket-n6r-belief-rollout-root/v2":
                if not row.get("policy_feature_baseline_reproduced"):
                    raise ValueError(
                        f"{path}:{line_number}: N6R baseline feature parity is absent"
                    )
                deserialize_policy_feature(row.get("policy_feature") or {})
            if not row.get("fully_complete"):
                raise ValueError(f"{path}:{line_number}: incomplete N6 root")
            if int(row.get("complete_pairs", -1)) != 24 or int(
                row.get("attempted_pairs", -1)
            ) != 24:
                raise ValueError(f"{path}:{line_number}: expected 24/24 paired samples")
            if len(row.get("candidate_indices") or ()) != 3:
                raise ValueError(f"{path}:{line_number}: expected three candidates")
            if len(row.get("particle_manifests") or ()) != 12:
                raise ValueError(f"{path}:{line_number}: expected 12 particle manifests")
            if row.get("errors"):
                raise ValueError(f"{path}:{line_number}: rollout errors are present")
            records.append((path, row))
            file_records += 1
        if file_records != int((manifest.get("counts") or {}).get("roots", -1)):
            raise ValueError(f"{path}: row count differs from its N6 manifest")
    if not records:
        raise ValueError("N6 analysis received no roots")
    input_schemas = {str(row["schema"]) for _, row in records}
    if len(input_schemas) != 1:
        raise ValueError(f"N6/N6R input schemas are mixed: {sorted(input_schemas)}")
    input_schema = next(iter(input_schemas))
    protocol = (
        "N6R_EXACT_FEATURE_BELIEF_ROLLOUT_PROTOCOL.md"
        if input_schema == "rocket-n6r-belief-rollout-root/v2"
        else "N6_BELIEF_ROLLOUT_DISTILLATION_PROTOCOL.md"
    )

    counts: Counter[str] = Counter()
    by_opponent: dict[str, Counter[str]] = defaultdict(Counter)
    estimates = []
    seen = set()
    oracle_gains = []
    admitted_gains = []
    for index, (path, row) in enumerate(records):
        key = (str(path), int(row["game_id"]), int(row["root_index"]))
        if key in seen:
            raise ValueError(f"duplicate N6 root: {key}")
        seen.add(key)
        estimate = estimate_root(
            row,
            seed=args.seed + 100_003 * index,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        opponent = str(row["opponent"])
        baseline_mean = estimate.action(estimate.baseline).mean
        best_mean = estimate.action(estimate.best).mean
        oracle_gain = best_mean - baseline_mean
        oracle_gains.append(oracle_gain)
        admitted_gain = oracle_gain if estimate.admissible_change else 0.0
        admitted_gains.append(admitted_gain)
        counts["roots"] += 1
        counts["fully_complete"] += int(bool(row.get("fully_complete")))
        counts["nominal_changes"] += int(estimate.best != estimate.baseline)
        counts["half_best_agreement"] += int(
            estimate.half_best[0] == estimate.half_best[1]
        )
        counts["admissible_changes"] += int(estimate.admissible_change)
        opponent_counts = by_opponent[opponent]
        opponent_counts["roots"] += 1
        opponent_counts["nominal_changes"] += int(estimate.best != estimate.baseline)
        opponent_counts["admissible_changes"] += int(estimate.admissible_change)
        estimates.append(
            {
                "source": str(path),
                "game_id": int(row["game_id"]),
                "root_index": int(row["root_index"]),
                "opponent": opponent,
                "turn": int(row["turn"]),
                "observation_fingerprint": row["observation_fingerprint"],
                "oracle_gain": oracle_gain,
                "admitted_gain": admitted_gain,
                **asdict(estimate),
            }
        )

    report = {
        "schema": "rocket-n6-belief-rollout-analysis/v1",
        "protocol": protocol,
        "input_root_schema": input_schema,
        "inputs": input_hashes,
        "input_manifests": manifest_hashes,
        "parameters": {
            "seed": args.seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "minimum_complete_pairs": 22,
            "strict_positive_clustered_lower_bound": True,
        },
        "counts": dict(counts),
        "rates": {
            "fully_complete": counts["fully_complete"] / counts["roots"],
            "nominal_change": counts["nominal_changes"] / counts["roots"],
            "half_best_agreement": counts["half_best_agreement"] / counts["roots"],
            "admissible_change": counts["admissible_changes"] / counts["roots"],
        },
        "mean_nominal_oracle_gain": statistics.fmean(oracle_gains),
        "mean_admitted_gain_per_root": statistics.fmean(admitted_gains),
        "by_opponent": {name: dict(value) for name, value in sorted(by_opponent.items())},
        "roots": estimates,
        "claims": {
            "official_descendant_execution": True,
            "hidden_completion_prior_official": False,
            "equilibrium_certificate": False,
            "arena_strength_improvement": False,
            "deployment_authority": False,
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "roots": counts["roots"],
                "admissible_changes": counts["admissible_changes"],
                "half_best_agreement": report["rates"]["half_best_agreement"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
