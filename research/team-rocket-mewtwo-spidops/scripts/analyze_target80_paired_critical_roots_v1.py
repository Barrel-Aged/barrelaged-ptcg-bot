#!/usr/bin/env python3
"""Audit and gate the preregistered Target-80 paired critical-root pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "TARGET80_PAIRED_CRITICAL_ROOT_V1_PROTOCOL.md"
COLLECTOR_SHA256 = "4f6d26ff7aef0b79341e589c79321a883f1201a6a4b2faf8d6fe4539e92d94dd"
EXPECTED = {
    "first": {"roots": 8, "seed": 856101},
    "second": {"roots": 8, "seed": 856201},
}
FIXED_PARAMETERS = {
    "capture_roots_per_game": 8,
    "candidates": 3,
    "minimum_alternative_support": 0.02,
    "particles": 8,
    "coin_rollouts": 2,
    "particle_offset": 0,
    "earliest_turn": 2,
    "latest_turn": 10,
    "max_source_attempts": 200,
    "max_live_decisions": 2_000,
    "max_rollout_decisions": 600,
    "root_budget_seconds": 600.0,
    "threads": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def _best(
    samples: Sequence[Mapping[str, Any]],
    candidates: Sequence[int],
) -> int | None:
    if not samples:
        return None
    means = {
        index: _mean(float(row["values"][str(index)]) for row in samples)
        for index in candidates
    }
    return max(means, key=lambda index: (means[index], -index))


def _delta(
    samples: Sequence[Mapping[str, Any]],
    candidate: int,
    baseline: int,
) -> float:
    return _mean(
        float(row["values"][str(candidate)]) - float(row["values"][str(baseline)])
        for row in samples
    )


def validate_shard(path: Path, seat: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = load_json(manifest_path)
    rows = load_rows(path)
    if manifest.get("schema") != "target80-paired-critical-root-manifest/v1":
        raise ValueError(f"{seat}: wrong manifest schema")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"{seat}: wrong protocol")
    if manifest.get("collector_sha256") != COLLECTOR_SHA256:
        raise ValueError(f"{seat}: collector hash changed")
    if manifest.get("output_sha256") != sha256(path):
        raise ValueError(f"{seat}: output hash mismatch")
    parameters = manifest.get("parameters") or {}
    expected = EXPECTED[seat]
    if parameters.get("seat") != seat:
        raise ValueError(f"{seat}: seat mismatch")
    for key, value in {**FIXED_PARAMETERS, **expected}.items():
        if parameters.get(key) != value:
            raise ValueError(
                f"{seat}: parameter {key} changed: {parameters.get(key)!r} != {value!r}"
            )
    if len(rows) != expected["roots"]:
        raise ValueError(f"{seat}: expected {expected['roots']} roots, found {len(rows)}")
    counts = manifest.get("counts") or {}
    if counts.get("retained_roots") != expected["roots"]:
        raise ValueError(f"{seat}: manifest root count mismatch")
    if counts.get("fully_complete_roots") != expected["roots"]:
        raise ValueError(f"{seat}: incomplete root in manifest")
    if counts.get("rollout_errors", 0) != 0:
        raise ValueError(f"{seat}: rollout errors in manifest")
    for field in (
        "genuine_official_engine_source_matches",
        "genuine_official_engine_descendant_transitions",
        "source_policy_exact_packaged_public_guards",
        "common_random_coin_sequences",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"{seat}: required manifest claim {field} is false")
    for field in (
        "visualizer_or_realized_hidden_truth_read",
        "raw_hidden_allocations_serialized",
        "leaf_value_substitution",
        "sampled_rollout_is_final_strategy",
        "deployment_authority",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"{seat}: forbidden manifest claim {field} is true")
    return manifest, rows


def analyze_root(row: Mapping[str, Any], expected_seat: str) -> dict[str, Any]:
    if row.get("schema") != "target80-paired-critical-root/v1":
        raise ValueError("wrong row schema")
    if row.get("protocol") != PROTOCOL:
        raise ValueError("wrong row protocol")
    if bool(row.get("started_first")) != (expected_seat == "first"):
        raise ValueError("row start-order mismatch")
    if not row.get("guarded_baseline_reproduced"):
        raise ValueError("guarded baseline parity failed")
    if row.get("errors"):
        raise ValueError("root contains rollout errors")
    if not row.get("fully_complete") or int(row.get("complete_pairs") or 0) != 16:
        raise ValueError("root does not contain all 16 paired sets")
    candidates = [int(index) for index in row["candidate_indices"]]
    baseline = int(row["baseline"])
    if baseline != int(row["baseline_action"][0]) or baseline not in candidates:
        raise ValueError("baseline/candidate mismatch")
    if len(candidates) not in (2, 3):
        raise ValueError("root must expose two or three supported candidates")
    probabilities = [float(value) for value in row["policy_probabilities"]]
    for index in candidates:
        if index != baseline and probabilities[index] < 0.02 - 1e-9:
            raise ValueError("candidate violates minimum reference support")
    observation = row["observation"]
    current = observation.get("current") or {}
    players = current.get("players") or ()
    own_seat = int(row["own_seat"])
    if len(players) != 2 or (players[1 - own_seat] or {}).get("hand") is not None:
        raise ValueError("opponent hidden hand leaked into retained observation")
    if any(key.startswith("_completion") for key in row):
        raise ValueError("raw hidden completion field serialized")
    samples = row["paired_samples"]
    particles = sorted({int(sample["particle"]) for sample in samples})
    coins = sorted({int(sample["coin_rollout"]) for sample in samples})
    if particles != list(range(8)) or coins != [0, 1]:
        raise ValueError("unexpected particle or coin support")
    full_best = _best(samples, candidates)
    if full_best != int(row["best"]):
        raise ValueError("serialized best action does not reproduce")
    half_best = [
        _best([sample for sample in samples if int(sample["particle"]) in half], candidates)
        for half in (set(range(4)), set(range(4, 8)))
    ]
    coin_best = [
        _best([sample for sample in samples if int(sample["coin_rollout"]) == coin], candidates)
        for coin in coins
    ]
    if half_best != [int(value) for value in row["half_best"]]:
        raise ValueError("serialized particle-half ranking does not reproduce")
    stable_particle = half_best[0] == half_best[1] == full_best
    stable_coin = coin_best[0] == coin_best[1] == full_best
    mean_delta = _delta(samples, int(full_best), baseline)
    half_deltas = [
        _delta(
            [sample for sample in samples if int(sample["particle"]) in half],
            int(full_best),
            baseline,
        )
        for half in (set(range(4)), set(range(4, 8)))
    ]
    coin_deltas = [
        _delta(
            [sample for sample in samples if int(sample["coin_rollout"]) == coin],
            int(full_best),
            baseline,
        )
        for coin in coins
    ]
    stable_change = bool(
        full_best != baseline
        and stable_particle
        and stable_coin
        and mean_delta >= 0.125
        and min(half_deltas) >= 0.0
        and min(coin_deltas) >= 0.0
    )
    flags = [
        name for name, value in (row.get("criticality") or {}).get("flags", {}).items() if value
    ]
    return {
        "seat": expected_seat,
        "root_id": int(row["root_id"]),
        "turn": int(row["turn"]),
        "baseline": baseline,
        "best": full_best,
        "stable_particle": stable_particle,
        "stable_coin": stable_coin,
        "stable_change": stable_change,
        "mean_delta": mean_delta,
        "half_deltas": half_deltas,
        "coin_deltas": coin_deltas,
        "critical_flags": flags,
        "guard_name": row.get("guard_name"),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    manifests = {}
    analyses = []
    source_hashes = {}
    for seat, path in (("first", args.first.resolve()), ("second", args.second.resolve())):
        manifest, rows = validate_shard(path, seat)
        manifests[seat] = manifest
        source_hashes[seat] = sha256(path)
        analyses.extend(analyze_root(row, seat) for row in rows)

    total = len(analyses)
    stable_particle = sum(int(row["stable_particle"]) for row in analyses)
    stable_coin = sum(int(row["stable_coin"]) for row in analyses)
    stable_changes = [row for row in analyses if row["stable_change"]]
    seat_changes = Counter(row["seat"] for row in stable_changes)
    flag_counts = Counter(flag for row in analyses for flag in row["critical_flags"])
    stable_change_mean_delta = _mean(row["mean_delta"] for row in stable_changes)
    gates = {
        "all_16_roots_integrity_complete": total == 16,
        "particle_half_ranking_stability_at_least_75pct": stable_particle / total >= 0.75,
        "coin_stream_ranking_stability_at_least_75pct": stable_coin / total >= 0.75,
        "at_least_three_stable_nonbaseline_changes": len(stable_changes) >= 3,
        "stable_change_in_each_seat": all(seat_changes[seat] >= 1 for seat in EXPECTED),
        "stable_change_mean_delta_at_least_0_125": stable_change_mean_delta >= 0.125,
        "at_least_three_criticality_categories": len(flag_counts) >= 3,
    }
    passes = all(gates.values())
    payload = {
        "schema": "target80-paired-critical-root-analysis/v1",
        "protocol": PROTOCOL,
        "collector_sha256": COLLECTOR_SHA256,
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "source_sha256": source_hashes,
        "root_count": total,
        "particle_stable_count": stable_particle,
        "particle_stable_rate": stable_particle / total,
        "coin_stable_count": stable_coin,
        "coin_stable_rate": stable_coin / total,
        "stable_change_count": len(stable_changes),
        "stable_change_rate": len(stable_changes) / total,
        "stable_change_count_by_seat": dict(seat_changes),
        "stable_change_mean_delta": stable_change_mean_delta,
        "critical_flag_counts": dict(sorted(flag_counts.items())),
        "gates": gates,
        "passes_pilot": passes,
        "authority": (
            "expanded paired-root collection only" if passes else "none; close exact pilot"
        ),
        "deployment_authority": False,
        "rows": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
