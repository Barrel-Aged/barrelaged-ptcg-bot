#!/usr/bin/env python3
"""Audit and gate the Target-80 paired strategic-plan milestone pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "TARGET80_PAIRED_MILESTONE_ROOT_V2_PROTOCOL.md"
COLLECTOR_SHA256 = "8e153f1baceb240ac522275aea869e170ce9bfdd2c0ad25ae4bb4118c068da17"
PLANS = ("baseline", "engine_resilience", "tempo_conversion")
PLAN_ORDER = {name: index for index, name in enumerate(PLANS)}
AXES = (
    "setup",
    "tempo",
    "resource_efficiency",
    "prize_control",
    "board_control",
    "optionality",
)
EXPECTED = {
    "first": {"roots": 6, "seed": 857101},
    "second": {"roots": 6, "seed": 857201},
}
FIXED_PARAMETERS = {
    "capture_roots_per_game": 8,
    "candidates": 3,
    "minimum_plan_action_support": 0.01,
    "particles": 8,
    "coin_rollouts": 2,
    "particle_offset": 0,
    "earliest_turn": 2,
    "latest_turn": 8,
    "horizon_turns": 2,
    "max_source_attempts": 200,
    "max_live_decisions": 2_000,
    "max_rollout_decisions": 250,
    "root_budget_seconds": 300.0,
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


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def best_plan(
    samples: Sequence[Mapping[str, Any]],
    *,
    particles: set[int] | None = None,
    coin: int | None = None,
) -> str | None:
    selected = [
        row
        for row in samples
        if (particles is None or int(row["particle"]) in particles)
        and (coin is None or int(row["coin_rollout"]) == coin)
    ]
    if not selected:
        return None
    means = {
        name: mean(float(row["values"][name]) for row in selected)
        for name in PLANS
    }
    return max(PLANS, key=lambda name: (means[name], -PLAN_ORDER[name]))


def delta(
    samples: Sequence[Mapping[str, Any]],
    plan: str,
    *,
    particles: set[int] | None = None,
    coin: int | None = None,
) -> float:
    selected = [
        row
        for row in samples
        if (particles is None or int(row["particle"]) in particles)
        and (coin is None or int(row["coin_rollout"]) == coin)
    ]
    return mean(
        float(row["values"][plan]) - float(row["values"]["baseline"])
        for row in selected
    )


def validate_shard(path: Path, seat: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = load_json(path.with_suffix(".manifest.json"))
    rows = load_rows(path)
    if manifest.get("schema") != "target80-paired-milestone-root-manifest/v2":
        raise ValueError(f"{seat}: wrong manifest schema")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"{seat}: wrong protocol")
    if manifest.get("collector_sha256") != COLLECTOR_SHA256:
        raise ValueError(f"{seat}: collector hash changed")
    if manifest.get("output_sha256") != sha256(path):
        raise ValueError(f"{seat}: output hash mismatch")
    parameters = manifest.get("parameters") or {}
    if parameters.get("seat") != seat:
        raise ValueError(f"{seat}: wrong seat")
    for key, value in {**FIXED_PARAMETERS, **EXPECTED[seat]}.items():
        if parameters.get(key) != value:
            raise ValueError(
                f"{seat}: parameter {key} changed: {parameters.get(key)!r} != {value!r}"
            )
    if len(rows) != EXPECTED[seat]["roots"]:
        raise ValueError(f"{seat}: wrong row count")
    counts = manifest.get("counts") or {}
    if counts.get("retained_roots") != EXPECTED[seat]["roots"]:
        raise ValueError(f"{seat}: manifest root count mismatch")
    if counts.get("fully_complete_roots") != EXPECTED[seat]["roots"]:
        raise ValueError(f"{seat}: incomplete roots")
    if counts.get("rollout_errors", 0) != 0:
        raise ValueError(f"{seat}: rollout errors")
    if tuple(manifest.get("strategic_axes") or ()) != AXES:
        raise ValueError(f"{seat}: strategic axes changed")
    if tuple((manifest.get("strategic_plans") or {}).keys()) != PLANS:
        raise ValueError(f"{seat}: strategic plans changed")
    if manifest.get("plan_max_abs_logit_bias") != 1.5:
        raise ValueError(f"{seat}: plan bias changed")
    for field in (
        "genuine_official_engine_source_matches",
        "genuine_official_engine_descendant_transitions",
        "source_policy_exact_packaged_public_guards",
        "common_random_coin_sequences",
        "plan_persists_through_horizon",
        "terminal_value_dominates_dense_score",
    ):
        if manifest.get(field) is not True:
            raise ValueError(f"{seat}: required claim {field} is false")
    for field in (
        "visualizer_or_realized_hidden_truth_read",
        "raw_hidden_allocations_serialized",
        "leaf_value_substitution",
        "sampled_rollout_is_final_strategy",
        "deployment_authority",
    ):
        if manifest.get(field) is not False:
            raise ValueError(f"{seat}: forbidden claim {field} is true")
    return manifest, rows


def analyze_root(row: Mapping[str, Any], seat: str) -> dict[str, Any]:
    if row.get("schema") != "target80-paired-milestone-root/v2":
        raise ValueError("wrong row schema")
    if row.get("protocol") != PROTOCOL:
        raise ValueError("wrong row protocol")
    if bool(row.get("started_first")) != (seat == "first"):
        raise ValueError("start-order mismatch")
    if not row.get("guarded_baseline_reproduced"):
        raise ValueError("guarded baseline parity failed")
    if tuple(row.get("plan_names") or ()) != PLANS:
        raise ValueError("plan names changed")
    plan_actions = row.get("plan_actions") or {}
    if set(plan_actions) != set(PLANS):
        raise ValueError("plan action set mismatch")
    if len({tuple(action) for action in plan_actions.values()}) < 2:
        raise ValueError("retained root lacks two distinct plan actions")
    if [int(index) for index in plan_actions["baseline"]] != [
        int(index) for index in row["baseline_action"]
    ]:
        raise ValueError("baseline plan action mismatch")
    probabilities = [float(value) for value in row["policy_probabilities"]]
    for name, action in plan_actions.items():
        if name != "baseline" and probabilities[int(action[0])] < 0.01 - 1e-9:
            raise ValueError("plan root action violates support floor")
    if row.get("errors") or not row.get("fully_complete"):
        raise ValueError("root incomplete or contains errors")
    if int(row.get("complete_pairs") or 0) != 16:
        raise ValueError("root lacks 16 paired sets")
    weights = {name: float(value) for name, value in (row.get("meta_weights") or {}).items()}
    if set(weights) != set(AXES) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("invalid meta weights")
    observation = row["observation"]
    current = observation.get("current") or {}
    players = current.get("players") or ()
    own_seat = int(row["own_seat"])
    if len(players) != 2 or (players[1 - own_seat] or {}).get("hand") is not None:
        raise ValueError("hidden opponent hand leaked")
    samples = row["paired_samples"]
    particles = sorted({int(sample["particle"]) for sample in samples})
    coins = sorted({int(sample["coin_rollout"]) for sample in samples})
    if particles != list(range(8)) or coins != [0, 1]:
        raise ValueError("particle/coin support changed")
    for sample in samples:
        if set(sample["values"]) != set(PLANS):
            raise ValueError("sample plan set mismatch")
        for plan in PLANS:
            axes = {name: float(value) for name, value in sample["axes"][plan].items()}
            if set(axes) != set(AXES) or any(not -1.000001 <= value <= 1.000001 for value in axes.values()):
                raise ValueError("invalid strategic axes")
            value = float(sample["values"][plan])
            if sample["terminals"][plan]:
                expected = 4.0 * float(sample["terminal_utilities"][plan])
            else:
                expected = sum(weights[name] * axes[name] for name in AXES)
            if abs(value - expected) > 1e-8:
                raise ValueError("milestone aggregate does not reproduce")
    full_best = best_plan(samples)
    if full_best != row["best"]:
        raise ValueError("full best plan does not reproduce")
    half_best = [
        best_plan(samples, particles=set(range(4))),
        best_plan(samples, particles=set(range(4, 8))),
    ]
    if half_best != list(row["half_best"]):
        raise ValueError("particle-half best plans do not reproduce")
    coin_best = [best_plan(samples, coin=coin) for coin in coins]
    stable_particle = half_best[0] == half_best[1] == full_best
    stable_coin = coin_best[0] == coin_best[1] == full_best
    mean_delta = delta(samples, str(full_best))
    half_deltas = [
        delta(samples, str(full_best), particles=set(range(4))),
        delta(samples, str(full_best), particles=set(range(4, 8))),
    ]
    coin_deltas = [delta(samples, str(full_best), coin=coin) for coin in coins]
    stable_change = bool(
        full_best != "baseline"
        and stable_particle
        and stable_coin
        and mean_delta >= 0.05
        and min(half_deltas) >= 0.0
        and min(coin_deltas) >= 0.0
    )
    return {
        "seat": seat,
        "root_id": int(row["root_id"]),
        "turn": int(row["turn"]),
        "plan_actions": plan_actions,
        "best": full_best,
        "stable_particle": stable_particle,
        "stable_coin": stable_coin,
        "stable_change": stable_change,
        "mean_delta": mean_delta,
        "half_deltas": half_deltas,
        "coin_deltas": coin_deltas,
        "critical_flags": [
            name
            for name, value in (row.get("criticality") or {}).get("flags", {}).items()
            if value
        ],
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    analyses = []
    source_hashes = {}
    for seat, path in (("first", args.first.resolve()), ("second", args.second.resolve())):
        _, rows = validate_shard(path, seat)
        source_hashes[seat] = sha256(path)
        analyses.extend(analyze_root(row, seat) for row in rows)
    total = len(analyses)
    particle_stable = sum(int(row["stable_particle"]) for row in analyses)
    coin_stable = sum(int(row["stable_coin"]) for row in analyses)
    stable_changes = [row for row in analyses if row["stable_change"]]
    seat_changes = Counter(row["seat"] for row in stable_changes)
    plan_changes = Counter(str(row["best"]) for row in stable_changes)
    flags = Counter(flag for row in analyses for flag in row["critical_flags"])
    stable_mean_delta = mean(row["mean_delta"] for row in stable_changes)
    gates = {
        "all_12_roots_integrity_complete": total == 12,
        "particle_half_ranking_stability_at_least_75pct": particle_stable / total >= 0.75,
        "coin_stream_ranking_stability_at_least_75pct": coin_stable / total >= 0.75,
        "at_least_three_stable_nonbaseline_plans": len(stable_changes) >= 3,
        "stable_change_in_each_seat": all(seat_changes[seat] >= 1 for seat in EXPECTED),
        "stable_change_mean_delta_at_least_0_05": stable_mean_delta >= 0.05,
        "at_least_three_criticality_categories": len(flags) >= 3,
    }
    passes = all(gates.values())
    payload = {
        "schema": "target80-paired-milestone-root-analysis/v2",
        "protocol": PROTOCOL,
        "collector_sha256": COLLECTOR_SHA256,
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "source_sha256": source_hashes,
        "root_count": total,
        "particle_stable_count": particle_stable,
        "particle_stable_rate": particle_stable / total,
        "coin_stable_count": coin_stable,
        "coin_stable_rate": coin_stable / total,
        "stable_change_count": len(stable_changes),
        "stable_change_rate": len(stable_changes) / total,
        "stable_change_count_by_seat": dict(seat_changes),
        "stable_change_count_by_plan": dict(plan_changes),
        "stable_change_mean_delta": stable_mean_delta,
        "critical_flag_counts": dict(sorted(flags.items())),
        "gates": gates,
        "passes_pilot": passes,
        "authority": "expanded plan-root collection only" if passes else "none; close exact V2 pilot",
        "deployment_authority": False,
        "rows": analyses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
