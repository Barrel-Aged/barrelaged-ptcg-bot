#!/usr/bin/env python3
"""Audit and score the frozen compact engine-resilience rule on fresh roots."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_target80_paired_milestone_roots_v2 as paired  # noqa: E402


PROTOCOL = "TARGET80_COMPACT_ENGINE_RULE_V5_TEST_PROTOCOL.md"
COLLECTOR_SHA256 = "fc6a3eddf3065c2334c47178c9a01d4a55b91cf9a6c358591d89af342f53ee89"
EXPECTED = {
    "test_first": ("first", 48, 861111),
    "test_second": ("second", 48, 861211),
}
FIXED = {
    "capture_roots_per_game": 8,
    "candidates": 3,
    "minimum_plan_action_support": 0.01,
    "particles": 8,
    "coin_rollouts": 2,
    "particle_offset": 0,
    "earliest_turn": 2,
    "latest_turn": 8,
    "max_source_attempts": 1_200,
    "max_live_decisions": 2_000,
    "max_rollout_decisions": 150,
    "root_budget_seconds": 180.0,
    "threads": 1,
}
OWN_TREE = "30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f"
OWN_DECK = "4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea"
OPPONENT_TREE = "49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else 0.0


def stable_teacher(row: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    samples = row["paired_samples"]
    best = paired.best_plan(samples)
    halves = [
        paired.best_plan(samples, particles=set(range(4))),
        paired.best_plan(samples, particles=set(range(4, 8))),
    ]
    coins = [paired.best_plan(samples, coin=index) for index in (0, 1)]
    deltas = {
        name: mean(
            float(sample["values"][name]) - float(sample["values"]["baseline"])
            for sample in samples
        )
        for name in paired.PLANS
    }
    stable = bool(
        best != "baseline"
        and halves[0] == halves[1] == best
        and coins[0] == coins[1] == best
        and deltas[str(best)] >= 0.05
        and min(
            paired.delta(samples, str(best), particles=set(range(4))),
            paired.delta(samples, str(best), particles=set(range(4, 8))),
            paired.delta(samples, str(best), coin=0),
            paired.delta(samples, str(best), coin=1),
        )
        >= 0.0
    )
    return (str(best) if stable else "baseline"), deltas


def target(row: Mapping[str, Any], plan: str) -> tuple[int | None, int | None]:
    option = row["plan_root_options"][plan]
    area = option.get("in_play_area")
    index = option.get("in_play_index")
    if area not in (4, 5) or index is None:
        return None, None
    current = row["observation"]["current"]
    own = current["players"][int(current["yourIndex"])]
    zone = own["active" if int(area) == 4 else "bench"]
    index = int(index)
    if not 0 <= index < len(zone):
        raise ValueError("plan option target is outside the public board")
    pokemon = zone[index]
    return int(pokemon["id"]), len(pokemon.get("energies") or ())


def compact_rule(row: Mapping[str, Any]) -> str:
    """The prospectively frozen public-state rule; no teacher fields allowed."""
    engine_option = row["plan_root_options"]["engine_resilience"]
    engine_card, engine_energy = target(row, "engine_resilience")
    baseline_card, baseline_energy = target(row, "baseline")
    engine_is_one_energy_spidops_attachment = bool(
        int(engine_option["type"]) == 8
        and int(engine_option.get("hand_card_id") or -1) in (1, 5, 15)
        and engine_card == 401
        and engine_energy == 1
    )
    baseline_yields_engine_window = bool(
        baseline_card is None
        or (
            baseline_card != 400
            and baseline_energy == 0
        )
    )
    return (
        "engine_resilience"
        if engine_is_one_energy_spidops_attachment and baseline_yields_engine_window
        else "baseline"
    )


def load(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()
    for raw in paths:
        path = raw.resolve()
        manifest = json.loads(
            path.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("schema") != "target80-compact-engine-rule-test-manifest/v5"
            or manifest.get("protocol") != PROTOCOL
            or manifest.get("collector_sha256") != COLLECTOR_SHA256
            or manifest.get("output_sha256") != sha256(path)
            or manifest.get("own_tree_sha256") != OWN_TREE
            or manifest.get("own_deck_sha256") != OWN_DECK
            or manifest.get("opponent_tree_sha256") != OPPONENT_TREE
        ):
            raise ValueError(f"{path}: frozen manifest mismatch")
        shard = str(manifest.get("shard") or "")
        if shard not in EXPECTED or shard in seen:
            raise ValueError(f"{path}: unexpected or duplicate shard {shard}")
        seen.add(shard)
        seat, count, seed = EXPECTED[shard]
        parameters = manifest.get("parameters") or {}
        for key, value in {**FIXED, "seat": seat, "roots": count, "seed": seed}.items():
            if parameters.get(key) != value:
                raise ValueError(f"{path}: parameter {key} changed")
        counts = manifest.get("counts") or {}
        if (
            counts.get("retained_roots") != count
            or counts.get("fully_complete_roots") != count
            or counts.get("complete_pairs") != count * 16
            or counts.get("rollout_errors", 0) != 0
        ):
            raise ValueError(f"{path}: incomplete manifest counts")
        file_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(file_rows) != count:
            raise ValueError(f"{path}: row count mismatch")
        for row in file_rows:
            if (
                row.get("schema") != "target80-compact-engine-rule-test-root/v5"
                or row.get("protocol") != PROTOCOL
                or row.get("shard") != shard
                or not row.get("fully_complete")
                or not row.get("guarded_baseline_reproduced")
                or int(row.get("complete_pairs") or 0) != 16
                or row.get("errors")
            ):
                raise ValueError(f"{path}: invalid root")
            compact_rule(row)
            stable_teacher(row)
            rows.append(row)
        hashes[shard] = sha256(path)
    if seen != set(EXPECTED):
        raise ValueError(f"expected {sorted(EXPECTED)}, got {sorted(seen)}")
    return rows, hashes


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows, hashes = load(args.input)
    teachers = [stable_teacher(row) for row in rows]
    labels = [item[0] for item in teachers]
    deltas = [item[1] for item in teachers]
    predictions = [compact_rule(row) for row in rows]
    changed = [i for i, value in enumerate(predictions) if value != "baseline"]
    positives = [i for i, value in enumerate(labels) if value != "baseline"]
    correct = [i for i in changed if predictions[i] == labels[i]]
    baseline_rows = [i for i, value in enumerate(labels) if value == "baseline"]
    predicted_deltas = [
        0.0 if prediction == "baseline" else float(deltas[i][prediction])
        for i, prediction in enumerate(predictions)
    ]
    changed_deltas = [predicted_deltas[i] for i in changed]
    seat_means = {
        name: mean(
            predicted_deltas[i]
            for i, row in enumerate(rows)
            if bool(row["started_first"]) == started_first
        )
        for name, started_first in (("first", True), ("second", False))
    }
    metrics = {
        "roots": len(rows),
        "matched_sets": sum(int(row["complete_pairs"]) for row in rows),
        "label_counts": dict(Counter(labels)),
        "prediction_counts": dict(Counter(predictions)),
        "change_count": len(changed),
        "change_rate": len(changed) / len(rows),
        "exact_nonbaseline_precision": len(correct) / max(1, len(changed)),
        "exact_nonbaseline_recall": len(correct) / max(1, len(positives)),
        "baseline_retention": sum(
            predictions[i] == "baseline" for i in baseline_rows
        )
        / max(1, len(baseline_rows)),
        "mean_teacher_delta_all_rows": mean(predicted_deltas),
        "mean_teacher_delta_changed_rows": mean(changed_deltas),
        "negative_changed_delta_rate": sum(value < -1e-9 for value in changed_deltas)
        / max(1, len(changed_deltas)),
        "seat_mean_teacher_delta": seat_means,
    }
    gates = {
        "integrity_96_roots_1536_sets": len(rows) == 96
        and metrics["matched_sets"] == 1_536,
        "change_rate_5_to_30pct": 0.05 <= metrics["change_rate"] <= 0.30,
        "precision_at_least_0_65": metrics["exact_nonbaseline_precision"] >= 0.65,
        "recall_at_least_0_20": metrics["exact_nonbaseline_recall"] >= 0.20,
        "baseline_retention_at_least_0_90": metrics["baseline_retention"] >= 0.90,
        "mean_delta_all_rows_at_least_0_04": metrics[
            "mean_teacher_delta_all_rows"
        ]
        >= 0.04,
        "negative_changed_rate_at_most_0_15": metrics[
            "negative_changed_delta_rate"
        ]
        <= 0.15,
        "both_start_orders_nonnegative": min(seat_means.values()) >= 0.0,
    }
    passes = all(gates.values())
    payload = {
        "schema": "target80-compact-engine-rule-test-analysis/v5",
        "protocol": PROTOCOL,
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "collector_sha256": COLLECTOR_SHA256,
        "source_sha256": hashes,
        "rule": (
            "engine plan attaches energy to a one-energy Spidops AND baseline "
            "has no public board target or targets a zero-energy non-Tarountula"
        ),
        "metrics": metrics,
        "gates": gates,
        "passes_test": passes,
        "authority": (
            "full-match official-engine development integration only"
            if passes
            else "none; close compact engine rule V5"
        ),
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
