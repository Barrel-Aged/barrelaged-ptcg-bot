#!/usr/bin/env python3
"""Fit and admit the frozen semantic empirical causal-rule overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from explore_empirical_causal_rules import fit_rules, metrics, predict, read_roots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "min_count": 5,
    "min_sources": 2,
    "min_positive": 3,
    "min_mean": 0.10,
    "max_harm_rate": 0.20,
    "min_decisive_precision": 0.75,
    "require_source_nonnegative": True,
}
BOOTSTRAP_SAMPLES = 10_000
SEED = 615_801


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-data", type=Path, action="append", required=True)
    parser.add_argument("--test-data", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "empirical_causal_rules_v1"),
    )
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_shard(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-frozen-causal-shard/v1":
        raise ValueError(f"{manifest_path}: unsupported frozen-shard schema")
    if Path(manifest["output"]).resolve() != path.resolve():
        raise ValueError(f"{manifest_path}: output path mismatch")
    if manifest.get("output_sha256") != digest(path):
        raise ValueError(f"{manifest_path}: frozen-shard hash mismatch")
    if not (
        int(manifest.get("first_game", 0)) == 1
        and int(manifest.get("last_game", 0)) == 30
        and int(manifest.get("games", 0)) == 30
        and bool(manifest.get("all_roots_fully_complete"))
        and manifest.get("roots_by_rollout_count")
        == {"2": int(manifest.get("roots", 0))}
    ):
        raise ValueError(f"{manifest_path}: incomplete frozen 30-game evidence")
    return manifest


def development_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["interventions"] >= 8
        and result["intervention_rate"] <= 0.35
        and result["non_harm_rate"] >= 0.90
        and result["decisive_precision"] >= 0.80
        and result["mean_paired_improvement"] > 0
        and result["game_clustered_95_interval"][0] > 0
    )


def admission_gate(result: dict[str, Any]) -> bool:
    return bool(
        result["interventions"] >= 8
        and result["intervention_rate"] <= 0.35
        and result["non_harm_rate"] >= 0.85
        and result["decisive_precision"] >= 0.70
        and result["mean_paired_improvement"] > 0
        and result["game_clustered_95_interval"][0] > 0
    )


def serializable_rules(rules: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "baseline": list(pattern[0]),
            "alternative": list(pattern[1]),
            "statistics": {
                **statistics,
                "source_means": dict(sorted(statistics["source_means"].items())),
            },
        }
        for pattern, statistics in sorted(rules.items())
    ]


def main() -> int:
    args = parse_args()
    development_paths = [path.resolve() for path in args.development_data]
    test_paths = [path.resolve() for path in args.test_data]
    if len(development_paths) != 3 or len(test_paths) != 3:
        raise ValueError("the frozen protocol requires exactly three development and three test shards")
    development_manifests = {
        str(path): verify_frozen_shard(path) for path in development_paths
    }
    development_roots = read_roots(development_paths)
    sources = sorted({root.source for root in development_roots})
    if len(sources) != 3:
        raise ValueError("development evidence does not contain three independent sources")

    cross_validated_rows = []
    fold_rule_counts = {}
    for held_source in sources:
        training = [root for root in development_roots if root.source != held_source]
        held = [root for root in development_roots if root.source == held_source]
        rules = fit_rules(training, **CONFIG)
        fold_rule_counts[held_source] = len(rules)
        cross_validated_rows.extend(predict(held, rules))
    development_metrics = metrics(cross_validated_rows, BOOTSTRAP_SAMPLES, SEED)
    development_admitted = development_gate(development_metrics)

    final_rules = fit_rules(development_roots, **CONFIG) if development_admitted else {}
    if development_admitted:
        # Test manifests, bytes, labels, and metrics remain unopened until this
        # exact point, after the configuration and cross-validation gate freeze.
        test_manifests = {str(path): verify_frozen_shard(path) for path in test_paths}
        test_roots = read_roots(test_paths)
        test_rows = predict(test_roots, final_rules)
        test_metrics: dict[str, Any] | None = metrics(
            test_rows, BOOTSTRAP_SAMPLES, SEED + 1
        )
        admitted = admission_gate(test_metrics)
        test_hashes: dict[str, str] | None = {str(path): digest(path) for path in test_paths}
    else:
        test_manifests = None
        test_metrics = None
        test_hashes = None
        admitted = False

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rule_artifact = {
        "schema": "rocket-empirical-causal-rules/v1",
        "action_signature": ["type", "hand_card_id", "in_play_area", "attack_id"],
        "config": CONFIG,
        "selection": "highest mean, lowest harm rate, greatest support, deterministic option index",
        "rules": serializable_rules(final_rules),
        "runtime_scope": {
            "candidate_count": 4,
            "latest_turn": 8,
            "own_seat": "second",
            "opponent_family": "publicly identified Marnie",
            "eligible_root": "first single-choice main root per turn",
        },
    }
    rules_path = output / "empirical_rules.json"
    rules_path.write_text(json.dumps(rule_artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema": "rocket-empirical-causal-rule-training/v1",
        "method": "low-capacity exact public action-semantic causal lookup",
        "config": CONFIG,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "source": {
            "development_data": {str(path): digest(path) for path in development_paths},
            "development_manifests": development_manifests,
            "test_data_listed": [str(path) for path in test_paths],
            "test_data_opened_sha256": test_hashes,
            "test_manifests_opened": test_manifests,
        },
        "development": {
            "roots": len(development_roots),
            "fold_rule_counts": fold_rule_counts,
            "metrics": development_metrics,
            "admitted": development_admitted,
        },
        "final_rule_count": len(final_rules),
        "test": {
            "opened": development_admitted,
            "metrics": test_metrics,
            "admitted": admitted,
        },
        "claims": {
            "official_descendant_execution": True,
            "realized_hidden_state_samples": True,
            "runtime_private_information": False,
            "public_information_set_expected_values": False,
            "equilibrium_certificate": False,
            "safe_strategy_certificate": False,
            "arena_strength_improvement": False,
        },
        "artifact": str(rules_path),
        "artifact_sha256": digest(rules_path),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
