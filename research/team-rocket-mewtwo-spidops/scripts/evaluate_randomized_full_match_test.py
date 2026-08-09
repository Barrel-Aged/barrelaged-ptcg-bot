#!/usr/bin/env python3
"""Open the sealed randomized test once for a frozen causal ridge artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import monotonic

import numpy as np
import torch

from train_randomized_full_match_ridge import (
    FROZEN_DECK_SHA,
    FROZEN_MODEL_SHA,
    Projection,
    bootstrap_weight_matrix,
    build_feature_sets,
    encode_trials,
    policy_metrics,
    policy_positions,
    predict_ensemble,
    read_deck,
    read_trials,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "randomized_full_match_ridge_v1"
        ),
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_value_calibrated_v1"
        ),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_neural_resolver_safe_tie_v3"
        ),
    )
    parser.add_argument("--test-data", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=731_991)
    return parser.parse_args()


def test_admissible(metrics: dict, *, minimum_policy_matches: int) -> bool:
    return bool(
        float(metrics["dr_improvement"]) > 0.0
        and float(metrics["dr_interval_95"][0]) > 0.0
        and float(metrics["ips_improvement"]) > 0.0
        and int(metrics["assigned_policy_matches"]) >= minimum_policy_matches
        and all(
            float(value) > -0.03
            for value in (metrics.get("opponent_dr_improvement") or {}).values()
        )
    )


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    started = monotonic()
    artifact_dir = args.artifact.resolve()
    artifact_path = artifact_dir / "causal_ridge.npz"
    manifest_path = artifact_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-randomized-full-match-ridge-artifact/v1":
        raise ValueError("unsupported frozen artifact manifest")
    if manifest.get("test_data_opened") is not False:
        raise ValueError("artifact was not frozen with test data sealed")
    if manifest.get("artifact_sha256") != sha256(artifact_path):
        raise ValueError("frozen artifact hash mismatch")
    candidate = args.candidate.resolve()
    if sha256(candidate / "model.pt") != FROZEN_MODEL_SHA:
        raise ValueError("candidate model hash mismatch")
    if sha256(candidate / "deck.csv") != FROZEN_DECK_SHA:
        raise ValueError("candidate deck hash mismatch")
    eligible_turn = int(manifest.get("eligible_turn", 2))
    raw, read_counts = read_trials(
        args.test_data,
        "test",
        allow_test=True,
        expected_turn=eligible_turn,
        expected_shortlist_size=int(manifest.get("shortlist_size", 0)),
        expected_protocol=str(manifest.get("protocol")),
    )
    if len({trial.opponent for trial in raw}) < 2:
        raise RuntimeError("sealed test does not contain both frozen opponents")
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    from soldier import features as features_module  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import attack_safe_logits, load_policy  # type: ignore

    deck = read_deck(candidate / "deck.csv")
    library_path = candidate / "deck_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else {}
    model = load_policy(candidate / "model.pt").eval()
    encoded, encode_counts = encode_trials(
        raw,
        model=model,
        deck=deck,
        library=library,
        belief_class=Belief,
        features_module=features_module,
        attack_safe_logits=attack_safe_logits,
        batch_size=args.batch_size,
    )
    arrays = np.load(artifact_path, allow_pickle=False)
    projection = Projection(
        global_mean=arrays["global_mean"],
        global_components=arrays["global_components"],
        option_mean=arrays["option_mean"],
        option_components=arrays["option_components"],
        difference_mean=arrays["difference_mean"],
        difference_components=arrays["difference_components"],
    )
    features, _, _ = build_feature_sets(
        encoded,
        projection,
        arrays["feature_mean"],
        arrays["feature_scale"],
    )
    predictions = predict_ensemble(arrays["weights"], features)
    policy, _ = policy_positions(
        predictions,
        float(manifest["confidence_multiplier"]),
        float(manifest["margin"]),
    )
    bootstrap = bootstrap_weight_matrix(
        encoded, args.bootstrap_samples, args.seed + 600_011
    )
    metrics = policy_metrics(encoded, predictions, policy, bootstrap)
    admitted = test_admissible(
        metrics,
        minimum_policy_matches=int(manifest.get("test_minimum_policy_matches", 30)),
    )
    report = {
        "schema": "rocket-randomized-full-match-ridge-test/v1",
        "protocol": manifest.get("protocol"),
        "admitted": admitted,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "frozen_gate": {
            "confidence_multiplier": float(manifest["confidence_multiplier"]),
            "margin": float(manifest["margin"]),
            "penalty": float(manifest["penalty"]),
        },
        "metrics": metrics,
        "read_counts": dict(read_counts),
        "encode_counts": dict(encode_counts),
        "test_data": [str(path.resolve()) for path in args.test_data],
        "test_data_sha256": {
            str(path.resolve()): sha256(path.resolve()) for path in args.test_data
        },
        "elapsed_seconds": monotonic() - started,
        "claim_boundary": (
            "untouched randomized official-match test evidence; fresh full-match "
            "admission and arena strength remain unevaluated"
        ),
    }
    report_path = artifact_dir / "test_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"admitted": admitted, "metrics": metrics, "report": str(report_path)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
