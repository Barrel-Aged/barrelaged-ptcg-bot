#!/usr/bin/env python3
"""Train the frozen pairwise causal recovery candidate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import monotonic
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from train_live_truth_causal_q import (
    EncodedRoot,
    encode_roots,
    read_deck,
    read_roots,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PairwisePreferenceHead(nn.Module):
    def __init__(self, token_width: int, hidden: int = 256) -> None:
        super().__init__()
        self.token_width = int(token_width)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.LayerNorm(3 * token_width + 1),
            nn.Linear(3 * token_width + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        global_token: torch.Tensor,
        baseline_token: torch.Tensor,
        alternative_tokens: torch.Tensor,
        baseline_logit: torch.Tensor,
        alternative_logits: torch.Tensor,
    ) -> torch.Tensor:
        count = alternative_tokens.shape[0]
        state = global_token.reshape(1, -1).expand(count, -1)
        baseline = baseline_token.reshape(1, -1).expand(count, -1)
        features = torch.cat(
            (
                state,
                alternative_tokens - baseline,
                alternative_tokens * baseline,
                (alternative_logits - baseline_logit).reshape(-1, 1),
            ),
            dim=-1,
        )
        return self.network(features).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_neural_resolver_safe_tie_v3"),
    )
    parser.add_argument("--train-data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, action="append", required=True)
    parser.add_argument("--test-data", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "live_truth_pairwise_v1"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-roots", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=615_501)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser.parse_args()


def pair_tensors(root: EncodedRoot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baseline_position = root.baseline_position
    positions = [index for index in range(len(root.candidates)) if index != baseline_position]
    if not positions:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
            torch.empty(0),
        )
    alternatives = torch.tensor(positions, dtype=torch.long)
    deltas = root.utilities[alternatives] - root.utilities[baseline_position]
    keep = deltas.abs() >= 0.5
    return alternatives[keep], (deltas[keep] > 0).float(), deltas[keep]


def pair_logits(
    head: PairwisePreferenceHead,
    root: EncodedRoot,
    alternative_positions: torch.Tensor,
) -> torch.Tensor:
    baseline_option = int(root.candidates[root.baseline_position])
    alternative_options = root.candidates[alternative_positions]
    return head(
        root.global_token,
        root.option_tokens[baseline_option],
        root.option_tokens[alternative_options],
        root.base_logits[baseline_option],
        root.base_logits[alternative_options],
    )


def count_classes(roots: Sequence[EncodedRoot]) -> tuple[int, int]:
    positive = negative = 0
    for root in roots:
        _, targets, _ = pair_tensors(root)
        positive += int(targets.sum())
        negative += len(targets) - int(targets.sum())
    return positive, negative


@torch.no_grad()
def pair_metrics(head: PairwisePreferenceHead, roots: Sequence[EncodedRoot], pos_weight: float) -> dict[str, Any]:
    head.eval()
    losses = []
    correct = positive = negative = 0
    for root in roots:
        positions, targets, _ = pair_tensors(root)
        if not len(positions):
            continue
        logits = pair_logits(head, root, positions)
        loss = Fnn.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=logits.new_tensor(pos_weight),
            reduction="none",
        )
        losses.extend(float(value) for value in loss)
        correct += int(((logits >= 0) == targets.bool()).sum())
        positive += int(targets.sum())
        negative += len(targets) - int(targets.sum())
    return {
        "pairs": positive + negative,
        "positive": positive,
        "negative": negative,
        "balanced_bce": statistics.fmean(losses) if losses else math.inf,
        "accuracy": correct / max(1, positive + negative),
    }


@torch.no_grad()
def root_predictions(head: PairwisePreferenceHead, roots: Sequence[EncodedRoot]) -> list[dict[str, Any]]:
    head.eval()
    rows = []
    for root in roots:
        positions = torch.tensor(
            [index for index in range(len(root.candidates)) if index != root.baseline_position],
            dtype=torch.long,
        )
        if not len(positions):
            continue
        probabilities = torch.sigmoid(pair_logits(head, root, positions))
        local = int(probabilities.argmax())
        alternative_position = int(positions[local])
        rows.append(
            {
                "game": root.game_key,
                "root": root.root_key,
                "probability": float(probabilities[local]),
                "alternative_position": alternative_position,
                "baseline_position": root.baseline_position,
                "alternative_utility": float(root.utilities[alternative_position]),
                "baseline_utility": float(root.utilities[root.baseline_position]),
            }
        )
    return rows


def gate_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    deltas = []
    interventions = correct = 0
    for row in rows:
        use = float(row["probability"]) >= threshold
        delta = float(row["alternative_utility"] - row["baseline_utility"]) if use else 0.0
        deltas.append(delta)
        interventions += int(use)
        correct += int(use and delta > 0)
    return {
        "roots": len(rows),
        "threshold": float(threshold),
        "interventions": interventions,
        "intervention_rate": interventions / max(1, len(rows)),
        "intervention_precision": correct / max(1, interventions),
        "mean_paired_improvement": statistics.fmean(deltas) if deltas else 0.0,
    }


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    probabilities = sorted({float(row["probability"]) for row in rows})
    candidates = {0.5, 1.0 + 1e-6}
    if probabilities:
        candidates.update(float(np.quantile(probabilities, q)) for q in np.linspace(0.5, 1.0, 51))
    table = [gate_metrics(rows, threshold) for threshold in sorted(candidates)]
    minimum = 5
    eligible = [
        item
        for item in table
        if minimum <= item["interventions"] <= math.floor(0.35 * len(rows))
        and item["intervention_precision"] >= 0.65
        and item["mean_paired_improvement"] > 0.0
    ]
    if not eligible:
        return 1.0 + 1e-6, table
    selected = max(
        eligible,
        key=lambda item: (
            item["mean_paired_improvement"],
            item["intervention_precision"],
            -item["intervention_rate"],
            item["threshold"],
        ),
    )
    return float(selected["threshold"]), table


def clustered_interval(
    rows: Sequence[Mapping[str, Any]], threshold: float, samples: int, seed: int
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        use = float(row["probability"]) >= threshold
        grouped[str(row["game"])].append(
            float(row["alternative_utility"] - row["baseline_utility"]) if use else 0.0
        )
    games = sorted(grouped)
    observed_values = [value for game in games for value in grouped[game]]
    observed = statistics.fmean(observed_values) if observed_values else 0.0
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        chosen = [games[rng.randrange(len(games))] for _ in games]
        values = [value for game in chosen for value in grouped[game]]
        boot.append(statistics.fmean(values) if values else 0.0)
    boot.sort()
    return (
        observed,
        boot[max(0, int(0.025 * len(boot)) - 1)],
        boot[min(len(boot) - 1, int(0.975 * len(boot)))],
    )


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


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    candidate = args.candidate.resolve()
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    from soldier import features as feature_module  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy  # type: ignore

    train_paths = [path.resolve() for path in args.train_data]
    validation_paths = [path.resolve() for path in args.validation_data]
    test_paths = [path.resolve() for path in args.test_data]
    train_raw, train_counts = read_roots(train_paths)
    validation_manifests = {
        str(path): verify_frozen_shard(path) for path in validation_paths
    }
    validation_raw, validation_counts = read_roots(validation_paths)
    deck = read_deck(candidate / "deck.csv")
    library_path = candidate / "deck_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else {}
    blueprint = load_policy(candidate / "model.pt").eval()
    for parameter in blueprint.parameters():
        parameter.requires_grad = False
    encoded, encode_counts = encode_roots(
        [*train_raw, *validation_raw],
        model=blueprint,
        deck=deck,
        library=library,
        belief_class=Belief,
        features_module=feature_module,
    )
    train_keys = {root.root_key for root in train_raw}
    validation_keys = {root.root_key for root in validation_raw}
    if train_keys & validation_keys:
        raise RuntimeError("training and validation roots overlap")
    data = {
        "train": [root for root in encoded if root.root_key in train_keys],
        "validation": [root for root in encoded if root.root_key in validation_keys],
    }
    if any(not values for values in data.values()):
        raise RuntimeError("pairwise split is empty")
    positive, negative = count_classes(data["train"])
    if positive == 0 or negative == 0:
        raise RuntimeError("training data lacks one pairwise class")
    pos_weight = negative / positive
    head = PairwisePreferenceHead(int(blueprint.h), args.hidden)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_score = math.inf
    best_epoch = 0
    patience = 0
    history = []
    started = monotonic()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data["train"])))
        random.Random(args.seed + epoch).shuffle(order)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = roots_used = pairs_used = 0
        accumulation = 0
        for position, index in enumerate(order, start=1):
            root = data["train"][index]
            alternatives, targets, _ = pair_tensors(root)
            if not len(alternatives):
                continue
            logits = pair_logits(head, root, alternatives)
            loss = Fnn.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=logits.new_tensor(pos_weight),
            )
            (loss / args.batch_roots).backward()
            loss_total += float(loss.detach())
            roots_used += 1
            pairs_used += len(targets)
            accumulation += 1
            if accumulation >= args.batch_roots or position == len(order):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        validation = pair_metrics(head, data["validation"], pos_weight)
        row = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, roots_used),
            "train_roots": roots_used,
            "train_pairs": pairs_used,
            "validation": validation,
            "elapsed_seconds": monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        score = float(validation["balanced_bce"])
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "schema": "rocket-live-truth-pairwise/v1",
                    "config": {"token_width": int(blueprint.h), "hidden": args.hidden},
                    "state_dict": head.state_dict(),
                },
                output / "pairwise.pt",
            )
        else:
            patience += 1
            if patience >= args.patience:
                break

    artifact = torch.load(output / "pairwise.pt", map_location="cpu", weights_only=True)
    best = PairwisePreferenceHead(**artifact["config"])
    best.load_state_dict(artifact["state_dict"])
    best.eval()
    pair_results = {
        name: pair_metrics(best, roots, pos_weight) for name, roots in data.items()
    }
    validation_rows = root_predictions(best, data["validation"])
    threshold, threshold_table = select_threshold(validation_rows)
    validation_gate = gate_metrics(validation_rows, threshold)
    selection_admitted = threshold <= 1.0
    if selection_admitted:
        # The untouched test files are deliberately not read until the frozen
        # model and validation-only intervention threshold have qualified.
        test_manifests = {
            str(path): verify_frozen_shard(path) for path in test_paths
        }
        test_raw, test_counts = read_roots(test_paths)
        test_encoded, test_encode_counts = encode_roots(
            test_raw,
            model=blueprint,
            deck=deck,
            library=library,
            belief_class=Belief,
            features_module=feature_module,
        )
        test_keys = {root.root_key for root in test_raw}
        if test_keys & (train_keys | validation_keys):
            raise RuntimeError("untouched test roots overlap development evidence")
        if not test_encoded:
            raise RuntimeError("untouched test split is empty")
        data["test"] = test_encoded
        pair_results["test"] = pair_metrics(best, test_encoded, pos_weight)
        test_rows = root_predictions(best, test_encoded)
        test_gate: dict[str, Any] | None = gate_metrics(test_rows, threshold)
        observed, lower, upper = clustered_interval(
            test_rows, threshold, args.bootstrap_samples, args.seed + 99
        )
        admitted = bool(
            test_gate["interventions"] >= 8
            and test_gate["intervention_rate"] <= 0.35
            and test_gate["intervention_precision"] >= 0.65
            and observed > 0.0
            and lower > 0.0
        )
        test_interval: list[float] | None = [lower, upper]
        test_observed: float | None = observed
        test_source_hashes: dict[str, str] | None = {
            str(path): digest(path) for path in test_paths
        }
    else:
        test_manifests = None
        test_counts = Counter()
        test_encode_counts = Counter()
        test_gate = None
        test_interval = None
        test_observed = None
        test_source_hashes = None
        admitted = False
    summary = {
        "schema": "rocket-live-truth-pairwise-training/v1",
        "method": "class-balanced pairwise causal preference on frozen contextual tokens",
        "source": {
            "candidate": str(candidate),
            "candidate_model_sha256": digest(candidate / "model.pt"),
            "train_data": {str(path): digest(path) for path in train_paths},
            "validation_data": {str(path): digest(path) for path in validation_paths},
            "validation_manifests": validation_manifests,
            "test_data_listed": [str(path) for path in test_paths],
            "test_data_opened_sha256": test_source_hashes,
            "test_manifests_opened": test_manifests,
        },
        "counts": {
            "train_read": dict(train_counts),
            "validation_read": dict(validation_counts),
            "test_read": dict(test_counts),
            "development_encoded": dict(encode_counts),
            "test_encoded": dict(test_encode_counts),
            "split_roots": {name: len(values) for name, values in data.items()},
            "train_positive_pairs": positive,
            "train_negative_pairs": negative,
        },
        "training": {
            "seed": args.seed,
            "positive_weight": pos_weight,
            "best_epoch": best_epoch,
            "history": history,
        },
        "pair_metrics": pair_results,
        "gate": {
            "threshold": threshold,
            "selection_table": threshold_table,
            "validation": validation_gate,
            "test": test_gate,
            "selection_admitted": selection_admitted,
            "test_opened": selection_admitted,
            "test_clustered_95_interval": test_interval,
            "test_observed_mean": test_observed,
            "admitted": admitted,
        },
        "claims": {
            "official_descendant_execution": True,
            "realized_hidden_state_samples": True,
            "common_random_coins": True,
            "public_information_set_expected_values": False,
            "equilibrium_certificate": False,
            "safe_strategy_certificate": False,
            "arena_strength_improvement": False,
        },
        "artifact": str(output / "pairwise.pt"),
        "artifact_sha256": digest(output / "pairwise.pt"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "threshold": threshold,
                "validation_gate": validation_gate,
                "test_gate": test_gate,
                "test_clustered_95_interval": test_interval,
                "admitted": admitted,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
