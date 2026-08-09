#!/usr/bin/env python3
"""Train and gate a conservative state-conditioned strategic-plan selector."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import analyze_target80_paired_milestone_roots_v2 as analysis  # noqa: E402
from rocketbot.policy_feature_snapshot import deserialize_policy_feature  # noqa: E402


PROTOCOL = "TARGET80_PLAN_SELECTOR_V4_PROTOCOL.md"
COLLECTOR_SHA256 = "ec0dac8c4943e773ac325b41f333ad1f56dbbefb5f02e598af37c96e8d56d710"
PLANS = analysis.PLANS
PLAN_INDEX = {name: index for index, name in enumerate(PLANS)}
MAX_POKES = 11
POKE_WIDTH = 47
OPTION_WIDTH = 58
GLOBAL_WIDTH = 102
VECTOR_SCHEMA = "target80-plan-selector-vector/v1"
EXPECTED_SHARDS = {
    "train_first_a": ("train", "first", 40, 859101),
    "train_first_b": ("train", "first", 40, 859102),
    "train_second_a": ("train", "second", 40, 859201),
    "train_second_b": ("train", "second", 40, 859202),
    "validation_first": ("validation", "first", 24, 859111),
    "validation_second": ("validation", "second", 24, 859211),
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
    "max_source_attempts": 1_200,
    "max_live_decisions": 2_000,
    "max_rollout_decisions": 150,
    "root_budget_seconds": 180.0,
    "threads": 1,
}
CONFIGS = (
    {
        "name": "conservative",
        "hidden": 96,
        "dropout": 0.10,
        "weight_decay": 1e-3,
        "learning_rate": 1e-3,
        "epochs": 120,
        "seed": 860101,
    },
    {
        "name": "balanced",
        "hidden": 128,
        "dropout": 0.10,
        "weight_decay": 3e-4,
        "learning_rate": 1e-3,
        "epochs": 150,
        "seed": 860102,
    },
    {
        "name": "capacity",
        "hidden": 192,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "learning_rate": 7e-4,
        "epochs": 150,
        "seed": 860103,
    },
)
THRESHOLDS = (0.55, 0.65, 0.75, 0.85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
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


def stable_teacher(row: Mapping[str, Any]) -> tuple[str, dict[str, float], bool]:
    samples = row["paired_samples"]
    full_best = analysis.best_plan(samples)
    half_best = [
        analysis.best_plan(samples, particles=set(range(4))),
        analysis.best_plan(samples, particles=set(range(4, 8))),
    ]
    coin_best = [analysis.best_plan(samples, coin=coin) for coin in (0, 1)]
    deltas = {
        name: mean(
            float(sample["values"][name]) - float(sample["values"]["baseline"])
            for sample in samples
        )
        for name in PLANS
    }
    half_deltas = [
        analysis.delta(samples, str(full_best), particles=set(range(4))),
        analysis.delta(samples, str(full_best), particles=set(range(4, 8))),
    ]
    coin_deltas = [
        analysis.delta(samples, str(full_best), coin=0),
        analysis.delta(samples, str(full_best), coin=1),
    ]
    stable_change = bool(
        full_best != "baseline"
        and half_best[0] == half_best[1] == full_best
        and coin_best[0] == coin_best[1] == full_best
        and deltas[str(full_best)] >= 0.05
        and min(half_deltas) >= 0.0
        and min(coin_deltas) >= 0.0
    )
    return (str(full_best) if stable_change else "baseline"), deltas, stable_change


def _fixed_rows(array: np.ndarray, rows: int, width: int) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"unexpected matrix shape {value.shape}; expected (*, {width})")
    result = np.zeros((rows, width), dtype=np.float32)
    count = min(rows, value.shape[0])
    result[:count] = value[:count]
    return result


def vectorize(row: Mapping[str, Any]) -> np.ndarray:
    feature = deserialize_policy_feature(row["policy_feature"])
    global_values = np.asarray(feature["global"], dtype=np.float32).reshape(-1)
    if global_values.shape != (GLOBAL_WIDTH,):
        raise ValueError(f"unexpected global shape {global_values.shape}")
    belief = np.concatenate(
        [
            np.asarray(feature["belief_global"], dtype=np.float32).reshape(-1),
            np.asarray(feature["belief_prized"], dtype=np.float32).reshape(-1),
        ]
    )
    if belief.shape != (6,):
        raise ValueError(f"unexpected belief shape {belief.shape}")
    pokes = _fixed_rows(feature["poke_feats"], MAX_POKES, POKE_WIDTH)
    poke_cards = np.zeros(MAX_POKES, dtype=np.float32)
    raw_poke_cards = np.asarray(feature["poke_card"], dtype=np.float32).reshape(-1)
    poke_cards[: min(MAX_POKES, raw_poke_cards.size)] = (
        raw_poke_cards[:MAX_POKES] / 1267.0
    )
    option_feats = np.asarray(feature["opt_feats"], dtype=np.float32)
    if option_feats.ndim != 2 or option_feats.shape[1] != OPTION_WIDTH:
        raise ValueError(f"unexpected option shape {option_feats.shape}")
    option_cards = np.asarray(feature["opt_card"], dtype=np.float32).reshape(-1)
    plan_actions = row["plan_actions"]
    plan_option_parts = []
    plan_scalar_parts = []
    logits = [float(value) for value in row["policy_logits"]]
    probabilities = [float(value) for value in row["policy_probabilities"]]
    for plan in PLANS:
        index = int(plan_actions[plan][0])
        if not 0 <= index < option_feats.shape[0]:
            raise ValueError("plan action exceeds option tensor")
        plan_option_parts.append(option_feats[index])
        plan_scalar_parts.extend(
            [
                float(np.clip(logits[index], -20.0, 20.0) / 20.0),
                float(probabilities[index]),
                float(option_cards[index] / 1267.0),
            ]
        )
    critical = row["criticality"]
    flags = critical["flags"]
    context = np.asarray(
        [
            float(bool(row["started_first"])),
            float(row["turn"]) / 10.0,
            float(np.clip(float(row["policy_margin"]), 0.0, 5.0) / 5.0),
            float(critical["ready_attacker_count"]) / 2.0,
            float(bool(critical["mewtwo_on_board"])),
            float(bool(critical["mewtwo_ready"])),
            float(critical["spidops_count"]) / 3.0,
            float(critical["own_prizes_remaining"]) / 6.0,
            float(critical["opponent_prizes_remaining"]) / 6.0,
            *[
                float(bool(flags[name]))
                for name in (
                    "low_policy_margin",
                    "attack_setup_tradeoff",
                    "missing_second_attacker",
                    "mewtwo_activation",
                    "spidops_engine",
                    "prize_threshold",
                )
            ],
        ],
        dtype=np.float32,
    )
    vector = np.concatenate(
        [
            global_values,
            belief,
            pokes.reshape(-1),
            poke_cards,
            *plan_option_parts,
            np.asarray(plan_scalar_parts, dtype=np.float32),
            context,
        ]
    ).astype(np.float32)
    if not np.isfinite(vector).all():
        raise ValueError("selector vector contains non-finite values")
    return vector


def validate_and_load(paths: Sequence[Path], split: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes = {}
    seen = set()
    for raw_path in paths:
        path = raw_path.resolve()
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "target80-plan-selector-manifest/v4":
            raise ValueError(f"{path}: wrong manifest schema")
        if manifest.get("protocol") != PROTOCOL or manifest.get("split") != split:
            raise ValueError(f"{path}: protocol/split mismatch")
        if manifest.get("collector_sha256") != COLLECTOR_SHA256:
            raise ValueError(f"{path}: collector changed")
        if manifest.get("output_sha256") != sha256(path):
            raise ValueError(f"{path}: output hash mismatch")
        shard = str(manifest.get("shard") or "")
        if shard not in EXPECTED_SHARDS or shard in seen:
            raise ValueError(f"{path}: unexpected or duplicate shard {shard}")
        seen.add(shard)
        expected_split, seat, count, seed = EXPECTED_SHARDS[shard]
        if expected_split != split:
            raise ValueError(f"{path}: shard belongs to {expected_split}")
        parameters = manifest.get("parameters") or {}
        expected = {**FIXED_PARAMETERS, "roots": count, "seed": seed, "seat": seat}
        for key, value in expected.items():
            if parameters.get(key) != value:
                raise ValueError(f"{path}: parameter {key} changed")
        if (manifest.get("counts") or {}).get("retained_roots") != count:
            raise ValueError(f"{path}: root count mismatch")
        if (manifest.get("counts") or {}).get("fully_complete_roots") != count:
            raise ValueError(f"{path}: incomplete roots")
        if (manifest.get("counts") or {}).get("rollout_errors", 0) != 0:
            raise ValueError(f"{path}: rollout errors")
        file_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(file_rows) != count:
            raise ValueError(f"{path}: row count mismatch")
        for row in file_rows:
            if (
                row.get("schema") != "target80-plan-selector-root/v4"
                or row.get("protocol") != PROTOCOL
                or row.get("split") != split
                or row.get("shard") != shard
                or not row.get("guarded_baseline_reproduced")
                or not row.get("fully_complete")
                or int(row.get("complete_pairs") or 0) != 16
                or row.get("errors")
            ):
                raise ValueError(f"{path}: invalid row integrity")
            key = (shard, int(row["root_id"]))
            if key in {(item["shard"], int(item["root_id"])) for item in rows}:
                raise ValueError(f"duplicate root {key}")
            vectorize(row)
            stable_teacher(row)
            rows.append(row)
        hashes[shard] = sha256(path)
    required = {
        name for name, (expected_split, _, _, _) in EXPECTED_SHARDS.items() if expected_split == split
    }
    if seen != required:
        raise ValueError(f"{split}: expected shards {sorted(required)}, got {sorted(seen)}")
    return rows, hashes


class Selector(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(48, hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(48, hidden // 2), len(PLANS)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


def train_config(
    config: Mapping[str, Any],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[Selector, list[float]]:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Selector(train_x.shape[1], int(config["hidden"]), float(config["dropout"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    losses = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(int(config["epochs"])):
        order = torch.randperm(train_x.shape[0], generator=generator)
        epoch_loss = 0.0
        model.train()
        for start in range(0, train_x.shape[0], 32):
            indices = order[start : start + 32]
            logits = model(train_x[indices])
            loss = nn.functional.cross_entropy(
                logits,
                train_y[indices],
                weight=class_weights,
                label_smoothing=0.03,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(indices)
        losses.append(epoch_loss / train_x.shape[0])
    model.eval()
    return model, losses


def evaluate(
    model: Selector,
    validation_x: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    deltas: Sequence[Mapping[str, float]],
    threshold: float,
) -> dict[str, Any]:
    with torch.no_grad():
        probabilities = torch.softmax(model(validation_x), dim=-1).cpu().numpy()
    predictions = []
    for values in probabilities:
        nonbaseline_index = int(np.argmax(values[1:])) + 1
        if (
            float(values[nonbaseline_index]) >= threshold
            and float(values[nonbaseline_index]) > float(values[0])
        ):
            predictions.append(PLANS[nonbaseline_index])
        else:
            predictions.append("baseline")
    changed = [index for index, value in enumerate(predictions) if value != "baseline"]
    positives = [index for index, value in enumerate(labels) if value != "baseline"]
    correct_changes = [
        index for index in changed if predictions[index] == labels[index]
    ]
    predicted_deltas = [
        0.0 if prediction == "baseline" else float(deltas[index][prediction])
        for index, prediction in enumerate(predictions)
    ]
    changed_deltas = [predicted_deltas[index] for index in changed]
    seat_means = {}
    for started_first in (True, False):
        selected = [
            predicted_deltas[index]
            for index, row in enumerate(rows)
            if bool(row["started_first"]) == started_first
        ]
        seat_means["first" if started_first else "second"] = mean(selected)
    baseline_rows = [index for index, value in enumerate(labels) if value == "baseline"]
    metrics = {
        "threshold": threshold,
        "change_count": len(changed),
        "change_rate": len(changed) / len(rows),
        "positive_label_count": len(positives),
        "nonbaseline_precision": len(correct_changes) / max(1, len(changed)),
        "nonbaseline_recall": len(correct_changes) / max(1, len(positives)),
        "baseline_retention": sum(
            int(predictions[index] == "baseline") for index in baseline_rows
        )
        / max(1, len(baseline_rows)),
        "mean_teacher_delta_all_rows": mean(predicted_deltas),
        "mean_teacher_delta_changed_rows": mean(changed_deltas),
        "negative_changed_delta_rate": sum(value < -1e-9 for value in changed_deltas)
        / max(1, len(changed_deltas)),
        "seat_mean_teacher_delta": seat_means,
        "prediction_counts": dict(Counter(predictions)),
        "label_counts": dict(Counter(labels)),
    }
    gates = {
        "change_rate_5_to_30pct": 0.05 <= metrics["change_rate"] <= 0.30,
        "nonbaseline_precision_at_least_0_65": metrics["nonbaseline_precision"] >= 0.65,
        "nonbaseline_recall_at_least_0_20": metrics["nonbaseline_recall"] >= 0.20,
        "baseline_retention_at_least_0_90": metrics["baseline_retention"] >= 0.90,
        "mean_teacher_delta_all_rows_at_least_0_04": metrics["mean_teacher_delta_all_rows"] >= 0.04,
        "negative_changed_delta_rate_at_most_0_15": metrics["negative_changed_delta_rate"] <= 0.15,
        "both_seat_mean_deltas_nonnegative": min(seat_means.values()) >= 0.0,
    }
    metrics["gates"] = gates
    metrics["eligible"] = all(gates.values())
    return metrics


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.mkdir(parents=True)
    train_rows, train_hashes = validate_and_load(args.train, "train")
    validation_rows, validation_hashes = validate_and_load(args.validation, "validation")
    train_vectors = np.stack([vectorize(row) for row in train_rows])
    validation_vectors = np.stack([vectorize(row) for row in validation_rows])
    train_teacher = [stable_teacher(row) for row in train_rows]
    validation_teacher = [stable_teacher(row) for row in validation_rows]
    train_labels = [item[0] for item in train_teacher]
    validation_labels = [item[0] for item in validation_teacher]
    validation_deltas = [item[1] for item in validation_teacher]
    label_counts = Counter(train_labels)
    if len(train_rows) != 160 or len(validation_rows) != 48:
        raise ValueError("unexpected train/validation row totals")
    if sum(value != "baseline" for value in train_labels) < 20:
        raise ValueError("training corpus has fewer than 20 stable nonbaseline labels")
    feature_mean = train_vectors.mean(axis=0).astype(np.float32)
    feature_std = train_vectors.std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-5] = 1.0
    train_x = torch.from_numpy((train_vectors - feature_mean) / feature_std)
    validation_x = torch.from_numpy((validation_vectors - feature_mean) / feature_std)
    train_y = torch.tensor([PLAN_INDEX[name] for name in train_labels], dtype=torch.long)
    counts = torch.tensor(
        [max(1, label_counts.get(name, 0)) for name in PLANS],
        dtype=torch.float32,
    )
    class_weights = torch.sqrt(float(len(train_rows)) / (len(PLANS) * counts))
    class_weights = torch.clamp(class_weights, 0.5, 3.0)

    results = []
    checkpoints = {}
    for config in CONFIGS:
        model, losses = train_config(config, train_x, train_y, class_weights)
        config_dir = output / str(config["name"])
        config_dir.mkdir()
        checkpoint_path = config_dir / "selector.pt"
        torch.save(
            {
                "schema": "target80-plan-selector-checkpoint/v4",
                "vector_schema": VECTOR_SCHEMA,
                "plans": PLANS,
                "input_dim": int(train_x.shape[1]),
                "feature_mean": torch.from_numpy(feature_mean),
                "feature_std": torch.from_numpy(feature_std),
                "config": dict(config),
                "state_dict": model.state_dict(),
                "train_source_sha256": train_hashes,
                "validation_source_sha256": validation_hashes,
            },
            checkpoint_path,
        )
        threshold_results = [
            evaluate(
                model,
                validation_x,
                validation_rows,
                validation_labels,
                validation_deltas,
                threshold,
            )
            for threshold in THRESHOLDS
        ]
        results.append(
            {
                "config": dict(config),
                "final_train_loss": losses[-1],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "thresholds": threshold_results,
            }
        )
        checkpoints[str(config["name"])] = checkpoint_path
    eligible = [
        (result, threshold)
        for result in results
        for threshold in result["thresholds"]
        if threshold["eligible"]
    ]
    selected = None
    if eligible:
        result, threshold = max(
            eligible,
            key=lambda item: (
                item[1]["mean_teacher_delta_all_rows"],
                item[1]["nonbaseline_precision"],
                -item[1]["change_rate"],
            ),
        )
        selected = {
            "config": result["config"]["name"],
            "threshold": threshold["threshold"],
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "validation_metrics": threshold,
        }
    payload = {
        "schema": "target80-plan-selector-training-result/v4",
        "protocol": PROTOCOL,
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "collector_sha256": COLLECTOR_SHA256,
        "vector_schema": VECTOR_SCHEMA,
        "input_dim": int(train_x.shape[1]),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_source_sha256": train_hashes,
        "validation_source_sha256": validation_hashes,
        "train_label_counts": dict(label_counts),
        "validation_label_counts": dict(Counter(validation_labels)),
        "configs": results,
        "selected": selected,
        "passes_validation": selected is not None,
        "authority": (
            "fresh selector test collection only"
            if selected is not None
            else "none; close exact V4 learner"
        ),
        "deployment_authority": False,
    }
    (output / "training_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
