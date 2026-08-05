#!/usr/bin/env python3
"""Calibrate the Rocket policy's critic from complete official arena games.

The deployed policy was transferred from a different deck, so its value head
was never trained to predict Rocket outcomes.  This script keeps every policy
parameter frozen and fits only the value MLP to game-level win/loss targets.
Games are split before feature extraction and each game contributes equal
total weight, preventing long games from dominating the critic.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fnn

from train_live_outcome_awr import GameInfo, read_game, split_games
from train_public_leader_v110_bc import Example, collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ValueExample:
    example: Example
    target: float
    weight: float
    progress: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "leader_bc_scale_v2",
    )
    parser.add_argument("--replays", type=Path, default=PROJECT_ROOT / "work" / "arena_scale_v2_609")
    parser.add_argument("--recorded-bot", default="HelbrosRocketSpidopsScaleBCV2")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_outcome_value",
    )
    parser.add_argument("--max-states-per-game", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(count: int, maximum: int) -> list[int]:
    if maximum <= 0 or count <= maximum:
        return list(range(count))
    return sorted(set(int(round(value)) for value in np.linspace(0, count - 1, maximum)))


def extract_split(
    games: list[GameInfo],
    recorded_bot: str,
    deck: list[int],
    deck_library: Any,
    belief_class: Any,
    features: Any,
    max_states_per_game: int,
) -> tuple[list[ValueExample], Counter[str]]:
    examples: list[ValueExample] = []
    counts: Counter[str] = Counter()
    for game in games:
        _, decisions = read_game(game.path, recorded_bot)
        belief = belief_class(deck, opponent_deck_priors=deck_library)
        game_features: list[dict[str, np.ndarray]] = []
        for record in decisions:
            observation = record.get("obs") or {}
            selection = observation.get("select") or {}
            if not selection.get("option"):
                counts["empty_options"] += 1
                continue
            try:
                game_features.append(
                    features.extract(observation, belief=belief, slot_link=True)
                )
            except Exception as error:
                counts[f"feature_error:{type(error).__name__}"] += 1
        selected = evenly_spaced_indices(len(game_features), max_states_per_game)
        if not selected:
            counts["empty_game"] += 1
            continue
        target = 1.0 if game.won else -1.0
        per_state_weight = 1.0 / len(selected)
        denominator = max(1, len(game_features) - 1)
        for index in selected:
            examples.append(
                ValueExample(
                    Example(game_features[index], 0, game.path.name),
                    target,
                    per_state_weight,
                    index / denominator,
                )
            )
        counts["games"] += 1
        counts["wins"] += int(game.won)
        counts["states"] += len(selected)
    return examples, counts


@torch.no_grad()
def metrics(model, examples: list[ValueExample], batch_size: int, device) -> dict[str, float | int]:
    model.eval()
    squared = absolute = weighted = weight_total = 0.0
    correct = 0
    predictions: list[float] = []
    targets: list[float] = []
    early_sq = late_sq = 0.0
    early_n = late_n = 0
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        batch = collate([value.example for value in selected], device)
        _, values = model.forward_batch(batch)
        prediction = values.detach().cpu().numpy()
        target = np.asarray([value.target for value in selected], dtype=np.float32)
        weights = np.asarray([value.weight for value in selected], dtype=np.float32)
        error = prediction - target
        squared += float(np.square(error).sum())
        absolute += float(np.abs(error).sum())
        weighted += float((np.square(error) * weights).sum())
        weight_total += float(weights.sum())
        correct += int(((prediction >= 0) == (target > 0)).sum())
        for item, err in zip(selected, error):
            if item.progress <= 0.33:
                early_sq += float(err * err)
                early_n += 1
            if item.progress >= 0.67:
                late_sq += float(err * err)
                late_n += 1
        predictions.extend(float(value) for value in prediction)
        targets.extend(float(value) for value in target)
    count = len(examples)
    win_predictions = [p for p, t in zip(predictions, targets) if t > 0]
    loss_predictions = [p for p, t in zip(predictions, targets) if t < 0]
    return {
        "examples": count,
        "mse": squared / max(1, count),
        "mae": absolute / max(1, count),
        "game_balanced_mse": weighted / max(weight_total, 1.0e-12),
        "sign_accuracy": correct / max(1, count),
        "mean_prediction_win": float(np.mean(win_predictions)) if win_predictions else 0.0,
        "mean_prediction_loss": float(np.mean(loss_predictions)) if loss_predictions else 0.0,
        "early_mse": early_sq / max(1, early_n),
        "late_mse": late_sq / max(1, late_n),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    candidate = args.candidate.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(candidate))

    from soldier import features
    from soldier.belief import Belief
    from soldier.net import load_policy, save_policy

    deck = [int(value) for value in (candidate / "deck.csv").read_text(encoding="utf-8").split()]
    if len(deck) != 60:
        raise ValueError(f"expected a 60-card deck, found {len(deck)}")
    library_path = candidate / "deck_library.json"
    deck_library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else None
    paths = sorted(args.replays.resolve().glob("*.jsonl"))
    games: list[GameInfo] = []
    for path in paths:
        try:
            game, _ = read_game(path, args.recorded_bot)
            games.append(game)
        except Exception:
            continue
    splits = split_games(games, args.seed)
    data: dict[str, list[ValueExample]] = {}
    extraction: dict[str, dict[str, int]] = {}
    for name, split in splits.items():
        data[name], counts = extract_split(
            split,
            args.recorded_bot,
            deck,
            deck_library,
            Belief,
            features,
            args.max_states_per_game,
        )
        extraction[name] = dict(counts)

    model_path = candidate / "model.pt"
    model = load_policy(model_path, map_location=device).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.value.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    base = {name: metrics(model, values, args.batch_size, device) for name, values in data.items()}

    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = output / "model.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data["train"])))
        random.shuffle(order)
        model.train()
        total_loss = total_weight = 0.0
        for start in range(0, len(order), args.batch_size):
            selected = [data["train"][index] for index in order[start : start + args.batch_size]]
            batch = collate([value.example for value in selected], device)
            targets = torch.as_tensor([value.target for value in selected], dtype=torch.float32, device=device)
            weights = torch.as_tensor([value.weight for value in selected], dtype=torch.float32, device=device)
            _, predictions = model.forward_batch(batch)
            per_item = Fnn.smooth_l1_loss(predictions, targets, reduction="none")
            loss = (per_item * weights).sum() / weights.sum().clamp(min=1.0e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float((per_item.detach() * weights).sum())
            total_weight += float(weights.sum())
        validation = metrics(model, data["validation"], args.batch_size, device)
        record = {
            "epoch": epoch,
            "train_game_balanced_smooth_l1": total_loss / max(total_weight, 1.0e-12),
            "validation": validation,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if float(validation["game_balanced_mse"]) < best_loss:
            best_loss = float(validation["game_balanced_mse"])
            best_epoch = epoch
            stale = 0
            save_policy(model, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    best = load_policy(best_path, map_location=device).to(device)
    tuned = {name: metrics(best, values, args.batch_size, device) for name, values in data.items()}
    report = {
        "schema_version": 1,
        "method": "game-balanced Monte Carlo Rocket critic calibration",
        "official_engine_source": True,
        "policy_parameters_changed": False,
        "value_parameters_changed": True,
        "candidate": str(candidate),
        "candidate_model_sha256": sha256(model_path),
        "replays": str(args.replays.resolve()),
        "recorded_bot": args.recorded_bot,
        "split_games": {name: [game.path.name for game in split] for name, split in splits.items()},
        "extraction": extraction,
        "hyperparameters": {
            key: value for key, value in vars(args).items() if isinstance(value, (int, float, str, bool))
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "base_metrics": base,
        "history": history,
        "best_epoch": best_epoch,
        "tuned_metrics": tuned,
        "model": str(best_path),
        "model_sha256": sha256(best_path),
        "elapsed_seconds": time.time() - started,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "model_sha256": report["model_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
