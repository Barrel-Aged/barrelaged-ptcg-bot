#!/usr/bin/env python3
"""Train and calibrate the locked N3 public-history style classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from time import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 786701
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-episodes", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-width", type=int, default=64)
    return parser.parse_args()


def load_split(paths: list[Path], expected_split: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    episodes = []
    hashes = {}
    labels = set()
    for raw in paths:
        path = raw.resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "rocket-opponent-style-sequences/v1":
            raise ValueError(f"unexpected style data schema: {path}")
        if payload.get("split") != expected_split:
            raise ValueError(f"style split mismatch: {path}")
        if payload.get("deck_sha256") != DECK_SHA256 or payload.get("base_model_sha256") != MODEL_SHA256:
            raise ValueError(f"style data substrate mismatch: {path}")
        if payload.get("hidden_information_used") is not False:
            raise ValueError(f"style data information-boundary mismatch: {path}")
        for episode in payload.get("episodes") or ():
            tokens = episode["tokens"]
            if tokens.ndim != 2 or tokens.shape[1] != 384 or tokens.shape[0] == 0:
                raise ValueError(f"invalid style token sequence: {path}")
            if not episode.get("started_second") or int(episode.get("exception_fallbacks", 0)):
                raise ValueError(f"invalid style episode correctness: {path}")
            labels.add(int(episode["label"]))
            episodes.append(
                {
                    "tokens": tokens.to(torch.float32),
                    "label": int(episode["label"]),
                    "source": str(path),
                    "game_index": int(episode["game_index"]),
                }
            )
        hashes[str(path)] = sha256(path)
    if labels != {0, 1}:
        raise ValueError(f"style split is not class-balanced in coverage: {labels}")
    counts = {label: sum(episode["label"] == label for episode in episodes) for label in (0, 1)}
    if counts[0] != counts[1]:
        raise ValueError(f"style split game counts are imbalanced: {counts}")
    return episodes, hashes


def episode_loss(model: Any, episode: dict[str, Any]) -> torch.Tensor:
    logits = model.sequence(episode["tokens"])
    target = torch.full_like(logits, float(episode["label"]))
    return F.binary_cross_entropy_with_logits(logits, target)


@torch.no_grad()
def raw_logits(model: Any, episodes: list[dict[str, Any]]) -> list[torch.Tensor]:
    model.eval()
    return [model.sequence(episode["tokens"]).detach() for episode in episodes]


def calibrate_temperature(
    logits: list[torch.Tensor], episodes: list[dict[str, Any]]
) -> float:
    log_temperature = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        losses = []
        for values, episode in zip(logits, episodes):
            target = torch.full_like(values, float(episode["label"]))
            losses.append(F.binary_cross_entropy_with_logits(values / temperature, target))
        loss = torch.stack(losses).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(0.05, 20.0).detach())


def activation(probabilities: torch.Tensor, threshold: float) -> tuple[bool, int | None]:
    consecutive = 0
    for index, probability in enumerate(probabilities.tolist(), start=1):
        consecutive = consecutive + 1 if probability >= threshold else 0
        if index >= 3 and consecutive >= 2:
            return True, index
    return False, None


def activation_metrics(
    probabilities: list[torch.Tensor],
    episodes: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    rows = []
    for values, episode in zip(probabilities, episodes):
        active, decision = activation(values, threshold)
        rows.append({"label": episode["label"], "active": active, "decision": decision})
    by_label = {}
    for label in (0, 1):
        subset = [row for row in rows if row["label"] == label]
        decisions = [int(row["decision"]) for row in subset if row["decision"] is not None]
        by_label[str(label)] = {
            "games": len(subset),
            "activations": sum(bool(row["active"]) for row in subset),
            "activation_rate": float(np.mean([row["active"] for row in subset])),
            "median_activation_decision": float(np.median(decisions)) if decisions else None,
        }
    final_predictions = [int(float(values[-1]) >= 0.5) for values in probabilities]
    accuracy = float(
        np.mean([prediction == episode["label"] for prediction, episode in zip(final_predictions, episodes)])
    )
    return {"threshold": threshold, "by_label": by_label, "final_step_accuracy": accuracy}


def select_threshold(
    probabilities: list[torch.Tensor], episodes: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    candidates = sorted(
        {
            0.5,
            1.0,
            *(
                min(1.0, max(0.5, float(value)))
                for values in probabilities
                for value in values.tolist()
                if 0.5 <= float(value) <= 1.0
            ),
        }
    )
    eligible = []
    for threshold in candidates:
        metrics = activation_metrics(probabilities, episodes, threshold)
        false_rate = float(metrics["by_label"]["0"]["activation_rate"])
        recovery_rate = float(metrics["by_label"]["1"]["activation_rate"])
        if false_rate <= 0.01 + 1.0e-12:
            eligible.append((recovery_rate, threshold, metrics))
    if not eligible:
        raise RuntimeError("no calibration threshold satisfies false-activation cap")
    recovery_rate, threshold, metrics = max(eligible, key=lambda item: (item[0], item[1]))
    return float(threshold), metrics


def main() -> int:
    args = parse_args()
    if args.epochs != 30 or args.hidden_width != 64:
        raise ValueError("N3 protocol locks 30 epochs and hidden width 64")
    if args.batch_episodes <= 0 or args.learning_rate <= 0:
        raise ValueError("invalid N3 training parameters")
    output = args.output.resolve()
    report_path = output.with_suffix(".training.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"refusing existing style classifier: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from opponent_style_runtime import OpponentStyleClassifier

    train, train_hashes = load_split(args.train, "train")
    calibration, calibration_hashes = load_split(args.calibration, "calibration")
    model = OpponentStyleClassifier(input_width=384, hidden_width=args.hidden_width)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = random.Random(SEED)
    history = []
    started = time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        rng.shuffle(order)
        model.train()
        epoch_losses = []
        gradient_norms = []
        for start in range(0, len(order), args.batch_episodes):
            batch = [train[index] for index in order[start : start + args.batch_episodes]]
            loss = torch.stack([episode_loss(model, episode) for episode in batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            gradient_norms.append(float(gradient_norm))
        record = {
            "epoch": epoch,
            "mean_loss": float(np.mean(epoch_losses)),
            "max_gradient_norm": max(gradient_norms),
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    calibration_logits = raw_logits(model, calibration)
    temperature = calibrate_temperature(calibration_logits, calibration)
    probabilities = [torch.sigmoid(values / temperature) for values in calibration_logits]
    threshold, calibration_metrics = select_threshold(probabilities, calibration)
    payload = {
        "schema": "rocket-opponent-style-classifier/v1",
        "protocol": "rocket-opponent-style-gate/v1",
        "config": {"input_width": 384, "hidden_width": args.hidden_width},
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "temperature": temperature,
        "activation_threshold": threshold,
        "minimum_decisions": 3,
        "consecutive_required": 2,
        "training_seed": SEED,
        "deck_sha256": DECK_SHA256,
        "base_model_sha256": MODEL_SHA256,
        "train_sources": train_hashes,
        "calibration_sources": calibration_hashes,
        "calibration_metrics": calibration_metrics,
    }
    torch.save(payload, output)
    report = {
        "schema": "rocket-opponent-style-training/v1",
        "protocol": "rocket-opponent-style-gate/v1",
        "train_games": len(train),
        "calibration_games": len(calibration),
        "training_seed": SEED,
        "epochs": history,
        "temperature": temperature,
        "activation_threshold": threshold,
        "minimum_decisions": 3,
        "consecutive_required": 2,
        "calibration_metrics": calibration_metrics,
        "train_sources": train_hashes,
        "calibration_sources": calibration_hashes,
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "elapsed_seconds": time() - started,
        "validation_opened": False,
        "claim_boundary": "train/calibration evidence only; untouched validation remains unopened",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
