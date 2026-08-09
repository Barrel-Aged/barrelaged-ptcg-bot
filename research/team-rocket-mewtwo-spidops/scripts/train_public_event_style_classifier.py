#!/usr/bin/env python3
"""Train and calibrate the locked N4 explicit-public-event classifier."""

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

from train_opponent_style_classifier import (
    calibrate_temperature,
    raw_logits,
    select_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 787701
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
FEATURE_WIDTH = 644


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
    return parser.parse_args()


def load_split(paths: list[Path], split: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    episodes = []
    hashes = {}
    for raw in paths:
        path = raw.resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "rocket-public-event-style-sequences/v1":
            raise ValueError(f"unexpected N4 sequence schema: {path}")
        if payload.get("protocol") != "rocket-public-event-style-gate/v1" or payload.get("split") != split:
            raise ValueError(f"N4 split/protocol mismatch: {path}")
        if payload.get("deck_sha256") != DECK_SHA256 or payload.get("base_model_sha256") != MODEL_SHA256:
            raise ValueError(f"N4 substrate mismatch: {path}")
        if payload.get("hidden_information_used") is not False:
            raise ValueError(f"N4 information boundary failed: {path}")
        for episode in payload.get("episodes") or ():
            tokens = episode["tokens"]
            if tokens.ndim != 2 or tuple(tokens.shape[1:]) != (FEATURE_WIDTH,):
                raise ValueError(f"N4 feature shape mismatch: {path}")
            if not episode.get("started_second") or int(episode.get("exception_fallbacks", 0)):
                raise ValueError(f"N4 correctness mismatch: {path}")
            episodes.append(
                {
                    "tokens": tokens.to(torch.float32),
                    "label": int(episode["label"]),
                    "source": str(path),
                    "game_index": int(episode["game_index"]),
                }
            )
        hashes[str(path)] = sha256(path)
    counts = {label: sum(episode["label"] == label for episode in episodes) for label in (0, 1)}
    if counts[0] != counts[1] or counts[0] == 0:
        raise ValueError(f"N4 split is not balanced: {counts}")
    return episodes, hashes


def loss_for(model: Any, episode: dict[str, Any]) -> torch.Tensor:
    logits = model.sequence(episode["tokens"])
    targets = torch.full_like(logits, float(episode["label"]))
    return F.binary_cross_entropy_with_logits(logits, targets)


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".training.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"refusing existing N4 classifier: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from public_event_style_classifier import PublicEventStyleClassifier

    train, train_hashes = load_split(args.train, "train")
    calibration, calibration_hashes = load_split(args.calibration, "calibration")
    model = PublicEventStyleClassifier(input_width=FEATURE_WIDTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-3)
    rng = random.Random(SEED)
    history = []
    started = time()
    for epoch in range(1, 31):
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        norms = []
        model.train()
        for start in range(0, len(order), 16):
            batch = [train[index] for index in order[start : start + 16]]
            loss = torch.stack([loss_for(model, episode) for episode in batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        record = {"epoch": epoch, "mean_loss": float(np.mean(losses)), "max_gradient_norm": max(norms)}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    calibration_logits = raw_logits(model, calibration)
    temperature = calibrate_temperature(calibration_logits, calibration)
    probabilities = [torch.sigmoid(values / temperature) for values in calibration_logits]
    threshold, metrics = select_threshold(probabilities, calibration)
    payload = {
        "schema": "rocket-public-event-style-classifier/v1",
        "protocol": "rocket-public-event-style-gate/v1",
        "config": {"input_width": FEATURE_WIDTH},
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
        "calibration_metrics": metrics,
    }
    torch.save(payload, output)
    report = {
        "schema": "rocket-public-event-style-training/v1",
        "protocol": "rocket-public-event-style-gate/v1",
        "train_games": len(train),
        "calibration_games": len(calibration),
        "training_seed": SEED,
        "epochs": history,
        "temperature": temperature,
        "activation_threshold": threshold,
        "minimum_decisions": 3,
        "consecutive_required": 2,
        "calibration_metrics": metrics,
        "train_sources": train_hashes,
        "calibration_sources": calibration_hashes,
        "artifact": str(output),
        "artifact_sha256": sha256(output),
        "elapsed_seconds": time() - started,
        "validation_opened": False,
        "claim_boundary": "N4 train/calibration only; validation remains unopened",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
