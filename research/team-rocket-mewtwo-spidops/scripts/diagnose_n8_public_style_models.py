#!/usr/bin/env python3
"""Development-only separability audit for N8 public opponent-plan models.

This script deliberately uses only the N4 training and calibration shards.  It
does not open N4's sealed validation shards and cannot authorize deployment.
The positive class is V110-like behavior because the N7 exploit is beneficial
only against that opponent; Recovery V4 false activation is the safety error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from time import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "neural_plan_n4" / "style_data"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "n8_safe_style_overlay"
    / "public_style_separability.json"
)
FEATURE_WIDTH = 644
SEED = 796001


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=30)
    return parser.parse_args()


def load(paths: Iterable[Path], split: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    episodes: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "rocket-public-event-style-sequences/v1":
            raise ValueError(f"unexpected N4 schema: {path}")
        if payload.get("split") != split or payload.get("hidden_information_used") is not False:
            raise ValueError(f"N4 split/information boundary mismatch: {path}")
        for episode in payload.get("episodes") or ():
            tokens = episode["tokens"].to(torch.float32)
            if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] != FEATURE_WIDTH:
                raise ValueError(f"invalid public sequence: {path}")
            # N4 label 0 is V110 and label 1 is Recovery.  N8 activates only
            # the V110 exploit, so reverse the target.
            episodes.append(
                {
                    "tokens": tokens,
                    "target": int(int(episode["label"]) == 0),
                    "game_index": int(episode["game_index"]),
                    "source": str(path.resolve()),
                }
            )
        hashes[str(path.resolve())] = digest(path)
    return episodes, hashes


class TokenLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_WIDTH)
        self.head = nn.Linear(FEATURE_WIDTH, 1)

    def sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(tokens)).squeeze(-1)


class TokenMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_WIDTH)
        self.body = nn.Sequential(
            nn.Linear(FEATURE_WIDTH, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.body(self.norm(tokens)).squeeze(-1)


class PublicGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(FEATURE_WIDTH)
        self.input = nn.Linear(FEATURE_WIDTH, 96)
        self.gru = nn.GRU(96, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        values = F.gelu(self.input(self.norm(tokens))).unsqueeze(0)
        hidden, _ = self.gru(values)
        return self.head(hidden.squeeze(0)).squeeze(-1)


def episode_loss(model: nn.Module, episode: dict[str, Any]) -> torch.Tensor:
    logits = model.sequence(episode["tokens"])
    targets = torch.full_like(logits, float(episode["target"]))
    # Give every game equal weight regardless of trajectory length.
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def probabilities(model: nn.Module, episodes: list[dict[str, Any]]) -> list[torch.Tensor]:
    model.eval()
    return [torch.sigmoid(model.sequence(episode["tokens"])).cpu() for episode in episodes]


def activates(values: torch.Tensor, threshold: float) -> tuple[bool, int | None]:
    consecutive = 0
    for index, probability in enumerate(values.tolist(), start=1):
        consecutive = consecutive + 1 if probability >= threshold else 0
        if index >= 3 and consecutive >= 2:
            return True, index
    return False, None


def activation_rows(
    values: list[torch.Tensor], episodes: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    rows = []
    for sequence, episode in zip(values, episodes):
        active, decision = activates(sequence, threshold)
        rows.append(
            {
                "target": int(episode["target"]),
                "active": bool(active),
                "decision": decision,
                "final_probability": float(sequence[-1]),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target, name in ((0, "recovery_v4"), (1, "v110")):
        subset = [row for row in rows if row["target"] == target]
        decisions = [int(row["decision"]) for row in subset if row["decision"] is not None]
        result[name] = {
            "games": len(subset),
            "activations": sum(bool(row["active"]) for row in subset),
            "activation_rate": float(np.mean([row["active"] for row in subset])),
            "median_activation_decision": float(np.median(decisions)) if decisions else None,
        }
    finals = np.asarray([row["final_probability"] for row in rows], dtype=np.float64)
    labels = np.asarray([row["target"] for row in rows], dtype=np.int64)
    # Rank-statistic AUC with deterministic average ranks for ties.
    order = np.argsort(finals, kind="mergesort")
    ranks = np.empty(len(finals), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and finals[order[end]] == finals[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    auc = (float(ranks[positives].sum()) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    result["final_step_auc"] = auc
    result["final_step_accuracy_at_half"] = float(np.mean((finals >= 0.5) == labels))
    return result


def choose_threshold(values: list[torch.Tensor], episodes: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    candidates = sorted(
        {1.0, *[float(value) for sequence in values for value in sequence.tolist()]},
        reverse=True,
    )
    eligible: list[tuple[float, float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = summarize(activation_rows(values, episodes, threshold))
        recovery_rate = float(metrics["recovery_v4"]["activation_rate"])
        v110_rate = float(metrics["v110"]["activation_rate"])
        if recovery_rate <= 0.01 + 1.0e-12:
            eligible.append((v110_rate, threshold, metrics))
    if not eligible:
        raise RuntimeError("no threshold satisfies Recovery false-activation cap")
    _, threshold, metrics = max(eligible, key=lambda item: (item[0], item[1]))
    return threshold, metrics


def train_model(
    name: str,
    model: nn.Module,
    fit: list[dict[str, Any]],
    tune: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    epochs: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-4, weight_decay=1.0e-3)
    rng = random.Random(SEED + sum(map(ord, name)))
    history = []
    for epoch in range(1, epochs + 1):
        order = list(range(len(fit)))
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), 16):
            batch = [fit[index] for index in order[start : start + 16]]
            loss = torch.stack([episode_loss(model, episode) for episode in batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "mean_loss": float(np.mean(losses))})
    tune_values = probabilities(model, tune)
    threshold, tune_metrics = choose_threshold(tune_values, tune)
    evaluation_metrics = summarize(
        activation_rows(probabilities(model, evaluation), evaluation, threshold)
    )
    return {
        "model": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "threshold_selected_on_tune": threshold,
        "fit_final_loss": history[-1]["mean_loss"],
        "tune": tune_metrics,
        "calibration_holdout": evaluation_metrics,
        "history": history,
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    data = args.data.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing diagnostic: {output}")
    train, train_hashes = load(sorted(data.glob("train_*.pt")), "train")
    evaluation, calibration_hashes = load(
        sorted(data.glob("calibration_*.pt")), "calibration"
    )
    # Deterministic within-class split: 150 fit and 50 threshold-tuning games
    # per opponent.  N4 calibration remains a diagnostic holdout.
    fit: list[dict[str, Any]] = []
    tune: list[dict[str, Any]] = []
    for target in (0, 1):
        subset = sorted(
            (episode for episode in train if episode["target"] == target),
            key=lambda episode: (episode["source"], episode["game_index"]),
        )
        if len(subset) != 200:
            raise ValueError(f"expected 200 N4 training games per class, got {len(subset)}")
        fit.extend(subset[:150])
        tune.extend(subset[150:])
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    started = time()
    results = []
    for name, constructor in (
        ("token_linear", TokenLinear),
        ("token_mlp", TokenMLP),
        ("public_gru", PublicGRU),
    ):
        torch.manual_seed(SEED + sum(map(ord, name)))
        result = train_model(name, constructor(), fit, tune, evaluation, args.epochs)
        results.append(result)
        print(json.dumps({key: result[key] for key in result if key != "history"}, sort_keys=True), flush=True)
    report = {
        "schema": "rocket-n8-public-style-separability/v1",
        "development_only": True,
        "deployment_authority": False,
        "hidden_information_used": False,
        "positive_class": "AshMarnieGeneralV110-like",
        "safety_error": "false activation against HelbrosMarnieGeneralRecoveryV4",
        "activation_rule": {"minimum_decisions": 3, "consecutive_required": 2},
        "threshold_tune_false_activation_cap": 0.01,
        "seed": SEED,
        "epochs": args.epochs,
        "fit_games": len(fit),
        "tune_games": len(tune),
        "calibration_holdout_games": len(evaluation),
        "train_sources": train_hashes,
        "calibration_sources": calibration_hashes,
        "sealed_validation_opened": False,
        "results": results,
        "elapsed_seconds": time() - started,
        "claim_boundary": "separability diagnosis only; no policy-strength or deployment claim",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
