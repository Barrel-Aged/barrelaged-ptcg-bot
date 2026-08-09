#!/usr/bin/env python3
"""Cross-fitted state-residual action-effect learner for Target-80 V2."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from time import monotonic
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from target80_randomized_advantage_v1 import (
    CACHE_SCHEMA,
    CORPUS_SHA256,
    GROUPS,
    binary_auc,
    calibration_probability,
    fit_logistic_calibration,
    sha256,
    state_dict_cpu,
    weighted_mean,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CACHE = PROJECT_ROOT / "work" / "target80_randomized_advantage_v1_cache" / "train_v2.pt"
DEVELOPMENT_CACHE = PROJECT_ROOT / "work" / "target80_randomized_advantage_v1_cache" / "test_v2.pt"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "target80_residual_advantage_v2"
STATE_SCHEMA = "target80-crossfit-state-baseline/v2"
MODEL_SCHEMA = "target80-residual-advantage-heads/v2"
EVALUATION_SCHEMA = "target80-residual-advantage-evaluation/v2"
STATE_FOLDS = 5
STATE_EPOCHS = 8
STATE_WIDTH = 128
STATE_SEED = 855100
SQRT_PROPENSITY_WEIGHT_CLIP = 4.0
THRESHOLDS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30)
CONFIGS: dict[str, dict[str, float | int]] = {
    "low_reg": {"regularization": 0.01, "seed": 855201},
    "balanced": {"regularization": 0.03, "seed": 855202},
    "high_reg": {"regularization": 0.08, "seed": 855203},
}


class StateHead(nn.Module):
    def __init__(self, hidden_dim: int, width: int = STATE_WIDTH) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(state).squeeze(-1))


class ResidualAdvantageHead(nn.Module):
    """State-option residual centered under the known behavior mixture."""

    def __init__(self, hidden_dim: int, width: int = 96) -> None:
        super().__init__()
        self.state_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU()
        )
        self.option_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU()
        )
        self.option_bias = nn.Linear(hidden_dim, 1)
        self.reference_scale = nn.Parameter(torch.zeros(()))
        self.scale = math.sqrt(width)

    def forward(
        self,
        state: torch.Tensor,
        option: torch.Tensor,
        reference_logit: torch.Tensor,
        behavior_probability: torch.Tensor,
    ) -> torch.Tensor:
        state_hidden = self.state_projection(state)
        option_hidden = self.option_projection(option)
        raw = torch.einsum("bh,boh->bo", state_hidden, option_hidden) / self.scale
        raw = raw + self.option_bias(option).squeeze(-1)
        raw = raw + self.reference_scale * reference_logit.clamp(-20.0, 20.0)
        bounded = 0.75 * torch.tanh(raw)
        return bounded - (bounded * behavior_probability).sum(1, keepdim=True)


def validate_cache(cache: Mapping[str, Any], split: str | None = None) -> None:
    if cache.get("schema") != CACHE_SCHEMA or cache.get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("V2 cache provenance mismatch")
    if split is not None and cache.get("split") != split:
        raise ValueError(f"V2 expected {split} cache")


def game_equal_weight(cache: Mapping[str, Any], rows: torch.Tensor) -> torch.Tensor:
    game = cache["game_id"][rows].long()
    weight = 1.0 / cache["game_lengths"][game].float().clamp_min(1.0)
    return weight / weight.mean().clamp_min(1.0e-12)


def train_one_state(
    cache: Mapping[str, Any], train_indices: list[int], *, seed: int,
    epochs: int, batch_size: int, device: torch.device
) -> StateHead:
    torch.manual_seed(seed)
    hidden = int(cache["global_token"].shape[1])
    model = StateHead(hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    for epoch in range(1, epochs + 1):
        order = list(train_indices); random.Random(seed + epoch).shuffle(order); model.train()
        for start in range(0, len(order), batch_size):
            rows = torch.tensor(order[start:start + batch_size], dtype=torch.long)
            state = cache["global_token"][rows].float().to(device)
            target = cache["outcome"][rows].float().to(device)
            weight = game_equal_weight(cache, rows).to(device)
            prediction = model(state)
            loss = weighted_mean(F.smooth_l1_loss(prediction, target, reduction="none"), weight)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model.eval()


@torch.no_grad()
def state_predict(
    cache: Mapping[str, Any], model: StateHead, indices: Sequence[int],
    *, batch_size: int, device: torch.device
) -> torch.Tensor:
    result = torch.empty(len(indices)); model.eval()
    for start in range(0, len(indices), batch_size):
        selected = list(indices[start:start + batch_size])
        state = cache["global_token"][selected].float().to(device)
        result[start:start + len(selected)] = model(state).cpu()
    return result


def train_state_baseline(
    *, cache_path: Path, output: Path, batch_size: int, threads: int
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing V2 state output: {output}")
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=True)
    validate_cache(cache, "train")
    torch.set_num_threads(max(1, threads)); device = torch.device("cpu")
    count = int(cache["outcome"].numel()); all_indices = list(range(count))
    game_fold = (cache["game_id"].long() * 37 + 13) % STATE_FOLDS
    crossfit = torch.empty(count); fold_rows = []; started = monotonic()
    for fold in range(STATE_FOLDS):
        validation = (game_fold == fold).nonzero().flatten().tolist()
        training = (game_fold != fold).nonzero().flatten().tolist()
        model = train_one_state(
            cache, training, seed=STATE_SEED + fold, epochs=STATE_EPOCHS,
            batch_size=batch_size, device=device,
        )
        crossfit[validation] = state_predict(cache, model, validation, batch_size=batch_size, device=device)
        fold_rows.append({"fold": fold, "train_choices": len(training), "held_out_choices": len(validation)})
        print(json.dumps(fold_rows[-1], sort_keys=True), flush=True)
    final_model = train_one_state(
        cache, all_indices, seed=STATE_SEED + 100, epochs=STATE_EPOCHS,
        batch_size=batch_size, device=device,
    )
    labels = (cache["outcome"].float() > 0).long()
    calibration = fit_logistic_calibration(crossfit, labels)
    probabilities = calibration_probability(crossfit, calibration)
    checkpoint = {
        "schema": STATE_SCHEMA, "hidden_dim": int(cache["global_token"].shape[1]),
        "width": STATE_WIDTH, "folds": STATE_FOLDS, "epochs": STATE_EPOCHS,
        "seed": STATE_SEED, "final_state": state_dict_cpu(final_model),
        "crossfit_prediction": crossfit, "calibration": calibration,
        "cache_sha256": sha256(cache_path.resolve()), "corpus_sha256": CORPUS_SHA256,
    }
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "state_baseline.pt"; torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": STATE_SCHEMA, "cache": str(cache_path.resolve()),
        "cache_sha256": sha256(cache_path.resolve()), "choices": count,
        "games": len(cache["files"]), "folds": fold_rows,
        "crossfit_auc": binary_auc(crossfit.tolist(), labels.tolist()),
        "crossfit_brier": float((probabilities - labels.float()).square().mean()),
        "crossfit_mae_raw": float((crossfit - cache["outcome"].float()).abs().mean()),
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": monotonic() - started, "development_only": True,
        "deployment_authority": False,
    }
    write_json(output / "training_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True); return report


def load_state(path: Path) -> tuple[dict[str, Any], StateHead]:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported V2 state checkpoint")
    model = StateHead(int(checkpoint["hidden_dim"]), int(checkpoint["width"]))
    model.load_state_dict(checkpoint["final_state"])
    return checkpoint, model.eval()


@dataclass(frozen=True)
class ActionBatch:
    state: torch.Tensor
    option: torch.Tensor
    reference_logit: torch.Tensor
    reference_probability: torch.Tensor
    behavior_probability: torch.Tensor
    support: torch.Tensor
    action: torch.Tensor
    baseline: torch.Tensor
    outcome: torch.Tensor
    propensity: torch.Tensor
    weight: torch.Tensor
    index: torch.Tensor
    group: torch.Tensor
    explored: torch.Tensor


def collate_action(
    cache: Mapping[str, Any], indices: Sequence[int], device: torch.device
) -> ActionBatch:
    rows = [int(index) for index in indices]; offset: torch.Tensor = cache["offset"]
    counts = [int(offset[index + 1] - offset[index]) for index in rows]
    maximum = max(counts); hidden = int(cache["global_token"].shape[1])
    option = torch.zeros(len(rows), maximum, hidden)
    reference = torch.full((len(rows), maximum), -1.0e9)
    reference_probability = torch.zeros(len(rows), maximum)
    behavior = torch.zeros(len(rows), maximum)
    support = torch.zeros(len(rows), maximum, dtype=torch.bool)
    for row, (index, count) in enumerate(zip(rows, counts, strict=True)):
        begin, end = int(offset[index]), int(offset[index]) + count
        option[row, :count] = cache["option_token"][begin:end].float()
        reference[row, :count] = cache["reference_logit"][begin:end]
        reference_probability[row, :count] = cache["reference_probability"][begin:end]
        behavior[row, :count] = cache["behavior_probability"][begin:end]
        support[row, :count] = cache["support"][begin:end]
    action = cache["action"][rows].long(); propensity = cache["propensity"][rows].float()
    # Square-root propensity correction retains randomized-action emphasis while
    # avoiding the severe ESS collapse observed with full 1/p weighting in V1.
    ratio = torch.clamp(
        torch.rsqrt(propensity.clamp_min(1.0e-8)),
        max=SQRT_PROPENSITY_WEIGHT_CLIP,
    )
    row_tensor = torch.tensor(rows, dtype=torch.long)
    base = game_equal_weight(cache, row_tensor)
    weight = base * ratio; weight = weight / weight.mean().clamp_min(1.0e-12)
    return ActionBatch(
        cache["global_token"][rows].float().to(device), option.to(device), reference.to(device),
        reference_probability.to(device), behavior.to(device), support.to(device),
        action.to(device), cache["baseline"][rows].long().to(device),
        cache["outcome"][rows].float().to(device), propensity.to(device), weight.to(device),
        row_tensor.to(device), cache["group_id"][rows].long().to(device),
        cache["explored"][rows].bool().to(device),
    )


def bootstrap_mask(indices: torch.Tensor, head: int) -> torch.Tensor:
    return ((indices * 43 + head * 19 + 7) % 5) != 0


def advantage_penalty(
    advantage: torch.Tensor, support: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    row = (advantage.square() * support.float()).sum(1) / support.sum(1).clamp_min(1)
    return weighted_mean(row, weight)


@torch.no_grad()
def predict(
    cache: Mapping[str, Any], state_model: StateHead,
    head1: ResidualAdvantageHead, head2: ResidualAdvantageHead,
    *, batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    count = int(cache["action"].numel())
    result = {name: torch.empty(count) for name in (
        "state", "adv1_taken", "adv2_taken", "q1_taken", "q2_taken",
        "margin1", "margin2",
    )}
    result["best1"] = torch.empty(count, dtype=torch.long)
    result["best2"] = torch.empty(count, dtype=torch.long)
    state_model.eval(); head1.eval(); head2.eval()
    for start in range(0, count, batch_size):
        rows = list(range(start, min(count, start + batch_size)))
        batch = collate_action(cache, rows, device); row = torch.arange(len(rows), device=device)
        state = state_model(batch.state)
        advantage1 = head1(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
        advantage2 = head2(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
        masked1 = advantage1.masked_fill(~batch.support, -1.0e9)
        masked2 = advantage2.masked_fill(~batch.support, -1.0e9)
        best1, best2 = masked1.argmax(1), masked2.argmax(1)
        sl = slice(start, start + len(rows))
        selected1, selected2 = advantage1[row, batch.action], advantage2[row, batch.action]
        result["state"][sl] = state.cpu()
        result["adv1_taken"][sl] = selected1.cpu(); result["adv2_taken"][sl] = selected2.cpu()
        result["q1_taken"][sl] = (state + selected1).cpu(); result["q2_taken"][sl] = (state + selected2).cpu()
        result["best1"][sl] = best1.cpu(); result["best2"][sl] = best2.cpu()
        result["margin1"][sl] = (advantage1[row, best1] - advantage1[row, batch.baseline]).cpu()
        result["margin2"][sl] = (advantage2[row, best2] - advantage2[row, batch.baseline]).cpu()
    return result


def stabilized_propensity_weights(cache: Mapping[str, Any]) -> torch.Tensor:
    ratio = torch.clamp(
        torch.rsqrt(cache["propensity"].float().clamp_min(1e-8)),
        max=SQRT_PROPENSITY_WEIGHT_CLIP,
    )
    game = cache["game_id"].long()
    return ratio / cache["game_lengths"][game].float().clamp_min(1)


def effective_sample(cache: Mapping[str, Any], mask: torch.Tensor) -> tuple[float, float]:
    weight = stabilized_propensity_weights(cache)[mask]
    ess = float(weight.sum().square() / weight.square().sum().clamp_min(1e-12))
    return ess, ess / max(1, int(mask.sum()))


def train_action(
    *, cache_path: Path, state_path: Path, output: Path, config_name: str,
    epochs: int, batch_size: int, threads: int
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing V2 action output: {output}")
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=True)
    validate_cache(cache, "train"); state_checkpoint, state_model = load_state(state_path)
    if state_checkpoint.get("cache_sha256") != sha256(cache_path.resolve()):
        raise ValueError("V2 crossfit state/cache mismatch")
    crossfit = state_checkpoint["crossfit_prediction"].float()
    config = CONFIGS[config_name]; seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads)); device = torch.device("cpu")
    hidden = int(cache["global_token"].shape[1]); head1 = ResidualAdvantageHead(hidden).to(device)
    torch.manual_seed(seed + 1); head2 = ResidualAdvantageHead(hidden).to(device)
    parameters = list(head1.parameters()) + list(head2.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3.0e-4, weight_decay=1.0e-4)
    count = int(cache["action"].numel()); history = []; started = monotonic()
    for epoch in range(1, epochs + 1):
        order = list(range(count)); random.Random(seed + epoch).shuffle(order)
        totals = Counter(); updates = 0; head1.train(); head2.train()
        for start in range(0, count, batch_size):
            rows = order[start:start + batch_size]; batch = collate_action(cache, rows, device)
            row = torch.arange(len(rows), device=device)
            advantage1 = head1(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
            advantage2 = head2(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
            target = batch.outcome - crossfit[batch.index.cpu()].to(device)
            selected1, selected2 = advantage1[row, batch.action], advantage2[row, batch.action]
            weight1 = batch.weight * bootstrap_mask(batch.index, 1).float()
            weight2 = batch.weight * bootstrap_mask(batch.index, 2).float()
            loss1 = weighted_mean(F.smooth_l1_loss(selected1, target, reduction="none"), weight1)
            loss2 = weighted_mean(F.smooth_l1_loss(selected2, target, reduction="none"), weight2)
            regularization = float(config["regularization"]) * (
                advantage_penalty(advantage1, batch.support, weight1)
                + advantage_penalty(advantage2, batch.support, weight2)
            )
            loss = loss1 + loss2 + regularization
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
            totals["head1"] += float(loss1.detach()); totals["head2"] += float(loss2.detach())
            totals["regularization"] += float(regularization.detach()); updates += 1
        row_history = {"epoch": epoch, **{key: value / max(1, updates) for key, value in totals.items()},
                       "seconds": monotonic() - started}
        history.append(row_history); print(json.dumps(row_history, sort_keys=True), flush=True)
    prediction = predict(cache, state_model, head1, head2, batch_size=batch_size, device=device)
    # Train-only calibration uses cross-fitted state predictions so the state
    # component was never fit on the calibrated row's own game.
    advantage = torch.minimum(prediction["adv1_taken"], prediction["adv2_taken"])
    q_crossfit = crossfit + advantage
    labels = (cache["outcome"].float() > 0).long()
    q_calibration = fit_logistic_calibration(q_crossfit, labels)
    checkpoint = {
        "schema": MODEL_SCHEMA, "config_name": config_name, "config": config,
        "epochs": epochs, "hidden_dim": hidden, "width": 96,
        "head1": state_dict_cpu(head1), "head2": state_dict_cpu(head2),
        "q_calibration": q_calibration, "state_calibration": state_checkpoint["calibration"],
        "state_checkpoint_sha256": sha256(state_path.resolve()),
        "cache_sha256": sha256(cache_path.resolve()), "corpus_sha256": CORPUS_SHA256,
        "weighting": "square-root inverse propensity, game normalized",
        "weight_clip": SQRT_PROPENSITY_WEIGHT_CLIP,
    }
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "residual_advantage_heads.pt"; torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": MODEL_SCHEMA, "config_name": config_name, "config": config,
        "epochs": epochs, "history": history, "cache": str(cache_path.resolve()),
        "state_checkpoint": str(state_path.resolve()), "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "trainer_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": monotonic() - started, "development_only": True,
        "deployment_authority": False,
    }
    write_json(output / "training_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True); return report


def load_action(path: Path) -> tuple[dict[str, Any], ResidualAdvantageHead, ResidualAdvantageHead]:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported V2 action checkpoint")
    head1 = ResidualAdvantageHead(int(checkpoint["hidden_dim"]), int(checkpoint["width"]))
    head2 = ResidualAdvantageHead(int(checkpoint["hidden_dim"]), int(checkpoint["width"]))
    head1.load_state_dict(checkpoint["head1"]); head2.load_state_dict(checkpoint["head2"])
    return checkpoint, head1.eval(), head2.eval()


def evaluation_metrics(
    cache: Mapping[str, Any], prediction: Mapping[str, torch.Tensor],
    checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    outcome = cache["outcome"].float(); labels = (outcome > 0).long()
    state = prediction["state"]
    advantage = torch.minimum(prediction["adv1_taken"], prediction["adv2_taken"])
    q = torch.minimum(prediction["q1_taken"], prediction["q2_taken"])
    state_probability = calibration_probability(state, checkpoint["state_calibration"])
    q_probability = calibration_probability(q, checkpoint["q_calibration"])
    explored = cache["explored"].bool(); group = cache["group_id"].long()
    game_lengths = cache["game_lengths"].long()
    starts = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(game_lengths, 0)[:-1]))
    first_mask = torch.zeros_like(explored); first_mask[starts] = True
    same_best = prediction["best1"] == prediction["best2"]
    nonbaseline = prediction["best1"] != cache["baseline"].long()
    margin = torch.minimum(prediction["margin1"], prediction["margin2"])

    def subset(mask: torch.Tensor) -> dict[str, Any]:
        count = int(mask.sum())
        if count == 0: return {"choices": 0}
        y = labels[mask]; target_residual = outcome[mask] - state[mask]
        ess, ratio = effective_sample(cache, mask)
        return {
            "choices": count, "wins": int(y.sum()),
            "q_auc": binary_auc(q[mask].tolist(), y.tolist()),
            "state_auc": binary_auc(state[mask].tolist(), y.tolist()),
            "q_brier": float((q_probability[mask] - y.float()).square().mean()),
            "state_brier": float((state_probability[mask] - y.float()).square().mean()),
            "advantage_residual_mae": float((advantage[mask] - target_residual).abs().mean()),
            "zero_residual_mae": float(target_residual.abs().mean()),
            "effective_sample_size": ess, "effective_sample_ratio": ratio,
        }
    thresholds = {}
    for threshold in THRESHOLDS:
        change = same_best & nonbaseline & (margin >= threshold)
        thresholds[str(threshold)] = {"changes": int(change.sum()),
                                      "change_rate": float(change.float().mean())}
    return {
        "choices": len(outcome), "games": len(cache["files"]),
        "overall": subset(torch.ones_like(explored)), "explored_actions": subset(explored),
        "baseline_actions": subset(~explored), "first_eligible_per_game": subset(first_mask),
        "by_group": {GROUPS[index]: subset(group == index) for index in range(len(GROUPS))},
        "head_best_agreement": float(same_best.float().mean()), "threshold_grid": thresholds,
    }


def evaluate(
    *, cache_path: Path, state_path: Path, heads_path: Path, output: Path,
    batch_size: int, threads: int
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists(): raise FileExistsError(f"refusing existing V2 evaluation: {output}")
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=True); validate_cache(cache)
    state_checkpoint, state_model = load_state(state_path)
    checkpoint, head1, head2 = load_action(heads_path)
    if checkpoint["state_checkpoint_sha256"] != sha256(state_path.resolve()):
        raise ValueError("V2 action/state provenance mismatch")
    torch.set_num_threads(max(1, threads))
    prediction = predict(cache, state_model, head1, head2, batch_size=batch_size, device=torch.device("cpu"))
    result = evaluation_metrics(cache, prediction, checkpoint)
    report = {
        "schema": EVALUATION_SCHEMA, "split": cache["split"],
        "config_name": checkpoint["config_name"], "cache": str(cache_path.resolve()),
        "cache_sha256": sha256(cache_path.resolve()), "state": str(state_path.resolve()),
        "state_sha256": sha256(state_path.resolve()), "heads": str(heads_path.resolve()),
        "heads_sha256": sha256(heads_path.resolve()), "metrics": result,
        "held_out": True, "hidden_information_used": False,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "development_only": True, "deployment_authority": False,
    }
    write_json(output, report); print(json.dumps(report, sort_keys=True), flush=True); return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    state = sub.add_parser("train-state"); state.add_argument("--cache", type=Path, default=TRAIN_CACHE)
    state.add_argument("--output", type=Path, required=True); state.add_argument("--batch-size", type=int, default=512)
    state.add_argument("--threads", type=int, default=1)
    action = sub.add_parser("train-action"); action.add_argument("--cache", type=Path, default=TRAIN_CACHE)
    action.add_argument("--state", type=Path, required=True); action.add_argument("--output", type=Path, required=True)
    action.add_argument("--config", choices=tuple(CONFIGS), required=True); action.add_argument("--epochs", type=int, default=10)
    action.add_argument("--batch-size", type=int, default=256); action.add_argument("--threads", type=int, default=1)
    evaluate_parser = sub.add_parser("evaluate"); evaluate_parser.add_argument("--cache", type=Path, required=True)
    evaluate_parser.add_argument("--state", type=Path, required=True); evaluate_parser.add_argument("--heads", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True); evaluate_parser.add_argument("--batch-size", type=int, default=256)
    evaluate_parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "train-state":
        train_state_baseline(cache_path=args.cache, output=args.output,
                             batch_size=args.batch_size, threads=args.threads)
    elif args.command == "train-action":
        train_action(cache_path=args.cache, state_path=args.state, output=args.output,
                     config_name=args.config, epochs=args.epochs,
                     batch_size=args.batch_size, threads=args.threads)
    elif args.command == "evaluate":
        evaluate(cache_path=args.cache, state_path=args.state, heads_path=args.heads,
                 output=args.output, batch_size=args.batch_size, threads=args.threads)
    else: raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
