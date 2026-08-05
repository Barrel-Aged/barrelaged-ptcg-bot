#!/usr/bin/env python3
"""Train and audit a conservative Team-Rocket-native offline action-value overlay.

N24 deliberately leaves the submitted policy bytes untouched.  It encodes the
already parity-audited N9 exact-Rocket trajectories with the frozen live v10
trunk, learns twin state-action return heads plus an expectile state-value head,
and admits an action change only when both Q heads prefer the same supported
alternative by a frozen margin.  Validation/test and official-engine execution
remain separate commands so an execution lock can be written before opening
held-out evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_functional

try:
    from train_live_outcome_awr import (
        collate_like_live_agent,
        deployment_logits,
        extract_examples,
    )
    from train_n9_balanced_pcgrad_awr import (
        BOT,
        CANDIDATE,
        DECK_SHA256,
        GROUPS,
        MODEL_SHA256,
        digest,
        load_games,
        split_games,
        validate_parity,
    )
except ModuleNotFoundError:  # pragma: no cover - module-style test import
    from scripts.train_live_outcome_awr import (
        collate_like_live_agent,
        deployment_logits,
        extract_examples,
    )
    from scripts.train_n9_balanced_pcgrad_awr import (
        BOT,
        CANDIDATE,
        DECK_SHA256,
        GROUPS,
        MODEL_SHA256,
        digest,
        load_games,
        split_games,
        validate_parity,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAYS = PROJECT_ROOT / "work" / "n9_balanced_exact_replays"
DEFAULT_PARITY = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "n9_balanced_pcgrad_awr"
    / "exact_agent_parity.json"
)
DEFAULT_CACHE = PROJECT_ROOT / "work" / "n24_native_iql"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "n24_native_iql"
)

CACHE_SCHEMA = "rocket-n24-frozen-token-cache/v1"
MODEL_SCHEMA = "rocket-n24-conservative-iql-heads/v1"
EVALUATION_SCHEMA = "rocket-n24-conservative-iql-evaluation/v1"
SEED = 824001


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _live_tokens(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact global/option tokens consumed by the live logit head."""

    if not hasattr(model, "_tokens"):
        raise TypeError("N24 requires the frozen live v10 relational trunk")
    return model._tokens(  # type: ignore[attr-defined]
        batch["global"],
        batch["belief_global"],
        batch["hand_card"],
        batch["my_discard_card"],
        batch["opp_discard_card"],
        batch["stadium_card"],
        batch["my_unseen_card"],
        batch["opp_unseen_card"],
        batch["poke_feats"],
        batch["poke_card"],
        batch["opt_feats"],
        batch["opt_unseen"],
        batch["opt_card"],
        batch["opt_attack"],
        batch["opt_target"],
        batch["poke_mask"],
        batch["opt_mask"],
        batch.get("hand_mask"),
        batch.get("my_discard_mask"),
        batch.get("opp_discard_mask"),
        batch.get("my_unseen_mask"),
        batch.get("opp_unseen_mask"),
        kpc=batch.get("known_prized_card"),
        kpm=batch.get("known_prized_mask"),
        efc=batch.get("effect_card"),
        bpz=batch.get("belief_prized"),
        ctc=batch.get("context_card"),
        pwk=batch.get("poke_weak"),
    )


def build_cache(
    *,
    split: str,
    candidate: Path,
    replays: Path,
    parity_report: Path,
    output: Path,
    threads: int,
) -> dict[str, Any]:
    """Encode one immutable game split without fitting or inspecting a model."""

    candidate = candidate.resolve()
    replays = replays.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N24 cache: {output}")
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported N24 split: {split}")
    if digest(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("N24 live model bytes changed")
    if digest(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("N24 Team Rocket deck bytes changed")
    validate_parity(parity_report.resolve(), candidate, replays)

    torch.set_num_threads(max(1, threads))
    device = torch.device("cpu")
    sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = [int(value) for value in (candidate / "deck.csv").read_text().split()]
    priors = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    games, source_counts = load_games(replays)
    selected_games = split_games(games)[split]
    model = load_policy(candidate / "model.pt", map_location=device).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    global_chunks: list[torch.Tensor] = []
    option_chunks: list[torch.Tensor] = []
    logit_chunks: list[torch.Tensor] = []
    availability_chunks: list[torch.Tensor] = []
    actions: list[int] = []
    game_ids: list[int] = []
    outcomes: list[int] = []
    remaining: list[int] = []
    offsets = [0]
    files: list[str] = []
    group_ids: list[int] = []
    game_lengths: list[int] = []
    extraction: dict[str, int] = {}
    parity_choices = 0
    started = monotonic()

    decisive_games = [game for game in selected_games if not game.draw]
    for game_id, game in enumerate(decisive_games):
        examples, counts = extract_examples(
            [game], BOT, deck, priors, Belief, features
        )
        for key, value in counts.items():
            extraction[key] = extraction.get(key, 0) + int(value)
        if not examples:
            raise ValueError(f"N24 game has no supported choices: {game.path}")
        batch, availability = collate_like_live_agent(examples, device)
        with torch.no_grad():
            global_token, option_token = _live_tokens(model, batch)
            reference_logits = model.logit_head(option_token).squeeze(-1)
            reference_logits = deployment_logits(
                reference_logits, availability, examples
            )
        labels = torch.tensor([example.action for example in examples])
        predictions = reference_logits.argmax(dim=1).cpu()
        if not torch.equal(predictions, labels):
            mismatches = int((predictions != labels).sum())
            raise ValueError(
                f"N24 frozen-token parity failed in {game.path.name}: {mismatches}"
            )
        parity_choices += len(examples)
        global_chunks.append(global_token.detach().cpu().to(torch.float16))
        files.append(str(game.path.resolve()))
        group_ids.append(GROUPS.index(game.opponent))
        game_lengths.append(len(examples))
        outcome = 1 if game.won else -1
        for index, example in enumerate(examples):
            option_count = int(example.feat["opt_feats"].shape[0])
            option_chunks.append(
                option_token[index, :option_count].detach().cpu().to(torch.float16)
            )
            logit_chunks.append(
                reference_logits[index, :option_count]
                .detach()
                .cpu()
                .to(torch.float32)
            )
            available = torch.as_tensor(
                example.feat["opt_avail"], dtype=torch.bool
            )
            for end_index in example.deployment_end_indices:
                if 0 <= end_index < len(available):
                    available[end_index] = False
            availability_chunks.append(available)
            offsets.append(offsets[-1] + option_count)
            actions.append(int(example.action))
            game_ids.append(game_id)
            outcomes.append(outcome)
            remaining.append(len(examples) - index - 1)
        if (game_id + 1) % 50 == 0 or game_id + 1 == len(decisive_games):
            print(
                json.dumps(
                    {
                        "cache_games": game_id + 1,
                        "split": split,
                        "choices": len(actions),
                        "seconds": round(monotonic() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    cache: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "split": split,
        "candidate": str(candidate),
        "model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "replays": str(replays),
        "replay_tree_sha256": "f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae",
        "parity_report": str(parity_report.resolve()),
        "parity_report_sha256": sha256(parity_report.resolve()),
        "source_counts": source_counts,
        "groups": GROUPS,
        "files": files,
        "group_id_by_game": torch.tensor(group_ids, dtype=torch.int8),
        "game_lengths": torch.tensor(game_lengths, dtype=torch.int16),
        "global_token": torch.cat(global_chunks, dim=0),
        "option_token": torch.cat(option_chunks, dim=0),
        "reference_logit": torch.cat(logit_chunks, dim=0),
        "available": torch.cat(availability_chunks, dim=0),
        "offset": torch.tensor(offsets, dtype=torch.int64),
        "action": torch.tensor(actions, dtype=torch.int16),
        "game_id": torch.tensor(game_ids, dtype=torch.int32),
        "outcome": torch.tensor(outcomes, dtype=torch.int8),
        "remaining": torch.tensor(remaining, dtype=torch.int16),
        "extraction": extraction,
        "parity_choices": parity_choices,
        "official_engine_source": True,
        "hidden_information_used": False,
        "runtime_opponent_identity_input": False,
        "elapsed_seconds": monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    manifest = {
        key: value
        for key, value in cache.items()
        if not torch.is_tensor(value)
    }
    manifest.update(
        {
            "games": len(files),
            "choices": len(actions),
            "option_rows": offsets[-1],
            "hidden_dim": int(cache["global_token"].shape[1]),
            "cache": str(output),
            "cache_sha256": sha256(output),
        }
    )
    _json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


class QHead(nn.Module):
    """Low-rank bounded state-option return head over frozen live tokens."""

    def __init__(self, hidden_dim: int, width: int = 64) -> None:
        super().__init__()
        self.state_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU()
        )
        self.option_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU()
        )
        self.state_bias = nn.Linear(hidden_dim, 1)
        self.option_bias = nn.Linear(hidden_dim, 1)
        self.reference_scale = nn.Parameter(torch.zeros(()))
        self.scale = math.sqrt(width)

    def forward(
        self,
        state: torch.Tensor,
        option: torch.Tensor,
        reference_logit: torch.Tensor,
    ) -> torch.Tensor:
        state_hidden = self.state_projection(state)
        option_hidden = self.option_projection(option)
        interaction = torch.einsum("bh,boh->bo", state_hidden, option_hidden)
        values = interaction / self.scale
        values = values + self.state_bias(state)
        values = values + self.option_bias(option).squeeze(-1)
        values = values + self.reference_scale * reference_logit.clamp(-20.0, 20.0)
        return torch.tanh(values)


class VHead(nn.Module):
    """Bounded public-state expectile value head."""

    def __init__(self, hidden_dim: int, width: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(state).squeeze(-1))


@dataclass(frozen=True)
class RaggedBatch:
    state: torch.Tensor
    option: torch.Tensor
    reference_logit: torch.Tensor
    available: torch.Tensor
    action: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor
    index: torch.Tensor


def collate_ragged(
    cache: Mapping[str, Any],
    indices: Sequence[int],
    *,
    gamma: float,
    device: torch.device,
) -> RaggedBatch:
    """Pad only the selected ragged option rows for one training/eval batch."""

    rows = [int(index) for index in indices]
    offset: torch.Tensor = cache["offset"]
    counts = [int(offset[index + 1] - offset[index]) for index in rows]
    maximum = max(counts)
    hidden = int(cache["global_token"].shape[1])
    option = torch.zeros(len(rows), maximum, hidden, dtype=torch.float32)
    reference = torch.full((len(rows), maximum), -1.0e9, dtype=torch.float32)
    available = torch.zeros(len(rows), maximum, dtype=torch.bool)
    for row, (index, count) in enumerate(zip(rows, counts, strict=True)):
        begin = int(offset[index])
        end = begin + count
        option[row, :count] = cache["option_token"][begin:end].float()
        reference[row, :count] = cache["reference_logit"][begin:end]
        available[row, :count] = cache["available"][begin:end]
    state = cache["global_token"][rows].float()
    action = cache["action"][rows].long()
    outcome = cache["outcome"][rows].float()
    remaining = cache["remaining"][rows].float()
    target = outcome * torch.pow(torch.full_like(remaining, gamma), remaining)
    game_id = cache["game_id"][rows].long()
    game_lengths = cache["game_lengths"][game_id].float()
    # Equal total base mass per trajectory.  The source split itself is exactly
    # opponent-balanced, so no runtime opponent identity is introduced.
    weight = 1.0 / game_lengths.clamp_min(1.0)
    weight = weight / weight.mean().clamp_min(1.0e-12)
    return RaggedBatch(
        state.to(device),
        option.to(device),
        reference.to(device),
        available.to(device),
        action.to(device),
        target.to(device),
        weight.to(device),
        torch.tensor(rows, dtype=torch.long, device=device),
    )


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0e-12)


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    expectile: float,
    weight: torch.Tensor,
) -> torch.Tensor:
    residual = target - prediction
    asymmetric = torch.where(
        residual >= 0.0,
        torch.full_like(residual, expectile),
        torch.full_like(residual, 1.0 - expectile),
    )
    return weighted_mean(asymmetric * residual.square(), weight)


def conservative_loss(
    values: torch.Tensor,
    action: torch.Tensor,
    available: torch.Tensor,
    weight: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    guarded = values.masked_fill(~available, -1.0e9)
    chosen = guarded.gather(1, action[:, None]).squeeze(1)
    soft_maximum = temperature * torch.logsumexp(guarded / temperature, dim=1)
    return weighted_mean(soft_maximum - chosen, weight)


def _bootstrap_mask(indices: torch.Tensor, head: int) -> torch.Tensor:
    # Deterministic 80% subsamples with different omissions and 60% overlap.
    return ((indices * 37 + head * 17 + 11) % 5) != 0


def _state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


@torch.no_grad()
def predict_batches(
    cache: Mapping[str, Any],
    q1: QHead,
    q2: QHead,
    value: VHead,
    *,
    gamma: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    q1.eval()
    q2.eval()
    value.eval()
    size = int(cache["action"].numel())
    taken1 = torch.empty(size)
    taken2 = torch.empty(size)
    values = torch.empty(size)
    mean1 = torch.empty(size)
    mean2 = torch.empty(size)
    best_index = torch.empty(size, dtype=torch.long)
    best_score = torch.empty(size)
    best_support = torch.empty(size)
    live_index = torch.empty(size, dtype=torch.long)
    eligible = torch.zeros(size, dtype=torch.bool)
    for start in range(0, size, batch_size):
        rows = list(range(start, min(size, start + batch_size)))
        batch = collate_ragged(cache, rows, gamma=gamma, device=device)
        first = q1(batch.state, batch.option, batch.reference_logit)
        second = q2(batch.state, batch.option, batch.reference_logit)
        state_value = value(batch.state)
        guarded_reference = batch.reference_logit.masked_fill(~batch.available, -1.0e9)
        probability = torch.softmax(guarded_reference, dim=1)
        live = guarded_reference.argmax(dim=1)
        lower = torch.minimum(first, second)
        supported = batch.available & (probability >= 0.02)
        lower_supported = lower.masked_fill(~supported, -1.0e9)
        best = lower_supported.argmax(dim=1)
        row_index = torch.arange(len(rows), device=device)
        score = (
            lower_supported[row_index, best]
            - lower[row_index, live]
        )
        head_agreement = (
            (first[row_index, best] > first[row_index, live])
            & (second[row_index, best] > second[row_index, live])
        )
        can_change = head_agreement & (best != live) & supported[row_index, best]
        target_slice = slice(start, start + len(rows))
        taken1[target_slice] = first[row_index, batch.action].cpu()
        taken2[target_slice] = second[row_index, batch.action].cpu()
        values[target_slice] = state_value.cpu()
        denominator = batch.available.sum(1).clamp_min(1)
        mean1[target_slice] = (
            first.masked_fill(~batch.available, 0.0).sum(1) / denominator
        ).cpu()
        mean2[target_slice] = (
            second.masked_fill(~batch.available, 0.0).sum(1) / denominator
        ).cpu()
        best_index[target_slice] = best.cpu()
        best_score[target_slice] = score.cpu()
        best_support[target_slice] = probability[row_index, best].cpu()
        live_index[target_slice] = live.cpu()
        eligible[target_slice] = can_change.cpu()
    return {
        "q1_taken": taken1,
        "q2_taken": taken2,
        "value": values,
        "q1_mean": mean1,
        "q2_mean": mean2,
        "best_index": best_index,
        "best_score": best_score,
        "best_support": best_support,
        "live_index": live_index,
        "eligible": eligible,
    }


def select_frozen_threshold(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    *,
    population: int,
    target_rate: float = 0.005,
    minimum: float = 0.05,
) -> float:
    """Choose one deterministic train-only threshold targeting 0.5% changes."""

    valid = scores[eligible & torch.isfinite(scores) & (scores >= minimum)]
    target = max(1, int(round(population * target_rate)))
    if valid.numel() < target:
        return float(minimum)
    ordered = torch.sort(valid, descending=True).values
    return float(max(minimum, ordered[target - 1].item()))


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Tie-aware binary ROC AUC without an optional sklearn dependency."""

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(int(label) for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(
            int(label) for _, label in pairs[index:end]
        )
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def fit_logistic_calibration(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    """Fit one train-only, opponent-blind Platt calibrator deterministically."""

    x = scores.detach().double()
    y = labels.detach().double()
    rate = float(y.mean().clamp(1.0e-6, 1.0 - 1.0e-6))
    parameter = nn.Parameter(
        torch.tensor([1.0, math.log(rate / (1.0 - rate))], dtype=torch.float64)
    )
    optimizer = torch.optim.LBFGS(
        [parameter], lr=0.5, max_iter=100, tolerance_grad=1.0e-12,
        tolerance_change=1.0e-12, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = parameter[0] * x + parameter[1]
        loss = torch_functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    if not torch.isfinite(parameter).all():
        raise RuntimeError("N24 train-only outcome calibrator is non-finite")
    return float(parameter[0].detach()), float(parameter[1].detach())


def evaluation_metrics(
    cache: Mapping[str, Any],
    prediction: Mapping[str, torch.Tensor],
    *,
    gamma: float,
    threshold: float,
    train_group_rates: Sequence[float],
    outcome_calibration: Sequence[float],
) -> dict[str, Any]:
    outcome = cache["outcome"].float()
    remaining = cache["remaining"].float()
    target = outcome * torch.pow(torch.full_like(remaining, gamma), remaining)
    q_taken = torch.minimum(prediction["q1_taken"], prediction["q2_taken"])
    q_mean = torch.minimum(prediction["q1_mean"], prediction["q2_mean"])
    labels = (outcome > 0).long()
    game_id = cache["game_id"].long()
    groups = cache["group_id_by_game"][game_id].long()
    game_lengths = cache["game_lengths"].long()
    game_starts = torch.cat(
        (torch.zeros(1, dtype=torch.long), torch.cumsum(game_lengths, dim=0)[:-1])
    )
    game_labels = labels[game_starts]
    game_scores = q_taken[game_starts]
    game_groups = cache["group_id_by_game"].long()
    choice_weight = 1.0 / game_lengths[game_id].float().clamp_min(1.0)
    change = prediction["eligible"] & (prediction["best_score"] >= threshold)
    reference_parity = prediction["live_index"].to(cache["action"].dtype).eq(
        cache["action"]
    )

    def subset(
        choice_mask: torch.Tensor,
        game_mask: torch.Tensor,
        group_rate: float,
    ) -> dict[str, float | int]:
        choice_count = int(choice_mask.sum())
        game_count = int(game_mask.sum())
        if choice_count == 0 or game_count == 0:
            return {"choices": choice_count, "games": game_count}
        q = q_taken[choice_mask]
        mean_q = q_mean[choice_mask]
        choice_target = target[choice_mask]
        weights = choice_weight[choice_mask]
        score = game_scores[game_mask]
        y = game_labels[game_mask]
        scale, bias = (float(value) for value in outcome_calibration)
        probability = torch.sigmoid(scale * score + bias)
        constant = torch.full_like(probability, float(group_rate))
        return {
            "choices": choice_count,
            "games": game_count,
            "wins": int(y.sum()),
            "first_choice_auc": binary_auc(score.tolist(), y.tolist()),
            "target_mae": float(
                ((q - choice_target).abs() * weights).sum()
                / weights.sum().clamp_min(1.0e-12)
            ),
            "brier": float((probability - y.float()).square().mean()),
            "constant_brier": float((constant - y.float()).square().mean()),
            "taken_vs_mean_mae_ratio": float(
                ((q - choice_target).abs() * weights).sum()
                / ((mean_q - choice_target).abs() * weights)
                .sum()
                .clamp_min(1.0e-12)
            ),
            "replacements": int(change[choice_mask].sum()),
            "replacement_rate": float(change[choice_mask].float().mean()),
        }

    group_rows = {
        GROUPS[group]: subset(
            groups == group,
            game_groups == group,
            float(train_group_rates[group]),
        )
        for group in range(len(GROUPS))
    }
    overall_rate = float(
        sum(train_group_rates) / max(1, len(train_group_rates))
    )
    changed_scores = prediction["best_score"][change]
    changed_support = prediction["best_support"][change]
    return {
        "choices": int(outcome.numel()),
        "games": len(cache["files"]),
        "reference_parity": int(reference_parity.sum()),
        "reference_mismatches": int((~reference_parity).sum()),
        "overall": subset(
            torch.ones_like(labels, dtype=torch.bool),
            torch.ones_like(game_labels, dtype=torch.bool),
            overall_rate,
        ),
        "by_opponent": group_rows,
        "overlay": {
            "threshold": threshold,
            "eligible_before_threshold": int(prediction["eligible"].sum()),
            "replacements": int(change.sum()),
            "replacement_rate": float(change.float().mean()),
            "minimum_changed_score": (
                float(changed_scores.min()) if changed_scores.numel() else None
            ),
            "mean_changed_score": (
                float(changed_scores.mean()) if changed_scores.numel() else None
            ),
            "minimum_reference_support": (
                float(changed_support.min()) if changed_support.numel() else None
            ),
        },
    }


def train_heads(
    *,
    cache_path: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    expectile: float,
    gamma: float,
    cql_coefficient: float,
    cql_temperature: float,
    seed: int,
    threads: int,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N24 training output: {output}")
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=True)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("split") != "train":
        raise ValueError("N24 training requires the frozen train cache")
    if cache.get("model_sha256") != MODEL_SHA256 or cache.get("deck_sha256") != DECK_SHA256:
        raise ValueError("N24 training cache substrate changed")
    if not 0.5 < expectile < 1.0:
        raise ValueError("IQL expectile must be in (0.5, 1)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    device = torch.device("cpu")
    hidden = int(cache["global_token"].shape[1])
    q1 = QHead(hidden).to(device)
    torch.manual_seed(seed + 1)
    q2 = QHead(hidden).to(device)
    torch.manual_seed(seed + 2)
    value = VHead(hidden).to(device)
    parameters = list(q1.parameters()) + list(q2.parameters()) + list(value.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    count = int(cache["action"].numel())
    history = []
    started = monotonic()
    for epoch in range(1, epochs + 1):
        order = list(range(count))
        random.Random(seed + epoch).shuffle(order)
        totals = {"q1": 0.0, "q2": 0.0, "value": 0.0, "cql1": 0.0, "cql2": 0.0}
        updates = 0
        q1.train()
        q2.train()
        value.train()
        for start in range(0, count, batch_size):
            rows = order[start : start + batch_size]
            batch = collate_ragged(cache, rows, gamma=gamma, device=device)
            first = q1(batch.state, batch.option, batch.reference_logit)
            second = q2(batch.state, batch.option, batch.reference_logit)
            row_index = torch.arange(len(rows), device=device)
            first_taken = first[row_index, batch.action]
            second_taken = second[row_index, batch.action]
            mask1 = _bootstrap_mask(batch.index, 1).float()
            mask2 = _bootstrap_mask(batch.index, 2).float()
            weight1 = batch.weight * mask1
            weight2 = batch.weight * mask2
            first_error = torch_functional.smooth_l1_loss(
                first_taken, batch.target, reduction="none"
            )
            second_error = torch_functional.smooth_l1_loss(
                second_taken, batch.target, reduction="none"
            )
            q1_loss = weighted_mean(first_error, weight1)
            q2_loss = weighted_mean(second_error, weight2)
            cql1 = conservative_loss(
                first,
                batch.action,
                batch.available,
                weight1,
                temperature=cql_temperature,
            )
            cql2 = conservative_loss(
                second,
                batch.action,
                batch.available,
                weight2,
                temperature=cql_temperature,
            )
            with torch.no_grad():
                expectile_target = torch.minimum(first_taken, second_taken)
            state_value = value(batch.state)
            value_loss = expectile_loss(
                state_value,
                expectile_target,
                expectile=expectile,
                weight=batch.weight,
            )
            loss = (
                q1_loss
                + q2_loss
                + value_loss
                + cql_coefficient * (cql1 + cql2)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            totals["q1"] += float(q1_loss.detach())
            totals["q2"] += float(q2_loss.detach())
            totals["value"] += float(value_loss.detach())
            totals["cql1"] += float(cql1.detach())
            totals["cql2"] += float(cql2.detach())
            updates += 1
        row = {
            "epoch": epoch,
            **{key: value_ / max(1, updates) for key, value_ in totals.items()},
            "seconds": monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    prediction = predict_batches(
        cache,
        q1,
        q2,
        value,
        gamma=gamma,
        batch_size=batch_size,
        device=device,
    )
    threshold = select_frozen_threshold(
        prediction["best_score"],
        prediction["eligible"],
        population=count,
    )
    game_outcomes = []
    for game_id, length in enumerate(cache["game_lengths"].tolist()):
        first_index = int((cache["game_id"] == game_id).nonzero()[0])
        game_outcomes.append(int(cache["outcome"][first_index]))
    train_group_rates = []
    for group in range(len(GROUPS)):
        selected = [
            outcome > 0
            for outcome, group_id in zip(
                game_outcomes, cache["group_id_by_game"].tolist(), strict=True
            )
            if int(group_id) == group
        ]
        train_group_rates.append(sum(selected) / len(selected))
    game_lengths_tensor = cache["game_lengths"].long()
    game_starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(game_lengths_tensor, dim=0)[:-1],
        )
    )
    q_taken = torch.minimum(prediction["q1_taken"], prediction["q2_taken"])
    outcome_calibration = fit_logistic_calibration(
        q_taken[game_starts],
        (cache["outcome"][game_starts] > 0).long(),
    )
    train_metrics = evaluation_metrics(
        cache,
        prediction,
        gamma=gamma,
        threshold=threshold,
        train_group_rates=train_group_rates,
        outcome_calibration=outcome_calibration,
    )
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "schema": MODEL_SCHEMA,
        "hidden_dim": hidden,
        "q1": _state_dict_cpu(q1),
        "q2": _state_dict_cpu(q2),
        "value": _state_dict_cpu(value),
        "threshold": threshold,
        "minimum_reference_probability": 0.02,
        "model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "expectile": expectile,
            "gamma": gamma,
            "cql_coefficient": cql_coefficient,
            "cql_temperature": cql_temperature,
            "seed": seed,
            "threads": threads,
        },
        "train_group_rates": train_group_rates,
        "outcome_calibration": outcome_calibration,
    }
    checkpoint_path = output / "iql_heads.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": MODEL_SCHEMA,
        "candidate": cache["candidate"],
        "model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256(cache_path.resolve()),
        "games": len(cache["files"]),
        "choices": count,
        "official_engine_source": True,
        "frozen_live_trunk": True,
        "live_policy_modified": False,
        "hidden_information_used": False,
        "runtime_opponent_identity_input": False,
        "training_method": {
            "q": "twin bootstrap Monte-Carlo discounted behavior-return heads",
            "v": "upper expectile of conservative twin logged-action Q",
            "conservatism": "CQL log-sum-exp penalty over legal alternatives",
            "overlay": "both heads must prefer a >=2%-reference-support alternative",
        },
        "hyperparameters": checkpoint["hyperparameters"],
        "threshold_selection": {
            "source": "train only",
            "target_replacement_rate": 0.005,
            "minimum_margin": 0.05,
            "selected_margin": threshold,
        },
        "train_group_rates": train_group_rates,
        "outcome_calibration": {
            "source": "train-only first-choice predictions",
            "opponent_identity_used": False,
            "scale": outcome_calibration[0],
            "bias": outcome_calibration[1],
        },
        "history": history,
        "train_metrics": train_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": monotonic() - started,
        "development_only": True,
        "deployment_authority": False,
    }
    _json(output / "training_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def load_heads(path: Path) -> tuple[dict[str, Any], QHead, QHead, VHead]:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported N24 head checkpoint")
    hidden = int(checkpoint["hidden_dim"])
    q1 = QHead(hidden)
    q2 = QHead(hidden)
    value = VHead(hidden)
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint["q2"])
    value.load_state_dict(checkpoint["value"])
    return checkpoint, q1.eval(), q2.eval(), value.eval()


def evaluate_heads(
    *,
    cache_path: Path,
    heads_path: Path,
    output: Path,
    batch_size: int,
    threads: int,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N24 evaluation: {output}")
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=True)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("split") not in {
        "validation", "test"
    }:
        raise ValueError("N24 evaluation requires a held-out cache")
    checkpoint, q1, q2, value = load_heads(heads_path)
    if checkpoint.get("model_sha256") != cache.get("model_sha256"):
        raise ValueError("N24 evaluation model provenance mismatch")
    if checkpoint.get("deck_sha256") != cache.get("deck_sha256"):
        raise ValueError("N24 evaluation deck provenance mismatch")
    torch.set_num_threads(max(1, threads))
    gamma = float(checkpoint["hyperparameters"]["gamma"])
    prediction = predict_batches(
        cache,
        q1,
        q2,
        value,
        gamma=gamma,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )
    metrics = evaluation_metrics(
        cache,
        prediction,
        gamma=gamma,
        threshold=float(checkpoint["threshold"]),
        train_group_rates=checkpoint["train_group_rates"],
        outcome_calibration=checkpoint["outcome_calibration"],
    )
    overall = metrics["overall"]
    opponents = list(metrics["by_opponent"].values())
    validation_gates = {
        "exact_reference_parity": metrics["reference_mismatches"] == 0,
        "overall_first_choice_auc_ge_0_56": float(overall["first_choice_auc"]) >= 0.56,
        "worst_opponent_first_choice_auc_ge_0_53": min(
            float(row["first_choice_auc"]) for row in opponents
        ) >= 0.53,
        "overall_brier_better_than_constant": float(overall["brier"]) < float(overall["constant_brier"]),
        "each_opponent_brier_not_worse_than_constant": all(
            float(row["brier"]) <= float(row["constant_brier"]) for row in opponents
        ),
        "action_specific_mae_ratio_le_0_98": float(overall["taken_vs_mean_mae_ratio"]) <= 0.98,
        "replacement_rate_ge_0_002": float(metrics["overlay"]["replacement_rate"]) >= 0.002,
        "replacement_rate_le_0_015": float(metrics["overlay"]["replacement_rate"]) <= 0.015,
        "minimum_two_replacements": int(metrics["overlay"]["replacements"]) >= 2,
        "reference_support_ge_0_02": (
            metrics["overlay"]["minimum_reference_support"] is not None
            and float(metrics["overlay"]["minimum_reference_support"]) >= 0.02 - 1.0e-7
        ),
    }
    admitted = all(validation_gates.values())
    report = {
        "schema": EVALUATION_SCHEMA,
        "split": cache["split"],
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256(cache_path.resolve()),
        "heads": str(heads_path.resolve()),
        "heads_sha256": sha256(heads_path.resolve()),
        "model_sha256": cache["model_sha256"],
        "deck_sha256": cache["deck_sha256"],
        "metrics": metrics,
        "gates": validation_gates,
        "admitted": admitted,
        "official_engine_source": True,
        "official_engine_candidate_execution": False,
        "live_policy_modified": False,
        "development_only": True,
        "deployment_authority": False,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="encode one frozen N9 split")
    cache.add_argument("--split", choices=("train", "validation", "test"), required=True)
    cache.add_argument("--candidate", type=Path, default=CANDIDATE)
    cache.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    cache.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--threads", type=int, default=1)

    train = subparsers.add_parser("train", help="fit N24 heads on train only")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3.0e-4)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--expectile", type=float, default=0.70)
    train.add_argument("--gamma", type=float, default=0.997)
    train.add_argument("--cql-coefficient", type=float, default=0.05)
    train.add_argument("--cql-temperature", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=SEED)
    train.add_argument("--threads", type=int, default=1)

    evaluate = subparsers.add_parser("evaluate", help="open one held-out cache")
    evaluate.add_argument("--cache", type=Path, required=True)
    evaluate.add_argument("--heads", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "cache":
        build_cache(
            split=args.split,
            candidate=args.candidate,
            replays=args.replays,
            parity_report=args.parity_report,
            output=args.output,
            threads=args.threads,
        )
    elif args.command == "train":
        train_heads(
            cache_path=args.cache,
            output=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            expectile=args.expectile,
            gamma=args.gamma,
            cql_coefficient=args.cql_coefficient,
            cql_temperature=args.cql_temperature,
            seed=args.seed,
            threads=args.threads,
        )
    elif args.command == "evaluate":
        evaluate_heads(
            cache_path=args.cache,
            heads_path=args.heads,
            output=args.output,
            batch_size=args.batch_size,
            threads=args.threads,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
