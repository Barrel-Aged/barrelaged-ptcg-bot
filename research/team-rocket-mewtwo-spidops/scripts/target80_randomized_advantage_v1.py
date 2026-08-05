#!/usr/bin/env python3
"""Build, train, and evaluate randomized-support action-advantage heads.

Unlike N24's deterministic behavior corpus, every cached choice here comes
from a known action mixture. The model centers action advantages under that
logged mixture, uses clipped inverse-propensity weights, and admits runtime
changes only when two independently bootstrapped heads agree on a supported
alternative.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_live_outcome_awr import (
    WeightedExample,
    collate_like_live_agent,
    deployment_logits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    PROJECT_ROOT / "work" / "target80_candidates" / "mimikyu_maxbelt_neural_control_v1"
)
CORPUS = PROJECT_ROOT / "work" / "target80_randomized_support_v1_corpus"
CORPUS_AUDIT = CORPUS / "CORPUS_AUDIT.json"
CACHE_ROOT = PROJECT_ROOT / "work" / "target80_randomized_advantage_v1_cache"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "target80_randomized_advantage_v1"

CACHE_SCHEMA = "target80-randomized-advantage-token-cache/v1"
MODEL_SCHEMA = "target80-randomized-advantage-heads/v1"
EVALUATION_SCHEMA = "target80-randomized-advantage-evaluation/v1"
CORPUS_SHA256 = "3dde57e3420070bf43f31ffd9fa38ac6a9015c6fa6cc3878a63dff776eb69602"
DECK_SHA256 = "4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea"
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
MAIN_SHA256 = "80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6"
BOT = "Target80MimikyuMaxBeltRandomSupportV1"
OPPONENTS = ("marnie", "starmie", "alakazam")
GROUPS = tuple(f"{opponent}_s{seat}" for opponent in OPPONENTS for seat in (0, 1))
MINIMUM_REFERENCE_SUPPORT = 0.03
IPW_CLIP = 20.0

CONFIGS: dict[str, dict[str, float | int]] = {
    "low_reg": {
        "epochs": 10,
        "width": 96,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "advantage_regularization": 0.01,
        "seed": 854001,
    },
    "balanced": {
        "epochs": 10,
        "width": 96,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "advantage_regularization": 0.03,
        "seed": 854002,
    },
    "high_reg": {
        "epochs": 10,
        "width": 96,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "advantage_regularization": 0.08,
        "seed": 854003,
    },
}
THRESHOLD_GRID = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _live_tokens(
    model: nn.Module, batch: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "_tokens"):
        raise TypeError("randomized advantage requires the frozen v10 relational trunk")
    return model._tokens(  # type: ignore[attr-defined]
        batch["global"], batch["belief_global"], batch["hand_card"],
        batch["my_discard_card"], batch["opp_discard_card"], batch["stadium_card"],
        batch["my_unseen_card"], batch["opp_unseen_card"], batch["poke_feats"],
        batch["poke_card"], batch["opt_feats"], batch["opt_unseen"],
        batch["opt_card"], batch["opt_attack"], batch["opt_target"],
        batch["poke_mask"], batch["opt_mask"], batch.get("hand_mask"),
        batch.get("my_discard_mask"), batch.get("opp_discard_mask"),
        batch.get("my_unseen_mask"), batch.get("opp_unseen_mask"),
        kpc=batch.get("known_prized_card"), kpm=batch.get("known_prized_mask"),
        efc=batch.get("effect_card"), bpz=batch.get("belief_prized"),
        ctc=batch.get("context_card"), pwk=batch.get("poke_weak"),
    )


def verify_inputs(corpus: Path, candidate: Path) -> dict[str, Any]:
    audit = json.loads((corpus / "CORPUS_AUDIT.json").read_text(encoding="utf-8"))
    if audit.get("corpus_sha256") != CORPUS_SHA256 or not audit.get("data_quality_admitted"):
        raise ValueError("frozen randomized-support corpus audit mismatch")
    if int(audit.get("games")) != 1_200 or int(audit.get("manifests")) != 18:
        raise ValueError("frozen corpus cardinality mismatch")
    if sha256(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("frozen Wobbuffet-free deck changed")
    if sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("frozen v10 model changed")
    if sha256(candidate / "main.py") != MAIN_SHA256:
        raise ValueError("frozen packaged behavior changed")
    return audit


def split_manifests(corpus: Path, split: str) -> list[Path]:
    paths = [
        corpus / f"{split}_{opponent}_s{seat}.manifest.json"
        for opponent in OPPONENTS
        for seat in (0, 1)
    ]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"incomplete {split} manifest set")
    return paths


def attack_end_indices(observation: Mapping[str, Any]) -> tuple[int, ...]:
    selection = observation.get("select") or {}
    options = selection.get("option") or []
    if min(int(selection.get("maxCount") or 0), len(options)) != 1:
        return ()
    if not any(option.get("type") == 13 for option in options):
        return ()
    return tuple(index for index, option in enumerate(options) if option.get("type") == 14)


def build_cache(
    *, split: str, corpus: Path, candidate: Path, output: Path, threads: int
) -> dict[str, Any]:
    corpus = corpus.resolve()
    candidate = candidate.resolve()
    output = output.resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing existing cache: {output}")
    if split not in {"train", "validation", "test"}:
        raise ValueError("unsupported cache split")
    audit = verify_inputs(corpus, candidate)
    torch.set_num_threads(max(1, threads))
    sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = [int(value) for value in (candidate / "deck.csv").read_text().split()]
    priors = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    model = load_policy(candidate / "model.pt", map_location="cpu").cpu().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    global_chunks: list[torch.Tensor] = []
    option_chunks: list[torch.Tensor] = []
    reference_chunks: list[torch.Tensor] = []
    reference_probability_chunks: list[torch.Tensor] = []
    behavior_chunks: list[torch.Tensor] = []
    support_chunks: list[torch.Tensor] = []
    option_type_chunks: list[torch.Tensor] = []
    actions: list[int] = []
    baselines: list[int] = []
    propensities: list[float] = []
    explored: list[bool] = []
    outcomes: list[int] = []
    game_ids: list[int] = []
    group_ids: list[int] = []
    selection_types: list[int] = []
    remaining: list[int] = []
    offsets = [0]
    game_lengths: list[int] = []
    game_group_ids: list[int] = []
    game_outcomes: list[int] = []
    files: list[str] = []
    replay_sha256: list[str] = []
    manifest_sha256: list[str] = []
    counts: Counter[str] = Counter()
    maximum_probability_error = 0.0
    maximum_logit_error = 0.0
    started = monotonic()

    game_id = 0
    for manifest_path in split_manifests(corpus, split):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha256.append(sha256(manifest_path))
        own_seat = int(manifest["own_seat"])
        opponent = str(manifest["opponent_key"])
        group_id = GROUPS.index(f"{opponent}_s{own_seat}")
        replay_root = Path(str(manifest_path)[: -len(".manifest.json")])
        for replay_info in manifest["replays"]:
            counts["source_games"] += 1
            path = replay_root / replay_info["file"]
            if sha256(path) != replay_info["sha256"]:
                raise ValueError(f"replay changed after corpus audit: {path}")
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            result = int(records[-1]["result"]["raw_result"])
            outcome = 1 if result == own_seat else 0 if result == 2 else -1
            belief = Belief(deck, opponent_deck_priors=priors)
            examples: list[WeightedExample] = []
            metadata: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for record in records[1:-1]:
                if int(record.get("p", -1)) != own_seat:
                    continue
                behavior = record.get("behavior") or {}
                source = str(behavior.get("reference_source") or "")
                if not source.startswith("neural_"):
                    counts[f"skipped:{source}"] += 1
                    continue
                observation = record["obs"]
                base = features.extract(observation, belief=belief, slot_link=True)
                if not behavior.get("eligible"):
                    counts["neural_ineligible"] += 1
                    continue
                action = list(record.get("action") or [])
                if len(action) != 1:
                    raise ValueError(f"eligible replay action is not single-choice: {path.name}")
                examples.append(
                    WeightedExample(
                        base,
                        int(action[0]),
                        path.name,
                        outcome > 0,
                        f"{opponent}_s{own_seat}",
                        deployment_end_indices=attack_end_indices(observation),
                    )
                )
                metadata.append((behavior, observation))
            if not examples:
                # A terminal game with no supported alternative contains no
                # action-identification row. Keep it in the admitted corpus,
                # record it explicitly, and omit it only from the token cache.
                counts["zero_eligible_games"] += 1
                continue
            batch, availability = collate_like_live_agent(examples, torch.device("cpu"))
            with torch.no_grad():
                global_token, option_token = _live_tokens(model, batch)
                recomputed = model.logit_head(option_token).squeeze(-1)
                recomputed = deployment_logits(recomputed, availability, examples)
            probabilities = torch.softmax(recomputed, dim=1)
            for row, (example, (behavior, observation)) in enumerate(zip(examples, metadata, strict=True)):
                option_count = int(example.feat["opt_feats"].shape[0])
                logits = recomputed[row, :option_count].detach().cpu().float()
                probs = probabilities[row, :option_count].detach().cpu().float()
                logged_probs = torch.tensor(behavior["reference_probabilities"], dtype=torch.float32)
                probability_error = float((probs - logged_probs).abs().max())
                maximum_probability_error = max(maximum_probability_error, probability_error)
                if probability_error > 2.0e-5:
                    raise ValueError(f"reference probability parity failed: {path.name} {probability_error}")
                logged_logits = behavior["reference_logits"]
                for index, logged in enumerate(logged_logits):
                    if logged is None:
                        if float(logits[index]) > -1.0e8:
                            raise ValueError(f"masked-logit parity failed: {path.name}")
                    else:
                        logit_error = abs(float(logits[index]) - float(logged))
                        maximum_logit_error = max(maximum_logit_error, logit_error)
                        if logit_error > 2.0e-5:
                            raise ValueError(f"reference logit parity failed: {path.name} {logit_error}")
                logged_behavior_probability = [
                    float(value) for value in behavior["behavior_probabilities"]
                ]
                behavior_probability = torch.tensor(logged_behavior_probability, dtype=torch.float32)
                support = behavior_probability > 0.0
                if not bool((support & (availability[row, :option_count] > 0.5)).equal(support)):
                    raise ValueError(f"behavior support exceeds feature availability: {path.name}")
                action = int(example.action)
                baseline = int(behavior["reference_argmax"])
                propensity = float(behavior["chosen_propensity"])
                if abs(logged_behavior_probability[action] - propensity) > 1.0e-12:
                    raise ValueError(f"chosen propensity parity failed: {path.name}")
                global_chunks.append(global_token[row].detach().cpu().to(torch.float16)[None])
                option_chunks.append(option_token[row, :option_count].detach().cpu().to(torch.float16))
                reference_chunks.append(logits)
                reference_probability_chunks.append(probs)
                behavior_chunks.append(behavior_probability)
                support_chunks.append(support)
                option_type_chunks.append(
                    torch.tensor(
                        [int(option.get("type") if option.get("type") is not None else -1)
                         for option in (observation.get("select") or {}).get("option") or []],
                        dtype=torch.int16,
                    )
                )
                offsets.append(offsets[-1] + option_count)
                actions.append(action)
                baselines.append(baseline)
                propensities.append(propensity)
                explored.append(bool(behavior["explored"]))
                outcomes.append(outcome)
                game_ids.append(game_id)
                group_ids.append(group_id)
                selection_types.append(int(behavior["selection_type"]))
            length = len(examples)
            remaining.extend(range(length - 1, -1, -1))
            game_lengths.append(length)
            game_group_ids.append(group_id)
            game_outcomes.append(outcome)
            files.append(str(path.resolve()))
            replay_sha256.append(str(replay_info["sha256"]))
            game_id += 1
            counts["games"] += 1
            counts["eligible"] += length
            counts["explored"] += sum(bool(item[0]["explored"]) for item in metadata)
            if game_id % 50 == 0:
                print(json.dumps({"split": split, "cached_games": game_id, "eligible": len(actions)}, sort_keys=True), flush=True)

    cache: dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "split": split,
        "candidate": str(candidate),
        "corpus": str(corpus),
        "corpus_sha256": CORPUS_SHA256,
        "deck_sha256": DECK_SHA256,
        "model_sha256": MODEL_SHA256,
        "main_sha256": MAIN_SHA256,
        "global_token": torch.cat(global_chunks),
        "option_token": torch.cat(option_chunks),
        "reference_logit": torch.cat(reference_chunks),
        "reference_probability": torch.cat(reference_probability_chunks),
        "behavior_probability": torch.cat(behavior_chunks),
        "support": torch.cat(support_chunks),
        "option_type": torch.cat(option_type_chunks),
        "offset": torch.tensor(offsets, dtype=torch.long),
        "action": torch.tensor(actions, dtype=torch.long),
        "baseline": torch.tensor(baselines, dtype=torch.long),
        "propensity": torch.tensor(propensities, dtype=torch.float32),
        "explored": torch.tensor(explored, dtype=torch.bool),
        "outcome": torch.tensor(outcomes, dtype=torch.int8),
        "game_id": torch.tensor(game_ids, dtype=torch.long),
        "group_id": torch.tensor(group_ids, dtype=torch.int8),
        "selection_type": torch.tensor(selection_types, dtype=torch.int8),
        "remaining": torch.tensor(remaining, dtype=torch.int16),
        "game_lengths": torch.tensor(game_lengths, dtype=torch.long),
        "game_group_id": torch.tensor(game_group_ids, dtype=torch.int8),
        "game_outcome": torch.tensor(game_outcomes, dtype=torch.int8),
        "files": files,
        "replay_sha256": replay_sha256,
        "manifest_sha256": manifest_sha256,
        "groups": GROUPS,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output)
    outcome_counts = Counter(game_outcomes)
    manifest = {
        "schema": CACHE_SCHEMA,
        "split": split,
        "cache": str(output),
        "cache_sha256": sha256(output),
        "corpus_audit": str(CORPUS_AUDIT.resolve()),
        "corpus_sha256": audit["corpus_sha256"],
        "games": len(files),
        "source_games": counts["source_games"],
        "zero_eligible_games": counts["zero_eligible_games"],
        "choices": len(actions),
        "option_rows": offsets[-1],
        "explored_choices": int(sum(explored)),
        "wins": int(outcome_counts[1]),
        "losses": int(outcome_counts[-1]),
        "draws": int(outcome_counts[0]),
        "maximum_reference_probability_error": maximum_probability_error,
        "maximum_reference_logit_error": maximum_logit_error,
        "exact_behavior_parity": True,
        "hidden_information_used": False,
        "official_engine_source": True,
        "builder_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": monotonic() - started,
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


class AdvantageHead(nn.Module):
    """Public state value plus behavior-centered state-option advantages."""

    def __init__(self, hidden_dim: int, width: int = 96) -> None:
        super().__init__()
        self.state_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU()
        )
        self.option_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU()
        )
        self.value_network = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1),
        )
        self.option_bias = nn.Linear(hidden_dim, 1)
        self.reference_scale = nn.Parameter(torch.zeros(()))
        self.interaction_scale = math.sqrt(width)

    def forward(
        self,
        state: torch.Tensor,
        option: torch.Tensor,
        reference_logit: torch.Tensor,
        behavior_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_hidden = self.state_projection(state)
        option_hidden = self.option_projection(option)
        raw = torch.einsum("bh,boh->bo", state_hidden, option_hidden) / self.interaction_scale
        raw = raw + self.option_bias(option).squeeze(-1)
        raw = raw + self.reference_scale * reference_logit.clamp(-20.0, 20.0)
        bounded = 0.75 * torch.tanh(raw)
        center = (bounded * behavior_probability).sum(dim=1, keepdim=True)
        advantage = bounded - center
        value = torch.tanh(self.value_network(state).squeeze(-1))
        q = value[:, None] + advantage
        return q, value, advantage


@dataclass(frozen=True)
class RaggedBatch:
    state: torch.Tensor
    option: torch.Tensor
    reference_logit: torch.Tensor
    reference_probability: torch.Tensor
    behavior_probability: torch.Tensor
    support: torch.Tensor
    action: torch.Tensor
    baseline: torch.Tensor
    target: torch.Tensor
    propensity: torch.Tensor
    weight: torch.Tensor
    index: torch.Tensor
    group: torch.Tensor
    explored: torch.Tensor


def collate_ragged(
    cache: Mapping[str, Any], indices: Sequence[int], device: torch.device
) -> RaggedBatch:
    rows = [int(index) for index in indices]
    offset: torch.Tensor = cache["offset"]
    counts = [int(offset[index + 1] - offset[index]) for index in rows]
    maximum = max(counts)
    hidden = int(cache["global_token"].shape[1])
    option = torch.zeros(len(rows), maximum, hidden)
    reference = torch.full((len(rows), maximum), -1.0e9)
    reference_probability = torch.zeros(len(rows), maximum)
    behavior_probability = torch.zeros(len(rows), maximum)
    support = torch.zeros(len(rows), maximum, dtype=torch.bool)
    for row, (index, count) in enumerate(zip(rows, counts, strict=True)):
        begin, end = int(offset[index]), int(offset[index]) + count
        option[row, :count] = cache["option_token"][begin:end].float()
        reference[row, :count] = cache["reference_logit"][begin:end]
        reference_probability[row, :count] = cache["reference_probability"][begin:end]
        behavior_probability[row, :count] = cache["behavior_probability"][begin:end]
        support[row, :count] = cache["support"][begin:end]
    propensity = cache["propensity"][rows].float()
    game_id = cache["game_id"][rows].long()
    base = 1.0 / cache["game_lengths"][game_id].float().clamp_min(1.0)
    inverse = torch.clamp(1.0 / propensity.clamp_min(1.0e-8), max=IPW_CLIP)
    weight = base * inverse
    weight = weight / weight.mean().clamp_min(1.0e-12)
    return RaggedBatch(
        cache["global_token"][rows].float().to(device), option.to(device),
        reference.to(device), reference_probability.to(device),
        behavior_probability.to(device), support.to(device),
        cache["action"][rows].long().to(device),
        cache["baseline"][rows].long().to(device),
        cache["outcome"][rows].float().to(device), propensity.to(device),
        weight.to(device), torch.tensor(rows, dtype=torch.long, device=device),
        cache["group_id"][rows].long().to(device), cache["explored"][rows].bool().to(device),
    )


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).sum() / weight.sum().clamp_min(1.0e-12)


def bootstrap_mask(indices: torch.Tensor, head: int) -> torch.Tensor:
    return ((indices * 37 + head * 17 + 11) % 5) != 0


def supported_advantage_penalty(
    advantage: torch.Tensor, support: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    per_row = (advantage.square() * support.float()).sum(1) / support.sum(1).clamp_min(1)
    return weighted_mean(per_row, weight)


def state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


@torch.no_grad()
def predict(
    cache: Mapping[str, Any], head1: AdvantageHead, head2: AdvantageHead,
    *, batch_size: int, device: torch.device
) -> dict[str, torch.Tensor]:
    count = int(cache["action"].numel())
    fields = {
        "q1_taken": torch.empty(count), "q2_taken": torch.empty(count),
        "v1": torch.empty(count), "v2": torch.empty(count),
        "q1_baseline": torch.empty(count), "q2_baseline": torch.empty(count),
        "best1": torch.empty(count, dtype=torch.long), "best2": torch.empty(count, dtype=torch.long),
        "margin1": torch.empty(count), "margin2": torch.empty(count),
    }
    head1.eval(); head2.eval()
    for start in range(0, count, batch_size):
        rows = list(range(start, min(count, start + batch_size)))
        batch = collate_ragged(cache, rows, device)
        q1, v1, _ = head1(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
        q2, v2, _ = head2(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
        q1 = q1.masked_fill(~batch.support, -1.0e9)
        q2 = q2.masked_fill(~batch.support, -1.0e9)
        row = torch.arange(len(rows), device=device)
        best1, best2 = q1.argmax(1), q2.argmax(1)
        sl = slice(start, start + len(rows))
        fields["q1_taken"][sl] = q1[row, batch.action].cpu()
        fields["q2_taken"][sl] = q2[row, batch.action].cpu()
        fields["v1"][sl] = v1.cpu(); fields["v2"][sl] = v2.cpu()
        fields["q1_baseline"][sl] = q1[row, batch.baseline].cpu()
        fields["q2_baseline"][sl] = q2[row, batch.baseline].cpu()
        fields["best1"][sl] = best1.cpu(); fields["best2"][sl] = best2.cpu()
        fields["margin1"][sl] = (q1[row, best1] - q1[row, batch.baseline]).cpu()
        fields["margin2"][sl] = (q2[row, best2] - q2[row, batch.baseline]).cpu()
    return fields


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positive = sum(labels); negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    rank_sum = 0.0; index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]: end += 1
        rank_sum += ((index + 1 + end) / 2.0) * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def fit_logistic_calibration(
    scores: torch.Tensor, labels: torch.Tensor
) -> tuple[float, float]:
    """Fit a deterministic train-only Platt calibrator."""

    x = scores.detach().double()
    y = labels.detach().double()
    rate = float(y.mean().clamp(1.0e-6, 1.0 - 1.0e-6))
    parameter = nn.Parameter(
        torch.tensor([1.0, math.log(rate / (1.0 - rate))], dtype=torch.float64)
    )
    optimizer = torch.optim.LBFGS(
        [parameter], lr=0.5, max_iter=100, tolerance_grad=1.0e-12,
        tolerance_change=1.0e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(parameter[0] * x + parameter[1], y)
        loss.backward()
        return loss

    optimizer.step(closure)
    if not torch.isfinite(parameter).all():
        raise RuntimeError("non-finite train-only outcome calibrator")
    return float(parameter[0].detach()), float(parameter[1].detach())


def calibration_probability(scores: torch.Tensor, calibration: Sequence[float]) -> torch.Tensor:
    return torch.sigmoid(float(calibration[0]) * scores + float(calibration[1]))


def effective_sample_size(cache: Mapping[str, Any], mask: torch.Tensor) -> tuple[float, float]:
    game_id = cache["game_id"].long()
    base = 1.0 / cache["game_lengths"][game_id].float().clamp_min(1.0)
    inverse = torch.clamp(1.0 / cache["propensity"].float().clamp_min(1e-8), max=IPW_CLIP)
    weight = (base * inverse)[mask]
    ess = float(weight.sum().square() / weight.square().sum().clamp_min(1e-12))
    return ess, ess / max(1, int(mask.sum()))


def metrics(
    cache: Mapping[str, Any], prediction: Mapping[str, torch.Tensor],
    checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    outcome = cache["outcome"].float()
    labels = (outcome > 0).long()
    q_score = torch.minimum(prediction["q1_taken"], prediction["q2_taken"])
    state_score = torch.minimum(prediction["v1"], prediction["v2"])
    offsets = cache["offset"].long(); action = cache["action"].long()
    reference_score = torch.empty(len(action))
    for index in range(len(action)):
        reference_score[index] = cache["reference_logit"][int(offsets[index]) + int(action[index])]
    game_lengths = cache["game_lengths"].long()
    game_starts = torch.cat((torch.zeros(1, dtype=torch.long), torch.cumsum(game_lengths, 0)[:-1]))
    explored = cache["explored"].bool()
    group = cache["group_id"].long()
    same_best = prediction["best1"] == prediction["best2"]
    nonbaseline = prediction["best1"] != cache["baseline"].long()
    minimum_margin = torch.minimum(prediction["margin1"], prediction["margin2"])

    def subset(mask: torch.Tensor) -> dict[str, Any]:
        count = int(mask.sum())
        if count == 0:
            return {"choices": 0}
        y = labels[mask]
        q = q_score[mask]
        reference = reference_score[mask]
        state = state_score[mask]
        q_probability = calibration_probability(q, checkpoint["q_calibration"])
        reference_probability = calibration_probability(reference, checkpoint["reference_calibration"])
        state_probability = calibration_probability(state, checkpoint["state_calibration"])
        ess, ratio = effective_sample_size(cache, mask)
        return {
            "choices": count,
            "wins": int(y.sum()),
            "q_auc": binary_auc(q.tolist(), y.tolist()),
            "reference_logit_auc": binary_auc(reference.tolist(), y.tolist()),
            "state_auc": binary_auc(state.tolist(), y.tolist()),
            "q_brier": float((q_probability - y.float()).square().mean()),
            "reference_logit_brier": float((reference_probability - y.float()).square().mean()),
            "state_brier": float((state_probability - y.float()).square().mean()),
            "q_mae": float((q_probability - y.float()).abs().mean()),
            "reference_logit_mae": float((reference_probability - y.float()).abs().mean()),
            "effective_sample_size": ess,
            "effective_sample_ratio": ratio,
        }

    thresholds = {}
    for threshold in THRESHOLD_GRID:
        change = same_best & nonbaseline & (minimum_margin >= threshold)
        thresholds[str(threshold)] = {
            "changes": int(change.sum()),
            "change_rate": float(change.float().mean()),
            "minimum_margin": float(minimum_margin[change].min()) if bool(change.any()) else None,
        }
    by_group = {GROUPS[index]: subset(group == index) for index in range(len(GROUPS))}
    return {
        "choices": len(action),
        "games": len(cache["files"]),
        "overall": subset(torch.ones_like(explored)),
        "first_eligible_per_game": subset(torch.zeros_like(explored).scatter(0, game_starts, True)),
        "explored_actions": subset(explored),
        "baseline_actions": subset(~explored),
        "by_group": by_group,
        "head_best_agreement": float(same_best.float().mean()),
        "threshold_grid": thresholds,
    }


def train(
    *, cache_path: Path, output: Path, config_name: str, batch_size: int, threads: int
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing training output: {output}")
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=True)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("split") != "train":
        raise ValueError("training requires frozen train cache")
    if cache.get("corpus_sha256") != CORPUS_SHA256:
        raise ValueError("training corpus mismatch")
    config = CONFIGS[config_name]
    seed = int(config["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads)); device = torch.device("cpu")
    hidden = int(cache["global_token"].shape[1]); width = int(config["width"])
    head1 = AdvantageHead(hidden, width).to(device)
    torch.manual_seed(seed + 1); head2 = AdvantageHead(hidden, width).to(device)
    parameters = list(head1.parameters()) + list(head2.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    count = int(cache["action"].numel()); history = []; started = monotonic()
    for epoch in range(1, int(config["epochs"]) + 1):
        order = list(range(count)); random.Random(seed + epoch).shuffle(order)
        totals = Counter(); updates = 0; head1.train(); head2.train()
        for start in range(0, count, batch_size):
            rows = order[start:start + batch_size]
            batch = collate_ragged(cache, rows, device)
            q1, _, advantage1 = head1(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
            q2, _, advantage2 = head2(batch.state, batch.option, batch.reference_logit, batch.behavior_probability)
            row = torch.arange(len(rows), device=device)
            selected1 = q1[row, batch.action]; selected2 = q2[row, batch.action]
            mask1 = bootstrap_mask(batch.index, 1).float(); mask2 = bootstrap_mask(batch.index, 2).float()
            weight1 = batch.weight * mask1; weight2 = batch.weight * mask2
            loss1 = weighted_mean(F.smooth_l1_loss(selected1, batch.target, reduction="none"), weight1)
            loss2 = weighted_mean(F.smooth_l1_loss(selected2, batch.target, reduction="none"), weight2)
            regularization1 = supported_advantage_penalty(advantage1, batch.support, weight1)
            regularization2 = supported_advantage_penalty(advantage2, batch.support, weight2)
            regularization = float(config["advantage_regularization"]) * (regularization1 + regularization2)
            loss = loss1 + loss2 + regularization
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
            totals["head1"] += float(loss1.detach()); totals["head2"] += float(loss2.detach())
            totals["regularization"] += float(regularization.detach()); updates += 1
        row_history = {"epoch": epoch, **{key: value / max(1, updates) for key, value in totals.items()},
                       "seconds": monotonic() - started}
        history.append(row_history); print(json.dumps(row_history, sort_keys=True), flush=True)

    prediction = predict(cache, head1, head2, batch_size=batch_size, device=device)
    q_score = torch.minimum(prediction["q1_taken"], prediction["q2_taken"])
    state_score = torch.minimum(prediction["v1"], prediction["v2"])
    offsets = cache["offset"].long(); action = cache["action"].long()
    reference_score = torch.tensor([
        float(cache["reference_logit"][int(offsets[index]) + int(action[index])])
        for index in range(len(action))
    ])
    labels = (cache["outcome"].float() > 0).long()
    q_calibration = fit_logistic_calibration(q_score, labels)
    reference_calibration = fit_logistic_calibration(reference_score, labels)
    state_calibration = fit_logistic_calibration(state_score, labels)
    checkpoint = {
        "schema": MODEL_SCHEMA, "config_name": config_name, "config": config,
        "hidden_dim": hidden, "width": width, "head1": state_dict_cpu(head1),
        "head2": state_dict_cpu(head2), "q_calibration": q_calibration,
        "reference_calibration": reference_calibration, "state_calibration": state_calibration,
        "minimum_reference_support": MINIMUM_REFERENCE_SUPPORT,
        "ipw_clip": IPW_CLIP, "corpus_sha256": CORPUS_SHA256,
        "deck_sha256": DECK_SHA256, "model_sha256": MODEL_SHA256,
    }
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "advantage_heads.pt"; torch.save(checkpoint, checkpoint_path)
    train_metrics = metrics(cache, prediction, checkpoint)
    report = {
        "schema": MODEL_SCHEMA, "config_name": config_name, "config": config,
        "cache": str(cache_path.resolve()), "cache_sha256": sha256(cache_path.resolve()),
        "games": len(cache["files"]), "choices": count, "history": history,
        "training": "twin bootstrap behavior-centered terminal-outcome regression",
        "propensity": {"known_behavior": True, "inverse_weighted": True, "clip": IPW_CLIP},
        "frozen_live_trunk": True, "hidden_information_used": False,
        "train_metrics": train_metrics, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path), "trainer_sha256": sha256(Path(__file__).resolve()),
        "elapsed_seconds": monotonic() - started, "development_only": True,
        "deployment_authority": False,
    }
    write_json(output / "training_report.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def load_heads(path: Path) -> tuple[dict[str, Any], AdvantageHead, AdvantageHead]:
    checkpoint = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != MODEL_SCHEMA:
        raise ValueError("unsupported advantage checkpoint")
    head1 = AdvantageHead(int(checkpoint["hidden_dim"]), int(checkpoint["width"]))
    head2 = AdvantageHead(int(checkpoint["hidden_dim"]), int(checkpoint["width"]))
    head1.load_state_dict(checkpoint["head1"]); head2.load_state_dict(checkpoint["head2"])
    return checkpoint, head1.eval(), head2.eval()


def evaluate(
    *, cache_path: Path, heads_path: Path, output: Path, batch_size: int, threads: int
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing evaluation: {output}")
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=True)
    if cache.get("schema") != CACHE_SCHEMA or cache.get("split") not in {"validation", "test"}:
        raise ValueError("evaluation requires a held-out cache")
    checkpoint, head1, head2 = load_heads(heads_path)
    if checkpoint.get("corpus_sha256") != cache.get("corpus_sha256"):
        raise ValueError("evaluation corpus provenance mismatch")
    torch.set_num_threads(max(1, threads))
    prediction = predict(cache, head1, head2, batch_size=batch_size, device=torch.device("cpu"))
    evaluation = metrics(cache, prediction, checkpoint)
    report = {
        "schema": EVALUATION_SCHEMA, "split": cache["split"],
        "config_name": checkpoint["config_name"], "cache": str(cache_path.resolve()),
        "cache_sha256": sha256(cache_path.resolve()), "heads": str(heads_path.resolve()),
        "heads_sha256": sha256(heads_path.resolve()), "metrics": evaluation,
        "held_out": True, "hidden_information_used": False,
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "development_only": True, "deployment_authority": False,
    }
    write_json(output, report); print(json.dumps(report, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--split", choices=("train", "validation", "test"), required=True)
    cache.add_argument("--corpus", type=Path, default=CORPUS)
    cache.add_argument("--candidate", type=Path, default=CANDIDATE)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--threads", type=int, default=1)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--cache", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--config", choices=tuple(CONFIGS), required=True)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--threads", type=int, default=1)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--cache", type=Path, required=True)
    evaluate_parser.add_argument("--heads", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=256)
    evaluate_parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "cache":
        build_cache(split=args.split, corpus=args.corpus, candidate=args.candidate,
                    output=args.output, threads=args.threads)
    elif args.command == "train":
        train(cache_path=args.cache, output=args.output, config_name=args.config,
              batch_size=args.batch_size, threads=args.threads)
    elif args.command == "evaluate":
        evaluate(cache_path=args.cache, heads_path=args.heads, output=args.output,
                 batch_size=args.batch_size, threads=args.threads)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
