"""Train and export the Phase 6AN integrated official controller.

This module is intentionally separate from the dependency-free deployed
runtime. PyTorch is used here to fit a frozen public-belief plan blueprint and
an autoregressive tactical policy. Exported tensors are ordinary JSON lists
consumed by :mod:`rocketbot.integrated_planner`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .competition import (
    CompetitionAgentConfig,
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    Phase6CompetitionAgent,
    SelectContext,
    SelectionContract,
    _safe_int,
    _state,
    resolve_official_deck,
)
from .integrated_planner import (
    ACTION_FEATURE_NAMES,
    ARCHETYPE_NAMES,
    INTEGRATED_PLANNER_SCHEMA_VERSION,
    INTENT_NAMES,
    PLAN_NAMES,
    PLANNER_FEATURE_NAMES,
    PREFIX_FEATURE_NAMES,
    RESOURCE_NAMES,
    STRATEGIC_AXES,
    TOKEN_FEATURE_NAMES,
    FrozenStrategicPlanner,
    PlannerMemory,
    _fuse_intent_evidence,
    _AXIS_TYPE_BIASES,
    _PLAN_TYPE_BIASES,
    action_feature_rows,
    behavior_intent_probabilities,
    planner_feature_items,
    prefix_feature_items,
    token_feature_rows,
)


TRAINING_SEED = 61_406
PLAN_HIDDEN_1 = 96
PLAN_HIDDEN_2 = 64
ACTION_STATE_HIDDEN = 96
TOKEN_HIDDEN = 48
ATTENTION_HIDDEN = 32
ACTION_HIDDEN = 64
PLAN_MIRROR_DESCENT_RATE = 1.75


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _normalizer(rows: Sequence[Mapping[str, float]], names: Sequence[str]) -> dict[str, Any]:
    if not rows:
        raise ValueError("normalizer requires at least one row")
    means: list[float] = []
    scales: list[float] = []
    for name in names:
        values = [float(row.get(name, 0.0)) for row in rows]
        mean = statistics.fmean(values)
        variance = statistics.fmean((value - mean) ** 2 for value in values)
        means.append(mean)
        scales.append(max(math.sqrt(variance), 1e-4))
    return {"names": list(names), "means": means, "scales": scales}


def _normalize_row(row: Mapping[str, float], normalizer: Mapping[str, Any]) -> list[float]:
    return [
        (float(row.get(str(name), 0.0)) - float(mean)) / max(float(scale), 1e-8)
        for name, mean, scale in zip(
            normalizer["names"], normalizer["means"], normalizer["scales"], strict=True
        )
    ]


def _finite_quantile(values: Sequence[float], coverage: float) -> float:
    if not values or not 0.0 < coverage < 1.0:
        raise ValueError("finite-sample quantile inputs are invalid")
    ordered = sorted(float(value) for value in values)
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * coverage))
    return ordered[rank - 1]


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    maximum = max(values)
    terms = [math.exp(value - maximum) for value in values]
    total = sum(terms)
    return tuple(value / total for value in terms)


@dataclass(frozen=True, slots=True)
class SurfaceExample:
    root_id: str
    split: str
    archetype_index: int
    intent_index: int
    features: Mapping[str, float]
    previous_plan: tuple[float, ...]
    plan_target: tuple[float, ...]
    plan_values: tuple[float, ...]
    strategic_target: tuple[float, ...]
    meta_target: tuple[float, ...]
    aggregate_target: float
    uncertainty_target: float
    gate_target: float
    resource_target: tuple[float, ...]


def _strategic_targets(features: Mapping[str, float]) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    turn = float(features.get("turn", 0.0))
    own_board = float(features.get("own_board_count", 0.0))
    opposing_board = float(features.get("opponent_board_count", 0.0))
    own_hp = float(features.get("own_board_hp", 0.0))
    opposing_hp = float(features.get("opponent_board_hp", 0.0))
    own_hand = float(features.get("own_hand_count", 0.0))
    opposing_hand = float(features.get("opponent_hand_count", 0.0))
    own_prizes = float(features.get("own_prize_count", 6.0))
    opposing_prizes = float(features.get("opponent_prize_count", 6.0))
    option_count = float(features.get("selection_option_count", 0.0))
    entropy = float(features.get("opponent_range_entropy", 0.0))
    values = (
        math.tanh((own_board - 3.0) / 2.0 + (own_hp - 500.0) / 900.0),
        math.tanh((opposing_prizes - own_prizes) / 2.0 + (own_board - opposing_board) / 4.0),
        math.tanh((own_hand - opposing_hand) / 7.0 + (features.get("own_deck_count", 0.0) - 15.0) / 45.0),
        math.tanh((opposing_prizes - own_prizes) / 1.75),
        math.tanh((own_hp - opposing_hp) / 550.0 + (own_board - opposing_board) / 3.0),
        math.tanh((option_count - 3.0) / 5.0 + own_hand / 14.0 + entropy / 4.0),
    )
    early = max(0.0, min(1.0, (12.0 - turn) / 12.0))
    close_prize = max(0.0, (3.0 - min(own_prizes, opposing_prizes)) / 3.0)
    raw_weights = (
        0.5 + 2.0 * early + max(0.0, 3.0 - own_board) / 2.0,
        0.8 + (1.0 - early) + close_prize,
        0.6 + max(0.0, 5.0 - own_hand) / 3.0,
        0.6 + 2.2 * close_prize,
        0.7 + max(0.0, opposing_board - own_board) / 2.0,
        0.7 + entropy / 2.0 + max(0.0, 3.0 - option_count) / 3.0,
    )
    weights = _softmax(raw_weights)
    aggregate = sum(value * weight for value, weight in zip(values, weights, strict=True))
    return values, weights, aggregate


def _resource_categories(runtime: Path, candidate_decks: Mapping[str, Sequence[int]]) -> dict[str, tuple[float, ...]]:
    runtime_text = str(runtime.resolve())
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    from cg.api import all_card_data  # type: ignore[import-not-found]

    metadata = {int(card.cardId): card for card in all_card_data()}
    result: dict[str, tuple[float, ...]] = {}
    for candidate, deck in candidate_decks.items():
        counts = Counter()
        for card_id in deck:
            card = metadata[int(card_id)]
            text = " ".join(str(skill.text).lower() for skill in card.skills)
            name = str(card.name).lower()
            if bool(card.basic):
                counts["basic_pokemon"] += 1
            if bool(card.stage1) or bool(card.stage2):
                counts["evolution"] += 1
            if int(card.cardType) in (5, 6):
                counts["energy"] += 1
            if "search your deck" in text or "look at" in text:
                counts["search"] += 1
            if "draw " in text:
                counts["draw"] += 1
            if "switch" in text or "retreat" in text:
                counts["switch_mobility"] += 1
            if "discard pile" in text and ("put" in text or "attach" in text):
                counts["recovery"] += 1
            if "opponent" in text and ("discard" in text or "switch" in text or "can't" in text):
                counts["disruption"] += 1
            if int(card.cardType) == 2 or "tool" in name:
                counts["tool"] += 1
            if int(card.cardType) == 4:
                counts["stadium"] += 1
        result[candidate] = tuple(
            min(0.98, max(0.02, 1.0 - (1.0 - counts[name] / 60.0) ** 7))
            for name in RESOURCE_NAMES
        )
    return result


def load_surface_examples(dataset_path: Path, runtime: Path, trajectory_path: Path) -> tuple[list[SurfaceExample], dict[str, Any]]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    candidate_decks = {
        str(item["candidate_id"]): tuple(int(value) for value in item["card_ids"])
        for item in trajectory["candidates"]
    }
    resource_targets = _resource_categories(runtime, candidate_decks)
    examples: list[SurfaceExample] = []
    for root in raw["roots"]:
        feature_map: dict[str, float] = {}
        for group in ("public", "private", "history"):
            feature_map.update({str(name): float(value) for name, value in root["features"][group]})
        blueprint_map = {str(name).removeprefix("plan:"): float(value) for name, value in root["features"]["blueprint_probabilities"]}
        blueprint = tuple(blueprint_map[name] for name in PLAN_NAMES)
        action_returns: dict[str, list[float]] = {name: [] for name in PLAN_NAMES}
        for surface in root["surfaces"]:
            for action in surface["action_values"]:
                name = str(action["action_id"]).removeprefix("plan:")
                action_returns[name].extend(float(value) for value in action["terminal_returns"])
        q_values = tuple(statistics.fmean(action_returns[name]) for name in PLAN_NAMES)
        standard_errors = tuple(
            statistics.stdev(action_returns[name]) / math.sqrt(len(action_returns[name]))
            for name in PLAN_NAMES
        )
        target_logits = [
            math.log(max(probability, 1e-8)) + PLAN_MIRROR_DESCENT_RATE * value
            for probability, value in zip(blueprint, q_values, strict=True)
        ]
        plan_target = _softmax(target_logits)
        strategic, meta, aggregate = _strategic_targets(feature_map)
        baseline = sum(probability * value for probability, value in zip(blueprint, q_values, strict=True))
        improvement = max(q_values) - baseline
        spread = max(q_values) - min(q_values)
        gate = float(improvement >= 0.12 or (max(standard_errors) >= 0.22 and spread >= 0.20))
        candidate = str(root["match"]["candidate_id_hypothesis_input"])
        intent = str(root["match"]["intent_id_supervision_only"])
        examples.append(
            SurfaceExample(
                root_id=str(root["root_id"]),
                split=str(root["match"]["calibration_partition"]),
                archetype_index=ARCHETYPE_NAMES.index(candidate),
                intent_index=INTENT_NAMES.index(intent),
                features=feature_map,
                previous_plan=blueprint,
                plan_target=plan_target,
                plan_values=q_values,
                strategic_target=strategic,
                meta_target=meta,
                aggregate_target=aggregate,
                uncertainty_target=max(standard_errors),
                gate_target=gate,
                resource_target=resource_targets[candidate],
            )
        )
    provenance = {
        "surface_dataset_fingerprint": str(raw["fingerprint"]),
        "surface_root_count": len(examples),
        "official_terminal_continuations": sum(
            len(action["terminal_returns"])
            for root in raw["roots"]
            for surface in root["surfaces"]
            for action in surface["action_values"]
        ),
        "partitions": dict(Counter(example.split for example in examples)),
    }
    return examples, provenance


class PlannerNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.planner_hidden_1 = nn.Linear(input_size + len(PLAN_NAMES), PLAN_HIDDEN_1)
        self.planner_hidden_2 = nn.Linear(PLAN_HIDDEN_1, PLAN_HIDDEN_2)
        self.archetype_head = nn.Linear(PLAN_HIDDEN_2, len(ARCHETYPE_NAMES))
        self.intent_head = nn.Linear(PLAN_HIDDEN_2, len(INTENT_NAMES))
        self.resource_head = nn.Linear(PLAN_HIDDEN_2, len(RESOURCE_NAMES))
        self.plan_policy_head = nn.Linear(PLAN_HIDDEN_2, len(PLAN_NAMES))
        self.belief_plan_bridge = nn.Linear(
            len(ARCHETYPE_NAMES) + len(INTENT_NAMES) + len(RESOURCE_NAMES),
            len(PLAN_NAMES),
            bias=True,
        )
        nn.init.zeros_(self.belief_plan_bridge.bias)
        self.plan_value_head = nn.Linear(PLAN_HIDDEN_2, len(PLAN_NAMES))
        self.strategic_head = nn.Linear(PLAN_HIDDEN_2, len(STRATEGIC_AXES))
        self.meta_head = nn.Linear(PLAN_HIDDEN_2, len(STRATEGIC_AXES))
        self.uncertainty_head = nn.Linear(PLAN_HIDDEN_2, 1)
        self.search_gate_head = nn.Linear(PLAN_HIDDEN_2, 1)
        nn.init.constant_(self.uncertainty_head.bias, 0.25)

    def forward(self, features: Tensor, previous_plan: Tensor) -> dict[str, Tensor]:
        hidden = torch.tanh(self.planner_hidden_1(torch.cat((features, previous_plan), dim=-1)))
        hidden = torch.tanh(self.planner_hidden_2(hidden))
        strategic = torch.tanh(self.strategic_head(hidden))
        meta_logits = self.meta_head(hidden)
        meta = torch.softmax(meta_logits, dim=-1)
        archetype_logits = self.archetype_head(hidden)
        intent_logits = self.intent_head(hidden)
        resource_logits = self.resource_head(hidden)
        belief_vector = torch.cat(
            (
                torch.softmax(archetype_logits, dim=-1),
                torch.softmax(intent_logits, dim=-1),
                torch.sigmoid(resource_logits),
            ),
            dim=-1,
        )
        plan_logits = self.plan_policy_head(hidden) + self.belief_plan_bridge(belief_vector)
        return {
            "hidden": hidden,
            "archetype_logits": archetype_logits,
            "intent_logits": intent_logits,
            "resource_logits": resource_logits,
            "plan_logits": plan_logits,
            "plan_values": torch.tanh(self.plan_value_head(hidden)),
            "strategic": strategic,
            "meta_logits": meta_logits,
            "meta": meta,
            "aggregate": (strategic * meta).sum(dim=-1),
            "uncertainty": torch.relu(self.uncertainty_head(hidden)).squeeze(-1),
            "gate_logits": self.search_gate_head(hidden).squeeze(-1),
        }


def _surface_tensors(examples: Sequence[SurfaceExample], normalizer: Mapping[str, Any]) -> dict[str, Tensor]:
    return {
        "x": torch.tensor([_normalize_row(example.features, normalizer) for example in examples], dtype=torch.float32),
        "previous": torch.tensor([example.previous_plan for example in examples], dtype=torch.float32),
        "archetype": torch.tensor([example.archetype_index for example in examples], dtype=torch.long),
        "intent": torch.tensor([example.intent_index for example in examples], dtype=torch.long),
        "resource": torch.tensor([example.resource_target for example in examples], dtype=torch.float32),
        "plan": torch.tensor([example.plan_target for example in examples], dtype=torch.float32),
        "q": torch.tensor([example.plan_values for example in examples], dtype=torch.float32),
        "strategic": torch.tensor([example.strategic_target for example in examples], dtype=torch.float32),
        "meta": torch.tensor([example.meta_target for example in examples], dtype=torch.float32),
        "aggregate": torch.tensor([example.aggregate_target for example in examples], dtype=torch.float32),
        "uncertainty": torch.tensor([example.uncertainty_target for example in examples], dtype=torch.float32),
        "gate": torch.tensor([example.gate_target for example in examples], dtype=torch.float32),
    }


def _planner_loss(outputs: Mapping[str, Tensor], target: Mapping[str, Tensor]) -> Tensor:
    plan_log = torch.log_softmax(outputs["plan_logits"], dim=-1)
    meta_log = torch.log_softmax(outputs["meta_logits"], dim=-1)
    return (
        F.cross_entropy(outputs["archetype_logits"], target["archetype"])
        + F.cross_entropy(outputs["intent_logits"], target["intent"])
        + 0.55 * F.binary_cross_entropy_with_logits(outputs["resource_logits"], target["resource"])
        + 1.4 * (-(target["plan"] * plan_log).sum(dim=-1).mean())
        + 1.8 * F.mse_loss(outputs["plan_values"], target["q"])
        + 0.65 * F.mse_loss(outputs["strategic"], target["strategic"])
        + 0.5 * (-(target["meta"] * meta_log).sum(dim=-1).mean())
        + 0.6 * F.mse_loss(outputs["aggregate"], target["aggregate"])
        + 0.35 * F.mse_loss(outputs["uncertainty"], target["uncertainty"])
        + 0.5 * F.binary_cross_entropy_with_logits(outputs["gate_logits"], target["gate"])
    )


def fit_planner(examples: Sequence[SurfaceExample]) -> tuple[PlannerNetwork, dict[str, Any], dict[str, Any]]:
    torch.manual_seed(TRAINING_SEED)
    random.seed(TRAINING_SEED)
    train = [example for example in examples if example.split == "train"]
    selection = [example for example in examples if example.split == "selection"]
    normalizer = _normalizer([example.features for example in train], PLANNER_FEATURE_NAMES)
    tensors = {name: _surface_tensors(rows, normalizer) for name, rows in (("train", train), ("selection", selection))}
    model = PlannerNetwork(len(PLANNER_FEATURE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=1e-4)
    best_state: dict[str, Tensor] | None = None
    best_selection = math.inf
    best_epoch = -1
    stale = 0
    for epoch in range(1_500):
        model.train()
        optimizer.zero_grad()
        output = model(tensors["train"]["x"], tensors["train"]["previous"])
        loss = _planner_loss(output, tensors["train"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(
                _planner_loss(
                    model(tensors["selection"]["x"], tensors["selection"]["previous"]),
                    tensors["selection"],
                )
            )
        if validation < best_selection - 1e-5:
            best_selection = validation
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 180:
            break
    if best_state is None:
        raise RuntimeError("planner training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics: dict[str, Any] = {"best_epoch": best_epoch, "selection_loss": best_selection}
    for split in ("train", "selection", "bounds", "untouched"):
        rows = [example for example in examples if example.split == split]
        target = _surface_tensors(rows, normalizer)
        with torch.no_grad():
            output = model(target["x"], target["previous"])
            metrics[split] = {
                "count": len(rows),
                "archetype_accuracy": float((output["archetype_logits"].argmax(-1) == target["archetype"]).float().mean()),
                "intent_accuracy": float((output["intent_logits"].argmax(-1) == target["intent"]).float().mean()),
                "plan_value_rmse": float(torch.sqrt(F.mse_loss(output["plan_values"], target["q"]))),
                "plan_policy_cross_entropy": float((-(target["plan"] * torch.log_softmax(output["plan_logits"], -1)).sum(-1)).mean()),
                "strategic_rmse": float(torch.sqrt(F.mse_loss(output["strategic"], target["strategic"]))),
                "gate_accuracy": float(((torch.sigmoid(output["gate_logits"]) >= 0.5) == (target["gate"] >= 0.5)).float().mean()),
            }
    return model, normalizer, metrics


def _temperature(logits: Tensor, target: Tensor, *, soft_target: bool) -> float:
    best = (math.inf, 1.0)
    for temperature in (0.40, 0.55, 0.70, 0.85, 1.0, 1.20, 1.50, 2.0, 2.5):
        log_probabilities = torch.log_softmax(logits / temperature, dim=-1)
        loss = (
            float((-(target * log_probabilities).sum(-1)).mean())
            if soft_target
            else float(F.nll_loss(log_probabilities, target))
        )
        best = min(best, (loss, temperature))
    return best[1]


def calibrate_planner(
    model: PlannerNetwork,
    examples: Sequence[SurfaceExample],
    normalizer: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = [example for example in examples if example.split == "selection"]
    selection_tensors = _surface_tensors(selection, normalizer)
    with torch.no_grad():
        selection_output = model(selection_tensors["x"], selection_tensors["previous"])
    temperatures = {
        "archetype_temperature": _temperature(selection_output["archetype_logits"], selection_tensors["archetype"], soft_target=False),
        "intent_temperature": _temperature(selection_output["intent_logits"], selection_tensors["intent"], soft_target=False),
        "plan_temperature": _temperature(selection_output["plan_logits"], selection_tensors["plan"], soft_target=True),
    }
    model_intent = torch.softmax(
        selection_output["intent_logits"] / temperatures["intent_temperature"], dim=-1
    )
    best_behavior = (math.inf, 0.0)
    for weight in (0.0, 0.10, 0.20, 0.30, 0.40, 0.55, 0.70):
        loss = 0.0
        for index, example in enumerate(selection):
            fused = _fuse_intent_evidence(
                tuple(float(value) for value in model_intent[index]),
                behavior_intent_probabilities(example.features),
                weight,
            )
            loss -= math.log(max(fused[example.intent_index], 1e-12))
        best_behavior = min(best_behavior, (loss / len(selection), weight))
    temperatures["behavior_evidence_weight"] = best_behavior[1]

    def search_rows(split: str) -> list[tuple[float, float]]:
        rows = [example for example in examples if example.split == split]
        tensors = _surface_tensors(rows, normalizer)
        with torch.no_grad():
            output = model(tensors["x"], tensors["previous"])
            gate_probabilities = torch.sigmoid(output["gate_logits"])
            plan_probabilities = torch.softmax(
                output["plan_logits"] / temperatures["plan_temperature"], dim=-1
            )
        result = []
        for index, example in enumerate(rows):
            baseline = sum(
                float(plan_probabilities[index, plan_index]) * example.plan_values[plan_index]
                for plan_index in range(len(PLAN_NAMES))
            )
            result.append((float(gate_probabilities[index]), max(example.plan_values) - baseline))
        return result

    selection_search = search_rows("selection")
    threshold_candidates: list[tuple[float, float]] = []
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        gated = [benefit for probability, benefit in selection_search if probability >= threshold]
        routine = [benefit for probability, benefit in selection_search if probability < threshold]
        if gated and routine and 0.15 <= len(gated) / len(selection_search) <= 0.70:
            threshold_candidates.append((statistics.fmean(gated) - statistics.fmean(routine), threshold))
    search_gate_threshold = max(threshold_candidates, default=(0.0, 0.50))[1]

    def advantage_rows(split: str) -> list[tuple[str, int, float]]:
        rows = [example for example in examples if example.split == split]
        tensors = _surface_tensors(rows, normalizer)
        with torch.no_grad():
            output = model(tensors["x"], tensors["previous"])
            predicted_q = output["plan_values"]
            probabilities = torch.softmax(output["plan_logits"] / temperatures["plan_temperature"], dim=-1)
        result: list[tuple[str, int, float]] = []
        for row_index, example in enumerate(rows):
            predicted_baseline = float((predicted_q[row_index] * probabilities[row_index]).sum())
            true_baseline = sum(
                float(probabilities[row_index, plan_index]) * example.plan_values[plan_index]
                for plan_index in range(len(PLAN_NAMES))
            )
            for plan_index in range(len(PLAN_NAMES)):
                predicted_advantage = float(predicted_q[row_index, plan_index]) - predicted_baseline
                true_advantage = example.plan_values[plan_index] - true_baseline
                result.append(
                    (
                        ARCHETYPE_NAMES[example.archetype_index],
                        plan_index,
                        abs(predicted_advantage - true_advantage),
                    )
                )
        return result

    bounds_rows = advantage_rows("bounds")
    radius = _finite_quantile([row[2] for row in bounds_rows], 0.95)
    untouched = advantage_rows("untouched")
    marginal = statistics.fmean(float(error <= radius + 1e-12) for _, _, error in untouched)
    slices = {
        archetype: statistics.fmean(
            float(error <= radius + 1e-12)
            for row_archetype, _, error in untouched
            if row_archetype == archetype
        )
        for archetype in ARCHETYPE_NAMES
    }
    admitted = bool(marginal >= 0.90 and min(slices.values()) >= 0.80)
    calibration = {
        **temperatures,
        "search_gate_threshold": search_gate_threshold,
        "counterfactual_bounds_admitted": admitted,
        "counterfactual_advantage_radius": radius,
        "safe_non_regression_tolerance": 0.01,
        "safe_improvement_margin": 0.05,
        "maximum_resolve_mass": 0.50,
        "plan_persistence": 0.35,
        "opponent_exploitation_admitted": False,
        "exploitation_weight": 0.0,
    }
    untouched_search = search_rows("untouched")
    gated_benefit = [benefit for probability, benefit in untouched_search if probability >= search_gate_threshold]
    routine_benefit = [benefit for probability, benefit in untouched_search if probability < search_gate_threshold]
    report = {
        "method": "95% split-conformal absolute counterfactual-advantage residual",
        "bounds_count": len(bounds_rows),
        "untouched_count": len(untouched),
        "radius": radius,
        "untouched_marginal_coverage": marginal,
        "untouched_archetype_coverage": slices,
        "admitted": admitted,
        "claim_boundary": "conditional empirical guard; not equilibrium safety",
        "selective_search": {
            "threshold_selected_on_selection": search_gate_threshold,
            "untouched_gated_count": len(gated_benefit),
            "untouched_routine_count": len(routine_benefit),
            "untouched_gated_mean_available_improvement": statistics.fmean(gated_benefit) if gated_benefit else 0.0,
            "untouched_routine_mean_available_improvement": statistics.fmean(routine_benefit) if routine_benefit else 0.0,
            "gain_concentrated_in_gated_states": bool(gated_benefit and routine_benefit and statistics.fmean(gated_benefit) > statistics.fmean(routine_benefit)),
        },
        "behavior_evidence": {
            "selection_weight": best_behavior[1],
            "selection_nll": best_behavior[0],
        },
    }
    return calibration, report


@dataclass(slots=True)
class ActionExample:
    split: str
    source: str
    match_id: str
    planner_row: Mapping[str, float]
    evaluation: tuple[float, ...]
    token_rows: tuple[Mapping[str, float], ...]
    action_rows: tuple[Mapping[str, float], ...]
    prefix_row: Mapping[str, float]
    target: int
    context: int


def _file_split(source: str, match_id: str) -> str:
    bucket = int.from_bytes(sha256(f"phase6an:{source}:{match_id}".encode()).digest()[:4], "big") % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def _planner_predict(
    model: PlannerNetwork,
    normalizer: Mapping[str, Any],
    calibration: Mapping[str, Any],
    features: Mapping[str, float],
    previous: Sequence[float],
) -> tuple[float, ...]:
    x = torch.tensor([_normalize_row(features, normalizer)], dtype=torch.float32)
    p = torch.tensor([list(previous)], dtype=torch.float32)
    with torch.no_grad():
        output = model(x, p)
        archetype = torch.softmax(output["archetype_logits"] / float(calibration["archetype_temperature"]), -1)[0]
        intent = torch.softmax(output["intent_logits"] / float(calibration["intent_temperature"]), -1)[0]
        resources = torch.sigmoid(output["resource_logits"])[0]
        plan = torch.softmax(output["plan_logits"] / float(calibration["plan_temperature"]), -1)[0]
        strategic = output["strategic"][0]
        meta = output["meta"][0]
        aggregate = output["aggregate"][0]
        uncertainty = output["uncertainty"][0]
    fused_intent = torch.tensor(
        _fuse_intent_evidence(
            tuple(float(value) for value in intent),
            behavior_intent_probabilities(features),
            float(calibration.get("behavior_evidence_weight", 0.0)),
        ),
        dtype=torch.float32,
    )
    return tuple(float(value) for value in torch.cat((archetype, fused_intent, resources, plan, strategic, meta, aggregate.view(1), uncertainty.view(1))))


def _parse_replay_file(
    path: Path,
    *,
    source: str,
    target_name: str | None,
    planner_model: PlannerNetwork,
    planner_normalizer: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> list[ActionExample]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    meta = rows[0].get("meta", {})
    match_id = str(meta.get("match_id", path.stem))
    if target_name is None:
        target_seat = int(meta["target_seat"])
    else:
        players = list(meta.get("players", ()))
        if target_name not in players:
            return []
        target_seat = players.index(target_name)
    split = _file_split(source, match_id)
    deck = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    prior = Phase6CompetitionAgent(
        deck,
        config=CompetitionAgentConfig(decision_budget_ms=1_000.0, search_budget_ms=0.0),
    )
    memory = PlannerMemory()
    examples: list[ActionExample] = []
    evaluation: tuple[float, ...]
    for row in rows[1:]:
        if "obs" not in row or int(row.get("p", -1)) != target_seat:
            continue
        observation = row["obs"]
        select = observation.get("select")
        if not isinstance(select, Mapping):
            continue
        contract = SelectionContract.from_observation(observation)
        try:
            action = list(contract.validate(row.get("action", ())))
        except Exception:
            continue
        memory.infer_opponent_history(observation)
        prior._update_memory(observation)
        options = list(select.get("option", ()))
        baseline_scores = tuple(prior._score_option(observation, option, index) for index, option in enumerate(options))
        planner_row = planner_feature_items(observation, memory)
        evaluation = _planner_predict(
            planner_model,
            planner_normalizer,
            calibration,
            planner_row,
            memory.plan_latent,
        )
        offset = 0
        archetype = evaluation[offset : offset + len(ARCHETYPE_NAMES)]; offset += len(ARCHETYPE_NAMES)
        intent = evaluation[offset : offset + len(INTENT_NAMES)]; offset += len(INTENT_NAMES)
        resources = evaluation[offset : offset + len(RESOURCE_NAMES)]; offset += len(RESOURCE_NAMES)
        plan = evaluation[offset : offset + len(PLAN_NAMES)]; offset += len(PLAN_NAMES)
        strategic = evaluation[offset : offset + len(STRATEGIC_AXES)]; offset += len(STRATEGIC_AXES)
        meta_weights = evaluation[offset : offset + len(STRATEGIC_AXES)]
        memory.archetype_posterior = list(archetype)
        memory.intent_posterior = list(intent)
        memory.resource_posterior = list(resources)
        turn = _safe_int(_state(observation).get("turn"), 0)
        is_main = _safe_int(select.get("context"), -1) == int(SelectContext.MAIN)
        if is_main and turn != memory.last_plan_turn:
            persistence = float(calibration["plan_persistence"])
            memory.plan_latent = [
                persistence * old + (1.0 - persistence) * new
                for old, new in zip(memory.plan_latent, plan, strict=True)
            ]
            total = sum(memory.plan_latent)
            memory.plan_latent = [value / total for value in memory.plan_latent]
            memory.last_plan_turn = turn
        tokens = tuple(token_feature_rows(observation))
        prefix: list[int] = []
        remaining = list(range(contract.option_count))
        for selected in action:
            include_stop = len(prefix) >= contract.min_count
            indices, action_rows = action_feature_rows(
                observation,
                baseline_scores,
                plan,
                meta_weights,
                prefix=prefix,
                remaining=remaining,
                include_stop=include_stop,
            )
            if selected not in indices:
                break
            examples.append(
                ActionExample(
                    split=split,
                    source=source,
                    match_id=match_id,
                    planner_row=planner_row,
                    evaluation=evaluation,
                    token_rows=tokens,
                    action_rows=tuple(action_rows),
                    prefix_row=prefix_feature_items(observation, prefix),
                    target=indices.index(selected),
                    context=_safe_int(select.get("context"), -1),
                )
            )
            prefix.append(selected)
            remaining.remove(selected)
        if len(prefix) == len(action) and len(prefix) >= contract.min_count and len(prefix) < contract.max_count:
            indices, action_rows = action_feature_rows(
                observation,
                baseline_scores,
                plan,
                meta_weights,
                prefix=prefix,
                remaining=remaining,
                include_stop=True,
            )
            examples.append(
                ActionExample(
                    split=split,
                    source=source,
                    match_id=match_id,
                    planner_row=planner_row,
                    evaluation=evaluation,
                    token_rows=tokens,
                    action_rows=tuple(action_rows),
                    prefix_row=prefix_feature_items(observation, prefix),
                    target=indices.index(None),
                    context=_safe_int(select.get("context"), -1),
                )
            )
        memory.observe_selection(observation, action)
    return examples


def load_action_examples(
    *,
    generated_directory: Path,
    archived_directory: Path,
    planner_model: PlannerNetwork,
    planner_normalizer: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[list[ActionExample], dict[str, Any]]:
    examples: list[ActionExample] = []
    files_by_split: Counter[str] = Counter()
    for source, directory, target_name in (
        ("compatibility_teacher", generated_directory, None),
        ("archived_prov3", archived_directory, "HelbrosRocketSpidopsProV3"),
    ):
        for path in sorted(directory.glob("*.jsonl")):
            rows = _parse_replay_file(
                path,
                source=source,
                target_name=target_name,
                planner_model=planner_model,
                planner_normalizer=planner_normalizer,
                calibration=calibration,
            )
            if rows:
                files_by_split[f"{source}:{rows[0].split}"] += 1
                examples.extend(rows)
    if not examples:
        raise RuntimeError("no tactical action examples were loaded")
    provenance = {
        "examples": len(examples),
        "examples_by_split": dict(Counter(example.split for example in examples)),
        "examples_by_source": dict(Counter(example.source for example in examples)),
        "files_by_source_split": dict(files_by_split),
        "nontrivial_examples": sum(len(example.action_rows) > 1 for example in examples),
    }
    return examples, provenance


class ActionNetwork(nn.Module):
    def __init__(self, planner_size: int, token_size: int, action_size: int) -> None:
        super().__init__()
        evaluation_size = (
            len(ARCHETYPE_NAMES)
            + len(INTENT_NAMES)
            + len(RESOURCE_NAMES)
            + len(PLAN_NAMES)
            + len(STRATEGIC_AXES)
            + len(STRATEGIC_AXES)
            + 2
        )
        self.action_state_hidden = nn.Linear(planner_size + evaluation_size, ACTION_STATE_HIDDEN)
        self.token_hidden = nn.Linear(token_size, TOKEN_HIDDEN)
        self.token_query = nn.Linear(TOKEN_HIDDEN, ATTENTION_HIDDEN)
        self.token_key = nn.Linear(TOKEN_HIDDEN, ATTENTION_HIDDEN)
        self.token_value = nn.Linear(TOKEN_HIDDEN, ATTENTION_HIDDEN)
        self.token_pool_query = nn.Linear(ACTION_STATE_HIDDEN, ATTENTION_HIDDEN)
        self.token_output = nn.Linear(ATTENTION_HIDDEN * 2, TOKEN_HIDDEN)
        self.action_query = nn.Linear(ACTION_STATE_HIDDEN + TOKEN_HIDDEN + len(PREFIX_FEATURE_NAMES), ACTION_HIDDEN)
        self.exploit_query = nn.Linear(ACTION_STATE_HIDDEN + len(INTENT_NAMES), ACTION_HIDDEN)
        self.action_hidden = nn.Linear(action_size, ACTION_HIDDEN)
        self.action_bias = nn.Linear(ACTION_HIDDEN, 1)
        self.raw_tactical_prior_weight = nn.Parameter(torch.tensor(0.55))
        self.raw_plan_alignment_weight = nn.Parameter(torch.tensor(-0.20))
        self.raw_axis_alignment_weight = nn.Parameter(torch.tensor(-0.20))

    def forward(
        self,
        state: Tensor,
        intent: Tensor,
        tokens: Tensor,
        token_mask: Tensor,
        actions: Tensor,
        action_mask: Tensor,
        prefix: Tensor,
        *,
        exploitation_weight: float = 0.10,
    ) -> Tensor:
        state_hidden = torch.tanh(self.action_state_hidden(state))
        token_hidden = torch.tanh(self.token_hidden(tokens))
        queries = self.token_query(token_hidden)
        keys = self.token_key(token_hidden)
        values = self.token_value(token_hidden)
        pair_scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(ATTENTION_HIDDEN)
        pair_scores = pair_scores.masked_fill(~token_mask[:, None, :], -1e9)
        pair_attention = torch.softmax(pair_scores, dim=-1)
        contexts = torch.matmul(pair_attention, values) * token_mask[:, :, None]
        pooled = contexts.sum(1) / token_mask.sum(1, keepdim=True).clamp_min(1)
        state_query = self.token_pool_query(state_hidden)
        pool_scores = (keys * state_query[:, None, :]).sum(-1) / math.sqrt(ATTENTION_HIDDEN)
        pool_scores = pool_scores.masked_fill(~token_mask, -1e9)
        pool_attention = torch.softmax(pool_scores, dim=-1)
        attended = (pool_attention[:, :, None] * values).sum(1)
        token_pool = torch.tanh(self.token_output(torch.cat((pooled, attended), -1)))
        query = torch.tanh(self.action_query(torch.cat((state_hidden, token_pool, prefix), -1)))
        exploit = torch.tanh(self.exploit_query(torch.cat((state_hidden, intent), -1)))
        action_hidden = torch.tanh(self.action_hidden(actions))
        logits = (action_hidden * query[:, None, :]).sum(-1) / math.sqrt(ACTION_HIDDEN)
        logits = logits + self.action_bias(action_hidden).squeeze(-1)
        logits = logits + exploitation_weight * (action_hidden * exploit[:, None, :]).sum(-1) / math.sqrt(ACTION_HIDDEN)
        # The tactical prior uses the normalized baseline-score column in the
        # fixed ACTION_FEATURE_NAMES contract.
        baseline_index = ACTION_FEATURE_NAMES.index("baseline_score")
        logits = logits + F.softplus(self.raw_tactical_prior_weight) * actions[:, :, baseline_index]
        plan_indices = [ACTION_FEATURE_NAMES.index(f"plan_alignment_{name}") for name in PLAN_NAMES]
        axis_indices = [ACTION_FEATURE_NAMES.index(f"axis_alignment_{name}") for name in STRATEGIC_AXES]
        logits = logits + F.softplus(self.raw_plan_alignment_weight) * actions[:, :, plan_indices].sum(-1)
        logits = logits + F.softplus(self.raw_axis_alignment_weight) * actions[:, :, axis_indices].sum(-1)
        return logits.masked_fill(~action_mask, -1e9)


def _action_batch(
    examples: Sequence[ActionExample],
    planner_normalizer: Mapping[str, Any],
    token_normalizer: Mapping[str, Any],
    action_normalizer: Mapping[str, Any],
    *,
    ablation: str = "full",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    batch = len(examples)
    max_tokens = max(len(example.token_rows) for example in examples)
    max_actions = max(len(example.action_rows) for example in examples)
    states: list[list[float]] = []
    intents: list[list[float]] = []
    tokens = torch.zeros((batch, max_tokens, len(TOKEN_FEATURE_NAMES)), dtype=torch.float32)
    token_mask = torch.zeros((batch, max_tokens), dtype=torch.bool)
    actions = torch.zeros((batch, max_actions, len(ACTION_FEATURE_NAMES)), dtype=torch.float32)
    action_mask = torch.zeros((batch, max_actions), dtype=torch.bool)
    prefixes: list[list[float]] = []
    targets: list[int] = []
    intent_start = len(ARCHETYPE_NAMES)
    intent_end = intent_start + len(INTENT_NAMES)
    for row_index, example in enumerate(examples):
        states.append(_normalize_row(example.planner_row, planner_normalizer) + list(example.evaluation))
        intents.append(list(example.evaluation[intent_start:intent_end]))
        for index, row in enumerate(example.token_rows):
            tokens[row_index, index] = torch.tensor(_normalize_row(row, token_normalizer))
            token_mask[row_index, index] = True
        for index, source_row in enumerate(example.action_rows):
            row = dict(source_row)
            option_type = next(
                (
                    option_index
                    for option_index in range(16)
                    if float(row.get(f"option_type_{option_index:02d}", 0.0)) > 0.5
                ),
                -1,
            )
            if ablation in {"no_plan", "no_beliefs"}:
                for name in PLAN_NAMES:
                    row[f"plan_alignment_{name}"] = (
                        _PLAN_TYPE_BIASES[name].get(option_type, 0.0) / len(PLAN_NAMES)
                    )
            if ablation == "no_meta":
                for name in STRATEGIC_AXES:
                    row[f"axis_alignment_{name}"] = (
                        _AXIS_TYPE_BIASES[name].get(option_type, 0.0) / len(STRATEGIC_AXES)
                    )
            actions[row_index, index] = torch.tensor(_normalize_row(row, action_normalizer))
            action_mask[row_index, index] = True
        prefixes.append([float(example.prefix_row.get(name, 0.0)) for name in PREFIX_FEATURE_NAMES])
        targets.append(example.target)
    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(intents, dtype=torch.float32),
        tokens,
        token_mask,
        actions,
        action_mask,
        torch.tensor(prefixes, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )


def _action_metrics(
    model: ActionNetwork,
    examples: Sequence[ActionExample],
    planner_normalizer: Mapping[str, Any],
    token_normalizer: Mapping[str, Any],
    action_normalizer: Mapping[str, Any],
    *,
    ablation: str = "full",
) -> dict[str, float | int]:
    total_loss = 0.0
    total = 0
    correct = 0
    nontrivial = 0
    nontrivial_correct = 0
    main_total = 0
    main_correct = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            rows = examples[start : start + 128]
            batch = _action_batch(
                rows,
                planner_normalizer,
                token_normalizer,
                action_normalizer,
                ablation=ablation,
            )
            state, intent, tokens, token_mask, actions, action_mask, prefix = batch[:-1]
            state = state.clone()
            tokens = tokens.clone()
            evaluation_start = len(PLANNER_FEATURE_NAMES)
            archetype_start = evaluation_start
            intent_start = archetype_start + len(ARCHETYPE_NAMES)
            resource_start = intent_start + len(INTENT_NAMES)
            plan_start = resource_start + len(RESOURCE_NAMES)
            strategic_start = plan_start + len(PLAN_NAMES)
            meta_start = strategic_start + len(STRATEGIC_AXES)
            aggregate_start = meta_start + len(STRATEGIC_AXES)
            if ablation == "no_beliefs":
                state[:, archetype_start:intent_start] = 1.0 / len(ARCHETYPE_NAMES)
                state[:, intent_start:resource_start] = 1.0 / len(INTENT_NAMES)
                state[:, resource_start:plan_start] = 0.5
                intent = torch.full_like(intent, 1.0 / len(INTENT_NAMES))
            elif ablation == "no_plan":
                state[:, plan_start:strategic_start] = 1.0 / len(PLAN_NAMES)
            elif ablation == "no_relational":
                tokens.zero_()
            elif ablation == "no_meta":
                state[:, meta_start:aggregate_start] = 1.0 / len(STRATEGIC_AXES)
                state[:, aggregate_start] = state[:, strategic_start:meta_start].mean(-1)
            elif ablation != "full":
                raise ValueError(f"unknown action ablation: {ablation}")
            logits = model(
                state,
                intent,
                tokens,
                token_mask,
                actions,
                action_mask,
                prefix,
                exploitation_weight=0.0,
            )
            target = batch[-1]
            losses = F.cross_entropy(logits, target, reduction="none")
            predictions = logits.argmax(-1)
            total_loss += float(losses.sum())
            total += len(rows)
            correct += int((predictions == target).sum())
            for index, example in enumerate(rows):
                hit = int(predictions[index]) == int(target[index])
                if len(example.action_rows) > 1:
                    nontrivial += 1
                    nontrivial_correct += int(hit)
                if example.context == int(SelectContext.MAIN):
                    main_total += 1
                    main_correct += int(hit)
    return {
        "count": total,
        "negative_log_likelihood": total_loss / max(total, 1),
        "top1_accuracy": correct / max(total, 1),
        "nontrivial_count": nontrivial,
        "nontrivial_top1_accuracy": nontrivial_correct / max(nontrivial, 1),
        "main_count": main_total,
        "main_top1_accuracy": main_correct / max(main_total, 1),
    }


def fit_action_policy(
    examples: Sequence[ActionExample],
    planner_normalizer: Mapping[str, Any],
) -> tuple[ActionNetwork, dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.manual_seed(TRAINING_SEED + 1)
    train = [example for example in examples if example.split == "train"]
    validation = [example for example in examples if example.split == "validation"]
    test = [example for example in examples if example.split == "test"]
    token_normalizer = _normalizer(
        [row for example in train for row in example.token_rows], TOKEN_FEATURE_NAMES
    )
    action_normalizer = _normalizer(
        [row for example in train for row in example.action_rows], ACTION_FEATURE_NAMES
    )
    model = ActionNetwork(len(PLANNER_FEATURE_NAMES), len(TOKEN_FEATURE_NAMES), len(ACTION_FEATURE_NAMES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=2e-4)
    generator = random.Random(TRAINING_SEED + 2)
    best_state: dict[str, Tensor] | None = None
    best_nll = math.inf
    best_epoch = -1
    stale = 0
    for epoch in range(45):
        model.train()
        shuffled = list(train)
        generator.shuffle(shuffled)
        for start in range(0, len(shuffled), 64):
            rows = shuffled[start : start + 64]
            batch = _action_batch(rows, planner_normalizer, token_normalizer, action_normalizer)
            optimizer.zero_grad()
            logits = model(*batch[:-1], exploitation_weight=0.10)
            loss = F.cross_entropy(logits, batch[-1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        metrics = _action_metrics(
            model, validation, planner_normalizer, token_normalizer, action_normalizer
        )
        nll = float(metrics["negative_log_likelihood"])
        if nll < best_nll - 1e-4:
            best_nll = nll
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    if best_state is None:
        raise RuntimeError("action training did not produce a checkpoint")
    model.load_state_dict(best_state)
    metrics = {
        "best_epoch": best_epoch,
        "tactical_prior_weight": float(F.softplus(model.raw_tactical_prior_weight)),
        "train": _action_metrics(model, train, planner_normalizer, token_normalizer, action_normalizer),
        "validation": _action_metrics(model, validation, planner_normalizer, token_normalizer, action_normalizer),
        "test": _action_metrics(model, test, planner_normalizer, token_normalizer, action_normalizer),
        "test_ablations": {
            ablation: _action_metrics(
                model,
                test,
                planner_normalizer,
                token_normalizer,
                action_normalizer,
                ablation=ablation,
            )
            for ablation in ("no_beliefs", "no_plan", "no_relational", "no_meta")
        },
    }
    return model, token_normalizer, action_normalizer, metrics


def _linear(layer: nn.Linear) -> dict[str, Any]:
    return {
        "weight": layer.weight.detach().cpu().tolist(),
        "bias": layer.bias.detach().cpu().tolist(),
    }


def export_integrated_artifact(
    *,
    planner: PlannerNetwork,
    action: ActionNetwork,
    planner_normalizer: Mapping[str, Any],
    token_normalizer: Mapping[str, Any],
    action_normalizer: Mapping[str, Any],
    calibration: Mapping[str, Any],
    evidence: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    layers = {
        name: _linear(getattr(planner, name))
        for name in (
            "planner_hidden_1",
            "planner_hidden_2",
            "archetype_head",
            "intent_head",
            "resource_head",
            "plan_policy_head",
            "belief_plan_bridge",
            "plan_value_head",
            "strategic_head",
            "meta_head",
            "uncertainty_head",
            "search_gate_head",
        )
    }
    layers.update(
        {
            name: _linear(getattr(action, name))
            for name in (
                "action_state_hidden",
                "token_hidden",
                "token_query",
                "token_key",
                "token_value",
                "token_pool_query",
                "token_output",
                "action_query",
                "exploit_query",
                "action_hidden",
                "action_bias",
            )
        }
    )
    artifact: dict[str, Any] = {
        "schema_version": INTEGRATED_PLANNER_SCHEMA_VERSION,
        "model_family": "public_belief_average_strategy_with_autoregressive_tactics",
        "plan_names": list(PLAN_NAMES),
        "strategic_axes": list(STRATEGIC_AXES),
        "intent_names": list(INTENT_NAMES),
        "resource_names": list(RESOURCE_NAMES),
        "archetype_names": list(ARCHETYPE_NAMES),
        "normalizers": {
            "planner": dict(planner_normalizer),
            "token": dict(token_normalizer),
            "action": dict(action_normalizer),
        },
        "layers": layers,
        "calibration": {
            **dict(calibration),
            "tactical_prior_weight": float(F.softplus(action.raw_tactical_prior_weight)),
            "plan_alignment_weight": float(F.softplus(action.raw_plan_alignment_weight)),
            "axis_alignment_weight": float(F.softplus(action.raw_axis_alignment_weight)),
        },
        "evidence": dict(evidence),
        "deployment_claims": {
            "official_engine_execution_in_training": True,
            "official_engine_inside_model": False,
            "neural_blueprint_is_final_strategy_provider": True,
            "heuristic_is_tactical_prior_only": True,
            "sampled_rollout_is_final_controller": False,
            "hidden_allocations_are_research_abstractions": True,
            "equilibrium_claimed": False,
        },
    }
    artifact["fingerprint"] = _fingerprint(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    # Loader verification catches both serialization and tensor-shape drift.
    FrozenStrategicPlanner(json.loads(output_path.read_text(encoding="utf-8")))
    return artifact


def train_integrated_controller(
    *,
    project_root: Path,
    archived_replays: Path,
    output_path: Path,
) -> dict[str, Any]:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    runtime = project_root / "work" / "official_kaggle" / "sample_submission"
    surfaces = project_root / "artifacts" / "phase6_expected_value_surfaces" / "expected_value_surface_dataset.json"
    trajectories = project_root / "artifacts" / "phase6_official_trajectory" / "official_trajectory_dataset.json"
    surface_examples, surface_provenance = load_surface_examples(surfaces, runtime, trajectories)
    planner, planner_normalizer, planner_metrics = fit_planner(surface_examples)
    calibration, bounds_report = calibrate_planner(planner, surface_examples, planner_normalizer)
    action_examples, action_provenance = load_action_examples(
        generated_directory=project_root / "work" / "integrated_teacher_replays",
        archived_directory=archived_replays,
        planner_model=planner,
        planner_normalizer=planner_normalizer,
        calibration=calibration,
    )
    action, token_normalizer, action_normalizer, action_metrics = fit_action_policy(
        action_examples, planner_normalizer
    )
    evidence = {
        "training_seed": TRAINING_SEED,
        "surface_provenance": surface_provenance,
        "planner_metrics": planner_metrics,
        "counterfactual_bounds": bounds_report,
        "action_provenance": action_provenance,
        "action_metrics": action_metrics,
        "protocol_document": "PHASE6AN_INTEGRATED_CONTROLLER_PROTOCOL.md",
    }
    artifact = export_integrated_artifact(
        planner=planner,
        action=action,
        planner_normalizer=planner_normalizer,
        token_normalizer=token_normalizer,
        action_normalizer=action_normalizer,
        calibration=calibration,
        evidence=evidence,
        output_path=output_path,
    )
    report = {
        "schema_version": 1,
        "artifact_path": str(output_path.resolve()),
        "artifact_fingerprint": artifact["fingerprint"],
        "artifact_sha256": sha256(output_path.read_bytes()).hexdigest(),
        "deck_resolution": resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        ).to_dict(),
        **evidence,
    }
    report["fingerprint"] = _fingerprint(report)
    report_path = output_path.with_name("integrated_training_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


__all__ = [
    "ActionExample",
    "ActionNetwork",
    "PlannerNetwork",
    "SurfaceExample",
    "calibrate_planner",
    "export_integrated_artifact",
    "fit_action_policy",
    "fit_planner",
    "load_action_examples",
    "load_surface_examples",
    "train_integrated_controller",
]
