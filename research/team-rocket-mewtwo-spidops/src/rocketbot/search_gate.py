"""Learned, calibrated selective-search gate for pivotal decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import statistics
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .beliefs import OPPONENT_INTENTS, RESOURCE_CATEGORIES, OpponentBeliefState, OpponentIntent
from .model import BeliefAwareEvaluation
from .observation import AgentObservation
from .plans import StrategicPlanProposer
from .policy import ActionPrefix, ActionToken
from .search import BeliefAwarePlanSearch, PlanSearchResult
from .strategy import STRATEGIC_AXES, StrategicAxis


SEARCH_GATE_FEATURE_NAMES: tuple[str, ...] = (
    "policy_uncertainty",
    "policy_entropy",
    "legal_action_pressure",
    "head_disagreement",
    "head_range",
    "plan_belief_entropy",
    "resource_belief_entropy",
    "hypothesis_entropy",
    "own_knockout_risk",
    "opponent_knockout_opportunity",
    "own_prize_closeness",
    "opponent_prize_closeness",
    "evolution_window",
    "disruption_window",
    "board_collapse_risk",
    "proposal_ambiguity",
    "proposal_entropy",
    "objective_diversity",
    "optionality_spread",
    "meta_value_neutrality",
    "turn_progress",
    "knockout_swing",
    "prize_swing",
    "budget_fraction",
)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


@dataclass(frozen=True, slots=True)
class PolicySignals:
    confidence: float
    entropy: float
    legal_action_count: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("policy confidence must lie in [0, 1]")
        if not math.isfinite(self.entropy) or self.entropy < 0.0:
            raise ValueError("policy entropy must be finite and non-negative")
        if self.legal_action_count <= 0:
            raise ValueError("legal_action_count must be positive")


@dataclass(frozen=True, slots=True)
class SearchComputeBank:
    capacity: float
    available: float
    refill_per_turn: float = 0.0
    spent_total: float = 0.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        values = (self.capacity, self.available, self.refill_per_turn, self.spent_total)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("compute-bank values must be finite and non-negative")
        if self.capacity <= 0.0 or self.available > self.capacity:
            raise ValueError("compute-bank capacity/availability are inconsistent")

    @property
    def fraction_available(self) -> float:
        return self.available / self.capacity

    def can_afford(self, cost: float) -> bool:
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("search cost must be finite and non-negative")
        return self.available + 1e-9 >= cost

    def spend(self, cost: float) -> "SearchComputeBank":
        if not self.can_afford(cost):
            raise ValueError("compute bank cannot afford the requested search")
        return SearchComputeBank(
            capacity=self.capacity,
            available=max(0.0, self.available - cost),
            refill_per_turn=self.refill_per_turn,
            spent_total=self.spent_total + cost,
        )

    def refill(self, turns: int = 1) -> "SearchComputeBank":
        if turns < 0:
            raise ValueError("turns cannot be negative")
        return SearchComputeBank(
            capacity=self.capacity,
            available=min(self.capacity, self.available + turns * self.refill_per_turn),
            refill_per_turn=self.refill_per_turn,
            spent_total=self.spent_total,
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "schema_version": self.schema_version,
            "capacity": self.capacity,
            "available": self.available,
            "refill_per_turn": self.refill_per_turn,
            "spent_total": self.spent_total,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchComputeBank":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            capacity=float(raw["capacity"]),
            available=float(raw["available"]),
            refill_per_turn=float(raw.get("refill_per_turn", 0.0)),
            spent_total=float(raw.get("spent_total", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class SearchGateFeatures:
    values: tuple[float, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if len(self.values) != len(SEARCH_GATE_FEATURE_NAMES):
            raise ValueError(
                f"search-gate features must contain {len(SEARCH_GATE_FEATURE_NAMES)} values"
            )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.values):
            raise ValueError("search-gate features must be finite and normalized to [0, 1]")

    def value(self, name: str) -> float:
        return self.values[SEARCH_GATE_FEATURE_NAMES.index(name)]

    def to_tensor(
        self,
        *,
        batch_dimension: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        tensor = torch.tensor(self.values, dtype=dtype, device=device)
        return tensor.unsqueeze(0) if batch_dimension else tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "values": {
                name: self.values[index]
                for index, name in enumerate(SEARCH_GATE_FEATURE_NAMES)
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchGateFeatures":
        values = raw["values"]
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            values=tuple(float(values[name]) for name in SEARCH_GATE_FEATURE_NAMES),
        )


class SearchGateFeatureExtractor:
    """Compute cheap search-value signals without running plan rollouts."""

    def __init__(self, proposer: StrategicPlanProposer, *, max_plans: int = 4) -> None:
        if max_plans <= 1:
            raise ValueError("at least two cheap plan proposals are required")
        self.proposer = proposer
        self.max_plans = max_plans

    def extract(
        self,
        observation: AgentObservation,
        evaluation: BeliefAwareEvaluation,
        belief: OpponentBeliefState,
        policy: PolicySignals,
        bank: SearchComputeBank,
        *,
        batch_index: int = 0,
    ) -> SearchGateFeatures:
        plans = tuple(
            self.proposer.propose(
                observation,
                evaluation,
                belief,
                top_k=self.max_plans,
                batch_index=batch_index,
            )
        )
        if len(plans) < 2:
            raise ValueError("search-gate features require at least two proposed plans")
        prior_total = sum(plan.prior for plan in plans)
        if prior_total <= 0.0:
            raise ValueError("search-gate plan priors must have positive mass")
        priors = sorted((plan.prior / prior_total for plan in plans), reverse=True)
        relative_margin = (
            (priors[0] - priors[1]) / max(priors[0], 1e-8)
            if len(priors) > 1
            else 1.0
        )
        proposal_ambiguity = 1.0 - _clamp01(relative_margin)
        proposal_entropy = -sum(prior * math.log(prior) for prior in priors if prior > 0.0)
        proposal_entropy /= math.log(max(len(priors), 2))
        objective_diversity = len({plan.objective for plan in plans}) / min(
            len(plans), len(STRATEGIC_AXES)
        )
        optionality_index = STRATEGIC_AXES.index(StrategicAxis.OPTIONALITY)
        optionality_values = [plan.axis_alignment[optionality_index] for plan in plans]
        optionality_spread = _clamp01(
            (max(optionality_values) - min(optionality_values)) / 2.0
        )

        head_values = [
            float(evaluation.head_values[axis][batch_index, 0].detach().cpu())
            for axis in STRATEGIC_AXES
        ]
        head_disagreement = _clamp01(statistics.pstdev(head_values))
        head_range = _clamp01((max(head_values) - min(head_values)) / 2.0)
        meta_value = float(evaluation.meta_value[batch_index, 0].detach().cpu())

        own = observation.public_state.player(observation.observer)
        opponent = observation.public_state.player(observation.observer.other)
        own_ko_risk = (
            _clamp01(own.active.damage / own.active.max_hp) if own.active else 1.0
        )
        opponent_ko_opportunity = (
            _clamp01(opponent.active.damage / opponent.active.max_hp)
            if opponent.active
            else 0.0
        )
        own_prize_closeness = _clamp01((3 - own.prizes_remaining) / 2.0)
        opponent_prize_closeness = _clamp01((3 - opponent.prizes_remaining) / 2.0)

        tarountula_in_play = any(
            pokemon.top_card.card_name == "Team Rocket's Tarountula"
            for pokemon in own.pokemon_in_play()
        )
        spidops_in_hand = any(
            card.card_name == "Team Rocket's Spidops" for card in own.hand
        )
        evolution_window = (
            1.0 if tarountula_in_play and spidops_in_hand
            else 0.55 if tarountula_in_play
            else 0.0
        )
        vulnerable_plan_mass = (
            belief.plan.probability(OpponentIntent.SETUP)
            + belief.plan.probability(OpponentIntent.RESOURCE_CONSERVATION)
            + belief.plan.probability(OpponentIntent.COMBO_ASSEMBLY)
        )
        disruption_window = _clamp01(
            min(len(opponent.hand) / 8.0, 1.0) * vulnerable_plan_mass
        )
        board_collapse_risk = _clamp01(
            (1.0 - len(own.bench) / 5.0) * (0.5 + 0.5 * own_ko_risk)
        )

        plan_entropy = belief.plan.entropy / math.log(len(OPPONENT_INTENTS))
        resource_entropy = belief.resource_entropy / (
            len(RESOURCE_CATEGORIES) * math.log(2.0)
        )
        if belief.hidden_hypotheses is None or len(belief.hidden_hypotheses.hypotheses) <= 1:
            hypothesis_entropy = 0.0
        else:
            hypothesis_entropy = belief.hidden_hypotheses.entropy / math.log(
                len(belief.hidden_hypotheses.hypotheses)
            )
        policy_entropy = policy.entropy / math.log(max(policy.legal_action_count, 2))
        legal_action_pressure = math.log1p(policy.legal_action_count - 1) / math.log(20.0)
        values = (
            1.0 - policy.confidence,
            _clamp01(policy_entropy),
            _clamp01(legal_action_pressure),
            head_disagreement,
            head_range,
            _clamp01(plan_entropy),
            _clamp01(resource_entropy),
            _clamp01(hypothesis_entropy),
            own_ko_risk,
            opponent_ko_opportunity,
            own_prize_closeness,
            opponent_prize_closeness,
            evolution_window,
            disruption_window,
            board_collapse_risk,
            proposal_ambiguity,
            _clamp01(proposal_entropy),
            _clamp01(objective_diversity),
            optionality_spread,
            _clamp01(1.0 - abs(meta_value)),
            _clamp01(observation.public_state.turn_number / 30.0),
            max(own_ko_risk, opponent_ko_opportunity),
            max(own_prize_closeness, opponent_prize_closeness),
            _clamp01(bank.fraction_available),
        )
        return SearchGateFeatures(tuple(float(value) for value in values))


@dataclass(slots=True)
class SearchGateModelOutput:
    raw_logit: torch.Tensor
    predicted_gain: torch.Tensor


class SearchGateNetwork(nn.Module):
    """Predict whether search pays for itself and its expected value gain."""

    def __init__(self, feature_dim: int = len(SEARCH_GATE_FEATURE_NAMES), hidden_dim: int = 48) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("search-gate dimensions must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.search_logit = nn.Linear(hidden_dim, 1)
        self.gain_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> SearchGateModelOutput:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"search-gate features must have shape [B, {self.feature_dim}]")
        embedding = self.trunk(features)
        return SearchGateModelOutput(
            raw_logit=self.search_logit(embedding).squeeze(-1),
            predicted_gain=self.gain_head(embedding).squeeze(-1),
        )


@dataclass(frozen=True, slots=True)
class SearchGateCalibrator:
    temperature: float = 1.0
    threshold: float = 0.5
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("gate temperature must be finite and positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("gate threshold must lie in [0, 1]")

    @classmethod
    def fit(
        cls,
        raw_logits: torch.Tensor,
        realized_gain: torch.Tensor,
        search_cost: torch.Tensor,
        *,
        candidate_temperatures: Sequence[float] | None = None,
        candidate_thresholds: Sequence[float] | None = None,
    ) -> "SearchGateCalibrator":
        if raw_logits.ndim != 1:
            raise ValueError("gate logits must have shape [B]")
        if raw_logits.numel() == 0:
            raise ValueError("at least one calibration example is required")
        if realized_gain.shape != raw_logits.shape or search_cost.shape != raw_logits.shape:
            raise ValueError("gain and cost must match gate logits")
        labels = realized_gain.gt(search_cost).float()
        temperatures = tuple(candidate_temperatures or (
            0.25, 0.35, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.75, 4.0
        ))
        if 1.0 not in temperatures:
            temperatures = temperatures + (1.0,)
        thresholds = tuple(
            candidate_thresholds
            or tuple(index / 20.0 for index in range(21))
        )
        if 0.0 not in thresholds:
            thresholds = thresholds + (0.0,)
        if 1.0 not in thresholds:
            thresholds = thresholds + (1.0,)
        if any(value <= 0.0 or not math.isfinite(value) for value in temperatures):
            raise ValueError("candidate temperatures must be finite and positive")
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("candidate thresholds must lie in [0, 1]")

        with torch.no_grad():
            losses = [
                float(F.binary_cross_entropy_with_logits(raw_logits / temperature, labels))
                for temperature in temperatures
            ]
            temperature = temperatures[min(range(len(temperatures)), key=losses.__getitem__)]
            probabilities = torch.sigmoid(raw_logits / temperature)
            utilities = []
            net_gain = realized_gain - search_cost
            for threshold in thresholds:
                invoked = probabilities.gt(threshold)
                utility = float(torch.where(invoked, net_gain, torch.zeros_like(net_gain)).mean())
                utilities.append(utility)
            best_index = max(
                range(len(thresholds)),
                key=lambda index: (utilities[index], thresholds[index]),
            )
        return cls(temperature=float(temperature), threshold=float(thresholds[best_index]))

    def probabilities(self, raw_logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw_logits / self.temperature)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "schema_version": self.schema_version,
            "temperature": self.temperature,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchGateCalibrator":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            temperature=float(raw["temperature"]),
            threshold=float(raw["threshold"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, encoded: str) -> "SearchGateCalibrator":
        return cls.from_dict(json.loads(encoded))


@dataclass(frozen=True, slots=True)
class SearchGateSupervision:
    features: SearchGateFeatures
    realized_search_gain: float
    search_cost: float
    source: str = "policy_search_comparison"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.realized_search_gain, self.search_cost)):
            raise ValueError("gate supervision gain/cost must be finite")
        if self.search_cost < 0.0 or not self.source.strip():
            raise ValueError("gate supervision cost/source are invalid")

    @property
    def should_search(self) -> bool:
        return self.realized_search_gain > self.search_cost

    @classmethod
    def from_outcome(
        cls,
        features: SearchGateFeatures,
        *,
        routine_value: float,
        search_result: PlanSearchResult,
        search_cost: float,
    ) -> "SearchGateSupervision":
        return cls(
            features=features,
            realized_search_gain=search_result.chosen.expected_value - routine_value,
            search_cost=search_cost,
            source="matched_policy_vs_plan_search",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "features": self.features.to_dict(),
            "realized_search_gain": self.realized_search_gain,
            "search_cost": self.search_cost,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchGateSupervision":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            features=SearchGateFeatures.from_dict(raw["features"]),
            realized_search_gain=float(raw["realized_search_gain"]),
            search_cost=float(raw["search_cost"]),
            source=str(raw.get("source", "policy_search_comparison")),
        )


@dataclass(frozen=True, slots=True)
class SearchGateDataset:
    examples: tuple[SearchGateSupervision, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("search-gate dataset cannot be empty")

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "examples": [example.to_dict() for example in self.examples],
            },
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, encoded: str) -> "SearchGateDataset":
        raw = json.loads(encoded)
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            examples=tuple(
                SearchGateSupervision.from_dict(example) for example in raw["examples"]
            ),
        )


@dataclass(slots=True)
class SearchGateBatch:
    features: torch.Tensor
    labels: torch.Tensor
    realized_gain: torch.Tensor
    search_cost: torch.Tensor

    @classmethod
    def from_examples(
        cls,
        examples: Sequence[SearchGateSupervision],
        *,
        device: torch.device | str | None = None,
    ) -> "SearchGateBatch":
        if not examples:
            raise ValueError("at least one gate example is required")
        return cls(
            features=torch.tensor(
                [example.features.values for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            labels=torch.tensor(
                [example.should_search for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            realized_gain=torch.tensor(
                [example.realized_search_gain for example in examples],
                dtype=torch.float32,
                device=device,
            ),
            search_cost=torch.tensor(
                [example.search_cost for example in examples],
                dtype=torch.float32,
                device=device,
            ),
        )


@dataclass(slots=True)
class SearchGateLoss:
    total: torch.Tensor
    classification: torch.Tensor
    gain_regression: torch.Tensor


def search_gate_loss(
    output: SearchGateModelOutput,
    batch: SearchGateBatch,
    *,
    gain_weight: float = 0.5,
) -> SearchGateLoss:
    if output.raw_logit.shape != batch.labels.shape:
        raise ValueError("gate output and labels have incompatible shapes")
    if output.predicted_gain.shape != batch.realized_gain.shape:
        raise ValueError("gate gain output and targets have incompatible shapes")
    if gain_weight < 0.0:
        raise ValueError("gain_weight cannot be negative")
    classification = F.binary_cross_entropy_with_logits(output.raw_logit, batch.labels)
    gain_regression = F.smooth_l1_loss(output.predicted_gain, batch.realized_gain)
    return SearchGateLoss(
        total=classification + gain_weight * gain_regression,
        classification=classification,
        gain_regression=gain_regression,
    )


@dataclass(frozen=True, slots=True)
class SearchGateMetrics:
    accuracy: float
    precision: float
    recall: float
    brier: float
    ece: float
    invocation_rate: float
    positive_invocation_rate: float
    negative_invocation_rate: float
    realized_net_utility: float
    oracle_net_utility: float

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def evaluate_search_gate(
    output: SearchGateModelOutput,
    batch: SearchGateBatch,
    calibrator: SearchGateCalibrator,
    *,
    calibration_bins: int = 10,
) -> SearchGateMetrics:
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")
    with torch.no_grad():
        probabilities = calibrator.probabilities(output.raw_logit)
        decisions = probabilities.gt(calibrator.threshold)
        labels = batch.labels.bool()
        true_positive = (decisions & labels).sum().float()
        false_positive = (decisions & ~labels).sum().float()
        false_negative = (~decisions & labels).sum().float()
        accuracy = decisions.eq(labels).float().mean()
        precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
        recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
        brier = ((probabilities - batch.labels) ** 2).mean()
        ece = probabilities.new_tensor(0.0)
        boundaries = torch.linspace(0.0, 1.0, calibration_bins + 1, device=probabilities.device)
        for index in range(calibration_bins):
            in_bin = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
                if index == calibration_bins - 1
                else probabilities < boundaries[index + 1]
            )
            if in_bin.any():
                ece += in_bin.float().mean() * (
                    probabilities[in_bin].mean() - batch.labels[in_bin].mean()
                ).abs()
        positive_rate = decisions[labels].float().mean() if labels.any() else probabilities.new_tensor(0.0)
        negative_rate = decisions[~labels].float().mean() if (~labels).any() else probabilities.new_tensor(0.0)
        net_gain = batch.realized_gain - batch.search_cost
        realized_utility = torch.where(decisions, net_gain, torch.zeros_like(net_gain)).mean()
        oracle_utility = net_gain.clamp_min(0.0).mean()
    return SearchGateMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        brier=float(brier),
        ece=float(ece),
        invocation_rate=float(decisions.float().mean()),
        positive_invocation_rate=float(positive_rate),
        negative_invocation_rate=float(negative_rate),
        realized_net_utility=float(realized_utility),
        oracle_net_utility=float(oracle_utility),
    )


@dataclass(frozen=True, slots=True)
class SearchGateDecision:
    should_search: bool
    wanted_search: bool
    budget_limited: bool
    calibrated_probability: float
    predicted_gain: float
    threshold: float
    search_cost: float
    reasons: tuple[str, ...]
    features: SearchGateFeatures
    bank_before: SearchComputeBank
    bank_after: SearchComputeBank
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.calibrated_probability <= 1.0:
            raise ValueError("calibrated gate probability must lie in [0, 1]")
        if not math.isfinite(self.predicted_gain):
            raise ValueError("predicted gate gain must be finite")
        if (
            not 0.0 <= self.threshold <= 1.0
            or not math.isfinite(self.search_cost)
            or self.search_cost < 0.0
        ):
            raise ValueError("gate threshold/cost are invalid")
        if not self.reasons or any(not reason.strip() for reason in self.reasons):
            raise ValueError("gate decision reasons must be non-empty")
        expected_wanted = self.calibrated_probability > self.threshold
        affordable = self.bank_before.can_afford(self.search_cost)
        expected_search = expected_wanted and affordable
        expected_limited = expected_wanted and not affordable
        if self.wanted_search is not expected_wanted:
            raise ValueError("wanted-search flag disagrees with calibrated threshold")
        if self.should_search is not expected_search:
            raise ValueError("search flag disagrees with approval and compute budget")
        if self.budget_limited is not expected_limited:
            raise ValueError("budget-limited flag disagrees with compute availability")
        expected_bank = (
            self.bank_before.spend(self.search_cost)
            if expected_search
            else self.bank_before
        )
        if self.bank_after != expected_bank:
            raise ValueError("gate decision contains an invalid compute-bank transition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "should_search": self.should_search,
            "wanted_search": self.wanted_search,
            "budget_limited": self.budget_limited,
            "calibrated_probability": self.calibrated_probability,
            "predicted_gain": self.predicted_gain,
            "threshold": self.threshold,
            "search_cost": self.search_cost,
            "reasons": list(self.reasons),
            "features": self.features.to_dict(),
            "bank_before": self.bank_before.to_dict(),
            "bank_after": self.bank_after.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchGateDecision":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            should_search=bool(raw["should_search"]),
            wanted_search=bool(raw["wanted_search"]),
            budget_limited=bool(raw["budget_limited"]),
            calibrated_probability=float(raw["calibrated_probability"]),
            predicted_gain=float(raw["predicted_gain"]),
            threshold=float(raw["threshold"]),
            search_cost=float(raw["search_cost"]),
            reasons=tuple(str(reason) for reason in raw["reasons"]),
            features=SearchGateFeatures.from_dict(raw["features"]),
            bank_before=SearchComputeBank.from_dict(raw["bank_before"]),
            bank_after=SearchComputeBank.from_dict(raw["bank_after"]),
        )


class CalibratedSearchGate:
    """Apply the learned gate, threshold, reasons, and compute-bank constraint."""

    def __init__(
        self,
        network: SearchGateNetwork,
        extractor: SearchGateFeatureExtractor,
        calibrator: SearchGateCalibrator,
    ) -> None:
        self.network = network
        self.extractor = extractor
        self.calibrator = calibrator

    @staticmethod
    def _reasons(features: SearchGateFeatures) -> tuple[str, ...]:
        reasons: list[str] = []
        if features.value("policy_uncertainty") >= 0.60:
            reasons.append("low_policy_confidence")
        if features.value("head_disagreement") >= 0.35:
            reasons.append("strategic_value_disagreement")
        if features.value("plan_belief_entropy") >= 0.65:
            reasons.append("high_opponent_belief_entropy")
        if features.value("knockout_swing") >= 0.70:
            reasons.append("knockout_threshold")
        if features.value("prize_swing") >= 0.70:
            reasons.append("prize_threshold")
        if features.value("evolution_window") >= 0.70:
            reasons.append("evolution_threshold")
        if features.value("disruption_window") >= 0.60:
            reasons.append("disruption_threshold")
        if features.value("board_collapse_risk") >= 0.70:
            reasons.append("board_collapse_threshold")
        if (
            features.value("proposal_ambiguity") >= 0.70
            and features.value("optionality_spread") >= 0.35
        ):
            reasons.append("close_plans_different_optionality")
        return tuple(reasons or ("learned_signal_combination",))

    def decide(
        self,
        observation: AgentObservation,
        evaluation: BeliefAwareEvaluation,
        belief: OpponentBeliefState,
        policy: PolicySignals,
        bank: SearchComputeBank,
        *,
        search_cost: float,
        batch_index: int = 0,
    ) -> SearchGateDecision:
        features = self.extractor.extract(
            observation,
            evaluation,
            belief,
            policy,
            bank,
            batch_index=batch_index,
        )
        self.network.eval()
        with torch.no_grad():
            output = self.network(features.to_tensor())
            probability = float(self.calibrator.probabilities(output.raw_logit)[0])
            predicted_gain = float(output.predicted_gain[0])
        wanted = probability > self.calibrator.threshold
        affordable = bank.can_afford(search_cost)
        should_search = wanted and affordable
        budget_limited = wanted and not affordable
        bank_after = bank.spend(search_cost) if should_search else bank
        reasons = list(self._reasons(features))
        if budget_limited:
            reasons.append("compute_budget_exhausted")
        elif not wanted:
            reasons.append("routine_policy_sufficient")
        return SearchGateDecision(
            should_search=should_search,
            wanted_search=wanted,
            budget_limited=budget_limited,
            calibrated_probability=probability,
            predicted_gain=predicted_gain,
            threshold=self.calibrator.threshold,
            search_cost=search_cost,
            reasons=tuple(reasons),
            features=features,
            bank_before=bank,
            bank_after=bank_after,
        )


@dataclass(frozen=True, slots=True)
class RoutineDecision:
    action_prefix: ActionPrefix
    value_estimate: float
    policy_confidence: float
    policy_entropy: float
    legal_action_count: int
    description: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.action_prefix.is_complete:
            raise ValueError("routine decision must contain a complete action prefix")
        if not math.isfinite(self.value_estimate):
            raise ValueError("routine value estimate must be finite")
        PolicySignals(self.policy_confidence, self.policy_entropy, self.legal_action_count)
        if not self.description.strip():
            raise ValueError("routine decision description must be non-empty")

    @property
    def policy_signals(self) -> PolicySignals:
        return PolicySignals(
            self.policy_confidence,
            self.policy_entropy,
            self.legal_action_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_prefix": [token.to_dict() for token in self.action_prefix.tokens],
            "value_estimate": self.value_estimate,
            "policy_confidence": self.policy_confidence,
            "policy_entropy": self.policy_entropy,
            "legal_action_count": self.legal_action_count,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RoutineDecision":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            action_prefix=ActionPrefix(
                tuple(ActionToken.from_dict(token) for token in raw["action_prefix"])
            ),
            value_estimate=float(raw["value_estimate"]),
            policy_confidence=float(raw["policy_confidence"]),
            policy_entropy=float(raw["policy_entropy"]),
            legal_action_count=int(raw["legal_action_count"]),
            description=str(raw["description"]),
        )


class DecisionMode(str, Enum):
    ROUTINE_POLICY = "routine_policy"
    EXPERT_SEARCH = "expert_search"


@dataclass(frozen=True, slots=True)
class SelectiveDecision:
    mode: DecisionMode
    routine: RoutineDecision
    gate: SearchGateDecision
    search_result: PlanSearchResult | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.mode is DecisionMode.EXPERT_SEARCH and self.search_result is None:
            raise ValueError("expert-search decision requires a search result")
        if self.mode is DecisionMode.ROUTINE_POLICY and self.search_result is not None:
            raise ValueError("routine decision cannot carry a search result")
        if (self.mode is DecisionMode.EXPERT_SEARCH) is not self.gate.should_search:
            raise ValueError("decision mode and gate outcome disagree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "routine": self.routine.to_dict(),
            "gate": self.gate.to_dict(),
            "search_result": (
                None if self.search_result is None else self.search_result.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SelectiveDecision":
        result = raw.get("search_result")
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            mode=DecisionMode(str(raw["mode"])),
            routine=RoutineDecision.from_dict(raw["routine"]),
            gate=SearchGateDecision.from_dict(raw["gate"]),
            search_result=None if result is None else PlanSearchResult.from_dict(result),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, encoded: str) -> "SelectiveDecision":
        return cls.from_dict(json.loads(encoded))


class SelectiveSearchPolicy:
    """Run expensive Phase 3 planning only when the calibrated gate approves."""

    def __init__(
        self,
        gate: CalibratedSearchGate,
        planner: BeliefAwarePlanSearch,
    ) -> None:
        self.gate = gate
        self.planner = planner

    def decide(
        self,
        observation: AgentObservation,
        evaluation: BeliefAwareEvaluation,
        belief: OpponentBeliefState,
        routine: RoutineDecision,
        bank: SearchComputeBank,
        *,
        search_cost: float = 1.0,
        batch_index: int = 0,
    ) -> SelectiveDecision:
        gate_decision = self.gate.decide(
            observation,
            evaluation,
            belief,
            routine.policy_signals,
            bank,
            search_cost=search_cost,
            batch_index=batch_index,
        )
        if gate_decision.should_search:
            result = self.planner.search(
                observation,
                evaluation,
                belief,
                batch_index=batch_index,
            )
            return SelectiveDecision(
                mode=DecisionMode.EXPERT_SEARCH,
                routine=routine,
                gate=gate_decision,
                search_result=result,
            )
        return SelectiveDecision(
            mode=DecisionMode.ROUTINE_POLICY,
            routine=routine,
            gate=gate_decision,
        )
