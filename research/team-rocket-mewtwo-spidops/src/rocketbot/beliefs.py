"""Serializable strategic-intent and inverse hidden-resource beliefs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping, Protocol, Sequence

from .observation import AgentObservation
from .state import ActionKind


class OpponentIntent(str, Enum):
    SETUP = "setup"
    PRIZE_RACE = "prize_race"
    LOCK_DISRUPTION = "lock_disruption"
    BOARD_DENIAL = "board_denial"
    RESOURCE_CONSERVATION = "resource_conservation"
    COMBO_ASSEMBLY = "combo_assembly"


OPPONENT_INTENTS: tuple[OpponentIntent, ...] = tuple(OpponentIntent)


class ResourceCategory(str, Enum):
    BASIC_POKEMON = "basic_pokemon"
    EVOLUTION = "evolution"
    ENERGY = "energy"
    SEARCH = "search"
    DRAW = "draw"
    SWITCH_MOBILITY = "switch_mobility"
    RECOVERY = "recovery"
    DISRUPTION = "disruption"
    TOOL = "tool"
    STADIUM = "stadium"


RESOURCE_CATEGORIES: tuple[ResourceCategory, ...] = tuple(ResourceCategory)


class OpponentStyle(str, Enum):
    RATIONAL = "rational"
    IMPERFECT = "imperfect"
    DECEPTIVE = "deceptive"

    @property
    def rationality(self) -> float:
        return {
            OpponentStyle.RATIONAL: 4.0,
            OpponentStyle.IMPERFECT: 0.75,
            OpponentStyle.DECEPTIVE: -0.75,
        }[self]


@dataclass(frozen=True, slots=True)
class OpponentPlanPosterior:
    """A calibrated categorical distribution over compact opponent intents."""

    probabilities: tuple[float, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if len(self.probabilities) != len(OPPONENT_INTENTS):
            raise ValueError(
                f"plan posterior must contain {len(OPPONENT_INTENTS)} probabilities"
            )
        if not all(math.isfinite(value) and value >= 0.0 for value in self.probabilities):
            raise ValueError("plan probabilities must be finite and non-negative")
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("plan probabilities must sum to one")

    @classmethod
    def uniform(cls) -> "OpponentPlanPosterior":
        probability = 1.0 / len(OPPONENT_INTENTS)
        return cls((probability,) * len(OPPONENT_INTENTS))

    @classmethod
    def normalized(cls, weights: Sequence[float]) -> "OpponentPlanPosterior":
        if len(weights) != len(OPPONENT_INTENTS):
            raise ValueError("one plan weight is required per intent")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("plan weights must be finite and non-negative")
        total = sum(weights)
        if total <= 0.0:
            return cls.uniform()
        return cls(tuple(float(weight / total) for weight in weights))

    def probability(self, intent: OpponentIntent) -> float:
        return self.probabilities[OPPONENT_INTENTS.index(intent)]

    @property
    def most_likely(self) -> OpponentIntent:
        return OPPONENT_INTENTS[max(range(len(self.probabilities)), key=self.probabilities.__getitem__)]

    @property
    def entropy(self) -> float:
        return -sum(
            probability * math.log(probability)
            for probability in self.probabilities
            if probability > 0.0
        )

    def update(self, likelihoods: Mapping[OpponentIntent, float]) -> "OpponentPlanPosterior":
        weights = tuple(
            self.probability(intent) * max(float(likelihoods.get(intent, 1.0)), 0.0)
            for intent in OPPONENT_INTENTS
        )
        return OpponentPlanPosterior.normalized(weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "probabilities": {
                intent.value: self.probability(intent) for intent in OPPONENT_INTENTS
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentPlanPosterior":
        probabilities = raw["probabilities"]
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            probabilities=tuple(float(probabilities[intent.value]) for intent in OPPONENT_INTENTS),
        )


@dataclass(frozen=True, slots=True)
class ResourceVector:
    """Minimum or hypothesized counts for coarse private-resource categories."""

    counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.counts) != len(RESOURCE_CATEGORIES):
            raise ValueError(
                f"resource vector must contain {len(RESOURCE_CATEGORIES)} counts"
            )
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in self.counts
        ):
            raise ValueError("resource counts must be non-negative integers")

    @classmethod
    def empty(cls) -> "ResourceVector":
        return cls((0,) * len(RESOURCE_CATEGORIES))

    @classmethod
    def from_mapping(
        cls,
        counts: Mapping[ResourceCategory, int],
    ) -> "ResourceVector":
        return cls(tuple(int(counts.get(category, 0)) for category in RESOURCE_CATEGORIES))

    def count(self, category: ResourceCategory) -> int:
        return self.counts[RESOURCE_CATEGORIES.index(category)]

    def covers(self, minimum: "ResourceVector") -> bool:
        return all(have >= needed for have, needed in zip(self.counts, minimum.counts))

    @property
    def presence(self) -> tuple[float, ...]:
        return tuple(float(count > 0) for count in self.counts)

    def to_dict(self) -> dict[str, int]:
        return {
            category.value: self.count(category) for category in RESOURCE_CATEGORIES
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResourceVector":
        return cls(tuple(int(raw.get(category.value, 0)) for category in RESOURCE_CATEGORIES))


@dataclass(frozen=True, slots=True)
class StrategicLineHypothesis:
    """A proposed opponent line and its minimum enabling private resources."""

    line_id: str
    intent: OpponentIntent
    action_signature: tuple[ActionKind, ...]
    minimum_resources: ResourceVector
    base_utility: float
    description: str

    def __post_init__(self) -> None:
        if not self.line_id.strip() or not self.description.strip():
            raise ValueError("line_id and description must be non-empty")
        if not self.action_signature:
            raise ValueError("a strategic line must contain at least one action")
        if not math.isfinite(self.base_utility):
            raise ValueError("line utility must be finite")


class PlausibleLineGenerator(Protocol):
    """Simulator adapters can replace the rule-agnostic reference generator."""

    def propose(
        self,
        observation: AgentObservation,
        plan_prior: OpponentPlanPosterior,
    ) -> Sequence[StrategicLineHypothesis]: ...


class IntentTemplateLineGenerator:
    """Deterministic reference lines for exercising inverse planning.

    These templates are strategic action-category sketches, not Pokemon card-rule
    simulation. A competition environment should supply legal, card-specific
    lines through `PlausibleLineGenerator` while preserving the same belief API.
    """

    def propose(
        self,
        observation: AgentObservation,
        plan_prior: OpponentPlanPosterior,
    ) -> tuple[StrategicLineHypothesis, ...]:
        opponent = observation.public_state.player(observation.observer.other)
        setup_pressure = (5 - len(opponent.bench)) / 5.0
        prize_pressure = (6 - opponent.prizes_remaining) / 6.0
        low_hand_pressure = max(0.0, (4 - len(opponent.hand)) / 4.0)
        specs = (
            (
                "setup_board",
                OpponentIntent.SETUP,
                (ActionKind.PLAY_POKEMON, ActionKind.PLAY_TRAINER, ActionKind.ATTACH_ENERGY, ActionKind.END_TURN),
                {
                    ResourceCategory.BASIC_POKEMON: 1,
                    ResourceCategory.SEARCH: 1,
                    ResourceCategory.ENERGY: 1,
                },
                1.0 + setup_pressure,
                "Develop a Pokemon, find a setup piece, and attach for a later turn.",
            ),
            (
                "push_prize_race",
                OpponentIntent.PRIZE_RACE,
                (ActionKind.ATTACH_ENERGY, ActionKind.ATTACK),
                {ResourceCategory.ENERGY: 1},
                1.0 + prize_pressure,
                "Convert an energy attachment into immediate prize pressure.",
            ),
            (
                "apply_lock",
                OpponentIntent.LOCK_DISRUPTION,
                (ActionKind.PLAY_TRAINER, ActionKind.USE_ABILITY, ActionKind.END_TURN),
                {ResourceCategory.DISRUPTION: 1},
                1.0,
                "Reduce the opponent's choices with disruption and a locking effect.",
            ),
            (
                "deny_board",
                OpponentIntent.BOARD_DENIAL,
                (ActionKind.PLAY_TRAINER, ActionKind.ATTACK),
                {ResourceCategory.DISRUPTION: 1, ResourceCategory.ENERGY: 1},
                1.0 + prize_pressure / 2.0,
                "Target a vulnerable board piece before it develops.",
            ),
            (
                "conserve_resources",
                OpponentIntent.RESOURCE_CONSERVATION,
                (ActionKind.USE_ABILITY, ActionKind.END_TURN),
                {ResourceCategory.DRAW: 1},
                1.0 + low_hand_pressure,
                "Use a renewable effect and retain scarce cards for a later pivot.",
            ),
            (
                "assemble_combo",
                OpponentIntent.COMBO_ASSEMBLY,
                (ActionKind.PLAY_TRAINER, ActionKind.EVOLVE, ActionKind.ATTACH_ENERGY, ActionKind.END_TURN),
                {
                    ResourceCategory.SEARCH: 1,
                    ResourceCategory.EVOLUTION: 1,
                    ResourceCategory.ENERGY: 1,
                },
                1.0 + setup_pressure / 2.0,
                "Assemble multiple enabling resources for a future payoff turn.",
            ),
        )
        return tuple(
            StrategicLineHypothesis(
                line_id=line_id,
                intent=intent,
                action_signature=signature,
                minimum_resources=ResourceVector.from_mapping(resources),
                base_utility=utility + math.log(max(plan_prior.probability(intent), 1e-6)) * 0.1,
                description=description,
            )
            for line_id, intent, signature, resources, utility, description in specs
        )


@dataclass(frozen=True, slots=True)
class HiddenResourceHypothesis:
    hypothesis_id: str
    intent: OpponentIntent
    resources: ResourceVector
    style: OpponentStyle
    probability: float
    explanation: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip() or not self.explanation.strip():
            raise ValueError("hypothesis_id and explanation must be non-empty")
        if not math.isfinite(self.probability) or self.probability < 0.0:
            raise ValueError("hypothesis probability must be finite and non-negative")

    def with_probability(self, probability: float) -> "HiddenResourceHypothesis":
        return HiddenResourceHypothesis(
            hypothesis_id=self.hypothesis_id,
            intent=self.intent,
            resources=self.resources,
            style=self.style,
            probability=probability,
            explanation=self.explanation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "intent": self.intent.value,
            "resources": self.resources.to_dict(),
            "style": self.style.value,
            "probability": self.probability,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HiddenResourceHypothesis":
        return cls(
            hypothesis_id=str(raw["hypothesis_id"]),
            intent=OpponentIntent(str(raw["intent"])),
            resources=ResourceVector.from_dict(raw["resources"]),
            style=OpponentStyle(str(raw["style"])),
            probability=float(raw["probability"]),
            explanation=str(raw["explanation"]),
        )


@dataclass(frozen=True, slots=True)
class HiddenResourcePosterior:
    """A normalized beam of explanatory private-resource hypotheses."""

    hypotheses: tuple[HiddenResourceHypothesis, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("hidden-resource posterior cannot be empty")
        identifiers = [hypothesis.hypothesis_id for hypothesis in self.hypotheses]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("hidden-resource hypothesis IDs must be unique")
        if not math.isclose(
            sum(hypothesis.probability for hypothesis in self.hypotheses),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("hidden-resource probabilities must sum to one")

    @classmethod
    def normalized(
        cls,
        hypotheses: Sequence[HiddenResourceHypothesis],
    ) -> "HiddenResourcePosterior":
        if not hypotheses:
            raise ValueError("at least one hidden-resource hypothesis is required")
        total = sum(hypothesis.probability for hypothesis in hypotheses)
        if total <= 0.0:
            probability = 1.0 / len(hypotheses)
            normalized = tuple(
                hypothesis.with_probability(probability) for hypothesis in hypotheses
            )
        else:
            normalized = tuple(
                hypothesis.with_probability(hypothesis.probability / total)
                for hypothesis in hypotheses
            )
        return cls(normalized)

    def probability(self, hypothesis_id: str) -> float:
        return next(
            hypothesis.probability
            for hypothesis in self.hypotheses
            if hypothesis.hypothesis_id == hypothesis_id
        )

    @property
    def entropy(self) -> float:
        return -sum(
            hypothesis.probability * math.log(hypothesis.probability)
            for hypothesis in self.hypotheses
            if hypothesis.probability > 0.0
        )

    @property
    def marginal_presence(self) -> tuple[float, ...]:
        return tuple(
            sum(
                hypothesis.probability
                for hypothesis in self.hypotheses
                if hypothesis.resources.count(category) > 0
            )
            for category in RESOURCE_CATEGORIES
        )

    @property
    def expected_counts(self) -> tuple[float, ...]:
        return tuple(
            sum(
                hypothesis.probability * hypothesis.resources.count(category)
                for hypothesis in self.hypotheses
            )
            for category in RESOURCE_CATEGORIES
        )

    @property
    def plan_posterior(self) -> OpponentPlanPosterior:
        return OpponentPlanPosterior.normalized(
            tuple(
                sum(
                    hypothesis.probability
                    for hypothesis in self.hypotheses
                    if hypothesis.intent is intent
                )
                for intent in OPPONENT_INTENTS
            )
        )

    def condition_on_reveal(
        self,
        category: ResourceCategory,
        *,
        present: bool = True,
        minimum_count: int = 1,
        observation_error: float = 0.01,
    ) -> "HiddenResourcePosterior":
        if minimum_count <= 0:
            raise ValueError("minimum_count must be positive")
        if not 0.0 < observation_error < 0.5:
            raise ValueError("observation_error must lie between zero and 0.5")
        reweighted = []
        for hypothesis in self.hypotheses:
            predicted = hypothesis.resources.count(category) >= minimum_count
            consistent = predicted is present
            likelihood = 1.0 - observation_error if consistent else observation_error
            reweighted.append(
                hypothesis.with_probability(hypothesis.probability * likelihood)
            )
        return HiddenResourcePosterior.normalized(reweighted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HiddenResourcePosterior":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            hypotheses=tuple(
                HiddenResourceHypothesis.from_dict(item) for item in raw["hypotheses"]
            ),
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
    def from_json(cls, encoded: str) -> "HiddenResourcePosterior":
        return cls.from_dict(json.loads(encoded))


class MinimumEnablingHypothesisGenerator:
    """Create a bounded beam from the minimum resources of proposed lines."""

    STYLE_PRIORS: Mapping[OpponentStyle, float] = {
        OpponentStyle.RATIONAL: 0.75,
        OpponentStyle.IMPERFECT: 0.20,
        OpponentStyle.DECEPTIVE: 0.05,
    }

    def __init__(self, beam_size: int = 32, unexplained_floor: float = 0.02) -> None:
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if not 0.0 < unexplained_floor < 1.0:
            raise ValueError("unexplained_floor must lie between zero and one")
        self.beam_size = beam_size
        self.unexplained_floor = unexplained_floor

    def generate(
        self,
        lines: Sequence[StrategicLineHypothesis],
        plan_prior: OpponentPlanPosterior,
    ) -> HiddenResourcePosterior:
        if not lines:
            raise ValueError("at least one plausible line is required")
        hypotheses: list[HiddenResourceHypothesis] = []
        for line in lines:
            plan_weight = max(plan_prior.probability(line.intent), 1e-6)
            utility_weight = math.exp(max(-10.0, min(line.base_utility, 10.0)))
            for style, style_prior in self.STYLE_PRIORS.items():
                hypotheses.append(
                    HiddenResourceHypothesis(
                        hypothesis_id=f"{line.line_id}:{style.value}",
                        intent=line.intent,
                        resources=line.minimum_resources,
                        style=style,
                        probability=plan_weight * utility_weight * style_prior,
                        explanation=(
                            f"Minimum resources for '{line.description}' under a "
                            f"{style.value} behavior model."
                        ),
                    )
                )

        fallback_intent = plan_prior.most_likely
        hypotheses.append(
            HiddenResourceHypothesis(
                hypothesis_id="unmodeled:imperfect",
                intent=fallback_intent,
                resources=ResourceVector.empty(),
                style=OpponentStyle.IMPERFECT,
                probability=self.unexplained_floor,
                explanation="Unmodeled or mistaken play requiring no asserted private resource.",
            )
        )
        hypotheses.sort(key=lambda hypothesis: hypothesis.probability, reverse=True)
        return HiddenResourcePosterior.normalized(hypotheses[: self.beam_size])


def _sequence_similarity(
    observed: Sequence[ActionKind],
    predicted: Sequence[ActionKind],
) -> float:
    if not observed and not predicted:
        return 1.0
    if not observed or not predicted:
        return 0.0
    rows = len(observed) + 1
    columns = len(predicted) + 1
    table = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        for column in range(1, columns):
            if observed[row - 1] is predicted[column - 1]:
                table[row][column] = table[row - 1][column - 1] + 1
            else:
                table[row][column] = max(
                    table[row - 1][column],
                    table[row][column - 1],
                )
    longest_common_subsequence = table[-1][-1]
    return longest_common_subsequence / max(len(observed), len(predicted))


@dataclass(frozen=True, slots=True)
class InversePlanningUpdate:
    observed_actions: tuple[ActionKind, ...]
    hidden_resources: HiddenResourcePosterior
    plan: OpponentPlanPosterior
    hypothesis_likelihoods: tuple[tuple[str, float], ...]


class InversePlanningBeliefUpdater:
    """Bayesian update using action rationality under each resource hypothesis."""

    def __init__(
        self,
        *,
        random_action_floor: float = 0.05,
        observation_floor: float = 0.01,
        intent_match_bonus: float = 0.75,
    ) -> None:
        if not 0.0 < random_action_floor < 1.0:
            raise ValueError("random_action_floor must lie between zero and one")
        if not 0.0 < observation_floor < 1.0:
            raise ValueError("observation_floor must lie between zero and one")
        self.random_action_floor = random_action_floor
        self.observation_floor = observation_floor
        self.intent_match_bonus = intent_match_bonus

    def update(
        self,
        prior: HiddenResourcePosterior,
        lines: Sequence[StrategicLineHypothesis],
        observed_actions: Sequence[ActionKind],
    ) -> InversePlanningUpdate:
        if not lines:
            raise ValueError("at least one plausible line is required")
        observed = tuple(observed_actions)
        reweighted: list[HiddenResourceHypothesis] = []
        likelihood_trace: list[tuple[str, float]] = []

        emissions = []
        for line in lines:
            similarity = _sequence_similarity(observed, line.action_signature)
            length_gap = abs(len(observed) - len(line.action_signature)) / max(
                len(observed), len(line.action_signature), 1
            )
            emission = self.observation_floor + (1.0 - self.observation_floor) * math.exp(
                -4.0 * (1.0 - similarity) - length_gap
            )
            emissions.append(emission)

        for hypothesis in prior.hypotheses:
            feasible_indices = [
                index
                for index, line in enumerate(lines)
                if hypothesis.resources.covers(line.minimum_resources)
            ]
            if feasible_indices:
                utilities = [
                    lines[index].base_utility
                    + self.intent_match_bonus
                    * float(lines[index].intent is hypothesis.intent)
                    for index in feasible_indices
                ]
                scaled = [hypothesis.style.rationality * utility for utility in utilities]
                maximum = max(scaled)
                exponentials = [math.exp(value - maximum) for value in scaled]
                denominator = sum(exponentials)
                rational_likelihood = sum(
                    (exponential / denominator) * emissions[index]
                    for exponential, index in zip(exponentials, feasible_indices)
                )
            else:
                rational_likelihood = self.observation_floor

            likelihood = (
                self.random_action_floor * self.observation_floor
                + (1.0 - self.random_action_floor) * rational_likelihood
            )
            likelihood_trace.append((hypothesis.hypothesis_id, likelihood))
            reweighted.append(
                hypothesis.with_probability(hypothesis.probability * likelihood)
            )

        posterior = HiddenResourcePosterior.normalized(reweighted)
        return InversePlanningUpdate(
            observed_actions=observed,
            hidden_resources=posterior,
            plan=posterior.plan_posterior,
            hypothesis_likelihoods=tuple(likelihood_trace),
        )


@dataclass(frozen=True, slots=True)
class OpponentBeliefState:
    """Serializable neural/symbolic belief state exposed to policy and search."""

    match_id: str
    observed_through_turn: int
    plan: OpponentPlanPosterior
    resource_probabilities: tuple[float, ...]
    hidden_hypotheses: HiddenResourcePosterior | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id must be non-empty")
        if self.observed_through_turn < 0:
            raise ValueError("observed_through_turn cannot be negative")
        if len(self.resource_probabilities) != len(RESOURCE_CATEGORIES):
            raise ValueError("one probability is required per resource category")
        if not all(
            math.isfinite(probability) and 0.0 <= probability <= 1.0
            for probability in self.resource_probabilities
        ):
            raise ValueError("resource probabilities must lie in [0, 1]")

    @property
    def context_vector(self) -> tuple[float, ...]:
        return self.plan.probabilities + self.resource_probabilities

    @property
    def resource_entropy(self) -> float:
        entropy = 0.0
        for probability in self.resource_probabilities:
            if 0.0 < probability < 1.0:
                entropy -= probability * math.log(probability)
                entropy -= (1.0 - probability) * math.log(1.0 - probability)
        return entropy

    def fuse_symbolic(
        self,
        symbolic: HiddenResourcePosterior,
        *,
        symbolic_weight: float = 0.5,
    ) -> "OpponentBeliefState":
        if not 0.0 <= symbolic_weight <= 1.0:
            raise ValueError("symbolic_weight must lie in [0, 1]")
        neural_weight = 1.0 - symbolic_weight
        plan = OpponentPlanPosterior.normalized(
            tuple(
                neural_weight * neural
                + symbolic_weight * symbolic.plan_posterior.probabilities[index]
                for index, neural in enumerate(self.plan.probabilities)
            )
        )
        resources = tuple(
            neural_weight * neural + symbolic_weight * inverse
            for neural, inverse in zip(
                self.resource_probabilities,
                symbolic.marginal_presence,
            )
        )
        return OpponentBeliefState(
            match_id=self.match_id,
            observed_through_turn=self.observed_through_turn,
            plan=plan,
            resource_probabilities=resources,
            hidden_hypotheses=symbolic,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "observed_through_turn": self.observed_through_turn,
            "plan": self.plan.to_dict(),
            "resource_probabilities": {
                category.value: self.resource_probabilities[index]
                for index, category in enumerate(RESOURCE_CATEGORIES)
            },
            "hidden_hypotheses": (
                None if self.hidden_hypotheses is None else self.hidden_hypotheses.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentBeliefState":
        resource_probabilities = raw["resource_probabilities"]
        hidden = raw.get("hidden_hypotheses")
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            match_id=str(raw["match_id"]),
            observed_through_turn=int(raw["observed_through_turn"]),
            plan=OpponentPlanPosterior.from_dict(raw["plan"]),
            resource_probabilities=tuple(
                float(resource_probabilities[category.value])
                for category in RESOURCE_CATEGORIES
            ),
            hidden_hypotheses=(
                None if hidden is None else HiddenResourcePosterior.from_dict(hidden)
            ),
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
    def from_json(cls, encoded: str) -> "OpponentBeliefState":
        return cls.from_dict(json.loads(encoded))
