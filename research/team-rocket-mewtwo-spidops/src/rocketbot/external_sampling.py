"""Auditable external-sampling MCCFR traversal primitives for Phase 6L.

This module implements the traversal in Algorithm 2 of Brown et al. (2019):
all actions are expanded at a traverser's information set, while one action is
sampled at chance and opponent information sets.  The resulting action-value
differences are *conditional* external-sampling advantages.  They are not
terminal-rollout labels, counterfactual-value safety bounds, or evidence that a
large game has converged to equilibrium.

The traversal is generic over :class:`AbstractInformationSetGame`.  Native
descendant transitions can therefore be genuine licensed-engine executions,
but any caller-supplied hidden-completion prior remains a research abstraction.
Budgets fail closed: partial traversals never return training samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
from typing import Any, Callable, Mapping, Sequence

from .game_theory import (
    AbstractGameState,
    AbstractInformationSetGame,
    ActionDistribution,
    ChanceOutcome,
    InformationSet,
    InformationSetActor,
    PublicBeliefState,
)


EXTERNAL_SAMPLING_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _finite(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


StrategyFunction = Callable[[InformationSet, PublicBeliefState], ActionDistribution]
InformationViewFunction = Callable[
    [AbstractInformationSetGame, AbstractGameState, int],
    tuple[InformationSet, PublicBeliefState],
]
ActionAdapter = Callable[
    [AbstractInformationSetGame, AbstractGameState, Any],
    Any,
]


@dataclass(frozen=True, slots=True)
class ExternalSamplingConfig:
    """Hard traversal limits; exceeding either aborts without usable samples."""

    maximum_transitions: int = 100_000
    maximum_depth: int = 4_096

    def __post_init__(self) -> None:
        if self.maximum_transitions <= 0 or self.maximum_depth <= 0:
            raise ValueError("external-sampling limits must be positive")


class ExternalSamplingBudgetExceeded(RuntimeError):
    """Raised before a partial traversal can be mistaken for CFR supervision."""


def _feature_payload(
    information_set: InformationSet,
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, tuple[tuple[str, float], ...]], ...],
]:
    action_features = tuple(
        (action.action_id, tuple((name, float(value)) for name, value in action.features))
        for action in information_set.legal_actions
    )
    return tuple(information_set.private_features), action_features


@dataclass(frozen=True, slots=True)
class ExternalSamplingAdvantageSample:
    """One Deep-CFR-style instantaneous advantage vector.

    Values are conditioned on the information set being reached by the sampled
    external block.  No explicit counterfactual reach multiplier belongs on
    this record; reach is represented by whether the block visits the set.
    """

    game_fingerprint: str
    traverser: int
    iteration: int
    information_set_id: str
    public_belief_fingerprint: str
    private_observation_fingerprint: str | None
    depth: int
    strategy: tuple[tuple[str, float], ...]
    action_values: tuple[tuple[str, float], ...]
    instantaneous_advantages: tuple[tuple[str, float], ...]
    public_features: tuple[tuple[str, float], ...]
    private_features: tuple[tuple[str, float], ...]
    action_features: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    source_official_engine_executed: bool
    hidden_completion_prior_official: bool

    def __post_init__(self) -> None:
        _validate_digest(self.game_fingerprint, "game_fingerprint")
        _validate_digest(self.public_belief_fingerprint, "public_belief_fingerprint")
        if self.private_observation_fingerprint is not None:
            _validate_digest(
                self.private_observation_fingerprint,
                "private_observation_fingerprint",
            )
        if self.traverser not in (0, 1) or self.iteration <= 0 or self.depth < 0:
            raise ValueError("external-sampling sample player/iteration/depth is invalid")
        if not self.information_set_id:
            raise ValueError("external-sampling sample requires an information-set ID")
        identifiers = tuple(action for action, _ in self.strategy)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("external-sampling action IDs must be unique and non-empty")
        for field_name in ("action_values", "instantaneous_advantages"):
            values = getattr(self, field_name)
            if tuple(action for action, _ in values) != identifiers:
                raise ValueError(f"{field_name} support/order must match strategy")
        if tuple(action for action, _ in self.action_features) != identifiers:
            raise ValueError("action-feature support/order must match strategy")
        probabilities = tuple(_finite(value, "strategy probability") for _, value in self.strategy)
        if any(value < 0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, abs_tol=1e-8
        ):
            raise ValueError("external-sampling strategy must be a distribution")
        for values in (self.action_values, self.instantaneous_advantages):
            for _, value in values:
                _finite(value, "external-sampling value")
        for name, value in self.public_features + self.private_features:
            if not name:
                raise ValueError("external-sampling feature names must be non-empty")
            _finite(value, f"feature {name}")
        for _, features in self.action_features:
            for name, value in features:
                if not name:
                    raise ValueError("external-sampling action feature names must be non-empty")
                _finite(value, f"action feature {name}")
        if self.hidden_completion_prior_official and not self.source_official_engine_executed:
            raise ValueError("an official hidden prior cannot be claimed without official execution")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "traverser": self.traverser,
            "iteration": self.iteration,
            "information_set_id": self.information_set_id,
            "public_belief_fingerprint": self.public_belief_fingerprint,
            "private_observation_fingerprint": self.private_observation_fingerprint,
            "depth": self.depth,
            "strategy": [list(item) for item in self.strategy],
            "action_values": [list(item) for item in self.action_values],
            "instantaneous_advantages": [
                list(item) for item in self.instantaneous_advantages
            ],
            "public_features": [list(item) for item in self.public_features],
            "private_features": [list(item) for item in self.private_features],
            "action_features": [
                [action_id, [list(item) for item in features]]
                for action_id, features in self.action_features
            ],
            "semantics": {
                "external_sampling_conditioned_on_information_set_reached": True,
                "counterfactual_reach_multiplier_applied": False,
                "terminal_rollout_label": False,
                "calibrated_counterfactual_value": False,
            },
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "selected_hidden_truth_persisted": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExternalSamplingAdvantageSample":
        provenance = raw["provenance"]
        result = cls(
            game_fingerprint=str(raw["game_fingerprint"]),
            traverser=int(raw["traverser"]),
            iteration=int(raw["iteration"]),
            information_set_id=str(raw["information_set_id"]),
            public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
            private_observation_fingerprint=(
                None
                if raw.get("private_observation_fingerprint") is None
                else str(raw["private_observation_fingerprint"])
            ),
            depth=int(raw["depth"]),
            strategy=tuple((str(key), float(value)) for key, value in raw["strategy"]),
            action_values=tuple(
                (str(key), float(value)) for key, value in raw["action_values"]
            ),
            instantaneous_advantages=tuple(
                (str(key), float(value))
                for key, value in raw["instantaneous_advantages"]
            ),
            public_features=tuple(
                (str(key), float(value)) for key, value in raw["public_features"]
            ),
            private_features=tuple(
                (str(key), float(value)) for key, value in raw["private_features"]
            ),
            action_features=tuple(
                (
                    str(action_id),
                    tuple((str(key), float(value)) for key, value in features),
                )
                for action_id, features in raw["action_features"]
            ),
            source_official_engine_executed=bool(
                provenance["source_official_engine_executed"]
            ),
            hidden_completion_prior_official=bool(
                provenance["hidden_completion_prior_official"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("external-sampling advantage fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ExternalSamplingStrategySample:
    """Current opponent strategy observed during the other player's traversal."""

    game_fingerprint: str
    player: int
    traverser: int
    iteration: int
    information_set_id: str
    public_belief_fingerprint: str
    depth: int
    strategy: tuple[tuple[str, float], ...]
    source_official_engine_executed: bool
    hidden_completion_prior_official: bool

    def __post_init__(self) -> None:
        _validate_digest(self.game_fingerprint, "game_fingerprint")
        _validate_digest(self.public_belief_fingerprint, "public_belief_fingerprint")
        if self.player not in (0, 1) or self.traverser not in (0, 1):
            raise ValueError("external-sampling strategy sample player is invalid")
        if self.player == self.traverser:
            raise ValueError("strategy memory is collected on the opposing traversal")
        if self.iteration <= 0 or self.depth < 0 or not self.information_set_id:
            raise ValueError("external-sampling strategy sample metadata is invalid")
        if not self.strategy or len({key for key, _ in self.strategy}) != len(self.strategy):
            raise ValueError("external-sampling strategy support must be unique")
        probabilities = tuple(_finite(value, "strategy probability") for _, value in self.strategy)
        if any(value < 0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, abs_tol=1e-8
        ):
            raise ValueError("external-sampling strategy must sum to one")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "player": self.player,
            "traverser": self.traverser,
            "iteration": self.iteration,
            "information_set_id": self.information_set_id,
            "public_belief_fingerprint": self.public_belief_fingerprint,
            "depth": self.depth,
            "strategy": [list(item) for item in self.strategy],
            "weight": self.iteration,
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "selected_hidden_truth_persisted": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExternalSamplingStrategySample":
        provenance = raw["provenance"]
        result = cls(
            game_fingerprint=str(raw["game_fingerprint"]),
            player=int(raw["player"]),
            traverser=int(raw["traverser"]),
            iteration=int(raw["iteration"]),
            information_set_id=str(raw["information_set_id"]),
            public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
            depth=int(raw["depth"]),
            strategy=tuple((str(key), float(value)) for key, value in raw["strategy"]),
            source_official_engine_executed=bool(
                provenance["source_official_engine_executed"]
            ),
            hidden_completion_prior_official=bool(
                provenance["hidden_completion_prior_official"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("external-sampling strategy fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ExternalSamplingTraversalResult:
    traverser: int
    iteration: int
    sampled_block_value: float
    transitions: int
    terminal_leaves: int
    maximum_depth_reached: int
    advantage_samples: tuple[ExternalSamplingAdvantageSample, ...]
    strategy_samples: tuple[ExternalSamplingStrategySample, ...]
    source_official_engine_executed: bool
    hidden_completion_prior_official: bool
    complete: bool = True

    def __post_init__(self) -> None:
        if self.traverser not in (0, 1) or self.iteration <= 0:
            raise ValueError("external-sampling traversal metadata is invalid")
        _finite(self.sampled_block_value, "sampled block value")
        if self.transitions <= 0 or self.terminal_leaves <= 0:
            raise ValueError("completed traversal requires transitions and terminal leaves")
        if self.maximum_depth_reached <= 0 or not self.complete:
            raise ValueError("only complete external-sampling traversals are materialized")
        if any(item.traverser != self.traverser for item in self.advantage_samples):
            raise ValueError("advantage samples must belong to the traverser")
        if any(item.traverser != self.traverser for item in self.strategy_samples):
            raise ValueError("strategy samples must belong to the traversal")
        if self.hidden_completion_prior_official and not self.source_official_engine_executed:
            raise ValueError("official hidden-prior provenance is inconsistent")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "algorithm": "external_sampling_mccfr_all_traverser_actions",
            "traverser": self.traverser,
            "iteration": self.iteration,
            "sampled_block_value": self.sampled_block_value,
            "transitions": self.transitions,
            "terminal_leaves": self.terminal_leaves,
            "maximum_depth_reached": self.maximum_depth_reached,
            "advantage_samples": [item.to_dict() for item in self.advantage_samples],
            "strategy_samples": [item.to_dict() for item in self.strategy_samples],
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "selected_hidden_truth_persisted": False,
            },
            "claims": {
                "complete_traversal_within_budget": self.complete,
                "full_game_equilibrium_convergence": False,
                "safe_resolving": False,
                "deployable_blueprint": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result


@dataclass(slots=True)
class _TraversalContext:
    transitions: int
    terminal_leaves: int
    maximum_depth_reached: int
    sampled_actions: dict[tuple[int, str], str]
    recorded_strategy_sets: set[tuple[int, str]]
    advantage_samples: list[ExternalSamplingAdvantageSample]
    strategy_samples: list[ExternalSamplingStrategySample]


def _sample_identifier(
    probabilities: Sequence[tuple[str, float]],
    rng: random.Random,
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for identifier, probability in probabilities:
        cumulative += probability
        if threshold <= cumulative + 1e-15:
            return identifier
    return probabilities[-1][0]


class ExternalSamplingTraversal:
    """One fail-closed external-sampling traversal for one traverser."""

    def __init__(self, config: ExternalSamplingConfig | None = None) -> None:
        self.config = config or ExternalSamplingConfig()

    def run(
        self,
        game: AbstractInformationSetGame,
        strategy: StrategyFunction,
        *,
        traverser: int,
        iteration: int,
        rng: random.Random,
        root_state: AbstractGameState | None = None,
        information_view: InformationViewFunction | None = None,
        action_adapter: ActionAdapter | None = None,
        forced_chance: Mapping[str, str] | None = None,
        forced_actions: Mapping[str, str] | None = None,
    ) -> ExternalSamplingTraversalResult:
        if traverser not in (0, 1) or iteration <= 0:
            raise ValueError("external-sampling traverser/iteration is invalid")
        context = _TraversalContext(0, 0, 0, {}, set(), [], [])
        official = bool(getattr(game, "source_official_engine_executed", False))
        prior_official = bool(getattr(game, "hidden_completion_prior_official", False))
        start = root_state if root_state is not None else game.root_state()
        view_function = information_view or (
            lambda boundary, state, player: (
                boundary.information_set(state, player=player),
                boundary.public_belief(state, player=player),
            )
        )
        adapt_action = action_adapter or (lambda boundary, state, action: action)
        value = self._visit(
            game,
            start,
            strategy,
            traverser=traverser,
            iteration=iteration,
            rng=rng,
            context=context,
            official=official,
            prior_official=prior_official,
            information_view=view_function,
            action_adapter=adapt_action,
            forced_chance=dict(forced_chance or {}),
            forced_actions=dict(forced_actions or {}),
        )
        return ExternalSamplingTraversalResult(
            traverser=traverser,
            iteration=iteration,
            sampled_block_value=value,
            transitions=context.transitions,
            terminal_leaves=context.terminal_leaves,
            maximum_depth_reached=context.maximum_depth_reached,
            advantage_samples=tuple(context.advantage_samples),
            strategy_samples=tuple(context.strategy_samples),
            source_official_engine_executed=official,
            hidden_completion_prior_official=prior_official,
        )

    def _transition(
        self,
        game: AbstractInformationSetGame,
        state: AbstractGameState,
        action: Any,
        *,
        chance_outcome: ChanceOutcome | None,
        context: _TraversalContext,
    ) -> AbstractGameState:
        if context.transitions >= self.config.maximum_transitions:
            raise ExternalSamplingBudgetExceeded(
                "external-sampling transition budget exhausted; partial samples discarded"
            )
        context.transitions += 1
        if chance_outcome is None:
            return game.transition(state, action)
        return game.transition(state, None, chance_outcome=chance_outcome)

    def _visit(
        self,
        game: AbstractInformationSetGame,
        state: AbstractGameState,
        strategy_function: StrategyFunction,
        *,
        traverser: int,
        iteration: int,
        rng: random.Random,
        context: _TraversalContext,
        official: bool,
        prior_official: bool,
        information_view: InformationViewFunction,
        action_adapter: ActionAdapter,
        forced_chance: Mapping[str, str],
        forced_actions: Mapping[str, str],
    ) -> float:
        context.maximum_depth_reached = max(context.maximum_depth_reached, state.depth)
        if state.depth > self.config.maximum_depth:
            raise ExternalSamplingBudgetExceeded(
                "external-sampling depth budget exhausted; partial samples discarded"
            )
        if state.terminal:
            context.terminal_leaves += 1
            return float(game.terminal_utility(state, traverser))
        if state.actor is InformationSetActor.CHANCE:
            outcomes = game.chance_outcomes(state)
            if not outcomes or not math.isclose(
                sum(item.probability for item in outcomes), 1.0, abs_tol=1e-8
            ):
                raise ValueError("chance outcomes must form a non-empty distribution")
            forced = forced_chance.get(state.state_id)
            identifier = forced or _sample_identifier(
                tuple((item.outcome_id, item.probability) for item in outcomes), rng
            )
            matches = [item for item in outcomes if item.outcome_id == identifier]
            if len(matches) != 1 or matches[0].probability <= 0:
                raise ValueError("forced/sampled chance outcome is not in positive support")
            child = self._transition(
                game,
                state,
                None,
                chance_outcome=matches[0],
                context=context,
            )
            return self._visit(
                game,
                child,
                strategy_function,
                traverser=traverser,
                iteration=iteration,
                rng=rng,
                context=context,
                official=official,
                prior_official=prior_official,
                information_view=information_view,
                action_adapter=action_adapter,
                forced_chance=forced_chance,
                forced_actions=forced_actions,
            )
        player = game.player_to_act(state)
        if player not in (0, 1):
            raise ValueError("nonterminal non-chance state requires player 0 or 1")
        information_set, belief = information_view(game, state, player)
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("external-sampling information view has mismatched belief")
        distribution = strategy_function(information_set, belief)
        distribution.validate_support(information_set)
        probabilities = tuple(
            (action.action_id, distribution.probability(action.action_id))
            for action in information_set.legal_actions
        )
        actions = {action.action_id: action for action in information_set.legal_actions}
        if player == traverser:
            values: list[tuple[str, float]] = []
            for action_id, _ in probabilities:
                child = self._transition(
                    game,
                    state,
                    action_adapter(game, state, actions[action_id]),
                    chance_outcome=None,
                    context=context,
                )
                values.append(
                    (
                        action_id,
                        self._visit(
                            game,
                            child,
                            strategy_function,
                            traverser=traverser,
                            iteration=iteration,
                            rng=rng,
                            context=context,
                            official=official,
                            prior_official=prior_official,
                            information_view=information_view,
                            action_adapter=action_adapter,
                            forced_chance=forced_chance,
                            forced_actions=forced_actions,
                        ),
                    )
                )
            value_by_action = dict(values)
            baseline = sum(
                probability * value_by_action[action_id]
                for action_id, probability in probabilities
            )
            private_features, action_features = _feature_payload(information_set)
            context.advantage_samples.append(
                ExternalSamplingAdvantageSample(
                    game_fingerprint=game.fingerprint,
                    traverser=traverser,
                    iteration=iteration,
                    information_set_id=information_set.info_set_id,
                    public_belief_fingerprint=belief.fingerprint,
                    private_observation_fingerprint=(
                        information_set.private_observation_fingerprint
                    ),
                    depth=information_set.depth,
                    strategy=probabilities,
                    action_values=tuple(values),
                    instantaneous_advantages=tuple(
                        (action_id, value - baseline) for action_id, value in values
                    ),
                    public_features=tuple(belief.public_features),
                    private_features=private_features,
                    action_features=action_features,
                    source_official_engine_executed=official,
                    hidden_completion_prior_official=prior_official,
                )
            )
            return baseline

        cache_key = (player, information_set.info_set_id)
        chosen = context.sampled_actions.get(cache_key)
        forced = forced_actions.get(information_set.info_set_id)
        if chosen is not None and forced is not None and chosen != forced:
            raise ValueError("forced action conflicts with the sampled pure strategy")
        if chosen is None:
            chosen = forced or _sample_identifier(probabilities, rng)
            if chosen not in actions or distribution.probability(chosen) <= 0:
                raise ValueError("forced/sampled opponent action is not in positive support")
            context.sampled_actions[cache_key] = chosen
        if cache_key not in context.recorded_strategy_sets:
            context.recorded_strategy_sets.add(cache_key)
            context.strategy_samples.append(
                ExternalSamplingStrategySample(
                    game_fingerprint=game.fingerprint,
                    player=player,
                    traverser=traverser,
                    iteration=iteration,
                    information_set_id=information_set.info_set_id,
                    public_belief_fingerprint=belief.fingerprint,
                    depth=information_set.depth,
                    strategy=probabilities,
                    source_official_engine_executed=official,
                    hidden_completion_prior_official=prior_official,
                )
            )
        child = self._transition(
            game,
            state,
            action_adapter(game, state, actions[chosen]),
            chance_outcome=None,
            context=context,
        )
        return self._visit(
            game,
            child,
            strategy_function,
            traverser=traverser,
            iteration=iteration,
            rng=rng,
            context=context,
            official=official,
            prior_official=prior_official,
            information_view=information_view,
            action_adapter=action_adapter,
            forced_chance=forced_chance,
            forced_actions=forced_actions,
        )


@dataclass(frozen=True, slots=True)
class ReservoirRecord:
    """One priority-reservoir entry with no dependence on insertion order state."""

    priority: str
    occurrence: int
    sample: ExternalSamplingAdvantageSample

    def __post_init__(self) -> None:
        _validate_digest(self.priority, "reservoir priority")
        if self.occurrence <= 0:
            raise ValueError("reservoir occurrence must be positive")


class AdvantageReservoir:
    """Deterministic hash-priority equivalent of uniform reservoir sampling.

    The SHA-256 priority is derived from a declared seed, occurrence number,
    and sample fingerprint.  Keeping the smallest priorities yields a bounded,
    reproducible uniform reservoir under the pseudorandom-hash assumption.
    Iteration weights remain sample metadata for later LCFR loss weighting;
    they do not alter admission probability.
    """

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("advantage reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.seen = 0
        self._records: list[ReservoirRecord] = []

    def add(self, sample: ExternalSamplingAdvantageSample) -> None:
        self.seen += 1
        priority = _fingerprint(
            {
                "schema": EXTERNAL_SAMPLING_SCHEMA_VERSION,
                "seed": self.seed,
                "occurrence": self.seen,
                "sample": sample.fingerprint,
            }
        )
        self._records.append(ReservoirRecord(priority, self.seen, sample))
        self._records.sort(key=lambda item: (item.priority, item.occurrence))
        if len(self._records) > self.capacity:
            self._records.pop()

    @property
    def samples(self) -> tuple[ExternalSamplingAdvantageSample, ...]:
        return tuple(record.sample for record in self._records)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "algorithm": "uniform_hash_priority_reservoir",
            "capacity": self.capacity,
            "seed": self.seed,
            "seen": self.seen,
            "records": [
                {
                    "priority": record.priority,
                    "occurrence": record.occurrence,
                    "sample": record.sample.to_dict(),
                }
                for record in self._records
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AdvantageReservoir":
        result = cls(int(raw["capacity"]), seed=int(raw["seed"]))
        result.seen = int(raw["seen"])
        result._records = [
            ReservoirRecord(
                priority=str(item["priority"]),
                occurrence=int(item["occurrence"]),
                sample=ExternalSamplingAdvantageSample.from_dict(item["sample"]),
            )
            for item in raw["records"]
        ]
        result._validate_restored()
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("advantage reservoir fingerprint mismatch")
        return result

    def _validate_restored(self) -> None:
        if self.seen < len(self._records) or len(self._records) > self.capacity:
            raise ValueError("restored advantage reservoir accounting is invalid")
        occurrences = [item.occurrence for item in self._records]
        if len(occurrences) != len(set(occurrences)) or any(
            occurrence > self.seen for occurrence in occurrences
        ):
            raise ValueError("restored advantage reservoir occurrences are invalid")
        for record in self._records:
            expected = _fingerprint(
                {
                    "schema": EXTERNAL_SAMPLING_SCHEMA_VERSION,
                    "seed": self.seed,
                    "occurrence": record.occurrence,
                    "sample": record.sample.fingerprint,
                }
            )
            if record.priority != expected:
                raise ValueError("restored advantage reservoir priority mismatch")
        self._records.sort(key=lambda item: (item.priority, item.occurrence))


@dataclass(frozen=True, slots=True)
class StrategyReservoirRecord:
    priority: str
    occurrence: int
    sample: ExternalSamplingStrategySample

    def __post_init__(self) -> None:
        _validate_digest(self.priority, "strategy reservoir priority")
        if self.occurrence <= 0:
            raise ValueError("strategy reservoir occurrence must be positive")


class StrategyReservoir:
    """Bounded uniform reservoir for the separate average-strategy memory."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("strategy reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.seen = 0
        self._records: list[StrategyReservoirRecord] = []

    def add(self, sample: ExternalSamplingStrategySample) -> None:
        self.seen += 1
        priority = _fingerprint(
            {
                "schema": EXTERNAL_SAMPLING_SCHEMA_VERSION,
                "seed": self.seed,
                "occurrence": self.seen,
                "sample": sample.fingerprint,
            }
        )
        self._records.append(StrategyReservoirRecord(priority, self.seen, sample))
        self._records.sort(key=lambda item: (item.priority, item.occurrence))
        if len(self._records) > self.capacity:
            self._records.pop()

    @property
    def samples(self) -> tuple[ExternalSamplingStrategySample, ...]:
        return tuple(record.sample for record in self._records)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "algorithm": "uniform_hash_priority_reservoir",
            "capacity": self.capacity,
            "seed": self.seed,
            "seen": self.seen,
            "records": [
                {
                    "priority": record.priority,
                    "occurrence": record.occurrence,
                    "sample": record.sample.to_dict(),
                }
                for record in self._records
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StrategyReservoir":
        result = cls(int(raw["capacity"]), seed=int(raw["seed"]))
        result.seen = int(raw["seen"])
        result._records = [
            StrategyReservoirRecord(
                priority=str(item["priority"]),
                occurrence=int(item["occurrence"]),
                sample=ExternalSamplingStrategySample.from_dict(item["sample"]),
            )
            for item in raw["records"]
        ]
        if result.seen < len(result._records) or len(result._records) > result.capacity:
            raise ValueError("restored strategy reservoir accounting is invalid")
        occurrences = [item.occurrence for item in result._records]
        if len(occurrences) != len(set(occurrences)) or any(
            occurrence > result.seen for occurrence in occurrences
        ):
            raise ValueError("restored strategy reservoir occurrences are invalid")
        for record in result._records:
            expected_priority = _fingerprint(
                {
                    "schema": EXTERNAL_SAMPLING_SCHEMA_VERSION,
                    "seed": result.seed,
                    "occurrence": record.occurrence,
                    "sample": record.sample.fingerprint,
                }
            )
            if record.priority != expected_priority:
                raise ValueError("restored strategy reservoir priority mismatch")
        result._records.sort(key=lambda item: (item.priority, item.occurrence))
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("strategy reservoir fingerprint mismatch")
        return result
