"""Bounded outcome-sampling MCCFR for large information-set games.

Outcome sampling follows one complete terminal history per traverser instead of
expanding every traverser action.  The recursive estimator here mirrors the
zero-baseline formulation used by OpenSpiel: the traverser's sampling policy is
an epsilon-uniform mixture with the current strategy, while chance and the
opponent are sampled from their current distributions.  Importance correction
is applied through prefix sampling reach and recursively corrected suffix
values.

The result is an unbiased *sampled counterfactual-regret estimate* when the
finite extensive-form assumptions hold.  It is not an all-action conditional
advantage, a calibrated counterfactual-value bound, a safety certificate, or a
claim that a neural/full-game strategy has converged.  Traversal limits fail
closed: an exception is raised before any partial sample can be returned.
"""

from __future__ import annotations

from dataclasses import dataclass
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


OUTCOME_SAMPLING_SCHEMA_VERSION = 1


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
ActionAdapter = Callable[[AbstractInformationSetGame, AbstractGameState, Any], Any]
BaselineFunction = Callable[
    [InformationSet, PublicBeliefState, int, int],
    Mapping[str, float],
]


@dataclass(frozen=True, slots=True)
class OutcomeSamplingConfig:
    """Exploration and hard traversal limits for one complete sampled path."""

    exploration: float = 0.6
    maximum_transitions: int = 4_096
    maximum_depth: int = 4_096

    def __post_init__(self) -> None:
        if not 0 < self.exploration <= 1 or not math.isfinite(self.exploration):
            raise ValueError("outcome-sampling exploration must be in (0, 1]")
        if self.maximum_transitions <= 0 or self.maximum_depth <= 0:
            raise ValueError("outcome-sampling limits must be positive")


class OutcomeSamplingBudgetExceeded(RuntimeError):
    """Raised before a capped trajectory can be exposed as training data."""


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


def _features(
    information_set: InformationSet,
) -> tuple[
    tuple[tuple[str, float], ...],
    tuple[tuple[str, tuple[tuple[str, float], ...]], ...],
]:
    return (
        tuple((name, float(value)) for name, value in information_set.private_features),
        tuple(
            (
                action.action_id,
                tuple((name, float(value)) for name, value in action.features),
            )
            for action in information_set.legal_actions
        ),
    )


def _validate_distribution(
    values: tuple[tuple[str, float], ...],
    identifiers: tuple[str, ...],
    field_name: str,
) -> None:
    if tuple(identifier for identifier, _ in values) != identifiers:
        raise ValueError(f"{field_name} support/order mismatch")
    probabilities = tuple(_finite(value, field_name) for _, value in values)
    if any(value < 0 for value in probabilities) or not math.isclose(
        sum(probabilities), 1.0, abs_tol=1e-8
    ):
        raise ValueError(f"{field_name} must be a probability distribution")


@dataclass(frozen=True, slots=True)
class OutcomeSamplingRegretSample:
    """One importance-weighted sampled counterfactual-regret vector."""

    game_fingerprint: str
    traverser: int
    iteration: int
    information_set_id: str
    public_belief_fingerprint: str
    private_observation_fingerprint: str | None
    depth: int
    current_strategy: tuple[tuple[str, float], ...]
    sampling_strategy: tuple[tuple[str, float], ...]
    sampled_action_id: str
    sampled_action_probability: float
    player_reach: float
    opponent_chance_reach: float
    sampling_reach: float
    action_baselines: tuple[tuple[str, float], ...]
    estimated_action_values: tuple[tuple[str, float], ...]
    estimated_policy_value: float
    sampled_regrets: tuple[tuple[str, float], ...]
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
            raise ValueError("outcome-sampling regret metadata is invalid")
        if not self.information_set_id:
            raise ValueError("outcome-sampling regret requires an information-set ID")
        identifiers = tuple(identifier for identifier, _ in self.current_strategy)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("outcome-sampling action support must be unique and non-empty")
        _validate_distribution(self.current_strategy, identifiers, "current_strategy")
        _validate_distribution(self.sampling_strategy, identifiers, "sampling_strategy")
        if self.sampled_action_id not in identifiers:
            raise ValueError("sampled action is outside outcome-sampling support")
        expected_probability = dict(self.sampling_strategy)[self.sampled_action_id]
        if not math.isclose(
            _finite(self.sampled_action_probability, "sampled_action_probability"),
            expected_probability,
            abs_tol=1e-12,
        ) or expected_probability <= 0:
            raise ValueError("sampled action probability is inconsistent")
        for field_name in (
            "player_reach",
            "opponent_chance_reach",
            "sampling_reach",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if not 0 <= value <= 1 + 1e-10:
                raise ValueError(f"{field_name} must be a reach probability")
        if self.sampling_reach <= 0:
            raise ValueError("outcome-sampling prefix reach must be positive")
        for field_name in (
            "action_baselines",
            "estimated_action_values",
            "sampled_regrets",
        ):
            values = getattr(self, field_name)
            if tuple(identifier for identifier, _ in values) != identifiers:
                raise ValueError(f"{field_name} support/order mismatch")
            for _, value in values:
                _finite(value, field_name)
        _finite(self.estimated_policy_value, "estimated_policy_value")
        if tuple(identifier for identifier, _ in self.action_features) != identifiers:
            raise ValueError("outcome-sampling action-feature support mismatch")
        for name, value in self.public_features + self.private_features:
            if not name:
                raise ValueError("outcome-sampling feature names must be non-empty")
            _finite(value, f"feature {name}")
        for _, features in self.action_features:
            for name, value in features:
                if not name:
                    raise ValueError("outcome-sampling action feature names must be non-empty")
                _finite(value, f"action feature {name}")
        if self.hidden_completion_prior_official and not self.source_official_engine_executed:
            raise ValueError("official hidden prior requires official source execution")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": OUTCOME_SAMPLING_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "traverser": self.traverser,
            "iteration": self.iteration,
            "information_set_id": self.information_set_id,
            "public_belief_fingerprint": self.public_belief_fingerprint,
            "private_observation_fingerprint": self.private_observation_fingerprint,
            "depth": self.depth,
            "current_strategy": [list(item) for item in self.current_strategy],
            "sampling_strategy": [list(item) for item in self.sampling_strategy],
            "sampled_action_id": self.sampled_action_id,
            "sampled_action_probability": self.sampled_action_probability,
            "reaches": {
                "player": self.player_reach,
                "opponent_and_chance": self.opponent_chance_reach,
                "sampling": self.sampling_reach,
            },
            "action_baselines": [list(item) for item in self.action_baselines],
            "estimated_action_values": [
                list(item) for item in self.estimated_action_values
            ],
            "estimated_policy_value": self.estimated_policy_value,
            "sampled_regrets": [list(item) for item in self.sampled_regrets],
            "public_features": [list(item) for item in self.public_features],
            "private_features": [list(item) for item in self.private_features],
            "action_features": [
                [action_id, [list(item) for item in features]]
                for action_id, features in self.action_features
            ],
            "semantics": {
                "outcome_sampling_importance_corrected": True,
                "pre_action_baseline_corrected": True,
                "all_actions_expanded": False,
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "OutcomeSamplingRegretSample":
        reaches = raw["reaches"]
        provenance = raw["provenance"]

        def pairs(name: str) -> tuple[tuple[str, float], ...]:
            return tuple((str(key), float(value)) for key, value in raw[name])

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
            current_strategy=pairs("current_strategy"),
            sampling_strategy=pairs("sampling_strategy"),
            sampled_action_id=str(raw["sampled_action_id"]),
            sampled_action_probability=float(raw["sampled_action_probability"]),
            player_reach=float(reaches["player"]),
            opponent_chance_reach=float(reaches["opponent_and_chance"]),
            sampling_reach=float(reaches["sampling"]),
            action_baselines=pairs("action_baselines"),
            estimated_action_values=pairs("estimated_action_values"),
            estimated_policy_value=float(raw["estimated_policy_value"]),
            sampled_regrets=pairs("sampled_regrets"),
            public_features=pairs("public_features"),
            private_features=pairs("private_features"),
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
            raise ValueError("outcome-sampling regret fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OutcomeSamplingStrategySample:
    """One stochastic average-strategy contribution for the traverser."""

    game_fingerprint: str
    player: int
    iteration: int
    information_set_id: str
    public_belief_fingerprint: str
    private_observation_fingerprint: str | None
    depth: int
    strategy: tuple[tuple[str, float], ...]
    importance_weight: float
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
        if self.player not in (0, 1) or self.iteration <= 0 or self.depth < 0:
            raise ValueError("outcome-sampling strategy metadata is invalid")
        if not self.information_set_id:
            raise ValueError("outcome-sampling strategy requires an information-set ID")
        identifiers = tuple(identifier for identifier, _ in self.strategy)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("outcome-sampling strategy support must be unique")
        _validate_distribution(self.strategy, identifiers, "strategy")
        if _finite(self.importance_weight, "importance_weight") < 0:
            raise ValueError("outcome-sampling strategy weight cannot be negative")
        if tuple(identifier for identifier, _ in self.action_features) != identifiers:
            raise ValueError("outcome-sampling strategy action-feature support mismatch")
        if self.hidden_completion_prior_official and not self.source_official_engine_executed:
            raise ValueError("official hidden prior requires official source execution")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": OUTCOME_SAMPLING_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "player": self.player,
            "iteration": self.iteration,
            "information_set_id": self.information_set_id,
            "public_belief_fingerprint": self.public_belief_fingerprint,
            "private_observation_fingerprint": self.private_observation_fingerprint,
            "depth": self.depth,
            "strategy": [list(item) for item in self.strategy],
            "importance_weight": self.importance_weight,
            "public_features": [list(item) for item in self.public_features],
            "private_features": [list(item) for item in self.private_features],
            "action_features": [
                [action_id, [list(item) for item in features]]
                for action_id, features in self.action_features
            ],
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "OutcomeSamplingStrategySample":
        provenance = raw["provenance"]

        def pairs(name: str) -> tuple[tuple[str, float], ...]:
            return tuple((str(key), float(value)) for key, value in raw[name])

        result = cls(
            game_fingerprint=str(raw["game_fingerprint"]),
            player=int(raw["player"]),
            iteration=int(raw["iteration"]),
            information_set_id=str(raw["information_set_id"]),
            public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
            private_observation_fingerprint=(
                None
                if raw.get("private_observation_fingerprint") is None
                else str(raw["private_observation_fingerprint"])
            ),
            depth=int(raw["depth"]),
            strategy=pairs("strategy"),
            importance_weight=float(raw["importance_weight"]),
            public_features=pairs("public_features"),
            private_features=pairs("private_features"),
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
            raise ValueError("outcome-sampling strategy fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OutcomeSamplingTraversalResult:
    traverser: int
    iteration: int
    terminal_utility: float
    estimated_root_value: float
    transitions: int
    maximum_depth_reached: int
    decision_nodes: int
    chance_nodes: int
    regret_samples: tuple[OutcomeSamplingRegretSample, ...]
    strategy_samples: tuple[OutcomeSamplingStrategySample, ...]
    source_official_engine_executed: bool
    hidden_completion_prior_official: bool
    complete: bool = True

    def __post_init__(self) -> None:
        if self.traverser not in (0, 1) or self.iteration <= 0:
            raise ValueError("outcome-sampling traversal metadata is invalid")
        _finite(self.terminal_utility, "terminal_utility")
        _finite(self.estimated_root_value, "estimated_root_value")
        if min(self.transitions, self.maximum_depth_reached, self.decision_nodes) <= 0:
            raise ValueError("complete outcome-sampling traversal accounting is invalid")
        if self.chance_nodes < 0 or not self.regret_samples or not self.strategy_samples:
            raise ValueError("complete traversal requires traverser samples")
        if any(item.traverser != self.traverser for item in self.regret_samples):
            raise ValueError("outcome-sampling regret sample has wrong traverser")
        if any(item.player != self.traverser for item in self.strategy_samples):
            raise ValueError("outcome-sampling strategy sample has wrong player")
        if not self.complete:
            raise ValueError("partial outcome-sampling traversals cannot materialize")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": OUTCOME_SAMPLING_SCHEMA_VERSION,
            "algorithm": "outcome_sampling_mccfr_zero_baseline",
            "traverser": self.traverser,
            "iteration": self.iteration,
            "terminal_utility": self.terminal_utility,
            "estimated_root_value": self.estimated_root_value,
            "transitions": self.transitions,
            "maximum_depth_reached": self.maximum_depth_reached,
            "decision_nodes": self.decision_nodes,
            "chance_nodes": self.chance_nodes,
            "regret_samples": [item.to_dict() for item in self.regret_samples],
            "strategy_samples": [item.to_dict() for item in self.strategy_samples],
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "selected_hidden_truth_persisted": False,
            },
            "claims": {
                "complete_terminal_history_within_budget": self.complete,
                "unbiased_estimator_requires_extensive_form_assumptions": True,
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
    maximum_depth_reached: int
    decision_nodes: int
    chance_nodes: int
    terminal_utility: float | None
    regret_samples: list[OutcomeSamplingRegretSample]
    strategy_samples: list[OutcomeSamplingStrategySample]


class OutcomeSamplingTraversal:
    """One fail-closed outcome-sampling trajectory for one traverser."""

    def __init__(self, config: OutcomeSamplingConfig | None = None) -> None:
        self.config = config or OutcomeSamplingConfig()

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
        baseline: BaselineFunction | None = None,
        forced_chance: Mapping[str, str] | None = None,
        forced_actions: Mapping[str, str] | None = None,
    ) -> OutcomeSamplingTraversalResult:
        if traverser not in (0, 1) or iteration <= 0:
            raise ValueError("outcome-sampling traverser/iteration is invalid")
        context = _TraversalContext(0, 0, 0, 0, None, [], [])
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
        baseline_function = baseline or (
            lambda information_set, belief, value_player, acting_player: {
                action.action_id: 0.0 for action in information_set.legal_actions
            }
        )
        estimated = self._visit(
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
            baseline=baseline_function,
            forced_chance=dict(forced_chance or {}),
            forced_actions=dict(forced_actions or {}),
            player_reach=1.0,
            opponent_chance_reach=1.0,
            sampling_reach=1.0,
        )
        if context.terminal_utility is None:
            raise RuntimeError("outcome-sampling traversal did not reach a terminal state")
        return OutcomeSamplingTraversalResult(
            traverser=traverser,
            iteration=iteration,
            terminal_utility=context.terminal_utility,
            estimated_root_value=estimated,
            transitions=context.transitions,
            maximum_depth_reached=context.maximum_depth_reached,
            decision_nodes=context.decision_nodes,
            chance_nodes=context.chance_nodes,
            regret_samples=tuple(context.regret_samples),
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
            raise OutcomeSamplingBudgetExceeded(
                "outcome-sampling transition budget exhausted; partial samples discarded"
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
        baseline: BaselineFunction,
        forced_chance: Mapping[str, str],
        forced_actions: Mapping[str, str],
        player_reach: float,
        opponent_chance_reach: float,
        sampling_reach: float,
    ) -> float:
        context.maximum_depth_reached = max(context.maximum_depth_reached, state.depth)
        if state.depth > self.config.maximum_depth:
            raise OutcomeSamplingBudgetExceeded(
                "outcome-sampling depth budget exhausted; partial samples discarded"
            )
        if state.terminal:
            utility = float(game.terminal_utility(state, traverser))
            context.terminal_utility = utility
            return utility
        if state.actor is InformationSetActor.CHANCE:
            context.chance_nodes += 1
            outcomes = game.chance_outcomes(state)
            if not outcomes or not math.isclose(
                sum(item.probability for item in outcomes), 1.0, abs_tol=1e-8
            ):
                raise ValueError("chance outcomes must form a non-empty distribution")
            forced = forced_chance.get(state.state_id)
            selected = forced or _sample_identifier(
                tuple((item.outcome_id, item.probability) for item in outcomes), rng
            )
            matches = [item for item in outcomes if item.outcome_id == selected]
            if len(matches) != 1 or matches[0].probability <= 0:
                raise ValueError("forced/sampled chance outcome is not in positive support")
            probability = matches[0].probability
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
                baseline=baseline,
                forced_chance=forced_chance,
                forced_actions=forced_actions,
                player_reach=player_reach,
                opponent_chance_reach=opponent_chance_reach * probability,
                sampling_reach=sampling_reach * probability,
            )

        player = game.player_to_act(state)
        if player not in (0, 1):
            raise ValueError("nonterminal non-chance state requires player 0 or 1")
        context.decision_nodes += 1
        information_set, belief = information_view(game, state, player)
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("outcome-sampling information view has mismatched belief")
        distribution = strategy_function(information_set, belief)
        distribution.validate_support(information_set)
        identifiers = tuple(action.action_id for action in information_set.legal_actions)
        current = tuple(
            (identifier, distribution.probability(identifier))
            for identifier in identifiers
        )
        raw_baselines = baseline(information_set, belief, traverser, player)
        if set(raw_baselines) != set(identifiers):
            raise ValueError("outcome-sampling baseline support must match legal actions")
        action_baselines = tuple(
            (identifier, _finite(raw_baselines[identifier], "action baseline"))
            for identifier in identifiers
        )
        if player == traverser:
            uniform = 1.0 / len(identifiers)
            sampling = tuple(
                (
                    identifier,
                    (1.0 - self.config.exploration) * probability
                    + self.config.exploration * uniform,
                )
                for identifier, probability in current
            )
        else:
            sampling = current
        forced = forced_actions.get(information_set.info_set_id)
        selected = forced or _sample_identifier(sampling, rng)
        actions = {
            action.action_id: action for action in information_set.legal_actions
        }
        sampling_probability = dict(sampling).get(selected, 0.0)
        if selected not in actions or sampling_probability <= 0:
            raise ValueError("forced/sampled action is not in positive sampling support")
        current_probability = distribution.probability(selected)
        child = self._transition(
            game,
            state,
            action_adapter(game, state, actions[selected]),
            chance_outcome=None,
            context=context,
        )
        child_value = self._visit(
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
            baseline=baseline,
            forced_chance=forced_chance,
            forced_actions=forced_actions,
            player_reach=(
                player_reach * current_probability
                if player == traverser
                else player_reach
            ),
            opponent_chance_reach=(
                opponent_chance_reach
                if player == traverser
                else opponent_chance_reach * current_probability
            ),
            sampling_reach=sampling_reach * sampling_probability,
        )
        baseline_by_action = dict(action_baselines)
        selected_baseline = baseline_by_action[selected]
        child_values = tuple(
            (
                identifier,
                (
                    baseline_by_action[identifier]
                    + (child_value - selected_baseline) / sampling_probability
                    if identifier == selected
                    else baseline_by_action[identifier]
                ),
            )
            for identifier in identifiers
        )
        value_estimate = sum(
            distribution.probability(identifier) * value
            for identifier, value in child_values
        )

        if player == traverser:
            reach_ratio = opponent_chance_reach / sampling_reach
            estimated_action_values = tuple(
                (identifier, value * reach_ratio)
                for identifier, value in child_values
            )
            estimated_policy_value = value_estimate * reach_ratio
            private_features, action_features = _features(information_set)
            context.regret_samples.append(
                OutcomeSamplingRegretSample(
                    game_fingerprint=game.fingerprint,
                    traverser=traverser,
                    iteration=iteration,
                    information_set_id=information_set.info_set_id,
                    public_belief_fingerprint=belief.fingerprint,
                    private_observation_fingerprint=(
                        information_set.private_observation_fingerprint
                    ),
                    depth=information_set.depth,
                    current_strategy=current,
                    sampling_strategy=sampling,
                    sampled_action_id=selected,
                    sampled_action_probability=sampling_probability,
                    player_reach=player_reach,
                    opponent_chance_reach=opponent_chance_reach,
                    sampling_reach=sampling_reach,
                    action_baselines=action_baselines,
                    estimated_action_values=estimated_action_values,
                    estimated_policy_value=estimated_policy_value,
                    sampled_regrets=tuple(
                        (identifier, value - estimated_policy_value)
                        for identifier, value in estimated_action_values
                    ),
                    public_features=tuple(belief.public_features),
                    private_features=private_features,
                    action_features=action_features,
                    source_official_engine_executed=official,
                    hidden_completion_prior_official=prior_official,
                )
            )
            context.strategy_samples.append(
                OutcomeSamplingStrategySample(
                    game_fingerprint=game.fingerprint,
                    player=traverser,
                    iteration=iteration,
                    information_set_id=information_set.info_set_id,
                    public_belief_fingerprint=belief.fingerprint,
                    private_observation_fingerprint=(
                        information_set.private_observation_fingerprint
                    ),
                    depth=information_set.depth,
                    strategy=current,
                    importance_weight=player_reach / sampling_reach,
                    public_features=tuple(belief.public_features),
                    private_features=private_features,
                    action_features=action_features,
                    source_official_engine_executed=official,
                    hidden_completion_prior_official=prior_official,
                )
            )
        return value_estimate


@dataclass(frozen=True, slots=True)
class OutcomeRegretReservoirRecord:
    priority: str
    occurrence: int
    sample: OutcomeSamplingRegretSample


class OutcomeRegretReservoir:
    """Deterministic hash-priority uniform reservoir for regret estimates."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("outcome regret reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.seen = 0
        self._records: list[OutcomeRegretReservoirRecord] = []

    def add(self, sample: OutcomeSamplingRegretSample) -> None:
        self.seen += 1
        priority = _fingerprint(
            {
                "schema": OUTCOME_SAMPLING_SCHEMA_VERSION,
                "seed": self.seed,
                "occurrence": self.seen,
                "sample": sample.fingerprint,
            }
        )
        self._records.append(OutcomeRegretReservoirRecord(priority, self.seen, sample))
        self._records.sort(key=lambda item: (item.priority, item.occurrence))
        if len(self._records) > self.capacity:
            self._records.pop()

    @property
    def samples(self) -> tuple[OutcomeSamplingRegretSample, ...]:
        return tuple(item.sample for item in self._records)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": OUTCOME_SAMPLING_SCHEMA_VERSION,
            "algorithm": "uniform_hash_priority_reservoir",
            "capacity": self.capacity,
            "seed": self.seed,
            "seen": self.seen,
            "records": [
                {
                    "priority": item.priority,
                    "occurrence": item.occurrence,
                    "sample": item.sample.to_dict(),
                }
                for item in self._records
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OutcomeRegretReservoir":
        result = cls(int(raw["capacity"]), seed=int(raw["seed"]))
        result.seen = int(raw["seen"])
        result._records = [
            OutcomeRegretReservoirRecord(
                priority=str(item["priority"]),
                occurrence=int(item["occurrence"]),
                sample=OutcomeSamplingRegretSample.from_dict(item["sample"]),
            )
            for item in raw["records"]
        ]
        result._validate_restored()
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("outcome regret reservoir fingerprint mismatch")
        return result

    def _validate_restored(self) -> None:
        if self.seen < len(self._records) or len(self._records) > self.capacity:
            raise ValueError("restored outcome regret reservoir accounting is invalid")
        occurrences = [item.occurrence for item in self._records]
        if len(occurrences) != len(set(occurrences)) or any(
            value <= 0 or value > self.seen for value in occurrences
        ):
            raise ValueError("restored outcome regret occurrences are invalid")
        for item in self._records:
            expected = _fingerprint(
                {
                    "schema": OUTCOME_SAMPLING_SCHEMA_VERSION,
                    "seed": self.seed,
                    "occurrence": item.occurrence,
                    "sample": item.sample.fingerprint,
                }
            )
            if item.priority != expected:
                raise ValueError("restored outcome regret priority mismatch")
        self._records.sort(key=lambda item: (item.priority, item.occurrence))


@dataclass(frozen=True, slots=True)
class OutcomeStrategyReservoirRecord:
    priority: str
    occurrence: int
    sample: OutcomeSamplingStrategySample


class OutcomeStrategyReservoir:
    """Separate bounded reservoir for stochastic average-strategy samples."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("outcome strategy reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.seen = 0
        self._records: list[OutcomeStrategyReservoirRecord] = []

    def add(self, sample: OutcomeSamplingStrategySample) -> None:
        self.seen += 1
        priority = _fingerprint(
            {
                "schema": OUTCOME_SAMPLING_SCHEMA_VERSION,
                "seed": self.seed,
                "occurrence": self.seen,
                "sample": sample.fingerprint,
            }
        )
        self._records.append(OutcomeStrategyReservoirRecord(priority, self.seen, sample))
        self._records.sort(key=lambda item: (item.priority, item.occurrence))
        if len(self._records) > self.capacity:
            self._records.pop()

    @property
    def samples(self) -> tuple[OutcomeSamplingStrategySample, ...]:
        return tuple(item.sample for item in self._records)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": OUTCOME_SAMPLING_SCHEMA_VERSION,
            "algorithm": "uniform_hash_priority_reservoir",
            "capacity": self.capacity,
            "seed": self.seed,
            "seen": self.seen,
            "records": [
                {
                    "priority": item.priority,
                    "occurrence": item.occurrence,
                    "sample": item.sample.to_dict(),
                }
                for item in self._records
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OutcomeStrategyReservoir":
        result = cls(int(raw["capacity"]), seed=int(raw["seed"]))
        result.seen = int(raw["seen"])
        result._records = [
            OutcomeStrategyReservoirRecord(
                priority=str(item["priority"]),
                occurrence=int(item["occurrence"]),
                sample=OutcomeSamplingStrategySample.from_dict(item["sample"]),
            )
            for item in raw["records"]
        ]
        if result.seen < len(result._records) or len(result._records) > result.capacity:
            raise ValueError("restored outcome strategy reservoir accounting is invalid")
        occurrences = [item.occurrence for item in result._records]
        if len(occurrences) != len(set(occurrences)) or any(
            value <= 0 or value > result.seen for value in occurrences
        ):
            raise ValueError("restored outcome strategy occurrences are invalid")
        for item in result._records:
            expected_priority = _fingerprint(
                {
                    "schema": OUTCOME_SAMPLING_SCHEMA_VERSION,
                    "seed": result.seed,
                    "occurrence": item.occurrence,
                    "sample": item.sample.fingerprint,
                }
            )
            if item.priority != expected_priority:
                raise ValueError("restored outcome strategy priority mismatch")
        result._records.sort(key=lambda item: (item.priority, item.occurrence))
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("outcome strategy reservoir fingerprint mismatch")
        return result
