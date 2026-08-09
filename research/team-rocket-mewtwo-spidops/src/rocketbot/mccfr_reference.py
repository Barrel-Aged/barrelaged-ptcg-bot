"""Exact Phase 6L validation for external-sampling MCCFR.

``RocketPrizeDuel`` is small enough to enumerate every chance outcome and every
opponent pure-strategy sample.  This lets us verify, to floating-point
precision, that :mod:`rocketbot.external_sampling` produces the conditional
advantages specified by external-sampling MCCFR before applying the traversal
to the much larger native game.

The convergence result in this module is only for the abstract microgame.  It
is not evidence of Pokemon TCG strength or native full-game convergence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
import math
import random
from typing import Any, Mapping

from .exact_game import (
    CFRTrainingResult,
    DuelState,
    RocketPrizeDuel,
    TabularStrategyProfile,
    evaluate_profile_from_state,
    exact_profile_metrics,
)
from .external_sampling import (
    EXTERNAL_SAMPLING_SCHEMA_VERSION,
    ExternalSamplingConfig,
    ExternalSamplingTraversal,
)
from .game_theory import ActionDistribution, InformationSet, PublicBeliefState


MCCFR_REFERENCE_SCHEMA_VERSION = 1
MCCFR_PRIMARY_SOURCES = (
    "https://proceedings.neurips.cc/paper_files/paper/2009/file/"
    "00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf",
    "https://webdocs.cs.ualberta.ca/~bowling/papers/09nips-mccfr-appendix.pdf",
    "https://proceedings.mlr.press/v97/brown19b/brown19b.pdf",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _seed(*items: Any) -> int:
    return int.from_bytes(sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class ExternalSamplingExpectationRow:
    traverser: int
    information_set_id: str
    action_id: str
    visit_probability: float
    exact_conditional_advantage: float
    enumerated_sample_advantage: float
    absolute_error: float

    def __post_init__(self) -> None:
        if self.traverser not in (0, 1) or not self.information_set_id or not self.action_id:
            raise ValueError("external-sampling audit row identity is invalid")
        for field_name in (
            "visit_probability",
            "exact_conditional_advantage",
            "enumerated_sample_advantage",
            "absolute_error",
        ):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if not 0 < self.visit_probability <= 1 + 1e-10 or self.absolute_error < 0:
            raise ValueError("external-sampling audit probability/error is invalid")


@dataclass(frozen=True, slots=True)
class ExternalSamplingExpectationAudit:
    game_fingerprint: str
    profile_fingerprint: str
    block_count_player_0: int
    block_count_player_1: int
    rows: tuple[ExternalSamplingExpectationRow, ...]
    maximum_absolute_error: float
    primary_sources: tuple[str, ...] = MCCFR_PRIMARY_SOURCES
    official_engine_executed: bool = False
    full_game_equilibrium_convergence: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.game_fingerprint, "game_fingerprint")
        _validate_digest(self.profile_fingerprint, "profile_fingerprint")
        if self.block_count_player_0 <= 0 or self.block_count_player_1 <= 0:
            raise ValueError("external-sampling expectation audit requires sampled blocks")
        if not self.rows or not math.isfinite(self.maximum_absolute_error):
            raise ValueError("external-sampling expectation audit is empty/non-finite")
        if self.maximum_absolute_error < 0:
            raise ValueError("external-sampling expectation error cannot be negative")
        if self.official_engine_executed or self.full_game_equilibrium_convergence:
            raise ValueError("microgame audit cannot promote native/equilibrium claims")
        if tuple(self.primary_sources) != MCCFR_PRIMARY_SOURCES:
            raise ValueError("external-sampling audit must bind the reviewed primary sources")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": MCCFR_REFERENCE_SCHEMA_VERSION,
            "external_sampling_schema_version": EXTERNAL_SAMPLING_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "block_count_player_0": self.block_count_player_0,
            "block_count_player_1": self.block_count_player_1,
            "rows": [asdict(row) for row in self.rows],
            "maximum_absolute_error": self.maximum_absolute_error,
            "primary_sources": list(self.primary_sources),
            "claims": {
                "exact_expectation_checked_on_abstract_microgame": True,
                "official_engine_executed": self.official_engine_executed,
                "full_game_equilibrium_convergence": (
                    self.full_game_equilibrium_convergence
                ),
                "pokemon_strength": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result


def _state_reaches(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}

    def visit(
        state: DuelState,
        reach_player_0: float,
        reach_player_1: float,
        chance_reach: float,
    ) -> None:
        reaches = (reach_player_0, reach_player_1, chance_reach)
        existing = result.get(state.state_id)
        if existing is not None:
            if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(existing, reaches)):
                raise RuntimeError("duel state has inconsistent unique-history reaches")
            return
        result[state.state_id] = reaches
        if state.terminal:
            return
        if state.cards == (None, None):
            for outcome in game.chance_outcomes(state):
                visit(
                    game.transition(state, None, chance_outcome=outcome),
                    reach_player_0,
                    reach_player_1,
                    chance_reach * outcome.probability,
                )
            return
        player = game.player_to_act(state)
        assert player is not None
        information_set = game.information_set(state, player=player)
        strategy = profile.strategy(information_set)
        for action in information_set.legal_actions:
            probability = strategy.probability(action.action_id)
            visit(
                game.transition(state, action),
                reach_player_0 * probability if player == 0 else reach_player_0,
                reach_player_1 * probability if player == 1 else reach_player_1,
                chance_reach,
            )

    visit(game.root_state(), 1.0, 1.0, 1.0)
    return result


def exact_conditional_advantages(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, float],
]:
    """Return ``regret / opponent+chance reach`` at every information set."""

    reaches = _state_reaches(game, profile)
    advantages: dict[str, dict[str, float]] = {}
    reach_masses: dict[str, float] = {}
    for information_set in game.information_sets():
        states = game.states_for_information_set(information_set.info_set_id)
        player = game.player_to_act(states[0])
        assert player is not None
        strategy = profile.strategy(information_set)
        totals = {action.action_id: 0.0 for action in information_set.legal_actions}
        mass = 0.0
        for state in states:
            reach_player_0, reach_player_1, chance_reach = reaches[state.state_id]
            counterfactual_reach = chance_reach * (
                reach_player_1 if player == 0 else reach_player_0
            )
            values = {
                action.action_id: evaluate_profile_from_state(
                    game,
                    profile,
                    game.transition(state, action),
                    player=player,
                )
                for action in information_set.legal_actions
            }
            baseline = sum(
                strategy.probability(action_id) * value
                for action_id, value in values.items()
            )
            for action_id, value in values.items():
                totals[action_id] += counterfactual_reach * (value - baseline)
            mass += counterfactual_reach
        if mass <= 0:
            raise RuntimeError("exact conditional advantage has zero counterfactual reach")
        advantages[information_set.info_set_id] = {
            action_id: value / mass for action_id, value in totals.items()
        }
        reach_masses[information_set.info_set_id] = mass
    return advantages, reach_masses


def _positive_pure_assignments(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
    *,
    player: int,
) -> tuple[tuple[dict[str, str], float], ...]:
    information_sets = tuple(
        item
        for item in game.information_sets()
        if item.info_set_id.startswith(f"p{player}:")
    )
    choices = []
    for information_set in information_sets:
        distribution = profile.strategy(information_set)
        supported = tuple(
            (action.action_id, distribution.probability(action.action_id))
            for action in information_set.legal_actions
            if distribution.probability(action.action_id) > 0
        )
        choices.append(supported)
    result = []
    for selected in product(*choices):
        assignment = {
            information_set.info_set_id: action_id
            for information_set, (action_id, _) in zip(
                information_sets, selected, strict=True
            )
        }
        probability = math.prod(probability for _, probability in selected)
        result.append((assignment, probability))
    return tuple(result)


def audit_external_sampling_expectation(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> ExternalSamplingExpectationAudit:
    """Exhaust every external-sampling block and compare with exact values."""

    exact, _ = exact_conditional_advantages(game, profile)
    weighted: dict[str, dict[str, float]] = {
        info_id: {action_id: 0.0 for action_id in values}
        for info_id, values in exact.items()
    }
    visit_probability = {info_id: 0.0 for info_id in exact}
    block_counts = [0, 0]
    root = game.root_state()
    traversal = ExternalSamplingTraversal(
        ExternalSamplingConfig(maximum_transitions=1_000, maximum_depth=32)
    )

    def strategy(
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("profile adapter received mismatched belief")
        return profile.strategy(information_set)

    for traverser in (0, 1):
        assignments = _positive_pure_assignments(
            game, profile, player=1 - traverser
        )
        for outcome in game.chance_outcomes(root):
            for forced_actions, assignment_probability in assignments:
                block_probability = outcome.probability * assignment_probability
                if block_probability <= 0:
                    continue
                result = traversal.run(
                    game,
                    strategy,
                    traverser=traverser,
                    iteration=1,
                    rng=random.Random(0),
                    forced_chance={root.state_id: outcome.outcome_id},
                    forced_actions=forced_actions,
                )
                block_counts[traverser] += 1
                seen: set[str] = set()
                for sample in result.advantage_samples:
                    if sample.information_set_id in seen:
                        raise RuntimeError("one external block revisited a traverser information set")
                    seen.add(sample.information_set_id)
                    visit_probability[sample.information_set_id] += block_probability
                    for action_id, value in sample.instantaneous_advantages:
                        weighted[sample.information_set_id][action_id] += (
                            block_probability * value
                        )

    rows = []
    for information_set in game.information_sets():
        info_id = information_set.info_set_id
        player = game.player_to_act(game.states_for_information_set(info_id)[0])
        assert player is not None
        mass = visit_probability[info_id]
        if mass <= 0:
            raise RuntimeError("external-sampling enumeration never visited an information set")
        for action in information_set.legal_actions:
            exact_value = exact[info_id][action.action_id]
            sampled_value = weighted[info_id][action.action_id] / mass
            rows.append(
                ExternalSamplingExpectationRow(
                    traverser=player,
                    information_set_id=info_id,
                    action_id=action.action_id,
                    visit_probability=mass,
                    exact_conditional_advantage=exact_value,
                    enumerated_sample_advantage=sampled_value,
                    absolute_error=abs(exact_value - sampled_value),
                )
            )
    ordered = tuple(
        sorted(rows, key=lambda item: (item.traverser, item.information_set_id, item.action_id))
    )
    return ExternalSamplingExpectationAudit(
        game_fingerprint=game.fingerprint,
        profile_fingerprint=profile.fingerprint,
        block_count_player_0=block_counts[0],
        block_count_player_1=block_counts[1],
        rows=ordered,
        maximum_absolute_error=max(item.absolute_error for item in ordered),
    )


@dataclass(frozen=True, slots=True)
class ExternalSamplingTrainingResult:
    iterations: int
    traversals: int
    transitions: int
    advantage_samples: int
    strategy_samples: int
    profile: TabularStrategyProfile
    maximum_positive_regret: float
    solver_fingerprint: str
    official_engine_executed: bool = False
    full_game_equilibrium_convergence: bool = False

    def __post_init__(self) -> None:
        if min(
            self.iterations,
            self.traversals,
            self.transitions,
            self.advantage_samples,
            self.strategy_samples,
        ) <= 0:
            raise ValueError("external-sampling training accounting must be positive")
        if self.traversals != 2 * self.iterations:
            raise ValueError("reference solver requires one traversal per player/iteration")
        if self.maximum_positive_regret < 0 or not math.isfinite(
            self.maximum_positive_regret
        ):
            raise ValueError("external-sampling maximum regret is invalid")
        _validate_digest(self.solver_fingerprint, "solver_fingerprint")
        if self.official_engine_executed or self.full_game_equilibrium_convergence:
            raise ValueError("reference solver cannot promote native/full-game claims")

    @property
    def metrics(self):
        return exact_profile_metrics(RocketPrizeDuel(), self.profile)

    def to_dict(self) -> dict[str, Any]:
        metrics = self.metrics
        return {
            "schema_version": MCCFR_REFERENCE_SCHEMA_VERSION,
            "algorithm": "tabular_external_sampling_mccfr_reference",
            "iterations": self.iterations,
            "traversals": self.traversals,
            "transitions": self.transitions,
            "advantage_samples": self.advantage_samples,
            "strategy_samples": self.strategy_samples,
            "profile": self.profile.to_dict(),
            "metrics": asdict(metrics),
            "maximum_positive_regret": self.maximum_positive_regret,
            "solver_fingerprint": self.solver_fingerprint,
            "primary_sources": list(MCCFR_PRIMARY_SOURCES),
            "claims": {
                "official_engine_executed": self.official_engine_executed,
                "abstract_microgame_convergence_measured": True,
                "full_game_equilibrium_convergence": (
                    self.full_game_equilibrium_convergence
                ),
                "pokemon_strength": False,
            },
        }


class ExternalSamplingMCCFRSolver:
    """Tabular external-sampling reference, intentionally limited to the duel."""

    def __init__(self, game: RocketPrizeDuel) -> None:
        self.game = game
        self.information_sets = {
            item.info_set_id: item for item in game.information_sets()
        }
        self.regrets = {
            identifier: {
                action.action_id: 0.0 for action in information_set.legal_actions
            }
            for identifier, information_set in self.information_sets.items()
        }
        self.strategy_sums = {
            identifier: {
                action.action_id: 0.0 for action in information_set.legal_actions
            }
            for identifier, information_set in self.information_sets.items()
        }
        self._iterations = 0
        self._traversals = 0
        self._transitions = 0
        self._advantage_samples = 0
        self._strategy_samples = 0

    def _strategy(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("MCCFR solver received mismatched information set/belief")
        positive = {
            action_id: max(0.0, value)
            for action_id, value in self.regrets[information_set.info_set_id].items()
        }
        if sum(positive.values()) <= 1e-15:
            positive = {action_id: 1.0 for action_id in positive}
        return ActionDistribution.normalized(positive)

    def train(
        self,
        iterations: int,
        *,
        seed: int = 6_019,
    ) -> ExternalSamplingTrainingResult:
        if iterations <= 0:
            raise ValueError("external-sampling iterations must be positive")
        traversal = ExternalSamplingTraversal(
            ExternalSamplingConfig(maximum_transitions=1_000, maximum_depth=32)
        )
        for offset in range(1, iterations + 1):
            iteration = self._iterations + offset
            for traverser in (0, 1):
                result = traversal.run(
                    self.game,
                    self._strategy,
                    traverser=traverser,
                    iteration=iteration,
                    rng=random.Random(_seed(seed, iteration, traverser)),
                )
                self._traversals += 1
                self._transitions += result.transitions
                self._advantage_samples += len(result.advantage_samples)
                self._strategy_samples += len(result.strategy_samples)
                for sample in result.advantage_samples:
                    regrets = self.regrets[sample.information_set_id]
                    for action_id, value in sample.instantaneous_advantages:
                        regrets[action_id] += value
                for sample in result.strategy_samples:
                    totals = self.strategy_sums[sample.information_set_id]
                    for action_id, probability in sample.strategy:
                        totals[action_id] += probability
        self._iterations += iterations
        profile = self.average_profile(
            label=f"external_sampling_mccfr_{self._iterations}_iterations"
        )
        maximum_regret = max(
            max(0.0, value)
            for regrets in self.regrets.values()
            for value in regrets.values()
        )
        solver_fingerprint = _fingerprint(
            {
                "schema": MCCFR_REFERENCE_SCHEMA_VERSION,
                "algorithm": "tabular_external_sampling_mccfr_reference",
                "game": self.game.fingerprint,
                "iterations": self._iterations,
                "traversals": self._traversals,
                "transitions": self._transitions,
                "regrets": self.regrets,
                "strategy_sums": self.strategy_sums,
                "seed": seed,
            }
        )
        return ExternalSamplingTrainingResult(
            iterations=self._iterations,
            traversals=self._traversals,
            transitions=self._transitions,
            advantage_samples=self._advantage_samples,
            strategy_samples=self._strategy_samples,
            profile=profile,
            maximum_positive_regret=maximum_regret,
            solver_fingerprint=solver_fingerprint,
        )

    def average_profile(
        self,
        *,
        label: str = "external_sampling_mccfr_average",
    ) -> TabularStrategyProfile:
        strategies = []
        for identifier in sorted(self.information_sets):
            totals = self.strategy_sums[identifier]
            if sum(totals.values()) <= 1e-15:
                values = {action_id: 1.0 for action_id in totals}
            else:
                values = totals
            strategies.append(
                (identifier, ActionDistribution.normalized(values))
            )
        return TabularStrategyProfile(tuple(strategies), label)

