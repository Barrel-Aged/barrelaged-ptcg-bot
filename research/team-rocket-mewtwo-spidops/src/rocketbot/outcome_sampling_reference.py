"""Exact expectation audit and tabular reference for outcome-sampling MCCFR.

``RocketPrizeDuel`` is small enough to enumerate every terminal trajectory
under the outcome-sampling exploration profile.  The audit therefore compares
the unconditional expectation of each sampled regret update with its exact CFR
immediate counterfactual regret.  This is the prerequisite for using the linear
trajectory sampler against the much larger native boundary.

All results in this module are abstract-microgame results.  They are neither
official Pokemon TCG execution nor evidence of native full-game equilibrium.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
from typing import Any

from .exact_game import (
    DuelState,
    RocketPrizeDuel,
    TabularStrategyProfile,
    exact_profile_metrics,
)
from .game_theory import ActionDistribution, InformationSet, PublicBeliefState
from .mccfr_reference import exact_conditional_advantages
from .outcome_sampling import (
    OUTCOME_SAMPLING_SCHEMA_VERSION,
    OutcomeSamplingConfig,
    OutcomeSamplingTraversal,
)


OUTCOME_SAMPLING_REFERENCE_SCHEMA_VERSION = 1
OUTCOME_SAMPLING_PRIMARY_SOURCES = (
    "https://proceedings.neurips.cc/paper_files/paper/2009/file/"
    "00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf",
    "https://webdocs.cs.ualberta.ca/~bowling/papers/09nips-mccfr-appendix.pdf",
    "https://raw.githubusercontent.com/google-deepmind/open_spiel/master/"
    "open_spiel/algorithms/outcome_sampling_mccfr.cc",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big"
    )


@dataclass(frozen=True, slots=True)
class OutcomeSamplingExpectationRow:
    traverser: int
    information_set_id: str
    action_id: str
    visit_probability: float
    exact_immediate_counterfactual_regret: float
    enumerated_sampled_regret: float
    absolute_error: float

    def __post_init__(self) -> None:
        if self.traverser not in (0, 1) or not self.information_set_id or not self.action_id:
            raise ValueError("outcome-sampling expectation-row identity is invalid")
        for field_name in (
            "visit_probability",
            "exact_immediate_counterfactual_regret",
            "enumerated_sampled_regret",
            "absolute_error",
        ):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if not 0 < self.visit_probability <= 1 + 1e-10 or self.absolute_error < 0:
            raise ValueError("outcome-sampling probability/error is invalid")


@dataclass(frozen=True, slots=True)
class OutcomeSamplingExpectationAudit:
    game_fingerprint: str
    profile_fingerprint: str
    exploration: float
    trajectory_count_player_0: int
    trajectory_count_player_1: int
    rows: tuple[OutcomeSamplingExpectationRow, ...]
    maximum_absolute_error: float
    primary_sources: tuple[str, ...] = OUTCOME_SAMPLING_PRIMARY_SOURCES
    official_engine_executed: bool = False
    full_game_equilibrium_convergence: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.game_fingerprint, "game_fingerprint")
        _validate_digest(self.profile_fingerprint, "profile_fingerprint")
        if not 0 < self.exploration <= 1:
            raise ValueError("outcome-sampling audit exploration is invalid")
        if min(self.trajectory_count_player_0, self.trajectory_count_player_1) <= 0:
            raise ValueError("outcome-sampling audit requires enumerated trajectories")
        if not self.rows or self.maximum_absolute_error < 0 or not math.isfinite(
            self.maximum_absolute_error
        ):
            raise ValueError("outcome-sampling audit rows/error are invalid")
        if self.primary_sources != OUTCOME_SAMPLING_PRIMARY_SOURCES:
            raise ValueError("outcome-sampling audit sources changed")
        if self.official_engine_executed or self.full_game_equilibrium_convergence:
            raise ValueError("microgame audit cannot promote native/full-game claims")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": OUTCOME_SAMPLING_REFERENCE_SCHEMA_VERSION,
            "outcome_sampling_schema_version": OUTCOME_SAMPLING_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "exploration": self.exploration,
            "trajectory_count_player_0": self.trajectory_count_player_0,
            "trajectory_count_player_1": self.trajectory_count_player_1,
            "rows": [asdict(item) for item in self.rows],
            "maximum_absolute_error": self.maximum_absolute_error,
            "primary_sources": list(self.primary_sources),
            "claims": {
                "exact_unconditional_expectation_checked": True,
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


@dataclass(frozen=True, slots=True)
class _EnumeratedTrajectory:
    probability: float
    forced_chance: tuple[tuple[str, str], ...]
    forced_actions: tuple[tuple[str, str], ...]


def _enumerated_trajectories(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
    *,
    traverser: int,
    exploration: float,
) -> tuple[_EnumeratedTrajectory, ...]:
    result: list[_EnumeratedTrajectory] = []

    def visit(
        state: DuelState,
        probability: float,
        forced_chance: dict[str, str],
        forced_actions: dict[str, str],
    ) -> None:
        if state.terminal:
            result.append(
                _EnumeratedTrajectory(
                    probability=probability,
                    forced_chance=tuple(sorted(forced_chance.items())),
                    forced_actions=tuple(sorted(forced_actions.items())),
                )
            )
            return
        if state.cards == (None, None):
            for outcome in game.chance_outcomes(state):
                next_chance = dict(forced_chance)
                next_chance[state.state_id] = outcome.outcome_id
                visit(
                    game.transition(state, None, chance_outcome=outcome),
                    probability * outcome.probability,
                    next_chance,
                    forced_actions,
                )
            return
        player = game.player_to_act(state)
        assert player is not None
        information_set = game.information_set(state, player=player)
        current = profile.strategy(information_set)
        uniform = 1.0 / len(information_set.legal_actions)
        for action in information_set.legal_actions:
            current_probability = current.probability(action.action_id)
            sample_probability = (
                (1.0 - exploration) * current_probability + exploration * uniform
                if player == traverser
                else current_probability
            )
            if sample_probability <= 0:
                continue
            previous = forced_actions.get(information_set.info_set_id)
            if previous is not None and previous != action.action_id:
                raise RuntimeError("one exact trajectory revisits an information set")
            next_actions = dict(forced_actions)
            next_actions[information_set.info_set_id] = action.action_id
            visit(
                game.transition(state, action),
                probability * sample_probability,
                forced_chance,
                next_actions,
            )

    visit(game.root_state(), 1.0, {}, {})
    total = sum(item.probability for item in result)
    if not math.isclose(total, 1.0, abs_tol=1e-12):
        raise RuntimeError("enumerated outcome-sampling trajectories do not sum to one")
    return tuple(result)


def audit_outcome_sampling_expectation(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
    *,
    exploration: float = 0.6,
) -> OutcomeSamplingExpectationAudit:
    """Exhaust sampled trajectories and compare updates with exact CFR regret."""

    conditional, reach_masses = exact_conditional_advantages(game, profile)
    exact = {
        info_id: {
            action_id: value * reach_masses[info_id]
            for action_id, value in values.items()
        }
        for info_id, values in conditional.items()
    }
    weighted = {
        info_id: {action_id: 0.0 for action_id in values}
        for info_id, values in exact.items()
    }
    visits = {info_id: 0.0 for info_id in exact}
    counts = [0, 0]
    traversal = OutcomeSamplingTraversal(
        OutcomeSamplingConfig(
            exploration=exploration,
            maximum_transitions=32,
            maximum_depth=32,
        )
    )

    def strategy(
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("outcome reference received mismatched belief")
        return profile.strategy(information_set)

    def deterministic_nonzero_baseline(
        information_set: InformationSet,
        belief: PublicBeliefState,
        value_player: int,
        acting_player: int,
    ) -> dict[str, float]:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("outcome baseline received mismatched belief")
        sign = 1.0 if value_player == acting_player else -1.0
        return {
            action.action_id: sign * 0.07 * (index + 1)
            for index, action in enumerate(information_set.legal_actions)
        }

    for traverser in (0, 1):
        trajectories = _enumerated_trajectories(
            game,
            profile,
            traverser=traverser,
            exploration=exploration,
        )
        counts[traverser] = len(trajectories)
        for trajectory in trajectories:
            result = traversal.run(
                game,
                strategy,
                traverser=traverser,
                iteration=1,
                rng=random.Random(0),
                forced_chance=dict(trajectory.forced_chance),
                forced_actions=dict(trajectory.forced_actions),
                baseline=deterministic_nonzero_baseline,
            )
            seen: set[str] = set()
            for sample in result.regret_samples:
                if sample.information_set_id in seen:
                    raise RuntimeError("outcome trajectory revisited an information set")
                seen.add(sample.information_set_id)
                visits[sample.information_set_id] += trajectory.probability
                for action_id, value in sample.sampled_regrets:
                    weighted[sample.information_set_id][action_id] += (
                        trajectory.probability * value
                    )

    rows = []
    for information_set in game.information_sets():
        info_id = information_set.info_set_id
        player = game.player_to_act(game.states_for_information_set(info_id)[0])
        assert player is not None
        if visits[info_id] <= 0:
            raise RuntimeError("outcome enumeration did not visit an information set")
        for action in information_set.legal_actions:
            exact_value = exact[info_id][action.action_id]
            sampled_value = weighted[info_id][action.action_id]
            rows.append(
                OutcomeSamplingExpectationRow(
                    traverser=player,
                    information_set_id=info_id,
                    action_id=action.action_id,
                    visit_probability=visits[info_id],
                    exact_immediate_counterfactual_regret=exact_value,
                    enumerated_sampled_regret=sampled_value,
                    absolute_error=abs(exact_value - sampled_value),
                )
            )
    ordered = tuple(
        sorted(
            rows,
            key=lambda item: (
                item.traverser,
                item.information_set_id,
                item.action_id,
            ),
        )
    )
    return OutcomeSamplingExpectationAudit(
        game_fingerprint=game.fingerprint,
        profile_fingerprint=profile.fingerprint,
        exploration=exploration,
        trajectory_count_player_0=counts[0],
        trajectory_count_player_1=counts[1],
        rows=ordered,
        maximum_absolute_error=max(item.absolute_error for item in ordered),
    )


@dataclass(frozen=True, slots=True)
class OutcomeSamplingTrainingResult:
    iterations: int
    traversals: int
    transitions: int
    regret_samples: int
    strategy_samples: int
    exploration: float
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
            self.regret_samples,
            self.strategy_samples,
        ) <= 0:
            raise ValueError("outcome-sampling training accounting must be positive")
        if self.traversals != 2 * self.iterations:
            raise ValueError("outcome reference requires one traversal/player/iteration")
        if not 0 < self.exploration <= 1:
            raise ValueError("outcome-sampling training exploration is invalid")
        if self.maximum_positive_regret < 0 or not math.isfinite(
            self.maximum_positive_regret
        ):
            raise ValueError("outcome-sampling maximum regret is invalid")
        _validate_digest(self.solver_fingerprint, "solver_fingerprint")
        if self.official_engine_executed or self.full_game_equilibrium_convergence:
            raise ValueError("reference solver cannot promote native/full-game claims")

    @property
    def metrics(self):
        return exact_profile_metrics(RocketPrizeDuel(), self.profile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": OUTCOME_SAMPLING_REFERENCE_SCHEMA_VERSION,
            "algorithm": "tabular_outcome_sampling_mccfr_reference",
            "iterations": self.iterations,
            "traversals": self.traversals,
            "transitions": self.transitions,
            "regret_samples": self.regret_samples,
            "strategy_samples": self.strategy_samples,
            "exploration": self.exploration,
            "profile": self.profile.to_dict(),
            "metrics": asdict(self.metrics),
            "maximum_positive_regret": self.maximum_positive_regret,
            "solver_fingerprint": self.solver_fingerprint,
            "primary_sources": list(OUTCOME_SAMPLING_PRIMARY_SOURCES),
            "claims": {
                "official_engine_executed": self.official_engine_executed,
                "abstract_microgame_convergence_measured": True,
                "full_game_equilibrium_convergence": (
                    self.full_game_equilibrium_convergence
                ),
                "pokemon_strength": False,
            },
        }


class OutcomeSamplingMCCFRSolver:
    """Tabular outcome-sampling reference, intentionally limited to the duel."""

    def __init__(self, game: RocketPrizeDuel, *, exploration: float = 0.6) -> None:
        self.game = game
        self.exploration = OutcomeSamplingConfig(exploration=exploration).exploration
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
        self._regret_samples = 0
        self._strategy_samples = 0

    def _strategy(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("outcome solver received mismatched belief")
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
        seed: int = 6_021,
    ) -> OutcomeSamplingTrainingResult:
        if iterations <= 0:
            raise ValueError("outcome-sampling iterations must be positive")
        traversal = OutcomeSamplingTraversal(
            OutcomeSamplingConfig(
                exploration=self.exploration,
                maximum_transitions=32,
                maximum_depth=32,
            )
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
                self._regret_samples += len(result.regret_samples)
                self._strategy_samples += len(result.strategy_samples)
                for sample in result.regret_samples:
                    totals = self.regrets[sample.information_set_id]
                    for action_id, value in sample.sampled_regrets:
                        totals[action_id] += value
                for sample in result.strategy_samples:
                    totals = self.strategy_sums[sample.information_set_id]
                    for action_id, probability in sample.strategy:
                        totals[action_id] += sample.importance_weight * probability
        self._iterations += iterations
        profile = self.average_profile(
            label=f"outcome_sampling_mccfr_{self._iterations}_iterations"
        )
        maximum_regret = max(
            max(0.0, value)
            for regrets in self.regrets.values()
            for value in regrets.values()
        )
        solver_fingerprint = _fingerprint(
            {
                "schema": OUTCOME_SAMPLING_REFERENCE_SCHEMA_VERSION,
                "algorithm": "tabular_outcome_sampling_mccfr_reference",
                "game": self.game.fingerprint,
                "exploration": self.exploration,
                "iterations": self._iterations,
                "traversals": self._traversals,
                "transitions": self._transitions,
                "regrets": self.regrets,
                "strategy_sums": self.strategy_sums,
                "seed": seed,
            }
        )
        return OutcomeSamplingTrainingResult(
            iterations=self._iterations,
            traversals=self._traversals,
            transitions=self._transitions,
            regret_samples=self._regret_samples,
            strategy_samples=self._strategy_samples,
            exploration=self.exploration,
            profile=profile,
            maximum_positive_regret=maximum_regret,
            solver_fingerprint=solver_fingerprint,
        )

    def average_profile(
        self,
        *,
        label: str = "outcome_sampling_mccfr_average",
    ) -> TabularStrategyProfile:
        strategies = []
        for identifier in sorted(self.information_sets):
            totals = self.strategy_sums[identifier]
            values = (
                totals
                if sum(totals.values()) > 1e-15
                else {action_id: 1.0 for action_id in totals}
            )
            strategies.append((identifier, ActionDistribution.normalized(values)))
        return TabularStrategyProfile(tuple(strategies), label)
