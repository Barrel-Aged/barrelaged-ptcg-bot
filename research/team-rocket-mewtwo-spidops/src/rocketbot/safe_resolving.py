"""Budget- and interruption-safe exact resolving for the Phase 6E gate.

This resolver is concrete only for ``RocketPrizeDuel``.  It completes a finite
candidate grid and exactly verifies every opponent private-state hypothesis, or
returns the neural blueprint distribution byte-for-byte.  Partial search output
is never exposed.  The implementation is a validation design for future
endgame/continual resolving, not an official Pokémon solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from time import perf_counter
from typing import Callable

from .counterfactual_calibration import (
    CalibratedBoundDecision,
    CalibratedNeuralCounterfactualBoundProvider,
    CounterfactualBoundContext,
)
from .exact_game import (
    RocketPrizeDuel,
    TabularStrategyProfile,
    evaluate_profile_from_state,
)
from .game_theory import (
    ActionDistribution,
    GameBoundaryKind,
    InformationSet,
    PublicBeliefState,
    ResolveBudget,
    ResolvedStrategy,
    SafeResolveRequest,
    SafetyCertificate,
)
from .neural_blueprint import NeuralDuelBlueprintProvider


class ResolveInterruption(str, Enum):
    NONE = "none"
    ITERATION_BUDGET = "iteration_budget"
    NODE_BUDGET = "node_budget"
    DEADLINE = "deadline"
    BOUND_GATE = "bound_gate"
    BASELINE_OUTSIDE_BOUND = "baseline_outside_bound"


@dataclass(frozen=True, slots=True)
class BudgetAwareResolveStatus:
    resolved: ResolvedStrategy
    completed_search: bool
    interruption: ResolveInterruption
    required_iterations: int
    required_nodes: int
    evaluated_iterations: int
    evaluated_nodes: int
    actor_player: int
    expected_actor_value_before: float | None
    expected_actor_value_after: float | None
    exact_actor_value_gain: float | None
    maximum_opponent_baseline_regression: float
    maximum_bound_violation: float
    strict_baseline_nonregression: bool
    boundary_kind: GameBoundaryKind
    official_engine_executed: bool

    def __post_init__(self) -> None:
        if self.actor_player not in (0, 1):
            raise ValueError("actor_player must be zero or one")
        for field_name in (
            "required_iterations",
            "required_nodes",
            "evaluated_iterations",
            "evaluated_nodes",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        for field_name in (
            "expected_actor_value_before",
            "expected_actor_value_after",
            "exact_actor_value_gain",
        ):
            value = getattr(self, field_name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite when present")
        if self.completed_search != (self.interruption is ResolveInterruption.NONE):
            raise ValueError("completion status must agree with interruption reason")
        if not self.completed_search and not self.resolved.fallback_to_blueprint:
            raise ValueError("interrupted search must return the blueprint")
        if self.completed_search and self.exact_actor_value_gain is None:
            raise ValueError("completed exact search must report its value gain")
        if self.maximum_opponent_baseline_regression < 0 or self.maximum_bound_violation < 0:
            raise ValueError("resolver violations cannot be negative")
        if self.official_engine_executed:
            raise ValueError("exact microgame resolving cannot claim official execution")
        if self.boundary_kind is not GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME:
            raise ValueError("exact microgame resolving must remain an abstraction")


class BudgetAwareExactSafeResolver:
    """All-or-blueprint grid resolver with exact per-hypothesis verification."""

    name = "budget_aware_exact_strict_nonregression_resolver"

    def __init__(
        self,
        base_profile: TabularStrategyProfile,
        *,
        grid_steps: int = 100,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if grid_steps <= 0:
            raise ValueError("grid_steps must be positive")
        self.base_profile = base_profile
        self.grid_steps = grid_steps
        self.clock = clock

    @staticmethod
    def _elapsed_ms(start: float, clock: Callable[[], float]) -> float:
        return max(0.0, (clock() - start) * 1_000.0)

    def _fallback(
        self,
        request: SafeResolveRequest,
        *,
        interruption: ResolveInterruption,
        start: float,
        actor: int,
        required_iterations: int,
        required_nodes: int,
        evaluated_iterations: int,
        evaluated_nodes: int,
    ) -> BudgetAwareResolveStatus:
        certificate = SafetyCertificate(
            safe=True,
            max_counterfactual_value_violation=0.0,
            tolerance=request.budget.safety_tolerance,
            confidence=request.opponent_value_bounds.confidence,
            bound_fingerprint=request.opponent_value_bounds.fingerprint,
            method=f"exact_blueprint_identity_fallback:{interruption.value}",
        )
        resolved = ResolvedStrategy(
            strategy=request.baseline_strategy,
            baseline_strategy=request.baseline_strategy,
            certificate=certificate,
            iterations=evaluated_iterations,
            expanded_nodes=evaluated_nodes,
            elapsed_ms=self._elapsed_ms(start, self.clock),
            fallback_to_blueprint=True,
            resolver_name=self.name,
        )
        return BudgetAwareResolveStatus(
            resolved=resolved,
            completed_search=False,
            interruption=interruption,
            required_iterations=required_iterations,
            required_nodes=required_nodes,
            evaluated_iterations=evaluated_iterations,
            evaluated_nodes=evaluated_nodes,
            actor_player=actor,
            expected_actor_value_before=None,
            expected_actor_value_after=None,
            exact_actor_value_gain=None,
            maximum_opponent_baseline_regression=0.0,
            maximum_bound_violation=0.0,
            strict_baseline_nonregression=True,
            boundary_kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            official_engine_executed=False,
        )

    def resolve_with_status(
        self,
        game: RocketPrizeDuel,
        request: SafeResolveRequest,
    ) -> BudgetAwareResolveStatus:
        if not isinstance(game, RocketPrizeDuel):
            raise TypeError("budget-aware exact resolver supports RocketPrizeDuel only")
        baseline = self.base_profile.strategy(request.information_set)
        if baseline != request.baseline_strategy:
            raise ValueError("resolver base profile does not match request baseline")
        states = game.states_for_information_set(request.information_set.info_set_id)
        actor = states[0].acting_player
        if actor is None or any(state.acting_player != actor for state in states):
            raise ValueError("resolver root must be one player's information set")
        opponent = 1 - actor
        if request.opponent_value_bounds.opponent_player != opponent:
            raise ValueError("counterfactual bounds target the wrong opponent")
        actions = request.information_set.legal_actions
        if len(actions) != 2:
            raise ValueError("budget-aware exact resolver currently requires two actions")
        state_by_hypothesis = {}
        for state in states:
            card = state.cards[opponent]
            if card is None:
                raise ValueError("resolver state lacks opponent private card")
            state_by_hypothesis[f"opponent:{card.label}"] = state
        weights = {
            item.hypothesis_id: item.weight
            for item in request.belief.opponent_range.hypotheses
        }
        bounds = {
            item.hypothesis_id: item for item in request.opponent_value_bounds.bounds
        }
        if set(state_by_hypothesis) != set(weights) or set(weights) != set(bounds):
            raise ValueError("states, belief range, and bounds must share hypotheses")
        first_probabilities = {index / self.grid_steps for index in range(self.grid_steps + 1)}
        first_probabilities.add(baseline.probability(actions[0].action_id))
        candidate_probabilities = tuple(sorted(first_probabilities))
        required_iterations = len(candidate_probabilities)
        # One exact baseline pass plus one pass per candidate and hypothesis.
        required_nodes = (required_iterations + 1) * len(states)
        start = self.clock()
        if request.budget.iterations < required_iterations:
            return self._fallback(
                request,
                interruption=ResolveInterruption.ITERATION_BUDGET,
                start=start,
                actor=actor,
                required_iterations=required_iterations,
                required_nodes=required_nodes,
                evaluated_iterations=0,
                evaluated_nodes=0,
            )
        if request.budget.max_nodes < required_nodes:
            return self._fallback(
                request,
                interruption=ResolveInterruption.NODE_BUDGET,
                start=start,
                actor=actor,
                required_iterations=required_iterations,
                required_nodes=required_nodes,
                evaluated_iterations=0,
                evaluated_nodes=0,
            )

        def deadline_exceeded() -> bool:
            return self._elapsed_ms(start, self.clock) >= request.budget.time_budget_ms

        evaluated_nodes = 0
        exact_baseline_opponent: dict[str, float] = {}
        exact_baseline_actor = 0.0
        for hypothesis_id, state in state_by_hypothesis.items():
            if deadline_exceeded():
                return self._fallback(
                    request,
                    interruption=ResolveInterruption.DEADLINE,
                    start=start,
                    actor=actor,
                    required_iterations=required_iterations,
                    required_nodes=required_nodes,
                    evaluated_iterations=0,
                    evaluated_nodes=evaluated_nodes,
                )
            value = evaluate_profile_from_state(
                game,
                self.base_profile,
                state,
                player=opponent,
            )
            exact_baseline_opponent[hypothesis_id] = value
            exact_baseline_actor += weights[hypothesis_id] * (-value)
            evaluated_nodes += 1
        maximum_baseline_bound_violation = max(
            max(0.0, exact_baseline_opponent[hypothesis_id] - bounds[hypothesis_id].upper)
            for hypothesis_id in state_by_hypothesis
        )
        if maximum_baseline_bound_violation > request.budget.safety_tolerance + 1e-12:
            return self._fallback(
                request,
                interruption=ResolveInterruption.BASELINE_OUTSIDE_BOUND,
                start=start,
                actor=actor,
                required_iterations=required_iterations,
                required_nodes=required_nodes,
                evaluated_iterations=0,
                evaluated_nodes=evaluated_nodes,
            )

        best: tuple[float, float, ActionDistribution, float, float] | None = None
        evaluated_iterations = 0
        for first_probability in candidate_probabilities:
            if deadline_exceeded():
                return self._fallback(
                    request,
                    interruption=ResolveInterruption.DEADLINE,
                    start=start,
                    actor=actor,
                    required_iterations=required_iterations,
                    required_nodes=required_nodes,
                    evaluated_iterations=evaluated_iterations,
                    evaluated_nodes=evaluated_nodes,
                )
            strategy = ActionDistribution(
                (
                    (actions[0].action_id, first_probability),
                    (actions[1].action_id, 1.0 - first_probability),
                )
            )
            candidate = self.base_profile.with_strategy(
                request.information_set.info_set_id,
                strategy,
                label="budget_aware_exact_resolve_candidate",
            )
            actor_value = 0.0
            maximum_bound_violation = 0.0
            maximum_baseline_regression = 0.0
            for hypothesis_id, state in state_by_hypothesis.items():
                if deadline_exceeded():
                    return self._fallback(
                        request,
                        interruption=ResolveInterruption.DEADLINE,
                        start=start,
                        actor=actor,
                        required_iterations=required_iterations,
                        required_nodes=required_nodes,
                        evaluated_iterations=evaluated_iterations,
                        evaluated_nodes=evaluated_nodes,
                    )
                opponent_value = evaluate_profile_from_state(
                    game,
                    candidate,
                    state,
                    player=opponent,
                )
                evaluated_nodes += 1
                actor_value += weights[hypothesis_id] * (-opponent_value)
                maximum_bound_violation = max(
                    maximum_bound_violation,
                    opponent_value - bounds[hypothesis_id].upper,
                )
                maximum_baseline_regression = max(
                    maximum_baseline_regression,
                    opponent_value - exact_baseline_opponent[hypothesis_id],
                )
            evaluated_iterations += 1
            maximum_bound_violation = max(0.0, maximum_bound_violation)
            maximum_baseline_regression = max(0.0, maximum_baseline_regression)
            if maximum_bound_violation > request.budget.safety_tolerance + 1e-12:
                continue
            if maximum_baseline_regression > request.budget.safety_tolerance + 1e-12:
                continue
            distance = sum(
                abs(
                    strategy.probability(action.action_id)
                    - baseline.probability(action.action_id)
                )
                for action in actions
            )
            score = (
                actor_value,
                -distance,
                strategy,
                maximum_baseline_regression,
                maximum_bound_violation,
            )
            if best is None or score[:2] > best[:2]:
                best = score
        if best is None:
            # The exact baseline is a candidate. Reaching this path means the
            # certification assumptions were inconsistent, so identity is the
            # only permissible output.
            return self._fallback(
                request,
                interruption=ResolveInterruption.BASELINE_OUTSIDE_BOUND,
                start=start,
                actor=actor,
                required_iterations=required_iterations,
                required_nodes=required_nodes,
                evaluated_iterations=evaluated_iterations,
                evaluated_nodes=evaluated_nodes,
            )
        actor_value, _, strategy, maximum_regression, maximum_violation = best
        gain = actor_value - exact_baseline_actor
        if gain < -request.budget.safety_tolerance - 1e-12:
            raise RuntimeError("strict resolver selected a lower-value candidate")
        if sum(
            abs(
                strategy.probability(action.action_id)
                - baseline.probability(action.action_id)
            )
            for action in actions
        ) <= 1e-12:
            strategy = baseline
        certificate = SafetyCertificate(
            safe=True,
            max_counterfactual_value_violation=max(
                maximum_regression,
                maximum_violation,
            ),
            tolerance=request.budget.safety_tolerance,
            confidence=request.opponent_value_bounds.confidence,
            bound_fingerprint=request.opponent_value_bounds.fingerprint,
            method="exact_per_hypothesis_bound_and_strict_baseline_nonregression",
        )
        resolved = ResolvedStrategy(
            strategy=strategy,
            baseline_strategy=baseline,
            certificate=certificate,
            iterations=evaluated_iterations,
            expanded_nodes=evaluated_nodes,
            elapsed_ms=self._elapsed_ms(start, self.clock),
            fallback_to_blueprint=strategy == baseline,
            resolver_name=self.name,
        )
        return BudgetAwareResolveStatus(
            resolved=resolved,
            completed_search=True,
            interruption=ResolveInterruption.NONE,
            required_iterations=required_iterations,
            required_nodes=required_nodes,
            evaluated_iterations=evaluated_iterations,
            evaluated_nodes=evaluated_nodes,
            actor_player=actor,
            expected_actor_value_before=exact_baseline_actor,
            expected_actor_value_after=actor_value,
            exact_actor_value_gain=max(0.0, gain),
            maximum_opponent_baseline_regression=maximum_regression,
            maximum_bound_violation=maximum_violation,
            strict_baseline_nonregression=maximum_regression <= request.budget.safety_tolerance,
            boundary_kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            official_engine_executed=False,
        )

    def resolve(
        self,
        game: RocketPrizeDuel,
        request: SafeResolveRequest,
    ) -> ResolvedStrategy:
        return self.resolve_with_status(game, request).resolved


@dataclass(frozen=True, slots=True)
class ExactSafePlanningDecision:
    strategy: ActionDistribution
    blueprint_strategy: ActionDistribution
    bound_decision: CalibratedBoundDecision
    resolve_status: BudgetAwareResolveStatus | None
    fallback_to_blueprint: bool
    reason: str
    boundary_kind: GameBoundaryKind
    official_engine_executed: bool

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("planning decision reason must be non-empty")
        if self.fallback_to_blueprint and self.strategy != self.blueprint_strategy:
            raise ValueError("pipeline fallback must return the blueprint exactly")
        if self.official_engine_executed:
            raise ValueError("exact safe planning cannot claim official execution")
        if self.boundary_kind is not GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME:
            raise ValueError("exact safe planning is an abstract boundary")


class ExactSafePlanningPipeline:
    """Fail-closed Phase 6D/6E composition for the exact validation game."""

    def __init__(
        self,
        *,
        game: RocketPrizeDuel,
        blueprint: NeuralDuelBlueprintProvider,
        bound_provider: CalibratedNeuralCounterfactualBoundProvider,
        resolver: BudgetAwareExactSafeResolver,
    ) -> None:
        self.game = game
        self.blueprint = blueprint
        self.bound_provider = bound_provider
        self.resolver = resolver

    def decide(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        budget: ResolveBudget,
        bound_context: CounterfactualBoundContext | None = None,
    ) -> ExactSafePlanningDecision:
        blueprint_strategy = self.blueprint.strategy(information_set, belief)
        bound_decision = self.bound_provider.decide_bounds(
            information_set,
            belief,
            context=bound_context,
        )
        if bound_decision.bounds is None:
            return ExactSafePlanningDecision(
                strategy=blueprint_strategy,
                blueprint_strategy=blueprint_strategy,
                bound_decision=bound_decision,
                resolve_status=None,
                fallback_to_blueprint=True,
                reason=f"bound gate fallback: {bound_decision.reason}",
                boundary_kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
                official_engine_executed=False,
            )
        status = self.resolver.resolve_with_status(
            self.game,
            SafeResolveRequest(
                information_set=information_set,
                belief=belief,
                baseline_strategy=blueprint_strategy,
                opponent_value_bounds=bound_decision.bounds,
                budget=budget,
            ),
        )
        return ExactSafePlanningDecision(
            strategy=status.resolved.strategy,
            blueprint_strategy=blueprint_strategy,
            bound_decision=bound_decision,
            resolve_status=status,
            fallback_to_blueprint=status.resolved.fallback_to_blueprint,
            reason=(
                "completed exact strict-nonregression grid"
                if status.completed_search
                else f"resolver fallback: {status.interruption.value}"
            ),
            boundary_kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            official_engine_executed=False,
        )
