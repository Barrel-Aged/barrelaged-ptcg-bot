"""Deterministic strategic-abstraction adapter for Phase 3 verification.

This is not a Pokemon rules engine. It provides a branch-safe simulator-shaped
adapter so plan search, posterior scenarios, and autoregressive execution can be
tested end to end before a competition environment adapter is connected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .beliefs import OPPONENT_INTENTS, OpponentIntent
from .observation import AgentObservation
from .plans import PlanDirective, PlanStep, StrategicPlanCandidate
from .policy import ActionComponent, ActionPrefix, ActionToken, TacticalCandidate
from .search import BeliefScenario, IntentOpponentBehaviorModel, RolloutOutcome
from .state import ActionKind
from .strategy import STRATEGIC_AXES


@dataclass(frozen=True, slots=True)
class AbstractRolloutState:
    axis_deltas: tuple[float, ...]
    resource_cost: float
    committed_directives: tuple[PlanDirective, ...] = ()
    tactical_choices: tuple[str, ...] = ()


class StrategicAbstractionAdapter:
    """Map tactical selections into strategic deltas under each belief scenario."""

    ACTION_OPTIONS: Mapping[PlanDirective, tuple[ActionKind, ...]] = {
        PlanDirective.DEVELOP_SPIDOPS: (
            ActionKind.EVOLVE,
            ActionKind.PLAY_POKEMON,
            ActionKind.PLAY_TRAINER,
        ),
        PlanDirective.PREPARE_MEWTWO: (
            ActionKind.ATTACH_ENERGY,
            ActionKind.PLAY_TRAINER,
            ActionKind.PLAY_POKEMON,
        ),
        PlanDirective.ACCELERATE_ENERGY: (
            ActionKind.ATTACH_ENERGY,
            ActionKind.USE_ABILITY,
            ActionKind.PLAY_TRAINER,
        ),
        PlanDirective.APPLY_DISRUPTION: (
            ActionKind.PLAY_TRAINER,
            ActionKind.USE_ABILITY,
            ActionKind.ATTACK,
        ),
        PlanDirective.CONTROL_BOARD: (
            ActionKind.USE_ABILITY,
            ActionKind.PLAY_TRAINER,
            ActionKind.ATTACK,
        ),
        PlanDirective.TAKE_PRIZE: (
            ActionKind.ATTACK,
            ActionKind.PLAY_TRAINER,
            ActionKind.USE_ABILITY,
        ),
        PlanDirective.RECOVER_RESOURCES: (
            ActionKind.PLAY_TRAINER,
            ActionKind.USE_ABILITY,
            ActionKind.END_TURN,
        ),
        PlanDirective.PRESERVE_OPTIONS: (
            ActionKind.USE_ABILITY,
            ActionKind.PLAY_TRAINER,
            ActionKind.END_TURN,
        ),
        PlanDirective.PIVOT_ATTACKER: (
            ActionKind.RETREAT,
            ActionKind.PLAY_TRAINER,
            ActionKind.USE_ABILITY,
        ),
        PlanDirective.BENCH_CLEFAIRY: (
            ActionKind.PLAY_POKEMON,
            ActionKind.PLAY_TRAINER,
            ActionKind.USE_ABILITY,
        ),
    }

    AXIS_DELTAS: Mapping[PlanDirective, tuple[float, ...]] = {
        PlanDirective.DEVELOP_SPIDOPS: (0.35, -0.03, 0.15, 0.00, 0.12, 0.18),
        PlanDirective.PREPARE_MEWTWO: (0.25, 0.12, 0.03, 0.08, 0.04, 0.02),
        PlanDirective.ACCELERATE_ENERGY: (0.10, 0.12, 0.28, 0.02, 0.03, 0.10),
        PlanDirective.APPLY_DISRUPTION: (0.00, 0.25, 0.04, 0.10, 0.30, 0.02),
        PlanDirective.CONTROL_BOARD: (0.00, 0.15, 0.02, 0.10, 0.35, 0.04),
        PlanDirective.TAKE_PRIZE: (0.00, 0.25, -0.10, 0.35, 0.10, -0.05),
        PlanDirective.RECOVER_RESOURCES: (0.04, -0.08, 0.30, 0.02, 0.04, 0.20),
        PlanDirective.PRESERVE_OPTIONS: (0.02, -0.04, 0.10, 0.02, 0.05, 0.35),
        PlanDirective.PIVOT_ATTACKER: (0.03, 0.16, -0.03, 0.05, 0.12, 0.22),
        PlanDirective.BENCH_CLEFAIRY: (0.16, 0.20, -0.04, 0.08, 0.04, 0.12),
    }

    RESOURCE_COST: Mapping[PlanDirective, float] = {
        PlanDirective.DEVELOP_SPIDOPS: 0.20,
        PlanDirective.PREPARE_MEWTWO: 0.15,
        PlanDirective.ACCELERATE_ENERGY: 0.15,
        PlanDirective.APPLY_DISRUPTION: 0.20,
        PlanDirective.CONTROL_BOARD: 0.15,
        PlanDirective.TAKE_PRIZE: 0.25,
        PlanDirective.RECOVER_RESOURCES: -0.15,
        PlanDirective.PRESERVE_OPTIONS: -0.05,
        PlanDirective.PIVOT_ATTACKER: 0.10,
        PlanDirective.BENCH_CLEFAIRY: 0.12,
    }

    def __init__(self, candidate_feature_dim: int) -> None:
        if candidate_feature_dim < 12:
            raise ValueError("strategic abstraction requires at least 12 candidate features")
        self.candidate_feature_dim = candidate_feature_dim

    def start(
        self,
        observation: AgentObservation,
        plan: StrategicPlanCandidate,
        scenario: BeliefScenario,
    ) -> AbstractRolloutState:
        return AbstractRolloutState((0.0,) * len(STRATEGIC_AXES), 0.0)

    def _candidate_features(
        self,
        *,
        quality: float,
        component: ActionComponent,
        plan: StrategicPlanCandidate,
        step: PlanStep,
        scenario: BeliefScenario,
    ) -> tuple[float, ...]:
        features = [0.0] * self.candidate_feature_dim
        objective_index = STRATEGIC_AXES.index(plan.objective)
        features[objective_index] = 1.0
        component_index = {
            ActionComponent.ACTION_KIND: 6,
            ActionComponent.MODE: 7,
            ActionComponent.END: 8,
        }.get(component, 9)
        features[component_index] = 1.0
        features[9] = OPPONENT_INTENTS.index(scenario.intent) / max(
            len(OPPONENT_INTENTS) - 1, 1
        )
        features[10] = (scenario.style.rationality + 1.0) / 5.0
        features[11] = sum(scenario.resources.presence) / len(scenario.resources.presence)
        if self.candidate_feature_dim > 12:
            features[12] = plan.prior
        if self.candidate_feature_dim > 13:
            features[13] = step.turn_offset / max(plan.horizon_turns, 1)
        features[-1] = quality
        return tuple(features)

    def _scenario_quality_bonus(
        self,
        directive: PlanDirective,
        scenario: BeliefScenario,
    ) -> float:
        if directive in (PlanDirective.APPLY_DISRUPTION, PlanDirective.CONTROL_BOARD):
            return 0.25 if scenario.intent is OpponentIntent.COMBO_ASSEMBLY else 0.0
        if directive in (PlanDirective.PIVOT_ATTACKER, PlanDirective.PRESERVE_OPTIONS):
            return 0.2 if scenario.intent is OpponentIntent.PRIZE_RACE else 0.0
        if directive is PlanDirective.TAKE_PRIZE:
            return 0.15 if scenario.intent in (
                OpponentIntent.SETUP,
                OpponentIntent.COMBO_ASSEMBLY,
            ) else 0.0
        return 0.0

    def next_candidates(
        self,
        rollout_state: AbstractRolloutState,
        plan: StrategicPlanCandidate,
        step: PlanStep,
        prefix: ActionPrefix,
        scenario: BeliefScenario,
    ) -> Sequence[TacticalCandidate]:
        if len(prefix.tokens) == 0:
            options = self.ACTION_OPTIONS[step.directive]
            return tuple(
                TacticalCandidate(
                    token=ActionToken(
                        ActionComponent.ACTION_KIND,
                        action.value,
                        action.value.replace("_", " ").title(),
                    ),
                    features=self._candidate_features(
                        quality=(1.0 if index == 0 else 0.62 - index * 0.08)
                        + self._scenario_quality_bonus(step.directive, scenario),
                        component=ActionComponent.ACTION_KIND,
                        plan=plan,
                        step=step,
                        scenario=scenario,
                    ),
                )
                for index, action in enumerate(options)
            )
        if len(prefix.tokens) == 1:
            aggressive_is_best = step.directive in (
                PlanDirective.APPLY_DISRUPTION,
                PlanDirective.CONTROL_BOARD,
                PlanDirective.TAKE_PRIZE,
            )
            conservative_is_best = step.directive in (
                PlanDirective.RECOVER_RESOURCES,
                PlanDirective.PRESERVE_OPTIONS,
            )
            modes = ("aggressive", "balanced", "conservative")
            return tuple(
                TacticalCandidate(
                    token=ActionToken(ActionComponent.MODE, mode, mode.title()),
                    features=self._candidate_features(
                        quality=(
                            1.0
                            if (mode == "aggressive" and aggressive_is_best)
                            or (mode == "conservative" and conservative_is_best)
                            or (mode == "balanced" and not aggressive_is_best and not conservative_is_best)
                            else 0.55
                        ),
                        component=ActionComponent.MODE,
                        plan=plan,
                        step=step,
                        scenario=scenario,
                    ),
                )
                for mode in modes
            )
        if len(prefix.tokens) == 2:
            return (
                TacticalCandidate(
                    token=ActionToken(ActionComponent.END, "end", "Commit action"),
                    features=self._candidate_features(
                        quality=1.0,
                        component=ActionComponent.END,
                        plan=plan,
                        step=step,
                        scenario=scenario,
                    ),
                ),
            )
        raise RuntimeError("abstract tactical action exceeded its three-token schema")

    def _matchup_factor(
        self,
        plan: StrategicPlanCandidate,
        scenario: BeliefScenario,
    ) -> float:
        factors = {
            "rocket_disruption": {
                OpponentIntent.COMBO_ASSEMBLY: 1.55,
                OpponentIntent.LOCK_DISRUPTION: 1.25,
            },
            "mewtwo_prize_push": {
                OpponentIntent.SETUP: 1.25,
                OpponentIntent.COMBO_ASSEMBLY: 1.15,
                OpponentIntent.PRIZE_RACE: 0.90,
            },
            "defensive_pivot": {
                OpponentIntent.PRIZE_RACE: 1.45,
                OpponentIntent.BOARD_DENIAL: 1.20,
            },
            "spidops_engine": {OpponentIntent.PRIZE_RACE: 0.72},
        }
        return factors.get(plan.plan_id, {}).get(scenario.intent, 1.0)

    def commit_action(
        self,
        rollout_state: AbstractRolloutState,
        plan: StrategicPlanCandidate,
        step: PlanStep,
        prefix: ActionPrefix,
        scenario: BeliefScenario,
    ) -> AbstractRolloutState:
        if not prefix.is_complete or len(prefix.tokens) != 3:
            raise ValueError("abstract action must contain kind, mode, and END")
        action_choice = prefix.tokens[0].choice_id
        mode = prefix.tokens[1].choice_id
        ideal_action = self.ACTION_OPTIONS[step.directive][0].value
        action_alignment = 1.0 if action_choice == ideal_action else 0.72
        mode_factor = {
            "aggressive": 1.15,
            "balanced": 1.0,
            "conservative": 0.86,
        }[mode]
        matchup_factor = self._matchup_factor(plan, scenario)
        delta = tuple(
            value * action_alignment * mode_factor * matchup_factor
            for value in self.AXIS_DELTAS[step.directive]
        )
        axis_deltas = tuple(
            prior + increment
            for prior, increment in zip(rollout_state.axis_deltas, delta)
        )
        resource_cost = rollout_state.resource_cost + self.RESOURCE_COST[step.directive]
        if mode == "aggressive":
            resource_cost += 0.05
        elif mode == "conservative":
            resource_cost -= 0.03
        return AbstractRolloutState(
            axis_deltas=axis_deltas,
            resource_cost=resource_cost,
            committed_directives=rollout_state.committed_directives + (step.directive,),
            tactical_choices=rollout_state.tactical_choices
            + (f"{step.directive.value}:{action_choice}:{mode}",),
        )

    def evaluate(
        self,
        rollout_state: AbstractRolloutState,
        plan: StrategicPlanCandidate,
        scenario: BeliefScenario,
    ) -> RolloutOutcome:
        weighted_gain = sum(
            alignment * delta
            for alignment, delta in zip(plan.axis_alignment, rollout_state.axis_deltas)
        ) / max(sum(abs(value) for value in plan.axis_alignment), 1.0)
        terminal_value = math.tanh(weighted_gain - 0.08 * rollout_state.resource_cost)
        objective_index = STRATEGIC_AXES.index(plan.objective)
        objective_achievement = math.tanh(rollout_state.axis_deltas[objective_index])
        predicted = IntentOpponentBehaviorModel.SIGNATURES[scenario.intent]
        return RolloutOutcome(
            axis_deltas=rollout_state.axis_deltas,
            terminal_value=terminal_value,
            objective_achievement=objective_achievement,
            predicted_opponent_actions=predicted,
            resource_cost=rollout_state.resource_cost,
            notes=rollout_state.tactical_choices,
        )
