"""Complete high-level strategic plan candidates and Team Rocket proposer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Protocol, Sequence

import torch

from .beliefs import OPPONENT_INTENTS, OpponentBeliefState, ResourceCategory
from .model import BeliefAwareEvaluation
from .observation import AgentObservation
from .strategy import STRATEGIC_AXES, StrategicAxis


class PlanDirective(str, Enum):
    DEVELOP_SPIDOPS = "develop_spidops"
    PREPARE_MEWTWO = "prepare_mewtwo"
    ACCELERATE_ENERGY = "accelerate_energy"
    APPLY_DISRUPTION = "apply_disruption"
    CONTROL_BOARD = "control_board"
    TAKE_PRIZE = "take_prize"
    RECOVER_RESOURCES = "recover_resources"
    PRESERVE_OPTIONS = "preserve_options"
    PIVOT_ATTACKER = "pivot_attacker"
    BENCH_CLEFAIRY = "bench_clefairy"


@dataclass(frozen=True, slots=True)
class PlanStep:
    step_id: str
    directive: PlanDirective
    turn_offset: int
    success_condition: str

    def __post_init__(self) -> None:
        if not self.step_id.strip() or not self.success_condition.strip():
            raise ValueError("plan step ID and success condition must be non-empty")
        if self.turn_offset < 0:
            raise ValueError("turn_offset cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "directive": self.directive.value,
            "turn_offset": self.turn_offset,
            "success_condition": self.success_condition,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanStep":
        return cls(
            step_id=str(raw["step_id"]),
            directive=PlanDirective(str(raw["directive"])),
            turn_offset=int(raw["turn_offset"]),
            success_condition=str(raw["success_condition"]),
        )


@dataclass(frozen=True, slots=True)
class StrategicPlanCandidate:
    """A complete, inspectable line proposed before low-level execution."""

    plan_id: str
    title: str
    objective: StrategicAxis
    horizon_turns: int
    steps: tuple[PlanStep, ...]
    completion_condition: str
    rationale: str
    required_resources: tuple[ResourceCategory, ...]
    axis_alignment: tuple[float, ...]
    opponent_matchups: tuple[float, ...]
    prior: float
    proposal_score: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.plan_id.strip() or not self.title.strip():
            raise ValueError("plan ID and title must be non-empty")
        if not self.completion_condition.strip() or not self.rationale.strip():
            raise ValueError("plan completion condition and rationale must be non-empty")
        if self.horizon_turns <= 0:
            raise ValueError("plan horizon must be positive")
        if not self.steps:
            raise ValueError("a complete plan requires at least one step")
        if max(step.turn_offset for step in self.steps) >= self.horizon_turns:
            raise ValueError("plan step lies outside its declared horizon")
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan step IDs must be unique")
        if len(self.axis_alignment) != len(STRATEGIC_AXES):
            raise ValueError("axis_alignment must contain six values")
        if len(self.opponent_matchups) != len(OPPONENT_INTENTS):
            raise ValueError("opponent_matchups must contain one value per intent")
        if not all(math.isfinite(value) for value in self.axis_alignment):
            raise ValueError("axis alignment values must be finite")
        if not all(math.isfinite(value) for value in self.opponent_matchups):
            raise ValueError("opponent matchup values must be finite")
        if not math.isfinite(self.prior) or not 0.0 <= self.prior <= 1.0:
            raise ValueError("plan prior must lie in [0, 1]")
        if not math.isfinite(self.proposal_score):
            raise ValueError("proposal score must be finite")

    @property
    def directive_signature(self) -> tuple[PlanDirective, ...]:
        return tuple(step.directive for step in self.steps)

    def with_prior(self, prior: float) -> "StrategicPlanCandidate":
        return StrategicPlanCandidate(
            plan_id=self.plan_id,
            title=self.title,
            objective=self.objective,
            horizon_turns=self.horizon_turns,
            steps=self.steps,
            completion_condition=self.completion_condition,
            rationale=self.rationale,
            required_resources=self.required_resources,
            axis_alignment=self.axis_alignment,
            opponent_matchups=self.opponent_matchups,
            prior=prior,
            proposal_score=self.proposal_score,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "title": self.title,
            "objective": self.objective.value,
            "horizon_turns": self.horizon_turns,
            "steps": [step.to_dict() for step in self.steps],
            "completion_condition": self.completion_condition,
            "rationale": self.rationale,
            "required_resources": [resource.value for resource in self.required_resources],
            "axis_alignment": list(self.axis_alignment),
            "opponent_matchups": list(self.opponent_matchups),
            "prior": self.prior,
            "proposal_score": self.proposal_score,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StrategicPlanCandidate":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            plan_id=str(raw["plan_id"]),
            title=str(raw["title"]),
            objective=StrategicAxis(str(raw["objective"])),
            horizon_turns=int(raw["horizon_turns"]),
            steps=tuple(PlanStep.from_dict(step) for step in raw["steps"]),
            completion_condition=str(raw["completion_condition"]),
            rationale=str(raw["rationale"]),
            required_resources=tuple(
                ResourceCategory(str(value)) for value in raw["required_resources"]
            ),
            axis_alignment=tuple(float(value) for value in raw["axis_alignment"]),
            opponent_matchups=tuple(float(value) for value in raw["opponent_matchups"]),
            prior=float(raw["prior"]),
            proposal_score=float(raw["proposal_score"]),
        )


class StrategicPlanProposer(Protocol):
    def propose(
        self,
        observation: AgentObservation,
        evaluation: BeliefAwareEvaluation,
        belief: OpponentBeliefState,
        *,
        top_k: int,
        batch_index: int = 0,
    ) -> Sequence[StrategicPlanCandidate]: ...


def _step(
    step_id: str,
    directive: PlanDirective,
    turn_offset: int,
    success: str,
) -> PlanStep:
    return PlanStep(step_id, directive, turn_offset, success)


class TeamRocketPlanProposer:
    """Propose distinct Mewtwo/Spidops plans from values, state, and beliefs."""

    def __init__(self, *, distinctness_threshold: float = 0.75) -> None:
        if not 0.0 <= distinctness_threshold <= 1.0:
            raise ValueError("distinctness_threshold must lie in [0, 1]")
        self.distinctness_threshold = distinctness_threshold

    @staticmethod
    def _templates() -> tuple[StrategicPlanCandidate, ...]:
        return (
            StrategicPlanCandidate(
                plan_id="spidops_engine",
                title="Build the Spidops engine",
                objective=StrategicAxis.SETUP,
                horizon_turns=2,
                steps=(
                    _step("establish", PlanDirective.DEVELOP_SPIDOPS, 0, "A Spidops line is established or protected."),
                    _step("accelerate", PlanDirective.ACCELERATE_ENERGY, 0, "Energy infrastructure advances without stranding the attacker."),
                    _step("preserve", PlanDirective.PRESERVE_OPTIONS, 1, "At least two credible next-turn lines remain."),
                ),
                completion_condition="Spidops infrastructure supports multiple powered attackers.",
                rationale="Invest in the deck's board and energy engine before committing Mewtwo.",
                required_resources=(ResourceCategory.BASIC_POKEMON, ResourceCategory.EVOLUTION, ResourceCategory.SEARCH),
                axis_alignment=(1.0, 0.1, 0.8, 0.0, 0.4, 0.8),
                opponent_matchups=(0.4, -0.4, 0.1, 0.0, 0.3, 0.2),
                prior=0.0,
                proposal_score=0.0,
            ),
            StrategicPlanCandidate(
                plan_id="mewtwo_prize_push",
                title="Convert Mewtwo pressure into prizes",
                objective=StrategicAxis.PRIZE_CONTROL,
                horizon_turns=1,
                steps=(
                    _step("prepare", PlanDirective.PREPARE_MEWTWO, 0, "Mewtwo is a credible attacker this turn."),
                    _step("power", PlanDirective.ACCELERATE_ENERGY, 0, "Required attack energy is assembled efficiently."),
                    _step("convert", PlanDirective.TAKE_PRIZE, 0, "The best available prize target is attacked."),
                ),
                completion_condition="Take a prize while retaining a follow-up attacker.",
                rationale="Exploit a tempo window before the opponent's board or combo matures.",
                required_resources=(ResourceCategory.ENERGY, ResourceCategory.SEARCH),
                axis_alignment=(0.2, 1.0, 0.1, 1.0, 0.4, -0.2),
                opponent_matchups=(0.8, 0.1, -0.2, 0.4, 0.5, 0.7),
                prior=0.0,
                proposal_score=0.0,
            ),
            StrategicPlanCandidate(
                plan_id="rocket_disruption",
                title="Disrupt the opponent's assembly line",
                objective=StrategicAxis.BOARD_CONTROL,
                horizon_turns=1,
                steps=(
                    _step("disrupt", PlanDirective.APPLY_DISRUPTION, 0, "Ariana, Giovanni, or another Rocket disruption changes their options."),
                    _step("control", PlanDirective.CONTROL_BOARD, 0, "The most important opposing board node is constrained."),
                    _step("pressure", PlanDirective.TAKE_PRIZE, 0, "Pressure converts without abandoning board control."),
                ),
                completion_condition="The opponent loses a key resource or board route before its payoff.",
                rationale="Attack the inferred plan and its minimum enabling resources, not only the active Pokemon.",
                required_resources=(ResourceCategory.DISRUPTION, ResourceCategory.ENERGY),
                axis_alignment=(0.0, 0.7, 0.2, 0.5, 1.0, 0.2),
                opponent_matchups=(0.1, 0.2, 0.8, 0.7, 0.6, 1.2),
                prior=0.0,
                proposal_score=0.0,
            ),
            StrategicPlanCandidate(
                plan_id="defensive_pivot",
                title="Pivot and preserve the long game",
                objective=StrategicAxis.OPTIONALITY,
                horizon_turns=2,
                steps=(
                    _step("pivot", PlanDirective.PIVOT_ATTACKER, 0, "A lower-liability attacker absorbs the next exchange."),
                    _step("recover", PlanDirective.RECOVER_RESOURCES, 0, "A discarded attacker or energy route returns."),
                    _step("preserve", PlanDirective.PRESERVE_OPTIONS, 1, "Mewtwo and a secondary attacker remain available."),
                ),
                completion_condition="Survive the exchange with at least two future attack lines.",
                rationale="Avoid losing the game to a forced prize map or resource collapse.",
                required_resources=(ResourceCategory.SWITCH_MOBILITY, ResourceCategory.RECOVERY),
                axis_alignment=(0.3, -0.2, 0.7, 0.5, 0.5, 1.0),
                opponent_matchups=(-0.1, 1.0, 0.3, 0.8, 0.2, 0.0),
                prior=0.0,
                proposal_score=0.0,
            ),
            StrategicPlanCandidate(
                plan_id="clefairy_countertempo",
                title="Create a Clefairy counter-tempo pivot",
                objective=StrategicAxis.TEMPO,
                horizon_turns=2,
                steps=(
                    _step("bench", PlanDirective.BENCH_CLEFAIRY, 0, "Clefairy enters only when its prize liability is justified."),
                    _step("pivot", PlanDirective.PIVOT_ATTACKER, 0, "The active route changes on favorable terms."),
                    _step("convert", PlanDirective.TAKE_PRIZE, 1, "The counter-tempo line converts into a prize."),
                ),
                completion_condition="Gain an attack-order or weakness window without exposing a free liability.",
                rationale="Use the one-of attacker as a matchup-specific tempo tool, not a default bench target.",
                required_resources=(ResourceCategory.BASIC_POKEMON, ResourceCategory.SWITCH_MOBILITY),
                axis_alignment=(0.3, 1.0, 0.0, 0.7, 0.3, 0.5),
                opponent_matchups=(0.2, 0.5, -0.1, 0.4, 0.3, 0.2),
                prior=0.0,
                proposal_score=0.0,
            ),
        )

    @staticmethod
    def _directive_similarity(
        left: StrategicPlanCandidate,
        right: StrategicPlanCandidate,
    ) -> float:
        left_set = set(left.directive_signature)
        right_set = set(right.directive_signature)
        return len(left_set & right_set) / max(len(left_set | right_set), 1)

    def propose(
        self,
        observation: AgentObservation,
        evaluation: BeliefAwareEvaluation,
        belief: OpponentBeliefState,
        *,
        top_k: int,
        batch_index: int = 0,
    ) -> tuple[StrategicPlanCandidate, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        templates = self._templates()
        if batch_index < 0 or batch_index >= evaluation.plan.shape[0]:
            raise IndexError("batch_index is outside the evaluation batch")
        axis_values = tuple(
            float(evaluation.head_values[axis][batch_index, 0].detach().cpu())
            for axis in STRATEGIC_AXES
        )
        own = observation.public_state.player(observation.observer)
        pokemon_names = [
            pokemon.top_card.card_name
            for pokemon in own.pokemon_in_play()
        ]
        has_active_mewtwo = bool(
            own.active and own.active.top_card.card_name == "Team Rocket's Mewtwo ex"
        )
        spidops_count = sum(name == "Team Rocket's Spidops" for name in pokemon_names)

        scored: list[StrategicPlanCandidate] = []
        for template in templates:
            axis_score = sum(
                alignment * value
                for alignment, value in zip(template.axis_alignment, axis_values)
            ) / max(sum(abs(value) for value in template.axis_alignment), 1.0)
            belief_score = sum(
                matchup * probability
                for matchup, probability in zip(
                    template.opponent_matchups,
                    belief.plan.probabilities,
                )
            )
            state_score = 0.0
            if template.plan_id == "spidops_engine":
                state_score += 0.35 * (1.0 - min(spidops_count / 2.0, 1.0))
            elif template.plan_id == "mewtwo_prize_push":
                state_score += 0.25 * float(has_active_mewtwo)
            elif template.plan_id == "defensive_pivot":
                state_score += 0.2 * (own.active.damage / own.active.max_hp if own.active else 0.0)
            score = axis_score + belief_score + state_score
            scored.append(
                StrategicPlanCandidate(
                    plan_id=template.plan_id,
                    title=template.title,
                    objective=template.objective,
                    horizon_turns=template.horizon_turns,
                    steps=template.steps,
                    completion_condition=template.completion_condition,
                    rationale=template.rationale,
                    required_resources=template.required_resources,
                    axis_alignment=template.axis_alignment,
                    opponent_matchups=template.opponent_matchups,
                    prior=0.0,
                    proposal_score=score,
                )
            )

        scored.sort(key=lambda plan: plan.proposal_score, reverse=True)
        selected: list[StrategicPlanCandidate] = []
        for candidate in scored:
            if all(
                self._directive_similarity(candidate, existing)
                <= self.distinctness_threshold
                for existing in selected
            ):
                selected.append(candidate)
            if len(selected) == min(top_k, len(scored)):
                break
        if not selected:
            raise RuntimeError("plan proposer failed to produce any candidate")

        logits = torch.tensor([candidate.proposal_score for candidate in selected])
        priors = torch.softmax(logits, dim=0).tolist()
        return tuple(
            candidate.with_prior(float(prior))
            for candidate, prior in zip(selected, priors)
        )
