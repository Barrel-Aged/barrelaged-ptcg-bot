"""Belief-aware search over complete strategic plans, not raw action trees."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import torch

from .beliefs import (
    OPPONENT_INTENTS,
    RESOURCE_CATEGORIES,
    HiddenResourceHypothesis,
    OpponentBeliefState,
    OpponentIntent,
    OpponentStyle,
    ResourceVector,
)
from .model import BeliefAwareEvaluation
from .observation import AgentObservation
from .plans import PlanStep, StrategicPlanCandidate, StrategicPlanProposer
from .policy import ActionPrefix, ActionToken, CandidateBatch, NeuralAutoregressiveTacticalPolicy, TacticalCandidate
from .state import ActionKind
from .strategy import STRATEGIC_AXES, StrategicAxis


@dataclass(frozen=True, slots=True)
class BeliefScenario:
    scenario_id: str
    prior_probability: float
    intent: OpponentIntent
    resources: ResourceVector
    style: OpponentStyle
    explanation: str

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.explanation.strip():
            raise ValueError("scenario ID and explanation must be non-empty")
        if not math.isfinite(self.prior_probability) or self.prior_probability < 0.0:
            raise ValueError("scenario prior must be finite and non-negative")

    def with_prior(self, probability: float) -> "BeliefScenario":
        return BeliefScenario(
            scenario_id=self.scenario_id,
            prior_probability=probability,
            intent=self.intent,
            resources=self.resources,
            style=self.style,
            explanation=self.explanation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "prior_probability": self.prior_probability,
            "intent": self.intent.value,
            "resources": self.resources.to_dict(),
            "style": self.style.value,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BeliefScenario":
        return cls(
            scenario_id=str(raw["scenario_id"]),
            prior_probability=float(raw["prior_probability"]),
            intent=OpponentIntent(str(raw["intent"])),
            resources=ResourceVector.from_dict(raw["resources"]),
            style=OpponentStyle(str(raw["style"])),
            explanation=str(raw["explanation"]),
        )


def scenarios_from_belief(
    belief: OpponentBeliefState,
    *,
    max_scenarios: int,
) -> tuple[BeliefScenario, ...]:
    """Materialize and renormalize the top posterior scenarios for search."""

    if max_scenarios <= 0:
        raise ValueError("max_scenarios must be positive")
    scenarios: list[BeliefScenario] = []
    if belief.hidden_hypotheses is not None:
        for hypothesis in belief.hidden_hypotheses.hypotheses:
            scenarios.append(
                BeliefScenario(
                    scenario_id=hypothesis.hypothesis_id,
                    prior_probability=hypothesis.probability,
                    intent=hypothesis.intent,
                    resources=hypothesis.resources,
                    style=hypothesis.style,
                    explanation=hypothesis.explanation,
                )
            )
    else:
        marginal_resources = ResourceVector.from_mapping(
            {
                category: int(belief.resource_probabilities[index] >= 0.5)
                for index, category in enumerate(RESOURCE_CATEGORIES)
            }
        )
        for intent in OPPONENT_INTENTS:
            scenarios.append(
                BeliefScenario(
                    scenario_id=f"neural:{intent.value}",
                    prior_probability=belief.plan.probability(intent),
                    intent=intent,
                    resources=marginal_resources,
                    style=OpponentStyle.IMPERFECT,
                    explanation="Scenario synthesized from neural plan and resource marginals.",
                )
            )
    scenarios.sort(key=lambda scenario: scenario.prior_probability, reverse=True)
    selected = scenarios[:max_scenarios]
    total = sum(scenario.prior_probability for scenario in selected)
    if total <= 0.0:
        probability = 1.0 / len(selected)
        return tuple(scenario.with_prior(probability) for scenario in selected)
    return tuple(
        scenario.with_prior(scenario.prior_probability / total) for scenario in selected
    )


class OpponentBehaviorModel(Protocol):
    def predict(
        self,
        observation: AgentObservation,
        scenario: BeliefScenario,
    ) -> Sequence[ActionKind]: ...


class IntentOpponentBehaviorModel:
    """Reference behavior predictor keyed by inferred intent and play style."""

    SIGNATURES: Mapping[OpponentIntent, tuple[ActionKind, ...]] = {
        OpponentIntent.SETUP: (
            ActionKind.PLAY_POKEMON,
            ActionKind.PLAY_TRAINER,
            ActionKind.ATTACH_ENERGY,
            ActionKind.END_TURN,
        ),
        OpponentIntent.PRIZE_RACE: (ActionKind.ATTACH_ENERGY, ActionKind.ATTACK),
        OpponentIntent.LOCK_DISRUPTION: (
            ActionKind.PLAY_TRAINER,
            ActionKind.USE_ABILITY,
            ActionKind.END_TURN,
        ),
        OpponentIntent.BOARD_DENIAL: (ActionKind.PLAY_TRAINER, ActionKind.ATTACK),
        OpponentIntent.RESOURCE_CONSERVATION: (
            ActionKind.USE_ABILITY,
            ActionKind.END_TURN,
        ),
        OpponentIntent.COMBO_ASSEMBLY: (
            ActionKind.PLAY_TRAINER,
            ActionKind.EVOLVE,
            ActionKind.ATTACH_ENERGY,
            ActionKind.END_TURN,
        ),
    }

    def predict(
        self,
        observation: AgentObservation,
        scenario: BeliefScenario,
    ) -> tuple[ActionKind, ...]:
        signature = self.SIGNATURES[scenario.intent]
        if scenario.style is OpponentStyle.RATIONAL:
            return signature
        if scenario.style is OpponentStyle.IMPERFECT:
            return signature[:-1] if len(signature) > 1 else signature
        return (ActionKind.USE_ABILITY, ActionKind.END_TURN)


def _sequence_similarity(
    observed: Sequence[ActionKind],
    predicted: Sequence[ActionKind],
) -> float:
    if not observed and not predicted:
        return 1.0
    if not observed or not predicted:
        return 0.0
    table = [[0] * (len(predicted) + 1) for _ in range(len(observed) + 1)]
    for row in range(1, len(observed) + 1):
        for column in range(1, len(predicted) + 1):
            if observed[row - 1] is predicted[column - 1]:
                table[row][column] = table[row - 1][column - 1] + 1
            else:
                table[row][column] = max(
                    table[row - 1][column], table[row][column - 1]
                )
    return table[-1][-1] / max(len(observed), len(predicted))


class OpponentBehaviorComparator:
    def __init__(self, *, likelihood_floor: float = 0.02, sharpness: float = 4.0) -> None:
        if not 0.0 < likelihood_floor < 1.0:
            raise ValueError("likelihood_floor must lie between zero and one")
        if sharpness <= 0.0:
            raise ValueError("sharpness must be positive")
        self.likelihood_floor = likelihood_floor
        self.sharpness = sharpness

    def likelihood(
        self,
        observed: Sequence[ActionKind],
        predicted: Sequence[ActionKind],
    ) -> float:
        if not observed:
            return 1.0
        similarity = _sequence_similarity(observed, predicted)
        length_gap = abs(len(observed) - len(predicted)) / max(
            len(observed), len(predicted), 1
        )
        return self.likelihood_floor + (1.0 - self.likelihood_floor) * math.exp(
            -self.sharpness * (1.0 - similarity) - length_gap
        )


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    scenario: BeliefScenario
    predicted_actions: tuple[ActionKind, ...]
    behavior_likelihood: float
    posterior_probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.behavior_likelihood <= 1.0:
            raise ValueError("behavior likelihood must lie in [0, 1]")
        if not 0.0 <= self.posterior_probability <= 1.0:
            raise ValueError("scenario posterior must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "predicted_actions": [action.value for action in self.predicted_actions],
            "behavior_likelihood": self.behavior_likelihood,
            "posterior_probability": self.posterior_probability,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ScenarioEvidence":
        return cls(
            scenario=BeliefScenario.from_dict(raw["scenario"]),
            predicted_actions=tuple(ActionKind(value) for value in raw["predicted_actions"]),
            behavior_likelihood=float(raw["behavior_likelihood"]),
            posterior_probability=float(raw["posterior_probability"]),
        )


@dataclass(frozen=True, slots=True)
class RolloutOutcome:
    axis_deltas: tuple[float, ...]
    terminal_value: float
    objective_achievement: float
    predicted_opponent_actions: tuple[ActionKind, ...]
    resource_cost: float
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.axis_deltas) != len(STRATEGIC_AXES):
            raise ValueError("rollout axis deltas must contain six values")
        numeric = self.axis_deltas + (
            self.terminal_value,
            self.objective_achievement,
            self.resource_cost,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("rollout outcome values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_deltas": list(self.axis_deltas),
            "terminal_value": self.terminal_value,
            "objective_achievement": self.objective_achievement,
            "predicted_opponent_actions": [
                action.value for action in self.predicted_opponent_actions
            ],
            "resource_cost": self.resource_cost,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RolloutOutcome":
        return cls(
            axis_deltas=tuple(float(value) for value in raw["axis_deltas"]),
            terminal_value=float(raw["terminal_value"]),
            objective_achievement=float(raw["objective_achievement"]),
            predicted_opponent_actions=tuple(
                ActionKind(value) for value in raw["predicted_opponent_actions"]
            ),
            resource_cost=float(raw["resource_cost"]),
            notes=tuple(str(note) for note in raw.get("notes", [])),
        )


class PlanExecutionAdapter(Protocol):
    """Branch-local simulator seam used by the autoregressive executor."""

    def start(
        self,
        observation: AgentObservation,
        plan: StrategicPlanCandidate,
        scenario: BeliefScenario,
    ) -> Any: ...

    def next_candidates(
        self,
        rollout_state: Any,
        plan: StrategicPlanCandidate,
        step: PlanStep,
        prefix: ActionPrefix,
        scenario: BeliefScenario,
    ) -> Sequence[TacticalCandidate]: ...

    def commit_action(
        self,
        rollout_state: Any,
        plan: StrategicPlanCandidate,
        step: PlanStep,
        prefix: ActionPrefix,
        scenario: BeliefScenario,
    ) -> Any: ...

    def evaluate(
        self,
        rollout_state: Any,
        plan: StrategicPlanCandidate,
        scenario: BeliefScenario,
    ) -> RolloutOutcome: ...


@dataclass(frozen=True, slots=True)
class TacticalSelection:
    token: ActionToken
    selected_probability: float
    log_probability: float
    entropy: float
    candidate_count: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.selected_probability <= 1.0:
            raise ValueError("selected probability must lie in [0, 1]")
        if not all(math.isfinite(value) for value in (
            self.log_probability, self.entropy
        )):
            raise ValueError("selection log probability and entropy must be finite")
        if self.entropy < 0.0 or self.candidate_count <= 0:
            raise ValueError("selection entropy/count are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token.to_dict(),
            "selected_probability": self.selected_probability,
            "log_probability": self.log_probability,
            "entropy": self.entropy,
            "candidate_count": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TacticalSelection":
        return cls(
            token=ActionToken.from_dict(raw["token"]),
            selected_probability=float(raw["selected_probability"]),
            log_probability=float(raw["log_probability"]),
            entropy=float(raw["entropy"]),
            candidate_count=int(raw["candidate_count"]),
        )


@dataclass(frozen=True, slots=True)
class ExecutedPlanStep:
    step: PlanStep
    prefix: ActionPrefix
    selections: tuple[TacticalSelection, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.to_dict(),
            "prefix": [token.to_dict() for token in self.prefix.tokens],
            "selections": [selection.to_dict() for selection in self.selections],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExecutedPlanStep":
        return cls(
            step=PlanStep.from_dict(raw["step"]),
            prefix=ActionPrefix(
                tuple(ActionToken.from_dict(token) for token in raw["prefix"])
            ),
            selections=tuple(
                TacticalSelection.from_dict(selection) for selection in raw["selections"]
            ),
        )


@dataclass(frozen=True, slots=True)
class TacticalRollout:
    executed_steps: tuple[ExecutedPlanStep, ...]
    outcome: RolloutOutcome
    candidates_scored: int
    estimated_raw_action_sequences: int
    mean_log_probability: float
    mean_entropy: float

    def __post_init__(self) -> None:
        if not self.executed_steps:
            raise ValueError("tactical rollout must execute at least one plan step")
        if any(not step.prefix.is_complete for step in self.executed_steps):
            raise ValueError("all executed plan steps must end with END")
        if self.candidates_scored <= 0 or self.estimated_raw_action_sequences <= 0:
            raise ValueError("rollout search counts must be positive")
        if not math.isfinite(self.mean_log_probability):
            raise ValueError("rollout mean log probability must be finite")
        if not math.isfinite(self.mean_entropy) or self.mean_entropy < 0.0:
            raise ValueError("rollout mean entropy must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_steps": [step.to_dict() for step in self.executed_steps],
            "outcome": self.outcome.to_dict(),
            "candidates_scored": self.candidates_scored,
            "estimated_raw_action_sequences": self.estimated_raw_action_sequences,
            "mean_log_probability": self.mean_log_probability,
            "mean_entropy": self.mean_entropy,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TacticalRollout":
        return cls(
            executed_steps=tuple(
                ExecutedPlanStep.from_dict(step) for step in raw["executed_steps"]
            ),
            outcome=RolloutOutcome.from_dict(raw["outcome"]),
            candidates_scored=int(raw["candidates_scored"]),
            estimated_raw_action_sequences=int(raw["estimated_raw_action_sequences"]),
            mean_log_probability=float(raw["mean_log_probability"]),
            mean_entropy=float(raw["mean_entropy"]),
        )


class AutoregressivePlanExecutor:
    """Greedily execute each proposed plan through the tactical policy."""

    def __init__(
        self,
        policy: NeuralAutoregressiveTacticalPolicy,
        *,
        max_tokens_per_step: int = 8,
        raw_sequence_cap: int = 10**12,
    ) -> None:
        if max_tokens_per_step <= 0 or raw_sequence_cap <= 0:
            raise ValueError("executor limits must be positive")
        self.policy = policy
        self.max_tokens_per_step = max_tokens_per_step
        self.raw_sequence_cap = raw_sequence_cap

    def execute(
        self,
        observation: AgentObservation,
        evaluation: BeliefAwareEvaluation,
        belief: OpponentBeliefState,
        plan: StrategicPlanCandidate,
        scenario: BeliefScenario,
        adapter: PlanExecutionAdapter,
        *,
        batch_index: int = 0,
    ) -> TacticalRollout:
        device = evaluation.state_embedding.device
        dtype = evaluation.state_embedding.dtype
        state_embedding = evaluation.state_embedding[batch_index : batch_index + 1].clone()
        branch_plan = evaluation.plan[batch_index : batch_index + 1].clone()
        belief_context = torch.tensor(
            [belief.context_vector],
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            hidden = self.policy.initial_hidden(
                state_embedding,
                branch_plan,
                belief_context,
            )
            rollout_state = adapter.start(observation, plan, scenario)
            executed_steps: list[ExecutedPlanStep] = []
            all_selections: list[TacticalSelection] = []
            candidates_scored = 0
            raw_sequences = 1

            for step in plan.steps:
                prefix = ActionPrefix()
                step_selections: list[TacticalSelection] = []
                for _ in range(self.max_tokens_per_step):
                    candidates = tuple(
                        adapter.next_candidates(
                            rollout_state,
                            plan,
                            step,
                            prefix,
                            scenario,
                        )
                    )
                    if not candidates:
                        raise RuntimeError(
                            f"adapter returned no candidates for {plan.plan_id}/{step.step_id}"
                        )
                    batch = CandidateBatch.from_single_step(
                        candidates,
                        dtype=dtype,
                        device=device,
                    )
                    logits = self.policy.score_next(hidden, batch)
                    probabilities = torch.softmax(logits, dim=-1)
                    selected_index = int(probabilities[0].argmax().item())
                    selected = candidates[selected_index]
                    selected_probability = float(probabilities[0, selected_index].item())
                    legal_probabilities = probabilities[0][batch.legal_mask[0]]
                    entropy = float(
                        (-(legal_probabilities * legal_probabilities.clamp_min(1e-12).log()).sum()).item()
                    )
                    legal_count = int(batch.legal_mask[0].sum().item())
                    selection = TacticalSelection(
                        token=selected.token,
                        selected_probability=selected_probability,
                        log_probability=math.log(max(selected_probability, 1e-12)),
                        entropy=entropy,
                        candidate_count=legal_count,
                    )
                    prefix = prefix.append(selected.token)
                    hidden = self.policy.advance(hidden, batch.features[:, selected_index])
                    step_selections.append(selection)
                    all_selections.append(selection)
                    candidates_scored += legal_count
                    raw_sequences = min(
                        self.raw_sequence_cap,
                        raw_sequences * max(legal_count, 1),
                    )
                    if prefix.is_complete:
                        rollout_state = adapter.commit_action(
                            rollout_state,
                            plan,
                            step,
                            prefix,
                            scenario,
                        )
                        break
                if not prefix.is_complete:
                    raise RuntimeError(
                        f"tactical decoder did not terminate {plan.plan_id}/{step.step_id}"
                    )
                executed_steps.append(
                    ExecutedPlanStep(step, prefix, tuple(step_selections))
                )

            outcome = adapter.evaluate(rollout_state, plan, scenario)
        mean_log_probability = sum(
            selection.log_probability for selection in all_selections
        ) / max(len(all_selections), 1)
        mean_entropy = sum(selection.entropy for selection in all_selections) / max(
            len(all_selections), 1
        )
        return TacticalRollout(
            executed_steps=tuple(executed_steps),
            outcome=outcome,
            candidates_scored=candidates_scored,
            estimated_raw_action_sequences=raw_sequences,
            mean_log_probability=mean_log_probability,
            mean_entropy=mean_entropy,
        )


@dataclass(frozen=True, slots=True)
class PlanBranchEvaluation:
    scenario_id: str
    scenario_probability: float
    behavior_likelihood: float
    strategic_values: tuple[float, ...]
    aggregate_value: float
    objective_value: float
    rollout: TacticalRollout

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("branch scenario ID must be non-empty")
        if not 0.0 <= self.scenario_probability <= 1.0:
            raise ValueError("branch scenario probability must lie in [0, 1]")
        if not 0.0 <= self.behavior_likelihood <= 1.0:
            raise ValueError("branch behavior likelihood must lie in [0, 1]")
        if len(self.strategic_values) != len(STRATEGIC_AXES):
            raise ValueError("branch strategic values must contain six values")
        if not all(math.isfinite(value) for value in self.strategic_values + (
            self.aggregate_value, self.objective_value
        )):
            raise ValueError("branch evaluation values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_probability": self.scenario_probability,
            "behavior_likelihood": self.behavior_likelihood,
            "strategic_values": list(self.strategic_values),
            "aggregate_value": self.aggregate_value,
            "objective_value": self.objective_value,
            "rollout": self.rollout.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanBranchEvaluation":
        return cls(
            scenario_id=str(raw["scenario_id"]),
            scenario_probability=float(raw["scenario_probability"]),
            behavior_likelihood=float(raw["behavior_likelihood"]),
            strategic_values=tuple(float(value) for value in raw["strategic_values"]),
            aggregate_value=float(raw["aggregate_value"]),
            objective_value=float(raw["objective_value"]),
            rollout=TacticalRollout.from_dict(raw["rollout"]),
        )


@dataclass(frozen=True, slots=True)
class PlanEvaluation:
    plan: StrategicPlanCandidate
    branches: tuple[PlanBranchEvaluation, ...]
    expected_strategic_values: tuple[float, ...]
    expected_value: float
    value_stddev: float
    objective_score: float
    policy_confidence: float
    policy_entropy: float
    search_score: float

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("plan evaluation requires belief branches")
        if not math.isclose(
            sum(branch.scenario_probability for branch in self.branches),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("plan branch probabilities must sum to one")
        if len(self.expected_strategic_values) != len(STRATEGIC_AXES):
            raise ValueError("expected strategic values must contain six values")
        numeric = self.expected_strategic_values + (
            self.expected_value,
            self.value_stddev,
            self.objective_score,
            self.policy_confidence,
            self.policy_entropy,
            self.search_score,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("plan evaluation values must be finite")
        if self.value_stddev < 0.0 or self.policy_entropy < 0.0:
            raise ValueError("plan uncertainty values cannot be negative")
        if not 0.0 <= self.policy_confidence <= 1.0:
            raise ValueError("policy confidence must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "branches": [branch.to_dict() for branch in self.branches],
            "expected_strategic_values": list(self.expected_strategic_values),
            "expected_value": self.expected_value,
            "value_stddev": self.value_stddev,
            "objective_score": self.objective_score,
            "policy_confidence": self.policy_confidence,
            "policy_entropy": self.policy_entropy,
            "search_score": self.search_score,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanEvaluation":
        return cls(
            plan=StrategicPlanCandidate.from_dict(raw["plan"]),
            branches=tuple(
                PlanBranchEvaluation.from_dict(branch) for branch in raw["branches"]
            ),
            expected_strategic_values=tuple(
                float(value) for value in raw["expected_strategic_values"]
            ),
            expected_value=float(raw["expected_value"]),
            value_stddev=float(raw["value_stddev"]),
            objective_score=float(raw["objective_score"]),
            policy_confidence=float(raw["policy_confidence"]),
            policy_entropy=float(raw["policy_entropy"]),
            search_score=float(raw["search_score"]),
        )


@dataclass(frozen=True, slots=True)
class PlanSearchUncertainty:
    scenario_entropy: float
    proposal_entropy: float
    chosen_value_stddev: float
    plan_margin: float

    def __post_init__(self) -> None:
        values = (
            self.scenario_entropy,
            self.proposal_entropy,
            self.chosen_value_stddev,
            self.plan_margin,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("search uncertainty values must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "scenario_entropy": self.scenario_entropy,
            "proposal_entropy": self.proposal_entropy,
            "chosen_value_stddev": self.chosen_value_stddev,
            "plan_margin": self.plan_margin,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanSearchUncertainty":
        return cls(**{key: float(raw[key]) for key in (
            "scenario_entropy", "proposal_entropy", "chosen_value_stddev", "plan_margin"
        )})


@dataclass(frozen=True, slots=True)
class PlanSearchStatistics:
    plans_proposed: int
    plans_searched: int
    scenarios_searched: int
    plan_scenario_branches: int
    tactical_candidates_scored: int
    estimated_raw_action_sequences: int
    estimated_branch_reduction: float

    def __post_init__(self) -> None:
        counts = (
            self.plans_proposed,
            self.plans_searched,
            self.scenarios_searched,
            self.plan_scenario_branches,
            self.tactical_candidates_scored,
            self.estimated_raw_action_sequences,
        )
        if any(count <= 0 for count in counts):
            raise ValueError("search statistics counts must be positive")
        if not math.isfinite(self.estimated_branch_reduction) or self.estimated_branch_reduction < 1.0:
            raise ValueError("estimated branch reduction must be finite and at least one")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "plans_proposed": self.plans_proposed,
            "plans_searched": self.plans_searched,
            "scenarios_searched": self.scenarios_searched,
            "plan_scenario_branches": self.plan_scenario_branches,
            "tactical_candidates_scored": self.tactical_candidates_scored,
            "estimated_raw_action_sequences": self.estimated_raw_action_sequences,
            "estimated_branch_reduction": self.estimated_branch_reduction,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanSearchStatistics":
        return cls(
            plans_proposed=int(raw["plans_proposed"]),
            plans_searched=int(raw["plans_searched"]),
            scenarios_searched=int(raw["scenarios_searched"]),
            plan_scenario_branches=int(raw["plan_scenario_branches"]),
            tactical_candidates_scored=int(raw["tactical_candidates_scored"]),
            estimated_raw_action_sequences=int(raw["estimated_raw_action_sequences"]),
            estimated_branch_reduction=float(raw["estimated_branch_reduction"]),
        )


@dataclass(frozen=True, slots=True)
class PlanSearchResult:
    match_id: str
    observed_opponent_actions: tuple[ActionKind, ...]
    chosen_plan_id: str
    chosen_objective: StrategicAxis
    inferred_opponent_intent: OpponentIntent
    base_strategic_values: tuple[float, ...]
    base_meta_value: float
    meta_weights: tuple[float, ...]
    belief: OpponentBeliefState
    scenario_evidence: tuple[ScenarioEvidence, ...]
    plan_evaluations: tuple[PlanEvaluation, ...]
    uncertainty: PlanSearchUncertainty
    statistics: PlanSearchStatistics
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.match_id.strip():
            raise ValueError("search result match ID must be non-empty")
        if not self.plan_evaluations or not self.scenario_evidence:
            raise ValueError("search result requires plans and scenario evidence")
        if self.chosen_plan_id not in {
            evaluation.plan.plan_id for evaluation in self.plan_evaluations
        }:
            raise ValueError("chosen plan is absent from plan evaluations")
        if len(self.base_strategic_values) != len(STRATEGIC_AXES):
            raise ValueError("base strategic values must contain six values")
        if len(self.meta_weights) != len(STRATEGIC_AXES):
            raise ValueError("meta weights must contain six values")
        if not math.isclose(sum(self.meta_weights), 1.0, abs_tol=1e-6):
            raise ValueError("meta weights must sum to one")
        if not math.isclose(
            sum(evidence.posterior_probability for evidence in self.scenario_evidence),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("scenario evidence probabilities must sum to one")
        if not math.isclose(
            sum(evaluation.plan.prior for evaluation in self.plan_evaluations),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("plan priors must sum to one")

    @property
    def chosen(self) -> PlanEvaluation:
        return next(
            evaluation
            for evaluation in self.plan_evaluations
            if evaluation.plan.plan_id == self.chosen_plan_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "observed_opponent_actions": [
                action.value for action in self.observed_opponent_actions
            ],
            "chosen_plan_id": self.chosen_plan_id,
            "chosen_objective": self.chosen_objective.value,
            "inferred_opponent_intent": self.inferred_opponent_intent.value,
            "base_strategic_values": list(self.base_strategic_values),
            "base_meta_value": self.base_meta_value,
            "meta_weights": list(self.meta_weights),
            "belief": self.belief.to_dict(),
            "scenario_evidence": [evidence.to_dict() for evidence in self.scenario_evidence],
            "plan_evaluations": [evaluation.to_dict() for evaluation in self.plan_evaluations],
            "uncertainty": self.uncertainty.to_dict(),
            "statistics": self.statistics.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanSearchResult":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            match_id=str(raw["match_id"]),
            observed_opponent_actions=tuple(
                ActionKind(value) for value in raw["observed_opponent_actions"]
            ),
            chosen_plan_id=str(raw["chosen_plan_id"]),
            chosen_objective=StrategicAxis(str(raw["chosen_objective"])),
            inferred_opponent_intent=OpponentIntent(str(raw["inferred_opponent_intent"])),
            base_strategic_values=tuple(
                float(value) for value in raw["base_strategic_values"]
            ),
            base_meta_value=float(raw["base_meta_value"]),
            meta_weights=tuple(float(value) for value in raw["meta_weights"]),
            belief=OpponentBeliefState.from_dict(raw["belief"]),
            scenario_evidence=tuple(
                ScenarioEvidence.from_dict(evidence) for evidence in raw["scenario_evidence"]
            ),
            plan_evaluations=tuple(
                PlanEvaluation.from_dict(evaluation) for evaluation in raw["plan_evaluations"]
            ),
            uncertainty=PlanSearchUncertainty.from_dict(raw["uncertainty"]),
            statistics=PlanSearchStatistics.from_dict(raw["statistics"]),
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
    def from_json(cls, encoded: str) -> "PlanSearchResult":
        return cls.from_dict(json.loads(encoded))


@dataclass(frozen=True, slots=True)
class PlanSearchBudget:
    max_plans: int = 4
    max_scenarios: int = 32

    def __post_init__(self) -> None:
        if self.max_plans <= 0 or self.max_scenarios <= 0:
            raise ValueError("search budget limits must be positive")


@dataclass(frozen=True, slots=True)
class PlanSearchConfig:
    risk_aversion: float = 0.15
    objective_weight: float = 0.15
    prior_weight: float = 0.05
    policy_confidence_weight: float = 0.03
    terminal_blend: float = 0.25
    resource_cost_weight: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.terminal_blend <= 1.0:
            raise ValueError("terminal_blend must lie in [0, 1]")
        if any(
            value < 0.0
            for value in (
                self.risk_aversion,
                self.objective_weight,
                self.prior_weight,
                self.policy_confidence_weight,
                self.resource_cost_weight,
            )
        ):
            raise ValueError("search weights cannot be negative")


class BeliefAwarePlanSearch:
    """Evaluate a few complete plans across posterior-weighted scenarios."""

    def __init__(
        self,
        proposer: StrategicPlanProposer,
        executor: AutoregressivePlanExecutor,
        adapter: PlanExecutionAdapter,
        *,
        behavior_model: OpponentBehaviorModel | None = None,
        behavior_comparator: OpponentBehaviorComparator | None = None,
        budget: PlanSearchBudget | None = None,
        config: PlanSearchConfig | None = None,
    ) -> None:
        self.proposer = proposer
        self.executor = executor
        self.adapter = adapter
        self.behavior_model = behavior_model or IntentOpponentBehaviorModel()
        self.behavior_comparator = behavior_comparator or OpponentBehaviorComparator()
        self.budget = budget or PlanSearchBudget()
        self.config = config or PlanSearchConfig()

    @staticmethod
    def _recent_opponent_actions(
        observation: AgentObservation,
    ) -> tuple[ActionKind, ...]:
        opponent_actions = [
            action
            for action in observation.actions
            if action.actor is observation.observer.other
        ]
        if not opponent_actions:
            return ()
        latest_turn = max(action.turn_number for action in opponent_actions)
        return tuple(
            action.kind
            for action in sorted(
                (action for action in opponent_actions if action.turn_number == latest_turn),
                key=lambda action: action.sequence_index,
            )
        )

    def _scenario_evidence(
        self,
        observation: AgentObservation,
        scenarios: Sequence[BeliefScenario],
        observed: Sequence[ActionKind],
    ) -> tuple[ScenarioEvidence, ...]:
        provisional: list[tuple[BeliefScenario, tuple[ActionKind, ...], float, float]] = []
        for scenario in scenarios:
            predicted = tuple(self.behavior_model.predict(observation, scenario))
            likelihood = self.behavior_comparator.likelihood(observed, predicted)
            provisional.append(
                (
                    scenario,
                    predicted,
                    likelihood,
                    scenario.prior_probability * likelihood,
                )
            )
        denominator = sum(item[3] for item in provisional)
        if denominator <= 0.0:
            denominator = 1.0
        return tuple(
            ScenarioEvidence(
                scenario=scenario,
                predicted_actions=predicted,
                behavior_likelihood=likelihood,
                posterior_probability=weight / denominator,
            )
            for scenario, predicted, likelihood, weight in provisional
        )

    def search(
        self,
        observation: AgentObservation,
        evaluation: BeliefAwareEvaluation,
        belief: OpponentBeliefState,
        *,
        batch_index: int = 0,
    ) -> PlanSearchResult:
        if observation.public_state.match_id != belief.match_id:
            raise ValueError("observation and belief must belong to the same match")
        original_plan = evaluation.plan.detach().clone()
        plans = tuple(
            self.proposer.propose(
                observation,
                evaluation,
                belief,
                top_k=self.budget.max_plans,
                batch_index=batch_index,
            )
        )
        if not plans:
            raise RuntimeError("plan proposer returned no plans")
        if not math.isclose(sum(plan.prior for plan in plans), 1.0, abs_tol=1e-6):
            raise ValueError("proposed plan priors must sum to one")
        scenarios = scenarios_from_belief(
            belief,
            max_scenarios=self.budget.max_scenarios,
        )
        observed = self._recent_opponent_actions(observation)
        evidence = self._scenario_evidence(observation, scenarios, observed)

        base_values = tuple(
            float(evaluation.head_values[axis][batch_index, 0].detach().cpu())
            for axis in STRATEGIC_AXES
        )
        meta_weights = tuple(
            float(value)
            for value in evaluation.meta_weights[batch_index].detach().cpu().tolist()
        )
        base_meta = float(evaluation.meta_value[batch_index, 0].detach().cpu())
        plan_evaluations: list[PlanEvaluation] = []

        total_candidates = 0
        total_raw_sequences = 0
        total_branches = 0
        for plan in plans:
            branches: list[PlanBranchEvaluation] = []
            for scenario_evidence in evidence:
                rollout = self.executor.execute(
                    observation,
                    evaluation,
                    belief,
                    plan,
                    scenario_evidence.scenario,
                    self.adapter,
                    batch_index=batch_index,
                )
                strategic_values = tuple(
                    max(-1.0, min(1.0, base + delta))
                    for base, delta in zip(base_values, rollout.outcome.axis_deltas)
                )
                strategic_aggregate = sum(
                    weight * value
                    for weight, value in zip(meta_weights, strategic_values)
                )
                aggregate = (
                    (1.0 - self.config.terminal_blend) * strategic_aggregate
                    + self.config.terminal_blend * rollout.outcome.terminal_value
                    - self.config.resource_cost_weight * rollout.outcome.resource_cost
                )
                objective_index = STRATEGIC_AXES.index(plan.objective)
                objective_value = 0.5 * strategic_values[objective_index] + 0.5 * min(
                    1.0, max(-1.0, rollout.outcome.objective_achievement)
                )
                branches.append(
                    PlanBranchEvaluation(
                        scenario_id=scenario_evidence.scenario.scenario_id,
                        scenario_probability=scenario_evidence.posterior_probability,
                        behavior_likelihood=scenario_evidence.behavior_likelihood,
                        strategic_values=strategic_values,
                        aggregate_value=aggregate,
                        objective_value=objective_value,
                        rollout=rollout,
                    )
                )
                total_candidates += rollout.candidates_scored
                total_raw_sequences += rollout.estimated_raw_action_sequences
                total_branches += 1

            expected_values = tuple(
                sum(
                    branch.scenario_probability * branch.strategic_values[index]
                    for branch in branches
                )
                for index in range(len(STRATEGIC_AXES))
            )
            expected = sum(
                branch.scenario_probability * branch.aggregate_value for branch in branches
            )
            variance = sum(
                branch.scenario_probability * (branch.aggregate_value - expected) ** 2
                for branch in branches
            )
            value_stddev = math.sqrt(max(variance, 0.0))
            objective_score = sum(
                branch.scenario_probability * branch.objective_value for branch in branches
            )
            weighted_log_probability = sum(
                branch.scenario_probability * branch.rollout.mean_log_probability
                for branch in branches
            )
            policy_confidence = math.exp(weighted_log_probability)
            policy_entropy = sum(
                branch.scenario_probability * branch.rollout.mean_entropy
                for branch in branches
            )
            search_score = (
                expected
                - self.config.risk_aversion * value_stddev
                + self.config.objective_weight * objective_score
                + self.config.prior_weight * math.log(max(plan.prior, 1e-12))
                + self.config.policy_confidence_weight * policy_confidence
            )
            plan_evaluations.append(
                PlanEvaluation(
                    plan=plan,
                    branches=tuple(branches),
                    expected_strategic_values=expected_values,
                    expected_value=expected,
                    value_stddev=value_stddev,
                    objective_score=objective_score,
                    policy_confidence=policy_confidence,
                    policy_entropy=policy_entropy,
                    search_score=search_score,
                )
            )

        plan_evaluations.sort(key=lambda item: item.search_score, reverse=True)
        chosen = plan_evaluations[0]
        plan_margin = (
            chosen.search_score - plan_evaluations[1].search_score
            if len(plan_evaluations) > 1
            else 0.0
        )
        scenario_entropy = -sum(
            item.posterior_probability * math.log(item.posterior_probability)
            for item in evidence
            if item.posterior_probability > 0.0
        )
        proposal_entropy = -sum(
            plan.prior * math.log(plan.prior) for plan in plans if plan.prior > 0.0
        )
        raw_sequences = max(total_raw_sequences, total_branches)
        statistics = PlanSearchStatistics(
            plans_proposed=len(plans),
            plans_searched=len(plan_evaluations),
            scenarios_searched=len(evidence),
            plan_scenario_branches=total_branches,
            tactical_candidates_scored=total_candidates,
            estimated_raw_action_sequences=raw_sequences,
            estimated_branch_reduction=raw_sequences / max(total_branches, 1),
        )
        if not torch.equal(evaluation.plan.detach(), original_plan):
            raise RuntimeError("search mutated the persistent plan tensor")
        return PlanSearchResult(
            match_id=observation.public_state.match_id,
            observed_opponent_actions=tuple(observed),
            chosen_plan_id=chosen.plan.plan_id,
            chosen_objective=chosen.plan.objective,
            inferred_opponent_intent=belief.plan.most_likely,
            base_strategic_values=base_values,
            base_meta_value=base_meta,
            meta_weights=meta_weights,
            belief=belief,
            scenario_evidence=evidence,
            plan_evaluations=tuple(plan_evaluations),
            uncertainty=PlanSearchUncertainty(
                scenario_entropy=scenario_entropy,
                proposal_entropy=proposal_entropy,
                chosen_value_stddev=chosen.value_stddev,
                plan_margin=plan_margin,
            ),
            statistics=statistics,
        )
