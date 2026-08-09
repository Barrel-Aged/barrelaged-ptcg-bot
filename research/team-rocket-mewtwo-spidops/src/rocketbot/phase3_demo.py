"""Deterministic end-to-end benchmark for belief-aware strategic plan search."""

from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from .abstract_simulator import StrategicAbstractionAdapter
from .beliefs import (
    OPPONENT_INTENTS,
    RESOURCE_CATEGORIES,
    IntentTemplateLineGenerator,
    MinimumEnablingHypothesisGenerator,
    OpponentBeliefState,
    OpponentIntent,
    OpponentPlanPosterior,
)
from .demo import build_demo_state
from .features import StateTensorizer
from .model import Phase2AgentConfig, Phase2BeliefAwareAgent
from .observation import ActionHistoryTensorizer, AgentObservation, ObservedAction
from .plans import TeamRocketPlanProposer
from .policy import CandidateBatch
from .search import (
    AutoregressivePlanExecutor,
    BeliefAwarePlanSearch,
    PlanSearchBudget,
    PlanSearchResult,
)
from .state import PlayerId
from .strategy import STRATEGIC_AXES


def _combo_observation() -> AgentObservation:
    state = build_demo_state()
    combo_actions = (
        "play_trainer",
        "evolve",
        "attach_energy",
        "end_turn",
    )
    from .state import ActionKind

    actions = tuple(
        ObservedAction(
            turn_number=state.turn_number - 1,
            sequence_index=index,
            actor=PlayerId.OPPONENT,
            kind=ActionKind(value),
            card_name=("Public combo piece" if index < 2 else None),
        )
        for index, value in enumerate(combo_actions)
    )
    return AgentObservation.from_state(state, actions=actions)


def _combo_belief(observation: AgentObservation) -> OpponentBeliefState:
    plan = OpponentPlanPosterior.normalized((0.04, 0.07, 0.07, 0.05, 0.09, 0.68))
    resource_probabilities = (
        0.45,
        0.90,
        0.86,
        0.92,
        0.62,
        0.20,
        0.15,
        0.28,
        0.18,
        0.22,
    )
    lines = IntentTemplateLineGenerator().propose(observation, plan)
    hidden = MinimumEnablingHypothesisGenerator().generate(lines, plan)
    return OpponentBeliefState(
        match_id=observation.public_state.match_id,
        observed_through_turn=observation.public_state.turn_number,
        plan=plan,
        resource_probabilities=resource_probabilities,
        hidden_hypotheses=hidden,
    )


def _train_candidate_quality_policy(
    agent: Phase2BeliefAwareAgent,
    *,
    steps: int = 140,
) -> float:
    """Teach the reference tactical policy to read candidate quality feature."""

    policy = agent.tactical_policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    batch_size = 48
    candidate_count = 4
    policy.train()
    for _ in range(steps):
        state = torch.randn(batch_size, agent.config.state_embedding_dim) * 0.25
        plan = torch.randn(batch_size, agent.config.plan_dim) * 0.25
        belief = torch.rand(batch_size, agent.config.belief_context_dim)
        features = torch.randn(
            batch_size,
            candidate_count,
            agent.config.candidate_feature_dim,
        ) * 0.2
        quality = torch.rand(batch_size, candidate_count)
        features[:, :, -1] = quality
        candidates = CandidateBatch(
            features=features,
            legal_mask=torch.ones(batch_size, candidate_count, dtype=torch.bool),
        )
        hidden = policy.initial_hidden(state, plan, belief)
        logits = policy.score_next(hidden, candidates)
        loss = F.cross_entropy(logits, quality.argmax(dim=-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

    policy.eval()
    with torch.no_grad():
        state = torch.randn(96, agent.config.state_embedding_dim) * 0.25
        plan = torch.randn(96, agent.config.plan_dim) * 0.25
        belief = torch.rand(96, agent.config.belief_context_dim)
        features = torch.randn(96, candidate_count, agent.config.candidate_feature_dim) * 0.2
        quality = torch.rand(96, candidate_count)
        features[:, :, -1] = quality
        candidates = CandidateBatch(
            features=features,
            legal_mask=torch.ones(96, candidate_count, dtype=torch.bool),
        )
        logits = policy.score_next(
            policy.initial_hidden(state, plan, belief),
            candidates,
        )
        accuracy = logits.argmax(dim=-1).eq(quality.argmax(dim=-1)).float().mean()
    return float(accuracy.item())


def run_phase3_demo() -> dict[str, object]:
    torch.manual_seed(20260726)
    observation = _combo_observation()
    belief = _combo_belief(observation)
    config = Phase2AgentConfig(
        state_embedding_dim=32,
        plan_dim=16,
        strategic_hidden_dim=32,
        meta_hidden_dim=24,
        candidate_feature_dim=16,
        policy_hidden_dim=32,
        history_hidden_dim=24,
        belief_embedding_dim=24,
    )
    agent = Phase2BeliefAwareAgent(config)
    quality_accuracy = _train_candidate_quality_policy(agent)

    state_features = StateTensorizer()(
        [observation.public_state],
        perspective=observation.observer,
    )
    action_history, history_mask = ActionHistoryTensorizer()(
        [observation],
        max_events=8,
    )
    agent.eval()
    with torch.no_grad():
        evaluation = agent(state_features, action_history, history_mask)
        evaluation.head_values = {
            axis: torch.zeros_like(evaluation.head_values[axis])
            for axis in STRATEGIC_AXES
        }
        evaluation.meta_value = torch.zeros_like(evaluation.meta_value)
        evaluation.meta_weights = torch.full_like(
            evaluation.meta_weights,
            1.0 / len(STRATEGIC_AXES),
        )
    plan_before = evaluation.plan.detach().clone()

    adapter = StrategicAbstractionAdapter(config.candidate_feature_dim)
    executor = AutoregressivePlanExecutor(agent.tactical_policy)
    planner = BeliefAwarePlanSearch(
        TeamRocketPlanProposer(),
        executor,
        adapter,
        budget=PlanSearchBudget(max_plans=4, max_scenarios=32),
    )
    result = planner.search(observation, evaluation, belief)
    roundtrip = PlanSearchResult.from_json(result.to_json()) == result
    branch_local_plan_preserved = torch.equal(plan_before, evaluation.plan.detach())

    combo_prior = sum(
        evidence.scenario.prior_probability
        for evidence in result.scenario_evidence
        if evidence.scenario.intent is OpponentIntent.COMBO_ASSEMBLY
    )
    combo_posterior = sum(
        evidence.posterior_probability
        for evidence in result.scenario_evidence
        if evidence.scenario.intent is OpponentIntent.COMBO_ASSEMBLY
    )
    completed_prefixes = all(
        executed.prefix.is_complete
        for plan in result.plan_evaluations
        for branch in plan.branches
        for executed in branch.rollout.executed_steps
    )
    priors_normalized = abs(
        sum(plan.plan.prior for plan in result.plan_evaluations) - 1.0
    ) < 1e-6
    distinct_signatures = {
        plan.plan.directive_signature for plan in result.plan_evaluations
    }
    all_six_values_returned = all(
        len(plan.expected_strategic_values) == len(STRATEGIC_AXES)
        and all(len(branch.strategic_values) == len(STRATEGIC_AXES) for branch in plan.branches)
        for plan in result.plan_evaluations
    )

    summary: dict[str, object] = {
        "status": "ok",
        "proposal": {
            "plan_count": len(result.plan_evaluations),
            "plans": [
                {
                    "plan_id": evaluation.plan.plan_id,
                    "objective": evaluation.plan.objective.value,
                    "prior": round(evaluation.plan.prior, 6),
                    "step_count": len(evaluation.plan.steps),
                    "search_score": round(evaluation.search_score, 6),
                    "expected_value": round(evaluation.expected_value, 6),
                    "value_stddev": round(evaluation.value_stddev, 6),
                }
                for evaluation in result.plan_evaluations
            ],
            "priors_normalized": priors_normalized,
            "distinct_plan_count": len(distinct_signatures),
        },
        "selection": {
            "chosen_plan_id": result.chosen_plan_id,
            "chosen_objective": result.chosen_objective.value,
            "inferred_opponent_intent": result.inferred_opponent_intent.value,
            "plan_margin": round(result.uncertainty.plan_margin, 6),
        },
        "behavior_evidence": {
            "observed_actions": [action.value for action in result.observed_opponent_actions],
            "scenario_count": len(result.scenario_evidence),
            "combo_prior_mass": round(combo_prior, 6),
            "combo_posterior_mass": round(combo_posterior, 6),
        },
        "tactical_execution": {
            "candidate_quality_accuracy": round(quality_accuracy, 6),
            "all_prefixes_complete": completed_prefixes,
            "branch_local_plan_preserved": branch_local_plan_preserved,
        },
        "search_efficiency": result.statistics.to_dict(),
        "rich_interface": {
            "base_value_head_count": len(result.base_strategic_values),
            "meta_weight_count": len(result.meta_weights),
            "belief_serialized": result.belief == belief,
            "all_branch_value_head_count": all_six_values_returned,
            "uncertainty_returned": result.uncertainty.chosen_value_stddev >= 0.0,
            "serialization_roundtrip": roundtrip,
        },
    }

    checks = {
        "policy executes candidate quality": quality_accuracy >= 0.90,
        "few distinct complete plans": (
            2 <= len(result.plan_evaluations) <= 4
            and len(distinct_signatures) == len(result.plan_evaluations)
            and completed_prefixes
        ),
        "plan priors normalized": priors_normalized,
        "observed behavior updates scenarios": combo_posterior > combo_prior,
        "belief-aware counter-plan selected": result.chosen_plan_id == "rocket_disruption",
        "full posterior searched": (
            belief.hidden_hypotheses is not None
            and result.statistics.scenarios_searched
            == len(belief.hidden_hypotheses.hypotheses)
        ),
        "one linear branch per plan/scenario": (
            result.statistics.plan_scenario_branches
            == result.statistics.plans_searched * result.statistics.scenarios_searched
        ),
        "plan search reduces branching": result.statistics.estimated_branch_reduction >= 50.0,
        "persistent plan branch safety": branch_local_plan_preserved,
        "rich values returned": all_six_values_returned,
        "search result serialization": roundtrip,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Phase 3 demo failed checks: {failed}; summary={summary}")
    return summary


def main() -> None:
    print(json.dumps(run_phase3_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
