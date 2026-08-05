"""Deterministic benchmark for learned, calibrated selective expert search."""

from __future__ import annotations

import json
import math
import random

import torch

from .abstract_simulator import StrategicAbstractionAdapter
from .beliefs import OPPONENT_INTENTS, RESOURCE_CATEGORIES, OpponentBeliefState, OpponentPlanPosterior
from .demo import build_demo_state
from .features import StateTensorizer
from .model import Phase2AgentConfig, Phase2BeliefAwareAgent
from .observation import ActionHistoryTensorizer, AgentObservation
from .phase3_demo import _combo_belief, _combo_observation, _train_candidate_quality_policy
from .plans import TeamRocketPlanProposer
from .policy import ActionComponent, ActionPrefix, ActionToken
from .search import AutoregressivePlanExecutor, BeliefAwarePlanSearch, PlanSearchBudget
from .search_gate import (
    SEARCH_GATE_FEATURE_NAMES,
    CalibratedSearchGate,
    DecisionMode,
    PolicySignals,
    RoutineDecision,
    SearchComputeBank,
    SearchGateBatch,
    SearchGateCalibrator,
    SearchGateDataset,
    SearchGateFeatureExtractor,
    SearchGateFeatures,
    SearchGateNetwork,
    SearchGateSupervision,
    SelectiveDecision,
    SelectiveSearchPolicy,
    evaluate_search_gate,
    search_gate_loss,
)
from .strategy import STRATEGIC_AXES


TRIGGER_GROUPS: tuple[tuple[str, ...], ...] = (
    ("policy_uncertainty", "policy_entropy"),
    ("head_disagreement", "head_range"),
    ("plan_belief_entropy", "hypothesis_entropy"),
    ("own_knockout_risk", "knockout_swing"),
    ("opponent_knockout_opportunity", "knockout_swing"),
    ("own_prize_closeness", "prize_swing"),
    ("opponent_prize_closeness", "prize_swing"),
    ("evolution_window",),
    ("disruption_window",),
    ("board_collapse_risk",),
    ("proposal_ambiguity", "proposal_entropy", "optionality_spread"),
)


def _jittered_anchor(
    anchor: SearchGateFeatures,
    rng: random.Random,
    amount: float = 0.025,
) -> SearchGateFeatures:
    return SearchGateFeatures(
        tuple(
            min(1.0, max(0.0, value + rng.uniform(-amount, amount)))
            for value in anchor.values
        )
    )


def _synthetic_gate_splits(
    pivotal_anchor: SearchGateFeatures,
    routine_anchor: SearchGateFeatures,
) -> tuple[
    tuple[SearchGateSupervision, ...],
    tuple[SearchGateSupervision, ...],
    tuple[SearchGateSupervision, ...],
]:
    rng = random.Random(20260726)
    train: list[SearchGateSupervision] = []
    validation: list[SearchGateSupervision] = []
    test: list[SearchGateSupervision] = []
    for repetition in range(30):
        destination = train if repetition < 20 else validation if repetition < 25 else test
        for trigger in TRIGGER_GROUPS:
            routine_values = {
                name: rng.uniform(0.03, 0.25) for name in SEARCH_GATE_FEATURE_NAMES
            }
            routine_values["budget_fraction"] = rng.uniform(0.45, 1.0)
            routine_values["meta_value_neutrality"] = rng.uniform(0.1, 0.8)
            destination.append(
                SearchGateSupervision(
                    features=SearchGateFeatures(
                        tuple(routine_values[name] for name in SEARCH_GATE_FEATURE_NAMES)
                    ),
                    realized_search_gain=rng.uniform(0.0, 0.035),
                    search_cost=0.08,
                    source="deterministic_routine_fixture",
                )
            )

            pivotal_values = {
                name: rng.uniform(0.08, 0.32) for name in SEARCH_GATE_FEATURE_NAMES
            }
            for name in trigger:
                pivotal_values[name] = rng.uniform(0.76, 0.98)
            pivotal_values["budget_fraction"] = rng.uniform(0.45, 1.0)
            if "proposal_ambiguity" in trigger:
                pivotal_values["optionality_spread"] = rng.uniform(0.70, 0.95)
            destination.append(
                SearchGateSupervision(
                    features=SearchGateFeatures(
                        tuple(pivotal_values[name] for name in SEARCH_GATE_FEATURE_NAMES)
                    ),
                    realized_search_gain=rng.uniform(0.23, 0.36),
                    search_cost=0.08,
                    source="deterministic_pivotal_fixture",
                )
            )

    for _ in range(20):
        train.append(
            SearchGateSupervision(
                _jittered_anchor(pivotal_anchor, rng),
                realized_search_gain=rng.uniform(0.24, 0.34),
                search_cost=0.08,
                source="integrated_pivotal_anchor",
            )
        )
        train.append(
            SearchGateSupervision(
                _jittered_anchor(routine_anchor, rng),
                realized_search_gain=rng.uniform(0.0, 0.03),
                search_cost=0.08,
                source="integrated_routine_anchor",
            )
        )
    return tuple(train), tuple(validation), tuple(test)


def _evaluate_agent(
    agent: Phase2BeliefAwareAgent,
    observation: AgentObservation,
):
    state_features = StateTensorizer()(
        [observation.public_state],
        perspective=observation.observer,
    )
    history, mask = ActionHistoryTensorizer()([observation], max_events=8)
    agent.eval()
    with torch.no_grad():
        return agent(state_features, history, mask)


def _set_head_values(evaluation, values: tuple[float, ...]) -> None:
    evaluation.head_values = {
        axis: torch.tensor([[values[index]]], dtype=evaluation.plan.dtype)
        for index, axis in enumerate(STRATEGIC_AXES)
    }
    evaluation.meta_weights = torch.full_like(
        evaluation.meta_weights,
        1.0 / len(STRATEGIC_AXES),
    )
    evaluation.meta_value = torch.tensor(
        [[sum(values) / len(values)]],
        dtype=evaluation.plan.dtype,
    )


def _train_gate(network: SearchGateNetwork, batch: SearchGateBatch, steps: int = 180) -> None:
    optimizer = torch.optim.Adam(network.parameters(), lr=0.02)
    network.train()
    for _ in range(steps):
        output = network(batch.features)
        loss = search_gate_loss(output, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()


def _routine_decision(*, pivotal: bool) -> RoutineDecision:
    return RoutineDecision(
        action_prefix=ActionPrefix().append(
            ActionToken(ActionComponent.END, "routine_end", "Commit routine policy action")
        ),
        value_estimate=0.08 if pivotal else 0.24,
        policy_confidence=0.24 if pivotal else 0.98,
        policy_entropy=math.log(4.0) * 0.95 if pivotal else 0.03,
        legal_action_count=4 if pivotal else 3,
        description=(
            "Low-confidence routine choice at a combo/disruption pivot."
            if pivotal
            else "High-confidence routine sequencing choice."
        ),
    )


def run_phase4_demo() -> dict[str, object]:
    torch.manual_seed(20260726)
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
    tactical_quality_accuracy = _train_candidate_quality_policy(agent)

    pivotal_observation = _combo_observation()
    pivotal_belief = _combo_belief(pivotal_observation)
    pivotal_evaluation = _evaluate_agent(agent, pivotal_observation)
    _set_head_values(pivotal_evaluation, (-0.55, 0.15, -0.10, 0.05, -0.60, 0.72))

    routine_observation = AgentObservation.from_state(build_demo_state())
    routine_belief = OpponentBeliefState(
        match_id=routine_observation.public_state.match_id,
        observed_through_turn=routine_observation.public_state.turn_number,
        plan=OpponentPlanPosterior.normalized((0.96, 0.008, 0.008, 0.008, 0.008, 0.008)),
        resource_probabilities=(0.03, 0.04, 0.96, 0.04, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02),
    )
    routine_evaluation = _evaluate_agent(agent, routine_observation)
    _set_head_values(routine_evaluation, (0.34, 0.35, 0.36, 0.35, 0.34, 0.36))

    proposer = TeamRocketPlanProposer()
    extractor = SearchGateFeatureExtractor(proposer)
    anchor_bank = SearchComputeBank(capacity=3.0, available=3.0, refill_per_turn=0.25)
    pivotal_routine = _routine_decision(pivotal=True)
    routine_routine = _routine_decision(pivotal=False)
    pivotal_anchor = extractor.extract(
        pivotal_observation,
        pivotal_evaluation,
        pivotal_belief,
        pivotal_routine.policy_signals,
        anchor_bank,
    )
    routine_anchor = extractor.extract(
        routine_observation,
        routine_evaluation,
        routine_belief,
        routine_routine.policy_signals,
        anchor_bank,
    )

    train_examples, validation_examples, test_examples = _synthetic_gate_splits(
        pivotal_anchor,
        routine_anchor,
    )
    dataset = SearchGateDataset(train_examples + validation_examples + test_examples)
    dataset_roundtrip = SearchGateDataset.from_json(dataset.to_json()) == dataset
    train_batch = SearchGateBatch.from_examples(train_examples)
    validation_batch = SearchGateBatch.from_examples(validation_examples)
    test_batch = SearchGateBatch.from_examples(test_examples)

    network = SearchGateNetwork(hidden_dim=48)
    _train_gate(network, train_batch)
    network.eval()
    with torch.no_grad():
        validation_output = network(validation_batch.features)
        calibrator = SearchGateCalibrator.fit(
            validation_output.raw_logit,
            validation_batch.realized_gain,
            validation_batch.search_cost,
        )
        test_output = network(test_batch.features)
        metrics = evaluate_search_gate(test_output, test_batch, calibrator)

    gate = CalibratedSearchGate(network, extractor, calibrator)
    planner = BeliefAwarePlanSearch(
        proposer,
        AutoregressivePlanExecutor(agent.tactical_policy),
        StrategicAbstractionAdapter(config.candidate_feature_dim),
        budget=PlanSearchBudget(max_plans=4, max_scenarios=32),
    )
    selective = SelectiveSearchPolicy(gate, planner)

    bank = SearchComputeBank(capacity=2.0, available=2.0, refill_per_turn=0.25)
    routine_decision = selective.decide(
        routine_observation,
        routine_evaluation,
        routine_belief,
        routine_routine,
        bank,
        search_cost=1.0,
    )
    pivotal_decision = selective.decide(
        pivotal_observation,
        pivotal_evaluation,
        pivotal_belief,
        pivotal_routine,
        routine_decision.gate.bank_after,
        search_cost=1.0,
    )
    exhausted_decision = selective.decide(
        pivotal_observation,
        pivotal_evaluation,
        pivotal_belief,
        pivotal_routine,
        SearchComputeBank(capacity=2.0, available=0.0),
        search_cost=1.0,
    )

    routine_roundtrip = (
        SelectiveDecision.from_json(routine_decision.to_json()) == routine_decision
    )
    pivotal_roundtrip = (
        SelectiveDecision.from_json(pivotal_decision.to_json()) == pivotal_decision
    )
    exhausted_roundtrip = (
        SelectiveDecision.from_json(exhausted_decision.to_json()) == exhausted_decision
    )
    calibration_roundtrip = (
        SearchGateCalibrator.from_json(calibrator.to_json()) == calibrator
    )

    summary: dict[str, object] = {
        "status": "ok",
        "dataset": {
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "test_examples": len(test_examples),
            "serialization_roundtrip": dataset_roundtrip,
        },
        "calibration": {
            "temperature": calibrator.temperature,
            "threshold": calibrator.threshold,
            "serialization_roundtrip": calibration_roundtrip,
            **{name: round(value, 6) for name, value in metrics.to_dict().items()},
        },
        "feature_signals": {
            "feature_count": len(SEARCH_GATE_FEATURE_NAMES),
            "pivotal_policy_uncertainty": round(pivotal_anchor.value("policy_uncertainty"), 6),
            "routine_policy_uncertainty": round(routine_anchor.value("policy_uncertainty"), 6),
            "pivotal_head_disagreement": round(pivotal_anchor.value("head_disagreement"), 6),
            "routine_head_disagreement": round(routine_anchor.value("head_disagreement"), 6),
            "pivotal_proposal_ambiguity": round(pivotal_anchor.value("proposal_ambiguity"), 6),
            "pivotal_optionality_spread": round(pivotal_anchor.value("optionality_spread"), 6),
        },
        "selective_execution": {
            "routine_mode": routine_decision.mode.value,
            "routine_gate_probability": round(routine_decision.gate.calibrated_probability, 6),
            "routine_budget_after": routine_decision.gate.bank_after.available,
            "pivotal_mode": pivotal_decision.mode.value,
            "pivotal_gate_probability": round(pivotal_decision.gate.calibrated_probability, 6),
            "pivotal_reasons": list(pivotal_decision.gate.reasons),
            "pivotal_budget_after": pivotal_decision.gate.bank_after.available,
            "pivotal_chosen_plan": (
                None if pivotal_decision.search_result is None
                else pivotal_decision.search_result.chosen_plan_id
            ),
            "exhausted_mode": exhausted_decision.mode.value,
            "exhausted_wanted_search": exhausted_decision.gate.wanted_search,
            "exhausted_budget_limited": exhausted_decision.gate.budget_limited,
            "search_invocations_across_routine_and_pivotal": int(
                routine_decision.mode is DecisionMode.EXPERT_SEARCH
            ) + int(pivotal_decision.mode is DecisionMode.EXPERT_SEARCH),
        },
        "verification": {
            "tactical_quality_accuracy": round(tactical_quality_accuracy, 6),
            "routine_roundtrip": routine_roundtrip,
            "pivotal_roundtrip": pivotal_roundtrip,
            "exhausted_roundtrip": exhausted_roundtrip,
        },
    }

    checks = {
        "dataset serialization": dataset_roundtrip,
        "calibrator serialization": calibration_roundtrip,
        "gate test accuracy": metrics.accuracy >= 0.94,
        "critical recall": metrics.positive_invocation_rate >= 0.90,
        "routine concentration": metrics.negative_invocation_rate <= 0.10,
        "positive net search utility": metrics.realized_net_utility > 0.0,
        "near-oracle utility": metrics.realized_net_utility >= 0.85 * metrics.oracle_net_utility,
        "routine bypasses search": (
            routine_decision.mode is DecisionMode.ROUTINE_POLICY
            and routine_decision.search_result is None
            and routine_decision.gate.bank_after == bank
        ),
        "pivotal invokes search": (
            pivotal_decision.mode is DecisionMode.EXPERT_SEARCH
            and pivotal_decision.search_result is not None
            and pivotal_decision.gate.bank_after.available == 1.0
        ),
        "budget prevents search": (
            exhausted_decision.mode is DecisionMode.ROUTINE_POLICY
            and exhausted_decision.gate.wanted_search
            and exhausted_decision.gate.budget_limited
        ),
        "selective decisions serialize": (
            routine_roundtrip and pivotal_roundtrip and exhausted_roundtrip
        ),
        "tactical policy ready": tactical_quality_accuracy >= 0.90,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Phase 4 demo failed checks: {failed}; summary={summary}")
    return summary


def main() -> None:
    print(json.dumps(run_phase4_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
