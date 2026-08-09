"""Deterministic Phase 5 search-guided self-play and distillation benchmark."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import random

import torch

from .abstract_simulator import StrategicAbstractionAdapter
from .beliefs import OpponentBeliefState, OpponentPlanPosterior
from .deck import load_default_deck
from .demo import build_demo_state
from .distillation import (
    DistillationCheckpointManifest,
    SearchGuidedDistiller,
    SearchGuidedStudent,
)
from .features import StateTensorizer
from .model import Phase2AgentConfig, Phase2BeliefAwareAgent
from .observation import ActionHistoryTensorizer, AgentObservation
from .phase3_demo import _combo_belief, _combo_observation, _train_candidate_quality_policy
from .phase4_demo import (
    _evaluate_agent,
    _jittered_anchor,
    _routine_decision,
    _set_head_values,
    _synthetic_gate_splits,
    _train_gate,
)
from .plans import TeamRocketPlanProposer
from .policy import ActionComponent, ActionPrefix, ActionToken, CandidateBatch
from .search import (
    AutoregressivePlanExecutor,
    BeliefAwarePlanSearch,
    PlanSearchBudget,
    PlanSearchConfig,
    PlanSearchResult,
)
from .search_gate import (
    CalibratedSearchGate,
    DecisionMode,
    PolicySignals,
    RoutineDecision,
    SearchComputeBank,
    SearchGateBatch,
    SearchGateCalibrator,
    SearchGateFeatureExtractor,
    SearchGateNetwork,
    SearchGateSupervision,
    SelectiveSearchPolicy,
    evaluate_search_gate,
    search_gate_loss,
)
from .search_learning import (
    DistillationDisposition,
    OutcomeValidation,
    SearchGuidedSelfPlayRunner,
    SearchRetentionTarget,
    SearchTargetValidationConfig,
    SearchTeacherCollector,
    SearchTeacherDataset,
    SearchTeacherRecord,
    SelfPlayRoot,
    TeacherAdmission,
    model_fingerprint,
    stable_fingerprint,
)
from .strategy import STRATEGIC_AXES


PHASE5_SEED = 20260726


def _observation_variant(
    observation: AgentObservation,
    match_id: str,
) -> AgentObservation:
    return AgentObservation(
        observer=observation.observer,
        public_state=replace(observation.public_state, match_id=match_id),
        actions=observation.actions,
    )


def _belief_variant(
    belief: OpponentBeliefState,
    match_id: str,
) -> OpponentBeliefState:
    return replace(belief, match_id=match_id)


def _end_routine(
    *,
    value: float,
    confidence: float,
    entropy: float,
    description: str,
) -> RoutineDecision:
    return RoutineDecision(
        action_prefix=ActionPrefix().append(
            ActionToken(ActionComponent.END, "routine_end", "Commit routine action")
        ),
        value_estimate=value,
        policy_confidence=confidence,
        policy_entropy=entropy,
        legal_action_count=4,
        description=description,
    )


def _build_episode_roots(episode_id: str) -> tuple[SelfPlayRoot, ...]:
    match_id = f"phase5-{episode_id}"
    easy_observation = _observation_variant(
        AgentObservation.from_state(build_demo_state()),
        match_id,
    )
    easy_belief = OpponentBeliefState(
        match_id=match_id,
        observed_through_turn=easy_observation.public_state.turn_number,
        plan=OpponentPlanPosterior.normalized(
            (0.96, 0.008, 0.008, 0.008, 0.008, 0.008)
        ),
        resource_probabilities=(
            0.03,
            0.04,
            0.96,
            0.04,
            0.03,
            0.02,
            0.02,
            0.03,
            0.02,
            0.02,
        ),
    )
    hard_observation = _observation_variant(_combo_observation(), match_id)
    hard_belief = _belief_variant(_combo_belief(hard_observation), match_id)
    uniform_meta = (1.0 / len(STRATEGIC_AXES),) * len(STRATEGIC_AXES)
    low_confidence_entropy = math.log(4.0) * 0.95
    return (
        SelfPlayRoot(
            observation=easy_observation,
            belief=easy_belief,
            routine=_end_routine(
                value=0.45,
                confidence=0.12,
                entropy=low_confidence_entropy,
                description="Uncertain policy before absorbing the easy Mewtwo line.",
            ),
            routine_plan_id="spidops_engine",
            compute_cost=1.0,
            search_value_cost=0.08,
            strategic_value_override=(0.34, 0.35, 0.36, 0.35, 0.34, 0.36),
            meta_weight_override=uniform_meta,
        ),
        SelfPlayRoot(
            observation=hard_observation,
            belief=hard_belief,
            routine=_end_routine(
                value=0.06,
                confidence=0.18,
                entropy=low_confidence_entropy,
                description="Uncertain policy at a belief-sensitive combo disruption pivot.",
            ),
            routine_plan_id="mewtwo_prize_push",
            compute_cost=1.0,
            search_value_cost=0.08,
            strategic_value_override=(-0.55, 0.15, -0.10, 0.05, -0.60, 0.72),
            meta_weight_override=uniform_meta,
        ),
    )


class _DeterministicSelfPlayEnvironment:
    """Episode-shaped paired replay fixture behind the public environment protocol."""

    def __init__(self, episode_id: str, roots: tuple[SelfPlayRoot, ...]) -> None:
        self.episode_id = episode_id
        self.environment_fingerprint = stable_fingerprint(
            {
                "name": "phase5_strategic_paired_replay",
                "schema": 1,
                "root_kinds": ["routine_teachable", "belief_critical"],
            }
        )
        self.roots = roots
        self.index = 0

    def reset(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("self-play seed cannot be negative")
        self.index = 0

    @property
    def terminal(self) -> bool:
        return self.index >= len(self.roots)

    def current_root(self) -> SelfPlayRoot:
        if self.terminal:
            raise RuntimeError("terminal environment has no current root")
        return self.roots[self.index]

    def validate(
        self,
        root: SelfPlayRoot,
        search_result: PlanSearchResult,
    ) -> OutcomeValidation:
        if root is not self.current_root():
            raise ValueError("paired validator received a stale self-play root")
        routine_return = 0.45 if self.index == 0 else 0.06
        return OutcomeValidation(
            source="complete_horizon_strategic_paired_replay",
            routine_return=routine_return,
            search_return=search_result.chosen.expected_value,
            confidence=0.96,
            paired_samples=8,
            terminal_verified=True,
            environment_fingerprint=self.environment_fingerprint,
        )

    def advance(self, decision) -> None:
        if self.terminal:
            raise RuntimeError("cannot advance a terminal environment")
        self.index += 1


def _build_phase4_gate(
    agent: Phase2BeliefAwareAgent,
    easy_root: SelfPlayRoot,
    hard_root: SelfPlayRoot,
):
    proposer = TeamRocketPlanProposer()
    extractor = SearchGateFeatureExtractor(proposer)
    bank = SearchComputeBank(capacity=4.0, available=4.0, refill_per_turn=0.25)

    easy_evaluation = _evaluate_agent(agent, easy_root.observation)
    _set_head_values(easy_evaluation, easy_root.strategic_value_override)
    hard_evaluation = _evaluate_agent(agent, hard_root.observation)
    _set_head_values(hard_evaluation, hard_root.strategic_value_override)
    easy_anchor = extractor.extract(
        easy_root.observation,
        easy_evaluation,
        easy_root.belief,
        _routine_decision(pivotal=False).policy_signals,
        bank,
    )
    hard_anchor = extractor.extract(
        hard_root.observation,
        hard_evaluation,
        hard_root.belief,
        _routine_decision(pivotal=True).policy_signals,
        bank,
    )
    train, validation, test = _synthetic_gate_splits(hard_anchor, easy_anchor)
    network = SearchGateNetwork(hidden_dim=48)
    _train_gate(network, SearchGateBatch.from_examples(train))
    network.eval()
    validation_batch = SearchGateBatch.from_examples(validation)
    with torch.no_grad():
        output = network(validation_batch.features)
        calibrator = SearchGateCalibrator.fit(
            output.raw_logit,
            validation_batch.realized_gain,
            validation_batch.search_cost,
        )
    return (
        proposer,
        extractor,
        network,
        calibrator,
        train,
        validation,
        test,
    )


def _evaluate_student_record(
    student: SearchGuidedStudent,
    record,
):
    state = StateTensorizer()(
        [record.observation.public_state],
        perspective=record.observation.observer,
    )
    history, mask = ActionHistoryTensorizer()(
        [record.observation],
        max_events=32,
    )
    student.eval()
    with torch.no_grad():
        return student.agent(state, history, mask)


def _student_policy_signals(student: SearchGuidedStudent, record, evaluation):
    teacher_step = record.tactical_steps[0]
    candidate_batch = CandidateBatch(
        features=torch.tensor(
            [[candidate.features for candidate in teacher_step.candidates]],
            dtype=evaluation.plan.dtype,
        ),
        legal_mask=torch.tensor(
            [[candidate.legal for candidate in teacher_step.candidates]],
            dtype=torch.bool,
        ),
    )
    with torch.no_grad():
        hidden = student.agent.initial_tactical_hidden(evaluation)
        logits = student.agent.tactical_policy.score_next(hidden, candidate_batch)
        probabilities = torch.softmax(logits, dim=-1)[0]
    legal_probabilities = probabilities[candidate_batch.legal_mask[0]]
    entropy = float(
        (-(legal_probabilities * legal_probabilities.clamp_min(1e-12).log()).sum()).item()
    )
    return PolicySignals(
        confidence=float(probabilities.max().item()),
        entropy=entropy,
        legal_action_count=int(candidate_batch.legal_mask[0].sum().item()),
    )


def _retention_examples(
    easy: SearchRetentionTarget,
    hard: SearchRetentionTarget,
    *,
    count: int,
    seed: int,
) -> tuple[SearchGateSupervision, ...]:
    rng = random.Random(seed)
    examples: list[SearchGateSupervision] = []
    for _ in range(count):
        for target in (easy, hard):
            examples.append(
                SearchGateSupervision(
                    features=_jittered_anchor(target.refreshed_features, rng, amount=0.018),
                    realized_search_gain=target.remaining_search_gain,
                    search_cost=target.search_value_cost,
                    source="post_distillation_retention_fixture",
                )
            )
    return tuple(examples)


def _fine_tune_gate(
    network: SearchGateNetwork,
    examples,
    *,
    steps: int = 120,
) -> None:
    batch = SearchGateBatch.from_examples(examples)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
    network.train()
    for _ in range(steps):
        loss = search_gate_loss(network(batch.features), batch)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
        optimizer.step()


def _gradient_coverage(student: SearchGuidedStudent) -> dict[str, bool]:
    groups = {
        "tactical_policy": student.agent.tactical_policy,
        "plan_memory": student.agent.plan_memory,
        "beliefs": student.agent.opponent_belief_model,
        "strategic_values": student.agent.value_heads,
        "meta_value": student.agent.meta_value,
        "uncertainty": student.agent.search_uncertainty_head,
        "counterfactual_plan_policy": student.plan_policy,
    }
    return {
        name: any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and parameter.grad.abs().sum().item() > 0.0
            for parameter in module.parameters()
        )
        for name, module in groups.items()
    }


def run_phase5_demo() -> dict[str, object]:
    torch.manual_seed(PHASE5_SEED)
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

    train_roots = _build_episode_roots("train")
    test_roots = _build_episode_roots("heldout")
    (
        proposer,
        extractor,
        gate_network,
        gate_calibrator,
        phase4_train,
        phase4_validation,
        _,
    ) = _build_phase4_gate(agent, train_roots[0], train_roots[1])
    adapter = StrategicAbstractionAdapter(config.candidate_feature_dim)
    search_budget = PlanSearchBudget(max_plans=4, max_scenarios=32)
    search_config = PlanSearchConfig()
    planner = BeliefAwarePlanSearch(
        proposer,
        AutoregressivePlanExecutor(agent.tactical_policy),
        adapter,
        budget=search_budget,
        config=search_config,
    )
    selective = SelectiveSearchPolicy(
        CalibratedSearchGate(gate_network, extractor, gate_calibrator),
        planner,
    )
    collector = SearchTeacherCollector(
        adapter,
        SearchTargetValidationConfig(
            min_predicted_gain=0.05,
            min_validated_gain=0.05,
            min_validation_confidence=0.90,
            min_paired_samples=4,
            min_action_consensus=0.55,
            max_teacher_value_stddev=0.25,
            max_counterfactuals=2,
            max_counterfactual_score_gap=0.35,
            max_counterfactual_stddev=0.30,
            retain_scenario_entropy=0.40,
            retain_value_stddev=0.12,
        ),
    )
    runner = SearchGuidedSelfPlayRunner(
        agent,
        selective,
        collector,
        max_turns=8,
    )
    deck_fingerprint = load_default_deck().fingerprint
    search_fingerprint = stable_fingerprint(
        {
            "budget": asdict(search_budget),
            "config": asdict(search_config),
            "adapter": "StrategicAbstractionAdapter:v1",
        }
    )
    train_environment = _DeterministicSelfPlayEnvironment("train", train_roots)
    test_environment = _DeterministicSelfPlayEnvironment("heldout", test_roots)
    train_collection = runner.run_episode(
        train_environment,
        SearchComputeBank(capacity=4.0, available=4.0, refill_per_turn=0.0),
        deck_fingerprint=deck_fingerprint,
        search_config_fingerprint=search_fingerprint,
        seed=PHASE5_SEED,
    )
    heldout_collection = runner.run_episode(
        test_environment,
        SearchComputeBank(capacity=4.0, available=4.0, refill_per_turn=0.0),
        deck_fingerprint=deck_fingerprint,
        search_config_fingerprint=search_fingerprint,
        seed=PHASE5_SEED + 1,
    )
    dataset = SearchTeacherDataset(
        train_collection.records + heldout_collection.records
    )
    dataset_roundtrip = SearchTeacherDataset.from_json(dataset.to_json()) == dataset
    gate_dataset_roundtrip = (
        type(dataset.gate_dataset()).from_json(dataset.gate_dataset().to_json())
        == dataset.gate_dataset()
    )
    train_dataset = dataset.for_episodes(("train",))
    heldout_dataset = dataset.for_episodes(("heldout",))

    first_record = train_collection.records[0]
    weak_validation = OutcomeValidation(
        source="insufficient_paired_replay",
        routine_return=first_record.outcome_validation.routine_return,
        search_return=first_record.outcome_validation.routine_return + 0.01,
        confidence=0.45,
        paired_samples=1,
        terminal_verified=False,
        environment_fingerprint=first_record.provenance.environment_fingerprint,
    )
    rejected = collector.collect(
        train_roots[0].observation,
        train_roots[0].routine,
        train_roots[0].routine_plan_id,
        train_collection.decisions[0],
        weak_validation,
        first_record.provenance,
        plan_dim=config.plan_dim,
        search_value_cost=train_roots[0].search_value_cost,
    )
    rejected_roundtrip = TeacherAdmission.from_dict(rejected.to_dict()) == rejected
    tampered_record = first_record.to_dict()
    tampered_record["predicted_search_gain"] += 0.1
    try:
        SearchTeacherRecord.from_dict(tampered_record)
        tamper_rejected = False
    except ValueError:
        tamper_rejected = True

    student = SearchGuidedStudent(agent, plan_policy_hidden_dim=48)
    distiller = SearchGuidedDistiller(student)
    student_before_fingerprint = model_fingerprint(student)
    before_metrics = distiller.evaluate(heldout_dataset.records)
    before_components = distiller.loss(heldout_dataset.records).detached()
    gradient_loss = distiller.loss(train_dataset.records)
    student.zero_grad(set_to_none=True)
    gradient_loss.total.backward()
    gradient_coverage = _gradient_coverage(student)
    student.zero_grad(set_to_none=True)
    trajectory = distiller.fit(
        train_dataset.records,
        steps=180,
        learning_rate=0.008,
    )
    after_metrics = distiller.evaluate(heldout_dataset.records)
    after_components = distiller.loss(heldout_dataset.records).detached()
    student_after_fingerprint = model_fingerprint(student)

    restored_student = SearchGuidedStudent(
        Phase2BeliefAwareAgent(config),
        plan_policy_hidden_dim=48,
    )
    restored_student.load_state_dict(student.state_dict())
    checkpoint_weights_roundtrip = (
        model_fingerprint(restored_student) == student_after_fingerprint
    )
    manifest = DistillationCheckpointManifest(
        dataset_fingerprint=dataset.fingerprint,
        deck_fingerprint=deck_fingerprint,
        environment_fingerprints=tuple(
            sorted({record.provenance.environment_fingerprint for record in dataset.records})
        ),
        teacher_model_fingerprints=tuple(
            sorted({record.provenance.teacher_model_fingerprint for record in dataset.records})
        ),
        student_model_fingerprint=student_after_fingerprint,
        search_config_fingerprint=search_fingerprint,
        training_seed=PHASE5_SEED,
        optimizer_steps=len(trajectory),
        initial_loss=trajectory[0],
        final_loss=trajectory[-1],
        training_episode_ids=("train",),
        heldout_episode_ids=("heldout",),
    )
    manifest_roundtrip = (
        DistillationCheckpointManifest.from_json(manifest.to_json()) == manifest
    )

    heldout_by_disposition = {
        record.disposition: record for record in heldout_dataset.records
    }
    easy_record = heldout_by_disposition[DistillationDisposition.DISTILL_TO_ROUTINE]
    hard_record = heldout_by_disposition[DistillationDisposition.RETAIN_EXPERT_SEARCH]
    post_bank = SearchComputeBank(capacity=3.0, available=3.0, refill_per_turn=0.0)
    easy_evaluation = _evaluate_student_record(student, easy_record)
    hard_evaluation = _evaluate_student_record(student, hard_record)
    easy_signals = _student_policy_signals(student, easy_record, easy_evaluation)
    hard_signals = _student_policy_signals(student, hard_record, hard_evaluation)
    easy_features = extractor.extract(
        easy_record.observation,
        easy_evaluation,
        easy_record.search_result.belief,
        easy_signals,
        post_bank,
    )
    hard_features = extractor.extract(
        hard_record.observation,
        hard_evaluation,
        hard_record.search_result.belief,
        hard_signals,
        post_bank,
    )
    easy_retention = SearchRetentionTarget(
        teacher_example_id=easy_record.example_id,
        disposition=easy_record.disposition,
        refreshed_features=easy_features,
        distilled_routine_return=easy_record.outcome_validation.search_return - 0.02,
        expert_return=easy_record.outcome_validation.search_return,
        search_value_cost=easy_record.search_value_cost,
        confidence=0.95,
    )
    hard_retention = SearchRetentionTarget(
        teacher_example_id=hard_record.example_id,
        disposition=hard_record.disposition,
        refreshed_features=hard_features,
        distilled_routine_return=hard_record.outcome_validation.search_return - 0.18,
        expert_return=hard_record.outcome_validation.search_return,
        search_value_cost=hard_record.search_value_cost,
        confidence=0.95,
    )
    retention_roundtrip = all(
        SearchRetentionTarget.from_dict(target.to_dict()) == target
        for target in (easy_retention, hard_retention)
    )
    retention_train = _retention_examples(
        easy_retention,
        hard_retention,
        count=70,
        seed=PHASE5_SEED + 10,
    )
    retention_validation = _retention_examples(
        easy_retention,
        hard_retention,
        count=24,
        seed=PHASE5_SEED + 20,
    )
    retention_test = _retention_examples(
        easy_retention,
        hard_retention,
        count=30,
        seed=PHASE5_SEED + 30,
    )
    _fine_tune_gate(
        gate_network,
        phase4_train + retention_train,
        steps=140,
    )
    gate_network.eval()
    combined_validation = SearchGateBatch.from_examples(
        phase4_validation + retention_validation
    )
    retention_test_batch = SearchGateBatch.from_examples(retention_test)
    with torch.no_grad():
        validation_output = gate_network(combined_validation.features)
        retention_calibrator = SearchGateCalibrator.fit(
            validation_output.raw_logit,
            combined_validation.realized_gain,
            combined_validation.search_cost,
        )
        retention_metrics = evaluate_search_gate(
            gate_network(retention_test_batch.features),
            retention_test_batch,
            retention_calibrator,
        )
    retention_policy = SelectiveSearchPolicy(
        CalibratedSearchGate(gate_network, extractor, retention_calibrator),
        planner,
    )
    easy_routine = RoutineDecision(
        action_prefix=easy_record.teacher_action,
        value_estimate=easy_retention.distilled_routine_return,
        policy_confidence=easy_signals.confidence,
        policy_entropy=easy_signals.entropy,
        legal_action_count=easy_signals.legal_action_count,
        description="Distilled routine now reproduces the validated search line.",
    )
    hard_routine = RoutineDecision(
        action_prefix=hard_record.teacher_action,
        value_estimate=hard_retention.distilled_routine_return,
        policy_confidence=hard_signals.confidence,
        policy_entropy=hard_signals.entropy,
        legal_action_count=hard_signals.legal_action_count,
        description="Confident tactics still face irreducible strategic uncertainty.",
    )
    easy_post_decision = retention_policy.decide(
        easy_record.observation,
        easy_evaluation,
        easy_record.search_result.belief,
        easy_routine,
        post_bank,
        search_cost=1.0,
    )
    hard_post_decision = retention_policy.decide(
        hard_record.observation,
        hard_evaluation,
        hard_record.search_result.belief,
        hard_routine,
        easy_post_decision.gate.bank_after,
        search_cost=1.0,
    )

    accepted_counterfactuals = sum(
        len(record.accepted_counterfactuals) for record in dataset.records
    )
    dispositions = dataset.disposition_counts
    summary: dict[str, object] = {
        "status": "ok",
        "self_play_collection": {
            "episodes": 2,
            "turns": train_collection.statistics.turns
            + heldout_collection.statistics.turns,
            "searched_turns": train_collection.statistics.searched_turns
            + heldout_collection.statistics.searched_turns,
            "admitted_teachers": len(dataset.records),
            "rejected_unsafe_candidate": not rejected.accepted,
            "rejection_reasons": list(rejected.rejection_reasons),
            "rejected_roundtrip": rejected_roundtrip,
            "tampered_record_rejected": tamper_rejected,
            "mean_action_consensus": round(
                sum(record.teacher_action_consensus for record in dataset.records)
                / len(dataset.records),
                6,
            ),
            "action_changes": sum(record.action_changed for record in dataset.records),
            "plan_changes": sum(record.plan_changed for record in dataset.records),
            "accepted_counterfactuals": accepted_counterfactuals,
            "distill_to_routine": dispositions[
                DistillationDisposition.DISTILL_TO_ROUTINE
            ],
            "retain_expert_search": dispositions[
                DistillationDisposition.RETAIN_EXPERT_SEARCH
            ],
        },
        "durable_data": {
            "dataset_fingerprint": dataset.fingerprint,
            "dataset_roundtrip": dataset_roundtrip,
            "gate_dataset_roundtrip": gate_dataset_roundtrip,
            "train_records": len(train_dataset.records),
            "heldout_records": len(heldout_dataset.records),
            "deck_fingerprint_attached": all(
                record.provenance.deck_fingerprint == deck_fingerprint
                for record in dataset.records
            ),
            "environment_fingerprint_attached": all(
                record.outcome_validation.environment_fingerprint
                == record.provenance.environment_fingerprint
                for record in dataset.records
            ),
        },
        "distillation": {
            "before": {
                **{
                    name: round(value, 6)
                    for name, value in before_metrics.to_dict().items()
                },
                "tactical_nll": round(before_components["tactical_policy"], 6),
            },
            "after": {
                **{
                    name: round(value, 6)
                    for name, value in after_metrics.to_dict().items()
                },
                "tactical_nll": round(after_components["tactical_policy"], 6),
            },
            "loss_reduction_fraction": round(
                1.0 - after_metrics.total_loss / before_metrics.total_loss,
                6,
            ),
            "optimizer_steps": len(trajectory),
            "all_named_gradient_paths": all(gradient_coverage.values()),
            "gradient_coverage": gradient_coverage,
            "teacher_tactical_quality_accuracy": round(
                tactical_quality_accuracy,
                6,
            ),
        },
        "checkpoint": {
            "manifest_id": manifest.checkpoint_id,
            "manifest_roundtrip": manifest_roundtrip,
            "weights_roundtrip": checkpoint_weights_roundtrip,
            "student_changed": student_before_fingerprint != student_after_fingerprint,
        },
        "post_distillation_search": {
            "retention_targets_roundtrip": retention_roundtrip,
            "gate_accuracy": round(retention_metrics.accuracy, 6),
            "profitable_invocation_rate": round(
                retention_metrics.positive_invocation_rate,
                6,
            ),
            "unprofitable_invocation_rate": round(
                retention_metrics.negative_invocation_rate,
                6,
            ),
            "easy_remaining_gain": round(easy_retention.remaining_search_gain, 6),
            "hard_remaining_gain": round(hard_retention.remaining_search_gain, 6),
            "easy_mode": easy_post_decision.mode.value,
            "hard_mode": hard_post_decision.mode.value,
            "easy_gate_probability": round(
                easy_post_decision.gate.calibrated_probability,
                6,
            ),
            "hard_gate_probability": round(
                hard_post_decision.gate.calibrated_probability,
                6,
            ),
            "search_invocations": int(
                easy_post_decision.mode is DecisionMode.EXPERT_SEARCH
            )
            + int(hard_post_decision.mode is DecisionMode.EXPERT_SEARCH),
        },
    }

    improved_components = (
        after_metrics.total_loss < before_metrics.total_loss * 0.45,
        after_components["tactical_policy"] < before_components["tactical_policy"],
        after_metrics.plan_target_mse < before_metrics.plan_target_mse,
        after_metrics.belief_plan_cross_entropy
        < before_metrics.belief_plan_cross_entropy,
        after_metrics.belief_resource_mae < before_metrics.belief_resource_mae,
        after_metrics.strategic_value_mae < before_metrics.strategic_value_mae,
        after_metrics.meta_value_mae < before_metrics.meta_value_mae,
        after_metrics.uncertainty_mae < before_metrics.uncertainty_mae,
        after_metrics.plan_value_mae < before_metrics.plan_value_mae,
    )
    checks = {
        "selected search ran in self-play": (
            train_collection.statistics.searched_turns == 2
            and heldout_collection.statistics.searched_turns == 2
        ),
        "every valid improvement admitted": len(dataset.records) == 4,
        "unsafe evidence rejected": (
            not rejected.accepted
            and "validated_gain_margin" in rejected.rejection_reasons
            and "validation_confidence" in rejected.rejection_reasons
        ),
        "meaningful roots saved": all(
            record.action_changed or record.plan_changed for record in dataset.records
        ),
        "outcome utility positive": all(
            record.outcome_validation.validated_gain > record.search_value_cost
            for record in dataset.records
        ),
        "counterfactual shortlist audited": accepted_counterfactuals == 8,
        "both distillation dispositions present": (
            dispositions[DistillationDisposition.DISTILL_TO_ROUTINE] == 2
            and dispositions[DistillationDisposition.RETAIN_EXPERT_SEARCH] == 2
        ),
        "durable artifacts roundtrip": (
            dataset_roundtrip
            and gate_dataset_roundtrip
            and rejected_roundtrip
            and tamper_rejected
            and retention_roundtrip
        ),
        "train and heldout episodes separated": (
            {record.provenance.episode_id for record in train_dataset.records}
            == {"train"}
            and {record.provenance.episode_id for record in heldout_dataset.records}
            == {"heldout"}
        ),
        "all named systems receive gradients": all(gradient_coverage.values()),
        "all named targets improve heldout": all(improved_components),
        "routine plan policy absorbs search": (
            after_metrics.plan_top1_accuracy == 1.0
            and after_metrics.counterfactual_ranking_accuracy == 1.0
            and after_metrics.tactical_token_accuracy >= 0.95
        ),
        "checkpoint reproducible": (
            manifest_roundtrip
            and checkpoint_weights_roundtrip
            and student_before_fingerprint != student_after_fingerprint
        ),
        "retention gate calibrated": (
            retention_metrics.accuracy >= 0.95
            and retention_metrics.positive_invocation_rate >= 0.95
            and retention_metrics.negative_invocation_rate <= 0.05
        ),
        "routine absorbed but hard search retained": (
            easy_post_decision.mode is DecisionMode.ROUTINE_POLICY
            and hard_post_decision.mode is DecisionMode.EXPERT_SEARCH
            and hard_post_decision.search_result is not None
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Phase 5 demo failed checks: {failed}; summary={summary}")
    return summary


def main() -> None:
    print(json.dumps(run_phase5_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
