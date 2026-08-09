"""Deterministic Phase 2 training, calibration, inverse-belief, and decision demo."""

from __future__ import annotations

from dataclasses import replace
import json

import torch
from torch.nn import functional as F

from .belief_model import BeliefCalibrator, evaluate_beliefs
from .belief_training import (
    BeliefBatchBuilder,
    BeliefSupervision,
    TrajectoryBeliefDataset,
    belief_supervision_loss,
)
from .beliefs import (
    OPPONENT_INTENTS,
    RESOURCE_CATEGORIES,
    HiddenResourcePosterior,
    IntentTemplateLineGenerator,
    InversePlanningBeliefUpdater,
    MinimumEnablingHypothesisGenerator,
    OpponentBeliefState,
    OpponentPlanPosterior,
    OpponentStyle,
    ResourceCategory,
)
from .demo import build_demo_state
from .model import Phase2AgentConfig, Phase2BeliefAwareAgent
from .observation import AgentObservation, ObservedAction
from .policy import CandidateBatch
from .state import PlayerId


def _make_supervision_splits() -> tuple[
    tuple[BeliefSupervision, ...],
    tuple[BeliefSupervision, ...],
    tuple[BeliefSupervision, ...],
]:
    base_observation = AgentObservation.from_state(build_demo_state())
    plan_prior = OpponentPlanPosterior.uniform()
    lines = IntentTemplateLineGenerator().propose(base_observation, plan_prior)
    train: list[BeliefSupervision] = []
    validation: list[BeliefSupervision] = []
    test: list[BeliefSupervision] = []
    for repetition in range(10):
        destination = train if repetition < 6 else validation if repetition < 8 else test
        for line in lines:
            turn_number = 5 + repetition
            state = replace(
                base_observation.public_state,
                match_id=f"phase2-{repetition}-{line.intent.value}",
                turn_number=turn_number,
            )
            actions = tuple(
                ObservedAction(
                    turn_number=turn_number - 1,
                    sequence_index=index,
                    actor=PlayerId.OPPONENT,
                    kind=kind,
                    card_name=(
                        "Revealed public action card"
                        if kind.value in {"play_trainer", "play_pokemon", "evolve"}
                        else None
                    ),
                )
                for index, kind in enumerate(line.action_signature)
            )
            observation = AgentObservation(
                observer=base_observation.observer,
                public_state=state,
                actions=actions,
            )
            present = tuple(
                category
                for category in RESOURCE_CATEGORIES
                if line.minimum_resources.count(category) > 0
            )
            destination.append(
                BeliefSupervision.fully_labeled(
                    observation,
                    revealed_intent=line.intent,
                    present=present,
                    source="deterministic_phase2_contract_fixture",
                )
            )
    return tuple(train), tuple(validation), tuple(test)


def _train_belief_model(
    agent: Phase2BeliefAwareAgent,
    train_batch,
    *,
    steps: int = 140,
) -> None:
    parameters = list(agent.state_encoder.parameters()) + list(
        agent.opponent_belief_model.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=0.025)
    agent.train()
    for _ in range(steps):
        embedding = agent.state_encoder(train_batch.state_features)
        output = agent.opponent_belief_model(
            embedding,
            train_batch.action_history,
            train_batch.history_mask,
        )
        loss = belief_supervision_loss(
            output,
            train_batch.plan_labels,
            train_batch.resource_targets,
            train_batch.resource_known_mask,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()


def _belief_output(agent, batch, calibrator=None):
    embedding = agent.state_encoder(batch.state_features)
    return agent.opponent_belief_model(
        embedding,
        batch.action_history,
        batch.history_mask,
        calibrator=calibrator,
    )


def _train_decision_probe(
    agent: Phase2BeliefAwareAgent,
    train_context: torch.Tensor,
    train_labels: torch.Tensor,
    test_context: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    steps: int = 180,
) -> tuple[float, float]:
    policy = agent.tactical_policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.025)
    number_of_actions = len(OPPONENT_INTENTS)

    def candidates(batch_size: int) -> CandidateBatch:
        features = torch.zeros(
            batch_size,
            number_of_actions,
            agent.config.candidate_feature_dim,
        )
        identity = torch.eye(number_of_actions)
        features[:, :, :number_of_actions] = identity.unsqueeze(0)
        return CandidateBatch(
            features=features,
            legal_mask=torch.ones(batch_size, number_of_actions, dtype=torch.bool),
        )

    train_candidates = candidates(train_context.shape[0])
    zero_state = torch.zeros(train_context.shape[0], agent.config.state_embedding_dim)
    zero_plan = torch.zeros(train_context.shape[0], agent.config.plan_dim)
    policy.train()
    for _ in range(steps):
        hidden = policy.initial_hidden(zero_state, zero_plan, train_context)
        logits = policy.score_next(hidden, train_candidates)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

    policy.eval()
    with torch.no_grad():
        test_candidates = candidates(test_context.shape[0])
        test_state = torch.zeros(test_context.shape[0], agent.config.state_embedding_dim)
        test_plan = torch.zeros(test_context.shape[0], agent.config.plan_dim)
        aware_hidden = policy.initial_hidden(test_state, test_plan, test_context)
        aware_logits = policy.score_next(aware_hidden, test_candidates)
        aware_accuracy = aware_logits.argmax(dim=-1).eq(test_labels).float().mean()

        uniform_context = torch.cat(
            (
                torch.full(
                    (test_context.shape[0], len(OPPONENT_INTENTS)),
                    1.0 / len(OPPONENT_INTENTS),
                ),
                torch.full(
                    (test_context.shape[0], len(RESOURCE_CATEGORIES)),
                    0.5,
                ),
            ),
            dim=-1,
        )
        ablated_hidden = policy.initial_hidden(test_state, test_plan, uniform_context)
        ablated_logits = policy.score_next(ablated_hidden, test_candidates)
        ablated_accuracy = ablated_logits.argmax(dim=-1).eq(test_labels).float().mean()
    return float(aware_accuracy.item()), float(ablated_accuracy.item())


def run_phase2_demo() -> dict[str, object]:
    """Validate the complete Phase 2 contract on deterministic synthetic traces."""

    torch.manual_seed(20260726)
    train_examples, validation_examples, test_examples = _make_supervision_splits()
    builder = BeliefBatchBuilder(max_history_events=8)
    train_batch = builder(train_examples)
    validation_batch = builder(validation_examples)
    test_batch = builder(test_examples)
    dataset = TrajectoryBeliefDataset(train_examples + validation_examples + test_examples)
    dataset_roundtrip = TrajectoryBeliefDataset.from_json(dataset.to_json()) == dataset

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
    _train_belief_model(agent, train_batch)
    agent.eval()
    with torch.no_grad():
        validation_uncalibrated = _belief_output(agent, validation_batch)
        calibrator = BeliefCalibrator.fit(
            validation_uncalibrated.raw_plan_logits,
            validation_batch.plan_labels,
            validation_uncalibrated.raw_resource_logits,
            validation_batch.resource_targets,
            resource_mask=validation_batch.resource_known_mask,
        )
        test_uncalibrated = _belief_output(agent, test_batch)
        test_calibrated = _belief_output(agent, test_batch, calibrator)
        uncalibrated_metrics = evaluate_beliefs(
            test_uncalibrated,
            test_batch.plan_labels,
            test_batch.resource_targets,
            resource_mask=test_batch.resource_known_mask,
        )
        calibrated_metrics = evaluate_beliefs(
            test_calibrated,
            test_batch.plan_labels,
            test_batch.resource_targets,
            resource_mask=test_batch.resource_known_mask,
        )
        integrated = agent(
            test_batch.state_features,
            test_batch.action_history,
            test_batch.history_mask,
            calibrator=calibrator,
        )
        captured = integrated.capture_belief(
            match_id=test_examples[0].observation.public_state.match_id,
            observed_through_turn=test_examples[0].observation.public_state.turn_number,
        )
        captured_roundtrip = OpponentBeliefState.from_json(captured.to_json()) == captured

        train_context = _belief_output(agent, train_batch, calibrator).context_vector.detach()
        test_context = test_calibrated.context_vector.detach()

    aware_accuracy, ablated_accuracy = _train_decision_probe(
        agent,
        train_context,
        train_batch.plan_labels,
        test_context,
        test_batch.plan_labels,
    )

    base_observation = AgentObservation.from_state(build_demo_state())
    prior_plan = OpponentPlanPosterior.uniform()
    lines = IntentTemplateLineGenerator().propose(base_observation, prior_plan)
    inverse_prior = MinimumEnablingHypothesisGenerator().generate(lines, prior_plan)
    setup_line = next(line for line in lines if line.intent.value == "setup")
    basic_index = RESOURCE_CATEGORIES.index(ResourceCategory.BASIC_POKEMON)
    basic_before = inverse_prior.marginal_presence[basic_index]
    inverse_update = InversePlanningBeliefUpdater().update(
        inverse_prior,
        lines,
        setup_line.action_signature,
    )
    basic_after = inverse_update.hidden_resources.marginal_presence[basic_index]
    revealed = inverse_update.hidden_resources.condition_on_reveal(
        ResourceCategory.BASIC_POKEMON,
        present=True,
    )
    inverse_roundtrip = (
        HiddenResourcePosterior.from_json(revealed.to_json()) == revealed
    )
    fused = captured.fuse_symbolic(inverse_update.hidden_resources)
    fused_roundtrip = OpponentBeliefState.from_json(fused.to_json()) == fused
    styles = {hypothesis.style for hypothesis in inverse_prior.hypotheses}

    result: dict[str, object] = {
        "status": "ok",
        "dataset": {
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "test_examples": len(test_examples),
            "serialization_roundtrip": dataset_roundtrip,
        },
        "belief_model": {
            "plan_classes": [intent.value for intent in OPPONENT_INTENTS],
            "resource_categories": [category.value for category in RESOURCE_CATEGORIES],
            "plan_probabilities_shape": list(integrated.belief.plan_probabilities.shape),
            "resource_probabilities_shape": list(
                integrated.belief.resource_probabilities.shape
            ),
            "context_shape": list(integrated.belief.context_vector.shape),
            "plan_accuracy": round(calibrated_metrics.plan_accuracy, 6),
            "resource_accuracy": round(calibrated_metrics.resource_accuracy, 6),
            "uncalibrated_plan_nll": round(uncalibrated_metrics.plan_nll, 6),
            "calibrated_plan_nll": round(calibrated_metrics.plan_nll, 6),
            "uncalibrated_plan_ece": round(uncalibrated_metrics.plan_ece, 6),
            "calibrated_plan_ece": round(calibrated_metrics.plan_ece, 6),
            "plan_temperature": calibrator.plan_temperature,
            "resource_temperature": calibrator.resource_temperature,
            "state_roundtrip": captured_roundtrip,
        },
        "inverse_planning": {
            "hypothesis_count": len(inverse_prior.hypotheses),
            "styles": sorted(style.value for style in styles),
            "basic_pokemon_marginal_before": round(basic_before, 6),
            "basic_pokemon_marginal_after_action": round(basic_after, 6),
            "posterior_roundtrip": inverse_roundtrip,
            "fused_belief_roundtrip": fused_roundtrip,
        },
        "downstream_decision_probe": {
            "belief_aware_accuracy": round(aware_accuracy, 6),
            "uninformed_accuracy": round(ablated_accuracy, 6),
            "accuracy_improvement": round(aware_accuracy - ablated_accuracy, 6),
        },
        "integrated_shapes": {
            "state_embedding": list(integrated.state_embedding.shape),
            "persistent_plan": list(integrated.plan.shape),
            "meta_value": list(integrated.meta_value.shape),
            "meta_weights": list(integrated.meta_weights.shape),
        },
    }

    required_styles = {
        OpponentStyle.RATIONAL,
        OpponentStyle.IMPERFECT,
        OpponentStyle.DECEPTIVE,
    }
    checks = {
        "dataset serialization": dataset_roundtrip,
        "belief serialization": captured_roundtrip and fused_roundtrip,
        "inverse serialization": inverse_roundtrip,
        "all behavior alternatives": required_styles.issubset(styles),
        "normalized plan beliefs": bool(
            torch.allclose(
                integrated.belief.plan_probabilities.sum(dim=-1),
                torch.ones(len(test_examples)),
                atol=1e-6,
            )
        ),
        "plan prediction": calibrated_metrics.plan_accuracy >= 0.95,
        "resource prediction": calibrated_metrics.resource_accuracy >= 0.90,
        "calibration": calibrated_metrics.plan_nll <= uncalibrated_metrics.plan_nll + 1e-6,
        "action-conditioned inverse update": basic_after > basic_before,
        "downstream decision improvement": aware_accuracy - ablated_accuracy >= 0.50,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Phase 2 demo failed checks: {failed}; result={result}")
    return result


def main() -> None:
    print(json.dumps(run_phase2_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
