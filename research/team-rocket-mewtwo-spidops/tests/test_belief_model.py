import unittest

import torch
from torch.nn import functional as F

from rocketbot.belief_model import BeliefCalibrator
from rocketbot.belief_training import (
    BeliefBatchBuilder,
    BeliefSupervision,
    TrajectoryBeliefDataset,
    belief_supervision_loss,
)
from rocketbot.beliefs import (
    OPPONENT_INTENTS,
    RESOURCE_CATEGORIES,
    IntentTemplateLineGenerator,
    OpponentPlanPosterior,
    ResourceCategory,
)
from rocketbot.demo import build_demo_state
from rocketbot.model import Phase2AgentConfig, Phase2BeliefAwareAgent
from rocketbot.observation import AgentObservation, ObservedAction
from rocketbot.state import PlayerId


class BeliefModelTests(unittest.TestCase):
    def _examples(self):
        base = AgentObservation.from_state(build_demo_state())
        lines = IntentTemplateLineGenerator().propose(base, OpponentPlanPosterior.uniform())
        examples = []
        for line in lines[:3]:
            actions = tuple(
                ObservedAction(
                    turn_number=2,
                    sequence_index=index,
                    actor=PlayerId.OPPONENT,
                    kind=kind,
                )
                for index, kind in enumerate(line.action_signature)
            )
            observation = AgentObservation(
                observer=base.observer,
                public_state=base.public_state,
                actions=actions,
            )
            present = tuple(
                category
                for category in RESOURCE_CATEGORIES
                if line.minimum_resources.count(category) > 0
            )
            examples.append(
                BeliefSupervision.fully_labeled(
                    observation,
                    revealed_intent=line.intent,
                    present=present,
                )
            )
        return tuple(examples)

    def setUp(self) -> None:
        torch.manual_seed(77)
        self.examples = self._examples()
        self.batch = BeliefBatchBuilder(max_history_events=6)(self.examples)
        self.agent = Phase2BeliefAwareAgent(
            Phase2AgentConfig(
                state_embedding_dim=24,
                plan_dim=12,
                strategic_hidden_dim=20,
                meta_hidden_dim=16,
                candidate_feature_dim=8,
                policy_hidden_dim=18,
                history_hidden_dim=14,
                belief_embedding_dim=16,
            )
        )

    def test_integrated_belief_shapes_normalization_and_conditioning(self) -> None:
        output = self.agent(
            self.batch.state_features,
            self.batch.action_history,
            self.batch.history_mask,
        )
        self.assertEqual(tuple(output.belief.plan_probabilities.shape), (3, 6))
        self.assertEqual(tuple(output.belief.resource_probabilities.shape), (3, 10))
        self.assertEqual(tuple(output.belief.context_vector.shape), (3, 16))
        torch.testing.assert_close(
            output.belief.plan_probabilities.sum(dim=-1),
            torch.ones(3),
        )
        self.assertEqual(tuple(output.plan.shape), (3, 12))
        self.assertEqual(tuple(output.meta_weights.shape), (3, 6))

        aware = self.agent.initial_tactical_hidden(output)
        ablated = self.agent.initial_tactical_hidden(
            output,
            belief_context=torch.zeros_like(output.belief.context_vector),
        )
        self.assertFalse(torch.equal(aware, ablated))

    def test_trajectory_loss_trains_plan_and_resource_heads(self) -> None:
        output = self.agent(
            self.batch.state_features,
            self.batch.action_history,
            self.batch.history_mask,
        )
        loss = belief_supervision_loss(
            output.belief,
            self.batch.plan_labels,
            self.batch.resource_targets,
            self.batch.resource_known_mask,
        )
        loss.total.backward()
        self.assertIsNotNone(self.agent.opponent_belief_model.plan_head.weight.grad)
        self.assertIsNotNone(self.agent.opponent_belief_model.resource_head.weight.grad)

    def test_supervision_dataset_serializes(self) -> None:
        dataset = TrajectoryBeliefDataset(self.examples)
        self.assertEqual(TrajectoryBeliefDataset.from_json(dataset.to_json()), dataset)

    def test_temperature_fit_cannot_worsen_its_calibration_objective(self) -> None:
        plan_logits = torch.tensor(
            [[3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0, 0.0, 0.0]]
        )
        plan_labels = torch.tensor([0, 1])
        resource_logits = torch.tensor([[2.0] * 10, [-2.0] * 10])
        resource_targets = torch.tensor([[1.0] * 10, [0.0] * 10])
        calibrator = BeliefCalibrator.fit(
            plan_logits,
            plan_labels,
            resource_logits,
            resource_targets,
        )
        self.assertLessEqual(
            float(F.cross_entropy(plan_logits / calibrator.plan_temperature, plan_labels)),
            float(F.cross_entropy(plan_logits, plan_labels)) + 1e-7,
        )
        self.assertLessEqual(
            float(
                F.binary_cross_entropy_with_logits(
                    resource_logits / calibrator.resource_temperature,
                    resource_targets,
                )
            ),
            float(F.binary_cross_entropy_with_logits(resource_logits, resource_targets)) + 1e-7,
        )
        self.assertEqual(BeliefCalibrator.from_json(calibrator.to_json()), calibrator)

    def test_partial_reveals_leave_unknown_categories_out_of_loss(self) -> None:
        example = BeliefSupervision.from_reveals(
            self.examples[0].observation,
            revealed_intent=OPPONENT_INTENTS[0],
            present=(ResourceCategory.ENERGY,),
        )
        self.assertEqual(sum(example.resource_known_mask), 1.0)


if __name__ == "__main__":
    unittest.main()
