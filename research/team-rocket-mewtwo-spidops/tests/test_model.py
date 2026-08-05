import unittest

import torch

from rocketbot.demo import build_demo_state
from rocketbot.features import STATE_FEATURE_NAMES, StateTensorizer
from rocketbot.model import AgentConfig, Phase1StrategicAgent
from rocketbot.plan import PlanLatent
from rocketbot.strategy import STRATEGIC_AXES


class StrategicModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)
        self.config = AgentConfig(
            state_embedding_dim=32,
            plan_dim=16,
            strategic_hidden_dim=24,
            meta_hidden_dim=20,
            candidate_feature_dim=8,
            policy_hidden_dim=28,
        )
        self.agent = Phase1StrategicAgent(self.config)
        self.features = StateTensorizer()([build_demo_state(), build_demo_state()])

    def test_feature_and_strategic_shapes(self) -> None:
        output = self.agent(self.features)
        self.assertEqual(tuple(self.features.shape), (2, len(STATE_FEATURE_NAMES)))
        self.assertEqual(tuple(output.state_embedding.shape), (2, 32))
        self.assertEqual(tuple(output.plan.shape), (2, 16))
        self.assertEqual(set(output.head_values), set(STRATEGIC_AXES))
        for value in output.head_values.values():
            self.assertEqual(tuple(value.shape), (2, 1))
            self.assertTrue(torch.all(value.abs() <= 1.0).item())
        self.assertEqual(tuple(output.meta_value.shape), (2, 1))
        self.assertEqual(tuple(output.meta_weights.shape), (2, 6))
        torch.testing.assert_close(
            output.meta_weights.sum(dim=-1),
            torch.ones(2),
        )

    def test_plan_persists_updates_and_serializes(self) -> None:
        first = self.agent(self.features)
        second = self.agent(self.features, first.plan)
        self.assertEqual(tuple(second.plan.shape), (2, 16))
        self.assertFalse(torch.equal(first.plan, second.plan))
        snapshot = second.capture_plan(
            match_id="test-match",
            created_turn=1,
            last_updated_turn=2,
            update_count=2,
            batch_index=1,
        )
        restored = PlanLatent.from_json(snapshot.to_json())
        self.assertEqual(restored, snapshot)
        self.assertEqual(tuple(restored.to_tensor().shape), (1, 16))

    def test_gradients_reach_all_strategic_heads(self) -> None:
        output = self.agent(self.features)
        loss = output.meta_value.mean() + sum(
            value.mean() for value in output.head_values.values()
        )
        loss.backward()
        for axis in STRATEGIC_AXES:
            parameters = list(self.agent.value_heads.heads[axis.value].parameters())
            self.assertTrue(any(parameter.grad is not None for parameter in parameters))


if __name__ == "__main__":
    unittest.main()
