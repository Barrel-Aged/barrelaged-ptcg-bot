import unittest

import torch

from rocketbot.features import StateTensorizer
from rocketbot.model import Phase2AgentConfig, Phase2BeliefAwareAgent
from rocketbot.observation import ActionHistoryTensorizer
from rocketbot.phase3_demo import _combo_belief, _combo_observation
from rocketbot.plans import StrategicPlanCandidate, TeamRocketPlanProposer
from rocketbot.strategy import STRATEGIC_AXES


class StrategicPlanProposerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.observation = _combo_observation()
        self.belief = _combo_belief(self.observation)
        config = Phase2AgentConfig(
            state_embedding_dim=24,
            plan_dim=12,
            strategic_hidden_dim=20,
            meta_hidden_dim=16,
            candidate_feature_dim=16,
            policy_hidden_dim=20,
            history_hidden_dim=16,
            belief_embedding_dim=16,
        )
        agent = Phase2BeliefAwareAgent(config)
        state_features = StateTensorizer()(
            [self.observation.public_state],
            perspective=self.observation.observer,
        )
        history, mask = ActionHistoryTensorizer()([self.observation], max_events=8)
        with torch.no_grad():
            self.evaluation = agent(state_features, history, mask)
            self.evaluation.head_values = {
                axis: torch.zeros_like(self.evaluation.head_values[axis])
                for axis in STRATEGIC_AXES
            }

    def test_proposer_returns_few_complete_distinct_normalized_plans(self) -> None:
        plans = TeamRocketPlanProposer().propose(
            self.observation,
            self.evaluation,
            self.belief,
            top_k=4,
        )
        self.assertEqual(len(plans), 4)
        self.assertAlmostEqual(sum(plan.prior for plan in plans), 1.0)
        self.assertEqual(plans[0].plan_id, "rocket_disruption")
        self.assertEqual(
            len({plan.directive_signature for plan in plans}),
            len(plans),
        )
        for plan in plans:
            self.assertGreater(len(plan.steps), 0)
            self.assertTrue(plan.completion_condition)
            self.assertTrue(all(step.turn_offset < plan.horizon_turns for step in plan.steps))
            self.assertEqual(StrategicPlanCandidate.from_dict(plan.to_dict()), plan)

    def test_top_k_is_respected(self) -> None:
        plans = TeamRocketPlanProposer().propose(
            self.observation,
            self.evaluation,
            self.belief,
            top_k=2,
        )
        self.assertEqual(len(plans), 2)
        self.assertAlmostEqual(sum(plan.prior for plan in plans), 1.0)


if __name__ == "__main__":
    unittest.main()
