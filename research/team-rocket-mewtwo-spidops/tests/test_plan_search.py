import unittest

import torch

from rocketbot.abstract_simulator import StrategicAbstractionAdapter
from rocketbot.features import StateTensorizer
from rocketbot.model import Phase2AgentConfig, Phase2BeliefAwareAgent
from rocketbot.observation import ActionHistoryTensorizer
from rocketbot.phase3_demo import _combo_belief, _combo_observation
from rocketbot.plans import TeamRocketPlanProposer
from rocketbot.search import (
    AutoregressivePlanExecutor,
    OpponentBehaviorComparator,
    PlanSearchStatistics,
    PlanSearchUncertainty,
    scenarios_from_belief,
)
from rocketbot.state import ActionKind


class PlanSearchPrimitiveTests(unittest.TestCase):
    def test_belief_scenarios_are_top_k_and_renormalized(self) -> None:
        observation = _combo_observation()
        scenarios = scenarios_from_belief(_combo_belief(observation), max_scenarios=7)
        self.assertEqual(len(scenarios), 7)
        self.assertAlmostEqual(sum(scenario.prior_probability for scenario in scenarios), 1.0)
        self.assertGreaterEqual(
            scenarios[0].prior_probability,
            scenarios[-1].prior_probability,
        )

    def test_behavior_comparator_rewards_matching_ordered_actions(self) -> None:
        comparator = OpponentBehaviorComparator()
        observed = (
            ActionKind.PLAY_TRAINER,
            ActionKind.EVOLVE,
            ActionKind.ATTACH_ENERGY,
            ActionKind.END_TURN,
        )
        exact = comparator.likelihood(observed, observed)
        mismatch = comparator.likelihood(
            observed,
            (ActionKind.ATTACK, ActionKind.END_TURN),
        )
        self.assertGreater(exact, mismatch)

    def test_autoregressive_executor_completes_every_plan_step(self) -> None:
        torch.manual_seed(21)
        observation = _combo_observation()
        belief = _combo_belief(observation)
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
            [observation.public_state], perspective=observation.observer
        )
        history, mask = ActionHistoryTensorizer()([observation], max_events=8)
        with torch.no_grad():
            evaluation = agent(state_features, history, mask)
        plan = TeamRocketPlanProposer().propose(
            observation,
            evaluation,
            belief,
            top_k=1,
        )[0]
        scenario = scenarios_from_belief(belief, max_scenarios=1)[0]
        rollout = AutoregressivePlanExecutor(agent.tactical_policy).execute(
            observation,
            evaluation,
            belief,
            plan,
            scenario,
            StrategicAbstractionAdapter(config.candidate_feature_dim),
        )
        self.assertEqual(len(rollout.executed_steps), len(plan.steps))
        self.assertTrue(all(step.prefix.is_complete for step in rollout.executed_steps))
        self.assertGreater(rollout.candidates_scored, 0)
        self.assertGreater(rollout.estimated_raw_action_sequences, 1)

    def test_search_artifacts_reject_invalid_counts_and_uncertainty(self) -> None:
        with self.assertRaises(ValueError):
            PlanSearchStatistics(
                plans_proposed=0,
                plans_searched=1,
                scenarios_searched=1,
                plan_scenario_branches=1,
                tactical_candidates_scored=1,
                estimated_raw_action_sequences=1,
                estimated_branch_reduction=1.0,
            )
        with self.assertRaises(ValueError):
            PlanSearchUncertainty(
                scenario_entropy=0.0,
                proposal_entropy=0.0,
                chosen_value_stddev=float("nan"),
                plan_margin=0.0,
            )


if __name__ == "__main__":
    unittest.main()
