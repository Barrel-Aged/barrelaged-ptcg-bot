from __future__ import annotations

import json
import random
import unittest

from rocketbot.exact_game import RocketPrizeDuel, uniform_profile
from rocketbot.game_theory import ActionDistribution
from rocketbot.outcome_sampling import (
    OutcomeRegretReservoir,
    OutcomeSamplingBudgetExceeded,
    OutcomeSamplingConfig,
    OutcomeSamplingRegretSample,
    OutcomeSamplingStrategySample,
    OutcomeSamplingTraversal,
    OutcomeStrategyReservoir,
)
from rocketbot.outcome_sampling_reference import (
    OutcomeSamplingMCCFRSolver,
    audit_outcome_sampling_expectation,
)


class OutcomeSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = RocketPrizeDuel()
        self.profile = uniform_profile(self.game)

    def strategy(self, information_set, belief):
        self.assertEqual(information_set.public_belief_fingerprint, belief.fingerprint)
        return self.profile.strategy(information_set)

    def test_exhaustive_paths_match_exact_immediate_counterfactual_regrets(self) -> None:
        audit = audit_outcome_sampling_expectation(
            self.game,
            self.profile,
            exploration=0.6,
        )
        self.assertEqual(audit.trajectory_count_player_0, 30)
        self.assertEqual(audit.trajectory_count_player_1, 30)
        self.assertEqual(len(audit.rows), 24)
        self.assertLess(audit.maximum_absolute_error, 1e-12)
        self.assertFalse(audit.official_engine_executed)
        self.assertFalse(audit.full_game_equilibrium_convergence)

    def test_fail_closed_traversal_and_separate_reservoir_roundtrip(self) -> None:
        with self.assertRaises(OutcomeSamplingBudgetExceeded):
            OutcomeSamplingTraversal(
                OutcomeSamplingConfig(
                    exploration=0.6,
                    maximum_transitions=1,
                    maximum_depth=32,
                )
            ).run(
                self.game,
                self.strategy,
                traverser=0,
                iteration=1,
                rng=random.Random(8),
            )

        result = OutcomeSamplingTraversal(
            OutcomeSamplingConfig(
                exploration=0.6,
                maximum_transitions=32,
                maximum_depth=32,
            )
        ).run(
            self.game,
            self.strategy,
            traverser=0,
            iteration=1,
            rng=random.Random(8),
        )
        self.assertTrue(result.complete)
        self.assertEqual(result.transitions, result.maximum_depth_reached)
        self.assertGreater(len(result.regret_samples), 0)
        self.assertEqual(len(result.regret_samples), len(result.strategy_samples))

        regret = OutcomeSamplingRegretSample.from_dict(
            json.loads(json.dumps(result.regret_samples[0].to_dict()))
        )
        strategy = OutcomeSamplingStrategySample.from_dict(
            json.loads(json.dumps(result.strategy_samples[0].to_dict()))
        )
        self.assertEqual(regret.fingerprint, result.regret_samples[0].fingerprint)
        self.assertEqual(strategy.fingerprint, result.strategy_samples[0].fingerprint)

        regrets = OutcomeRegretReservoir(2, seed=17)
        strategies = OutcomeStrategyReservoir(2, seed=23)
        for item in result.regret_samples:
            regrets.add(item)
        for item in result.strategy_samples:
            strategies.add(item)
        restored_regrets = OutcomeRegretReservoir.from_dict(
            json.loads(json.dumps(regrets.to_dict()))
        )
        restored_strategies = OutcomeStrategyReservoir.from_dict(
            json.loads(json.dumps(strategies.to_dict()))
        )
        self.assertEqual(restored_regrets.fingerprint, regrets.fingerprint)
        self.assertEqual(restored_strategies.fingerprint, strategies.fingerprint)

    def test_tabular_outcome_sampling_has_bounded_reference_exploitability(self) -> None:
        result = OutcomeSamplingMCCFRSolver(
            self.game,
            exploration=0.6,
        ).train(5_000, seed=6_021)
        self.assertEqual(result.traversals, 10_000)
        self.assertLess(result.metrics.nash_conv, 0.1)
        self.assertFalse(result.official_engine_executed)
        self.assertFalse(result.full_game_equilibrium_convergence)


if __name__ == "__main__":
    unittest.main()
