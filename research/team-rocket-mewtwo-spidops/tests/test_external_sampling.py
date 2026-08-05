import json
import random
import unittest

from rocketbot.exact_game import (
    ANALYTIC_GAME_VALUE_PLAYER_0,
    RocketPrizeDuel,
    exact_profile_metrics,
    uniform_profile,
)
from rocketbot.external_sampling import (
    AdvantageReservoir,
    ExternalSamplingBudgetExceeded,
    ExternalSamplingConfig,
    ExternalSamplingTraversal,
    StrategyReservoir,
)
from rocketbot.mccfr_reference import (
    ExternalSamplingMCCFRSolver,
    audit_external_sampling_expectation,
)


class ExternalSamplingTests(unittest.TestCase):
    def test_exhaustive_blocks_match_exact_conditional_advantages(self):
        game = RocketPrizeDuel()
        profile = uniform_profile(game)
        audit = audit_external_sampling_expectation(game, profile)
        self.assertEqual(audit.block_count_player_0, 384)
        self.assertEqual(audit.block_count_player_1, 384)
        self.assertEqual(len(audit.rows), 24)
        self.assertLess(audit.maximum_absolute_error, 2e-12)
        self.assertFalse(audit.official_engine_executed)
        self.assertFalse(audit.full_game_equilibrium_convergence)
        serialized = json.dumps(audit.to_dict(), sort_keys=True)
        self.assertNotIn("selected_completion", serialized)
        self.assertNotIn("opponent_hand", serialized)

    def test_traversal_is_complete_fail_closed_and_reservoir_is_reproducible(self):
        game = RocketPrizeDuel()
        profile = uniform_profile(game)

        def strategy(information_set, belief):
            self.assertEqual(
                information_set.public_belief_fingerprint, belief.fingerprint
            )
            return profile.strategy(information_set)

        traversal = ExternalSamplingTraversal(
            ExternalSamplingConfig(maximum_transitions=100, maximum_depth=32)
        )
        result = traversal.run(
            game,
            strategy,
            traverser=0,
            iteration=7,
            rng=random.Random(11),
        )
        self.assertTrue(result.complete)
        self.assertGreater(result.terminal_leaves, 0)
        self.assertTrue(result.advantage_samples)
        self.assertTrue(result.strategy_samples)
        self.assertTrue(all(sample.traverser == 0 for sample in result.advantage_samples))
        self.assertTrue(all(sample.player == 1 for sample in result.strategy_samples))
        self.assertFalse(result.source_official_engine_executed)
        self.assertFalse(result.hidden_completion_prior_official)

        with self.assertRaises(ExternalSamplingBudgetExceeded):
            ExternalSamplingTraversal(
                ExternalSamplingConfig(maximum_transitions=1, maximum_depth=32)
            ).run(
                game,
                strategy,
                traverser=0,
                iteration=1,
                rng=random.Random(0),
            )

        first = AdvantageReservoir(3, seed=101)
        second = AdvantageReservoir(3, seed=101)
        for _ in range(10):
            for reservoir in (first, second):
                reservoir.add(result.advantage_samples[0])
        self.assertEqual(first.seen, 10)
        self.assertEqual(len(first.samples), 3)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            json.loads(json.dumps(first.to_dict()))["fingerprint"],
            first.fingerprint,
        )
        restored = AdvantageReservoir.from_dict(
            json.loads(json.dumps(first.to_dict()))
        )
        self.assertEqual(restored.fingerprint, first.fingerprint)

        strategies = StrategyReservoir(2, seed=202)
        for sample in result.strategy_samples:
            strategies.add(sample)
        restored_strategies = StrategyReservoir.from_dict(
            json.loads(json.dumps(strategies.to_dict()))
        )
        self.assertEqual(restored_strategies.fingerprint, strategies.fingerprint)

    def test_tabular_external_sampling_converges_only_on_reference_game(self):
        game = RocketPrizeDuel()
        baseline = exact_profile_metrics(game, uniform_profile(game))
        result = ExternalSamplingMCCFRSolver(game).train(5_000, seed=6_019)
        self.assertEqual(result.traversals, 10_000)
        self.assertGreater(result.advantage_samples, 0)
        self.assertGreater(result.strategy_samples, 0)
        self.assertLess(result.metrics.nash_conv, 0.03)
        self.assertLess(result.metrics.nash_conv, baseline.nash_conv * 0.04)
        self.assertAlmostEqual(
            result.metrics.expected_value_player_0,
            ANALYTIC_GAME_VALUE_PLAYER_0,
            delta=0.002,
        )
        self.assertFalse(result.official_engine_executed)
        self.assertFalse(result.full_game_equilibrium_convergence)


if __name__ == "__main__":
    unittest.main()
