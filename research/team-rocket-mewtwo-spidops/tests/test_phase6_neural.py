import unittest

from rocketbot.phase6_neural import run_phase6_neural_blueprint


class Phase6NeuralTests(unittest.TestCase):
    def test_phase6c_compares_and_restores_both_neural_candidates(self):
        artifacts = run_phase6_neural_blueprint()
        evidence = artifacts.evidence
        claims = dict(evidence.claims)
        self.assertEqual(
            evidence.status,
            "phase6c_neural_microgame_blueprint_validated",
        )
        self.assertEqual(evidence.selected_strategy_mode, "policy")
        self.assertLess(evidence.policy_candidate.metrics.nash_conv, 0.001)
        self.assertLess(evidence.regret_candidate.metrics.nash_conv, 0.005)
        self.assertEqual(
            evidence.policy_roundtrip.maximum_strategy_probability_delta,
            0.0,
        )
        self.assertEqual(
            evidence.regret_roundtrip.maximum_counterfactual_value_delta,
            0.0,
        )
        self.assertEqual(
            claims["neural_game_theoretic_blueprint"],
            "validated_exact_microgame_only",
        )
        self.assertEqual(claims["full_game_neural_blueprint"], "not_trained")
        self.assertEqual(claims["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
