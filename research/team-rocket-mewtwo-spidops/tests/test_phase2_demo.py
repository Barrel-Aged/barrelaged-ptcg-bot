import unittest

from rocketbot.phase2_demo import run_phase2_demo


class Phase2DemoTests(unittest.TestCase):
    def test_phase2_contract_end_to_end(self) -> None:
        result = run_phase2_demo()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["belief_model"]["plan_accuracy"], 1.0)
        self.assertGreaterEqual(
            result["downstream_decision_probe"]["accuracy_improvement"],
            0.5,
        )
        self.assertGreater(
            result["inverse_planning"]["basic_pokemon_marginal_after_action"],
            result["inverse_planning"]["basic_pokemon_marginal_before"],
        )


if __name__ == "__main__":
    unittest.main()
