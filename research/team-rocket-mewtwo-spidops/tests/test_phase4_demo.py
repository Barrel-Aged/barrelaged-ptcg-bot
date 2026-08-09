import unittest

from rocketbot.phase4_demo import run_phase4_demo


class Phase4DemoTests(unittest.TestCase):
    def test_phase4_selective_search_contract_end_to_end(self) -> None:
        result = run_phase4_demo()
        calibration = result["calibration"]
        execution = result["selective_execution"]

        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(calibration["accuracy"], 0.94)
        self.assertGreaterEqual(calibration["positive_invocation_rate"], 0.90)
        self.assertLessEqual(calibration["negative_invocation_rate"], 0.10)
        self.assertGreater(calibration["realized_net_utility"], 0.0)
        self.assertEqual(execution["routine_mode"], "routine_policy")
        self.assertEqual(execution["pivotal_mode"], "expert_search")
        self.assertEqual(execution["exhausted_mode"], "routine_policy")
        self.assertTrue(execution["exhausted_wanted_search"])
        self.assertTrue(execution["exhausted_budget_limited"])
        self.assertEqual(
            execution["search_invocations_across_routine_and_pivotal"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
