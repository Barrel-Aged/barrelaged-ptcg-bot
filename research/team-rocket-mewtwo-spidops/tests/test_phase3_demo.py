import unittest

from rocketbot.phase3_demo import run_phase3_demo


class Phase3DemoTests(unittest.TestCase):
    def test_phase3_contract_end_to_end(self) -> None:
        result = run_phase3_demo()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selection"]["chosen_plan_id"], "rocket_disruption")
        self.assertGreater(
            result["behavior_evidence"]["combo_posterior_mass"],
            result["behavior_evidence"]["combo_prior_mass"],
        )
        self.assertGreaterEqual(
            result["search_efficiency"]["estimated_branch_reduction"],
            50.0,
        )
        self.assertTrue(result["rich_interface"]["serialization_roundtrip"])


if __name__ == "__main__":
    unittest.main()
