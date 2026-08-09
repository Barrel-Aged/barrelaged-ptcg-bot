from pathlib import Path
import unittest

from rocketbot.phase6_demo import default_official_runtime, run_phase6_demo


@unittest.skipUnless(default_official_runtime().is_dir(), "official competition runtime not staged")
class Phase6DemoTests(unittest.TestCase):
    def test_official_engine_phase6_contract_end_to_end(self) -> None:
        result = run_phase6_demo(repetitions=1)
        evaluation = result["evaluation"]
        deck = result["deck"]

        self.assertEqual(result["status"], "external_blocked")
        self.assertEqual(result["software_status"], "ok")
        self.assertEqual(evaluation["benchmark_id"], "official-cg-phase6-r1")
        self.assertEqual(evaluation["tactical"]["illegal_action_rate"], 0.0)
        self.assertEqual(evaluation["tactical"]["fallback_rate"], 0.0)
        self.assertEqual(evaluation["tactical"]["soft_timeout_rate"], 0.0)
        self.assertGreater(evaluation["tactical"]["decisions"], 1000)
        self.assertEqual(len(evaluation["by_ablation"]), 4)
        self.assertEqual(len(evaluation["by_archetype"]), 6)
        self.assertGreater(evaluation["search"]["invocations"], 0)
        self.assertEqual(len(result["replay_reviews"]), 6)
        self.assertFalse(deck["exact_resolution"]["playable"])
        self.assertEqual(deck["exact_resolution"]["unresolved"], ["Prism Tower"])
        self.assertTrue(deck["compatibility_resolution"]["playable"])
        self.assertEqual(
            result["external_status"]["superhuman_strength"], "not_established"
        )


if __name__ == "__main__":
    unittest.main()

