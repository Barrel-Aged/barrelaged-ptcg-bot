import unittest

from rocketbot.phase5_demo import run_phase5_demo


class Phase5DemoTests(unittest.TestCase):
    def test_phase5_search_guided_learning_contract_end_to_end(self) -> None:
        result = run_phase5_demo()
        collection = result["self_play_collection"]
        distillation = result["distillation"]
        retention = result["post_distillation_search"]

        self.assertEqual(result["status"], "ok")
        self.assertEqual(collection["searched_turns"], 4)
        self.assertEqual(collection["admitted_teachers"], 4)
        self.assertTrue(collection["rejected_unsafe_candidate"])
        self.assertTrue(collection["tampered_record_rejected"])
        self.assertEqual(collection["accepted_counterfactuals"], 8)
        self.assertTrue(distillation["all_named_gradient_paths"])
        self.assertGreater(distillation["loss_reduction_fraction"], 0.80)
        self.assertEqual(distillation["after"]["plan_top1_accuracy"], 1.0)
        self.assertEqual(
            distillation["after"]["counterfactual_ranking_accuracy"],
            1.0,
        )
        self.assertEqual(retention["easy_mode"], "routine_policy")
        self.assertEqual(retention["hard_mode"], "expert_search")
        self.assertEqual(retention["search_invocations"], 1)
        self.assertTrue(result["checkpoint"]["weights_roundtrip"])


if __name__ == "__main__":
    unittest.main()
