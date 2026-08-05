import unittest

from rocketbot.phase6_foundation import run_phase6_foundation


class Phase6FoundationTests(unittest.TestCase):
    def test_phase6ab_public_replay_and_exact_game_evidence(self):
        artifacts = run_phase6_foundation(cfr_iterations=20_000)
        summary = artifacts.public_replays.summary
        evidence = artifacts.evidence
        claims = dict(evidence.claims)
        self.assertEqual(evidence.status, "phase6ab_foundation_validated")
        self.assertEqual(summary["matches"], 6)
        self.assertEqual(summary["episodes"], 12)
        self.assertEqual(summary["frames"], 1_232)
        self.assertEqual(summary["decision_frames"], 610)
        self.assertTrue(summary["validated"])
        self.assertEqual(summary["unknown_event_types"], [])
        self.assertGreater(summary["redacted_events"], 0)
        self.assertGreater(summary["supervision_only_reveal_targets"], 0)
        self.assertFalse(evidence.official_engine_executed_by_game)
        self.assertLess(evidence.analytic_metrics.nash_conv, 1e-12)
        self.assertLess(evidence.cfr_metrics.nash_conv, 0.005)
        self.assertLess(evidence.cfr_value_error, 1e-4)
        self.assertTrue(evidence.safe_resolve.safe)
        self.assertGreater(evidence.safe_resolve.actor_value_gain, 0)
        self.assertEqual(
            claims["neural_game_theoretic_blueprint"],
            "trained_exact_microgame_only_full_game_not_trained",
        )
        self.assertEqual(
            claims["official_information_set_bridge"],
            "one_step_shadow_validated_full_game_not_implemented",
        )
        self.assertEqual(claims["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
