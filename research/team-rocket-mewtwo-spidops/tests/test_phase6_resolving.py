import unittest

from rocketbot.phase6_resolving import run_phase6_safe_resolving


class Phase6ResolvingEvidenceTests(unittest.TestCase):
    def test_phase6e_exact_resolving_and_interruption_evidence(self):
        evidence = run_phase6_safe_resolving()
        claims = dict(evidence.claims)
        self.assertEqual(
            evidence.status,
            "phase6e_exact_interruption_safe_resolving_validated",
        )
        self.assertEqual(evidence.endgame.roots, 9)
        self.assertEqual(evidence.continual_root.roots, 3)
        self.assertEqual(evidence.endgame.completed_searches, 9)
        self.assertEqual(evidence.continual_root.completed_searches, 3)
        self.assertGreater(evidence.total_exact_actor_value_gain, 0.0)
        self.assertTrue(evidence.all_returned_strategies_legal)
        self.assertTrue(evidence.all_completed_certificates_safe)
        self.assertTrue(evidence.all_completed_strict_nonregression)
        self.assertTrue(
            all(
                item.returned_blueprint_exactly
                for item in evidence.interruption_fallbacks
            )
        )
        self.assertEqual(claims["official_full_game_resolver"], "not_implemented")
        self.assertEqual(claims["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
