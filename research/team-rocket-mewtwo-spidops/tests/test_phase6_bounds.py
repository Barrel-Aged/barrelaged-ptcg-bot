import unittest

from rocketbot.phase6_bounds import run_phase6_counterfactual_bounds


class Phase6BoundEvidenceTests(unittest.TestCase):
    def test_phase6d_evidence_is_exact_scoped_and_fail_closed(self):
        artifacts = run_phase6_counterfactual_bounds()
        evidence = artifacts.evidence
        claims = dict(evidence.claims)
        self.assertEqual(
            evidence.status,
            "phase6d_exact_counterfactual_bounds_validated",
        )
        self.assertEqual(evidence.validation_coverage, 1.0)
        self.assertEqual(evidence.audit.false_safe_certificates, 0)
        self.assertTrue(evidence.audit.passes_exact_gate)
        self.assertGreater(evidence.widened_half_width, evidence.base_half_width)
        self.assertEqual(
            claims["counterfactual_value_intervals"],
            "validated_exact_microgame_only",
        )
        self.assertEqual(claims["full_game_counterfactual_bounds"], "not_calibrated")
        self.assertEqual(claims["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
