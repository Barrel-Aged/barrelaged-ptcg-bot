from pathlib import Path
import unittest

from rocketbot.phase6_bridge import run_phase6_official_bridge_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT_ROOT / "work" / "official_kaggle" / "sample_submission"


@unittest.skipUnless((RUNTIME / "cg").is_dir(), "staged official runtime unavailable")
class Phase6BridgeEvidenceTests(unittest.TestCase):
    def test_live_shadow_bridge_roundtrips_without_changing_native_actions(self):
        artifacts = run_phase6_official_bridge_audit(
            project_root=PROJECT_ROOT,
            runtime_directory=RUNTIME,
            match_limit=2,
        )
        evidence = artifacts.evidence
        claims = dict(evidence.claims)
        self.assertTrue(evidence.official_engine_executed)
        self.assertEqual(len(evidence.matches), 2)
        self.assertGreater(evidence.target_decisions, 0)
        self.assertEqual(evidence.bridge_errors, 0)
        self.assertEqual(
            evidence.exact_selection_roundtrips,
            evidence.target_decisions,
        )
        self.assertEqual(evidence.hidden_canary_invariant_decisions, evidence.target_decisions)
        self.assertEqual(evidence.native_illegal_actions, 0)
        self.assertEqual(evidence.native_fallbacks, 0)
        self.assertEqual(evidence.native_soft_timeouts, 0)
        self.assertEqual(evidence.shadow_action_changes, 0)
        self.assertEqual(claims["full_game_neural_blueprint"], "not_trained")
        self.assertEqual(claims["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
