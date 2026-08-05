import json
from pathlib import Path
import unittest

from rocketbot.counterfactual_calibration import (
    BoundGateFailure,
    CalibratedNeuralCounterfactualBoundProvider,
    CounterfactualBoundCalibrationArtifact,
    CounterfactualBoundContext,
    audit_calibrated_counterfactual_bounds,
    calibrate_neural_counterfactual_bounds,
)
from rocketbot.exact_game import RocketPrizeDuel
from rocketbot.neural_blueprint import (
    NeuralBlueprintArtifact,
    restore_neural_duel_blueprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CounterfactualCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = json.loads(
            (
                PROJECT_ROOT
                / "artifacts"
                / "phase6_neural_blueprint"
                / "selected_neural_blueprint.json"
            ).read_text(encoding="utf-8")
        )
        cls.game = RocketPrizeDuel()
        cls.blueprint = restore_neural_duel_blueprint(
            cls.game,
            NeuralBlueprintArtifact.from_dict(raw),
        )
        cls.calibration = calibrate_neural_counterfactual_bounds(
            cls.game,
            cls.blueprint,
        )

    def _representative(self):
        identifier = next(
            item.information_set_id
            for item in self.calibration.examples
            if item.split == "validation"
        )
        information_set = next(
            item for item in self.game.information_sets() if item.info_set_id == identifier
        )
        state = self.game.states_for_information_set(identifier)[0]
        actor = state.acting_player
        self.assertIsNotNone(actor)
        return information_set, self.game.public_belief(state, player=actor)

    def test_calibration_split_coverage_roundtrip_and_tamper_rejection(self):
        artifact = self.calibration
        self.assertEqual(artifact.metrics.calibration_examples, 12)
        self.assertEqual(artifact.metrics.validation_examples, 12)
        self.assertEqual(artifact.metrics.validation_coverage, 1.0)
        self.assertEqual(artifact.metrics.maximum_validation_miss, 0.0)
        self.assertTrue(artifact.passes_exact_gate)

        encoded = json.loads(json.dumps(artifact.to_dict()))
        restored = CounterfactualBoundCalibrationArtifact.from_dict(encoded)
        self.assertEqual(restored, artifact)
        self.assertEqual(restored.fingerprint, artifact.fingerprint)

        encoded["metrics"]["conformal_radius"] += 0.01
        with self.assertRaises(ValueError):
            CounterfactualBoundCalibrationArtifact.from_dict(encoded)

    def test_uncertainty_widens_and_high_risk_contexts_fail_closed(self):
        information_set, belief = self._representative()
        provider = CalibratedNeuralCounterfactualBoundProvider(
            self.game,
            self.blueprint,
            self.calibration,
        )
        base = provider.decide_bounds(information_set, belief)
        widened = provider.decide_bounds(
            information_set,
            belief,
            context=CounterfactualBoundContext(
                abstraction_error_bound=0.01,
                ood_score=0.20,
                incomplete_search_fraction=0.25,
                required_confidence=0.90,
            ),
        )
        self.assertIsNotNone(base.bounds)
        self.assertIsNotNone(widened.bounds)
        self.assertAlmostEqual(widened.half_width - base.half_width, 0.045)

        for context in (
            CounterfactualBoundContext(ood_score=0.51),
            CounterfactualBoundContext(incomplete_search_fraction=0.51),
            CounterfactualBoundContext(required_confidence=0.91),
        ):
            decision = provider.decide_bounds(
                information_set,
                belief,
                context=context,
            )
            self.assertTrue(decision.fallback_to_blueprint)
            self.assertIsNone(decision.bounds)

        with self.assertRaises(BoundGateFailure):
            provider.bounds(information_set, belief, object())

    def test_exact_audit_finds_no_false_safe_certificate(self):
        audit = audit_calibrated_counterfactual_bounds(
            self.game,
            self.blueprint,
            self.calibration,
        )
        self.assertTrue(audit.passes_exact_gate)
        self.assertEqual(audit.validation_information_sets, 6)
        self.assertEqual(audit.validation_hypotheses, 12)
        self.assertEqual(audit.false_safe_certificates, 0)
        self.assertEqual(audit.maximum_true_interval_violation, 0.0)
        self.assertEqual(audit.maximum_certified_bound_violation, 0.0)
        self.assertEqual(audit.maximum_regression_beyond_declared_uncertainty, 0.0)


if __name__ == "__main__":
    unittest.main()
