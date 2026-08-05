from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    NATIVE_PUBLIC_FEATURE_NAMES,
    PUBLIC_LEAF_EVALUATION_NAMES,
    PUBLIC_LEAF_REGULARIZATION_GRID,
    NativeLeafModelDataset,
    NativePublicLeafCalibration,
    Phase6NativePublicLeafEvidence,
    PublicRidgeLeafArtifact,
    restore_public_ridge_leaf,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_native_leaf_model"
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_native_public_leaf"


class NativePublicLeafArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training = NativeLeafModelDataset.from_dict(
            json.loads(
                (TRAINING_DIRECTORY / "native_low_noise_leaf_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.fresh = NativeLeafModelDataset.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "fresh_native_leaf_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.model = PublicRidgeLeafArtifact.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "public_ridge_leaf.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.calibration = NativePublicLeafCalibration.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "public_leaf_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.evidence = Phase6NativePublicLeafEvidence.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "phase6v_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    def test_training_is_public_only_and_split_safe(self) -> None:
        model = self.model
        self.assertEqual(model.source_dataset_fingerprint, self.training.fingerprint)
        self.assertEqual(model.regularization_grid, PUBLIC_LEAF_REGULARIZATION_GRID)
        self.assertEqual(model.selected_regularization, 1_000.0)
        self.assertEqual(model.trained_on_variants, (0, 1))
        self.assertEqual(len(model.train_example_ids), 12)
        self.assertEqual(
            set(model.train_example_ids),
            {
                item.example_id
                for item in self.training.examples
                if item.variant in (0, 1)
            },
        )
        self.assertEqual(
            model.public_normalizer.fit_rows_fingerprint,
            model.train_rows_fingerprint,
        )
        self.assertEqual(model.public_normalizer.names, NATIVE_PUBLIC_FEATURE_NAMES)
        self.assertTrue(model.public_features_only)
        self.assertFalse(model.selected_truth_identifier_input)
        self.assertFalse(model.counterfactual_value_provider)
        self.assertFalse(model.safe_resolver_compatible)
        self.assertFalse(model.official_engine_executed)
        best = min(
            model.cross_validation_metrics,
            key=lambda item: (
                item.leave_candidate_out_rmse,
                item.regularization,
            ),
        )
        self.assertEqual(best.regularization, model.selected_regularization)

    def test_public_model_roundtrip_predictions_are_exact(self) -> None:
        restored = restore_public_ridge_leaf(self.model)
        roundtrip = PublicRidgeLeafArtifact.from_dict(self.model.to_dict())
        roundtrip_provider = restore_public_ridge_leaf(roundtrip)
        for example in self.training.examples + self.fresh.examples:
            prediction = restored.predict(example.public_features)
            self.assertEqual(prediction, roundtrip_provider.predict(example.public_features))
            self.assertGreaterEqual(prediction, -1.0)
            self.assertLessEqual(prediction, 1.0)

    def test_fresh_native_holdout_is_disjoint_and_fail_closed(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertNotEqual(self.training.fingerprint, self.fresh.fingerprint)
        self.assertNotEqual(
            self.training.particle_family_fingerprint,
            self.fresh.particle_family_fingerprint,
        )
        self.assertTrue(
            {item.particle_fingerprint for item in self.training.examples}.isdisjoint(
                item.particle_fingerprint for item in self.fresh.examples
            )
        )
        self.assertEqual(len(self.fresh.examples), 24)
        self.assertEqual(self.fresh.rollouts_per_leaf, 10)
        self.assertEqual(
            evidence.fresh_native_transition_count,
            self.fresh.native_transition_count,
        )
        self.assertGreater(self.fresh.native_transition_count, 0)
        self.assertTrue(
            all(item.source_descendant_official_engine_executed for item in self.fresh.examples)
        )
        self.assertEqual(calibration.training_holdout_overlap_count, 0)
        self.assertEqual(len(calibration.calibration_example_ids), 6)
        self.assertEqual(len(calibration.validation_example_ids), 6)
        self.assertTrue(
            set(calibration.calibration_example_ids).isdisjoint(
                calibration.validation_example_ids
            )
        )
        self.assertEqual(
            tuple(item.model_name for item in calibration.evaluation_metrics),
            PUBLIC_LEAF_EVALUATION_NAMES,
        )
        public_metric = calibration.evaluation_metrics[0]
        self.assertTrue(calibration.outperformed_best_reference)
        self.assertLess(
            public_metric.validation_root_mean_square_error,
            calibration.best_reference_root_mean_square_error,
        )
        self.assertEqual(calibration.best_reference_model, "zero_baseline")
        self.assertAlmostEqual(calibration.validation_coverage, 1.0)
        self.assertGreater(calibration.mean_interval_width, 1.25)
        self.assertFalse(calibration.empirical_fixed_profile_leaf_gate_passed)
        self.assertFalse(calibration.counterfactual_value_gate_passed)
        self.assertFalse(calibration.safe_resolver_compatible)
        with self.assertRaises(ValueError):
            replace(calibration, empirical_fixed_profile_leaf_gate_passed=True)
        self.assertEqual(evidence.calibration_fingerprint, calibration.fingerprint)
        self.assertFalse(evidence.counterfactual_values_calibrated)
        self.assertFalse(evidence.public_belief_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        with self.assertRaises(ValueError):
            replace(evidence, empirical_fixed_profile_leaf_gate_passed=True)
        with self.assertRaises(ValueError):
            replace(evidence, safe_resolver_enabled=True)

    def test_serialized_evidence_rejects_tampering(self) -> None:
        raw = self.evidence.to_dict()
        raw["validation_mean_absolute_error"] += 0.1
        with self.assertRaises(ValueError):
            Phase6NativePublicLeafEvidence.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
