from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    PUBLIC_LEAF_EVALUATION_NAMES,
    ROOT_HELDOUT_TRAINING_VARIANTS,
    NativeLeafModelDataset,
    Phase6RootHeldoutLeafEvidence,
    PublicRidgeLeafArtifact,
    RootHeldoutLeafCalibration,
    restore_public_ridge_leaf,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_native_leaf_model"
CALIBRATION_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_native_public_leaf"
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_root_heldout_leaf"


def _load_dataset(path: Path) -> NativeLeafModelDataset:
    return NativeLeafModelDataset.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


class RootHeldoutLeafArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training = _load_dataset(
            TRAINING_DIRECTORY / "native_low_noise_leaf_dataset.json"
        )
        cls.calibration_dataset = _load_dataset(
            CALIBRATION_DIRECTORY / "fresh_native_leaf_dataset.json"
        )
        cls.validation_dataset = _load_dataset(
            ARTIFACT_DIRECTORY / "fresh_validation_leaf_dataset.json"
        )
        cls.model = PublicRidgeLeafArtifact.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "root_heldout_public_leaf.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.calibration = RootHeldoutLeafCalibration.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "root_heldout_leaf_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.evidence = Phase6RootHeldoutLeafEvidence.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "phase6w_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    def test_three_complete_collections_are_identity_disjoint(self) -> None:
        datasets = (self.training, self.calibration_dataset, self.validation_dataset)
        self.assertEqual(len({item.fingerprint for item in datasets}), 3)
        for dataset in datasets:
            self.assertEqual(len(dataset.examples), 24)
            self.assertEqual(dataset.rollouts_per_leaf, 10)
            self.assertEqual(
                {
                    (item.candidate_id_supervision_only, item.variant)
                    for item in dataset.examples
                },
                {
                    (candidate_id, variant)
                    for candidate_id in dataset.candidate_ids
                    for variant in ROOT_HELDOUT_TRAINING_VARIANTS
                },
            )
        for index, left in enumerate(datasets):
            for right in datasets[index + 1 :]:
                self.assertTrue(
                    {item.particle_fingerprint for item in left.examples}.isdisjoint(
                        item.particle_fingerprint for item in right.examples
                    )
                )
                self.assertTrue(
                    {item.source_state_id for item in left.examples}.isdisjoint(
                        item.source_state_id for item in right.examples
                    )
                )
                self.assertTrue(
                    {
                        item.public_history_fingerprint for item in left.examples
                    }.isdisjoint(
                        item.public_history_fingerprint for item in right.examples
                    )
                )

    def test_model_is_fit_only_to_complete_training_collection(self) -> None:
        model = self.model
        self.assertEqual(model.source_dataset_fingerprint, self.training.fingerprint)
        self.assertEqual(model.trained_on_variants, ROOT_HELDOUT_TRAINING_VARIANTS)
        self.assertEqual(len(model.train_example_ids), 24)
        self.assertEqual(model.selected_regularization, 0.1)
        self.assertEqual(
            model.public_normalizer.fit_rows_fingerprint,
            model.train_rows_fingerprint,
        )
        self.assertTrue(model.public_features_only)
        self.assertFalse(model.selected_truth_identifier_input)
        self.assertFalse(model.counterfactual_value_provider)
        self.assertFalse(model.safe_resolver_compatible)
        self.assertFalse(model.official_engine_executed)
        provider = restore_public_ridge_leaf(model)
        for example in self.validation_dataset.examples:
            value = provider.predict(example.public_features)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

    def test_root_heldout_result_rejects_point_model_and_interval(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertTrue(calibration.collection_disjoint_validation)
        self.assertEqual(calibration.conformal_rank, 23)
        self.assertEqual(len(calibration.calibration_example_ids), 24)
        self.assertEqual(len(calibration.validation_example_ids), 24)
        self.assertAlmostEqual(calibration.calibration_coverage, 23 / 24)
        self.assertAlmostEqual(calibration.validation_coverage, 22 / 24)
        self.assertGreater(calibration.mean_interval_width, 1.25)
        self.assertEqual(
            tuple(item.model_name for item in calibration.evaluation_metrics),
            PUBLIC_LEAF_EVALUATION_NAMES,
        )
        self.assertEqual(calibration.best_reference_model, "zero_baseline")
        self.assertFalse(calibration.outperformed_best_reference)
        self.assertGreater(
            calibration.validation_root_mean_square_error,
            calibration.best_reference_root_mean_square_error,
        )
        self.assertFalse(calibration.empirical_fixed_profile_leaf_gate_passed)
        self.assertFalse(calibration.counterfactual_value_gate_passed)
        self.assertFalse(calibration.safe_resolver_compatible)
        self.assertEqual(
            evidence.fresh_native_transition_count,
            self.validation_dataset.native_transition_count,
        )
        self.assertGreater(self.validation_dataset.native_transition_count, 0)
        self.assertTrue(
            all(
                item.source_descendant_official_engine_executed
                for item in self.validation_dataset.examples
            )
        )
        self.assertEqual(evidence.calibration_fingerprint, calibration.fingerprint)
        self.assertFalse(evidence.counterfactual_values_calibrated)
        self.assertFalse(evidence.public_belief_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        with self.assertRaises(ValueError):
            replace(calibration, empirical_fixed_profile_leaf_gate_passed=True)
        with self.assertRaises(ValueError):
            replace(evidence, empirical_fixed_profile_leaf_gate_passed=True)
        with self.assertRaises(ValueError):
            replace(evidence, safe_resolver_enabled=True)


if __name__ == "__main__":
    unittest.main()
