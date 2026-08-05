from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    LEAF_MODEL_NAMES,
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeLeafModelCalibration,
    NativeLeafModelDataset,
    Phase6NativeLeafModelEvidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_native_leaf_model"


class NativeLeafModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = NativeLeafModelDataset.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "native_low_noise_leaf_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.calibration = NativeLeafModelCalibration.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "native_leaf_model_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.evidence = Phase6NativeLeafModelEvidence.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "phase6u_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    def test_ten_rollout_allocation_dataset_is_redacted_and_precise(self) -> None:
        dataset = self.dataset
        self.assertEqual(len(dataset.examples), 24)
        self.assertEqual(dataset.rollouts_per_leaf, 10)
        self.assertEqual(
            len({item.particle_fingerprint for item in dataset.examples}),
            24,
        )
        self.assertEqual(
            len({item.particle_id for item in dataset.examples if item.split == "calibration"}),
            18,
        )
        self.assertEqual(
            len({item.particle_id for item in dataset.examples if item.split == "validation"}),
            6,
        )
        self.assertTrue(dataset.allocation_heldout)
        self.assertFalse(dataset.exact_counterfactual_targets)
        self.assertEqual(dataset.raw_private_card_ids_persisted, 0)
        self.assertGreater(dataset.native_transition_count, 0)
        for example in dataset.examples:
            self.assertEqual(len(example.terminal_returns), 10)
            self.assertLessEqual(example.rollout_standard_error, 1.0 / 3.0 + 1e-12)
            self.assertEqual(tuple(name for name, _ in example.predictions), LEAF_MODEL_NAMES)
            self.assertEqual(
                tuple(name for name, _ in example.public_features),
                NATIVE_PUBLIC_FEATURE_NAMES,
            )
            self.assertEqual(
                tuple(name for name, _ in example.private_features),
                NATIVE_PRIVATE_FEATURE_NAMES,
            )
            self.assertFalse(example.target_exact_counterfactual_value)
            self.assertFalse(example.selected_hidden_truth_identifier_input)
            self.assertFalse(example.raw_private_card_ids_persisted)

    def test_low_noise_target_still_rejects_biased_leaf_models_and_safety(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertEqual(calibration.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(evidence.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(evidence.calibration_fingerprint, calibration.fingerprint)
        self.assertEqual(calibration.selected_model, "zero_baseline")
        self.assertEqual(
            tuple(item.model_name for item in calibration.model_metrics),
            LEAF_MODEL_NAMES,
        )
        self.assertAlmostEqual(calibration.maximum_rollout_standard_error, 1.0 / 3.0)
        self.assertAlmostEqual(calibration.validation_coverage, 1.0)
        self.assertAlmostEqual(calibration.mean_interval_width, 2.0)
        self.assertEqual(calibration.informative_interval_fraction, 0.0)
        self.assertFalse(calibration.empirical_fixed_profile_leaf_gate_passed)
        self.assertFalse(calibration.counterfactual_value_gate_passed)
        self.assertFalse(evidence.counterfactual_values_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        with self.assertRaises(ValueError):
            replace(evidence, counterfactual_values_calibrated=True)


if __name__ == "__main__":
    unittest.main()
