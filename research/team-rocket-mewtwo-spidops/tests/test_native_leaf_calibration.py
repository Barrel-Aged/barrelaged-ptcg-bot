from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

from rocketbot.phase6_native_leaf_calibration import (
    NativeLeafCalibrationArtifact,
    NativeLeafContinuationDataset,
    Phase6NativeLeafCalibrationEvidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeLeafCalibrationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = PROJECT_ROOT / "artifacts" / "phase6_native_leaf_calibration"
        cls.dataset_raw = json.loads(
            (directory / "native_leaf_continuation_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        cls.calibration_raw = json.loads(
            (directory / "native_leaf_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        cls.evidence_raw = json.loads(
            (directory / "phase6s_evidence.json").read_text(encoding="utf-8")
        )
        cls.dataset = NativeLeafContinuationDataset.from_dict(cls.dataset_raw)
        cls.calibration = NativeLeafCalibrationArtifact.from_dict(
            cls.calibration_raw
        )
        cls.evidence = Phase6NativeLeafCalibrationEvidence.from_dict(
            cls.evidence_raw
        )

    def test_dataset_has_disjoint_order_particles_and_no_truth_features(self) -> None:
        dataset = self.dataset
        self.assertEqual(len(dataset.examples), 18)
        self.assertEqual(dataset.rollouts_per_leaf, 3)
        self.assertTrue(dataset.hidden_zone_order_heldout)
        self.assertFalse(dataset.hidden_zone_allocation_heldout)
        self.assertFalse(dataset.supplied_public_range_calibrated)
        self.assertFalse(dataset.exact_counterfactual_targets)
        self.assertEqual(dataset.raw_private_card_ids_persisted, 0)
        calibration_particles = {
            item.particle_id
            for item in dataset.examples
            if item.particle_split == "calibration"
        }
        validation_particles = {
            item.particle_id
            for item in dataset.examples
            if item.particle_split == "validation"
        }
        self.assertEqual(len(calibration_particles), 12)
        self.assertEqual(len(validation_particles), 6)
        self.assertFalse(calibration_particles & validation_particles)
        self.assertTrue(
            all(not item.selected_hidden_truth_in_policy_features for item in dataset.examples)
        )
        self.assertTrue(
            all(not item.target_exact_counterfactual_value for item in dataset.examples)
        )
        restored = NativeLeafContinuationDataset.from_dict(
            json.loads(dataset.to_json())
        )
        self.assertEqual(restored.fingerprint, dataset.fingerprint)

    def test_empirical_interval_is_vacuous_and_cannot_promote_safety(self) -> None:
        calibration = self.calibration
        self.assertEqual(calibration.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(calibration.validation_coverage, 1.0)
        self.assertEqual(calibration.mean_interval_width, 2.0)
        self.assertEqual(calibration.informative_interval_fraction, 0.0)
        self.assertGreater(calibration.maximum_rollout_standard_error, 0.35)
        self.assertFalse(calibration.empirical_rollout_leaf_gate_passed)
        self.assertFalse(calibration.counterfactual_value_gate_passed)
        self.assertFalse(calibration.public_belief_calibration_gate_passed)
        self.assertTrue(
            math.isclose(
                calibration.public_range_nll,
                calibration.uniform_range_nll,
                abs_tol=1e-12,
            )
        )
        self.assertAlmostEqual(calibration.public_range_top1_accuracy, 1.0 / 6.0)

    def test_evidence_enables_nothing_and_rejects_capability_tampering(self) -> None:
        evidence = self.evidence
        self.assertEqual(evidence.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(
            evidence.calibration_fingerprint, self.calibration.fingerprint
        )
        self.assertTrue(evidence.heldout_particle_order_validated)
        self.assertFalse(evidence.heldout_particle_allocation_validated)
        self.assertFalse(evidence.public_belief_calibrated)
        self.assertFalse(evidence.counterfactual_values_independently_calibrated)
        self.assertFalse(evidence.counterfactual_bounds_built)
        self.assertFalse(evidence.safe_resolver_executed)
        self.assertFalse(evidence.exploitation_executed)
        self.assertFalse(evidence.official_strategy_control_enabled)
        self.assertEqual(dict(evidence.claims)["phase6_complete"], "false")

        tampered = copy.deepcopy(self.evidence_raw)
        tampered.pop("fingerprint", None)
        tampered["execution"]["counterfactual_bounds_built"] = True
        with self.assertRaises(ValueError):
            Phase6NativeLeafCalibrationEvidence.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
