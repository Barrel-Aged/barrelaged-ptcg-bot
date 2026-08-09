from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from rocketbot import (
    AlignedPublicRangeCalibration,
    AlignedPublicRangeDataset,
    Phase6AlignedPublicRangeEvidence,
    PrototypePublicRangeModel,
    PublicRootAlignedValueDataset,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_aligned_public_range"
SOURCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6_aligned_counterfactual_values"
    / "public_root_aligned_value_dataset.json"
)


class AlignedPublicRangeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads(
                (ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8")
            )

        cls.source = PublicRootAlignedValueDataset.from_dict(
            json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        )
        cls.dataset_raw = raw("aligned_public_range_dataset.json")
        cls.model_raw = raw("prototype_public_range_model.json")
        cls.calibration_raw = raw("aligned_public_range_calibration.json")
        cls.evidence_raw = raw("phase6aa_evidence.json")
        cls.dataset = AlignedPublicRangeDataset.from_dict(cls.dataset_raw)
        cls.model = PrototypePublicRangeModel.from_dict(cls.model_raw)
        cls.calibration = AlignedPublicRangeCalibration.from_dict(
            cls.calibration_raw
        )
        cls.evidence = Phase6AlignedPublicRangeEvidence.from_dict(
            cls.evidence_raw
        )

    def test_dataset_is_root_disjoint_public_only_and_source_bound(self) -> None:
        dataset = self.dataset
        self.assertEqual(
            dataset.source_phase6z_dataset_fingerprint,
            self.source.fingerprint,
        )
        self.assertEqual(len(dataset.examples), 18)
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertTrue(dataset.root_split_disjoint)
        self.assertEqual(len({item.source_root_id for item in dataset.examples}), 18)
        for split in ("train", "calibration", "validation"):
            self.assertEqual(sum(item.split == split for item in dataset.examples), 6)
        self.assertFalse(dataset.target_candidate_identifier_in_features)
        self.assertFalse(dataset.private_features_present)
        self.assertFalse(dataset.transformation_official_engine_executed)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)

    def test_provider_uses_train_prototypes_calibration_selection_and_opponent_range(self) -> None:
        model = self.model
        train_roots = {
            item.source_root_id for item in self.dataset.examples if item.split == "train"
        }
        calibration_roots = {
            item.source_root_id
            for item in self.dataset.examples
            if item.split == "calibration"
        }
        validation_roots = {
            item.source_root_id
            for item in self.dataset.examples
            if item.split == "validation"
        }
        self.assertEqual(set(model.train_root_ids), train_roots)
        self.assertEqual(set(model.calibration_root_ids), calibration_roots)
        self.assertTrue(train_roots.isdisjoint(calibration_roots | validation_roots))
        self.assertTrue(calibration_roots.isdisjoint(validation_roots))
        self.assertEqual(model.selected_temperature, 30.0)
        self.assertEqual(model.selected_uniform_mix, 0.05)
        example = next(item for item in self.dataset.examples if item.split == "validation")
        opponent_range = model.predict(example.public_features)
        self.assertEqual(
            {item.hypothesis_id for item in opponent_range.hypotheses},
            set(self.dataset.candidate_ids),
        )
        self.assertTrue(
            math.isclose(
                sum(item.weight for item in opponent_range.hypotheses),
                1.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(all(not item.hidden_resource_counts for item in opponent_range.hypotheses))
        self.assertFalse(model.selected_truth_identifier_input)
        self.assertFalse(model.hidden_hand_or_resource_posterior)

    def test_archetype_signal_passes_but_official_range_and_safety_do_not(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertEqual(calibration.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(calibration.model_fingerprint, self.model.fingerprint)
        self.assertTrue(calibration.root_split_gate_passed)
        self.assertTrue(calibration.heldout_archetype_signal_gate_passed)
        self.assertLess(calibration.calibration.nll, calibration.uniform_nll)
        self.assertLess(calibration.validation.nll, calibration.uniform_nll)
        self.assertEqual(calibration.validation.top1_accuracy, 5 / 6)
        self.assertFalse(calibration.sufficient_calibration_roots_gate_passed)
        self.assertFalse(calibration.sufficient_validation_roots_gate_passed)
        self.assertFalse(calibration.official_public_range_calibrated)
        self.assertFalse(calibration.hidden_hand_or_resource_posterior)
        self.assertFalse(calibration.counterfactual_value_compatible)
        self.assertFalse(calibration.safe_resolver_compatible)
        self.assertFalse(evidence.official_public_range_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        with self.assertRaises(ValueError):
            replace(calibration, official_public_range_calibrated=True)
        with self.assertRaises(ValueError):
            replace(self.model, hidden_hand_or_resource_posterior=True)

    def test_every_artifact_json_shape_roundtrips_exactly(self) -> None:
        for artifact, raw in (
            (self.dataset, self.dataset_raw),
            (self.model, self.model_raw),
            (self.calibration, self.calibration_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)


if __name__ == "__main__":
    unittest.main()
