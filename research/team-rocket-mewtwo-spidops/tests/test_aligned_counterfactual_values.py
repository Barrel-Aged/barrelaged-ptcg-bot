from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    ALIGNED_VALUE_PARTICLE_VARIANTS,
    ALIGNED_VALUE_SPLITS,
    HypothesisConditionedValueModel,
    AlignedValueCalibration,
    Phase6AlignedValueEvidence,
    PublicRootAlignedValueDataset,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "phase6_aligned_counterfactual_values"
)


class AlignedCounterfactualValueArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads(
                (ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8")
            )

        cls.dataset_raw = raw("public_root_aligned_value_dataset.json")
        cls.model_raw = raw("hypothesis_conditioned_value_model.json")
        cls.calibration_raw = raw("aligned_value_calibration.json")
        cls.evidence_raw = raw("phase6z_evidence.json")
        cls.dataset = PublicRootAlignedValueDataset.from_dict(cls.dataset_raw)
        cls.model = HypothesisConditionedValueModel.from_dict(cls.model_raw)
        cls.calibration = AlignedValueCalibration.from_dict(cls.calibration_raw)
        cls.evidence = Phase6AlignedValueEvidence.from_dict(cls.evidence_raw)

    def test_public_roots_are_aligned_complete_and_collection_disjoint(self) -> None:
        dataset = self.dataset
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertEqual(len(dataset.roots), 18)
        self.assertEqual(len(dataset.surfaces), 54)
        self.assertEqual(dataset.action_target_count, 270)
        self.assertEqual(dataset.terminal_rollout_count, 810)
        self.assertEqual(dataset.particle_variants_per_root, 3)
        self.assertEqual(dataset.rollouts_per_action, 3)
        self.assertTrue(dataset.public_root_alignment_verified)
        self.assertTrue(dataset.collection_disjoint_roots_verified)
        self.assertEqual(len({root.root_id for root in dataset.roots}), 18)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)
        for split in ALIGNED_VALUE_SPLITS:
            self.assertEqual(sum(root.split == split for root in dataset.roots), 6)
        for root in dataset.roots:
            group = [surface for surface in dataset.surfaces if surface.root_id == root.root_id]
            self.assertEqual(len(group), ALIGNED_VALUE_PARTICLE_VARIANTS)
            self.assertEqual(
                len({surface.hidden_allocation_fingerprint for surface in group}),
                ALIGNED_VALUE_PARTICLE_VARIANTS,
            )
            self.assertEqual({surface.public_evidence_adjustments for surface in group}, {0})
            self.assertEqual({surface.public_history_fingerprint for surface in group}, {root.public_history_fingerprint})
            self.assertEqual({surface.information_set_id for surface in group}, {root.information_set_id})

    def test_model_uses_only_training_roots_and_hypothesis_action_inputs(self) -> None:
        model = self.model
        dataset = self.dataset
        train_roots = {root.root_id for root in dataset.roots if root.split == "train"}
        heldout_roots = {root.root_id for root in dataset.roots if root.split != "train"}
        self.assertEqual(set(model.train_root_ids), train_roots)
        self.assertTrue(set(model.train_root_ids).isdisjoint(heldout_roots))
        self.assertEqual(len(model.train_row_ids), 90)
        self.assertTrue(model.hypothesis_identifier_input)
        self.assertTrue(model.conditional_rollout_target)
        self.assertFalse(model.counterfactual_value_provider)
        surface = dataset.surfaces[0]
        predictions = [
            model.predict(
                surface.public_features,
                surface.candidate_id_hypothesis_input,
                value.action_id,
            )
            for value in surface.action_values
        ]
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in predictions))
        with self.assertRaises(ValueError):
            model.predict(surface.public_features, "unknown-candidate", "balanced_setup")

    def test_calibration_is_heldout_but_downstream_safety_stays_closed(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertEqual(calibration.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(calibration.model_fingerprint, self.model.fingerprint)
        self.assertEqual(evidence.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(evidence.model_fingerprint, self.model.fingerprint)
        self.assertEqual(evidence.calibration_fingerprint, calibration.fingerprint)
        self.assertEqual(len(calibration.action_metrics), 5)
        self.assertTrue(calibration.public_root_alignment_gate_passed)
        self.assertTrue(calibration.collection_disjoint_gate_passed)
        self.assertFalse(calibration.native_counterfactual_value_gate_passed)
        self.assertFalse(calibration.safe_resolver_compatible)
        self.assertFalse(calibration.official_strategy_control_enabled)
        self.assertFalse(evidence.native_counterfactual_values_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        with self.assertRaises(ValueError):
            replace(self.dataset, acting_player_hand_range_complete=True)
        with self.assertRaises(ValueError):
            replace(calibration, native_counterfactual_value_gate_passed=True)

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
