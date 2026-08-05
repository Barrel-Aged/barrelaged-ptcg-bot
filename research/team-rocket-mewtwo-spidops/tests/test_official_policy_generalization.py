from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from rocketbot import (
    OFFICIAL_INTENT_POLICY_VARIANT_IDS,
    OFFICIAL_POLICY_MINIMUM_SLICE_TOP1,
    OFFICIAL_POLICY_PARTICLES_PER_CANDIDATE,
    OFFICIAL_POLICY_ROOTS_PER_CANDIDATE_INTENT_SPLIT,
    OFFICIAL_POLICY_VARIANT_BY_SPLIT,
    OfficialJointBeliefProvider,
    OfficialPolicyHeldoutCalibration,
    OfficialPolicyHeldoutIntentModel,
    OfficialPolicyVariantDataset,
    OfficialPublicRangeModel,
    OfficialRangeTruthDataset,
    OpponentIntent,
    OptionType,
    Phase6OfficialIntentEvidence,
    Phase6OfficialPolicyGeneralizationEvidence,
    StrategicIntentPriorityBaseline,
    official_intent_policy_variant_fingerprint,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "phase6_official_policy_generalization"
)
SOURCE_AD_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_official_range_calibration"
SOURCE_AE_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_official_intent_calibration"


class OfficialPolicyGeneralizationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(directory: Path, name: str) -> dict:
            return json.loads((directory / name).read_text(encoding="utf-8"))

        cls.source_ad_dataset = OfficialRangeTruthDataset.from_dict(
            raw(SOURCE_AD_DIRECTORY, "official_range_truth_dataset.json")
        )
        cls.source_ad_model = OfficialPublicRangeModel.from_dict(
            raw(SOURCE_AD_DIRECTORY, "official_public_range_model.json")
        )
        cls.source_ae_evidence = Phase6OfficialIntentEvidence.from_dict(
            raw(SOURCE_AE_DIRECTORY, "phase6ae_evidence.json")
        )
        cls.dataset_raw = raw(
            ARTIFACT_DIRECTORY, "official_policy_variant_dataset.json"
        )
        cls.model_raw = raw(
            ARTIFACT_DIRECTORY, "official_policy_heldout_model.json"
        )
        cls.calibration_raw = raw(
            ARTIFACT_DIRECTORY, "official_policy_heldout_calibration.json"
        )
        cls.evidence_raw = raw(ARTIFACT_DIRECTORY, "phase6af_evidence.json")
        cls.dataset = OfficialPolicyVariantDataset.from_dict(cls.dataset_raw)
        cls.model = OfficialPolicyHeldoutIntentModel.from_dict(cls.model_raw)
        cls.calibration = OfficialPolicyHeldoutCalibration.from_dict(
            cls.calibration_raw
        )
        cls.evidence = Phase6OfficialPolicyGeneralizationEvidence.from_dict(
            cls.evidence_raw
        )

    def test_collection_has_four_split_isolated_policy_implementations(self) -> None:
        dataset = self.dataset
        self.assertEqual(
            dataset.source_phase6ae_evidence_fingerprint,
            self.source_ae_evidence.fingerprint,
        )
        self.assertEqual(
            dataset.source_phase6ad_dataset_fingerprint,
            self.source_ad_dataset.fingerprint,
        )
        self.assertEqual(
            dataset.roots_per_candidate_intent_split,
            OFFICIAL_POLICY_ROOTS_PER_CANDIDATE_INTENT_SPLIT,
        )
        self.assertEqual(
            dataset.particles_per_candidate,
            OFFICIAL_POLICY_PARTICLES_PER_CANDIDATE,
        )
        self.assertEqual(dataset.policy_variant_by_split, OFFICIAL_POLICY_VARIANT_BY_SPLIT)
        self.assertEqual(len(dataset.examples), 432)
        self.assertEqual(
            dict(dataset.roots_per_split),
            {"train": 144, "calibration": 144, "validation": 144},
        )
        variants = {
            split: {
                item.policy_variant_id_supervision_only
                for item in dataset.examples
                if item.base_example.split == split
            }
            for split in ("train", "calibration", "validation")
        }
        self.assertEqual(variants["train"], {"canonical", "engine_first"})
        self.assertEqual(variants["calibration"], {"hand_first"})
        self.assertEqual(variants["validation"], {"action_first"})
        self.assertTrue(variants["train"].isdisjoint(variants["calibration"] | variants["validation"]))
        self.assertTrue(variants["calibration"].isdisjoint(variants["validation"]))
        self.assertTrue(dataset.policy_variants_split_disjoint)
        self.assertTrue(dataset.target_policy_fixed_across_intents_and_variants)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)

    def test_variant_identity_is_supervision_only_and_public_trace_boundary_holds(self) -> None:
        for item in self.dataset.examples:
            base = item.base_example
            self.assertFalse(item.policy_variant_identifier_selected_for_model_input)
            self.assertFalse(item.policy_variant_is_human_opponent_truth)
            self.assertFalse(base.candidate_identifier_selected_for_model_input)
            self.assertFalse(base.intent_identifier_selected_for_model_input)
            self.assertFalse(base.truth_resource_counts_selected_for_model_input)
            self.assertFalse(base.target_seat_action_semantics_selected_for_model_input)
            self.assertEqual(base.raw_private_card_ids_persisted, 0)
            self.assertGreater(len(base.prefix_trace.opponent_main_action_types), 0)
            self.assertTrue(
                all(
                    not step.selected_public_main_action_types or step.actor == 1
                    for step in base.prefix_trace.steps
                )
            )

    def test_model_crossfits_training_policies_and_never_selects_validation_roots(self) -> None:
        model = self.model
        base = self.dataset.intent_dataset_view
        train_ids = {
            item.source_root_id for item in base.examples if item.split == "train"
        }
        calibration_ids = {
            item.source_root_id
            for item in base.examples
            if item.split == "calibration"
        }
        validation_ids = {
            item.source_root_id
            for item in base.examples
            if item.split == "validation"
        }
        self.assertEqual(set(model.intent_model.train_root_ids), train_ids)
        self.assertEqual(
            set(model.intent_model.calibration_root_ids), calibration_ids
        )
        self.assertTrue(train_ids.isdisjoint(calibration_ids | validation_ids))
        self.assertTrue(calibration_ids.isdisjoint(validation_ids))
        self.assertEqual(
            model.training_policy_crossfit_variants,
            ("canonical", "engine_first"),
        )
        self.assertIn("leave_one_variant_out", model.selection_protocol)
        self.assertFalse(model.policy_variant_identifier_selected_for_model_input)
        self.assertFalse(model.validation_policy_variant_selected_during_training)
        self.assertFalse(model.intent_labels_are_human_latent_truth)
        self.assertFalse(model.official_engine_executed)

    def test_heldout_joint_provider_is_normalized_and_all_family_slices_pass(self) -> None:
        calibration = self.calibration
        self.assertTrue(calibration.source_phase6ae_gate_passed)
        self.assertTrue(calibration.policy_variant_split_gate_passed)
        self.assertTrue(calibration.heldout_intent_family_slice_gate_passed)
        self.assertTrue(calibration.implementation_heldout_joint_range_calibrated)
        self.assertTrue(
            calibration.joint_calibration.in_scope_joint_candidate_intent_resource_range_calibrated
        )
        self.assertTrue(
            all(
                item.top1_accuracy >= OFFICIAL_POLICY_MINIMUM_SLICE_TOP1
                for item in calibration.validation_intent_family_slices
            )
        )
        example = next(
            item.base_example
            for item in self.dataset.examples
            if item.base_example.split == "validation"
        )
        provider = OfficialJointBeliefProvider(
            self.source_ad_model,
            self.model.intent_model,
        )
        opponent_range = provider.opponent_range(
            example.candidate_public_features,
            example.intent_public_features,
            example.candidate_particle_summaries,
        )
        self.assertEqual(len(opponent_range.hypotheses), 36)
        self.assertTrue(
            math.isclose(
                sum(item.weight for item in opponent_range.hypotheses),
                1.0,
                abs_tol=1e-12,
            )
        )

    def test_checked_evidence_passes_proxy_transfer_only_and_stays_fail_closed(self) -> None:
        evidence = self.evidence
        self.assertEqual(evidence.root_count, 432)
        self.assertEqual(evidence.roots_per_split, 144)
        self.assertEqual(evidence.total_public_consistent_particles, 82_944)
        self.assertTrue(evidence.implementation_heldout_joint_range_calibrated)
        self.assertGreaterEqual(
            evidence.minimum_validation_intent_slice_top1,
            OFFICIAL_POLICY_MINIMUM_SLICE_TOP1,
        )
        self.assertFalse(evidence.human_latent_intent_calibrated)
        self.assertFalse(evidence.full_official_public_belief_calibrated)
        self.assertFalse(evidence.counterfactual_value_compatible)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.opponent_exploitation_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)

    def test_artifacts_roundtrip_and_capability_tampering_is_rejected(self) -> None:
        for artifact, raw in (
            (self.dataset, self.dataset_raw),
            (self.model, self.model_raw),
            (self.calibration, self.calibration_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        with self.assertRaises(ValueError):
            replace(self.dataset, intent_labels_are_human_latent_truth=True)
        with self.assertRaises(ValueError):
            replace(self.model, validation_policy_variant_selected_during_training=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, safe_resolver_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)

    def test_policy_variant_fingerprints_and_semantic_choices_are_distinct(self) -> None:
        fingerprints = {
            official_intent_policy_variant_fingerprint(variant)
            for variant in OFFICIAL_INTENT_POLICY_VARIANT_IDS
        }
        self.assertEqual(len(fingerprints), len(OFFICIAL_INTENT_POLICY_VARIANT_IDS))
        observation = {
            "current": {"turn": 1},
            "select": {
                "type": 1,
                "context": 0,
                "minCount": 1,
                "maxCount": 1,
                "option": [
                    {"type": int(OptionType.PLAY)},
                    {"type": int(OptionType.ATTACH)},
                    {"type": int(OptionType.EVOLVE)},
                    {"type": int(OptionType.ABILITY)},
                    {"type": int(OptionType.END)},
                ],
            },
        }
        canonical = StrategicIntentPriorityBaseline(
            tuple(range(60)),
            intent=OpponentIntent.SETUP,
            reverse=False,
            variant_id="canonical",
        )
        hand_first = StrategicIntentPriorityBaseline(
            tuple(range(60)),
            intent=OpponentIntent.SETUP,
            reverse=False,
            variant_id="hand_first",
        )
        self.assertNotEqual(canonical.choose(observation), hand_first.choose(observation))


if __name__ == "__main__":
    unittest.main()
