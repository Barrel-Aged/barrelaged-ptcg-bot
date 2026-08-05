from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from rocketbot import (
    OFFICIAL_INTENT_FEATURE_NAMES,
    OFFICIAL_INTENT_MAIN_ACTION_BUDGET,
    OFFICIAL_INTENT_PARTICLES_PER_CANDIDATE,
    OFFICIAL_INTENT_ROOTS_PER_CANDIDATE_INTENT_SPLIT,
    OPPONENT_INTENTS,
    OfficialIntentRangeModel,
    OfficialIntentTruthDataset,
    OfficialJointBeliefCalibration,
    OfficialJointBeliefProvider,
    OfficialPublicRangeModel,
    OfficialRangeCalibration,
    OfficialRangeTruthDataset,
    OpponentIntent,
    OptionType,
    Phase6OfficialIntentEvidence,
    StrategicIntentPriorityBaseline,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_official_intent_calibration"
SOURCE_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_official_range_calibration"


class OfficialIntentCalibrationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(directory: Path, name: str) -> dict:
            return json.loads((directory / name).read_text(encoding="utf-8"))

        cls.source_dataset = OfficialRangeTruthDataset.from_dict(
            raw(SOURCE_DIRECTORY, "official_range_truth_dataset.json")
        )
        cls.source_model = OfficialPublicRangeModel.from_dict(
            raw(SOURCE_DIRECTORY, "official_public_range_model.json")
        )
        cls.source_calibration = OfficialRangeCalibration.from_dict(
            raw(SOURCE_DIRECTORY, "official_range_calibration.json")
        )
        cls.dataset_raw = raw(
            ARTIFACT_DIRECTORY, "official_intent_truth_dataset.json"
        )
        cls.model_raw = raw(
            ARTIFACT_DIRECTORY, "official_intent_range_model.json"
        )
        cls.calibration_raw = raw(
            ARTIFACT_DIRECTORY, "official_joint_belief_calibration.json"
        )
        cls.evidence_raw = raw(ARTIFACT_DIRECTORY, "phase6ae_evidence.json")
        cls.dataset = OfficialIntentTruthDataset.from_dict(cls.dataset_raw)
        cls.model = OfficialIntentRangeModel.from_dict(cls.model_raw)
        cls.calibration = OfficialJointBeliefCalibration.from_dict(
            cls.calibration_raw
        )
        cls.evidence = Phase6OfficialIntentEvidence.from_dict(cls.evidence_raw)

    def test_collection_is_seat_decoupled_split_complete_and_public_only(self) -> None:
        dataset = self.dataset
        self.assertEqual(
            dataset.source_phase6ad_dataset_fingerprint,
            self.source_dataset.fingerprint,
        )
        self.assertEqual(
            dataset.source_phase6ad_model_fingerprint,
            self.source_model.fingerprint,
        )
        self.assertEqual(
            dataset.roots_per_candidate_intent_split,
            OFFICIAL_INTENT_ROOTS_PER_CANDIDATE_INTENT_SPLIT,
        )
        self.assertEqual(
            dataset.particles_per_candidate,
            OFFICIAL_INTENT_PARTICLES_PER_CANDIDATE,
        )
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertEqual(len(dataset.intent_ids), 6)
        self.assertEqual(len(dataset.examples), 324)
        self.assertEqual(
            dict(dataset.roots_per_split),
            {"train": 108, "calibration": 108, "validation": 108},
        )
        self.assertEqual(
            {
                (
                    item.split,
                    item.candidate_id_supervision_only,
                    item.intent_id_supervision_only,
                    item.collection_ordinal,
                )
                for item in dataset.examples
            },
            {
                (split, candidate, intent, ordinal)
                for split in ("train", "calibration", "validation")
                for candidate in dataset.candidate_ids
                for intent in dataset.intent_ids
                for ordinal in range(3)
            },
        )
        self.assertTrue(dataset.collection_split_disjoint)
        self.assertTrue(dataset.target_policy_fixed_across_intents)
        self.assertTrue(dataset.opponent_only_action_semantics)
        self.assertTrue(dataset.observer_orientation_validated)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)

    def test_trace_and_feature_boundaries_exclude_target_behavior_and_private_ids(self) -> None:
        for example in self.dataset.examples:
            self.assertEqual(
                tuple(name for name, _ in example.intent_public_features),
                OFFICIAL_INTENT_FEATURE_NAMES,
            )
            self.assertFalse(example.candidate_identifier_selected_for_model_input)
            self.assertFalse(example.intent_identifier_selected_for_model_input)
            self.assertFalse(example.truth_resource_counts_selected_for_model_input)
            self.assertFalse(
                example.target_seat_action_semantics_selected_for_model_input
            )
            self.assertEqual(example.raw_private_card_ids_persisted, 0)
            self.assertTrue(example.prefix_trace.target_policy_fixed_across_intents)
            self.assertTrue(
                example.prefix_trace.opponent_policy_is_declared_intent_proxy
            )
            self.assertGreater(len(example.prefix_trace.opponent_main_action_types), 0)
            for step in example.prefix_trace.steps:
                self.assertTrue(step.official_battle_selection_executed)
                self.assertEqual(step.raw_selection_indices_persisted, 0)
                self.assertEqual(step.raw_private_card_ids_persisted, 0)
                if step.selected_public_main_action_types:
                    self.assertEqual(step.actor, 1)
                    self.assertEqual(step.selection_context, 0)

    def test_train_only_model_emits_normalized_joint_candidate_intent_range(self) -> None:
        train_ids = {
            item.source_root_id for item in self.dataset.examples if item.split == "train"
        }
        calibration_ids = {
            item.source_root_id
            for item in self.dataset.examples
            if item.split == "calibration"
        }
        validation_ids = {
            item.source_root_id
            for item in self.dataset.examples
            if item.split == "validation"
        }
        self.assertEqual(set(self.model.train_root_ids), train_ids)
        self.assertEqual(set(self.model.calibration_root_ids), calibration_ids)
        self.assertTrue(train_ids.isdisjoint(calibration_ids | validation_ids))
        self.assertTrue(calibration_ids.isdisjoint(validation_ids))
        self.assertTrue(self.model.public_features_only)
        self.assertTrue(self.model.opponent_action_semantics_only)
        self.assertFalse(self.model.selected_candidate_identifier_input)
        self.assertFalse(self.model.selected_intent_identifier_input)
        self.assertFalse(self.model.target_policy_action_input)
        self.assertFalse(self.model.intent_labels_are_official_latent_truth)
        self.assertFalse(self.model.official_engine_executed)

        example = next(
            item for item in self.dataset.examples if item.split == "validation"
        )
        provider = OfficialJointBeliefProvider(self.source_model, self.model)
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
        self.assertEqual(
            {item.intent_id for item in opponent_range.hypotheses},
            {item.value for item in OPPONENT_INTENTS},
        )
        self.assertTrue(
            all(len(item.hidden_resource_counts) == 96 for item in opponent_range.hypotheses)
        )

    def test_preregistered_in_scope_gates_pass_but_latent_and_control_gates_stay_closed(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertTrue(calibration.source_phase6ad_gate_passed)
        self.assertTrue(calibration.root_split_gate_passed)
        self.assertTrue(calibration.sufficient_calibration_roots_gate_passed)
        self.assertTrue(calibration.sufficient_validation_roots_gate_passed)
        self.assertTrue(calibration.observer_and_seat_decoupling_gate_passed)
        self.assertTrue(calibration.heldout_candidate_transfer_gate_passed)
        self.assertTrue(calibration.heldout_declared_intent_proxy_gate_passed)
        self.assertTrue(calibration.heldout_joint_range_gate_passed)
        self.assertTrue(calibration.heldout_hidden_resource_gate_passed)
        self.assertTrue(
            calibration.in_scope_joint_candidate_intent_resource_range_calibrated
        )
        self.assertGreaterEqual(
            calibration.validation.intent_top1_accuracy,
            calibration.minimum_intent_top1,
        )
        self.assertLessEqual(
            calibration.validation.intent_ece,
            calibration.intent_ece_limit,
        )
        self.assertLessEqual(
            calibration.validation.joint_ece,
            calibration.joint_ece_limit,
        )
        self.assertLessEqual(
            calibration.validation.resource_ece,
            calibration.resource_ece_limit,
        )
        self.assertTrue(evidence.declared_policy_intent_proxy_calibrated)
        self.assertTrue(
            evidence.in_scope_joint_candidate_intent_resource_range_calibrated
        )
        self.assertEqual(evidence.total_public_consistent_particles, 62_208)
        self.assertFalse(evidence.opponent_latent_intent_calibrated)
        self.assertFalse(evidence.full_official_public_belief_calibrated)
        self.assertFalse(evidence.counterfactual_value_compatible)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.opponent_exploitation_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)

    def test_checked_artifacts_roundtrip_and_reject_promoted_claims(self) -> None:
        for artifact, raw in (
            (self.dataset, self.dataset_raw),
            (self.model, self.model_raw),
            (self.calibration, self.calibration_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        with self.assertRaises(ValueError):
            replace(self.model, intent_labels_are_official_latent_truth=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, counterfactual_value_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, safe_resolver_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, opponent_latent_intent_calibrated=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)

    def test_declared_proxy_action_budgets_are_deterministic_and_resettable(self) -> None:
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
                    {"type": int(OptionType.RETREAT)},
                    {"type": int(OptionType.ATTACK)},
                    {"type": int(OptionType.END)},
                ],
            },
        }
        end_index = 6
        for intent in OPPONENT_INTENTS:
            policy = StrategicIntentPriorityBaseline(
                tuple(range(60)), intent=intent, reverse=False
            )
            budget = OFFICIAL_INTENT_MAIN_ACTION_BUDGET[intent]
            for _ in range(budget):
                self.assertNotEqual(policy.choose(observation), [end_index])
            self.assertEqual(policy.choose(observation), [end_index])
            policy.reset()
            if budget:
                self.assertNotEqual(policy.choose(observation), [end_index])
            else:
                self.assertEqual(policy.choose(observation), [end_index])


if __name__ == "__main__":
    unittest.main()
