from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from rocketbot import (
    NATIVE_PUBLIC_FEATURE_NAMES,
    OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE,
    OFFICIAL_RANGE_RESOURCE_NAMES,
    OFFICIAL_RANGE_ROOTS_PER_CANDIDATE_PER_SPLIT,
    JointOfficialValueDataset,
    OfficialPublicRangeModel,
    OfficialRangeCalibration,
    OfficialRangeTruthDataset,
    OpponentRange,
    Phase6OfficialRangeEvidence,
    RangeHypothesis,
    native_observer_public_feature_items,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_official_range_calibration"
SOURCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6_joint_official_values"
    / "joint_official_value_dataset.json"
)


def _observer_fixture() -> dict:
    player_0 = {
        "active": [],
        "bench": [],
        "benchMax": 5,
        "deckCount": 42,
        "discard": [{"id": 431}],
        "prize": [None] * 6,
        "handCount": 7,
        "hand": None,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }
    player_1 = {
        **player_0,
        "deckCount": 39,
        "discard": [{"id": 555}],
        "handCount": 9,
        # The prompt belongs to player 1, so these identities are private to
        # player 1 and must not enter player 0's observer-oriented features.
        "hand": [
            {"id": 1001, "serial": index, "playerIndex": 1}
            for index in range(9)
        ],
    }
    return {
        "select": {
            "type": 1,
            "context": 8,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": 3, "area": 2, "index": 0, "playerIndex": 1}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "search_begin_input": None,
        "current": {
            "turn": 7,
            "turnActionCount": 4,
            "yourIndex": 1,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [player_0, player_1],
        },
    }


def _uniform_range() -> OpponentRange:
    return OpponentRange(
        (
            RangeHypothesis("candidate:a", 0.5, (), "unknown", "unknown", "test"),
            RangeHypothesis("candidate:b", 0.5, (), "unknown", "unknown", "test"),
        )
    )


class OfficialRangeCalibrationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads((ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8"))

        cls.source = JointOfficialValueDataset.from_dict(
            json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        )
        cls.dataset_raw = raw("official_range_truth_dataset.json")
        cls.model_raw = raw("official_public_range_model.json")
        cls.calibration_raw = raw("official_range_calibration.json")
        cls.evidence_raw = raw("phase6ad_evidence.json")
        cls.dataset = OfficialRangeTruthDataset.from_dict(cls.dataset_raw)
        cls.model = OfficialPublicRangeModel.from_dict(cls.model_raw)
        cls.calibration = OfficialRangeCalibration.from_dict(cls.calibration_raw)
        cls.evidence = Phase6OfficialRangeEvidence.from_dict(cls.evidence_raw)

    def test_collection_is_genuine_split_complete_and_public_only(self) -> None:
        dataset = self.dataset
        self.assertEqual(dataset.source_phase6ac_dataset_fingerprint, self.source.fingerprint)
        self.assertEqual(dataset.runtime_fingerprint, self.source.runtime_fingerprint)
        self.assertEqual(dataset.deck_fingerprint, self.source.deck_fingerprint)
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertEqual(dataset.roots_per_candidate_per_split, 10)
        self.assertEqual(dataset.particles_per_candidate, 32)
        self.assertEqual(len(dataset.examples), 180)
        self.assertEqual(dict(dataset.roots_per_split), {
            "train": 60,
            "calibration": 60,
            "validation": 60,
        })
        self.assertEqual(dataset.live_selection_count, 5_178)
        self.assertEqual(
            {
                (item.split, item.candidate_id_supervision_only, item.collection_ordinal)
                for item in dataset.examples
            },
            {
                (split, candidate, ordinal)
                for split in ("train", "calibration", "validation")
                for candidate in dataset.candidate_ids
                for ordinal in range(10)
            },
        )
        self.assertTrue(dataset.collection_split_disjoint)
        self.assertTrue(dataset.observer_orientation_validated)
        self.assertTrue(dataset.official_battle_roots_executed)
        self.assertTrue(dataset.visualizer_truth_supervision_only)
        self.assertFalse(dataset.model_transformation_official_engine_executed)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)

    def test_truth_and_particle_accounting_preserves_the_public_zone_sizes(self) -> None:
        self.assertEqual(len(OFFICIAL_RANGE_RESOURCE_NAMES), 96)
        for example in self.dataset.examples:
            self.assertEqual(example.observer_player, 0)
            self.assertEqual(example.root_actor, 1)
            self.assertEqual(
                tuple(name for name, _ in example.public_features),
                NATIVE_PUBLIC_FEATURE_NAMES,
            )
            self.assertTrue(example.prefix_trace.official_battle_root_executed)
            self.assertTrue(example.prefix_trace.official_action_prefix_complete)
            self.assertTrue(
                all(
                    step.official_battle_selection_executed
                    for step in example.prefix_trace.steps
                )
            )
            self.assertEqual(len(example.candidate_particle_summaries), 6)
            for summary in example.candidate_particle_summaries:
                self.assertEqual(
                    summary.particle_count,
                    OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE,
                )
                zone_sums = tuple(
                    sum(summary.expected_resource_counts[start : start + 32])
                    for start in (0, 32, 64)
                )
                for observed, expected in zip(
                    zone_sums,
                    example.truth_zone_counts,
                    strict=True,
                ):
                    self.assertTrue(math.isclose(observed, expected, abs_tol=1e-8))
            true_summary = next(
                summary
                for summary in example.candidate_particle_summaries
                if summary.candidate_id == example.candidate_id_supervision_only
            )
            self.assertEqual(true_summary.maximum_public_evidence_adjustments, 0)

    def test_provider_is_split_isolated_and_emits_a_normalized_range(self) -> None:
        model = self.model
        train = {
            item.source_root_id for item in self.dataset.examples if item.split == "train"
        }
        calibration = {
            item.source_root_id
            for item in self.dataset.examples
            if item.split == "calibration"
        }
        validation = {
            item.source_root_id
            for item in self.dataset.examples
            if item.split == "validation"
        }
        self.assertEqual(set(model.train_root_ids), train)
        self.assertEqual(set(model.calibration_root_ids), calibration)
        self.assertTrue(train.isdisjoint(calibration | validation))
        self.assertTrue(calibration.isdisjoint(validation))
        self.assertTrue(model.public_features_only)
        self.assertTrue(model.hidden_resource_particle_marginals)
        self.assertFalse(model.selected_truth_identifier_input)
        self.assertFalse(model.opponent_intent_modeled)
        self.assertFalse(model.official_engine_executed)

        example = next(item for item in self.dataset.examples if item.split == "validation")
        opponent_range = model.opponent_range(
            example.public_features,
            example.candidate_particle_summaries,
        )
        self.assertEqual(len(opponent_range.hypotheses), 6)
        self.assertTrue(
            math.isclose(
                sum(item.weight for item in opponent_range.hypotheses),
                1.0,
                abs_tol=1e-12,
            )
        )
        self.assertTrue(
            all(
                len(item.hidden_resource_counts) == len(OFFICIAL_RANGE_RESOURCE_NAMES)
                for item in opponent_range.hypotheses
            )
        )
        self.assertTrue(all(item.intent_id == "unknown" for item in opponent_range.hypotheses))

    def test_preregistered_in_scope_gates_pass_but_downstream_gates_stay_closed(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertEqual(calibration.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(calibration.model_fingerprint, self.model.fingerprint)
        self.assertTrue(calibration.root_split_gate_passed)
        self.assertTrue(calibration.sufficient_calibration_roots_gate_passed)
        self.assertTrue(calibration.sufficient_validation_roots_gate_passed)
        self.assertTrue(calibration.observer_orientation_gate_passed)
        self.assertTrue(calibration.heldout_candidate_calibration_gate_passed)
        self.assertTrue(calibration.heldout_hidden_resource_calibration_gate_passed)
        self.assertTrue(calibration.in_scope_official_hidden_card_range_calibrated)
        self.assertLess(calibration.validation.candidate_nll, calibration.uniform_candidate_nll)
        self.assertLess(
            calibration.validation.resource_presence_brier,
            calibration.validation.uniform_candidate_resource_brier,
        )
        self.assertLessEqual(
            calibration.validation.candidate_expected_calibration_error,
            calibration.candidate_ece_limit,
        )
        self.assertLessEqual(
            calibration.validation.resource_expected_calibration_error,
            calibration.resource_ece_limit,
        )
        self.assertTrue(evidence.in_scope_official_hidden_card_range_calibrated)
        self.assertEqual(evidence.total_public_consistent_particles, 34_560)
        self.assertFalse(evidence.opponent_intent_range_calibrated)
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
            replace(self.dataset, model_transformation_official_engine_executed=True)
        with self.assertRaises(ValueError):
            replace(self.model, opponent_intent_modeled=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, full_official_public_belief_calibrated=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, safe_resolver_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)

    def test_player_zero_features_ignore_player_one_private_hand_identities(self) -> None:
        observation = _observer_fixture()
        alternate = deepcopy(observation)
        alternate["current"]["players"][1]["hand"] = [
            {"id": 9000 + index, "serial": 100 + index, "playerIndex": 1}
            for index in range(9)
        ]
        baseline_features = native_observer_public_feature_items(
            observation,
            _uniform_range(),
            observer_player=0,
        )
        alternate_features = native_observer_public_feature_items(
            alternate,
            _uniform_range(),
            observer_player=0,
        )
        self.assertEqual(baseline_features, alternate_features)

        public_change = deepcopy(observation)
        public_change["current"]["players"][1]["discard"].append({"id": 777})
        self.assertNotEqual(
            baseline_features,
            native_observer_public_feature_items(
                public_change,
                _uniform_range(),
                observer_player=0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
