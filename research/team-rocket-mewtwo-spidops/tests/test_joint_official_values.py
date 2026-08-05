from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import unittest

from rocketbot import (
    JointOfficialTruthValueSurface,
    JointOfficialValueDataset,
    Phase6JointOfficialValueEvidence,
    PublicRootAlignedValueDataset,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_joint_official_values"
SOURCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6_aligned_counterfactual_values"
    / "public_root_aligned_value_dataset.json"
)


class JointOfficialValueArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PublicRootAlignedValueDataset.from_dict(
            json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        )
        cls.dataset_raw = json.loads(
            (ARTIFACT_DIRECTORY / "joint_official_value_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        cls.evidence_raw = json.loads(
            (ARTIFACT_DIRECTORY / "phase6ac_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        cls.dataset = JointOfficialValueDataset.from_dict(cls.dataset_raw)
        cls.evidence = Phase6JointOfficialValueEvidence.from_dict(
            cls.evidence_raw
        )

    def test_joint_roots_cover_every_split_candidate_and_bind_prefix(self) -> None:
        dataset = self.dataset
        self.assertEqual(
            dataset.source_phase6z_dataset_fingerprint,
            self.source.fingerprint,
        )
        self.assertEqual(dataset.runtime_fingerprint, self.source.runtime_fingerprint)
        self.assertEqual(dataset.deck_fingerprint, self.source.deck_fingerprint)
        self.assertEqual(len(dataset.surfaces), 18)
        self.assertEqual(
            {(item.split, item.candidate_id) for item in dataset.surfaces},
            {
                (split, candidate)
                for split in ("train", "calibration", "validation")
                for candidate in dataset.candidate_ids
            },
        )
        for surface in dataset.surfaces:
            self.assertEqual(
                surface.prefix_trace.root_public_history_fingerprint,
                surface.public_history_fingerprint,
            )
            self.assertTrue(surface.joint_prefix_truth_value_root)
            self.assertTrue(surface.realized_hidden_state_official_sample)
            self.assertTrue(surface.descendant_transitions_official_engine_executed)

    def test_truth_reinitialization_and_next_live_step_match_exactly(self) -> None:
        for surface in self.dataset.surfaces:
            self.assertTrue(surface.visualizer_root_public_parity)
            self.assertTrue(surface.visualizer_root_acting_private_zone_parity)
            self.assertTrue(surface.native_root_public_parity)
            self.assertTrue(surface.native_root_private_parity)
            self.assertTrue(surface.native_live_next_public_parity)
            self.assertTrue(surface.native_live_next_private_parity)
            self.assertGreaterEqual(surface.live_replay_coin_count, 0)
            self.assertFalse(surface.hidden_truth_selected_for_policy_input)
            self.assertEqual(surface.raw_private_card_ids_persisted, 0)
        self.assertTrue(self.dataset.live_native_replay_fidelity_gate_passed)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)
        serialized = json.dumps(self.dataset_raw, sort_keys=True)
        for forbidden in (
            '"your_deck"',
            '"your_prize"',
            '"opponent_deck"',
            '"opponent_prize"',
            '"opponent_hand"',
            '"search_begin_input"',
            '"observation_json"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_all_five_plan_values_have_repeated_terminal_rollouts(self) -> None:
        dataset = self.dataset
        self.assertEqual(dataset.action_target_count, 90)
        self.assertEqual(dataset.terminal_rollout_count, 270)
        self.assertGreater(dataset.native_transition_count, 0)
        self.assertGreater(dataset.macro_transition_count, 0)
        for surface in dataset.surfaces:
            self.assertEqual(len(surface.action_values), 5)
            self.assertAlmostEqual(
                sum(value.blueprint_probability for value in surface.action_values),
                1.0,
            )
            for value in surface.action_values:
                self.assertEqual(len(value.terminal_returns), 3)
                self.assertAlmostEqual(
                    value.empirical_opponent_value,
                    statistics.mean(value.terminal_returns),
                )
                self.assertAlmostEqual(
                    value.rollout_standard_error,
                    statistics.stdev(value.terminal_returns) / math.sqrt(3),
                )

    def test_common_chance_ratio_is_known_while_absolute_reach_and_safety_are_not(self) -> None:
        dataset = self.dataset
        evidence = self.evidence
        self.assertTrue(dataset.common_chance_likelihood_ratio_available)
        for surface in dataset.surfaces:
            ratio = surface.common_chance_ratio
            self.assertTrue(ratio.target_and_sampling_chance_law_identical)
            self.assertEqual(ratio.counterfactual_likelihood_ratio, 1.0)
            self.assertFalse(ratio.absolute_chance_reach_available)
            self.assertFalse(surface.absolute_chance_reach_available)
            self.assertFalse(surface.calibrated_counterfactual_value)
        self.assertFalse(dataset.repeated_truth_samples_per_information_set)
        self.assertFalse(dataset.calibrated_counterfactual_values)
        self.assertFalse(dataset.safe_resolver_compatible)
        self.assertFalse(dataset.opponent_exploitation_compatible)
        self.assertFalse(dataset.official_strategy_control_enabled)
        self.assertFalse(evidence.absolute_chance_reach_available)
        self.assertFalse(evidence.calibrated_counterfactual_values)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.opponent_exploitation_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)

    def test_fail_closed_claims_and_json_shapes(self) -> None:
        surface = self.dataset.surfaces[0]
        with self.assertRaises(ValueError):
            replace(surface, hidden_truth_selected_for_policy_input=True)
        with self.assertRaises(ValueError):
            replace(surface, absolute_chance_reach_available=True)
        with self.assertRaises(ValueError):
            replace(surface, calibrated_counterfactual_value=True)
        with self.assertRaises(ValueError):
            replace(self.dataset, safe_resolver_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)
        self.assertEqual(self.dataset.to_dict(), self.dataset_raw)
        self.assertEqual(self.evidence.to_dict(), self.evidence_raw)
        self.assertEqual(
            JointOfficialTruthValueSurface.from_dict(surface.to_dict()),
            surface,
        )


if __name__ == "__main__":
    unittest.main()
