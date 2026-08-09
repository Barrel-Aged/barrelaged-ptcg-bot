from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    OfficialPrefixDecisionStep,
    OfficialPrefixReachDataset,
    OfficialPrefixReachTrace,
    Phase6OfficialPrefixReachEvidence,
    PublicRootAlignedValueDataset,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_official_prefix_reach"
SOURCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6_aligned_counterfactual_values"
    / "public_root_aligned_value_dataset.json"
)


class OfficialPrefixReachArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = PublicRootAlignedValueDataset.from_dict(
            json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
        )
        cls.dataset_raw = json.loads(
            (ARTIFACT_DIRECTORY / "official_prefix_reach_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        cls.evidence_raw = json.loads(
            (ARTIFACT_DIRECTORY / "phase6ab_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        cls.dataset = OfficialPrefixReachDataset.from_dict(cls.dataset_raw)
        cls.evidence = Phase6OfficialPrefixReachEvidence.from_dict(
            cls.evidence_raw
        )

    def test_dataset_has_full_split_candidate_support_and_genuine_steps(self) -> None:
        dataset = self.dataset
        self.assertEqual(
            dataset.source_phase6z_dataset_fingerprint,
            self.source.fingerprint,
        )
        self.assertEqual(dataset.runtime_fingerprint, self.source.runtime_fingerprint)
        self.assertEqual(dataset.deck_fingerprint, self.source.deck_fingerprint)
        self.assertEqual(len(dataset.traces), 18)
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertEqual(
            {(trace.split, trace.candidate_id) for trace in dataset.traces},
            {
                (split, candidate)
                for split in ("train", "calibration", "validation")
                for candidate in dataset.candidate_ids
            },
        )
        self.assertGreater(dataset.official_selection_execution_count, 0)
        self.assertGreater(dataset.player_0_decisions, 0)
        self.assertGreater(dataset.player_1_decisions, 0)
        for trace in dataset.traces:
            self.assertTrue(trace.official_battle_root_executed)
            self.assertTrue(trace.official_action_prefix_complete)
            self.assertGreater(trace.player_0_decisions, 0)
            self.assertGreater(trace.player_1_decisions, 0)
            self.assertTrue(
                all(step.official_battle_selection_executed for step in trace.steps)
            )

    def test_public_chains_are_contiguous_and_identifiers_are_redacted(self) -> None:
        for trace in self.dataset.traces:
            self.assertEqual(
                tuple(step.step_index for step in trace.steps),
                tuple(range(len(trace.steps))),
            )
            for left, right in zip(trace.steps, trace.steps[1:]):
                self.assertEqual(
                    left.public_state_after_fingerprint,
                    right.public_state_before_fingerprint,
                )
                self.assertEqual(
                    left.public_history_after_fingerprint,
                    right.public_history_before_fingerprint,
                )
            self.assertEqual(
                trace.steps[-1].public_state_after_fingerprint,
                trace.root_public_state_fingerprint,
            )
            self.assertEqual(
                trace.steps[-1].public_history_after_fingerprint,
                trace.root_public_history_fingerprint,
            )
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)
        serialized = json.dumps(self.dataset_raw, sort_keys=True)
        self.assertNotIn('"selected_indices"', serialized)
        self.assertNotIn('"option"', serialized)
        self.assertNotIn('"hand"', serialized)

    def test_behavior_reach_is_exact_but_chance_and_counterfactual_reach_are_null(self) -> None:
        dataset = self.dataset
        evidence = self.evidence
        self.assertTrue(dataset.behavior_action_ledger_complete)
        self.assertTrue(dataset.declared_deterministic_behavior_reach_available)
        self.assertEqual(dataset.official_chance_probability_coverage, 0.0)
        for trace in dataset.traces:
            self.assertEqual(trace.declared_player_0_behavior_reach, 1.0)
            self.assertEqual(trace.declared_player_1_behavior_reach, 1.0)
            self.assertEqual(trace.declared_player_behavior_sampling_reach, 1.0)
            self.assertIsNone(trace.official_chance_reach)
            self.assertIsNone(trace.full_sampling_reach)
            self.assertIsNone(trace.counterfactual_importance_weight)
            self.assertFalse(trace.official_chance_probability_available)
            self.assertFalse(trace.upstream_official_reach_complete)
            self.assertFalse(trace.aligned_with_phase6z_value_root)
            for step in trace.steps:
                self.assertEqual(step.declared_behavior_probability, 1.0)
                self.assertEqual(step.sampling_probability, 1.0)
                self.assertFalse(step.official_chance_probability_available)
        self.assertTrue(evidence.behavior_action_ledger_gate_passed)
        self.assertTrue(evidence.deterministic_behavior_reach_gate_passed)
        self.assertFalse(evidence.official_chance_reach_available)
        self.assertFalse(evidence.full_upstream_reach_available)
        self.assertFalse(evidence.official_counterfactual_importance_weights_available)
        self.assertFalse(evidence.phase6z_value_root_alignment_available)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.opponent_exploitation_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)

    def test_fail_closed_dataclasses_reject_promoted_claims(self) -> None:
        trace = self.dataset.traces[0]
        step = trace.steps[0]
        with self.assertRaises(ValueError):
            replace(step, official_chance_probability_available=True)
        with self.assertRaises(ValueError):
            replace(trace, official_chance_reach=1.0)
        with self.assertRaises(ValueError):
            replace(trace, counterfactual_importance_weight=1.0)
        with self.assertRaises(ValueError):
            replace(self.dataset, safe_resolver_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)

    def test_artifact_json_shapes_roundtrip_exactly(self) -> None:
        self.assertEqual(self.dataset.to_dict(), self.dataset_raw)
        self.assertEqual(self.evidence.to_dict(), self.evidence_raw)
        self.assertEqual(
            OfficialPrefixDecisionStep.from_dict(
                self.dataset.traces[0].steps[0].to_dict()
            ),
            self.dataset.traces[0].steps[0],
        )
        self.assertEqual(
            OfficialPrefixReachTrace.from_dict(
                self.dataset.traces[0].to_dict()
            ),
            self.dataset.traces[0],
        )


if __name__ == "__main__":
    unittest.main()
