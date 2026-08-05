from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    PHASE6_BRANCH_DELTA_NAMES,
    BranchConditionedDataset,
    BranchValueCalibration,
    BranchValueModel,
    Phase6BranchValueEvidence,
    ExpectedValueDataset,
    build_branch_value_examples,
    restore_branch_value_model,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_branch_conditioned_values"
DESIGN_DATASET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "phase6_expected_value_surfaces"
    / "expected_value_surface_dataset.json"
)


class BranchConditionedValueArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads(
                (ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8")
            )

        cls.dataset_raw = raw("branch_conditioned_dataset.json")
        cls.model_raw = raw("branch_value_model.json")
        cls.calibration_raw = raw("branch_value_calibration.json")
        cls.evidence_raw = raw("phase6ai_evidence.json")
        cls.dataset = BranchConditionedDataset.from_dict(cls.dataset_raw)
        cls.model = BranchValueModel.from_dict(cls.model_raw)
        cls.calibration = BranchValueCalibration.from_dict(cls.calibration_raw)
        cls.evidence = Phase6BranchValueEvidence.from_dict(cls.evidence_raw)
        cls.examples = build_branch_value_examples(cls.dataset)
        cls.design = ExpectedValueDataset.from_dict(
            json.loads(DESIGN_DATASET_PATH.read_text(encoding="utf-8"))
        )

    def test_panel_is_fresh_complete_and_branch_aligned(self) -> None:
        dataset = self.dataset
        self.assertEqual(dataset.source_phase6ah_design_dataset_fingerprint, self.design.fingerprint)
        self.assertEqual(set(dataset.design_root_ids), {item.root_id for item in self.design.roots})
        self.assertFalse(
            set(dataset.design_root_ids) & {item.root_id for item in dataset.roots}
        )
        self.assertEqual(len(dataset.roots), 180)
        self.assertEqual(dataset.phase6ah_roots_reused, 0)
        self.assertEqual(dataset.terminal_rollout_count, 14_400)
        self.assertEqual(sum(len(item.branch_observations) for item in dataset.roots), 1_800)
        self.assertEqual(len(self.examples), 180)
        for root, example in zip(dataset.roots, self.examples, strict=True):
            self.assertEqual(root.root_id, example.root_id)
            self.assertTrue(root.match_id.startswith("phase6ai-"))
            self.assertEqual(len(root.branch_observations), 10)
            self.assertEqual(len(example.actions), 5)
            for observation in root.branch_observations:
                self.assertEqual(
                    tuple(name for name, _ in observation.public_deltas),
                    PHASE6_BRANCH_DELTA_NAMES,
                )
                self.assertTrue(observation.official_native_transition_executed)
                self.assertFalse(observation.hidden_completion_truth_selected_as_feature)
                self.assertEqual(observation.post_branch_private_features_persisted, 0)

    def test_model_features_are_grouped_causal_and_leakage_safe(self) -> None:
        root_names = self.model.root_normalizer.names
        branch_names = self.model.branch_normalizer.names
        self.assertEqual(len(root_names), 208)
        self.assertEqual(len(branch_names), 45)
        self.assertTrue(any(name.startswith("public::") for name in root_names))
        self.assertTrue(any(name.startswith("private::") for name in root_names))
        self.assertTrue(any(name.startswith("history::") for name in root_names))
        self.assertTrue(any(name.startswith("candidate_hypothesis::") for name in root_names))
        self.assertFalse(any(name.startswith("plan_action::") for name in root_names))
        self.assertTrue(any(name.startswith("public_delta_mean::") for name in branch_names))
        self.assertTrue(any(name.startswith("public_delta_range::") for name in branch_names))
        self.assertEqual(sum(name.startswith("plan_action::") for name in branch_names), 5)
        self.assertIn("immediate_terminal_rate", branch_names)
        self.assertFalse(
            any(
                token in name
                for name in root_names + branch_names
                for token in (
                    "intent_id",
                    "policy_variant",
                    "supervision_only",
                    "terminal_return",
                    "rollout_depth",
                    "hidden_completion",
                )
            )
        )
        self.assertTrue(self.model.grouped_five_action_objective)
        self.assertTrue(self.model.public_branch_features_only)
        self.assertFalse(self.model.audit_labels_as_features)
        self.assertFalse(self.model.terminal_rollout_results_as_features)
        self.assertIsNotNone(restore_branch_value_model(self.model).network)

    def test_artifacts_roundtrip_bind_sources_and_persist_no_private_zones(self) -> None:
        for artifact, raw in (
            (self.dataset, self.dataset_raw),
            (self.model, self.model_raw),
            (self.calibration, self.calibration_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        self.assertEqual(self.model.source_dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(
            self.calibration.source_model_fingerprint,
            self.model.fingerprint,
        )
        self.assertEqual(
            self.evidence.calibration_fingerprint,
            self.calibration.fingerprint,
        )
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

    def test_calibration_is_fresh_implementation_heldout_and_fail_closed(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertTrue(calibration.implementation_holdout_gate_passed)
        self.assertEqual(
            evidence.action_sensitive_point_model_validated,
            calibration.action_sensitive_point_model_validated,
        )
        self.assertEqual(
            evidence.conditional_expected_mean_model_admitted,
            calibration.conditional_expected_mean_model_admitted,
        )
        self.assertEqual(
            {item.dimension for item in calibration.slice_metrics},
            {"action", "candidate", "intent"},
        )
        self.assertFalse(calibration.counterfactual_value_compatible)
        self.assertFalse(calibration.equilibrium_convergence_validated)
        self.assertFalse(calibration.safe_resolver_enabled)
        self.assertFalse(calibration.opponent_exploitation_enabled)
        self.assertFalse(calibration.official_strategy_control_enabled)
        self.assertFalse(evidence.counterfactual_value_compatible)
        self.assertFalse(evidence.equilibrium_convergence_validated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.opponent_exploitation_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)

    def test_capability_and_provenance_tampering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.dataset, phase6ah_roots_reused=1)
        with self.assertRaises(ValueError):
            replace(self.model, official_engine_executed=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, counterfactual_value_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)


if __name__ == "__main__":
    unittest.main()
