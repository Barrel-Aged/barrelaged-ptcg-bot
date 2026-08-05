from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    BranchConditionedDataset,
    BranchHybridCalibration,
    BranchHybridEvaluation,
    BranchHybridModel,
    BranchHybridValidationDataset,
    Phase6BranchHybridEvidence,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "phase6_semiparametric_branch_hybrid"
)
SOURCE_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_branch_conditioned_values"


class SemiparametricBranchHybridArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(directory: Path, name: str) -> dict:
            return json.loads((directory / name).read_text(encoding="utf-8"))

        cls.dataset_raw = raw(
            ARTIFACT_DIRECTORY,
            "branch_hybrid_validation_dataset.json",
        )
        cls.model_raw = raw(ARTIFACT_DIRECTORY, "branch_hybrid_model.json")
        cls.calibration_raw = raw(
            ARTIFACT_DIRECTORY,
            "branch_hybrid_calibration.json",
        )
        cls.evaluation_raw = raw(
            ARTIFACT_DIRECTORY,
            "branch_hybrid_evaluation.json",
        )
        cls.evidence_raw = raw(ARTIFACT_DIRECTORY, "phase6aj_evidence.json")
        cls.dataset = BranchHybridValidationDataset.from_dict(cls.dataset_raw)
        cls.model = BranchHybridModel.from_dict(cls.model_raw)
        cls.calibration = BranchHybridCalibration.from_dict(cls.calibration_raw)
        cls.evaluation = BranchHybridEvaluation.from_dict(cls.evaluation_raw)
        cls.evidence = Phase6BranchHybridEvidence.from_dict(cls.evidence_raw)
        cls.source = BranchConditionedDataset.from_dict(
            raw(SOURCE_DIRECTORY, "branch_conditioned_dataset.json")
        )

    def test_fresh_panel_is_complete_disjoint_and_officially_executed(self) -> None:
        dataset = self.dataset
        validation_ids = {item.root_id for item in dataset.roots}
        self.assertEqual(len(validation_ids), 36)
        self.assertFalse(validation_ids & set(dataset.phase6ah_design_root_ids))
        self.assertFalse(validation_ids & set(dataset.phase6ai_source_root_ids))
        self.assertEqual(dataset.terminal_rollout_count, 2_880)
        self.assertEqual(sum(len(item.branch_observations) for item in dataset.roots), 360)
        self.assertEqual(sum(len(item.value_root.surfaces) for item in dataset.roots), 72)
        self.assertEqual(
            {
                (
                    item.value_root.candidate_id_hypothesis_input,
                    item.value_root.intent_id_supervision_only,
                )
                for item in dataset.roots
            },
            {
                (candidate, intent)
                for candidate in dataset.candidate_ids
                for intent in dataset.intent_ids
            },
        )
        for root in dataset.roots:
            self.assertEqual(root.value_root.policy_variant_id_supervision_only, "action_first")
            self.assertTrue(
                all(
                    observation.official_native_transition_executed
                    for observation in root.branch_observations
                )
            )

    def test_model_and_bounds_were_frozen_without_validation_targets(self) -> None:
        self.assertEqual(self.model.source_dataset_fingerprint, self.source.fingerprint)
        self.assertEqual(len(self.model.selection_root_ids), 36)
        self.assertEqual(len(self.model.selection_metrics), 16)
        self.assertEqual(self.model.selected_mean_blend, 0.5)
        self.assertEqual(self.model.selected_advantage_scale, 1.0)
        self.assertFalse(self.model.learned_action_advantage_used)
        self.assertTrue(self.model.causal_branch_heuristic_is_complete_action_advantage)
        self.assertFalse(self.model.validation_targets_used_for_selection)
        self.assertEqual(len(self.calibration.calibration_root_ids), 36)
        self.assertFalse(self.calibration.validation_targets_used_for_calibration)
        self.assertFalse(
            set(self.evaluation.validation_root_ids) & set(self.model.selection_root_ids)
        )
        self.assertFalse(
            set(self.evaluation.validation_root_ids)
            & set(self.calibration.calibration_root_ids)
        )

    def test_artifacts_roundtrip_bind_every_stage_and_leak_no_private_zones(self) -> None:
        for artifact, raw in (
            (self.dataset, self.dataset_raw),
            (self.model, self.model_raw),
            (self.calibration, self.calibration_raw),
            (self.evaluation, self.evaluation_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        self.assertEqual(
            self.evaluation.validation_dataset_fingerprint,
            self.dataset.fingerprint,
        )
        self.assertEqual(self.evaluation.source_model_fingerprint, self.model.fingerprint)
        self.assertEqual(
            self.evaluation.source_calibration_fingerprint,
            self.calibration.fingerprint,
        )
        self.assertEqual(self.evidence.evaluation_fingerprint, self.evaluation.fingerprint)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)

    def test_preregistered_result_validates_point_signal_but_rejects_intervals(self) -> None:
        evaluation = self.evaluation
        self.assertTrue(evaluation.sampling_precision_gate_passed)
        self.assertTrue(evaluation.point_mse_gate_passed)
        self.assertTrue(evaluation.contrast_relative_gate_passed)
        self.assertTrue(evaluation.contrast_bootstrap_gate_passed)
        self.assertTrue(evaluation.implementation_holdout_gate_passed)
        self.assertTrue(evaluation.action_sensitive_point_model_validated)
        self.assertFalse(evaluation.interval_gate_passed)
        self.assertFalse(evaluation.conditional_expected_mean_model_admitted)
        self.assertGreater(evaluation.validation_mean_width, 1.75)
        self.assertFalse(evaluation.counterfactual_value_compatible)
        self.assertFalse(evaluation.safe_resolver_enabled)
        self.assertFalse(evaluation.opponent_exploitation_enabled)
        self.assertFalse(evaluation.official_strategy_control_enabled)

    def test_capability_and_provenance_tampering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.dataset, validation_targets_used_for_selection=True)
        with self.assertRaises(ValueError):
            replace(self.model, learned_action_advantage_used=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, safe_resolver_enabled=True)
        with self.assertRaises(ValueError):
            replace(self.evaluation, counterfactual_value_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)


if __name__ == "__main__":
    unittest.main()
