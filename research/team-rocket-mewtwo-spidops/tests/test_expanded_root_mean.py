from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    ExpandedRootMeanCalibration,
    ExpandedRootMeanEvaluation,
    ExpandedRootMeanModel,
    ExpandedRootMeanPanel,
    PHASE6_EXPANDED_ROOT_MEAN_FOLDS,
    PHASE6_EXPANDED_ROOT_MEAN_FORMS,
    PHASE6_EXPANDED_ROOT_MEAN_PENALTY_PER_ROOT_GRID,
    Phase6ExpandedRootMeanEvidence,
    restore_expanded_root_mean_model,
)
from rocketbot.phase6_branch_conditioned_values import (
    build_branch_value_examples_from_roots,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_expanded_root_mean"


class ExpandedRootMeanArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads(
                (ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8")
            )

        cls.model_raw = raw("expanded_root_mean_model.json")
        cls.calibration_panel_raw = raw("calibration_panel.json")
        cls.calibration_raw = raw("expanded_root_mean_calibration.json")
        cls.validation_panel_raw = raw("validation_panel.json")
        cls.evaluation_raw = raw("expanded_root_mean_evaluation.json")
        cls.evidence_raw = raw("phase6ak_evidence.json")
        cls.model = ExpandedRootMeanModel.from_dict(cls.model_raw)
        cls.calibration_panel = ExpandedRootMeanPanel.from_dict(
            cls.calibration_panel_raw
        )
        cls.calibration = ExpandedRootMeanCalibration.from_dict(cls.calibration_raw)
        cls.validation_panel = ExpandedRootMeanPanel.from_dict(
            cls.validation_panel_raw
        )
        cls.evaluation = ExpandedRootMeanEvaluation.from_dict(cls.evaluation_raw)
        cls.evidence = Phase6ExpandedRootMeanEvidence.from_dict(cls.evidence_raw)

    def test_development_ledger_and_ridge_selection_are_frozen(self) -> None:
        model = self.model
        self.assertEqual(len(model.design_root_ids), 216)
        self.assertEqual(len(set(model.design_root_ids)), 216)
        self.assertEqual(model.selected_model_form, "root_heuristic_residual")
        self.assertEqual(model.selected_penalty_per_root, 5.0)
        self.assertEqual(
            {
                (metric.model_form, metric.penalty_per_root)
                for metric in model.selection_metrics
            },
            {
                (form, penalty)
                for form in PHASE6_EXPANDED_ROOT_MEAN_FORMS
                for penalty in PHASE6_EXPANDED_ROOT_MEAN_PENALTY_PER_ROOT_GRID
            },
        )
        self.assertEqual(len(model.selection_metrics), 12)
        expected_heldout = {
            "action_first": 36,
            "phase6aj_action_first": 36,
            "canonical": 36,
            "engine_first": 36,
            "hand_first": 72,
        }
        for metric in model.selection_metrics:
            self.assertEqual(
                tuple(fold.fold_id for fold in metric.fold_metrics),
                PHASE6_EXPANDED_ROOT_MEAN_FOLDS,
            )
            for fold in metric.fold_metrics:
                self.assertEqual(fold.heldout_root_count, expected_heldout[fold.fold_id])
                self.assertEqual(fold.fit_root_count, 216 - fold.heldout_root_count)
        self.assertTrue(model.all_prior_targets_are_development_data)
        self.assertFalse(model.intent_labels_as_features)
        self.assertFalse(model.policy_labels_as_features)
        self.assertFalse(model.learned_action_advantage_used)

    def test_fresh_panels_are_complete_mutually_disjoint_and_officially_executed(self) -> None:
        calibration = self.calibration_panel
        validation = self.validation_panel
        self.assertEqual(calibration.role, "calibration")
        self.assertEqual(validation.role, "validation")
        calibration_ids = {item.root_id for item in calibration.roots}
        validation_ids = {item.root_id for item in validation.roots}
        design_ids = set(self.model.design_root_ids)
        self.assertEqual(len(calibration_ids), 36)
        self.assertEqual(len(validation_ids), 36)
        self.assertFalse(calibration_ids & validation_ids)
        self.assertFalse(calibration_ids & design_ids)
        self.assertFalse(validation_ids & design_ids)
        for panel in (calibration, validation):
            self.assertEqual(panel.terminal_rollout_count, 2_880)
            self.assertEqual(
                sum(len(item.branch_observations) for item in panel.roots),
                360,
            )
            self.assertEqual(
                sum(len(item.value_root.surfaces) for item in panel.roots),
                72,
            )
            self.assertEqual(
                {
                    (
                        item.value_root.candidate_id_hypothesis_input,
                        item.value_root.intent_id_supervision_only,
                    )
                    for item in panel.roots
                },
                {
                    (candidate, intent)
                    for candidate in panel.candidate_ids
                    for intent in panel.intent_ids
                },
            )
            for root in panel.roots:
                self.assertEqual(
                    root.value_root.policy_variant_id_supervision_only,
                    "action_first",
                )
                self.assertTrue(
                    all(
                        observation.official_native_transition_executed
                        for observation in root.branch_observations
                    )
                )

    def test_artifacts_roundtrip_bind_every_stage_and_leak_no_private_zones(self) -> None:
        for artifact, raw in (
            (self.model, self.model_raw),
            (self.calibration_panel, self.calibration_panel_raw),
            (self.calibration, self.calibration_raw),
            (self.validation_panel, self.validation_panel_raw),
            (self.evaluation, self.evaluation_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        self.assertEqual(
            self.calibration_panel.source_model_fingerprint,
            self.model.fingerprint,
        )
        self.assertEqual(
            self.validation_panel.source_model_fingerprint,
            self.model.fingerprint,
        )
        self.assertEqual(
            self.calibration.calibration_panel_fingerprint,
            self.calibration_panel.fingerprint,
        )
        self.assertEqual(
            self.evaluation.validation_panel_fingerprint,
            self.validation_panel.fingerprint,
        )
        self.assertEqual(
            self.evaluation.source_calibration_fingerprint,
            self.calibration.fingerprint,
        )
        self.assertEqual(self.evidence.evaluation_fingerprint, self.evaluation.fingerprint)
        self.assertEqual(_raw_private_key_count(self.calibration_panel_raw), 0)
        self.assertEqual(_raw_private_key_count(self.validation_panel_raw), 0)

    def test_frozen_provider_reproduces_validation_metric(self) -> None:
        examples = build_branch_value_examples_from_roots(
            self.validation_panel.roots,
            self.validation_panel.candidate_ids,
            expected_count=36,
        )
        predictions = restore_expanded_root_mean_model(self.model).predict_examples(
            examples
        )
        squared_errors = [
            (prediction - target.empirical_player0_value) ** 2
            for root_predictions, example in zip(predictions, examples, strict=True)
            for prediction, target in zip(
                root_predictions,
                example.actions,
                strict=True,
            )
        ]
        self.assertAlmostEqual(
            sum(squared_errors) / len(squared_errors),
            self.evaluation.validation_point_metric.mse,
        )

    def test_result_validates_point_signal_but_rejects_slice_coverage(self) -> None:
        evaluation = self.evaluation
        metric = evaluation.validation_point_metric
        self.assertTrue(evaluation.sampling_precision_gate_passed)
        self.assertTrue(evaluation.point_mse_gate_passed)
        self.assertTrue(evaluation.contrast_relative_gate_passed)
        self.assertTrue(evaluation.contrast_retention_gate_passed)
        self.assertTrue(evaluation.contrast_bootstrap_gate_passed)
        self.assertTrue(evaluation.collection_holdout_gate_passed)
        self.assertTrue(evaluation.action_sensitive_point_model_validated)
        self.assertLess(metric.mse, metric.zero_mse)
        self.assertLess(metric.mse, metric.root_heuristic_mse)
        self.assertLess(metric.mse, metric.branch_heuristic_mse)
        self.assertLess(metric.mse, evaluation.phase6aj_baseline_validation_mse)
        self.assertGreaterEqual(evaluation.validation_coverage, 0.90)
        self.assertLessEqual(evaluation.validation_mean_width, 1.75)
        self.assertLess(evaluation.minimum_validation_slice_coverage, 0.75)
        self.assertFalse(evaluation.interval_gate_passed)
        self.assertFalse(evaluation.conditional_expected_mean_model_admitted)
        self.assertFalse(evaluation.counterfactual_value_compatible)
        self.assertFalse(evaluation.safe_resolver_enabled)
        self.assertFalse(evaluation.opponent_exploitation_enabled)
        self.assertFalse(evaluation.official_strategy_control_enabled)

    def test_capability_and_provenance_tampering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.model, official_engine_executed=True)
        with self.assertRaises(ValueError):
            replace(self.calibration_panel, prior_roots_reused=1)
        with self.assertRaises(ValueError):
            replace(self.calibration, validation_roots_collected_before_freeze=1)
        with self.assertRaises(ValueError):
            replace(self.evaluation, counterfactual_value_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)


if __name__ == "__main__":
    unittest.main()
