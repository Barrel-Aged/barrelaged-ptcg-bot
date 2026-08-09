from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import unittest

from rocketbot import (
    HighPrecisionCalibration,
    HighPrecisionEvaluation,
    HighPrecisionModelReference,
    HighPrecisionPanel,
    PHASE6_HIGH_PRECISION_CALIBRATION_COVERAGE,
    PHASE6_HIGH_PRECISION_CALIBRATION_QUANTILE_RANK,
    PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION,
    PHASE6_HIGH_PRECISION_SOURCE_MODEL_FINGERPRINT,
    Phase6HighPrecisionEvidence,
    build_high_precision_examples,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "phase6_high_precision_confirmation"
)


class HighPrecisionConfirmationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads(
                (ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8")
            )

        cls.reference_raw = raw("high_precision_model_reference.json")
        cls.calibration_panel_raw = raw("calibration_panel.json")
        cls.calibration_raw = raw("high_precision_calibration.json")
        cls.validation_panel_raw = raw("validation_panel.json")
        cls.evaluation_raw = raw("high_precision_evaluation.json")
        cls.evidence_raw = raw("phase6al_evidence.json")
        cls.reference = HighPrecisionModelReference.from_dict(cls.reference_raw)
        cls.calibration_panel = HighPrecisionPanel.from_dict(
            cls.calibration_panel_raw
        )
        cls.calibration = HighPrecisionCalibration.from_dict(cls.calibration_raw)
        cls.validation_panel = HighPrecisionPanel.from_dict(
            cls.validation_panel_raw
        )
        cls.evaluation = HighPrecisionEvaluation.from_dict(cls.evaluation_raw)
        cls.evidence = Phase6HighPrecisionEvidence.from_dict(cls.evidence_raw)

    def test_reference_is_frozen_and_all_prior_roots_are_development(self) -> None:
        reference = self.reference
        self.assertEqual(
            reference.source_model_fingerprint,
            PHASE6_HIGH_PRECISION_SOURCE_MODEL_FINGERPRINT,
        )
        self.assertEqual(len(reference.development_root_ids), 288)
        self.assertEqual(len(set(reference.development_root_ids)), 288)
        self.assertEqual(reference.source_feature_count, 208)
        self.assertFalse(reference.source_model_retrained)
        self.assertTrue(reference.phase6ak_targets_declared_development)
        self.assertFalse(reference.intent_labels_as_model_input)
        self.assertFalse(reference.counterfactual_value_provider)

    def test_panels_are_complete_disjoint_and_sequentially_bound(self) -> None:
        calibration = self.calibration_panel
        validation = self.validation_panel
        self.assertEqual(calibration.role, "calibration")
        self.assertEqual(validation.role, "validation")
        calibration_ids = {item.root_id for item in calibration.roots}
        validation_ids = {item.root_id for item in validation.roots}
        calibration_batch_ids = {
            batch.root_id for item in calibration.roots for batch in item.batches
        }
        validation_batch_ids = {
            batch.root_id for item in validation.roots for batch in item.batches
        }
        self.assertEqual(len(calibration_ids), 36)
        self.assertEqual(len(validation_ids), 36)
        self.assertFalse(calibration_ids & validation_ids)
        self.assertFalse(calibration_batch_ids & validation_batch_ids)
        self.assertFalse(calibration_batch_ids & set(self.reference.development_root_ids))
        self.assertFalse(validation_batch_ids & set(self.reference.development_root_ids))
        for panel in (calibration, validation):
            self.assertEqual(panel.branch_observation_count, 720)
            self.assertEqual(panel.terminal_rollout_count, 5_760)
            self.assertEqual(
                {(item.candidate_id, item.intent_id) for item in panel.roots},
                {
                    (candidate, intent)
                    for candidate in panel.candidate_ids
                    for intent in panel.intent_ids
                },
            )
            for root in panel.roots:
                self.assertEqual(len(root.batches), 2)
                self.assertEqual(root.branch_observation_count, 20)
                self.assertEqual(root.terminal_rollout_count, 160)
                completions = {
                    surface.completion_fingerprint
                    for batch in root.batches
                    for surface in batch.value_root.surfaces
                }
                self.assertEqual(len(completions), 4)
        self.assertEqual(
            self.calibration.calibration_panel_fingerprint,
            calibration.fingerprint,
        )
        self.assertEqual(
            self.evaluation.validation_panel_fingerprint,
            validation.fingerprint,
        )

    def test_every_action_aggregates_32_returns_with_exact_standard_error(self) -> None:
        for panel in (self.calibration_panel, self.validation_panel):
            examples = build_high_precision_examples(panel.roots, panel.candidate_ids)
            self.assertEqual(len(examples), 36)
            for example in examples:
                self.assertEqual(len(example.root_features), 208)
                self.assertEqual(len(example.actions), 5)
                for action in example.actions:
                    self.assertEqual(
                        len(action.terminal_returns),
                        PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION,
                    )
                    expected = statistics.stdev(action.terminal_returns) / math.sqrt(
                        PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION
                    )
                    self.assertAlmostEqual(action.target_standard_error, expected)

    def test_artifacts_roundtrip_bind_every_stage_and_leak_no_private_zones(self) -> None:
        for artifact, raw in (
            (self.reference, self.reference_raw),
            (self.calibration_panel, self.calibration_panel_raw),
            (self.calibration, self.calibration_raw),
            (self.validation_panel, self.validation_panel_raw),
            (self.evaluation, self.evaluation_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        self.assertEqual(
            self.calibration.calibration_coverage_target,
            PHASE6_HIGH_PRECISION_CALIBRATION_COVERAGE,
        )
        self.assertEqual(
            self.calibration.calibration_quantile_rank,
            PHASE6_HIGH_PRECISION_CALIBRATION_QUANTILE_RANK,
        )
        self.assertEqual(
            self.evaluation.source_reference_fingerprint,
            self.reference.fingerprint,
        )
        self.assertEqual(
            self.evaluation.source_calibration_fingerprint,
            self.calibration.fingerprint,
        )
        self.assertEqual(self.evidence.evaluation_fingerprint, self.evaluation.fingerprint)
        self.assertEqual(_raw_private_key_count(self.calibration_panel_raw), 0)
        self.assertEqual(_raw_private_key_count(self.validation_panel_raw), 0)

    def test_gate_flags_are_consistent_and_downstream_remains_fail_closed(self) -> None:
        evaluation = self.evaluation
        self.assertTrue(evaluation.sampling_precision_gate_passed)
        self.assertTrue(evaluation.collection_holdout_gate_passed)
        self.assertTrue(evaluation.point_mse_gate_passed)
        self.assertTrue(evaluation.contrast_relative_gate_passed)
        self.assertTrue(evaluation.contrast_retention_gate_passed)
        self.assertFalse(evaluation.contrast_bootstrap_gate_passed)
        self.assertFalse(evaluation.action_sensitive_point_model_validated)
        self.assertFalse(evaluation.interval_gate_passed)
        self.assertFalse(evaluation.conditional_expected_mean_model_admitted)
        self.assertAlmostEqual(evaluation.validation_coverage, 8.0 / 9.0)
        self.assertAlmostEqual(
            evaluation.minimum_validation_slice_coverage,
            2.0 / 3.0,
        )
        self.assertLessEqual(evaluation.maximum_target_standard_error, 0.20)
        self.assertLess(evaluation.validation_point_metric.mse, 0.45)
        self.assertLess(
            evaluation.validation_point_metric.mse,
            evaluation.validation_point_metric.zero_mse,
        )
        self.assertLess(
            evaluation.validation_point_metric.mse,
            evaluation.validation_point_metric.root_heuristic_mse,
        )
        self.assertLess(
            evaluation.validation_point_metric.mse,
            evaluation.validation_point_metric.branch_heuristic_mse,
        )
        self.assertLess(
            evaluation.validation_point_metric.mse,
            evaluation.phase6aj_baseline_validation_mse,
        )
        self.assertLess(evaluation.zero_contrast_bootstrap_lower_bound, 0.0)
        self.assertEqual(
            evaluation.conditional_expected_mean_model_admitted,
            evaluation.action_sensitive_point_model_validated
            and evaluation.interval_gate_passed,
        )
        self.assertFalse(evaluation.counterfactual_value_compatible)
        self.assertFalse(evaluation.equilibrium_convergence_validated)
        self.assertFalse(evaluation.safe_resolver_enabled)
        self.assertFalse(evaluation.opponent_exploitation_enabled)
        self.assertFalse(evaluation.official_strategy_control_enabled)
        self.assertEqual(
            self.evidence.status,
            "high_precision_confirmation_rejected_downstream_fail_closed",
        )

    def test_capability_and_provenance_tampering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.reference, source_model_retrained=True)
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
