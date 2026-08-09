from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from rocketbot import (
    CounterfactualReachExpectationAudit,
    NativeCounterfactualReachAdmission,
    NativeCounterfactualReachDataset,
    OpponentDeviationCalibration,
    OpponentDeviationDataset,
    Phase6NativeCounterfactualReachEvidence,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = (
    PROJECT_ROOT / "artifacts" / "phase6_native_counterfactual_reach"
)


class NativeCounterfactualReachArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads(
                (ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8")
            )

        cls.source_raw = raw("source_opponent_deviation_dataset.json")
        cls.audit_raw = raw("exact_reach_audit.json")
        cls.source_calibration_raw = raw(
            "source_opponent_deviation_calibration.json"
        )
        cls.dataset_raw = raw("native_counterfactual_reach_dataset.json")
        cls.admission_raw = raw("native_counterfactual_reach_admission.json")
        cls.evidence_raw = raw("phase6y_evidence.json")
        cls.source = OpponentDeviationDataset.from_dict(cls.source_raw)
        cls.source_calibration = OpponentDeviationCalibration.from_dict(
            cls.source_calibration_raw
        )
        cls.audit = CounterfactualReachExpectationAudit.from_dict(cls.audit_raw)
        cls.dataset = NativeCounterfactualReachDataset.from_dict(cls.dataset_raw)
        cls.admission = NativeCounterfactualReachAdmission.from_dict(cls.admission_raw)
        cls.evidence = Phase6NativeCounterfactualReachEvidence.from_dict(
            cls.evidence_raw
        )

    def test_exact_reach_audit_is_exhaustive_and_abstract_only(self) -> None:
        self.assertEqual(len(self.audit.rows), 24)
        self.assertEqual(self.audit.maximum_absolute_error, 0.0)
        self.assertEqual(self.audit.maximum_importance_weight, 2.0)
        self.assertTrue(self.audit.exact_expectation_passed)
        self.assertFalse(self.audit.official_engine_executed)
        self.assertFalse(self.audit.full_game_native_counterfactual_values_calibrated)

    def test_native_ledger_links_every_source_surface_and_action(self) -> None:
        dataset = self.dataset
        self.assertEqual(
            dataset.source_opponent_deviation_dataset_fingerprint,
            self.source.fingerprint,
        )
        self.assertEqual(
            dataset.source_opponent_deviation_calibration_fingerprint,
            self.source_calibration.fingerprint,
        )
        self.assertEqual(
            self.source_calibration.dataset_fingerprint,
            self.source.fingerprint,
        )
        self.assertEqual(dataset.exact_reach_audit_fingerprint, self.audit.fingerprint)
        self.assertEqual(len(dataset.surfaces), 12)
        self.assertEqual(dataset.action_target_count, 60)
        self.assertEqual(dataset.terminal_rollout_count, 600)
        self.assertEqual(dataset.native_transition_count, 70_741)
        self.assertEqual(dataset.unique_public_history_count, 12)
        self.assertEqual(dataset.minimum_importance_weight, 1.0)
        self.assertEqual(dataset.maximum_importance_weight, 1.0)
        self.assertEqual(_raw_private_key_count(self.dataset_raw), 0)
        source_by_fingerprint = {
            surface.fingerprint: surface for surface in self.source.surfaces
        }
        self.assertEqual(
            set(source_by_fingerprint),
            {surface.source_surface_fingerprint for surface in dataset.surfaces},
        )
        for surface in dataset.surfaces:
            source = source_by_fingerprint[surface.source_surface_fingerprint]
            self.assertEqual(surface.source_surface_id, source.surface_id)
            self.assertEqual(surface.source_state_id, source.source_state_id)
            self.assertEqual(surface.reach_trace.steps, ())
            self.assertEqual(surface.reach_trace.player_1_reach, 1.0)
            self.assertEqual(surface.reach_trace.importance_weight, 1.0)
            self.assertTrue(
                surface.source_descendant_official_engine_executed
            )
            for reach_value, source_value in zip(
                surface.action_values,
                source.action_values,
                strict=True,
            ):
                self.assertEqual(reach_value.action_id, source_value.action_id)
                self.assertAlmostEqual(
                    reach_value.empirical_conditional_value,
                    source_value.empirical_opponent_value,
                )
                self.assertTrue(
                    math.isclose(
                        reach_value.weighted_counterfactual_numerator_contribution,
                        source_value.empirical_opponent_value,
                        abs_tol=1e-12,
                    )
                )

    def test_admission_and_evidence_keep_downstream_gates_closed(self) -> None:
        admission = self.admission
        evidence = self.evidence
        self.assertEqual(admission.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(evidence.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(evidence.admission_fingerprint, admission.fingerprint)
        self.assertTrue(admission.exact_reach_semantics_gate_passed)
        self.assertTrue(admission.native_reach_ledger_gate_passed)
        self.assertEqual(admission.maximum_histories_per_public_information_set, 1)
        self.assertFalse(admission.native_counterfactual_value_gate_passed)
        self.assertFalse(admission.safe_resolver_compatible)
        self.assertFalse(evidence.native_counterfactual_values_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        with self.assertRaises(ValueError):
            replace(self.dataset, native_counterfactual_values_calibrated=True)
        with self.assertRaises(ValueError):
            replace(admission, native_counterfactual_value_gate_passed=True)
        with self.assertRaises(ValueError):
            replace(
                self.dataset.surfaces[0],
                reach_trace=replace(
                    self.dataset.surfaces[0].reach_trace,
                    upstream_official_reach_complete=True,
                ),
            )

    def test_every_artifact_json_shape_roundtrips_exactly(self) -> None:
        for artifact, raw in (
            (self.source, self.source_raw),
            (self.source_calibration, self.source_calibration_raw),
            (self.audit, self.audit_raw),
            (self.dataset, self.dataset_raw),
            (self.admission, self.admission_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)


if __name__ == "__main__":
    unittest.main()
