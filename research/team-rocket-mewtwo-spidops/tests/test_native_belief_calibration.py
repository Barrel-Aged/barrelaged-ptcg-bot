from __future__ import annotations

import json
from pathlib import Path
import unittest

from rocketbot import (
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeAllocationRangeDataset,
    NativeBeliefCalibrationArtifact,
    Phase6NativeBeliefCalibrationEvidence,
    run_phase6_native_belief_calibration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeBeliefCalibrationTests(unittest.TestCase):
    def test_phase6t_uses_heldout_allocations_and_public_only_range_inputs(self) -> None:
        artifacts = run_phase6_native_belief_calibration(
            project_root=PROJECT_ROOT,
            independent_roots=1,
            particle_variants=3,
            observations_per_particle=2,
            maximum_transitions_per_particle=96,
        )
        dataset = artifacts.dataset
        calibration = artifacts.calibration
        evidence = artifacts.evidence

        self.assertEqual(
            evidence.status,
            "phase6t_native_allocation_range_diagnostic_validated_partial",
        )
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertEqual(len(dataset.rows), 36)
        self.assertEqual(evidence.particle_count, 18)
        self.assertEqual(evidence.validation_rows, 12)
        self.assertEqual(evidence.candidate_strata, 6)
        self.assertTrue(dataset.allocation_heldout)
        self.assertFalse(dataset.selected_truth_in_model_input)
        self.assertEqual(dataset.raw_private_card_ids_persisted, 0)
        self.assertGreater(dataset.native_transition_count, 0)
        self.assertEqual(
            len({row.particle_fingerprint for row in dataset.rows}),
            18,
        )
        for row in dataset.rows:
            self.assertTrue(row.source_descendant_official_engine_executed)
            self.assertFalse(row.hidden_completion_truth_input)
            self.assertFalse(row.private_feature_input)
            self.assertFalse(row.future_reveal_input)
            self.assertEqual(
                tuple(name for name, _ in row.model_input_public_features),
                NATIVE_PUBLIC_FEATURE_NAMES,
            )
            self.assertEqual(
                {name for name, _ in row.neural_distribution},
                set(dataset.candidate_ids),
            )
            self.assertAlmostEqual(
                sum(value for _, value in row.neural_distribution),
                1.0,
            )
        self.assertFalse(calibration.public_belief_calibrated)
        self.assertFalse(calibration.official_hidden_state_posterior)
        self.assertFalse(evidence.counterfactual_values_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)

    def test_checked_phase6t_artifacts_roundtrip_and_remain_fail_closed(self) -> None:
        directory = PROJECT_ROOT / "artifacts" / "phase6_native_belief_calibration"
        dataset = NativeAllocationRangeDataset.from_dict(
            json.loads(
                (directory / "native_allocation_range_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        calibration = NativeBeliefCalibrationArtifact.from_dict(
            json.loads(
                (directory / "native_belief_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        evidence = Phase6NativeBeliefCalibrationEvidence.from_dict(
            json.loads(
                (directory / "phase6t_evidence.json").read_text(encoding="utf-8")
            )
        )
        self.assertEqual(calibration.dataset_fingerprint, dataset.fingerprint)
        self.assertEqual(evidence.dataset_fingerprint, dataset.fingerprint)
        self.assertEqual(evidence.calibration_fingerprint, calibration.fingerprint)
        self.assertEqual(len(dataset.rows), 216)
        self.assertEqual(evidence.validation_rows, 72)
        self.assertFalse(calibration.allocation_generalization_gate_passed)
        self.assertGreater(
            calibration.validation.neural.nll,
            calibration.validation.uniform.nll,
        )
        self.assertGreater(
            calibration.validation.neural.nll,
            calibration.validation.causal.nll,
        )
        self.assertFalse(evidence.public_belief_calibrated)


if __name__ == "__main__":
    unittest.main()
