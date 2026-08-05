from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import unittest

from rocketbot import (
    NATIVE_PUBLIC_FEATURE_NAMES,
    OPPONENT_DEVIATION_ACTION_IDS,
    OpponentDeviationCalibration,
    OpponentDeviationDataset,
    Phase6OpponentDeviationEvidence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_opponent_deviation_values"


class OpponentDeviationValueArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = OpponentDeviationDataset.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "opponent_deviation_dataset.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        cls.calibration = OpponentDeviationCalibration.from_dict(
            json.loads(
                (
                    ARTIFACT_DIRECTORY / "opponent_deviation_calibration.json"
                ).read_text(encoding="utf-8")
            )
        )
        cls.evidence = Phase6OpponentDeviationEvidence.from_dict(
            json.loads(
                (ARTIFACT_DIRECTORY / "phase6x_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
        )

    def test_dataset_has_complete_hypothesis_action_support(self) -> None:
        dataset = self.dataset
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertEqual(len(dataset.surfaces), 12)
        self.assertEqual(dataset.rollouts_per_action, 10)
        self.assertEqual(dataset.terminal_rollout_count, 600)
        self.assertEqual(
            {
                (surface.candidate_id_supervision_only, surface.variant)
                for surface in dataset.surfaces
            },
            {
                (candidate_id, variant)
                for candidate_id in dataset.candidate_ids
                for variant in (0, 1)
            },
        )
        self.assertEqual(
            {surface.split for surface in dataset.surfaces if surface.variant == 0},
            {"calibration"},
        )
        self.assertEqual(
            {surface.split for surface in dataset.surfaces if surface.variant == 1},
            {"validation"},
        )
        self.assertEqual(
            len({surface.particle_fingerprint for surface in dataset.surfaces}),
            12,
        )
        self.assertTrue(dataset.common_random_numbers_across_actions)
        self.assertTrue(dataset.source_descendants_official_engine_executed)
        self.assertTrue(dataset.conditional_action_values_only)
        self.assertFalse(dataset.opponent_reach_weighted)
        self.assertFalse(dataset.exact_counterfactual_targets)
        self.assertEqual(dataset.raw_private_card_ids_persisted, 0)
        self.assertGreater(dataset.native_transition_count, 0)

    def test_artifact_json_shapes_roundtrip_exactly(self) -> None:
        for filename, artifact in (
            ("opponent_deviation_dataset.json", self.dataset),
            ("opponent_deviation_calibration.json", self.calibration),
            ("phase6x_evidence.json", self.evidence),
        ):
            raw = json.loads((ARTIFACT_DIRECTORY / filename).read_text(encoding="utf-8"))
            self.assertEqual(artifact.to_dict(), raw)

    def test_each_surface_roundtrips_empirical_values_and_precision(self) -> None:
        for surface in self.dataset.surfaces:
            self.assertEqual(
                tuple(name for name, _ in surface.public_features),
                NATIVE_PUBLIC_FEATURE_NAMES,
            )
            self.assertEqual(
                tuple(action.action_id for action in surface.action_values),
                OPPONENT_DEVIATION_ACTION_IDS,
            )
            self.assertAlmostEqual(
                sum(action.blueprint_probability for action in surface.action_values),
                1.0,
            )
            self.assertTrue(surface.source_descendant_official_engine_executed)
            self.assertTrue(surface.conditional_on_sampled_reach)
            self.assertFalse(surface.opponent_reach_weighted)
            self.assertFalse(surface.exact_counterfactual_values)
            self.assertFalse(surface.selected_hidden_truth_identifier_input)
            self.assertFalse(surface.raw_private_card_ids_persisted)
            for action in surface.action_values:
                self.assertEqual(len(action.terminal_returns), 10)
                self.assertAlmostEqual(
                    action.empirical_opponent_value,
                    statistics.mean(action.terminal_returns),
                )
                self.assertAlmostEqual(
                    action.rollout_standard_error,
                    statistics.stdev(action.terminal_returns) / math.sqrt(10),
                )
                self.assertLessEqual(action.rollout_standard_error, 1 / 3 + 1e-12)
            with self.assertRaises(ValueError):
                replace(surface, exact_counterfactual_values=True)

    def test_conditional_signal_passes_but_counterfactual_safety_stays_false(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertEqual(calibration.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(evidence.dataset_fingerprint, self.dataset.fingerprint)
        self.assertEqual(evidence.calibration_fingerprint, calibration.fingerprint)
        self.assertTrue(calibration.conditional_deviation_substrate_gate_passed)
        self.assertFalse(calibration.opponent_counterfactual_value_gate_passed)
        self.assertFalse(calibration.safe_resolver_compatible)
        self.assertEqual(calibration.calibration.candidate_strata, 6)
        self.assertEqual(calibration.validation.candidate_strata, 6)
        self.assertEqual(
            calibration.validation.nontrivial_action_surface_fraction,
            1.0,
        )
        self.assertAlmostEqual(
            calibration.validation.maximum_rollout_standard_error,
            1 / 3,
        )
        self.assertGreater(calibration.validation.mean_action_span, 0.1)
        self.assertEqual(
            calibration.validation.blueprint_top_action_matches_empirical_best_fraction,
            0.0,
        )
        self.assertFalse(evidence.opponent_counterfactual_values_calibrated)
        self.assertFalse(evidence.public_belief_calibrated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        with self.assertRaises(ValueError):
            replace(calibration, opponent_counterfactual_value_gate_passed=True)
        with self.assertRaises(ValueError):
            replace(evidence, safe_resolver_enabled=True)
        with self.assertRaises(ValueError):
            replace(self.dataset, exact_counterfactual_targets=True)


if __name__ == "__main__":
    unittest.main()
