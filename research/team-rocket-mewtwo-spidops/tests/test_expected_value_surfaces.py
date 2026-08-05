from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import unittest

from rocketbot import (
    PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT,
    PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL,
    PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION,
    ExpectedValueCalibration,
    ExpectedValueDataset,
    ExpectedValueModel,
    Phase6ExpectedValueEvidence,
    build_expected_value_rows,
    restore_expected_value_model,
)
from rocketbot.phase6_native_external_sampling import _raw_private_key_count
from rocketbot.phase6_expected_value_surfaces import _target_reverse_for_capture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_expected_value_surfaces"


class ExpectedValueSurfaceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads((ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8"))

        cls.dataset_raw = raw("expected_value_surface_dataset.json")
        cls.model_raw = raw("expected_value_model.json")
        cls.calibration_raw = raw("expected_value_calibration.json")
        cls.evidence_raw = raw("phase6ah_evidence.json")
        cls.dataset = ExpectedValueDataset.from_dict(cls.dataset_raw)
        cls.model = ExpectedValueModel.from_dict(cls.model_raw)
        cls.calibration = ExpectedValueCalibration.from_dict(cls.calibration_raw)
        cls.evidence = Phase6ExpectedValueEvidence.from_dict(cls.evidence_raw)

    def test_frozen_grid_and_repeated_surface_accounting_are_complete(self) -> None:
        dataset = self.dataset
        self.assertEqual(dataset.policy_protocol, PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL)
        self.assertEqual(len(dataset.roots), 180)
        self.assertEqual(
            dict(dataset.split_root_counts),
            {"train": 72, "calibration": 72, "validation": 36},
        )
        self.assertEqual(sum(len(root.surfaces) for root in dataset.roots), 360)
        self.assertEqual(dataset.terminal_rollout_count, 14_400)
        self.assertEqual(len(build_expected_value_rows(dataset)), 900)
        roles = Counter(
            root.calibration_partition
            for root in dataset.roots
            if root.split == "calibration"
        )
        self.assertEqual(roles, Counter({"selection": 36, "bounds": 36}))
        for root in dataset.roots:
            self.assertEqual(len(root.surfaces), PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT)
            self.assertEqual(root.terminal_rollout_count, 80)
            for surface in root.surfaces:
                self.assertEqual(len(surface.action_values), 5)
                self.assertTrue(surface.root_replay_public_parity)
                self.assertTrue(surface.root_replay_acting_private_parity)
                self.assertFalse(surface.hidden_allocation_official)
                for estimate in surface.action_values:
                    self.assertEqual(
                        len(estimate.terminal_returns),
                        PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION,
                    )
                    self.assertAlmostEqual(
                        estimate.empirical_player0_value,
                        statistics.mean(estimate.terminal_returns),
                    )

    def test_capture_retries_alternate_direction_without_changing_attempt_one(self) -> None:
        match_id = "phase6ah-direction-regression"
        directions = tuple(
            _target_reverse_for_capture(match_id, attempt)
            for attempt in range(1, 5)
        )
        self.assertEqual(directions[0], directions[2])
        self.assertEqual(directions[1], directions[3])
        self.assertNotEqual(directions[0], directions[1])
        with self.assertRaises(ValueError):
            _target_reverse_for_capture(match_id, 0)

    def test_model_schema_excludes_intent_and_policy_implementation(self) -> None:
        names = self.model.normalizer.names
        prefixes = Counter(name.split("::", 1)[0] for name in names)
        self.assertGreater(prefixes["public"], 0)
        self.assertGreater(prefixes["private"], 0)
        self.assertEqual(prefixes["history"], 16)
        self.assertEqual(prefixes["candidate_hypothesis"], 6)
        self.assertEqual(prefixes["plan_action"], 5)
        self.assertFalse(
            any(
                token in name
                for name in names
                for token in ("intent_id", "policy_variant", "supervision_only")
            )
        )
        self.assertTrue(
            self.model.policy_visible_features_plus_candidate_and_action_only
        )
        self.assertFalse(self.model.intent_identifier_as_feature)
        self.assertFalse(self.model.policy_variant_identifier_as_feature)
        self.assertFalse(self.model.official_engine_executed)
        self.assertFalse(self.model.equilibrium_trained)
        self.assertFalse(self.model.counterfactual_value_provider)
        self.assertIsNotNone(restore_expected_value_model(self.model).network)

    def test_empirical_intervals_cover_mean_uncertainty_and_stay_fail_closed(self) -> None:
        calibration = self.calibration
        evidence = self.evidence
        self.assertTrue(calibration.implementation_holdout_gate_passed)
        self.assertEqual(
            evidence.implementation_heldout_point_model_validated,
            calibration.point_model_gate_passed,
        )
        self.assertEqual(
            evidence.implementation_heldout_conditional_expected_mean_admitted,
            calibration.conditional_expected_mean_model_admitted,
        )
        self.assertTrue(math.isfinite(calibration.conformal_radius))
        self.assertEqual(
            {item.dimension for item in calibration.slice_metrics},
            {"action", "candidate", "intent"},
        )
        self.assertFalse(calibration.counterfactual_value_compatible)
        self.assertFalse(calibration.safe_resolver_compatible)
        self.assertFalse(calibration.opponent_exploitation_enabled)
        self.assertFalse(calibration.official_strategy_control_enabled)
        self.assertFalse(evidence.counterfactual_value_compatible)
        self.assertFalse(evidence.equilibrium_convergence_validated)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.opponent_exploitation_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)

    def test_artifacts_roundtrip_bind_sources_and_persist_no_raw_hidden_zones(self) -> None:
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

    def test_capability_and_protocol_tampering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.dataset, hidden_allocation_prior_official=True)
        with self.assertRaises(ValueError):
            replace(self.model, counterfactual_value_provider=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, safe_resolver_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)
        with self.assertRaises(ValueError):
            replace(
                self.evidence,
                implementation_heldout_point_model_validated=(
                    not self.evidence.implementation_heldout_point_model_validated
                ),
            )


if __name__ == "__main__":
    unittest.main()
