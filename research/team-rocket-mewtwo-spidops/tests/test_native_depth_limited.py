from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from rocketbot.phase6_native_depth_limited import (
    NativeDepthLimitedPlanDataset,
    Phase6NativeDepthLimitedEvidence,
    audit_depth_limited_exact_expectation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeDepthLimitedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        directory = PROJECT_ROOT / "artifacts" / "phase6_native_depth_limited"
        cls.dataset_raw = json.loads(
            (directory / "depth_limited_native_plan_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        cls.evidence_raw = json.loads(
            (directory / "phase6r_evidence.json").read_text(encoding="utf-8")
        )
        cls.dataset = NativeDepthLimitedPlanDataset.from_dict(cls.dataset_raw)
        cls.evidence = Phase6NativeDepthLimitedEvidence.from_dict(cls.evidence_raw)

    def test_exact_cutoff_preserves_pre_cutoff_expectations(self) -> None:
        audit = audit_depth_limited_exact_expectation()
        self.assertEqual(audit.audited_regret_entries, 18)
        self.assertLess(audit.maximum_absolute_error, 1e-12)
        self.assertGreater(audit.exact_leaf_count, 0)

    def test_live_dataset_is_abstract_redacted_and_roundtrips(self) -> None:
        dataset = self.dataset
        self.assertEqual(
            NativeDepthLimitedPlanDataset.from_dict(
                json.loads(dataset.to_json())
            ).fingerprint,
            dataset.fingerprint,
        )
        self.assertTrue(dataset.source_descendant_official_engine_executed)
        self.assertFalse(dataset.target_boundary_official_engine_executed)
        self.assertFalse(dataset.hidden_completion_prior_official)
        self.assertFalse(dataset.leaf_values_calibrated)
        self.assertFalse(dataset.counterfactual_values_calibrated)
        self.assertEqual(dataset.raw_private_card_ids_persisted, 0)
        self.assertEqual(dataset.leaf_estimate_count, dataset.cutoff_trajectory_count)
        samples = (
            dataset.player_0_regrets.samples
            + dataset.player_1_regrets.samples
            + dataset.average_strategy.samples
        )
        self.assertTrue(samples)
        self.assertTrue(
            all(not sample.source_official_engine_executed for sample in samples)
        )

    def test_evidence_reduces_observed_tail_but_enables_no_capability(self) -> None:
        evidence = self.evidence
        self.assertEqual(evidence.dataset_fingerprint, self.dataset.fingerprint)
        self.assertLess(evidence.p95_ratio_to_phase6p, 1.0)
        self.assertLess(evidence.maximum_ratio_to_phase6p, 1.0)
        self.assertFalse(evidence.leaf_values_calibrated)
        self.assertFalse(evidence.counterfactual_bounds_built)
        self.assertFalse(evidence.safe_resolver_executed)
        self.assertFalse(evidence.exploitation_executed)
        self.assertFalse(evidence.official_strategy_control_enabled)
        self.assertEqual(dict(evidence.claims)["phase6_complete"], "false")

        tampered = copy.deepcopy(self.evidence_raw)
        tampered.pop("fingerprint", None)
        tampered["execution"]["safe_resolver_executed"] = True
        with self.assertRaises(ValueError):
            Phase6NativeDepthLimitedEvidence.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
