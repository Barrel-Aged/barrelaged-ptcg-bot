from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    PHASE6_OFFICIAL_VALUE_LEAF_POLICY_SPLITS,
    PHASE6_OFFICIAL_VALUE_LEAF_REPLICAS,
    OfficialValueLeafCalibration,
    OfficialValueLeafManifest,
    OfficialValueLeafModel,
    Phase6OfficialValueLeafEvidence,
    restore_official_value_leaf,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRECTORY = PROJECT_ROOT / "artifacts" / "phase6_official_value_leaf"


class OfficialValueLeafArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        def raw(name: str) -> dict:
            return json.loads((ARTIFACT_DIRECTORY / name).read_text(encoding="utf-8"))

        cls.manifest_raw = raw("official_value_leaf_manifest.json")
        cls.model_raw = raw("official_value_leaf_model.json")
        cls.calibration_raw = raw("official_value_leaf_calibration.json")
        cls.evidence_raw = raw("phase6ag_evidence.json")
        cls.manifest = OfficialValueLeafManifest.from_dict(cls.manifest_raw)
        cls.model = OfficialValueLeafModel.from_dict(cls.model_raw)
        cls.calibration = OfficialValueLeafCalibration.from_dict(
            cls.calibration_raw
        )
        cls.evidence = Phase6OfficialValueLeafEvidence.from_dict(cls.evidence_raw)

    def test_collection_grid_is_complete_split_isolated_and_balanced(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest.policy_variant_splits, PHASE6_OFFICIAL_VALUE_LEAF_POLICY_SPLITS)
        self.assertEqual(manifest.replicas_per_stratum, PHASE6_OFFICIAL_VALUE_LEAF_REPLICAS)
        self.assertEqual(len(manifest.matches), 288)
        self.assertEqual(
            dict(manifest.split_match_counts),
            {"train": 144, "calibration": 72, "validation": 72},
        )
        variants_by_split = {
            split: {
                item.policy_variant_id
                for item in manifest.matches
                if item.split == split
            }
            for split in ("train", "calibration", "validation")
        }
        self.assertEqual(variants_by_split["train"], {"canonical", "engine_first"})
        self.assertEqual(variants_by_split["calibration"], {"hand_first"})
        self.assertEqual(variants_by_split["validation"], {"action_first"})
        self.assertTrue(variants_by_split["train"].isdisjoint(variants_by_split["calibration"] | variants_by_split["validation"]))
        self.assertTrue(variants_by_split["calibration"].isdisjoint(variants_by_split["validation"]))

        roles: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for item in manifest.matches:
            if item.split == "calibration":
                roles[(item.candidate_id, item.intent_id)][
                    item.calibration_partition
                ] += 1
        self.assertEqual(len(roles), 36)
        self.assertTrue(
            all(counts == Counter({"selection": 1, "bounds": 1}) for counts in roles.values())
        )

    def test_compact_artifacts_roundtrip_exactly_and_are_fingerprint_bound(self) -> None:
        for artifact, raw in (
            (self.manifest, self.manifest_raw),
            (self.model, self.model_raw),
            (self.calibration, self.calibration_raw),
            (self.evidence, self.evidence_raw),
        ):
            self.assertEqual(artifact.to_dict(), raw)
        self.assertEqual(self.model.source_manifest_fingerprint, self.manifest.fingerprint)
        self.assertEqual(self.calibration.source_model_fingerprint, self.model.fingerprint)
        self.assertEqual(self.evidence.calibration_fingerprint, self.calibration.fingerprint)
        self.assertEqual(
            self.evidence.dataset_fingerprint,
            "be1fdfba69e1048101ac5d39f72e98c628f7b474b0c45862e3c70b01063b207b",
        )
        self.assertEqual(
            self.evidence.fingerprint,
            "2112d91fdb5fe6adfbaee72b86346846e87806818368f6054919fd3ecaf97673",
        )

    def test_model_uses_only_public_private_and_range_inputs(self) -> None:
        names = self.model.normalizer.names
        prefixes = Counter(name.split("::", 1)[0] for name in names)
        self.assertEqual(
            prefixes,
            Counter({"public": 127, "private": 75, "range_probability": 6}),
        )
        self.assertEqual(len(names), 208)
        forbidden = ("policy_variant", "intent_id", "truth_candidate", "outcome")
        self.assertFalse(any(token in name for token in forbidden for name in names))
        self.assertTrue(self.model.public_plus_acting_private_features_only)
        self.assertFalse(self.model.supervision_identifiers_as_features)
        self.assertFalse(self.model.official_engine_executed)
        self.assertFalse(self.model.equilibrium_trained)
        self.assertFalse(self.model.counterfactual_value_provider)
        self.assertFalse(self.model.deployable_blueprint)
        self.assertIsNotNone(restore_official_value_leaf(self.model).network)

    def test_point_leaf_passes_both_baselines_but_intervals_fail_closed(self) -> None:
        evidence = self.evidence
        self.assertEqual(evidence.match_count, 288)
        self.assertEqual(evidence.episode_count, 576)
        self.assertEqual(evidence.trajectory_step_count, 30_668)
        self.assertEqual(evidence.live_selection_count, 30_668)
        self.assertAlmostEqual(evidence.validation_episode_mse, 0.9387949616063226)
        self.assertLess(evidence.validation_episode_mse, evidence.validation_zero_episode_mse)
        self.assertLess(evidence.validation_episode_mse, evidence.validation_heuristic_episode_mse)
        self.assertTrue(evidence.implementation_heldout_point_leaf_validated)
        self.assertTrue(self.calibration.point_model_gate_passed)
        self.assertTrue(self.calibration.implementation_holdout_gate_passed)
        self.assertFalse(self.calibration.progress_interval_gate_passed)
        self.assertFalse(evidence.implementation_heldout_value_leaf_calibrated)
        self.assertFalse(evidence.counterfactual_value_compatible)
        self.assertFalse(evidence.safe_resolver_enabled)
        self.assertFalse(evidence.opponent_exploitation_enabled)
        self.assertFalse(evidence.official_strategy_control_enabled)
        self.assertEqual(
            dict(evidence.claims)["expected_value_intervals"],
            "rejected_as_too_wide_or_undercovered",
        )

    def test_capability_and_protocol_tampering_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.model, official_engine_executed=True)
        with self.assertRaises(ValueError):
            replace(self.calibration, counterfactual_value_compatible=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, official_strategy_control_enabled=True)
        with self.assertRaises(ValueError):
            replace(self.evidence, implementation_heldout_point_leaf_validated=False)
        with self.assertRaises(ValueError):
            replace(
                self.evidence,
                claims=tuple(
                    (name, "rejected_against_point_baselines" if name == "point_leaf" else value)
                    for name, value in self.evidence.claims
                ),
            )

        crossed = deepcopy(self.manifest_raw)
        crossed["policy_variant_splits"][3][1] = "train"
        with self.assertRaises(ValueError):
            OfficialValueLeafManifest.from_dict(crossed)


if __name__ == "__main__":
    unittest.main()
