import copy
import json
from pathlib import Path
import unittest

from rocketbot.native_blueprint_features import (
    NATIVE_ACTION_FEATURE_NAMES,
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
)
from rocketbot.phase6_native_external_sampling import (
    NativeExternalSamplingDataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativeExternalSamplingArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = (
            PROJECT_ROOT
            / "artifacts"
            / "phase6_native_external_sampling"
            / "native_external_sampling_dataset.json"
        )
        cls.raw = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.dataset = NativeExternalSamplingDataset.from_dict(cls.raw)

    def test_checked_in_native_wiring_artifact_roundtrips_without_private_zones(self):
        dataset = self.dataset
        self.assertEqual(dataset.fingerprint, self.raw["fingerprint"])
        self.assertTrue(dataset.source_official_engine_executed)
        self.assertFalse(dataset.hidden_completion_prior_official)
        self.assertTrue(dataset.fixed_completion_endgame_roots)
        self.assertFalse(dataset.selected_completion_truth_policy_input)
        self.assertFalse(dataset.full_game_deep_cfr_iterations)
        self.assertFalse(dataset.equilibrium_convergence_validated)
        self.assertGreater(dataset.player_0_advantages.seen, 0)
        self.assertGreater(dataset.player_1_advantages.seen, 0)
        self.assertEqual(dataset.average_strategy.seen, 0)
        self.assertNotEqual(
            dataset.source_blueprint_game_fingerprint,
            dataset.game_fingerprint,
        )

        serialized = json.dumps(dataset.to_dict(), sort_keys=True)
        for forbidden in (
            '"your_deck"',
            '"your_prize"',
            '"opponent_deck"',
            '"opponent_prize"',
            '"opponent_hand"',
            '"opponent_active"',
            '"observation_json"',
            '"search_id"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_native_samples_have_exact_feature_widths_and_provenance(self):
        samples = (
            self.dataset.player_0_advantages.samples
            + self.dataset.player_1_advantages.samples
        )
        self.assertEqual(len(samples), len(self.dataset.traversals))
        for sample in samples:
            self.assertEqual(
                tuple(name for name, _ in sample.public_features),
                NATIVE_PUBLIC_FEATURE_NAMES,
            )
            self.assertEqual(
                tuple(name for name, _ in sample.private_features),
                NATIVE_PRIVATE_FEATURE_NAMES,
            )
            for _, features in sample.action_features:
                self.assertEqual(
                    tuple(name for name, _ in features),
                    NATIVE_ACTION_FEATURE_NAMES,
                )
            self.assertTrue(sample.source_official_engine_executed)
            self.assertFalse(sample.hidden_completion_prior_official)

    def test_native_dataset_rejects_raw_private_zone_injection(self):
        tampered = copy.deepcopy(self.raw)
        tampered["opponent_hand"] = [1, 2, 3]
        with self.assertRaises(ValueError):
            NativeExternalSamplingDataset.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
