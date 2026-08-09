from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot.native_blueprint_features import (
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
)
from rocketbot.native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanOutcomeDataset,
    native_plan_sample_view,
    restore_native_plan_blueprint,
)
from rocketbot.native_plan_game import (
    NATIVE_PLAN_ACTION_FEATURE_NAMES,
    NATIVE_PLAN_ACTIONS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NativePlanBlueprintArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = PROJECT_ROOT / "artifacts" / "phase6_native_plan_blueprint"
        cls.dataset_raw = json.loads(
            (cls.directory / "native_plan_outcome_dataset.json").read_text(
                encoding="utf-8"
            )
        )
        cls.model_raw = json.loads(
            (cls.directory / "native_plan_blueprint.json").read_text(
                encoding="utf-8"
            )
        )
        cls.evidence = json.loads(
            (cls.directory / "phase6o_evidence.json").read_text(encoding="utf-8")
        )
        cls.dataset = NativePlanOutcomeDataset.from_dict(cls.dataset_raw)
        cls.artifact = NativePlanBlueprintArtifact.from_dict(cls.model_raw)

    def test_live_dataset_roundtrips_and_has_both_player_targets(self) -> None:
        dataset = self.dataset
        self.assertEqual(dataset.fingerprint, self.dataset_raw["fingerprint"])
        self.assertEqual(dataset.fingerprint, self.evidence["dataset_fingerprint"])
        self.assertEqual(self.artifact.dataset_fingerprint, dataset.fingerprint)
        self.assertEqual(self.artifact.fingerprint, self.evidence["model_fingerprint"])
        self.assertEqual(len(dataset.candidate_ids), 6)
        self.assertEqual(len(set(dataset.candidate_ids)), 6)
        self.assertGreater(dataset.player_0_regrets.seen, 0)
        self.assertGreater(dataset.player_1_regrets.seen, 0)
        self.assertGreater(dataset.average_strategy.seen, 0)
        self.assertTrue(dataset.source_official_engine_executed)
        self.assertFalse(dataset.dataset_official_engine_executed)
        self.assertFalse(dataset.hidden_completion_prior_official)
        self.assertFalse(dataset.equilibrium_convergence_validated)

        samples = (
            dataset.player_0_regrets.samples
            + dataset.player_1_regrets.samples
            + dataset.average_strategy.samples
        )
        expected_actions = tuple(action.action_id for action in NATIVE_PLAN_ACTIONS)
        for sample in samples:
            self.assertEqual(
                tuple(name for name, _ in sample.public_features),
                NATIVE_PUBLIC_FEATURE_NAMES,
            )
            self.assertEqual(
                tuple(name for name, _ in sample.private_features),
                NATIVE_PRIVATE_FEATURE_NAMES,
            )
            self.assertEqual(
                tuple(action_id for action_id, _ in sample.action_features),
                expected_actions,
            )
            for _, features in sample.action_features:
                self.assertEqual(
                    tuple(name for name, _ in features),
                    NATIVE_PLAN_ACTION_FEATURE_NAMES,
                )

    def test_three_models_restore_with_exact_support_but_no_safety_claim(self) -> None:
        artifact = NativePlanBlueprintArtifact.from_dict(
            json.loads(self.artifact.to_json())
        )
        first = restore_native_plan_blueprint(artifact)
        second = restore_native_plan_blueprint(artifact)
        sample = self.dataset.player_0_regrets.samples[0]
        information_set, belief = native_plan_sample_view(sample)
        average_first = first.strategy(information_set, belief)
        average_second = second.strategy(information_set, belief)
        self.assertEqual(average_first, average_second)
        expected_support = {action.action_id for action in NATIVE_PLAN_ACTIONS}
        self.assertEqual(set(average_first.as_dict()), expected_support)
        for player in (0, 1):
            current = first.current_strategy(
                information_set,
                belief,
                player=player,
            )
            self.assertEqual(set(current.as_dict()), expected_support)
        self.assertFalse(artifact.equilibrium_convergence_validated)
        self.assertFalse(artifact.deployable_blueprint)
        self.assertFalse(artifact.calibrated_counterfactual_values)
        self.assertFalse(artifact.safe_resolver_compatible)
        with self.assertRaises(RuntimeError):
            first.counterfactual_values(information_set, belief, player=1)
        with self.assertRaises(ValueError):
            replace(artifact, deployable_blueprint=True)

    def test_artifact_rejects_private_zone_and_capability_promotion(self) -> None:
        tampered = copy.deepcopy(self.dataset_raw)
        tampered.pop("fingerprint", None)
        tampered["claims"]["raw_private_card_ids_persisted"] = 1
        with self.assertRaises(ValueError):
            NativePlanOutcomeDataset.from_dict(tampered)

        serialized = json.dumps(self.dataset_raw, sort_keys=True)
        for forbidden in (
            '"your_deck"',
            '"your_prize"',
            '"opponent_deck"',
            '"opponent_prize"',
            '"opponent_hand"',
            '"observation_json"',
            '"search_id"',
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(
            self.evidence["status"],
            "phase6o_native_turn_plan_blueprint_validated_partial",
        )
        self.assertEqual(self.evidence["claims"]["phase6_complete"], "false")
        self.assertEqual(
            self.evidence["claims"]["native_descendant_transitions"],
            "genuine_official_search_execution",
        )
        self.assertEqual(
            self.evidence["claims"]["turn_plan_actions"],
            "research_abstraction_not_official_options",
        )


if __name__ == "__main__":
    unittest.main()
