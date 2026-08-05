from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import random
import unittest

from rocketbot import (
    CommonChanceLawReachRatio,
    CounterfactualReachExpectationAudit,
    CounterfactualReachTrace,
    RocketPrizeDuel,
    UniformBlueprintStub,
    audit_counterfactual_reach_expectation,
    counterfactual_importance_weight,
    common_chance_law_counterfactual_ratio,
    load_default_deck,
    sample_first_player_decision_reach,
    uniform_profile,
)


class CounterfactualReachTests(unittest.TestCase):
    def test_common_opaque_chance_law_cancels_without_absolute_reach(self) -> None:
        chance_law = "a" * 64
        record = CommonChanceLawReachRatio(
            target_player=1,
            chance_law_fingerprint=chance_law,
            player_0_target_reach=0.6,
            player_1_target_reach=0.2,
            player_0_sampling_reach=0.3,
            player_1_sampling_reach=0.4,
        )
        expected = 0.6 / (0.3 * 0.4)
        self.assertAlmostEqual(record.counterfactual_likelihood_ratio, expected)
        self.assertIsNone(record.to_dict()["derived"]["absolute_chance_reach"])
        self.assertEqual(
            CommonChanceLawReachRatio.from_dict(record.to_dict()),
            record,
        )
        with self.assertRaises(ValueError):
            replace(record, absolute_chance_reach_available=True)
        with self.assertRaises(ValueError):
            replace(record, target_and_sampling_chance_law_identical=False)

    def test_common_chance_ratio_matches_explicit_weights_for_any_chance_mass(self) -> None:
        ratio = common_chance_law_counterfactual_ratio(
            target_player=0,
            player_0_target_reach=0.7,
            player_1_target_reach=0.5,
            player_0_sampling_reach=0.25,
            player_1_sampling_reach=0.8,
        )
        for chance in (0.01, 0.2, 0.5, 1.0):
            explicit = counterfactual_importance_weight(
                target_player=0,
                chance_reach=chance,
                player_0_reach=0.7,
                player_1_reach=0.5,
                sampling_reach=chance * 0.25 * 0.8,
            )
            self.assertAlmostEqual(ratio, explicit)

    def test_importance_weight_excludes_target_player_reach(self) -> None:
        weight_0 = counterfactual_importance_weight(
            target_player=0,
            chance_reach=0.2,
            player_0_reach=0.25,
            player_1_reach=0.4,
            sampling_reach=0.02,
        )
        weight_1 = counterfactual_importance_weight(
            target_player=1,
            chance_reach=0.2,
            player_0_reach=0.25,
            player_1_reach=0.4,
            sampling_reach=0.02,
        )
        self.assertAlmostEqual(weight_0, 4.0)
        self.assertAlmostEqual(weight_1, 2.5)

    def test_exact_importance_expectation_matches_counterfactual_values(self) -> None:
        game = RocketPrizeDuel()
        audit = audit_counterfactual_reach_expectation(game, uniform_profile(game))
        self.assertEqual(len(audit.rows), 24)
        self.assertEqual(audit.maximum_absolute_error, 0.0)
        self.assertEqual(audit.maximum_importance_weight, 2.0)
        self.assertTrue(audit.exact_expectation_passed)
        self.assertFalse(audit.official_engine_executed)
        self.assertFalse(audit.full_game_native_counterfactual_values_calibrated)
        restored = CounterfactualReachExpectationAudit.from_dict(
            json.loads(audit.to_json())
        )
        self.assertEqual(restored.to_dict(), audit.to_dict())

    def test_first_opponent_decision_trace_is_redacted_and_fail_closed(self) -> None:
        game = RocketPrizeDuel()
        provider = UniformBlueprintStub(
            game_fingerprint=game.fingerprint,
            deck_fingerprint=load_default_deck().fingerprint,
        )
        state, trace = sample_first_player_decision_reach(
            game,
            provider,
            target_player=1,
            rng=random.Random(17),
            maximum_transitions=8,
            redaction_salt="unit-test-redaction",
            source_descendants_official_engine_executed=False,
        )
        self.assertEqual(game.player_to_act(state), 1)
        self.assertEqual(trace.player_1_reach, 1.0)
        self.assertEqual(trace.importance_weight, 1.0)
        self.assertTrue(trace.target_and_sampling_strategy_match)
        self.assertFalse(trace.upstream_official_reach_complete)
        serialized = trace.to_dict()
        for step in serialized["steps"]:
            self.assertNotIn("selected_identifier", step)
            self.assertEqual(len(step["selected_identifier_fingerprint"]), 64)
        self.assertEqual(
            CounterfactualReachTrace.from_dict(serialized).to_dict(),
            serialized,
        )
        tampered = deepcopy(serialized)
        tampered["derived_reach"]["player_1"] = 0.5
        with self.assertRaises(ValueError):
            CounterfactualReachTrace.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
