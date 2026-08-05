from __future__ import annotations

from pathlib import Path
import unittest

from rocketbot import run_phase6_native_search_game
from rocketbot.official_search_game import (
    HiddenZoneCompletion,
    permute_hidden_zone_completion,
    sample_hidden_zone_completion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OfficialNativeSearchGameTests(unittest.TestCase):
    def test_hidden_allocation_particles_preserve_public_evidence_and_counts(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "turn": 0,
                "turnActionCount": 0,
                "players": [
                    {
                        "hand": [{"id": 1}, {"id": 2}],
                        "discard": [{"id": 3}],
                        "active": [{"id": 4}],
                        "bench": [],
                        "prize": [{"id": 5}, None, None, None, None, None],
                        "deckCount": 50,
                        "handCount": 2,
                    },
                    {
                        "hand": [],
                        "discard": [{"id": 101}],
                        "active": [None],
                        "bench": [{"id": 102}],
                        "prize": [{"id": 103}, None, None, None, None, None],
                        "deckCount": 44,
                        "handCount": 7,
                    },
                ],
            },
            "select": {},
            "logs": [],
        }
        arguments = {
            "probability": 0.5,
            "your_candidate_id": "rocket",
            "opponent_candidate_id": "opponent",
            "your_archetype_id": "rocket",
            "opponent_archetype_id": "opponent",
            "your_style_id": "unknown",
            "opponent_style_id": "unknown",
            "your_card_ids": tuple(range(1, 61)),
            "opponent_card_ids": tuple(range(101, 161)),
            "your_deck_fingerprint": "a" * 64,
            "opponent_deck_fingerprint": "b" * 64,
            "basic_card_ids": (104, 105, 106),
            "source": "unit-test sampled public-consistent allocation",
        }
        left = sample_hidden_zone_completion(
            observation,
            hypothesis_id="allocation-left",
            allocation_seed=17,
            **arguments,
        )
        right = sample_hidden_zone_completion(
            observation,
            hypothesis_id="allocation-right",
            allocation_seed=29,
            **arguments,
        )

        self.assertEqual(len(left.your_prize), 6)
        self.assertEqual(len(left.your_deck), 50)
        self.assertEqual(len(left.opponent_prize), 6)
        self.assertEqual(len(left.opponent_hand), 7)
        self.assertEqual(len(left.opponent_deck), 44)
        self.assertEqual(len(left.opponent_active), 1)
        self.assertIn(5, left.your_prize)
        self.assertIn(103, left.opponent_prize)
        self.assertIn(left.opponent_active[0], arguments["basic_card_ids"])
        self.assertEqual(
            sorted(left.your_deck + left.your_prize),
            sorted(right.your_deck + right.your_prize),
        )
        self.assertEqual(
            sorted(
                left.opponent_deck
                + left.opponent_prize
                + left.opponent_hand
                + left.opponent_active
            ),
            sorted(
                right.opponent_deck
                + right.opponent_prize
                + right.opponent_hand
                + right.opponent_active
            ),
        )
        self.assertTrue(
            any(
                sorted(getattr(left, field_name))
                != sorted(getattr(right, field_name))
                for field_name in (
                    "your_deck",
                    "your_prize",
                    "opponent_deck",
                    "opponent_prize",
                    "opponent_hand",
                    "opponent_active",
                )
            )
        )
        self.assertFalse(left.manifest["raw_card_ids_persisted"])
        self.assertNotIn("opponent_hand", left.manifest)

    def test_hidden_completion_order_particles_preserve_zone_multisets(self) -> None:
        base = HiddenZoneCompletion(
            hypothesis_id="base",
            probability=1.0,
            root_player_index=0,
            your_candidate_id="rocket",
            opponent_candidate_id="opponent",
            your_archetype_id="rocket",
            opponent_archetype_id="opponent",
            your_style_id="unknown",
            opponent_style_id="unknown",
            your_deck_fingerprint="a" * 64,
            opponent_deck_fingerprint="b" * 64,
            your_deck=(1, 2, 3, 4),
            your_prize=(5, 6, 7),
            opponent_deck=(8, 9, 10, 11),
            opponent_prize=(12, 13, 14),
            opponent_hand=(15, 16, 17),
            opponent_active=(18,),
            manual_coin=True,
            public_evidence_adjustments=0,
            source="unit-test public-consistent completion",
        )
        particle = permute_hidden_zone_completion(
            base,
            hypothesis_id="heldout-order-1",
            probability=0.25,
            permutation_seed=17,
        )
        for field_name in (
            "your_deck",
            "your_prize",
            "opponent_deck",
            "opponent_prize",
            "opponent_hand",
            "opponent_active",
        ):
            self.assertEqual(
                sorted(getattr(particle, field_name)),
                sorted(getattr(base, field_name)),
            )
        self.assertNotEqual(particle.fingerprint, base.fingerprint)
        self.assertNotIn("your_deck", particle.manifest)
        self.assertNotIn("opponent_hand", particle.manifest)
        self.assertFalse(particle.manifest["raw_card_ids_persisted"])

    def test_phase6j_branches_native_states_without_promoting_hidden_prior(self) -> None:
        artifacts = run_phase6_native_search_game(project_root=PROJECT_ROOT)
        evidence = artifacts.evidence
        self.assertEqual(
            evidence.status,
            "phase6j_branchable_native_search_game_validated_partial",
        )
        self.assertTrue(evidence.source_official_engine_executed)
        self.assertEqual(evidence.game_boundary_kind, "abstract_information_set_game")
        self.assertFalse(evidence.game_official_engine_executed)
        self.assertFalse(evidence.hidden_completion_prior_official)
        self.assertTrue(evidence.branchable_native_transitions)
        self.assertEqual(evidence.completion_count, 6)
        self.assertAlmostEqual(evidence.completion_probability_sum, 1.0)
        self.assertEqual(evidence.raw_private_card_ids_persisted, 0)
        self.assertTrue(evidence.root_public_observation_equivalence)
        self.assertTrue(evidence.root_private_observation_equivalence)
        self.assertTrue(evidence.root_information_set_invariant_across_completions)
        self.assertTrue(evidence.root_public_belief_invariant_across_completions)
        self.assertFalse(evidence.selected_completion_truth_enters_policy_payload)
        self.assertGreaterEqual(evidence.root_legal_action_count, 2)
        self.assertEqual(
            evidence.root_distinct_alternative_children,
            evidence.root_legal_action_count,
        )
        self.assertTrue(evidence.repeated_root_transition_deterministic)
        self.assertTrue(evidence.tampered_action_rejected)
        self.assertTrue(evidence.tampered_chance_outcome_rejected)
        self.assertTrue(evidence.all_scenarios_terminal)
        self.assertTrue(evidence.both_players_traversed_in_every_scenario)
        self.assertTrue(all(item.midgame_branch_verified for item in evidence.scenarios))
        self.assertTrue(all(item.terminal_zero_sum_exact for item in evidence.scenarios))
        self.assertTrue(all(item.terminal_depth > 1 for item in evidence.scenarios))
        self.assertGreater(evidence.native_transition_count, 0)
        self.assertGreaterEqual(evidence.native_state_count, evidence.native_transition_count)
        for manifest in evidence.completion_manifests:
            self.assertFalse(manifest["raw_card_ids_persisted"])
            self.assertNotIn("your_deck", manifest)
            self.assertNotIn("opponent_hand", manifest)
        claims = dict(evidence.claims)
        self.assertEqual(claims["equilibrium_blueprint"], "not_trained")
        self.assertEqual(claims["phase6_complete"], "false")


if __name__ == "__main__":
    unittest.main()
