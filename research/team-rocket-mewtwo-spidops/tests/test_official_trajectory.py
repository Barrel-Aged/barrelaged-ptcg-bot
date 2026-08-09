from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot.official_trajectory import (
    complete_replay_actions,
    private_frame_payload,
    private_feature_items,
    public_frame_payload,
)
from rocketbot.phase6_trajectory import run_phase6_official_trajectory
from rocketbot.public_replay import AgentVisibleSelection, PublicReplayDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OfficialTrajectoryTests(unittest.TestCase):
    def test_complete_ordered_actions(self):
        selection = AgentVisibleSelection(
            selection_type="Card",
            context="Discard",
            min_count=1,
            max_count=2,
            remain_damage_counter=0,
            remain_energy_cost=0,
            options=tuple(
                (("type", "Card"), ("index", index)) for index in range(3)
            ),
            visible_search_card_ids=(),
            effect_card_id=None,
            context_card_id=None,
        )
        actions = complete_replay_actions(selection)
        self.assertEqual(len(actions), 9)
        self.assertIn((2, 0), {item.official_option_indices for item in actions})

    def test_phase6h_logged_trajectory_and_private_separation(self):
        artifacts = run_phase6_official_trajectory(project_root=PROJECT_ROOT)
        dataset = artifacts.dataset
        evidence = artifacts.evidence
        summary = dataset.summary
        self.assertEqual(
            evidence.status,
            "phase6h_logged_trajectory_substrate_validated_partial",
        )
        self.assertEqual(summary["steps"], 610)
        self.assertEqual(summary["behavior_labeled_steps"], 610)
        self.assertEqual(summary["behavior_missing_steps"], 0)
        self.assertEqual(summary["terminal_transition_links"], 12)
        self.assertEqual(summary["maximum_complete_action_count"], 260)
        self.assertFalse(summary["branchable_transition_game"])
        self.assertFalse(summary["dataset_official_engine_executed"])
        self.assertTrue(summary["source_official_engine_executed"])
        self.assertLess(summary["public_range_nll"], summary["uniform_range_nll"])
        self.assertLess(
            summary["public_range_brier"], summary["uniform_range_brier"]
        )
        self.assertEqual(dataset.leakage_audit.public_state_mismatches, 0)
        self.assertEqual(dataset.leakage_audit.public_history_mismatches, 0)
        self.assertTrue(evidence.serialization_roundtrip_exact)
        self.assertEqual(dict(evidence.claims)["phase6_complete"], "false")

        source_raw = json.loads(
            (
                PROJECT_ROOT
                / "artifacts"
                / "phase6_game_theory_foundation"
                / "public_replay_dataset.json"
            ).read_text(encoding="utf-8")
        )
        source = PublicReplayDataset.from_dict(source_raw)
        frame = next(
            frame
            for episode in source.episodes
            for frame in episode.frames
            if frame.selection is not None
            and frame.players[frame.observer_player].known_hand_card_ids
        )
        own = frame.players[frame.observer_player]
        mutated_own = replace(
            own,
            known_hand_card_ids=tuple(
                987_654_321 for _ in own.known_hand_card_ids
            ),
        )
        players = list(frame.players)
        players[frame.observer_player] = mutated_own
        mutated = replace(frame, players=(players[0], players[1]))
        self.assertEqual(public_frame_payload(frame), public_frame_payload(mutated))
        self.assertNotEqual(private_frame_payload(frame), private_frame_payload(mutated))
        self.assertNotEqual(private_feature_items(frame), private_feature_items(mutated))


if __name__ == "__main__":
    unittest.main()
