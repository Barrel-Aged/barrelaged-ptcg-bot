from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.train_live_outcome_awr import (
    DEFAULT_EXACT_PARITY_REPORT,
    PROJECT_ROOT,
    GameInfo,
    WeightedExample,
    collate_like_live_agent,
    deployment_end_indices,
    deployment_logits,
    exact_parity_exclusions,
    read_game,
    split_games,
)


BOT = "HelbrosRocketSpidopsScaleBCV2"


class LiveOutcomeAwrTests(unittest.TestCase):
    def test_read_game_retains_draw(self) -> None:
        records = [
            {"meta": {"players": [BOT, "AshMarnieGeneralV151"]}},
            {
                "p": 0,
                "action": [0],
                "obs": {
                    "current": {"firstPlayer": 1},
                    "select": {"option": [{"type": 1}], "minCount": 1, "maxCount": 1},
                },
            },
            {"result": {"winner": None}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draw.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            game, decisions = read_game(path, BOT)

        self.assertTrue(game.draw)
        self.assertFalse(game.won)
        self.assertFalse(game.started_first)
        self.assertEqual(game.family, "marnie")
        self.assertEqual(len(decisions), 1)

    def test_chronological_split_uses_filename_order(self) -> None:
        games = [
            GameInfo(Path(f"g{index:02d}.jsonl"), 1, False, "Marnie", "marnie", index % 2 == 0, False)
            for index in reversed(range(20))
        ]
        split = split_games(
            games,
            7,
            mode="chronological",
            train_fraction=0.70,
            validation_fraction=0.15,
        )
        self.assertEqual([game.path.name for game in split["train"]], [f"g{i:02d}.jsonl" for i in range(14)])
        self.assertEqual([game.path.name for game in split["validation"]], ["g14.jsonl", "g15.jsonl", "g16.jsonl"])
        self.assertEqual([game.path.name for game in split["test"]], ["g17.jsonl", "g18.jsonl", "g19.jsonl"])

    def test_deployment_guard_masks_end_after_logits(self) -> None:
        observation = {
            "select": {
                "option": [{"type": 8}, {"type": 13}, {"type": 14}],
                "minCount": 1,
                "maxCount": 1,
            }
        }
        example = WeightedExample(
            {},
            1,
            "game",
            True,
            "group",
            deployment_end_indices=deployment_end_indices(observation),
        )
        source = torch.tensor([[0.1, 0.2, 9.0]])
        guarded = deployment_logits(source, torch.ones_like(source), [example])
        self.assertEqual(float(source[0, 2]), 9.0)
        self.assertLess(float(guarded[0, 2]), -1.0e8)
        self.assertEqual(int(guarded.argmax(dim=1)[0]), 1)

    def test_deployment_guard_does_not_change_multiselect(self) -> None:
        observation = {
            "select": {
                "option": [{"type": 13}, {"type": 14}],
                "minCount": 2,
                "maxCount": 2,
            }
        }
        self.assertEqual(deployment_end_indices(observation), ())

    def test_live_collation_separates_availability_from_token_presence(self) -> None:
        examples = [
            WeightedExample({"opt_feats": torch.zeros((2, 1))}, 1, "a", True, "g"),
            WeightedExample({"opt_feats": torch.zeros((3, 1))}, 2, "b", False, "g"),
        ]
        historical_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with patch(
            "scripts.train_live_outcome_awr.collate",
            return_value={"opt_mask": historical_mask.clone()},
        ):
            batch, availability = collate_like_live_agent(examples, torch.device("cpu"))

        self.assertTrue(torch.equal(availability, historical_mask))
        self.assertTrue(
            torch.equal(
                batch["opt_mask"],
                torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
            )
        )

    def test_frozen_exact_parity_manifest(self) -> None:
        candidate = (
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "leader_bc_scale_v2"
        ).resolve()
        replays = (PROJECT_ROOT / "work" / "arena_scale_v2_all_20260728").resolve()
        exclusions, metadata = exact_parity_exclusions(
            DEFAULT_EXACT_PARITY_REPORT,
            candidate=candidate,
            replays=replays,
            recorded_bot=BOT,
            family_filter="marnie",
            started="second",
        )
        self.assertEqual(len(exclusions), 10)
        self.assertEqual(metadata["eligible_games"], 1458)


if __name__ == "__main__":
    unittest.main()
