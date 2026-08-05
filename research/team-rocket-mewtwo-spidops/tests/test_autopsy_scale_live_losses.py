from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.autopsy_scale_live_losses import BOT, analyze_game, rate


class ScaleReplayAutopsyTests(unittest.TestCase):
    def test_null_winner_is_retained_as_draw(self) -> None:
        records = [
            {"meta": {"players": [BOT, "Opponent"]}},
            {
                "p": 0,
                "action": [],
                "obs": {
                    "current": {
                        "firstPlayer": 0,
                        "yourIndex": 0,
                        "turn": 1,
                        "players": [
                            {"active": [], "bench": [], "prize": [], "deckCount": 53},
                            {"active": [], "bench": [], "prize": [], "deckCount": 53},
                        ],
                    },
                    "select": {"context": 0, "option": []},
                },
            },
            {"result": {"winner": None, "steps": 1}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draw.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            game = analyze_game(path)

        self.assertFalse(game["won"])
        self.assertFalse(game["lost"])
        self.assertTrue(game["draw"])
        self.assertEqual(
            rate([game]),
            {"games": 1, "wins": 0, "losses": 0, "draws": 1, "win_rate": 0.0},
        )

    def test_custom_bot_identity_is_supported(self) -> None:
        custom_bot = "HelbrosRocketSpidopsScaleStopV3"
        records = [
            {"meta": {"players": ["Opponent", custom_bot]}},
            {
                "p": 1,
                "action": [],
                "obs": {
                    "current": {
                        "firstPlayer": 0,
                        "yourIndex": 1,
                        "turn": 1,
                        "players": [
                            {"active": [], "bench": [], "prize": [], "deckCount": 53},
                            {"active": [], "bench": [], "prize": [], "deckCount": 53},
                        ],
                    },
                    "select": {"context": 0, "option": []},
                },
            },
            {"result": {"winner": 1, "steps": 1}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            game = analyze_game(path, custom_bot)

        self.assertTrue(game["won"])
        self.assertEqual(game["seat"], 1)
        self.assertEqual(game["opponent"], "Opponent")


if __name__ == "__main__":
    unittest.main()
