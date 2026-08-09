from __future__ import annotations

import json
import tarfile
from pathlib import Path
import tempfile
import unittest

from scripts.finalize_arena_rank_goal import (
    BOT,
    DECK_SHA256,
    MODEL_SHA256,
    MONITOR_SCHEMA,
    NotReadyError,
    SECONDARY_BOT,
    SECONDARY_MODEL_SHA256,
    TARGETS,
    build_seal_manifest,
    parse_ratings,
    target_admitted_by_monitor,
    validate_seal_manifest,
    validate_current_ratings,
    validate_monitor_status,
    validate_tar_members,
)


def rating_row(
    *, games: int = 300, rank: int = 1, name: str = BOT
) -> dict[str, object]:
    return {
        "rank": rank,
        "name": name,
        "rating": 900.5,
        "wins": games - 100,
        "losses": 100,
        "draws": 0,
        "games": games,
    }


def admitted_status(
    *, name: str = BOT, model_sha256: str = MODEL_SHA256
) -> dict[str, object]:
    row = rating_row(name=name)
    goal = {
        "name": name,
        "present": True,
        "deck_sha256": DECK_SHA256,
        "model_sha256": model_sha256,
        "rating_row": row,
    }
    return {
        "schema": MONITOR_SCHEMA,
        "read_only": True,
        "target_games": 300,
        "qualifying_deck_sha256": DECK_SHA256,
        "goal_met": True,
        "goal_bot": goal,
        "bots": [goal],
        "top": [row],
    }


class FinalArenaRankGoalTests(unittest.TestCase):
    def test_parse_ratings_preserves_full_bot_name_and_own_count(self) -> None:
        rows = parse_ratings(
            "rank bot rating record\n"
            f"1 {BOT} 901.2 (201-99-0)\n"
            "2 TrainingPin - Spark Starmie V20 850.0 (20-10-1)\n"
        )
        self.assertEqual(rows[0]["name"], BOT)
        self.assertEqual(rows[0]["games"], 300)
        self.assertEqual(rows[1]["name"], "TrainingPin - Spark Starmie V20")
        self.assertEqual(rows[1]["games"], 31)

    def test_monitor_gate_accepts_one_exact_admitted_identity(self) -> None:
        self.assertEqual(validate_monitor_status(admitted_status()), rating_row())

    def test_monitor_and_live_gates_accept_secondary_identity_separately(self) -> None:
        status = admitted_status(
            name=SECONDARY_BOT, model_sha256=SECONDARY_MODEL_SHA256
        )
        target = target_admitted_by_monitor(status)
        self.assertEqual(target, TARGETS[SECONDARY_BOT])
        expected = rating_row(name=SECONDARY_BOT)
        self.assertEqual(validate_monitor_status(status, target), expected)
        self.assertEqual(validate_current_ratings([expected], target), expected)

    def test_monitor_gate_rejects_provisional_rank_one(self) -> None:
        status = admitted_status()
        status["goal_met"] = False
        status["goal_bot"] = None
        status["bot"] = rating_row(games=54)
        with self.assertRaises(NotReadyError):
            validate_monitor_status(status)

    def test_monitor_gate_rejects_combined_or_inconsistent_count(self) -> None:
        status = admitted_status()
        status["goal_bot"]["rating_row"] = rating_row(games=250)
        with self.assertRaises(NotReadyError):
            validate_monitor_status(status)

    def test_live_gate_requires_exact_bot_at_rank_one_and_300(self) -> None:
        qualifying = rating_row(games=305)
        self.assertEqual(validate_current_ratings([qualifying]), qualifying)
        with self.assertRaises(NotReadyError):
            validate_current_ratings([rating_row(games=299)])
        with self.assertRaises(NotReadyError):
            validate_current_ratings([rating_row(games=305, rank=2)])

    def test_tar_gate_rejects_duplicate_and_unsafe_members(self) -> None:
        safe = tarfile.TarInfo("main.py")
        safe.size = 0
        validate_tar_members([safe], frozenset({"main.py"}))

        duplicate = tarfile.TarInfo("main.py")
        duplicate.size = 0
        with self.assertRaises(ValueError):
            validate_tar_members([safe, duplicate], frozenset({"main.py"}))

        unsafe = tarfile.TarInfo("../main.py")
        unsafe.size = 0
        with self.assertRaises(ValueError):
            validate_tar_members([unsafe], frozenset({"../main.py"}))

    def test_tar_gate_rejects_links(self) -> None:
        link = tarfile.TarInfo("main.py")
        link.type = tarfile.SYMTYPE
        link.linkname = "elsewhere.py"
        with self.assertRaises(ValueError):
            validate_tar_members([link], frozenset({"main.py"}))

    def test_evidence_seal_detects_tampering_and_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "audit.json"
            evidence.write_text("{}\n", encoding="utf-8")
            manifest = build_seal_manifest(root)
            (root / "seal_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(validate_seal_manifest(root), manifest)

            evidence.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_seal_manifest(root)

            evidence.write_text("{}\n", encoding="utf-8")
            (root / "unexpected.txt").write_text("extra", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_seal_manifest(root)


if __name__ == "__main__":
    unittest.main()
