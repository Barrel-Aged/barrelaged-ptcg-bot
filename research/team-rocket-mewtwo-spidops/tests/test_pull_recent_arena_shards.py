from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.pull_recent_arena_shards import recent_run_ids, safe_extract_short_names


class RecentArenaShardTests(unittest.TestCase):
    def test_recent_runs_are_newest_first_and_bounded(self) -> None:
        status = {
            "runs": [
                {"run": "one"},
                {"run": "two"},
                {"run": "three"},
                {"run": "four"},
            ]
        }
        self.assertEqual(recent_run_ids(status, 3), ["four", "three", "two"])

    def test_missing_run_ledger_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recent_run_ids({}, 3)

    def test_shard_extraction_uses_short_names_and_preserves_mapping(self) -> None:
        data = b'{"result":{"winner":0}}\n'
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            member = tarfile.TarInfo("very/long/original_replay_name.jsonl")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = safe_extract_short_names(output.getvalue(), root)
            self.assertEqual((root / "game_0000.jsonl").read_bytes(), data)
            self.assertEqual(
                mapping[0]["original_name"], "very/long/original_replay_name.jsonl"
            )
            self.assertEqual(
                json.loads((root / "archive_members.json").read_text()), mapping
            )

    def test_shard_extraction_rejects_unsafe_paths(self) -> None:
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            member = tarfile.TarInfo("../escape.jsonl")
            member.size = 0
            archive.addfile(member, io.BytesIO())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                safe_extract_short_names(output.getvalue(), Path(directory))


if __name__ == "__main__":
    unittest.main()
