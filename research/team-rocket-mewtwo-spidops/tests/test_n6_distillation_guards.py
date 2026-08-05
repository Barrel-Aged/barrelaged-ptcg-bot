from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.train_belief_rollout_distillation_n6 import apply_attack_guard, read_rows


class N6DistillationGuardTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, **overrides: object) -> None:
        row: dict[str, object] = {
            "schema": "rocket-n6-belief-rollout-root/v1",
            "fully_complete": True,
            "complete_pairs": 24,
            "attempted_pairs": 24,
            "errors": [],
            "game_id": 1,
            "root_index": 0,
        }
        row.update(overrides)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        from scripts.train_belief_rollout_distillation_n6 import sha256

        manifest = {
            "schema": "rocket-n6-belief-rollout-manifest/v1",
            "genuine_official_engine_descendant_transitions": True,
            "visualizer_or_realized_hidden_truth_read": False,
            "raw_hidden_allocations_serialized": False,
            "output_sha256": sha256(path),
            "counts": {"roots": 1},
        }
        path.with_suffix(".manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8"
        )

    def test_accepts_fully_complete_error_free_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roots.jsonl"
            self._write(path)
            rows, hashes = read_rows([path])
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                set(hashes),
                {str(path.resolve()), str(path.with_suffix(".manifest.json").resolve())},
            )

    def test_rejects_incomplete_or_errorful_root(self) -> None:
        cases = (
            {"fully_complete": False},
            {"complete_pairs": 23},
            {"errors": ["rollout failed"]},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / f"bad_{index}.jsonl"
                self._write(path, **overrides)
                with self.assertRaises(ValueError):
                    read_rows([path])

    def test_rejects_jsonl_changed_after_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.jsonl"
            self._write(path)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows([path])

    def test_attack_guard_matches_deployment_post_logit_mask(self) -> None:
        logits = torch.tensor([0.25, 0.5, 4.0], dtype=torch.float32)
        observation = {
            "select": {
                "option": [
                    {"type": 8},
                    {"type": 13},
                    {"type": 14},
                ]
            }
        }

        guarded = apply_attack_guard(logits, observation)

        self.assertEqual(float(logits[2]), 4.0, "source logits must remain unchanged")
        self.assertEqual(float(guarded[0]), 0.25)
        self.assertEqual(float(guarded[1]), 0.5)
        self.assertTrue(torch.isneginf(guarded[2]))
        self.assertEqual(int(torch.argmax(guarded)), 1)

    def test_attack_guard_is_noop_without_attack(self) -> None:
        logits = torch.tensor([0.25, 4.0], dtype=torch.float32)
        observation = {"select": {"option": [{"type": 8}, {"type": 14}]}}
        self.assertIs(apply_attack_guard(logits, observation), logits)


if __name__ == "__main__":
    unittest.main()
