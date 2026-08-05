from __future__ import annotations

import unittest

import torch

from scripts.build_n25_complementary_composition import (
    HEAD_KEYS,
    composition_audit,
)


class N25CompositionTest(unittest.TestCase):
    def fixtures(self) -> tuple[dict[str, torch.Tensor], ...]:
        live = {
            "body": torch.tensor([1.0]),
            "value.weight": torch.tensor([0.4]),
            "logit_head.bias": torch.tensor([0.1]),
            "logit_head.weight": torch.tensor([[0.2, 0.3]]),
        }
        control = {key: value.clone() for key, value in live.items()}
        control["value.weight"] += 0.5
        n8b = {key: value.clone() for key, value in control.items()}
        n8b["logit_head.bias"] += 0.02
        n8b["logit_head.weight"] += torch.tensor([[0.04, -0.01]])
        n15 = {key: value.clone() for key, value in live.items()}
        n15["body"] += 2.0
        candidate = {key: value.clone() for key, value in n15.items()}
        for key in HEAD_KEYS:
            candidate[key] = n15[key] + n8b[key] - control[key]
        return live, control, n8b, n15, candidate

    def test_exact_composition_passes(self) -> None:
        self.assertTrue(composition_audit(*self.fixtures())["passes"])

    def test_non_head_drift_fails(self) -> None:
        live, control, n8b, n15, candidate = self.fixtures()
        candidate["body"] += 0.01
        result = composition_audit(live, control, n8b, n15, candidate)
        self.assertFalse(result["passes"])
        self.assertEqual(result["non_head_changes_vs_n15"], ["body"])

    def test_n8b_value_leak_fails(self) -> None:
        live, control, n8b, n15, candidate = self.fixtures()
        n8b["value.weight"] += 0.01
        result = composition_audit(live, control, n8b, n15, candidate)
        self.assertFalse(result["passes"])
        self.assertEqual(result["source_delta_keys"], [
            "logit_head.bias",
            "logit_head.weight",
            "value.weight",
        ])


if __name__ == "__main__":
    unittest.main()
