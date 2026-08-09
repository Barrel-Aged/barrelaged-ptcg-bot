from __future__ import annotations

import torch
import unittest

from scripts.train_n15_terminal_consolidation import (
    compare_state_dicts,
    terminal_admission,
)


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("seen", torch.tensor([3]))


def preservation(retention: float, kl: float, groups: tuple[float, float]) -> dict:
    return {
        "top1_retention": retention,
        "mean_forward_kl": kl,
        "by_group": {
            "a": {"top1_retention": groups[0]},
            "b": {"top1_retention": groups[1]},
        },
    }


class N15TerminalTests(unittest.TestCase):
    def test_state_comparison_is_tensor_exact(self) -> None:
        left = Tiny()
        right = Tiny()
        self.assertTrue(compare_state_dicts(left, right)["passes"])
        with torch.no_grad():
            right.weight[0] = torch.nextafter(
                right.weight[0], torch.tensor(float("inf"))
            )
        result = compare_state_dicts(left, right)
        self.assertFalse(result["passes"])
        self.assertEqual(result["unequal_keys"], ["weight"])

    def test_terminal_admission_requires_every_gate(self) -> None:
        fit = {"fitted": 48, "by_cell": {"a": 1, "b": 1, "c": 1, "d": 1}}
        n6 = preservation(0.98, 0.05, (1.0, 1.0))
        n9 = preservation(0.99, 0.02, (0.985, 0.985))
        parity = {"top_action_mismatches": 0, "max_finite_relative_logit_error": 2e-5}
        prefixes = {
            str(step): {"state_equality": {"passes": True}}
            for step in (62, 90, 149, 286)
        }
        serialization = {"passes": True}
        result = terminal_admission(
            fit,
            n6,
            n9,
            parity,
            parity,
            ["logit_adapter.3.weight"],
            prefixes,
            serialization,
        )
        self.assertTrue(result["passes"])
        n9["top1_retention"] = 0.989999
        result = terminal_admission(
            fit,
            n6,
            n9,
            parity,
            parity,
            ["logit_adapter.3.weight"],
            prefixes,
            serialization,
        )
        self.assertFalse(result["passes"])
        self.assertFalse(result["gates"]["n9_retention_ge_0_99"])


if __name__ == "__main__":
    unittest.main()
