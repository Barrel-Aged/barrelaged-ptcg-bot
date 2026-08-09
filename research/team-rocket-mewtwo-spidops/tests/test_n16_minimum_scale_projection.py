from __future__ import annotations

import numpy as np
import torch
import unittest

from scripts.build_n16_minimum_scale_projection import (
    constraint_interval,
    direction_audit,
    rounded_scale,
)


class Tiny(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.cfg = {"adapter_scale": 1.0}
        self.keep = torch.nn.Parameter(torch.tensor([2.0]))
        self.logit_adapter = torch.nn.Sequential(
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
            torch.nn.Linear(1, 1, bias=False),
        )
        with torch.no_grad():
            self.logit_adapter[-1].weight.fill_(value)


class N16ProjectionTests(unittest.TestCase):
    def test_constraint_interval_and_upward_rounding(self) -> None:
        # Teacher 1 begins 1.0 behind competitor 0 and gains 2.0 per unit scale.
        result = constraint_interval(
            [np.asarray([1.0, 0.0])],
            [np.asarray([0.0, 2.0])],
            [1],
            margin=0.02,
        )
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["lower"], 0.51)
        self.assertEqual(rounded_scale(result["lower"]), 0.510001)

    def test_negative_slope_creates_upper_bound(self) -> None:
        result = constraint_interval(
            [np.asarray([0.0, 1.0])],
            [np.asarray([0.0, -2.0])],
            [1],
            margin=0.02,
        )
        self.assertTrue(result["feasible"])
        self.assertAlmostEqual(result["upper"], 0.49)

    def test_direction_audit_requires_only_scaled_final_projection(self) -> None:
        terminal = Tiny(4.0)
        candidate = Tiny(2.0)
        result = direction_audit(terminal, candidate, 0.5)
        self.assertTrue(result["passes"])
        self.assertTrue(result["config_exact"])
        with torch.no_grad():
            candidate.keep.add_(1.0)
        self.assertFalse(direction_audit(terminal, candidate, 0.5)["passes"])


if __name__ == "__main__":
    unittest.main()
