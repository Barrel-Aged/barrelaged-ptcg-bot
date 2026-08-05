from __future__ import annotations

import json
import unittest

import torch

from scripts import train_n12_counterfactual_margin_adapter as support
from scripts.train_n17_replicated_target_residual import (
    EXPECTED_EXPANDED_CELLS,
    EXPECTED_STRICT_CELLS,
    TARGET_AUDIT,
    admission,
    balanced_cell_weights,
    configure_adapter,
    partition_targets,
)


class TinyResidualPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_rank = 32
        self.logit_adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(384),
            torch.nn.Linear(384, 32, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(32, 1, bias=False),
        )
        torch.nn.init.zeros_(self.logit_adapter[-1].weight)
        self.frozen = torch.nn.Linear(3, 3)


class N17ResidualTests(unittest.TestCase):
    def test_configure_adapter_freezes_every_original_parameter(self) -> None:
        model = TinyResidualPolicy()
        trainable = configure_adapter(model)
        self.assertEqual(sum(parameter.numel() for parameter in trainable), 13_088)
        self.assertTrue(all(parameter.requires_grad for parameter in model.logit_adapter.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.frozen.parameters()))

    def test_frozen_target_partition_and_cell_mass(self) -> None:
        examples, _ = support.load_n6_examples(support.CORPUS, support.AUDIT)
        audit = json.loads(TARGET_AUDIT.read_text(encoding="utf-8"))
        strict, expanded, anchors = partition_targets(examples, audit)
        self.assertEqual((len(strict), len(expanded), len(anchors)), (48, 122, 469))
        strict_weights = balanced_cell_weights(strict, EXPECTED_STRICT_CELLS)
        expanded_weights = balanced_cell_weights(expanded, EXPECTED_EXPANDED_CELLS)
        self.assertAlmostEqual(float(strict_weights.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(expanded_weights.sum()), 1.0, places=6)

    def test_admission_requires_every_expanded_cell(self) -> None:
        strict = {"examples": 48, "fitted": 48}
        expanded = {
            "examples": 122,
            "fitted": 112,
            "by_cell": {
                name: {"fit_rate": 0.90} for name in EXPECTED_EXPANDED_CELLS
            },
        }
        preservation_n6 = {"top1_retention": 0.99, "mean_forward_kl": 0.01}
        preservation_n9 = {
            "top1_retention": 0.995,
            "mean_forward_kl": 0.01,
            "by_group": {
                name: {"top1_retention": 0.99} for name in support.GROUPS
            },
        }
        parity = {"top_action_mismatches": 0, "max_finite_relative_logit_error": 1e-6}
        passed = admission(
            strict,
            expanded,
            preservation_n6,
            preservation_n9,
            parity,
            parity,
            ["logit_adapter.3.weight"],
            {"tensor_exact": True},
            True,
        )
        self.assertTrue(passed["passes"])
        first = next(iter(expanded["by_cell"].values()))
        first["fit_rate"] = 0.84
        failed = admission(
            strict,
            expanded,
            preservation_n6,
            preservation_n9,
            parity,
            parity,
            ["logit_adapter.3.weight"],
            {"tensor_exact": True},
            True,
        )
        self.assertFalse(failed["passes"])
        self.assertFalse(failed["gates"]["expanded_each_cell_fit_ge_0_85"])


if __name__ == "__main__":
    unittest.main()
