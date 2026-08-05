from __future__ import annotations

import numpy as np
import torch
import unittest

from scripts.train_n14_localized_contextual_residual import (
    cached_logits,
    configure_adapter,
)
from scripts.train_n12_counterfactual_margin_adapter import N12Example


class TinyResidualPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter_rank = 16
        self.logit_adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(384),
            torch.nn.Linear(384, 16, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(16, 1, bias=False),
        )
        torch.nn.init.zeros_(self.logit_adapter[-1].weight)
        self.frozen = torch.nn.Linear(3, 3)


class N14ResidualTests(unittest.TestCase):
    def test_zero_adapter_preserves_cached_logits(self) -> None:
        model = TinyResidualPolicy()
        example = N12Example(feat={"opt_feats": np.zeros((3, 1), dtype=np.float32)}, action=0, game="x", won=False, group="g")
        example.contextual_options = np.random.default_rng(4).normal(size=(3, 384)).astype(np.float32)
        example.reference_logits = np.asarray([1.0, 0.25, -0.5], dtype=np.float32)
        logits = cached_logits(model.logit_adapter, [example], torch.device("cpu"))
        self.assertTrue(torch.equal(logits[0], torch.tensor([1.0, 0.25, -0.5])))

    def test_configure_adapter_freezes_every_original_parameter(self) -> None:
        model = TinyResidualPolicy()
        trainable = configure_adapter(model)
        self.assertEqual(sum(parameter.numel() for parameter in trainable), 6_928)
        self.assertTrue(all(parameter.requires_grad for parameter in model.logit_adapter.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.frozen.parameters()))


if __name__ == "__main__":
    unittest.main()

