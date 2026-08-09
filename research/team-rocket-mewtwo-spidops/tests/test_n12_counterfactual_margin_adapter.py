from __future__ import annotations

import copy
import numpy as np
import torch
import unittest

from scripts.train_n12_counterfactual_margin_adapter import (
    GROUPS,
    N12Example,
    backward_anchor_objective,
    cached_deployed,
    correction_cell_weights,
    forward_kl,
    permitted_changes,
    repeated_order,
    retention_loss,
    smooth_margin_loss,
)


def _example(cell: str) -> N12Example:
    return N12Example(
        feat={},
        action=0,
        game="test",
        won=False,
        group=cell.split(":", 1)[0],
        cell=cell,
    )


class N12CounterfactualMarginTests(unittest.TestCase):
    def test_correction_weights_equalize_four_cells(self) -> None:
        cells = [
            f"{GROUPS[0]}:turn_2",
            f"{GROUPS[0]}:turn_4",
            f"{GROUPS[1]}:turn_2",
            f"{GROUPS[1]}:turn_4",
        ]
        examples = [
            *[_example(cells[0]) for _ in range(2)],
            *[_example(cells[1]) for _ in range(4)],
            *[_example(cells[2]) for _ in range(3)],
            _example(cells[3]),
        ]
        weights = correction_cell_weights(examples).numpy()
        self.assertTrue(np.isclose(weights.sum(), 1.0))
        for cell in cells:
            mass = sum(
                weight for weight, example in zip(weights, examples)
                if example.cell == cell
            )
            self.assertTrue(np.isclose(mass, 0.25))

    def test_smooth_margin_decreases_as_margin_improves(self) -> None:
        target = torch.tensor([0.1])
        poor = smooth_margin_loss(torch.tensor([-1.0]), target, temperature=0.25)
        adequate = smooth_margin_loss(torch.tensor([0.1]), target, temperature=0.25)
        strong = smooth_margin_loss(torch.tensor([1.0]), target, temperature=0.25)
        self.assertGreater(float(poor), float(adequate))
        self.assertGreater(float(adequate), float(strong))

    def test_repeated_order_is_deterministic_and_complete(self) -> None:
        first = repeated_order(7, 30, 812101)
        second = repeated_order(7, 30, 812101)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        self.assertEqual(set(first[:7]), set(range(7)))

    def test_permitted_changes_rejects_empty_or_broad_updates(self) -> None:
        self.assertTrue(
            permitted_changes(["encoder.layers.2.norm2.weight", "logit_head.weight"])
        )
        self.assertFalse(permitted_changes([]))
        self.assertFalse(permitted_changes(["encoder.layers.1.norm1.weight"]))

    def test_anchor_microbatches_match_logical_batch_gradient(self) -> None:
        class TinyCachedPolicy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=4,
                    nhead=2,
                    dim_feedforward=8,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(layer, 1)
                self.logit_head = torch.nn.Linear(4, 1)

        torch.manual_seed(17)
        logical = TinyCachedPolicy().eval()
        micro = copy.deepcopy(logical)
        examples = []
        for index in range(5):
            rng = np.random.default_rng(100 + index)
            example = _example(f"{GROUPS[index % 2]}:turn_2")
            example.feat = {"opt_feats": np.zeros((3, 1), dtype=np.float32)}
            example.cached_tokens = rng.normal(size=(4, 4)).astype(np.float32)
            example.option_start = 1
            example.deployment_available = np.ones(3, dtype=np.bool_)
            example.reference_probabilities = np.asarray([0.6, 0.3, 0.1], dtype=np.float32)
            example.reference_top = 0
            example.reference_margin = 0.2
            examples.append(example)

        logits = cached_deployed(logical, examples, torch.device("cpu"))
        full_loss = 1.3 * forward_kl(logits, examples) + 0.4 * retention_loss(
            logits, examples, cap=0.25, temperature=0.25
        )
        full_loss.backward()
        backward_anchor_objective(
            micro,
            examples,
            kl_coefficient=1.3,
            retention_coefficient=0.4,
            retention_cap=0.25,
            temperature=0.25,
            microbatch_size=2,
        )
        for first, second in zip(logical.parameters(), micro.parameters()):
            if first.grad is None or second.grad is None:
                self.assertIsNone(first.grad)
                self.assertIsNone(second.grad)
            else:
                self.assertTrue(
                    torch.allclose(first.grad, second.grad, atol=2.0e-6, rtol=2.0e-5)
                )
