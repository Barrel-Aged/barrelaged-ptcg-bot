from __future__ import annotations

import unittest

import torch

from scripts.n24_native_iql import (
    binary_auc,
    conservative_loss,
    expectile_loss,
    fit_logistic_calibration,
    select_frozen_threshold,
)


class N24NativeIQLTest(unittest.TestCase):
    def test_expectile_penalizes_underestimate_more_at_upper_expectile(self) -> None:
        prediction = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, -1.0])
        weight = torch.ones(2)
        loss = expectile_loss(
            prediction, target, expectile=0.70, weight=weight
        )
        self.assertAlmostEqual(float(loss), 0.5, places=6)
        positive_only = expectile_loss(
            prediction[:1], target[:1], expectile=0.70, weight=weight[:1]
        )
        negative_only = expectile_loss(
            prediction[:1], target[1:], expectile=0.70, weight=weight[:1]
        )
        self.assertGreater(float(positive_only), float(negative_only))

    def test_conservative_loss_is_zero_for_single_available_action(self) -> None:
        values = torch.tensor([[0.25, -0.50]])
        action = torch.tensor([0])
        available = torch.tensor([[True, False]])
        loss = conservative_loss(
            values,
            action,
            available,
            torch.ones(1),
            temperature=0.25,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_threshold_targets_population_rate_and_respects_floor(self) -> None:
        scores = torch.linspace(0.001, 1.0, 1000)
        eligible = torch.ones(1000, dtype=torch.bool)
        threshold = select_frozen_threshold(
            scores, eligible, population=1000, target_rate=0.005
        )
        self.assertGreaterEqual(threshold, 0.05)
        self.assertEqual(int((scores >= threshold).sum()), 5)

    def test_binary_auc_handles_ties(self) -> None:
        self.assertAlmostEqual(binary_auc([0.0, 0.0], [0, 1]), 0.5)
        self.assertAlmostEqual(binary_auc([0.0, 1.0], [0, 1]), 1.0)

    def test_train_only_logistic_calibration_tracks_ordered_labels(self) -> None:
        scores = torch.tensor([-1.0, -0.5, 0.5, 1.0])
        labels = torch.tensor([0, 0, 1, 1])
        scale, bias = fit_logistic_calibration(scores, labels)
        self.assertGreater(scale, 0.0)
        probability = torch.sigmoid(scale * scores + bias)
        self.assertLess(float(probability[:2].max()), 0.5)
        self.assertGreater(float(probability[2:].min()), 0.5)


if __name__ == "__main__":
    unittest.main()
