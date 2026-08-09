from __future__ import annotations

import math
import unittest

from rocketbot.n6_teacher import (
    clustered_bootstrap_interval,
    estimate_root,
    redistribution_coefficient,
)


def _row(candidate_values: tuple[float, ...], baseline: float = -1.0) -> dict:
    samples = []
    for particle, candidate in enumerate(candidate_values):
        for coin in range(2):
            samples.append(
                {
                    "particle": particle,
                    "coin_rollout": coin,
                    "values": {"0": baseline, "1": candidate, "2": baseline},
                }
            )
    return {
        "baseline": 0,
        "candidate_indices": [0, 1, 2],
        "paired_samples": samples,
    }


class N6TeacherTests(unittest.TestCase):
    def test_clustered_bootstrap_is_deterministic(self) -> None:
        first = clustered_bootstrap_interval(
            [1.0, 0.5, 1.0, 0.5], seed=7, replicates=500
        )
        second = clustered_bootstrap_interval(
            [1.0, 0.5, 1.0, 0.5], seed=7, replicates=500
        )
        self.assertEqual(first, second)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertGreater(first[0], 0.0)

    def test_estimate_root_admits_replicated_positive_action(self) -> None:
        estimate = estimate_root(
            _row((1.0,) * 12),
            seed=11,
            bootstrap_replicates=500,
        )
        self.assertEqual(estimate.best, 1)
        self.assertEqual(estimate.half_best, (1, 1))
        self.assertTrue(estimate.admissible_change)
        self.assertTrue(math.isclose(estimate.action(1).paired_difference, 2.0))

    def test_estimate_root_rejects_unstable_halves(self) -> None:
        estimate = estimate_root(
            _row((1.0,) * 6 + (-0.5,) * 6, baseline=0.0),
            seed=13,
            bootstrap_replicates=500,
        )
        self.assertEqual(estimate.half_best, (1, 0))
        self.assertLess(estimate.half_paired_differences[1] or 0.0, 0.0)
        self.assertFalse(estimate.admissible_change)

    def test_redistribution_coefficient_is_bounded(self) -> None:
        self.assertEqual(redistribution_coefficient([0.0, 0.5], 12), 0.5)
        self.assertEqual(redistribution_coefficient([-1.0, 1.0], 12), 1.0)
        self.assertEqual(redistribution_coefficient([0.0, 0.5], 3), 0.25)
