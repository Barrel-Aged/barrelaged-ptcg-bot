from __future__ import annotations

import unittest

from scripts import train_n13_calibrated_norm2_adapter as n13


class N13CalibrationTests(unittest.TestCase):
    def test_single_locked_calibration(self) -> None:
        args = n13.locked_args()
        self.assertEqual(args.learning_rate, 5.0e-5)
        self.assertEqual(args.steps, 400)
        self.assertEqual(args.gradient_cap, 0.25)
        self.assertEqual(args.seed, 812101)
        self.assertEqual(args.output, n13.OUTPUT)

    def test_calibration_is_exactly_one_hundred_x_n12(self) -> None:
        self.assertAlmostEqual(n13.LEARNING_RATE / 5.0e-7, 100.0)


if __name__ == "__main__":
    unittest.main()
