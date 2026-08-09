from __future__ import annotations

import copy
import unittest

import numpy as np

from rocketbot.policy_feature_snapshot import (
    deserialize_policy_feature,
    serialize_policy_feature,
)


class PolicyFeatureSnapshotTests(unittest.TestCase):
    def test_round_trip_preserves_values_shapes_and_dtypes(self) -> None:
        feature = {
            "global": np.asarray([0.25, 1.0], dtype=np.float32),
            "opt_card": np.asarray([4, 9, 0], dtype=np.int64),
            "opt_feats": np.asarray([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        }
        restored = deserialize_policy_feature(serialize_policy_feature(feature))
        self.assertEqual(set(restored), set(feature))
        for key in feature:
            self.assertEqual(restored[key].dtype, feature[key].dtype)
            np.testing.assert_array_equal(restored[key], feature[key])

    def test_rejects_post_snapshot_mutation(self) -> None:
        payload = serialize_policy_feature(
            {"global": np.asarray([0.25, 1.0], dtype=np.float32)}
        )
        changed = copy.deepcopy(payload)
        changed["tensors"]["global"]["values"][0] = 0.5
        with self.assertRaises(ValueError):
            deserialize_policy_feature(changed)


if __name__ == "__main__":
    unittest.main()
