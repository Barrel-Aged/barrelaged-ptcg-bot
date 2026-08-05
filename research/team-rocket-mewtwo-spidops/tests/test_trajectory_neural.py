from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from rocketbot import (
    OfficialTrajectoryDataset,
    TrajectoryBootstrapArtifact,
    restore_logged_trajectory_bootstrap,
    run_phase6_trajectory_neural_bootstrap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrajectoryNeuralBootstrapTests(unittest.TestCase):
    def test_phase6i_is_split_safe_restorable_and_explicitly_non_equilibrium(self) -> None:
        artifacts = run_phase6_trajectory_neural_bootstrap(project_root=PROJECT_ROOT)
        evidence = artifacts.evidence
        artifact = artifacts.artifact
        self.assertEqual(
            evidence.status,
            "phase6i_logged_neural_bootstrap_validated_partial",
        )
        self.assertEqual(
            (evidence.train_steps, evidence.calibration_steps, evidence.validation_steps),
            (385, 86, 139),
        )
        self.assertEqual(evidence.split_match_overlap_count, 0)
        self.assertTrue(evidence.normalizers_fit_train_only)
        self.assertTrue(evidence.required_head_gradients_present)
        self.assertLess(evidence.final_training_loss, evidence.initial_training_loss)
        self.assertTrue(evidence.roundtrip.artifact_serialization_exact)
        self.assertTrue(evidence.roundtrip.legal_support_exact)
        self.assertEqual(
            max(
                evidence.roundtrip.maximum_action_probability_delta,
                evidence.roundtrip.maximum_outcome_value_delta,
                evidence.roundtrip.maximum_range_probability_delta,
                evidence.roundtrip.maximum_next_public_feature_delta,
            ),
            0.0,
        )
        self.assertFalse(artifact.model_official_engine_executed)
        self.assertFalse(artifact.equilibrium_trained)
        self.assertFalse(artifact.deployable_blueprint)
        self.assertFalse(artifact.counterfactual_value_provider)
        self.assertEqual(
            dict(evidence.claims)["behavior_policy_target"],
            "preserved_heuristic_imitation_not_final_strategy",
        )

        restored_artifact = TrajectoryBootstrapArtifact.from_dict(
            json.loads(artifact.to_json())
        )
        restored = restore_logged_trajectory_bootstrap(restored_artifact)
        dataset = OfficialTrajectoryDataset.from_dict(
            json.loads(
                (
                    PROJECT_ROOT
                    / "artifacts"
                    / "phase6_official_trajectory"
                    / "official_trajectory_dataset.json"
                ).read_text(encoding="utf-8")
            )
        )
        step = dataset.steps[0]
        prediction = restored.predict_step(step)
        self.assertEqual(
            tuple(name for name, _ in prediction.action_probabilities),
            tuple(action.action_id for action in step.information_set.legal_actions),
        )
        self.assertAlmostEqual(
            sum(value for _, value in prediction.action_probabilities),
            1.0,
        )
        public_range = restored.predict_public_range(
            step.public_belief.public_features
        )
        self.assertEqual(public_range, prediction.opponent_range)
        self.assertAlmostEqual(sum(value for _, value in public_range), 1.0)
        self.assertFalse(hasattr(restored, "counterfactual_values"))
        self.assertTrue(restored.claim_boundary["behavior_imitation_only"])

        with self.assertRaises(ValueError):
            replace(artifact, deployable_blueprint=True)


if __name__ == "__main__":
    unittest.main()
