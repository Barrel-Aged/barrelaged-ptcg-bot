import unittest

import torch

from rocketbot.distillation import (
    PLAN_DESCRIPTOR_NAMES,
    DistillationCheckpointManifest,
    DistillationLossWeights,
    PlanValuePolicyHead,
    plan_descriptor,
)
from rocketbot.plans import TeamRocketPlanProposer
from rocketbot.search_learning import stable_fingerprint


class DistillationPrimitiveTests(unittest.TestCase):
    def test_plan_policy_shapes_and_gradients(self) -> None:
        torch.manual_seed(9)
        head = PlanValuePolicyHead(12, 8, hidden_dim=16)
        state = torch.randn(2, 12)
        plan = torch.randn(2, 8)
        descriptors = torch.randn(2, 3, len(PLAN_DESCRIPTOR_NAMES))
        output = head(state, plan, descriptors)
        self.assertEqual(output.expected_value.shape, (2, 3))
        self.assertEqual(output.search_score.shape, (2, 3))
        self.assertEqual(output.uncertainty.shape, (2, 3))
        self.assertTrue(torch.all(output.uncertainty >= 0.0).item())
        loss = (
            output.expected_value.square().mean()
            + output.search_score.square().mean()
            + output.uncertainty.mean()
        )
        loss.backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in head.parameters())
        )

    def test_plan_descriptor_has_public_fixed_schema(self) -> None:
        descriptor = plan_descriptor(TeamRocketPlanProposer._templates()[0])
        self.assertEqual(len(descriptor), len(PLAN_DESCRIPTOR_NAMES))
        self.assertTrue(all(torch.isfinite(torch.tensor(descriptor)).tolist()))

    def test_checkpoint_manifest_is_fingerprinted_and_split_safe(self) -> None:
        fingerprint = stable_fingerprint("artifact")
        manifest = DistillationCheckpointManifest(
            dataset_fingerprint=fingerprint,
            deck_fingerprint=fingerprint,
            environment_fingerprints=(fingerprint,),
            teacher_model_fingerprints=(fingerprint,),
            student_model_fingerprint=fingerprint,
            search_config_fingerprint=fingerprint,
            training_seed=11,
            optimizer_steps=20,
            initial_loss=2.0,
            final_loss=0.2,
            training_episode_ids=("train",),
            heldout_episode_ids=("test",),
        )
        self.assertEqual(
            DistillationCheckpointManifest.from_json(manifest.to_json()),
            manifest,
        )
        with self.assertRaises(ValueError):
            DistillationCheckpointManifest(
                dataset_fingerprint=fingerprint,
                deck_fingerprint=fingerprint,
                environment_fingerprints=(fingerprint,),
                teacher_model_fingerprints=(fingerprint,),
                student_model_fingerprint=fingerprint,
                search_config_fingerprint=fingerprint,
                training_seed=11,
                optimizer_steps=20,
                initial_loss=2.0,
                final_loss=0.2,
                training_episode_ids=("same",),
                heldout_episode_ids=("same",),
            )

    def test_distillation_weights_reject_all_zero(self) -> None:
        with self.assertRaises(ValueError):
            DistillationLossWeights(
                tactical_policy=0.0,
                plan_memory=0.0,
                belief_plan=0.0,
                belief_resources=0.0,
                strategic_values=0.0,
                meta_value=0.0,
                uncertainty=0.0,
                plan_value=0.0,
                plan_score=0.0,
                plan_uncertainty=0.0,
                counterfactual_ranking=0.0,
            )


if __name__ == "__main__":
    unittest.main()
