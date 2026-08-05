import unittest

import torch

from rocketbot.model import Phase2AgentConfig, Phase2BeliefAwareAgent
from rocketbot.phase5_demo import _build_episode_roots
from rocketbot.plans import TeamRocketPlanProposer
from rocketbot.search_gate import SEARCH_GATE_FEATURE_NAMES, SearchGateFeatures
from rocketbot.search_learning import (
    ArtifactProvenance,
    DistillationDisposition,
    OutcomeValidation,
    SearchRetentionTarget,
    TacticalTeacherStep,
    TeacherCandidate,
    encode_strategic_plan,
    model_fingerprint,
    stable_fingerprint,
)


class SearchLearningPrimitiveTests(unittest.TestCase):
    def test_provenance_and_model_fingerprints_are_stable_and_sensitive(self) -> None:
        fingerprint = stable_fingerprint({"environment": "fixture", "version": 1})
        provenance = ArtifactProvenance(
            deck_fingerprint=fingerprint,
            environment_fingerprint=fingerprint,
            teacher_model_fingerprint=fingerprint,
            search_config_fingerprint=fingerprint,
            random_seed=7,
            episode_id="episode-7",
        )
        self.assertEqual(
            ArtifactProvenance.from_dict(provenance.to_dict()),
            provenance,
        )

        torch.manual_seed(4)
        agent = Phase2BeliefAwareAgent(
            Phase2AgentConfig(
                state_embedding_dim=16,
                plan_dim=8,
                strategic_hidden_dim=16,
                meta_hidden_dim=12,
                candidate_feature_dim=16,
                policy_hidden_dim=16,
                history_hidden_dim=12,
                belief_embedding_dim=12,
            )
        )
        before = model_fingerprint(agent)
        self.assertEqual(before, model_fingerprint(agent))
        with torch.no_grad():
            next(agent.parameters()).add_(0.01)
        self.assertNotEqual(before, model_fingerprint(agent))

    def test_tactical_teacher_step_roundtrips_and_rejects_illegal_target(self) -> None:
        root = _build_episode_roots("unit")[0]
        token = root.routine.action_prefix.tokens[0]
        selected = TeacherCandidate(token, (0.0, 1.0), legal=True)
        alternative = TeacherCandidate(token, (1.0, 0.0), legal=True)
        step = TacticalTeacherStep("step-1", 0, (selected, alternative), 0)
        self.assertEqual(TacticalTeacherStep.from_dict(step.to_dict()), step)
        with self.assertRaises(ValueError):
            TacticalTeacherStep(
                "step-1",
                0,
                (TeacherCandidate(token, (0.0, 1.0), legal=False), alternative),
                0,
            )

    def test_plan_encoding_is_fixed_width_and_identity_sensitive(self) -> None:
        plans = TeamRocketPlanProposer._templates()
        first = encode_strategic_plan(plans[0], 16)
        second = encode_strategic_plan(plans[1], 16)
        self.assertEqual(len(first), 16)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in first))
        self.assertNotEqual(first, second)

    def test_retention_target_separates_absorbed_and_hard_roots(self) -> None:
        features = SearchGateFeatures(
            tuple(0.5 for _ in SEARCH_GATE_FEATURE_NAMES)
        )
        example_id = stable_fingerprint("teacher")
        absorbed = SearchRetentionTarget(
            teacher_example_id=example_id,
            disposition=DistillationDisposition.DISTILL_TO_ROUTINE,
            refreshed_features=features,
            distilled_routine_return=0.58,
            expert_return=0.60,
            search_value_cost=0.08,
            confidence=0.95,
        )
        hard = SearchRetentionTarget(
            teacher_example_id=example_id,
            disposition=DistillationDisposition.RETAIN_EXPERT_SEARCH,
            refreshed_features=features,
            distilled_routine_return=0.40,
            expert_return=0.60,
            search_value_cost=0.08,
            confidence=0.95,
        )
        self.assertFalse(absorbed.should_search)
        self.assertTrue(hard.should_search)
        self.assertEqual(SearchRetentionTarget.from_dict(absorbed.to_dict()), absorbed)
        self.assertFalse(absorbed.to_supervision().should_search)
        self.assertTrue(hard.to_supervision().should_search)
        with self.assertRaises(ValueError):
            SearchRetentionTarget(
                teacher_example_id=example_id,
                disposition=DistillationDisposition.DISTILL_TO_ROUTINE,
                refreshed_features=features,
                distilled_routine_return=0.40,
                expert_return=0.60,
                search_value_cost=0.08,
                confidence=0.95,
            )

    def test_outcome_validation_distinguishes_gain_from_search_cost(self) -> None:
        fingerprint = stable_fingerprint("environment")
        evidence = OutcomeValidation(
            source="paired_replay",
            routine_return=0.30,
            search_return=0.46,
            confidence=0.92,
            paired_samples=6,
            terminal_verified=True,
            environment_fingerprint=fingerprint,
        )
        self.assertAlmostEqual(evidence.validated_gain, 0.16)
        self.assertEqual(OutcomeValidation.from_dict(evidence.to_dict()), evidence)


if __name__ == "__main__":
    unittest.main()
