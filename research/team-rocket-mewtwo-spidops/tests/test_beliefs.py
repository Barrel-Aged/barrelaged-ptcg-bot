import unittest

from rocketbot.beliefs import (
    OPPONENT_INTENTS,
    RESOURCE_CATEGORIES,
    HiddenResourcePosterior,
    IntentTemplateLineGenerator,
    InversePlanningBeliefUpdater,
    MinimumEnablingHypothesisGenerator,
    OpponentBeliefState,
    OpponentPlanPosterior,
    OpponentStyle,
    ResourceCategory,
    ResourceVector,
)
from rocketbot.demo import build_demo_state
from rocketbot.observation import AgentObservation


class BeliefRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = AgentObservation.from_state(build_demo_state())
        self.plan = OpponentPlanPosterior.uniform()
        self.lines = IntentTemplateLineGenerator().propose(self.observation, self.plan)

    def test_plan_posterior_is_normalized_serializable_and_updatable(self) -> None:
        self.assertAlmostEqual(sum(self.plan.probabilities), 1.0)
        updated = self.plan.update({OPPONENT_INTENTS[0]: 8.0})
        self.assertGreater(
            updated.probability(OPPONENT_INTENTS[0]),
            self.plan.probability(OPPONENT_INTENTS[0]),
        )
        self.assertEqual(
            OpponentPlanPosterior.from_dict(updated.to_dict()),
            updated,
        )

    def test_minimum_enabling_beam_preserves_behavior_alternatives(self) -> None:
        posterior = MinimumEnablingHypothesisGenerator().generate(self.lines, self.plan)
        styles = {hypothesis.style for hypothesis in posterior.hypotheses}
        self.assertEqual(
            styles,
            {
                OpponentStyle.RATIONAL,
                OpponentStyle.IMPERFECT,
                OpponentStyle.DECEPTIVE,
            },
        )
        self.assertAlmostEqual(sum(h.probability for h in posterior.hypotheses), 1.0)
        self.assertEqual(HiddenResourcePosterior.from_json(posterior.to_json()), posterior)

    def test_observed_setup_actions_reweight_resources_and_intent(self) -> None:
        prior = MinimumEnablingHypothesisGenerator().generate(self.lines, self.plan)
        setup_line = next(line for line in self.lines if line.intent.value == "setup")
        category_index = RESOURCE_CATEGORIES.index(ResourceCategory.BASIC_POKEMON)
        before = prior.marginal_presence[category_index]
        update = InversePlanningBeliefUpdater().update(
            prior,
            self.lines,
            setup_line.action_signature,
        )
        after = update.hidden_resources.marginal_presence[category_index]
        self.assertGreater(after, before)
        self.assertEqual(update.plan.most_likely.value, "setup")

        revealed = update.hidden_resources.condition_on_reveal(
            ResourceCategory.BASIC_POKEMON,
            present=True,
        )
        self.assertGreater(revealed.marginal_presence[category_index], after)

    def test_neural_and_symbolic_beliefs_fuse_and_roundtrip(self) -> None:
        symbolic = MinimumEnablingHypothesisGenerator().generate(self.lines, self.plan)
        state = OpponentBeliefState(
            match_id="belief-test",
            observed_through_turn=5,
            plan=self.plan,
            resource_probabilities=(0.5,) * len(RESOURCE_CATEGORIES),
        )
        fused = state.fuse_symbolic(symbolic, symbolic_weight=0.75)
        self.assertIsNotNone(fused.hidden_hypotheses)
        self.assertEqual(OpponentBeliefState.from_json(fused.to_json()), fused)
        self.assertEqual(len(fused.context_vector), len(OPPONENT_INTENTS) + len(RESOURCE_CATEGORIES))

    def test_resource_vector_rejects_fractional_counts(self) -> None:
        with self.assertRaises(ValueError):
            ResourceVector((0,) * (len(RESOURCE_CATEGORIES) - 1) + (0.5,))


if __name__ == "__main__":
    unittest.main()
