import unittest

from rocketbot.counterfactual_calibration import (
    CalibratedNeuralCounterfactualBoundProvider,
    CounterfactualBoundContext,
)
from rocketbot.exact_game import RocketPrizeDuel
from rocketbot.game_theory import ResolveBudget
from rocketbot.neural_blueprint import provider_profile
from rocketbot.phase6_bounds import run_phase6_counterfactual_bounds
from rocketbot.safe_resolving import (
    BudgetAwareExactSafeResolver,
    ExactSafePlanningPipeline,
    ResolveInterruption,
)


class TickingClock:
    def __init__(self, step_seconds):
        self.value = 0.0
        self.step_seconds = step_seconds

    def __call__(self):
        result = self.value
        self.value += self.step_seconds
        return result


class SafeResolvingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        phase6d = run_phase6_counterfactual_bounds()
        cls.game = RocketPrizeDuel()
        cls.blueprint = phase6d.blueprint
        cls.profile = provider_profile(
            cls.game,
            cls.blueprint,
            label="safe_resolving_test_profile",
        )
        cls.bound_provider = CalibratedNeuralCounterfactualBoundProvider(
            cls.game,
            cls.blueprint,
            phase6d.calibration,
        )
        cls.resolver = BudgetAwareExactSafeResolver(cls.profile)
        cls.pipeline = ExactSafePlanningPipeline(
            game=cls.game,
            blueprint=cls.blueprint,
            bound_provider=cls.bound_provider,
            resolver=cls.resolver,
        )
        cls.information_set = next(
            item
            for item in cls.game.information_sets()
            if cls.game.states_for_information_set(item.info_set_id)[0].depth >= 2
        )
        state = cls.game.states_for_information_set(cls.information_set.info_set_id)[0]
        cls.belief = cls.game.public_belief(state, player=state.acting_player)

    @staticmethod
    def full_budget():
        return ResolveBudget(
            iterations=102,
            max_nodes=206,
            time_budget_ms=1_000.0,
            safety_tolerance=1e-12,
        )

    def test_complete_search_is_exactly_verified_and_nonregressing(self):
        decision = self.pipeline.decide(
            self.information_set,
            self.belief,
            budget=self.full_budget(),
        )
        status = decision.resolve_status
        self.assertIsNotNone(status)
        self.assertTrue(status.completed_search)
        self.assertEqual(status.interruption, ResolveInterruption.NONE)
        self.assertEqual(status.required_iterations, 102)
        self.assertEqual(status.required_nodes, 206)
        self.assertTrue(status.resolved.certificate.safe)
        self.assertTrue(status.strict_baseline_nonregression)
        self.assertLessEqual(status.maximum_opponent_baseline_regression, 1e-12)
        decision.strategy.validate_support(self.information_set)

    def test_iteration_node_and_deadline_interruptions_return_exact_blueprint(self):
        for budget, expected in (
            (
                ResolveBudget(101, 206, 1_000.0, 1e-12),
                ResolveInterruption.ITERATION_BUDGET,
            ),
            (
                ResolveBudget(102, 205, 1_000.0, 1e-12),
                ResolveInterruption.NODE_BUDGET,
            ),
        ):
            decision = self.pipeline.decide(
                self.information_set,
                self.belief,
                budget=budget,
            )
            self.assertEqual(decision.strategy, decision.blueprint_strategy)
            self.assertEqual(decision.resolve_status.interruption, expected)
            self.assertTrue(decision.resolve_status.resolved.fallback_to_blueprint)

        deadline_pipeline = ExactSafePlanningPipeline(
            game=self.game,
            blueprint=self.blueprint,
            bound_provider=self.bound_provider,
            resolver=BudgetAwareExactSafeResolver(
                self.profile,
                clock=TickingClock(0.001),
            ),
        )
        deadline = deadline_pipeline.decide(
            self.information_set,
            self.belief,
            budget=ResolveBudget(102, 206, 0.5, 1e-12),
        )
        self.assertEqual(deadline.strategy, deadline.blueprint_strategy)
        self.assertEqual(
            deadline.resolve_status.interruption,
            ResolveInterruption.DEADLINE,
        )

    def test_bound_gate_skips_resolver_and_returns_blueprint(self):
        decision = self.pipeline.decide(
            self.information_set,
            self.belief,
            budget=self.full_budget(),
            bound_context=CounterfactualBoundContext(ood_score=0.51),
        )
        self.assertTrue(decision.fallback_to_blueprint)
        self.assertIsNone(decision.resolve_status)
        self.assertEqual(decision.strategy, decision.blueprint_strategy)


if __name__ == "__main__":
    unittest.main()
