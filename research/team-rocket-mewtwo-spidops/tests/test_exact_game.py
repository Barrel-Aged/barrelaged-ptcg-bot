import unittest

from rocketbot.deck import load_default_deck
from rocketbot.exact_game import (
    ANALYTIC_GAME_VALUE_PLAYER_0,
    DuelCard,
    ExactGridSafeResolver,
    ExactTabularBlueprintProvider,
    RocketPrizeDuel,
    TabularCFRSolver,
    TabularStrategyProfile,
    analytic_equilibrium_profile,
    deal_count,
    exact_counterfactual_bounds,
    exact_profile_metrics,
    information_set_action_count,
    terminal_state_count,
    uniform_profile,
)
from rocketbot.game_theory import (
    GameBoundaryKind,
    ResolveBudget,
    SafeResolveRequest,
)


class ExactGameTests(unittest.TestCase):
    def test_exact_game_counts_information_sets_and_boundary(self):
        game = RocketPrizeDuel()
        self.assertEqual(game.provenance.kind, GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME)
        self.assertFalse(game.provenance.official_engine_executed)
        self.assertEqual(len(game.all_states()), 55)
        self.assertEqual(terminal_state_count(game), 30)
        self.assertEqual(len(game.information_sets()), 12)
        self.assertEqual(information_set_action_count(game), 24)
        self.assertEqual(deal_count(game), 6)
        state = next(
            item
            for item in game.all_states()
            if item.cards[0] is DuelCard.SPIDOPS and item.history == ()
        )
        info_set = game.information_set(state, player=0)
        belief = game.public_belief(state, player=0)
        self.assertIsNotNone(info_set.private_observation_fingerprint)
        self.assertEqual(
            {item.hypothesis_id for item in belief.opponent_range.hypotheses},
            {"opponent:mimikyu", "opponent:mewtwo"},
        )

    def test_analytic_equilibrium_and_cfr_have_exact_exploitability_metrics(self):
        game = RocketPrizeDuel()
        uniform = exact_profile_metrics(game, uniform_profile(game))
        analytic_profile = analytic_equilibrium_profile(game)
        analytic = exact_profile_metrics(game, analytic_profile)
        self.assertAlmostEqual(
            analytic.expected_value_player_0,
            ANALYTIC_GAME_VALUE_PLAYER_0,
            places=12,
        )
        self.assertLess(analytic.nash_conv, 1e-12)
        self.assertGreater(uniform.nash_conv, 0.9)
        restored = TabularStrategyProfile.from_dict(analytic_profile.to_dict())
        self.assertEqual(restored.fingerprint, analytic_profile.fingerprint)

        cfr = TabularCFRSolver(game).train(2_000)
        self.assertLess(cfr.metrics.nash_conv, 0.02)
        self.assertLess(cfr.metrics.nash_conv, uniform.nash_conv * 0.03)
        self.assertAlmostEqual(
            cfr.metrics.expected_value_player_0,
            ANALYTIC_GAME_VALUE_PLAYER_0,
            delta=5e-4,
        )

    def test_exact_safe_resolver_improves_without_violating_any_private_hypothesis_bound(self):
        game = RocketPrizeDuel()
        profile = uniform_profile(game)
        state = next(
            item
            for item in game.all_states()
            if item.cards[0] is DuelCard.SPIDOPS and item.history == ()
        )
        info_set = game.information_set(state, player=0)
        belief = game.public_belief(state, player=0)
        provider = ExactTabularBlueprintProvider(
            game,
            profile,
            deck_fingerprint=load_default_deck().fingerprint,
            algorithm="unit_validation_tabular",
            training_steps=1,
        )
        baseline = provider.strategy(info_set, belief)
        bounds = exact_counterfactual_bounds(
            provider,
            info_set,
            belief,
            opponent_player=1,
        )
        result = ExactGridSafeResolver(profile, grid_steps=100).resolve(
            game,
            SafeResolveRequest(
                information_set=info_set,
                belief=belief,
                baseline_strategy=baseline,
                opponent_value_bounds=bounds,
                budget=ResolveBudget(101, 1_000, 1_000.0, 1e-12),
            ),
        )
        self.assertTrue(result.certificate.safe)
        self.assertEqual(result.certificate.max_counterfactual_value_violation, 0.0)
        self.assertEqual(result.strategy.argmax, "push")
        self.assertGreater(result.strategy.probability("push"), baseline.probability("push"))
        self.assertFalse(result.fallback_to_blueprint)
        self.assertEqual(result.expanded_nodes, 202)


if __name__ == "__main__":
    unittest.main()
