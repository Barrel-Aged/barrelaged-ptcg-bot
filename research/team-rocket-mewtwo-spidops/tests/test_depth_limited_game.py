from __future__ import annotations

from collections import defaultdict
import random
import unittest

from rocketbot.depth_limited_game import (
    DepthLimitedInformationSetGame,
    ExactProfileLeafValueProvider,
    LeafValueEstimate,
)
from rocketbot.exact_game import RocketPrizeDuel, analytic_equilibrium_profile
from rocketbot.game_theory import InformationSetActor
from rocketbot.outcome_sampling import OutcomeSamplingConfig, OutcomeSamplingTraversal
from rocketbot.outcome_sampling_reference import (
    audit_outcome_sampling_expectation,
)


def _enumerated_wrapper_trajectories(
    game: DepthLimitedInformationSetGame,
    profile,
    *,
    traverser: int,
    exploration: float,
):
    """Enumerate sampling paths using the wrapper's own state identifiers.

    Forced chance outcomes are keyed by state ID.  A depth-limited wrapper has
    intentionally different IDs from its source game, so reusing source-game
    paths would silently leave chance sampling uncontrolled.
    """

    result = []

    def visit(state, probability, forced_chance, forced_actions):
        if state.terminal:
            result.append((probability, forced_chance, forced_actions))
            return
        if state.actor is InformationSetActor.CHANCE:
            for outcome in game.chance_outcomes(state):
                next_chance = dict(forced_chance)
                next_chance[state.state_id] = outcome.outcome_id
                visit(
                    game.transition(state, None, chance_outcome=outcome),
                    probability * outcome.probability,
                    next_chance,
                    forced_actions,
                )
            return
        player = game.player_to_act(state)
        assert player is not None
        information_set = game.information_set(state, player=player)
        current = profile.strategy(information_set)
        uniform = 1.0 / len(information_set.legal_actions)
        for action in information_set.legal_actions:
            current_probability = current.probability(action.action_id)
            sample_probability = (
                (1.0 - exploration) * current_probability + exploration * uniform
                if player == traverser
                else current_probability
            )
            if sample_probability <= 0:
                continue
            previous = forced_actions.get(information_set.info_set_id)
            if previous is not None and previous != action.action_id:
                raise RuntimeError("one wrapper path revisits an information set")
            next_actions = dict(forced_actions)
            next_actions[information_set.info_set_id] = action.action_id
            visit(
                game.transition(state, action),
                probability * sample_probability,
                forced_chance,
                next_actions,
            )

    visit(game.root_state(), 1.0, {}, {})
    assert abs(sum(item[0] for item in result) - 1.0) < 1e-12
    return result


class DepthLimitedGameTests(unittest.TestCase):
    def test_exact_leaf_preserves_pre_cutoff_regret_expectations(self) -> None:
        game = RocketPrizeDuel()
        profile = analytic_equilibrium_profile(game)
        full_audit = audit_outcome_sampling_expectation(game, profile)
        exact = {
            (item.traverser, item.information_set_id, item.action_id): (
                item.exact_immediate_counterfactual_regret
            )
            for item in full_audit.rows
        }
        wrapper = DepthLimitedInformationSetGame(
            game,
            ExactProfileLeafValueProvider(game, profile),
            maximum_source_depth_delta=3,
        )
        traversal = OutcomeSamplingTraversal(
            OutcomeSamplingConfig(
                exploration=full_audit.exploration,
                maximum_transitions=32,
                maximum_depth=32,
            )
        )
        weighted: dict[tuple[int, str, str], float] = defaultdict(float)
        for traverser in (0, 1):
            for probability, forced_chance, forced_actions in (
                _enumerated_wrapper_trajectories(
                wrapper,
                profile,
                traverser=traverser,
                exploration=full_audit.exploration,
                )
            ):
                result = traversal.run(
                    wrapper,
                    lambda information_set, belief: profile.strategy(information_set),
                    traverser=traverser,
                    iteration=1,
                    rng=random.Random(0),
                    forced_chance=forced_chance,
                    forced_actions=forced_actions,
                )
                self.assertFalse(result.source_official_engine_executed)
                for sample in result.regret_samples:
                    for action_id, regret in sample.sampled_regrets:
                        weighted[(traverser, sample.information_set_id, action_id)] += (
                            probability * regret
                        )
        self.assertGreaterEqual(len(weighted), 4)
        maximum_error = max(
            abs(value - exact[key]) for key, value in weighted.items()
        )
        self.assertLess(maximum_error, 1e-12)
        self.assertTrue(wrapper.leaf_estimates)
        self.assertTrue(all(item.calibrated for item in wrapper.leaf_estimates))
        self.assertTrue(
            all(item.lower == item.value == item.upper for item in wrapper.leaf_estimates)
        )
        self.assertFalse(wrapper.provenance.official_engine_executed)

    def test_cutoff_leaf_cannot_claim_official_terminal_execution(self) -> None:
        with self.assertRaises(ValueError):
            LeafValueEstimate(
                player=0,
                value=0.0,
                lower=-1.0,
                upper=1.0,
                confidence=0.0,
                source="invalid test leaf",
                calibrated=False,
                official_terminal_result=True,
            )


if __name__ == "__main__":
    unittest.main()
