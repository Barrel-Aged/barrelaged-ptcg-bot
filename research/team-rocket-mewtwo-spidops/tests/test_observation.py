from dataclasses import replace
import unittest

from rocketbot.demo import build_demo_state
from rocketbot.observation import (
    ACTION_HISTORY_FEATURE_NAMES,
    ActionHistoryTensorizer,
    AgentObservation,
    ObservedAction,
)
from rocketbot.state import ActionKind, CardInstance, GameAction, PlayerId


class AgentObservationTests(unittest.TestCase):
    def test_private_card_identities_are_redacted_without_losing_counts(self) -> None:
        truth = build_demo_state()
        opponent = truth.player(PlayerId.OPPONENT)
        named_secret = CardInstance(
            instance_id=opponent.hand[0].instance_id,
            card_name="Secret Opponent Combo Card",
            owner=PlayerId.OPPONENT,
            revealed=False,
        )
        opponent_truth = replace(
            opponent,
            hand=(named_secret,) + opponent.hand[1:],
        )
        truth = replace(
            truth,
            players=(truth.player(PlayerId.SELF), opponent_truth),
        )
        observation = AgentObservation.from_state(truth)
        observed_opponent = observation.public_state.player(PlayerId.OPPONENT)
        observed_self = observation.public_state.player(PlayerId.SELF)

        self.assertEqual(len(observed_opponent.hand), len(opponent.hand))
        self.assertTrue(all(card.card_name is None for card in observed_opponent.hand))
        self.assertTrue(all(card.card_name is None for card in observed_self.deck))
        self.assertTrue(all(card.card_name is None for card in observed_self.prizes))
        self.assertTrue(all(card.card_name is not None for card in observed_self.hand))
        self.assertEqual(AgentObservation.from_json(observation.to_json()), observation)

    def test_private_opponent_draw_is_removed_from_public_action(self) -> None:
        action = GameAction(
            sequence_index=0,
            actor=PlayerId.OPPONENT,
            kind=ActionKind.DRAW,
            source_instance_id="opponent-secret-card",
            target_instance_id="opponent-hand-slot",
            card_name="Secret Drawn Card",
        )
        observed = ObservedAction.from_game_action(
            action,
            turn_number=4,
            observer=PlayerId.SELF,
        )
        self.assertIsNone(observed.card_name)
        self.assertIsNone(observed.source_instance_id)
        self.assertIsNone(observed.target_instance_id)

    def test_action_history_tensorization_has_explicit_mask(self) -> None:
        state = build_demo_state()
        action = ObservedAction(
            turn_number=2,
            sequence_index=0,
            actor=PlayerId.OPPONENT,
            kind=ActionKind.PLAY_TRAINER,
            card_name="Public Trainer",
        )
        observation = AgentObservation.from_state(state, actions=(action,))
        history, mask = ActionHistoryTensorizer()([observation], max_events=4)
        self.assertEqual(tuple(history.shape), (1, 4, len(ACTION_HISTORY_FEATURE_NAMES)))
        self.assertEqual(mask.tolist(), [[True, False, False, False]])
        self.assertGreater(float(history[0, 0].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
