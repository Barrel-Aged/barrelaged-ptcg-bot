from dataclasses import replace
import unittest

from rocketbot.demo import build_demo_state
from rocketbot.state import (
    ActionKind,
    GameAction,
    GameState,
    MatchTrace,
    PlayerId,
    TurnPhase,
    TurnRecord,
)


class StateAndHistoryTests(unittest.TestCase):
    def test_game_state_json_is_stable_and_roundtrips(self) -> None:
        state = build_demo_state()
        encoded = state.to_json()
        restored = GameState.from_json(encoded)
        self.assertEqual(restored, state)
        self.assertEqual(restored.to_json(), encoded)
        self.assertEqual(restored.digest, state.digest)

    def test_match_trace_chains_state_digests_and_roundtrips(self) -> None:
        state = build_demo_state()
        next_state = replace(
            state,
            turn_number=state.turn_number + 1,
            acting_player=PlayerId.OPPONENT,
            phase=TurnPhase.DRAW,
        )
        action = GameAction(
            sequence_index=0,
            actor=PlayerId.SELF,
            kind=ActionKind.END_TURN,
            payload=(("reason", "demo"),),
        )
        record = TurnRecord(
            turn_number=state.turn_number,
            acting_player=PlayerId.SELF,
            state_before_digest=state.digest,
            actions=(action,),
            state_after_digest=next_state.digest,
        )
        trace = MatchTrace(state, state).append_turn(record, next_state)
        restored = MatchTrace.from_json(trace.to_json())
        self.assertEqual(restored, trace)
        self.assertEqual(restored.current_state, next_state)

    def test_trace_rejects_a_broken_digest_link(self) -> None:
        state = build_demo_state()
        broken = TurnRecord(
            turn_number=state.turn_number,
            acting_player=PlayerId.SELF,
            state_before_digest="not-the-state-digest",
            actions=(),
            state_after_digest=state.digest,
        )
        with self.assertRaises(ValueError):
            MatchTrace(state, state, (broken,))


if __name__ == "__main__":
    unittest.main()
