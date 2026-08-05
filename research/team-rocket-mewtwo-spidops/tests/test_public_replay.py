import json
import unittest

from rocketbot.public_replay import (
    OfficialReplayConverter,
    PublicReplayDataset,
)


def card(card_id, serial, owner, name):
    return {"id": card_id, "serial": serial, "playerIndex": owner, "name": name}


def pokemon(card_id, serial, owner, name):
    result = card(card_id, serial, owner, name)
    result.update(
        {
            "hp": 100,
            "maxHp": 100,
            "appearThisTurn": False,
            "energyCards": [],
            "tools": [],
            "preEvolution": [],
        }
    )
    return result


def player(index, *, hand, deck, prize, active=()):
    return {
        "active": list(active),
        "bench": [],
        "benchMax": 5,
        "deckCount": len(deck),
        "discard": [],
        "prize": list(prize),
        "handCount": len(hand),
        "hand": list(hand),
        "deck": list(deck),
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def frame(*, actor, players, logs, selected=None):
    return {
        "select": {
            "type": "Main",
            "context": "Main",
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": "End"}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": logs,
        "current": {
            "turn": 1,
            "turnActionCount": 1,
            "yourIndex": actor,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "lookingCount": 0,
            "looking": None,
            "players": players,
        },
        "selected": selected,
        "ver": 1,
    }


class PublicReplayTests(unittest.TestCase):
    def synthetic_replay(self):
        own_hand = [card(100, 10, 0, "own-visible")]
        opponent_hand = [card(200, 99, 1, "OPPONENT-HAND-CANARY")]
        own_deck = [card(101, 11, 0, "OWN-DECK-CANARY")]
        opponent_deck = [card(201, 98, 1, "OPPONENT-DECK-CANARY")]
        own_prize = [card(102, 12, 0, "OWN-PRIZE-CANARY")]
        opponent_prize = [card(202, 97, 1, "OPPONENT-PRIZE-CANARY")]
        draw = {"type": "Draw", "playerIndex": 1, "cardId": 200, "serial": 99}
        first = frame(
            actor=0,
            players=[
                player(0, hand=own_hand, deck=own_deck, prize=own_prize),
                player(1, hand=opponent_hand, deck=opponent_deck, prize=opponent_prize),
            ],
            logs=[draw],
        )
        reveal = {
            "type": "MoveCard",
            "playerIndex": 1,
            "cardId": 200,
            "serial": 99,
            "fromArea": 2,
            "toArea": 4,
        }
        second = frame(
            actor=1,
            players=[
                player(0, hand=own_hand, deck=own_deck, prize=own_prize),
                player(
                    1,
                    hand=[],
                    deck=opponent_deck,
                    prize=opponent_prize,
                    active=[pokemon(200, 99, 1, "now-public")],
                ),
            ],
            logs=[draw, reveal],
            selected=[0],
        )
        return [first, second]

    def test_truth_replay_is_redacted_per_perspective_and_roundtrips(self):
        episodes = OfficialReplayConverter().convert_match(
            self.synthetic_replay(),
            match_id="synthetic-public-redaction",
            engine_sha256="a" * 64,
            card_pool_sha256="b" * 64,
            environment_fingerprint="c" * 64,
            deck_fingerprints=("d" * 64, "e" * 64),
        )
        player_zero, player_one = episodes
        self.assertTrue(player_zero.audit.passed)
        self.assertTrue(player_one.audit.passed)
        self.assertEqual(player_zero.frames[0].players[0].known_hand_card_ids, (100,))
        self.assertEqual(player_zero.frames[0].players[1].known_hand_card_ids, ())
        self.assertEqual(player_one.frames[0].players[1].known_hand_card_ids, (200,))
        self.assertIsNotNone(player_zero.frames[0].selection)
        self.assertIsNone(player_zero.frames[1].selection)
        self.assertIsNotNone(player_one.frames[1].selection)
        # The official visualizer writes a prompt's chosen indices on the next
        # frame. The converter must shift that label back to its source prompt.
        self.assertEqual(player_zero.frames[0].chosen_indices, (0,))
        self.assertIsNone(player_one.frames[1].chosen_indices)
        self.assertEqual(
            player_zero.frames[0].public_history_fingerprint,
            player_one.frames[0].public_history_fingerprint,
        )
        self.assertEqual(
            player_zero.frames[1].public_history_fingerprint,
            player_one.frames[1].public_history_fingerprint,
        )

        opponent_draw = player_zero.frames[0].events[0]
        self.assertTrue(opponent_draw.card_identity_redacted)
        self.assertNotIn("cardId", dict(opponent_draw.fields))
        public_reveal = player_zero.frames[1].events[0]
        self.assertFalse(public_reveal.card_identity_redacted)
        self.assertEqual(dict(public_reveal.fields)["cardId"], 200)
        self.assertEqual(len(player_zero.reveal_targets), 1)
        self.assertEqual(player_zero.reveal_targets[0].revealed_at_frame, 1)
        self.assertEqual(player_zero.reveal_targets[0].may_supervise_prefix_through, 0)

        frame_text = json.dumps(
            [item.to_dict() for item in player_zero.frames], sort_keys=True
        )
        self.assertNotIn("CANARY", frame_text)
        self.assertNotIn('"serial"', frame_text)
        self.assertNotIn('"hand"', frame_text)
        self.assertNotIn('"deck"', frame_text)
        self.assertNotIn('"prize"', frame_text)

        dataset = PublicReplayDataset(episodes)
        restored = PublicReplayDataset.from_dict(json.loads(dataset.to_json()))
        self.assertEqual(restored.fingerprint, dataset.fingerprint)
        self.assertTrue(restored.validated)
        self.assertFalse(dataset.summary["cross_perspective_join_allowed"])
        self.assertEqual(dataset.summary["schema_version"], 2)

    def test_phase6ag_converter_extensions_are_opt_in_and_causal(self):
        default = OfficialReplayConverter()
        self.assertEqual(
            default.fingerprint,
            "a6b863b88400b5eb1e66cc3ade167a9036be455208303704bc1fcdadb54ca308",
        )
        enhanced = OfficialReplayConverter(
            extra_known_event_types=("Coin",),
            extra_public_scalar_keys=("head",),
            include_coarse_main_action_events=True,
        )
        self.assertNotEqual(enhanced.fingerprint, default.fingerprint)

        replay = self.synthetic_replay()
        coin = {"type": "Coin", "playerIndex": 0, "head": True}
        replay[0]["logs"].append(coin)
        replay[1]["logs"].insert(1, coin)
        player_zero, player_one = enhanced.convert_match(
            replay,
            match_id="synthetic-phase6ag-history",
            engine_sha256="a" * 64,
            card_pool_sha256="b" * 64,
            environment_fingerprint="c" * 64,
            deck_fingerprints=("d" * 64, "e" * 64),
        )
        for episode in (player_zero, player_one):
            self.assertTrue(episode.audit.passed)
            self.assertEqual(episode.audit.unknown_event_types, ())
            coin_event = next(
                event
                for frame_item in episode.frames
                for event in frame_item.events
                if event.event_type == "Coin"
            )
            self.assertIs(dict(coin_event.fields)["head"], True)
            coarse = next(
                event
                for frame_item in episode.frames
                for event in frame_item.events
                if event.event_type == "CoarseMainAction"
            )
            self.assertEqual(
                dict(coarse.fields),
                {"optionType": "End", "playerIndex": 0},
            )
            self.assertEqual(coarse.card_identity_redacted, False)


if __name__ == "__main__":
    unittest.main()
