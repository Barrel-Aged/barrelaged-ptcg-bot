import unittest

from rocketbot.competition import SelectionContract
from rocketbot.official_bridge import (
    HIDDEN_CANARY_CARD_ID,
    OfficialDecisionBridge,
    audit_official_bridge_decision,
    complete_selection_count,
    sanitize_official_public_state,
)


def observation_fixture():
    player = {
        "active": [],
        "bench": [],
        "benchMax": 5,
        "deckCount": 47,
        "discard": [],
        "prize": [None] * 6,
        "handCount": 7,
        "hand": [{"id": 431, "serial": 1, "playerIndex": 0}],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }
    opponent = dict(player)
    opponent["hand"] = None
    return {
        "select": {
            "type": 1,
            "context": 8,
            "minCount": 1,
            "maxCount": 2,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [
                {"type": 3, "area": 2, "index": index, "playerIndex": 0}
                for index in range(3)
            ],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "search_begin_input": None,
        "current": {
            "turn": 2,
            "turnActionCount": 3,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [player, opponent],
        },
    }


class OfficialBridgeTests(unittest.TestCase):
    def test_complete_ordered_selection_roundtrip_and_canary_redaction(self):
        observation = observation_fixture()
        bridge = OfficialDecisionBridge(
            ruleset_fingerprint="1" * 64,
            official_engine_executed=False,
        )
        belief = bridge.public_belief_from_official_observation(observation)
        information_set = bridge.information_set_from_official_observation(
            observation,
            belief,
        )
        contract = SelectionContract.from_observation(observation)
        self.assertEqual(complete_selection_count(contract), 9)
        self.assertEqual(len(information_set.legal_actions), 9)
        strategy = bridge.one_hot_selection_strategy(information_set, [2, 0])
        self.assertEqual(bridge.official_selection(strategy, observation), [2, 0])

        audit = audit_official_bridge_decision(
            bridge,
            observation,
            [2, 0],
            match_id="synthetic-bridge-test",
            decision_index=0,
        )
        self.assertTrue(audit.selection_roundtrip_exact)
        self.assertTrue(audit.every_abstract_action_contract_valid)
        self.assertTrue(audit.hidden_canary_invariant)
        self.assertTrue(audit.own_private_sensitivity)
        self.assertFalse(audit.unknown_option_fields)

        public_text = str(sanitize_official_public_state(observation))
        self.assertNotIn(str(HIDDEN_CANARY_CARD_ID), public_text)
        self.assertNotIn("serial", public_text)


if __name__ == "__main__":
    unittest.main()
