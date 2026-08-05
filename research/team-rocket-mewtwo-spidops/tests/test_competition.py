import json
import unittest

from rocketbot.competition import (
    AblationMode,
    CompetitionAgentConfig,
    EXACT_DECK_COUNTS,
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    Phase6CompetitionAgent,
    SelectionContract,
    UnsupportedCardPoolError,
    resolve_official_deck,
)
from rocketbot.deck import load_default_deck


def observation(*, options, context=0, opponent_id=723, opponent_hp=150):
    return {
        "select": {
            "type": 0,
            "context": context,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "current": {
            "turn": 5,
            "turnActionCount": 2,
            "yourIndex": 0,
            "firstPlayer": 0,
            "result": -1,
            "stadium": [],
            "looking": None,
            "players": [
                {
                    "active": [
                        {
                            "id": 431,
                            "hp": 280,
                            "maxHp": 280,
                            "energies": [5, 5, 0],
                            "energyCards": [{"id": 15}],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [
                        {
                            "id": 401,
                            "hp": 130,
                            "maxHp": 130,
                            "energies": [1],
                            "energyCards": [{"id": 1}],
                            "tools": [],
                            "preEvolution": [{"id": 400}],
                        },
                        {"id": 400, "hp": 50, "maxHp": 50, "energies": [], "energyCards": [], "tools": [], "preEvolution": []},
                        {"id": 414, "hp": 120, "maxHp": 120, "energies": [], "energyCards": [], "tools": [], "preEvolution": []},
                    ],
                    "deckCount": 35,
                    "discard": [],
                    "prize": [None] * 4,
                    "handCount": 5,
                    "hand": [],
                },
                {
                    "active": [
                        {
                            "id": opponent_id,
                            "hp": opponent_hp,
                            "maxHp": opponent_hp,
                            "energies": [],
                            "energyCards": [],
                            "tools": [],
                            "preEvolution": [],
                        }
                    ],
                    "bench": [],
                    "deckCount": 35,
                    "discard": [{"id": opponent_id}],
                    "prize": [None] * 3,
                    "handCount": 4,
                    "hand": None,
                },
            ],
        },
    }


class _SearchStub:
    def __init__(self, bonus):
        self.bonus = bonus
        self.calls = 0

    def advise(self, observation, candidate_indices, *, deadline, root_player):
        self.calls += 1
        return self.bonus


class CompetitionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy = resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION
        )

    def test_exact_deck_fails_closed_and_proxy_is_explicit(self) -> None:
        deck = load_default_deck()
        self.assertEqual(dict(EXACT_DECK_COUNTS), deck.card_counts)
        exact = resolve_official_deck()
        self.assertFalse(exact.playable)
        self.assertEqual(exact.unresolved, ("Prism Tower",))
        self.assertEqual(len(exact.card_ids), 58)
        with self.assertRaises(UnsupportedCardPoolError):
            resolve_official_deck(fail_on_unresolved=True)

        self.assertTrue(self.proxy.playable)
        self.assertFalse(self.proxy.exact)
        self.assertEqual(len(self.proxy.card_ids), 60)
        self.assertEqual(self.proxy.substitutions[0].count, 2)
        self.assertEqual(len(self.proxy.fingerprint), 64)

    def test_selection_contract_rejects_every_illegal_shape(self) -> None:
        contract = SelectionContract(min_count=1, max_count=2, option_count=3)
        self.assertEqual(contract.validate([2, 0]), (2, 0))
        self.assertEqual(contract.fallback(), (0,))
        for invalid in ([], [0, 1, 2], [0, 0], [-1], [3], [True]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    contract.validate(invalid)

    def test_policy_is_legal_persistent_belief_aware_and_serializable(self) -> None:
        agent = Phase6CompetitionAgent(self.proxy)
        self.assertEqual(agent.choose({"select": None}), list(self.proxy.card_ids))
        obs = observation(
            options=[
                {"type": 14},
                {"type": 13, "attackId": 608},
            ]
        )
        self.assertEqual(agent.choose(obs), [1])
        self.assertGreater(agent.memory.archetype_posterior["abomasnow"], 0.5)
        self.assertEqual(agent.memory.fallback_count, 0)
        self.assertEqual(agent.memory.telemetry[-1].strategic.values.__len__(), 6)
        json.dumps(agent.summary())

    def test_search_can_change_a_critical_action_and_remains_auditable(self) -> None:
        advisor = _SearchStub({0: 1000.0, 1: 0.0})
        agent = Phase6CompetitionAgent(
            self.proxy,
            config=CompetitionAgentConfig(
                ablation=AblationMode.FULL,
                decision_budget_ms=100.0,
                search_budget_ms=50.0,
                criticality_threshold=0.0,
            ),
            search_advisor=advisor,
        )
        obs = observation(
            options=[
                {"type": 14},
                {"type": 13, "attackId": 608},
            ]
        )
        self.assertEqual(agent.choose(obs), [0])
        self.assertEqual(advisor.calls, 1)
        telemetry = agent.memory.telemetry[-1]
        self.assertTrue(telemetry.search_used)
        self.assertTrue(telemetry.search_changed_action)
        self.assertEqual(telemetry.policy_selection, (1,))

    def test_search_failure_returns_valid_failsafe(self) -> None:
        class BrokenSearch:
            def advise(self, *args, **kwargs):
                raise RuntimeError("injected failure")

        agent = Phase6CompetitionAgent(
            self.proxy,
            config=CompetitionAgentConfig(
                ablation=AblationMode.FULL,
                decision_budget_ms=100.0,
                search_budget_ms=50.0,
                criticality_threshold=0.0,
            ),
            search_advisor=BrokenSearch(),
        )
        obs = observation(options=[{"type": 14}, {"type": 13, "attackId": 608}])
        self.assertEqual(agent.choose(obs), [0])
        self.assertEqual(agent.memory.fallback_count, 1)
        self.assertTrue(agent.memory.telemetry[-1].fallback_used)


if __name__ == "__main__":
    unittest.main()

