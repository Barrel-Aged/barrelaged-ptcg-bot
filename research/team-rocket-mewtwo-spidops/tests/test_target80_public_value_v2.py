from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "work" / "target80_candidates" / "murkrow_public_value_plan_v2"
if str(CANDIDATE) not in sys.path:
    sys.path.insert(0, str(CANDIDATE))

from soldier.public_value import (  # noqa: E402
    extract_public_value_features,
    load_public_value,
)


def _observation():
    return {
        "select": {"type": 0, "minCount": 1, "maxCount": 1, "option": [{"type": 14}]},
        "current": {
            "turn": 5,
            "turnActionCount": 1,
            "yourIndex": 0,
            "firstPlayer": 1,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "stadium": [{"id": 1259, "playerIndex": 1}],
            "players": [
                {
                    "active": [{"id": 401, "hp": 130, "maxHp": 130, "energies": [1, 1], "energyCards": [{"id": 1}, {"id": 1}], "tools": [], "preEvolution": [{"id": 400}]}],
                    "bench": [{"id": 431, "hp": 280, "maxHp": 280, "energies": [], "energyCards": [], "tools": [], "preEvolution": []}],
                    "discard": [{"id": 1216}],
                    "prize": [None] * 5,
                    "handCount": 2,
                    "hand": [{"id": 15}, {"id": 1218}],
                    "deckCount": 38,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
                {
                    "active": [{"id": 648, "hp": 320, "maxHp": 320, "energies": [7, 7], "energyCards": [{"id": 7}, {"id": 7}], "tools": [], "preEvolution": [{"id": 646}, {"id": 647}]}],
                    "bench": [{"id": 112, "hp": 110, "maxHp": 110, "energies": [], "energyCards": [], "tools": [], "preEvolution": []}],
                    "discard": [{"id": 1227}],
                    "prize": [None] * 6,
                    "handCount": 4,
                    "hand": None,
                    "deckCount": 40,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
            ],
        },
    }


def test_deck_and_model_integrity():
    deck = [int(value) for value in (CANDIDATE / "deck.csv").read_text().split()]
    assert len(deck) == 60
    assert deck.count(432) == 0
    assert deck.count(463) == 3
    loaded = load_public_value(CANDIDATE / "public_value.pt")
    assert loaded.mean.shape == loaded.std.shape == (202,)


def test_features_ignore_opponent_hidden_identities_but_use_own_hand():
    observation = _observation()
    opponent_deck = [7] * 10 + [646] * 4 + [647] * 3 + [648] * 3 + [112] * 4 + [1219] * 4 + [1227] * 4 + [1259] * 4 + [1152] * 4 + [1097] * 3 + [1086] * 4 + [104] * 2 + [860] * 2 + [1079] * 3 + [1080, 1122, 1137, 1182, 1231, 1142]
    assert len(opponent_deck) == 60
    original = extract_public_value_features(observation, root_player=0, opponent_deck=opponent_deck)
    hidden_changed = copy.deepcopy(observation)
    hidden_changed["current"]["players"][1]["hand"] = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    hidden_changed["current"]["players"][1]["prize"] = [{"id": 999}] * 6
    assert extract_public_value_features(hidden_changed, root_player=0, opponent_deck=opponent_deck) == original
    own_changed = copy.deepcopy(observation)
    own_changed["current"]["players"][0]["hand"][0]["id"] = 1
    assert extract_public_value_features(own_changed, root_player=0, opponent_deck=opponent_deck) != original


def test_loaded_value_is_finite_probability():
    observation = _observation()
    opponent_deck = [7] * 10 + [646] * 4 + [647] * 3 + [648] * 3 + [112] * 4 + [1219] * 4 + [1227] * 4 + [1259] * 4 + [1152] * 4 + [1097] * 3 + [1086] * 4 + [104] * 2 + [860] * 2 + [1079] * 3 + [1080, 1122, 1137, 1182, 1231, 1142]
    value = load_public_value(CANDIDATE / "public_value.pt").predict(
        observation, root_player=0, opponent_deck=opponent_deck
    )
    assert 0.0 <= value <= 1.0
