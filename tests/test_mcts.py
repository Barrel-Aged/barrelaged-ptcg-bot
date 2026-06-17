"""Tests for MctsPolicy (Sprint 6).

Unit tests cover determinization sizing (no engine). The full strength measurement (MCTS beats the
heuristic ~88% head-to-head) is run via the head-to-head harness, not the fast suite.
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CG_AVAILABLE = importlib.util.find_spec("cg") is not None
needs_cg = pytest.mark.skipif(not CG_AVAILABLE, reason="cg engine not on path")


@needs_cg
def test_determinize_sizes_match_state():
    """search_begin requires predicted hidden info to match the real counts exactly."""
    import random

    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    from bot.deck_id.identifier import DeckIdentifier
    from bot.eval.gauntlet import load_gauntlet_decks
    from bot.policies.determinize import determinize
    from bot.policies.heuristic import HeuristicPolicy

    deck = load_gauntlet_decks()["dragapult-ex"]
    hp = HeuristicPolicy()
    ident = DeckIdentifier(your_index=0)
    obs, start = battle_start(deck, deck)
    assert start.errorPlayer < 0
    checks = 0
    while obs["current"]["result"] < 0 and checks < 10:
        typed = to_observation_class(obs)
        ident.observe(typed)
        if obs["current"]["yourIndex"] == 0 and obs["current"]["turn"] >= 1:
            kw = determinize(typed, deck, ident.predicted_deck(), random.Random(0))
            st = typed.current
            me, opp = st.players[0], st.players[1]
            assert len(kw["your_deck"]) == me.deckCount
            assert len(kw["your_prize"]) == len(me.prize)
            assert len(kw["opponent_deck"]) == opp.deckCount
            assert len(kw["opponent_hand"]) == opp.handCount
            checks += 1
        obs = battle_select(hp.select(obs, None))
    battle_finish()
    assert checks > 0


@needs_cg
def test_mcts_plays_legally():
    from cg.game import battle_finish, battle_select, battle_start

    from bot.eval.gauntlet import load_gauntlet_decks
    from bot.policies.heuristic import HeuristicPolicy
    from bot.policies.mcts import MctsPolicy

    deck = load_gauntlet_decks()["dragapult-ex"]
    mp = MctsPolicy(our_deck=deck, move_budget_s=0.03, min_samples=1)
    hp = HeuristicPolicy()
    obs, start = battle_start(deck, deck)
    n = 0
    while obs["current"]["result"] < 0 and n < 30:
        yi = obs["current"]["yourIndex"]
        sel = (mp if yi == 0 else hp).select(obs, None)
        opt_n = len(obs["select"]["option"])
        assert isinstance(sel, list) and all(isinstance(i, int) and 0 <= i < opt_n for i in sel)
        obs = battle_select(sel)
        n += 1
    battle_finish()
