"""Regression guards for resource discipline (B-22 'don't deck yourself out').

Decking-out was the dominant failure (≈100% of ns-zoroark/dragapult losses) before the deck-out
guards; this locks in the fix so a future edit can't silently reintroduce it.

- Fast unit test: the thresholds are LOADED FROM the knowledge file (file-driven, not hardcoded).
- Integration test (needs the cg engine; skipped otherwise): on a draw-heavy deck the heuristic
  must not deck itself out in most games.
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CG_AVAILABLE = importlib.util.find_spec("cg") is not None
needs_cg = pytest.mark.skipif(not CG_AVAILABLE, reason="cg engine not on path")


def test_resource_thresholds_loaded_from_knowledge_file():
    from bot import knowledge
    rp = knowledge.resource_policy()
    file_block = knowledge.beginner()["resource_policy"]
    # the policy values come from the JSON, not code defaults
    assert rp["draw_supporter_if_hand_at_most"] == file_block["draw_supporter_if_hand_at_most"]
    assert rp["search_if_hand_at_most"] == file_block["search_if_hand_at_most"]
    assert rp["deck_out_guard"] == file_block["deck_out_guard"]


@needs_cg
def test_heuristic_does_not_deck_out_on_draw_heavy_deck():
    """On ns-zoroark (a draw engine that previously decked out ~100% of losses), deck-out losses
    must stay well below the pre-fix level. Threshold is loose to tolerate eval non-determinism."""
    from bot.eval.gauntlet import Competitor, _agent_for, load_gauntlet_decks
    from bot.policies.heuristic import HeuristicPolicy
    from bot.policies.random_policy import RandomPolicy
    from cg.game import battle_finish, battle_select, battle_start

    deck = load_gauntlet_decks()["ns-zoroark-ex"]
    games, deckout_losses = 20, 0
    for g in range(games):
        h = _agent_for(Competitor("h", deck, HeuristicPolicy))
        r = _agent_for(Competitor("r", deck, RandomPolicy))
        agents = [h, r] if g % 2 == 0 else [r, h]
        hero = 0 if g % 2 == 0 else 1
        obs, start = battle_start(deck, deck)
        if start.errorPlayer >= 0:
            battle_finish()
            continue
        while obs["current"]["result"] < 0:
            obs = battle_select(agents[obs["current"]["yourIndex"]].act(obs))
        st = obs["current"]
        if st["result"] != 2 and st["result"] != hero and st["players"][hero]["deckCount"] == 0:
            deckout_losses += 1
        battle_finish()
    rate = deckout_losses / games
    assert rate < 0.35, f"deck-out-loss rate {rate:.0%} too high — resource discipline regressed"
