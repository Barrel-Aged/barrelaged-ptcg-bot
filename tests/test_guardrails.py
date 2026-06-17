"""Guardrail tests (Sprint 2 / Story S2.5) — the bot's Definition-of-Done gates.

The agent must, for any observation: (1) return only LEGAL selections, (2) never raise, (3) return
its 60-card deck at the initial step. These run without the `cg` engine by feeding synthetic
observations covering the validity edge cases. A real-engine self-play smoke test lives separately
(needs `cg` on PYTHONPATH) and is skipped when unavailable.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.agent import Agent, _is_legal  # noqa: E402


@pytest.fixture
def bot():
    return Agent()


def _select(n_options, min_count, max_count):
    return {"select": {"option": [{"type": 14} for _ in range(n_options)],
                       "minCount": min_count, "maxCount": max_count}}


def test_initial_step_returns_60_card_deck(bot):
    out = bot.act({"select": None})
    assert isinstance(out, list) and len(out) == 60 and all(isinstance(i, int) for i in out)


@pytest.mark.parametrize("n,mn,mx", [(1, 1, 1), (3, 1, 1), (5, 0, 3), (2, 2, 2), (8, 1, 4), (1, 0, 0)])
def test_selection_is_always_legal(bot, n, mn, mx):
    obs = _select(n, mn, mx)
    out = bot.act(obs)
    assert _is_legal(out, obs["select"]), f"illegal selection {out} for n={n} min={mn} max={mx}"


def test_agent_never_raises_on_malformed_select(bot):
    # Missing/odd fields must still yield a (possibly empty) legal-ish response, not an exception.
    for bad in ({"select": {"option": [], "minCount": 0, "maxCount": 0}},
                {"select": {"option": [{"type": 1}], "maxCount": 1}},
                {"select": {"option": [{"type": 1}, {"type": 2}], "minCount": 1, "maxCount": 1}}):
        out = bot.act(bad)
        assert isinstance(out, list)


def test_is_legal_rejects_bad_selections():
    sel = {"option": [0, 1, 2], "minCount": 1, "maxCount": 2}
    assert _is_legal([0], sel)
    assert _is_legal([0, 1], sel)
    assert not _is_legal([0, 0], sel)        # duplicate
    assert not _is_legal([3], sel)           # out of range
    assert not _is_legal([], sel)            # below minCount
    assert not _is_legal([0, 1, 2], sel)     # above maxCount
