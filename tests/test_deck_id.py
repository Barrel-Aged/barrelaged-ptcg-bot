"""Tests for DeckIdentifier (Sprint 5).

Fast unit tests (no engine): the Bayesian matcher identifies an archetype from its discriminative
cards, and predicted_deck returns a legal 60-card hypothesis. Integration accuracy is measured by
`python -m bot.deck_id.eval_accuracy` (needs cg; ~94% top-1 by turn 4).
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.deck_id.identifier import DeckIdentifier, _load  # noqa: E402


def _unique_cards():
    """Map archetype id -> a card id that appears ONLY in that archetype (its signature)."""
    archs = _load()
    out = {}
    for a in archs:
        for cid in a["_cards"]:
            if sum(1 for b in archs if cid in b["_cards"]) == 1:
                out[a["id"]] = cid
                break
    return out


@pytest.mark.parametrize("arch_id,card_id", list(_unique_cards().items()))
def test_unique_card_identifies_archetype(arch_id, card_id):
    di = DeckIdentifier(your_index=0)
    di._seen = {1000: card_id}   # inject one signature card as evidence
    assert di.best() == arch_id, f"signature card {card_id} should identify {arch_id}"


def test_posterior_is_a_distribution():
    di = DeckIdentifier(your_index=0)
    post = di.posterior()
    assert abs(sum(post.values()) - 1.0) < 1e-9
    assert len(post) == len(_load())


def test_predicted_deck_is_60_cards():
    di = DeckIdentifier(your_index=0)
    for a in _load():
        deck = di.predicted_deck(a["id"])
        assert len(deck) == 60, f"{a['id']} predicted deck has {len(deck)} cards"
        assert all(isinstance(c, int) for c in deck)


def test_no_evidence_falls_back_to_prior():
    di = DeckIdentifier(your_index=0)
    # with zero evidence the top pick is the highest-meta-share archetype (dragapult)
    assert di.best() == "dragapult-ex"
