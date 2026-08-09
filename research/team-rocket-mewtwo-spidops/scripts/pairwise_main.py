"""Kaggle entry point template for an admitted causal-pairwise overlay."""

from __future__ import annotations

import json
import os
import sys


_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/kaggle_simulations/agent"
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import torch  # noqa: E402

torch.set_num_threads(1)

from soldier.net import load_policy  # noqa: E402
from soldier.pairwise import PairwiseRouter, load_pairwise  # noqa: E402


def _find(name: str) -> str:
    local = os.path.join(_DIR, name)
    return local if os.path.exists(local) else "/kaggle_simulations/agent/" + name


with open(_find("deck.csv"), encoding="utf-8") as handle:
    _DECK = [int(line) for line in handle.read().splitlines() if line.strip()][:60]

with open(_find("pairwise_gate.json"), encoding="utf-8") as handle:
    _GATE = json.load(handle)

_NET = load_policy(_find("model.pt")).eval()
_PAIRWISE = load_pairwise(_find("pairwise.pt")).eval()
_DECK_LIBRARY_PATH = _find("deck_library.json")
if os.path.exists(_DECK_LIBRARY_PATH):
    with open(_DECK_LIBRARY_PATH, encoding="utf-8") as handle:
        _OPPONENT_DECK_PRIORS = json.load(handle)
else:
    _OPPONENT_DECK_PRIORS = None
_AGENT = None


def _fresh_agent() -> PairwiseRouter:
    return PairwiseRouter(
        _NET,
        _PAIRWISE,
        deck_ids=_DECK,
        opponent_deck_priors=_OPPONENT_DECK_PRIORS,
        threshold=float(_GATE["threshold"]),
        candidates=int(_GATE.get("candidates", 4)),
        latest_turn=int(_GATE.get("latest_turn", 8)),
    )


def agent(observation: dict):
    """Return the exact Rocket deck at setup and a legal action thereafter."""
    global _AGENT
    if observation.get("select") is None:
        _AGENT = _fresh_agent()
        return _DECK
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _AGENT(observation)
