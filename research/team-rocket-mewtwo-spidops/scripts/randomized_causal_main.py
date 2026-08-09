"""Packaged entry point for the randomized full-match causal overlay."""

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
from soldier.randomized_causal_runtime import CausalRidgeRouter, load_causal_ridge  # noqa: E402


def _find(name):
    local = os.path.join(_DIR, name)
    return local if os.path.exists(local) else "/kaggle_simulations/agent/" + name


with open(_find("deck.csv"), encoding="utf-8") as handle:
    _DECK = [int(value) for value in handle.read().split()][:60]
with open(_find("deck_library.json"), encoding="utf-8") as handle:
    _DECK_LIBRARY = json.load(handle)
with open(_find("causal_ridge_manifest.json"), encoding="utf-8") as handle:
    _CONFIG = json.load(handle)

_NET = load_policy(_find("model.pt")).eval()
_ARTIFACT = load_causal_ridge(_find("causal_ridge.npz"))
_AGENT = None


def _fresh_agent():
    return CausalRidgeRouter(
        _NET,
        _ARTIFACT,
        _CONFIG,
        deck_ids=_DECK,
        opponent_deck_priors=_DECK_LIBRARY,
        candidates=int(_CONFIG.get("shortlist_size", 4)),
        eligible_turn=int(_CONFIG.get("eligible_turn", 2)),
    )


def agent(observation):
    global _AGENT
    if observation.get("select") is None:
        _AGENT = _fresh_agent()
        return _DECK
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _AGENT(observation)
