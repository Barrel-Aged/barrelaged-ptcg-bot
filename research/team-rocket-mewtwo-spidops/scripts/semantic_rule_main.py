"""Kaggle entry point template for the bounded semantic rule overlay."""

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
from soldier.semantic_rules import SemanticRuleRouter, load_semantic_rules  # noqa: E402


def _find(name):
    local = os.path.join(_DIR, name)
    return local if os.path.exists(local) else "/kaggle_simulations/agent/" + name


with open(_find("deck.csv"), encoding="utf-8") as handle:
    _DECK = [int(line) for line in handle.read().splitlines() if line.strip()][:60]
with open(_find("deck_library.json"), encoding="utf-8") as handle:
    _OPPONENT_DECK_PRIORS = json.load(handle)

_NET = load_policy(_find("model.pt")).eval()
_RULES = load_semantic_rules(_find("semantic_rules.json"))
_AGENT = None


def _fresh_agent():
    return SemanticRuleRouter(
        _NET,
        _RULES,
        deck_ids=_DECK,
        opponent_deck_priors=_OPPONENT_DECK_PRIORS,
        candidates=4,
        latest_turn=8,
    )


def agent(observation):
    global _AGENT
    if observation.get("select") is None:
        _AGENT = _fresh_agent()
        return _DECK
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _AGENT(observation)
