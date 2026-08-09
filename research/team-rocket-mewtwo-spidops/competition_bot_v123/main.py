"""Public-BC Mega Lucario with a calibrated deterministic safety fallback."""
from __future__ import annotations

import os
import sys
import json

_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/kaggle_simulations/agent"
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import torch  # noqa: E402

torch.set_num_threads(1)

from hybrid_policy import ConfidenceSafeHybridAgent  # noqa: E402
from soldier.net import load_policy  # noqa: E402


def _find(name: str) -> str:
    local = os.path.join(_DIR, name)
    return local if os.path.exists(local) else "/kaggle_simulations/agent/" + name


with open(_find("deck.csv"), encoding="utf-8") as handle:
    _DECK = [int(line) for line in handle.read().splitlines() if line.strip()][:60]

_NET = load_policy(_find("model.pt")).eval()
_DECK_LIBRARY_PATH = _find("deck_library.json")
if os.path.exists(_DECK_LIBRARY_PATH):
    with open(_DECK_LIBRARY_PATH, encoding="utf-8") as handle:
        _OPPONENT_DECK_PRIORS = json.load(handle)
else:
    _OPPONENT_DECK_PRIORS = None
_AGENT = None
_THRESHOLD_PATH = _find("HYBRID_CALIBRATION.json")
if os.path.exists(_THRESHOLD_PATH):
    with open(_THRESHOLD_PATH, encoding="utf-8") as handle:
        _HYBRID_THRESHOLD = float(json.load(handle)["selected_threshold"])
else:
    _HYBRID_THRESHOLD = 0.0


def _fresh_agent() -> ConfidenceSafeHybridAgent:
    return ConfidenceSafeHybridAgent(
        _NET,
        deck_ids=list(_DECK),
        threshold=_HYBRID_THRESHOLD,
        opponent_deck_priors=_OPPONENT_DECK_PRIORS,
    )


def agent(observation: dict):
    """Return the deck at setup and legal policy actions during a game."""
    global _AGENT
    if observation.get("select") is None:
        # A module can be reused across arena games; belief state must not be.
        _AGENT = _fresh_agent()
        return _DECK
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _AGENT(observation)
