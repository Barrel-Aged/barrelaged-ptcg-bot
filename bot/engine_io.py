"""Engine I/O helpers: deck loading, observation access, and always-legal selection.

Selection validity (RULES_AND_REQUIREMENTS §4, from the engine's `search_step` error codes):
  minCount <= len(returned) <= maxCount · every index in [0, len(option)) · no duplicates.
"""
from __future__ import annotations

import os

DECK_SIZE = 60


def load_deck(path: str | None = None) -> list[int]:
    """Read deck.csv (60 card IDs, one integer per line) from `path`, CWD, or the Kaggle mount."""
    candidates = [path] if path else []
    candidates += ["deck.csv", "/kaggle_simulations/agent/deck.csv",
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deck.csv")]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as f:
                ids = [int(line) for line in f.read().splitlines() if line.strip()]
            if len(ids) != DECK_SIZE:
                raise ValueError(f"{p}: expected {DECK_SIZE} card IDs, got {len(ids)}")
            return ids
    raise FileNotFoundError("deck.csv not found in any known location")


def get_select(obs_dict: dict) -> dict | None:
    """Return the `select` block, or None at the initial deck-selection step."""
    return obs_dict.get("select") if isinstance(obs_dict, dict) else None


def num_options(select: dict) -> int:
    return len(select.get("option") or [])


def safe_selection(select: dict) -> list[int]:
    """An always-legal selection for any `select`: the first `maxCount` option indices.

    `maxCount` never exceeds `len(option)` and is >= `minCount`, so `range(maxCount)` is always
    within bounds, has no duplicates, and satisfies the count window. This is the last-resort
    fallback that guarantees the agent never forfeits on an illegal/empty selection.
    """
    max_count = int(select.get("maxCount", 0))
    return list(range(max_count))
