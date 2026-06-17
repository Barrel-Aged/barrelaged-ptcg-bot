"""Loads the heuristics knowledge base that DRIVES HeuristicPolicy.

The bot's play is an interpreter of `heuristics_beginner.json`'s executable `turn_policy` /
`opening_policy` (ordered action steps, each tagged with B-rule IDs) — so editing the JSON changes
the bot's behavior. `heuristics.json` (advanced H-rules) is loaded for the prize-trade references
(H-31..H-39) used in targeting. The files are vendored into `bot/knowledge/` so the submission is
self-contained (no internet at eval).
"""
from __future__ import annotations

import json
import os

_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
_BEGINNER = None
_ADVANCED = None


def _load(name: str) -> dict:
    with open(os.path.join(_DIR, name)) as f:
        return json.load(f)


def beginner() -> dict:
    global _BEGINNER
    if _BEGINNER is None:
        _BEGINNER = _load("heuristics_beginner.json")
    return _BEGINNER


def advanced() -> dict:
    global _ADVANCED
    if _ADVANCED is None:
        _ADVANCED = _load("heuristics.json")
    return _ADVANCED


def turn_policy() -> list[dict]:
    """Ordered MAIN-phase steps: [{step, action, rules:[B-..]}...]."""
    return beginner()["turn_policy"]


def opening_policy() -> list[dict]:
    return beginner()["opening_policy"]


def rules_index() -> dict[str, dict]:
    """B-rule id -> {title, rule, why, priority, maps_to}."""
    return {r["id"]: r for r in beginner().get("rules", [])}
