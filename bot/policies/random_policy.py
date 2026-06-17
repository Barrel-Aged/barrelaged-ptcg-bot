"""RandomPolicy — the official 'random' baseline (the Phase-1 yardstick).

Mirrors the competition's reference random agent: a random legal selection each decision.
"""
from __future__ import annotations

import random

from bot.policies.base import Policy


class RandomPolicy(Policy):
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select(self, obs_dict: dict, budget) -> list[int]:
        sel = obs_dict["select"]
        n = len(sel.get("option") or [])
        k = int(sel.get("maxCount", 0))
        if n == 0 or k == 0:
            return []
        return self._rng.sample(range(n), min(k, n))
