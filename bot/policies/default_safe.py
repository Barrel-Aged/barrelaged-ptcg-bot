"""DefaultSafePolicy — the always-legal baseline (Sprint 2).

Picks the first `maxCount` option indices for every decision. Legal everywhere, never raises, never
times out. It plays badly on purpose: it exists so the scaffold (Agent, budget, eval harness,
guardrail tests) is exercised end-to-end before the real heuristics arrive in Sprint 3 / Phase 1.
"""
from __future__ import annotations

from bot.engine_io import safe_selection
from bot.policies.base import Policy


class DefaultSafePolicy(Policy):
    def select(self, obs_dict: dict, budget) -> list[int]:
        return safe_selection(obs_dict["select"])
