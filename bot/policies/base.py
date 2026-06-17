"""Policy interface — the swappable decision core (ARCHITECTURE §2).

Phases plug in here: DefaultSafePolicy (Sprint 2 baseline) → HeuristicPolicy (Phase 1) →
MctsPolicy (Phase 2) → RlPolicy (Phase 3). Every policy MUST return a legal selection and must
not raise (the Agent wraps calls, but policies should be robust on their own too).
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def select(self, obs_dict: dict, budget) -> list[int]:
        """Return chosen option indices for `obs_dict['select']` (never called at the deck step)."""
        raise NotImplementedError
