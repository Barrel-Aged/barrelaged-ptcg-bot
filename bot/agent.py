"""Agent — wires the policy stack to the engine contract (ARCHITECTURE §2).

Responsibilities: return the deck at the initial step; route selections to the active Policy under
the BudgetManager; and guarantee a legal, non-raising response at every decision point.
"""
from __future__ import annotations

from bot.budget import BudgetManager
from bot.engine_io import get_select, load_deck, safe_selection
from bot.policies.base import Policy
from bot.policies.default_safe import DefaultSafePolicy


class Agent:
    def __init__(self, policy: Policy | None = None, deck_path: str | None = None):
        self._deck = load_deck(deck_path)
        self._policy = policy or DefaultSafePolicy()
        self._budget = BudgetManager()

    def act(self, obs_dict: dict) -> list[int]:
        select = get_select(obs_dict)
        if select is None:
            return self._deck                      # initial deck-selection step
        self._budget.update(obs_dict)
        self._budget.start_move()
        try:
            choice = self._policy.select(obs_dict, self._budget)
            return choice if _is_legal(choice, select) else safe_selection(select)
        except Exception:
            return safe_selection(select)           # never forfeit on a crash
        finally:
            self._budget.end_move()


def _is_legal(choice, select: dict) -> bool:
    if not isinstance(choice, list) or any(not isinstance(i, int) for i in choice):
        return False
    n = len(select.get("option") or [])
    if len(set(choice)) != len(choice):
        return False
    if not all(0 <= i < n for i in choice):
        return False
    return int(select.get("minCount", 0)) <= len(choice) <= int(select.get("maxCount", 0))
