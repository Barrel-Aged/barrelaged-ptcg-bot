"""BudgetManager — the per-episode compute bank (RULES_AND_REQUIREMENTS §9).

The engine gives each agent a ~600 s compute bank for the WHOLE episode (kaggle_environments
`remainingOverageTime`, `actTimeout=0`), under a 3000 s wall. Overrunning the bank = TIMEOUT = loss.
So we budget across the game: cheap on routine turns, more on pivotal ones, always leaving a margin.

This is the scaffold the Phase-2 MCTS will consult (`time_for_move` / `should_continue`). The
Phase-1 default-safe policy doesn't search, so it simply never approaches the bank.
"""
from __future__ import annotations

import time

DEFAULT_BANK_S = 600.0   # remainingOverageTime default from cabt.json
SAFETY_MARGIN_S = 30.0   # never spend the last N seconds of the bank


class BudgetManager:
    def __init__(self, bank_s: float = DEFAULT_BANK_S, safety_margin_s: float = SAFETY_MARGIN_S):
        self._bank_s = bank_s
        self._margin_s = safety_margin_s
        self._spent_s = 0.0
        self._move_start: float | None = None

    def update(self, obs_dict: dict) -> None:
        """Sync with the engine's authoritative remaining-time signal when present."""
        rem = obs_dict.get("remainingOverageTime") if isinstance(obs_dict, dict) else None
        if rem is not None:
            try:
                self._spent_s = max(self._spent_s, self._bank_s - float(rem))
            except (TypeError, ValueError):
                pass

    def remaining_s(self) -> float:
        return max(0.0, self._bank_s - self._spent_s - self._margin_s)

    def time_for_move(self, est_moves_left: int = 40) -> float:
        """Soft per-move allowance: spread the usable bank over the moves we expect to make."""
        return self.remaining_s() / max(1, est_moves_left)

    def start_move(self) -> None:
        self._move_start = time.monotonic()

    def elapsed_move_s(self) -> float:
        return 0.0 if self._move_start is None else time.monotonic() - self._move_start

    def end_move(self) -> None:
        if self._move_start is not None:
            self._spent_s += time.monotonic() - self._move_start
            self._move_start = None

    def should_continue(self, move_allowance_s: float) -> bool:
        """For anytime search: keep iterating only while within this move's allowance and the bank."""
        return self.elapsed_move_s() < move_allowance_s and self.remaining_s() > 0
