"""MctsPolicy (Phase 2) — flat Information-Set Monte Carlo over the Search API.

For the current decision, evaluate each legal action by repeatedly: determinize hidden info (our
decklist + the DeckIdentifier's predicted opponent deck), step the action in the Search API, and roll
out to a terminal/horizon with the HeuristicPolicy — averaging the win value over many
determinizations. Pick the highest-value action. This adds the lookahead the reactive heuristic
can't do (does this attack/play actually lead to winning?), while staying anytime: it falls back to
the heuristic move under time pressure or on any search error.

Flat MC (not a UCT tree) is the v1 — simpler, robust, and already captures consequence-evaluation.
"""
from __future__ import annotations

import random

from cg.api import SelectContext, search_begin, search_end, search_step, to_observation_class

from bot.deck_id.identifier import DeckIdentifier
from bot.policies.base import Policy
from bot.policies.determinize import determinize
from bot.policies.heuristic import HeuristicPolicy

ROLLOUT_CAP = 60          # max search steps per rollout before falling back to a heuristic value
DEFAULT_MOVE_BUDGET_S = 1.0   # soft per-move wall when no BudgetManager is supplied
MCTS_CONTEXTS = {SelectContext.MAIN, SelectContext.ATTACK}  # where lookahead pays off; else heuristic


class MctsPolicy(Policy):
    def __init__(self, our_deck: list[int] | None = None, seed: int = 0,
                 move_budget_s: float | None = None, min_samples: int = 20):
        self._heur = HeuristicPolicy()
        self._ident = DeckIdentifier(your_index=0)
        self._our_deck = our_deck
        self._rng = random.Random(seed)
        self._move_budget_s = move_budget_s   # hard per-move cap (overrides budget-derived deadline)
        self._min_samples = min_samples

    def set_deck(self, deck: list[int]) -> None:
        self._our_deck = deck

    def select(self, obs_dict: dict, budget=None) -> list[int]:
        obs = to_observation_class(obs_dict)
        self._ident._your_index = obs.current.yourIndex if obs.current else 0
        self._ident.observe(obs)
        sel = obs.select
        # Only search where lookahead matters and choice is non-trivial; else play the heuristic.
        if sel is None or sel.context not in MCTS_CONTEXTS or len(sel.option) < 2 \
                or self._our_deck is None or int(sel.maxCount) != 1:
            return self._heur.select_typed(obs)
        try:
            return [self._search(obs, budget)]
        except Exception:
            return self._heur.select_typed(obs)

    # ---- flat Monte-Carlo over determinizations ----------------------------------------------
    def _search(self, obs, budget) -> int:
        n_actions = len(obs.select.option)
        totals = [0.0] * n_actions
        counts = [0] * n_actions
        predicted = self._ident.predicted_deck()
        if self._move_budget_s is not None:
            deadline = self._move_budget_s
        else:
            deadline = min(budget.time_for_move(), 1.5) if budget is not None else DEFAULT_MOVE_BUDGET_S
        import time
        start = time.monotonic()
        iters = 0
        while time.monotonic() - start < deadline:
            kw = determinize(obs, self._our_deck, predicted, self._rng)
            root = search_begin(obs, **kw)
            for a in range(n_actions):
                child = search_step(root.searchId, [a])
                totals[a] += self._rollout(child)
                counts[a] += 1
            search_end()
            iters += 1
            if iters >= 2 and min(counts) >= self._min_samples:    # enough samples — stop early
                break
        if not any(counts):
            return self._heur.select_typed(obs)[0]
        means = [totals[a] / counts[a] if counts[a] else -1e9 for a in range(n_actions)]
        return max(range(n_actions), key=lambda a: means[a])

    def _rollout(self, state) -> float:
        """Play to terminal/horizon with the heuristic; return value in [-1,1] from our perspective."""
        me = self._ident._your_index
        sid = state.searchId
        obs = state.observation
        for _ in range(ROLLOUT_CAP):
            cur = obs.current
            if cur is None:
                break
            if cur.result >= 0:
                return 0.0 if cur.result == 2 else (1.0 if cur.result == me else -1.0)
            sel = self._heur.select_typed(obs)
            state = search_step(sid, sel)
            sid = state.searchId
            obs = state.observation
        # Horizon reached: estimate by the prize-trade (our taken vs opp taken), in [-1,1].
        cur = obs.current
        if cur is None:
            return 0.0
        my_taken = 6 - len(cur.players[me].prize)
        opp_taken = 6 - len(cur.players[1 - me].prize)
        return max(-1.0, min(1.0, (my_taken - opp_taken) / 6.0))
