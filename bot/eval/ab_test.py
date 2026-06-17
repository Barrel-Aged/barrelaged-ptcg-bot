"""Variance-aware A/B evaluation — measure effects honestly, with confidence intervals.

The cg engine is non-deterministic (internal shuffle/coin RNG, no seed) and that variance is
irreducible, so common-random-numbers pairing is impossible — the only lever is sample size. This
runs each arm over many games (in parallel, since cg is one-battle-per-process), reports win-rate
with a 95% Wilson confidence interval, and only calls a difference real when the CIs separate.

Lesson encoded here: a single 12–16-game number is meaningless for a ±25pt effect (we saw the same
config give 33% and 69%). Use this, not ad-hoc loops, for any MCTS comparison.

Factories must be picklable for the spawn pool — use the module-level `make_*` builders (or
functools.partial of them), NOT lambdas.
"""
from __future__ import annotations

import math
import multiprocessing as mp

from bot.eval.gauntlet import Competitor, play_game
from bot.policies.heuristic import HeuristicPolicy
from bot.policies.mcts import MctsPolicy


# ---- picklable policy builders (module-level → safe for spawn) --------------------------------
def make_heuristic(**kw):
    return HeuristicPolicy(**kw)


def make_mcts(deck, n_iterations=20, force_archetype=None, hedge=False, seed=0):
    return MctsPolicy(our_deck=deck, n_iterations=n_iterations, force_archetype=force_archetype,
                      hedge=hedge, seed=seed)


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (robust at small n / extreme p)."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, center - half), min(1.0, center + half)


def _one_game(args):
    arm_a, arm_b, swap = args
    a, b = (arm_b, arm_a) if swap else (arm_a, arm_b)
    winner, _, _ = play_game(a, b)
    if winner is None:
        return 0.5
    return 1.0 if winner == arm_a.name else 0.0   # arm_a's score


def run_arm(name: str, hero_factory, opp_factory, opp_deck, hero_deck, games: int, workers: int):
    """Play `name` (hero_factory piloting hero_deck) vs opp over `games`; return (wins, draws, n)."""
    arm = Competitor(name, hero_deck, hero_factory)
    opp = Competitor("opp", opp_deck, opp_factory)
    tasks = [(arm, opp, g % 2 == 1) for g in range(games)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        scores = pool.map(_one_game, tasks)
    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    return wins, draws, len(scores)


def decisive(wins: int, draws: int, n: int) -> tuple[int, int]:
    """Reduce to decisive games (exclude draws) — win-rate must be measured among games that had a
    winner, else high-draw decks (control mirrors) read ~30% even for identical arms."""
    return wins, max(1, n - draws)


def compare(label_a, fac_a, label_b, fac_b, opp_factory, opp_deck, hero_deck,
            games: int = 120, workers: int | None = None) -> dict:
    """Run two arms vs the same opponent; report DECISIVE win-rates + 95% CIs (draws excluded)."""
    workers = workers or max(1, mp.cpu_count() - 2)
    wa, da, na = run_arm(label_a, fac_a, opp_factory, opp_deck, hero_deck, games, workers)
    wb, db, nb = run_arm(label_b, fac_b, opp_factory, opp_deck, hero_deck, games, workers)
    wa_d, na_d = decisive(wa, da, na)
    wb_d, nb_d = decisive(wb, db, nb)
    pa, pb = wa_d / na_d, wb_d / nb_d
    loa, hia = wilson_ci(wa_d, na_d)
    lob, hib = wilson_ci(wb_d, nb_d)
    separated = hia < lob or hib < loa
    return {"a": (label_a, pa, loa, hia, wa, na), "b": (label_b, pb, lob, hib, wb, nb),
            "delta": pa - pb, "significant": separated}


def fmt(res: dict) -> str:
    la, pa, loa, hia, wa, na = res["a"]
    lb, pb, lob, hib, wb, nb = res["b"]
    return (f"  {la:16} {pa*100:5.1f}%  CI[{loa*100:4.1f},{hia*100:4.1f}]  ({wa}/{na})\n"
            f"  {lb:16} {pb*100:5.1f}%  CI[{lob*100:4.1f},{hib*100:4.1f}]  ({wb}/{nb})\n"
            f"  delta {res['delta']*100:+.1f}pt — {'SIGNIFICANT (CIs separate)' if res['significant'] else 'not significant (CIs overlap)'}")
