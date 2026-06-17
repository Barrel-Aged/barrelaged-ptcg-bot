"""Re-run the deck-ID and hedge ablations PROPERLY — large N, parallel, with confidence intervals.

Usage (repo root, cg on path):  python -m bot.eval.run_ab --games 120 --iters 10
The __main__ guard is REQUIRED (spawn pool).
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bot.eval.ab_test import compare, fmt, make_heuristic, make_mcts  # noqa: E402
from bot.eval.gauntlet import load_gauntlet_decks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=120, help="games per arm per opponent")
    ap.add_argument("--iters", type=int, default=10, help="fixed MCTS determinization iterations")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--opponents", nargs="+", default=["ns-zoroark-ex", "hydrapple-ex"])
    args = ap.parse_args()

    decks = load_gauntlet_decks()
    hd = decks["dragapult-ex"]
    real = functools.partial(make_mcts, deck=hd, n_iterations=args.iters, force_archetype=None)
    naive = functools.partial(make_mcts, deck=hd, n_iterations=args.iters, force_archetype="dragapult-ex")

    print(f"DECK-ID ABLATION (proper): real deck-ID vs naive, vs Heuristic(opp), "
          f"{args.games} games/arm, {args.iters} iters\n")
    for opp in args.opponents:
        t = time.monotonic()
        res = compare("real-deckID", real, "naive", naive, make_heuristic, decks[opp], hd,
                      games=args.games, workers=args.workers)
        print(f"vs {opp}  ({time.monotonic()-t:.0f}s):")
        print(fmt(res))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
