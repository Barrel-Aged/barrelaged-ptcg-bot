"""CLI: run the bot against the prechosen-deck gauntlet.

Built on the curated decks (`bot/eval/gauntlet/*.csv`) — independent of the daily strategy analyzer.
The `if __name__ == "__main__"` guard is REQUIRED: the parallel runner uses a 'spawn' pool, whose
workers re-import this module; without the guard they would re-run the gauntlet (spawn fork-bomb).

Usage (from the repo root, with cg on PYTHONPATH):
    python -m bot.eval.run_gauntlet --deck dragapult-ex --games 10 --workers 14
    python -m bot.eval.run_gauntlet --sequential --games 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bot.eval.gauntlet import Competitor, load_gauntlet_decks, run_gauntlet  # noqa: E402
from bot.eval.parallel import run_gauntlet_parallel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the bot vs the prechosen-deck gauntlet.")
    ap.add_argument("--deck", default="dragapult-ex", help="hero deck (a gauntlet archetype id)")
    ap.add_argument("--games", type=int, default=10, help="games vs each opponent")
    ap.add_argument("--workers", type=int, default=None, help="parallel workers (default cpu-2)")
    ap.add_argument("--sequential", action="store_true", help="run single-process")
    args = ap.parse_args()

    decks = load_gauntlet_decks()
    if args.deck not in decks:
        sys.exit(f"unknown deck '{args.deck}'. options: {', '.join(sorted(decks))}")
    hero = Competitor(f"hero({args.deck})", decks[args.deck])
    opponents = [Competitor(n, decks[n]) for n in decks if n != args.deck]

    t = time.monotonic()
    if args.sequential:
        standings = run_gauntlet(hero, opponents, games_each=args.games)
    else:
        standings = run_gauntlet_parallel(hero, opponents, games_each=args.games, workers=args.workers)
    dt = time.monotonic() - t

    n_games = len(opponents) * args.games
    print(f"{n_games} games in {dt:.1f}s ({n_games / dt:.1f} games/s, "
          f"{'sequential' if args.sequential else f'{args.workers or os.cpu_count()-2} workers'})\n")
    print(standings.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
