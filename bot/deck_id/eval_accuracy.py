"""Measure DeckIdentifier top-1 accuracy by turn against opponents piloting known archetypes.

Gate (Sprint 5 / M4): top-1 archetype accuracy >= 75% by turn 4. Run from the repo root with cg on
the path:  python -m bot.deck_id.eval_accuracy [--games 20]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from bot.deck_id.identifier import DeckIdentifier  # noqa: E402
from bot.eval.gauntlet import Competitor, _agent_for, load_gauntlet_decks  # noqa: E402
from bot.policies.heuristic import HeuristicPolicy  # noqa: E402

TURN_CHECKPOINTS = [2, 4, 6]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20, help="games per opponent archetype")
    ap.add_argument("--hero", default="dragapult-ex")
    args = ap.parse_args()

    decks = load_gauntlet_decks()
    hero_deck = decks[args.hero]
    archetypes = [a for a in decks if a != args.hero] + [args.hero]  # include the mirror

    # accuracy[checkpoint] = [correct, total]; plus end-of-game
    acc = {t: [0, 0] for t in TURN_CHECKPOINTS}
    acc["end"] = [0, 0]
    per_arch_end = {}

    for true_arch in archetypes:
        opp_deck = decks[true_arch]
        for g in range(args.games):
            # hero is always player 0, opponent player 1 (first-player balance is irrelevant to
            # deck-ID accuracy; what matters is the identifier sees the opponent's true deck).
            hero = _agent_for(Competitor("h", hero_deck, HeuristicPolicy))
            opp = _agent_for(Competitor("o", opp_deck, HeuristicPolicy))
            agents = [hero, opp]
            ident = DeckIdentifier(your_index=0)

            obs, start = battle_start(hero_deck, opp_deck)
            if start.errorPlayer >= 0:
                battle_finish()
                continue
            done_cp = set()
            while obs["current"]["result"] < 0:
                typed = to_observation_class(obs)
                ident.observe(typed)
                turn = obs["current"]["turn"]
                for t in TURN_CHECKPOINTS:
                    if turn >= t and t not in done_cp and ident.evidence_count() >= 0:
                        done_cp.add(t)
                        acc[t][1] += 1
                        if ident.best() == true_arch:
                            acc[t][0] += 1
                obs = battle_select(agents[obs["current"]["yourIndex"]].act(obs))
            battle_finish()
            acc["end"][1] += 1
            correct = ident.best() == true_arch
            acc["end"][0] += correct
            per_arch_end.setdefault(true_arch, [0, 0])
            per_arch_end[true_arch][0] += correct
            per_arch_end[true_arch][1] += 1

    print(f"DeckIdentifier top-1 accuracy (hero={args.hero}, {args.games} games/opponent):")
    for t in TURN_CHECKPOINTS:
        c, n = acc[t]
        print(f"  by turn {t}: {100*c/n:.0f}%  ({c}/{n})")
    c, n = acc["end"]
    print(f"  end of game: {100*c/n:.0f}%  ({c}/{n})")
    gate = acc[4]
    print(f"\n  GATE (>=75% by turn 4): {'PASS' if 100*gate[0]/gate[1] >= 75 else 'FAIL'} "
          f"({100*gate[0]/gate[1]:.0f}%)")
    print("\n  per-opponent end accuracy:")
    for a, (c, n) in sorted(per_arch_end.items(), key=lambda kv: kv[1][0]/kv[1][1]):
        print(f"    {a:20} {100*c/n:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
