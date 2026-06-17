"""Granular play-pattern auditor — trace one game turn-by-turn and flag basic-play defects.

Not about win-rate. It plays a game (audited hero = player 0), records every decision the hero makes,
groups them by turn, and reports a readable trace + flags for: decking out, never developing the
board (bench/evolve), passing up beneficial setup (evolve/attach/draw available but turn ended),
not attacking when able, and out-of-order sequencing (attach/attack before an available draw/search).

Usage:  python -m bot.eval.audit --deck dragapult-ex [--policy heuristic|mcts] [--opp <deck>]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cg.api import CardType, OptionType, SelectContext, all_card_data, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from bot.eval.gauntlet import load_gauntlet_decks  # noqa: E402
from bot.policies.heuristic import HeuristicPolicy  # noqa: E402

CARDS = None
OPT = {v: k for k, v in OptionType.__members__.items()}


def _cards():
    global CARDS
    if CARDS is None:
        CARDS = {c.cardId: c for c in all_card_data()}
    return CARDS


def _has_benchable_basic(obs, hand) -> bool:
    for o in obs.select.option:
        if o.type == OptionType.PLAY and o.index is not None and o.index < len(hand):
            c = _cards().get(hand[o.index].id)
            if c and c.cardType == CardType.POKEMON and c.basic:
                return True
    return False


def _has_draw_supporter(obs, hand) -> bool:
    for o in obs.select.option:
        if o.type == OptionType.PLAY and o.index is not None and o.index < len(hand):
            c = _cards().get(hand[o.index].id)
            if c and c.cardType == CardType.SUPPORTER:
                return True
    return False


def _opt_types(obs) -> set:
    return {o.type for o in obs.select.option}


def audit_game(hero_deck, opp_deck, make_hero, hero_name="hero", step_cap=2000):
    from bot.eval.gauntlet import Competitor, _agent_for
    # Use the robust Agent path (handles edge obs + never hangs) and log from OUTSIDE the policy.
    agents = [_agent_for(Competitor("hero", hero_deck, make_hero)),
              _agent_for(Competitor("opp", opp_deck, HeuristicPolicy))]
    obs, start = battle_start(hero_deck, opp_deck)
    if start.errorPlayer >= 0:
        print(f"deck error {start.errorType}")
        return
    turns = defaultdict(lambda: {"played": [], "avail": set(), "draw_avail_lowhand": False,
                                 "bench_avail": False, "attack_avail": False, "attacked": False,
                                 "evolve_avail": False, "evolved": False, "drew_or_searched": False,
                                 "attached_or_attacked_first": False})
    steps = 0
    while obs["current"]["result"] < 0 and steps < step_cap:
        steps += 1
        yi = obs["current"]["yourIndex"]
        sel = agents[yi].act(obs)
        if yi == 0 and obs["select"] and obs["select"]["context"] == SelectContext.MAIN:
            typed = to_observation_class(obs)
            me = typed.current.players[0]
            t = typed.current.turn
            rec = turns[t]
            avail = _opt_types(typed)
            rec["avail"] |= avail
            hand = me.hand or []
            chosen = typed.select.option[sel[0]].type if sel else OptionType.END
            rec["played"].append(OPT.get(chosen, str(chosen)))
            # snapshot state once per turn
            rec["state"] = (me.active[0].id if me.active and me.active[0] else None,
                            len(me.active[0].energies) if me.active and me.active[0] else 0,
                            len(me.bench or []), me.handCount, me.deckCount, len(me.prize))
            if _has_draw_supporter(typed, hand) and me.handCount <= 4:
                rec["draw_avail_lowhand"] = True
            if _has_benchable_basic(typed, hand):
                rec["bench_avail"] = True
            if OptionType.EVOLVE in avail:
                rec["evolve_avail"] = True
            if chosen == OptionType.EVOLVE:
                rec["evolved"] = True
            if OptionType.ATTACK in avail:
                rec["attack_avail"] = True
            if chosen == OptionType.ATTACK:
                rec["attacked"] = True
            if chosen in (OptionType.PLAY,) and _has_draw_supporter(typed, hand):
                rec["drew_or_searched"] = True
            if chosen in (OptionType.ATTACH, OptionType.ATTACK) and not rec["drew_or_searched"] \
                    and _has_draw_supporter(typed, hand):
                rec["attached_or_attacked_first"] = True
    res = obs["current"]["result"]
    final = obs["current"]["players"][0]
    deck_out = final["deckCount"] == 0
    battle_finish()

    # ---- report ----
    print(f"\n===== AUDIT: {hero_name} (result: {'WIN' if res == 0 else 'LOSS' if res == 1 else 'DRAW'}) =====")
    print(f"final: deck={final['deckCount']} prizesLeft={len(final['prize'])} bench={len(final['bench'] or [])} "
          f"{'⚠️ DECKED OUT' if deck_out else ''}")
    print("turn | active(E) bench hand deck przL | actions")
    defects = []
    ever_evolved = False
    attacks = 0
    for t in sorted(turns):
        r = turns[t]
        st = r.get("state", (None, 0, 0, 0, 0, 6))
        aid, en, bench, hand, deck, prz = st
        acts = ",".join(r["played"])
        flags = []
        if r["draw_avail_lowhand"] and "PLAY" not in r["played"]:
            flags.append("⚠passed-draw@lowhand")
        if r["evolve_avail"] and not r["evolved"]:
            flags.append("⚠evolve-avail-not-taken")
        if r["attack_avail"] and not r["attacked"]:
            flags.append("⚠attack-avail-not-taken")
        if r["attached_or_attacked_first"]:
            flags.append("⚠attach/attack-before-draw")
        if r["evolved"]:
            ever_evolved = True
        if r["attacked"]:
            attacks += 1
        fs = "  " + " ".join(flags) if flags else ""
        print(f"  {t:2} | id{aid}({en}E) b{bench} h{hand} d{deck} p{prz} | {acts}{fs}")
        for f in flags:
            defects.append(f)
    # game-level defects
    print("\nDEFECT SUMMARY:")
    if deck_out:
        print("  ✗ DECK-OUT (ran out of cards)")
    if not ever_evolved and any(OptionType.EVOLVE in turns[t]["avail"] for t in turns):
        print("  ✗ NEVER EVOLVED despite evolve being available")
    if attacks == 0:
        print("  ✗ NEVER ATTACKED all game")
    from collections import Counter
    dc = Counter(defects)
    for d, n in dc.most_common():
        print(f"  ⚠ {d} ×{n}")
    if not deck_out and ever_evolved and attacks > 0 and not dc:
        print("  (no basic-play defects flagged)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="dragapult-ex")
    ap.add_argument("--opp", default=None, help="opponent deck (default: mirror)")
    ap.add_argument("--games", type=int, default=1)
    args = ap.parse_args()
    decks = load_gauntlet_decks()
    hero_deck = decks[args.deck]
    opp_deck = decks[args.opp] if args.opp else hero_deck
    for _ in range(args.games):
        audit_game(hero_deck, opp_deck, HeuristicPolicy, hero_name=args.deck)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
