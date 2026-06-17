"""HeuristicPolicy (Phase 1) — an INTERPRETER of the heuristics knowledge base.

It executes `heuristics_beginner.json`'s `turn_policy` (9 ordered steps, each tagged with B-rules)
and `opening_policy` directly: at each MAIN decision it walks the steps IN FILE ORDER and takes the
first one whose action is legally available — so the JSON *is* the strategy and editing it changes
play. Each step is mapped to engine options by the B-rule IDs it carries (the code is the
interpreter; the order/intent/rules live in the file). Prize-trade targeting follows B-15 / H-31..H-39.

Never raises; never returns an illegal selection (falls back to the always-legal default).
"""
from __future__ import annotations

from cg.api import (AreaType, CardType, OptionType, SelectContext, all_attack,
                    all_card_data, to_observation_class)

from bot import knowledge
from bot.engine_io import safe_selection
from bot.policies.base import Policy

# Which engine action-"kind" a turn_policy step represents, signalled by the B-rule it carries.
# (First matching rule in the step decides its kind; precedence handles multi-rule steps.)
RULE_KIND = {
    "B-02": "draw_supporter", "B-05": "item_search", "B-06": "evolve", "B-18": "ability",
    "B-04": "attach", "B-14": "attach", "B-10": "stadium", "B-20": "stadium",
    "B-07": "attack", "B-08": "attack", "B-19": "attack", "B-16": "attack",
    "B-09": "bench", "B-01": "bench",
}
KIND_PRECEDENCE = ["draw_supporter", "item_search", "bench", "evolve", "ability", "attach",
                   "stadium", "attack"]

_CARD_TABLE = None
_ATTACK_TABLE = None


def _step_kind(step: dict) -> str | None:
    kinds = {RULE_KIND[r] for r in step.get("rules", []) if r in RULE_KIND}
    for k in KIND_PRECEDENCE:
        if k in kinds:
            return k
    return None


def _prize_value(card) -> int:
    return 3 if card.megaEx else 2 if card.ex else 1   # prizes the opponent takes on KO


class HeuristicPolicy(Policy):
    def __init__(self):
        self._cards = None
        self._attacks = None
        # Precompute the ordered (kind, step) plan once from the knowledge base.
        self._plan = [(k, s) for s in knowledge.turn_policy() if (k := _step_kind(s))]

    def _ensure_tables(self):
        global _CARD_TABLE, _ATTACK_TABLE
        if _CARD_TABLE is None:
            _CARD_TABLE = {c.cardId: c for c in all_card_data()}
            _ATTACK_TABLE = {a.attackId: a for a in all_attack()}
        self._cards, self._attacks = _CARD_TABLE, _ATTACK_TABLE

    # ---- dispatch ----------------------------------------------------------------------------
    def select(self, obs_dict: dict, budget) -> list[int]:
        self._ensure_tables()
        obs = to_observation_class(obs_dict)
        sel = obs.select
        ctx = sel.context
        if not sel.option:
            return safe_selection(obs_dict["select"])

        if ctx == SelectContext.MAIN:
            return [self._run_turn_policy(obs)]
        # Promote/active (H-33: put your best attacker up). Bench (H-37/H-32/B-09: prefer
        # single-prize, don't volunteer 2-prize liabilities you don't need).
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
            return self._counted(sel, self._rank_attacker(obs))
        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return self._counted(sel, self._rank_bench(obs))
        if ctx == SelectContext.ATTACK:
            return [self._best_attack(obs)]
        if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
            return self._counted(sel, self._rank_targets(obs))   # B-15: hit 2-prize targets
        if ctx == SelectContext.DRAW_COUNT:
            return [self._max_number(sel.option)]
        return safe_selection(obs_dict["select"])

    # ---- the turn_policy interpreter ---------------------------------------------------------
    def _run_turn_policy(self, obs) -> int:
        """Walk turn_policy steps in file order; take the first available action (B-18 sequencing)."""
        opts = obs.select.option
        you = obs.current.yourIndex
        hand = obs.current.players[you].hand or []
        for kind, _step in self._plan:
            matches = [i for i, o in enumerate(opts) if self._matches(kind, o, hand)]
            if matches:
                return self._choose(kind, matches, opts, obs)
        # No policy step applies → end the turn (B-16: otherwise maximize setup and pass).
        for i, o in enumerate(opts):
            if o.type == OptionType.END:
                return i
        return 0

    def _matches(self, kind: str, o, hand) -> bool:
        t = o.type
        if kind == "draw_supporter":
            return t == OptionType.PLAY and self._card_of(o, hand) and \
                self._card_of(o, hand).cardType == CardType.SUPPORTER
        if kind == "item_search":
            return t == OptionType.PLAY and self._card_of(o, hand) and \
                self._card_of(o, hand).cardType == CardType.ITEM
        if kind == "bench":
            c = self._card_of(o, hand)
            return t == OptionType.PLAY and c and c.cardType == CardType.POKEMON and c.basic
        if kind == "evolve":
            return t == OptionType.EVOLVE
        if kind == "ability":
            return t == OptionType.ABILITY
        if kind == "attach":
            return t == OptionType.ATTACH
        if kind == "stadium":
            c = self._card_of(o, hand)
            return t == OptionType.PLAY and c and c.cardType == CardType.STADIUM
        if kind == "attack":
            return t == OptionType.ATTACK
        return False

    def _choose(self, kind: str, matches: list[int], opts, obs) -> int:
        if kind == "attach":   # B-04/B-14: energy onto the active attacker
            for i in matches:
                if getattr(opts[i], "inPlayArea", None) == AreaType.ACTIVE:
                    return i
            return matches[0]
        if kind == "attack":   # B-07/B-08/B-15: KO, preferring 2-prize targets
            return self._best_attack(obs, restrict=matches)
        return matches[0]

    # ---- attacks / targeting -----------------------------------------------------------------
    def _best_attack(self, obs, restrict=None) -> int:
        opp = obs.current.players[1 - obs.current.yourIndex]
        opp_hp = opp.active[0].hp if (opp.active and opp.active[0]) else None
        opp_val = _prize_value(self._cards[opp.active[0].id]) if (opp.active and opp.active[0]
                                                                  and opp.active[0].id in self._cards) else 1
        idxs = restrict if restrict is not None else range(len(obs.select.option))
        best_i, best_s = (idxs[0] if restrict else 0), float("-inf")
        for i in idxs:
            o = obs.select.option[i]
            atk = self._attacks.get(o.attackId) if o.attackId is not None else None
            dmg = atk.damage if atk else 0
            s = dmg
            if opp_hp is not None and dmg >= opp_hp:
                s += 120 + 60 * opp_val   # KO bonus, scaled by prizes taken (B-15)
            if s > best_s:
                best_i, best_s = i, s
        return best_i

    def _rank_targets(self, obs) -> list[int]:
        scored = []
        for i, o in enumerate(obs.select.option):
            c = self._cards.get(o.cardId) if o.cardId is not None else None
            scored.append((_prize_value(c) if c else 0, i))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    def _rank_attacker(self, obs) -> list[int]:
        """H-33: promote the best attacker (highest HP as a threat proxy; 2-prize is fine here)."""
        scored = []
        for i, o in enumerate(obs.select.option):
            c = self._cards.get(o.cardId) if o.cardId is not None else None
            scored.append((c.hp if c else 0, i))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    def _rank_bench(self, obs) -> list[int]:
        """H-37/H-32/B-09: prefer single-prize benchers (less prize liability), HP as tiebreak."""
        scored = []
        for i, o in enumerate(obs.select.option):
            c = self._cards.get(o.cardId) if o.cardId is not None else None
            liability = _prize_value(c) if c else 1          # 1 / 2 / 3 prizes on KO
            scored.append((-liability, c.hp if c else 0, i))  # fewest prizes first, then HP
        scored.sort(reverse=True)
        return [i for *_, i in scored]

    # ---- helpers -----------------------------------------------------------------------------
    def _card_of(self, o, hand):
        if o.index is not None and o.index < len(hand):
            return self._cards.get(hand[o.index].id)
        return None

    @staticmethod
    def _counted(sel, ranked: list[int]) -> list[int]:
        k = min(max(int(sel.minCount), 1), int(sel.maxCount), len(ranked)) if int(sel.maxCount) >= 1 \
            else int(sel.minCount)
        return sorted(ranked[:k])

    @staticmethod
    def _max_number(opts) -> int:
        best_i, best_n = 0, float("-inf")
        for i, o in enumerate(opts):
            n = o.number if o.number is not None else 0
            if n > best_n:
                best_i, best_n = i, n
        return best_i
