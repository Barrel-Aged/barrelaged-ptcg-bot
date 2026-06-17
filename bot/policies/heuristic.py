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

# Resource-discipline thresholds (B-02 "if hand low", B-05 "pieces you need", B-11 "keep an out",
# B-22 "don't deck out") are loaded from heuristics_beginner.json's resource_policy — tuning lives in
# the knowledge base, not in code.
_RP = knowledge.resource_policy()
DRAW_IF_HAND_AT_MOST = _RP["draw_supporter_if_hand_at_most"]
SEARCH_IF_HAND_AT_MOST = _RP["search_if_hand_at_most"]
DECK_OUT_GUARD = _RP["deck_out_guard"]

_CARD_TABLE = None
_ATTACK_TABLE = None
_DRAW_ABILITY_IDS = None   # card IDs whose Ability text draws cards (deck-out risk when deck is low)


def _safe_typed(sel) -> list[int]:
    """Always-legal selection from a typed SelectData: the first maxCount option indices."""
    return list(range(int(sel.maxCount)))


def _step_kind(step: dict) -> str | None:
    kinds = {RULE_KIND[r] for r in step.get("rules", []) if r in RULE_KIND}
    for k in KIND_PRECEDENCE:
        if k in kinds:
            return k
    return None


def _prize_value(card) -> int:
    return 3 if card.megaEx else 2 if card.ex else 1   # prizes the opponent takes on KO


def _effective_damage(base: int, attacker_type, defender_card) -> int:
    """Apply Weakness (×2) / Resistance (−30) — the prize-trade lever in the Mega era (H-31/H-35).
    A heuristic estimate (the engine computes exact values incl. attack effects); good enough to
    judge KOs and pick attacks/targets."""
    dmg = base
    if attacker_type is not None and defender_card is not None:
        if defender_card.weakness is not None and defender_card.weakness == attacker_type:
            dmg *= 2
        if defender_card.resistance is not None and defender_card.resistance == attacker_type:
            dmg = max(0, dmg - 30)
    return dmg


class HeuristicPolicy(Policy):
    # NOTE: the default tweaks (gate_draw_abilities_by_hand, dig_when_undeveloped) are A/B-validated
    # wins for AGGRO/MIDRANGE/SETUP decks (+11–15pt) but HURT CONTROL decks (slowking −17pt) — they
    # bias toward committing/attacking. If piloting a control deck, construct with both = False.
    def __init__(self, gate_draw_abilities_by_hand: bool = True, smart_retreat: bool = False,
                 dig_when_undeveloped: bool = True,
                 promote_charged: bool = False):
        self._cards = None
        self._attacks = None
        # Tweak (A/B'd): when the active can't attack, retreat to a clearly-more-charged bench
        # attacker (addresses mid-game stalls). Conservatively gated — a naive version regressed.
        self._smart_retreat = smart_retreat
        # Tweak (A/B'd): when the board is undeveloped (no evolved attacker in play), keep digging
        # for pieces even with a fuller hand — the real-ladder brick (lone Dreepy) sat passively.
        self._dig_when_undeveloped = dig_when_undeveloped
        self._dig_relax = False   # set per-decision in _run_turn_policy
        # Tweak (A/B'd): also skip draw-Abilities when the hand is already large (not just on deck-out)
        # — the traces showed hands bloating to 20+ via greedy ability draws, delaying commitment.
        self._gate_draw_abil_hand = gate_draw_abilities_by_hand
        self._promote_charged = promote_charged   # A/B: promote the most-charged attacker
        # Precompute the ordered (kind, step) plan once from the knowledge base.
        self._plan = [(k, s) for s in knowledge.turn_policy() if (k := _step_kind(s))]

    def _ensure_tables(self):
        global _CARD_TABLE, _ATTACK_TABLE, _DRAW_ABILITY_IDS
        if _CARD_TABLE is None:
            _CARD_TABLE = {c.cardId: c for c in all_card_data()}
            _ATTACK_TABLE = {a.attackId: a for a in all_attack()}
            _DRAW_ABILITY_IDS = {c.cardId for c in _CARD_TABLE.values()
                                 if any("draw" in (sk.text or "").lower() for sk in (c.skills or []))}
        self._cards, self._attacks = _CARD_TABLE, _ATTACK_TABLE

    # ---- dispatch ----------------------------------------------------------------------------
    def select(self, obs_dict: dict, budget=None) -> list[int]:
        self._ensure_tables()
        return self.select_typed(to_observation_class(obs_dict))

    def select_typed(self, obs) -> list[int]:
        """Decide from an already-typed Observation (used by MCTS rollouts over the Search API)."""
        self._ensure_tables()
        sel = obs.select
        ctx = sel.context
        if not sel.option:
            return _safe_typed(sel)

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
        return _safe_typed(sel)

    # ---- the turn_policy interpreter ---------------------------------------------------------
    def _run_turn_policy(self, obs) -> int:
        """Walk turn_policy steps in file order; take the first available action (B-18 sequencing)."""
        opts = obs.select.option
        you = obs.current.yourIndex
        me = obs.current.players[you]
        hand = me.hand or []
        hand_n, deck_n = me.handCount, me.deckCount
        # "Undeveloped" = no evolved (Stage 1/2) Pokémon in play yet → keep digging for the line.
        if self._dig_when_undeveloped:
            in_play = list(me.active or []) + list(me.bench or [])
            self._dig_relax = not any(
                (c := self._cards.get(p.id)) and (c.stage1 or c.stage2) for p in in_play if p)
        for kind, _step in self._plan:
            matches = [i for i, o in enumerate(opts) if self._matches(kind, o, hand, hand_n, deck_n, me)]
            if matches:
                return self._choose(kind, matches, opts, obs)
        # Can't act per policy. Conservative smart-retreat (A/B'd): only when the active CAN'T attack
        # AND a bench Pokémon is clearly more charged (≥2 energy and more than the active) — i.e. a
        # real attacker is waiting. The earlier naive "any bench has energy" version regressed.
        if self._smart_retreat and not any(o.type == OptionType.ATTACK for o in opts):
            retreats = [i for i, o in enumerate(opts) if o.type == OptionType.RETREAT]
            if retreats:
                active_e = len(me.active[0].energies) if (me.active and me.active[0]) else 0
                best_bench_e = max((len(p.energies) for p in (me.bench or [])), default=0)
                if best_bench_e >= 2 and best_bench_e > active_e:
                    return retreats[0]
        for i, o in enumerate(opts):
            if o.type == OptionType.END:
                return i
        return 0

    def _matches(self, kind: str, o, hand, hand_n: int, deck_n: int, me) -> bool:
        t = o.type
        draw_cap = (8 if self._dig_relax else DRAW_IF_HAND_AT_MOST)
        search_cap = (8 if self._dig_relax else SEARCH_IF_HAND_AT_MOST)
        if kind == "draw_supporter":
            if hand_n > draw_cap or deck_n <= DECK_OUT_GUARD:
                return False   # hand not low / would risk decking out (B-02, B-22)
            return t == OptionType.PLAY and self._card_of(o, hand) and \
                self._card_of(o, hand).cardType == CardType.SUPPORTER
        if kind == "item_search":
            if hand_n > search_cap or deck_n <= DECK_OUT_GUARD:
                return False   # don't over-thin (B-05 "pieces you need", B-22)
            return t == OptionType.PLAY and self._card_of(o, hand) and \
                self._card_of(o, hand).cardType == CardType.ITEM
        if kind == "bench":
            c = self._card_of(o, hand)
            return t == OptionType.PLAY and c and c.cardType == CardType.POKEMON and c.basic
        if kind == "evolve":
            return t == OptionType.EVOLVE
        if kind == "ability":
            if t != OptionType.ABILITY:
                return False
            # Skip draw-Abilities on deck-out risk (B-22) or — if the tweak is on — a bloated hand.
            if deck_n <= DECK_OUT_GUARD or (self._gate_draw_abil_hand and hand_n > DRAW_IF_HAND_AT_MOST):
                cid = self._board_card_id(o, me)
                if cid is not None and cid in _DRAW_ABILITY_IDS:
                    return False
            return True
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
    def _my_active_type(self, obs):
        me = obs.current.players[obs.current.yourIndex]
        if me.active and me.active[0] and me.active[0].id in self._cards:
            return self._cards[me.active[0].id].energyType
        return None

    def _best_attack(self, obs, restrict=None) -> int:
        opp = obs.current.players[1 - obs.current.yourIndex]
        opp_card = self._cards.get(opp.active[0].id) if (opp.active and opp.active[0]) else None
        opp_hp = opp.active[0].hp if (opp.active and opp.active[0]) else None
        opp_val = _prize_value(opp_card) if opp_card else 1
        atk_type = self._my_active_type(obs)
        idxs = list(restrict) if restrict is not None else list(range(len(obs.select.option)))
        best_i, best_s = idxs[0], float("-inf")
        for i in idxs:
            o = obs.select.option[i]
            atk = self._attacks.get(o.attackId) if o.attackId is not None else None
            eff = _effective_damage(atk.damage if atk else 0, atk_type, opp_card)
            s = eff
            if opp_hp is not None and eff >= opp_hp:
                s += 120 + 60 * opp_val   # KO bonus, scaled by prizes taken (B-15/H-31)
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
        """H-33: promote the best attacker. With promote_charged, prefer the most-CHARGED Pokémon
        (energy attached) so the new active can actually attack next turn, then HP."""
        me = obs.current.players[obs.current.yourIndex]
        bench = me.bench or []
        scored = []
        for i, o in enumerate(obs.select.option):
            c = self._cards.get(o.cardId) if o.cardId is not None else None
            energy = 0
            if self._promote_charged and o.area == AreaType.BENCH and o.index is not None \
                    and o.index < len(bench):
                energy = len(bench[o.index].energies)
            scored.append((energy, c.hp if c else 0, i))
        scored.sort(reverse=True)
        return [i for *_, i in scored]

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
    def _board_card_id(o, me):
        """Resolve the CardData id of an in-play option (Ability) from its area/index."""
        poke = None
        if o.area == AreaType.ACTIVE and me.active:
            poke = me.active[0]
        elif o.area == AreaType.BENCH and o.index is not None and o.index < len(me.bench):
            poke = me.bench[o.index]
        return poke.id if poke else None

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
