"""HeuristicPolicy (Phase 1) — rule-based play grounded in the heuristics knowledge base.

Maps each `SelectContext` to logic derived from `heuristics_beginner.json`'s executable
`turn_policy`/`opening_policy` + the prize-trade heuristics (H-31..H-39). MAIN is handled by scoring
each legal option so that setup actions resolve before attacks (turn_policy order), then attacking.
Unhandled contexts fall through to the always-legal default. Never raises; never returns illegal.

This is a general v1 (works across decks via card data, not deck-specific hardcoding). Tune the
SCORE_* constants from gauntlet results.
"""
from __future__ import annotations

from cg.api import (CardType, OptionType, SelectContext, all_attack, all_card_data,
                    to_observation_class)

from bot.engine_io import safe_selection
from bot.policies.base import Policy

# MAIN option base priorities (turn_policy order: abilities/draw/search/evolve before attaching,
# attaching before attacking, attacking before ending). Tunable.
PRItem = {
    OptionType.ABILITY: 80,   # draw/search abilities first (after we can see our hand)
    OptionType.EVOLVE: 70,    # advance toward the main attacker
    OptionType.PLAY: 60,      # supporters/items (refined below by card type)
    OptionType.ATTACH: 45,    # energy onto an attacker
    OptionType.ATTACK: 25,    # after setup is exhausted (boosted by damage below)
    OptionType.END: 2,        # end the turn rather than make a negative-value play
    OptionType.DISCARD: 1,    # don't volunteer to discard our own resources
    OptionType.RETREAT: 0,    # don't retreat without a reason (burns energy) — refine later
}
SCORE_SUPPORTER_BONUS = 25    # draw/utility supporter ranks near the top of the turn
SCORE_ATTACH_ACTIVE_BONUS = 20
SCORE_ATTACK_DMG_DIV = 8.0    # +damage/8 so a big attack can beat low-value setup
SCORE_ATTACK_KO_BONUS = 120   # an attack that KOs the opponent's active takes a prize → do it


def _opp_active_hp(obs) -> int | None:
    opp = obs.current.players[1 - obs.current.yourIndex]
    if opp.active and opp.active[0] is not None:
        return opp.active[0].hp
    return None


# Module-level caches: the native card/attack tables load once per process (not per game/instance).
_CARD_TABLE = None
_ATTACK_TABLE = None


class HeuristicPolicy(Policy):
    def __init__(self):
        self._cards = None
        self._attacks = None

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
        opts = sel.option
        if not opts:
            return safe_selection(obs_dict["select"])

        if ctx == SelectContext.MAIN:
            return [self._best_main(obs)]
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE, SelectContext.SWITCH):
            return self._counted(sel, self._rank_pokemon_options(obs))
        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH, SelectContext.TO_FIELD):
            return self._counted(sel, self._rank_pokemon_options(obs))
        if ctx == SelectContext.ATTACK:
            return [self._best_attack(obs)]
        if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                   SelectContext.DAMAGE_COUNTER_ANY):
            return self._counted(sel, self._rank_damage_targets(obs))
        if ctx == SelectContext.DRAW_COUNT:
            return [self._max_number_option(opts)]          # draw as many as offered
        # yes/no & everything else → default-safe (legal, neutral). Refined in later sprints.
        return safe_selection(obs_dict["select"])

    # ---- MAIN --------------------------------------------------------------------------------
    def _best_main(self, obs) -> int:
        from cg.api import AreaType
        you = obs.current.yourIndex
        hand = obs.current.players[you].hand or []
        opp_hp = _opp_active_hp(obs)
        best_i, best_s = 0, float("-inf")
        for i, o in enumerate(obs.select.option):
            s = PRItem.get(o.type, 5)
            if o.type == OptionType.PLAY and o.index is not None and o.index < len(hand):
                card = self._cards.get(hand[o.index].id)
                if card and card.cardType == CardType.SUPPORTER:
                    s += SCORE_SUPPORTER_BONUS
            elif o.type == OptionType.ATTACH and getattr(o, "inPlayArea", None) is not None:
                if o.inPlayArea == AreaType.ACTIVE:
                    s += SCORE_ATTACH_ACTIVE_BONUS
            elif o.type == OptionType.ATTACK and o.attackId is not None:
                atk = self._attacks.get(o.attackId)
                if atk:
                    s += atk.damage / SCORE_ATTACK_DMG_DIV
                    if opp_hp is not None and atk.damage >= opp_hp:   # KO → take a prize
                        s += SCORE_ATTACK_KO_BONUS
            if s > best_s:
                best_i, best_s = i, s
        return best_i

    # ---- attacks -----------------------------------------------------------------------------
    def _best_attack(self, obs) -> int:
        opp_hp = _opp_active_hp(obs)
        best_i, best_s = 0, float("-inf")
        for i, o in enumerate(obs.select.option):
            dmg = 0
            if o.attackId is not None:
                atk = self._attacks.get(o.attackId)
                dmg = atk.damage if atk else 0
            s = dmg + (SCORE_ATTACK_KO_BONUS if opp_hp is not None and dmg >= opp_hp else 0)
            if s > best_s:
                best_i, best_s = i, s
        return best_i

    # ---- pokemon ranking (setup / promote): prefer higher-HP, non-2-prize liabilities --------
    def _rank_pokemon_options(self, obs) -> list[int]:
        scored = []
        for i, o in enumerate(obs.select.option):
            card = self._cards.get(o.cardId) if o.cardId is not None else None
            score = 0
            if card:
                score = card.hp - (50 if (card.ex or card.megaEx) else 0)  # avoid loose 2-prizers
            scored.append((score, i))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    # ---- damage targeting: hit the highest-value opponent Pokémon (prize trade) ---------------
    def _rank_damage_targets(self, obs) -> list[int]:
        scored = []
        for i, o in enumerate(obs.select.option):
            card = self._cards.get(o.cardId) if o.cardId is not None else None
            value = 0
            if card:
                value = (200 if card.megaEx else 100 if card.ex else 50)  # prizes on KO
            scored.append((value, i))
        scored.sort(reverse=True)
        return [i for _, i in scored]

    # ---- helpers -----------------------------------------------------------------------------
    @staticmethod
    def _counted(sel, ranked_indices: list[int]) -> list[int]:
        """Take the top-`maxCount` ranked indices, at least `minCount`."""
        k = max(int(sel.minCount), 1) if int(sel.maxCount) >= 1 else int(sel.minCount)
        k = min(k, int(sel.maxCount), len(ranked_indices))
        return sorted(ranked_indices[:k])

    @staticmethod
    def _max_number_option(opts) -> int:
        best_i, best_n = 0, float("-inf")
        for i, o in enumerate(opts):
            n = o.number if o.number is not None else 0
            if n > best_n:
                best_i, best_n = i, n
        return best_i
