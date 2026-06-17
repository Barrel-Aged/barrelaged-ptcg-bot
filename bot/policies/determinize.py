"""Determinization for the Search API — fill in hidden info so `search_begin` can run.

IS-MCTS needs a concrete guess of every hidden card: our own deck+prizes (we know our decklist but not
the order/which are prized) and the opponent's deck+hand+prizes (from the DeckIdentifier's predicted
archetype). `search_begin` requires the counts to match the real state exactly, so this samples to the
right sizes and pads defensively (a wrong guess just makes one search rollout less accurate).
"""
from __future__ import annotations

import collections
import random

BASIC_ENERGY = 1   # filler when a predicted pool is too small (a valid card id)


def _visible_ids(player) -> list[int]:
    ids = []
    for poke in list(player.active or []) + list(player.bench or []):
        if poke is None:
            continue
        ids.append(poke.id)
        for grp in (poke.energyCards, poke.tools, poke.preEvolution):
            ids.extend(c.id for c in grp or [])
    ids.extend(c.id for c in player.discard or [])
    return ids


def _draw(pool: list[int], n: int, rng: random.Random) -> tuple[list[int], list[int]]:
    """Take n ids from pool (without replacement; pad with BASIC_ENERGY if short). Returns (taken, rest)."""
    if n <= 0:
        return [], pool
    if len(pool) >= n:
        idx = rng.sample(range(len(pool)), n)
        taken = [pool[i] for i in idx]
        rest = [pool[i] for i in range(len(pool)) if i not in set(idx)]
        return taken, rest
    return pool + [BASIC_ENERGY] * (n - len(pool)), []


def _unseen(full: list[int], visible: list[int]) -> list[int]:
    rem = collections.Counter(full)
    for cid in visible:
        if rem[cid] > 0:
            rem[cid] -= 1
    out = []
    for cid, c in rem.items():
        out.extend([cid] * c)
    return out


def determinize(obs, our_deck: list[int], predicted_opp_deck: list[int], rng: random.Random) -> dict:
    """Return kwargs for search_begin: your_deck/your_prize/opponent_deck/opponent_prize/
    opponent_hand/opponent_active, sized to the real state."""
    state = obs.current
    you = state.yourIndex
    me, opp = state.players[you], state.players[1 - you]

    # our hidden cards = decklist minus what we can see (hand is visible to us)
    my_visible = _visible_ids(me) + [c.id for c in (me.hand or [])]
    mine = _unseen(our_deck, my_visible)
    rng.shuffle(mine)
    your_deck, mine = _draw(mine, me.deckCount, rng)
    your_prize, _ = _draw(mine, len(me.prize), rng)

    # opponent hidden cards = predicted decklist minus their visible board/discard
    opp_unseen = _unseen(predicted_opp_deck, _visible_ids(opp))
    rng.shuffle(opp_unseen)
    opponent_deck, opp_unseen = _draw(opp_unseen, opp.deckCount, rng)
    opponent_prize, opp_unseen = _draw(opp_unseen, len(opp.prize), rng)
    opponent_hand, opp_unseen = _draw(opp_unseen, opp.handCount, rng)

    # opponent active only required when it is face-down
    opponent_active = []
    if opp.active and opp.active[0] is None:
        # pick any Basic Pokémon id from the predicted deck as a stand-in (engine requires a Pokémon)
        opponent_active = [predicted_opp_deck[0] if predicted_opp_deck else BASIC_ENERGY]

    return {
        "your_deck": your_deck, "your_prize": your_prize,
        "opponent_deck": opponent_deck, "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand, "opponent_active": opponent_active,
    }
