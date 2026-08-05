"""Wobbuffet-free ScaleBC policy with validated tactical and board guards."""
from __future__ import annotations

import os
import sys
import json

_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/kaggle_simulations/agent"
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import torch  # noqa: E402

torch.set_num_threads(1)

from soldier import features as _features  # noqa: E402
from soldier.net import (  # noqa: E402
    NNAgent,
    attack_safe_logits,
    feat_to_tensors,
    load_policy,
)
from soldier.features import _opt_card_id  # noqa: E402
from soldier.cards import CARDS  # noqa: E402


def _find(name: str) -> str:
    local = os.path.join(_DIR, name)
    return local if os.path.exists(local) else "/kaggle_simulations/agent/" + name


with open(_find("deck.csv"), encoding="utf-8") as handle:
    _DECK = [int(line) for line in handle.read().splitlines() if line.strip()][:60]

_NET = load_policy(_find("model.pt")).eval()
_DECK_LIBRARY_PATH = _find("deck_library.json")
if os.path.exists(_DECK_LIBRARY_PATH):
    with open(_DECK_LIBRARY_PATH, encoding="utf-8") as handle:
        _OPPONENT_DECK_PRIORS = json.load(handle)
else:
    _OPPONENT_DECK_PRIORS = None
_AGENT = None


POKE_PAD = 1152
TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
MEWTWO_EX = 431
MIMIKYU = 434
BRAVE_BANGLE = 1175
ROCKET_RUSH = 560
OPTION_PLAY = 7
OPTION_ATTACK = 13
OPTION_END = 14


# Repelling Veil stops effects of attacks done to Basic Team Rocket Pokemon,
# but not damage or Abilities.  These public evolution families identify the
# official-engine matchups in which preserving the fifth bench slot has a
# concrete defensive purpose.  A family signal may be in play or discarded;
# hidden cards and the opponent's private hand are never inspected.
_EFFECT_ATTACK_FAMILY_NAMES = frozenset(
    {
        "Dreepy",
        "Drakloak",
        "Dragapult ex",
        "Abra",
        "Kadabra",
        "Alakazam",
        "Snorunt",
        "Froslass",
        "Mega Froslass ex",
        "Duskull",
        "Dusclops",
        "Dusknoir",
        "Munkidori",
        "Munkidori ex",
        "Team Rocket's Murkrow",
        "Team Rocket's Wobbuffet",
    }
)


def _early_spidops_search_override(observation: dict):
    """Turn an early Poke Pad into a missing Spidops evolution card.

    In 610 official live games, when both Tarountula and Spidops were legal Pad
    targets and an unevolved Tarountula lacked a matching Spidops in hand, the
    live policy still chose another Tarountula 347 times.  This guard applies
    only through turn four and stops as soon as every in-play Tarountula has an
    evolution card reserved in hand.
    """
    select = observation.get("select") or {}
    effect = select.get("effect") or {}
    if effect.get("id") != POKE_PAD:
        return None
    current = observation.get("current") or {}
    if int(current.get("turn") or 0) > 4:
        return None
    your_index = current.get("yourIndex")
    players = current.get("players") or []
    if your_index not in (0, 1) or len(players) != 2:
        return None
    mine = players[your_index] or {}
    board = (mine.get("active") or []) + (mine.get("bench") or [])
    tarountula = sum(
        isinstance(card, dict) and card.get("id") == TAROUNTULA for card in board
    )
    hand_spidops = sum(
        isinstance(card, dict) and card.get("id") == SPIDOPS
        for card in (mine.get("hand") or [])
    )
    if tarountula <= hand_spidops:
        return None
    for index, option in enumerate(select.get("option") or []):
        if _opt_card_id(option, current, your_index, select) == SPIDOPS:
            return [index]
    return None


def _spidops_bangle_lethal_override(observation: dict):
    """Attach Bangle only when it converts a legal Rocket Rush into an ex KO."""
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    if select.get("type") != 0:  # MAIN
        return None
    your_index = current.get("yourIndex")
    players = current.get("players") or []
    if your_index not in (0, 1) or len(players) != 2:
        return None
    mine = players[your_index] or {}
    theirs = players[1 - your_index] or {}
    hand = mine.get("hand") or []
    active = mine.get("active") or []
    opponent_active = theirs.get("active") or []
    if not active or not opponent_active:
        return None
    if not isinstance(active[0], dict) or active[0].get("id") != SPIDOPS:
        return None
    target = opponent_active[0]
    if not isinstance(target, dict) or target.get("id") not in CARDS:
        return None
    target_meta = CARDS[target["id"]]
    if not target_meta.ex:
        return None
    if not any(
        option.get("type") == 13 and option.get("attackId") == ROCKET_RUSH
        for option in select.get("option") or []
    ):
        return None
    in_play = active + (mine.get("bench") or [])
    rocket_count = sum(
        isinstance(card, dict)
        and card.get("id") in CARDS
        and CARDS[card["id"]].name.startswith("Team Rocket's")
        for card in in_play
    )
    multiplier = 2 if target_meta.weakness == 1 else 1
    resistance = 30 if target_meta.resistance == 1 else 0
    base_damage = max(0, 30 * rocket_count * multiplier - resistance)
    boosted_damage = max(0, (30 * rocket_count + 30) * multiplier - resistance)
    remaining_hp = int(target.get("hp") or 0)
    if not base_damage < remaining_hp <= boosted_damage:
        return None
    for option_index, option in enumerate(select.get("option") or []):
        if option.get("type") != 8:  # ATTACH
            continue
        hand_index = option.get("index")
        if not isinstance(hand_index, int) or not 0 <= hand_index < len(hand):
            continue
        card = hand[hand_index]
        if not isinstance(card, dict) or card.get("id") != BRAVE_BANGLE:
            continue
        if option.get("inPlayArea") == 4 and option.get("inPlayIndex") == 0:
            return [option_index]
    return None


def _opponent_revealed_effect_attack_family(theirs: dict) -> bool:
    """Return whether public opponent cards reveal a Repelling Veil matchup."""
    public_cards = (
        (theirs.get("active") or [])
        + (theirs.get("bench") or [])
        + (theirs.get("discard") or [])
    )
    for card in public_cards:
        if not isinstance(card, dict):
            continue
        meta = CARDS.get(card.get("id"))
        if meta is not None and meta.name in _EFFECT_ATTACK_FAMILY_NAMES:
            return True
    return False


def _articuno_bench_plan(observation: dict):
    """Force Articuno into the last slot or mask only redundant Basic plays.

    The guard is deliberately narrow: it activates at four occupied bench
    slots, only after the opponent publicly reveals an attack-effect family,
    and only while no Articuno is already in play.  If Articuno is available it
    is played.  Otherwise, the fifth slot is withheld only from a duplicate
    Mewtwo/Mimikyu or a fourth Tarountula; core setup plays remain available.
    """
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    if select.get("type") != 0 or int(select.get("maxCount") or 0) != 1:
        return None, frozenset()
    your_index = current.get("yourIndex")
    players = current.get("players") or []
    if your_index not in (0, 1) or len(players) != 2:
        return None, frozenset()
    mine = players[your_index] or {}
    theirs = players[1 - your_index] or {}
    bench = mine.get("bench") or []
    if len(bench) != 4 or not _opponent_revealed_effect_attack_family(theirs):
        return None, frozenset()

    board = (mine.get("active") or []) + bench
    board_ids = [
        card.get("id") for card in board if isinstance(card, dict)
    ]
    if ARTICUNO in board_ids:
        return None, frozenset()

    options = select.get("option") or []
    for option_index, option in enumerate(options):
        if option.get("type") != OPTION_PLAY:
            continue
        if _opt_card_id(option, current, your_index, select) == ARTICUNO:
            return [option_index], frozenset()

    forbidden = set()
    for option_index, option in enumerate(options):
        if option.get("type") != OPTION_PLAY:
            continue
        card_id = _opt_card_id(option, current, your_index, select)
        meta = CARDS.get(card_id)
        if meta is None or meta.cardType != 0 or not meta.basic:
            continue
        if card_id in (MEWTWO_EX, MIMIKYU) and card_id in board_ids:
            forbidden.add(option_index)
        elif card_id == TAROUNTULA and board_ids.count(TAROUNTULA) >= 3:
            forbidden.add(option_index)
    return None, frozenset(forbidden)


def _fallback_allowed_action(select: dict, allowed: list[int]):
    """Return a deterministic legal fallback without reopening masked plays."""
    options = select.get("option") or []
    for index in allowed:
        if options[index].get("type") == OPTION_ATTACK:
            return [index]
    for index in allowed:
        if options[index].get("type") != OPTION_END:
            return [index]
    return [allowed[0]]


@torch.no_grad()
def _masked_greedy_action(observation: dict, forbidden: frozenset[int]):
    """Run the frozen policy once while removing reserved-slot plays."""
    select = observation.get("select") or {}
    options = select.get("option") or []
    allowed = [index for index in range(len(options)) if index not in forbidden]
    if not allowed:
        return _AGENT(observation)
    try:
        base = _features.extract(
            observation,
            belief=_AGENT.belief,
            slot_link=_AGENT.slot_link,
        )
        feat = feat_to_tensors(base, _AGENT.device)
        logits, _ = _AGENT.net(feat)
        if _AGENT.forbid_end_when_attack:
            logits = attack_safe_logits(logits, options)
        logits = logits.clone()
        for index in forbidden:
            logits[index] = -torch.inf
        index = int(torch.argmax(logits))
        if index in allowed and bool(torch.isfinite(logits[index])):
            return [index]
    except Exception:
        pass
    return _fallback_allowed_action(select, allowed)


def _fresh_agent() -> NNAgent:
    return NNAgent(
        _NET,
        greedy=True,
        deck_ids=_DECK,
        forbid_end_when_attack=True,
        opponent_deck_priors=_OPPONENT_DECK_PRIORS,
    )


def agent(observation: dict):
    """Return the deck at setup and legal policy actions during a game."""
    global _AGENT
    if observation.get("select") is None:
        # A module can be reused across arena games; belief state must not be.
        _AGENT = _fresh_agent()
        return _DECK
    search_override = _early_spidops_search_override(observation)
    if search_override is not None:
        return search_override
    bangle_override = _spidops_bangle_lethal_override(observation)
    if bangle_override is not None:
        return bangle_override
    articuno_override, forbidden = _articuno_bench_plan(observation)
    if articuno_override is not None:
        return articuno_override
    if _AGENT is None:
        _AGENT = _fresh_agent()
    if forbidden:
        return _masked_greedy_action(observation, forbidden)
    return _AGENT(observation)
