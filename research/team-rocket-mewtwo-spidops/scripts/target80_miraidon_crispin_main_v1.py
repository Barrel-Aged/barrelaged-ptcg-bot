"""Zero-Wobbuffet Rocket/Miraidon research route for public Starmie boards.

The neural policy remains the default.  After Staryu or Mega Starmie ex is
public, a narrow deterministic bridge may acquire, charge, promote, and attack
with Miraidon ex.  This bridge is a structural mechanism probe, not a final
planner or a deployment-strength claim.
"""
from __future__ import annotations

import json
import os
import sys

_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else "/kaggle_simulations/agent"
)
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import torch  # noqa: E402

torch.set_num_threads(1)

from soldier.cards import CARDS  # noqa: E402
from soldier.features import _opt_card_id  # noqa: E402
from soldier.net import NNAgent, load_policy  # noqa: E402


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
_STARMIE_SEEN = False
_PENDING_MIRAIDON_SWITCH = False


POKE_PAD = 1152
ULTRA_BALL = 1121
CRISPIN = 1198
BASIC_GRASS = 1
BASIC_LIGHTNING = 4
TAROUNTULA = 400
SPIDOPS = 401
MIRAIDON_EX = 957
STARYU = 1030
MEGA_STARMIE_EX = 1031
BRAVE_BANGLE = 1175
ROCKET_RUSH = 560
HADRON_SPARK = 1377

ACTIVE = 4
BENCH = 5
MAIN = 0
TO_ACTIVE = 4
TO_HAND = 7
DISCARD = 8
ATTACH_FROM = 21
ATTACH_TO = 22


def _players(observation: dict):
    current = observation.get("current") or {}
    me = current.get("yourIndex")
    players = current.get("players") or []
    if me not in (0, 1) or len(players) != 2:
        return current, None, {}, {}
    return current, int(me), players[me] or {}, players[1 - me] or {}


def _board(player: dict) -> list[dict]:
    return [
        card
        for card in ((player.get("active") or []) + (player.get("bench") or []))
        if isinstance(card, dict)
    ]


def _observe_starmie(observation: dict) -> bool:
    global _STARMIE_SEEN
    _, me, _, theirs = _players(observation)
    if me is not None and any(
        card.get("id") in (STARYU, MEGA_STARMIE_EX) for card in _board(theirs)
    ):
        _STARMIE_SEEN = True
    return _STARMIE_SEEN


def _miraidon_slots(mine: dict) -> list[tuple[int, int, dict]]:
    answer: list[tuple[int, int, dict]] = []
    for index, card in enumerate(mine.get("active") or []):
        if isinstance(card, dict) and card.get("id") == MIRAIDON_EX:
            answer.append((ACTIVE, index, card))
    for index, card in enumerate(mine.get("bench") or []):
        if isinstance(card, dict) and card.get("id") == MIRAIDON_EX:
            answer.append((BENCH, index, card))
    return answer


def _energy_units(card: dict) -> tuple[int, int]:
    units = card.get("energies") or []
    total = len(units)
    lightning = sum(
        value == BASIC_LIGHTNING
        or (isinstance(value, dict) and value.get("type") == BASIC_LIGHTNING)
        for value in units
    )
    if lightning == 0:
        lightning = sum(
            isinstance(value, dict) and value.get("id") == BASIC_LIGHTNING
            for value in (card.get("energyCards") or [])
        )
    return total, lightning


def _miraidon_ready(card: dict) -> bool:
    total, lightning = _energy_units(card)
    return total >= 3 and lightning >= 2


def _card_id(option: dict, select: dict, current: dict, me: int) -> int:
    return int(_opt_card_id(option, current, me, select) or 0)


def _choose_ids(
    select: dict,
    current: dict,
    me: int,
    wanted: list[int],
) -> list[int] | None:
    options = select.get("option") or []
    maximum = min(int(select.get("maxCount") or 0), len(options))
    minimum = min(int(select.get("minCount") or 0), maximum)
    picked: list[int] = []
    for wanted_id in wanted:
        for index, option in enumerate(options):
            if index not in picked and _card_id(option, select, current, me) == wanted_id:
                picked.append(index)
                break
        if len(picked) >= maximum:
            break
    return picked if len(picked) >= minimum else None


def _target_option(
    select: dict,
    area: int,
    index: int,
    *,
    option_type: int | None = None,
) -> int | None:
    for option_index, option in enumerate(select.get("option") or []):
        if option_type is not None and option.get("type") != option_type:
            continue
        option_area = option.get("inPlayArea", option.get("area"))
        option_index_in_area = option.get("inPlayIndex", option.get("index"))
        if option_area == area and option_index_in_area == index:
            return option_index
    return None


def _ultra_ball_prompt_override(
    observation: dict, current: dict, me: int, mine: dict
) -> list[int] | None:
    select = observation.get("select") or {}
    effect = select.get("effect") or {}
    if effect.get("id") != ULTRA_BALL:
        return None
    options = select.get("option") or []
    if any(
        _card_id(option, select, current, me) == MIRAIDON_EX
        for option in options
    ):
        return _choose_ids(select, current, me, [MIRAIDON_EX])
    if select.get("context") == DISCARD:
        protected = {MIRAIDON_EX, BASIC_LIGHTNING, BASIC_GRASS, CRISPIN}
        ranked = sorted(
            range(len(options)),
            key=lambda index: (
                _card_id(options[index], select, current, me) in protected,
                _card_id(options[index], select, current, me) in (SPIDOPS, TAROUNTULA),
                index,
            ),
        )
        minimum = int(select.get("minCount") or 0)
        maximum = min(int(select.get("maxCount") or 0), len(ranked))
        count = maximum if maximum >= minimum else minimum
        return ranked[:count]
    return None


def _crispin_prompt_override(
    observation: dict,
    current: dict,
    me: int,
    mine: dict,
) -> list[int] | None:
    select = observation.get("select") or {}
    effect = select.get("effect") or {}
    if effect.get("id") != CRISPIN:
        return None
    slots = _miraidon_slots(mine)
    if not slots:
        return None
    area, slot_index, miraidon = slots[0]
    total, lightning = _energy_units(miraidon)
    context = select.get("context")
    options = select.get("option") or []

    # Initial deck search: acquire one Lightning and one Grass.
    deck_choices = [
        index
        for index, option in enumerate(options)
        if option.get("area") == 1
        and _card_id(option, select, current, me)
        in (BASIC_LIGHTNING, BASIC_GRASS)
    ]
    if deck_choices and int(select.get("maxCount") or 0) >= 2:
        return _choose_ids(
            select, current, me, [BASIC_LIGHTNING, BASIC_GRASS]
        )

    # If the engine asks which searched Energy goes to hand, leave Grass in
    # hand while Miraidon still needs a second Lightning attachment.
    if context == TO_HAND:
        preferred = (
            [BASIC_GRASS, BASIC_LIGHTNING]
            if lightning < 2
            else [BASIC_LIGHTNING, BASIC_GRASS]
        )
        choice = _choose_ids(select, current, me, preferred)
        if choice is not None:
            return choice

    # If it asks which Energy to attach, choose Lightning first.
    if context == ATTACH_FROM:
        preferred = (
            [BASIC_LIGHTNING, BASIC_GRASS]
            if lightning < 2
            else [BASIC_GRASS, BASIC_LIGHTNING]
        )
        choice = _choose_ids(select, current, me, preferred)
        if choice is not None:
            return choice

    if context == ATTACH_TO:
        target = _target_option(select, area, slot_index)
        if target is not None:
            return [target]
    return None


def _miraidon_starmie_override(observation: dict) -> list[int] | None:
    global _PENDING_MIRAIDON_SWITCH
    if not _observe_starmie(observation):
        return None
    select = observation.get("select") or {}
    current, me, mine, _ = _players(observation)
    if me is None:
        return None

    prompt = _ultra_ball_prompt_override(observation, current, me, mine)
    if prompt is not None:
        return prompt
    prompt = _crispin_prompt_override(observation, current, me, mine)
    if prompt is not None:
        return prompt

    slots = _miraidon_slots(mine)
    context = select.get("context")
    if context in (TO_ACTIVE, 3) and slots:
        ready = [slot for slot in slots if _miraidon_ready(slot[2])]
        chosen = ready[0] if ready else slots[0]
        target = _target_option(select, chosen[0], chosen[1])
        if target is not None and (_PENDING_MIRAIDON_SWITCH or ready):
            _PENDING_MIRAIDON_SWITCH = False
            return [target]

    if select.get("type") != MAIN:
        return None
    options = select.get("option") or []
    hand = mine.get("hand") or []

    active = (mine.get("active") or [])
    active_miraidon = bool(
        active and isinstance(active[0], dict) and active[0].get("id") == MIRAIDON_EX
    )
    if active_miraidon:
        for index, option in enumerate(options):
            if option.get("type") == 13 and option.get("attackId") == HADRON_SPARK:
                return [index]

    if not slots:
        # Bench Miraidon from hand before spending cards on search.
        for index, option in enumerate(options):
            if option.get("type") == 7 and _card_id(option, select, current, me) == MIRAIDON_EX:
                return [index]
        # Ultra Ball is the only routed acquisition action. The prompt guard
        # above handles discards and chooses Miraidon if it remains in deck.
        for index, option in enumerate(options):
            if option.get("type") == 7 and _card_id(option, select, current, me) == ULTRA_BALL:
                return [index]
        return None

    area, slot_index, miraidon = slots[0]
    total, lightning = _energy_units(miraidon)
    if (total < 3 or lightning < 2) and not bool(current.get("supporterPlayed")):
        for index, option in enumerate(options):
            if option.get("type") == 7 and _card_id(option, select, current, me) == CRISPIN:
                return [index]

    # Manual attachment after Crispin: satisfy the two-Lightning requirement,
    # then the remaining Colorless unit with Grass or Lightning.
    attachment_preferences = (
        [BASIC_LIGHTNING]
        if lightning < 2
        else [BASIC_GRASS, BASIC_LIGHTNING]
    )
    for preferred in attachment_preferences:
        for index, option in enumerate(options):
            if option.get("type") != 8:
                continue
            hand_index = option.get("index")
            if not isinstance(hand_index, int) or not 0 <= hand_index < len(hand):
                continue
            card = hand[hand_index]
            if not isinstance(card, dict) or card.get("id") != preferred:
                continue
            if (
                option.get("inPlayArea") == area
                and option.get("inPlayIndex") == slot_index
            ):
                return [index]

    if not active_miraidon and _miraidon_ready(miraidon):
        for index, option in enumerate(options):
            if option.get("type") == 12:
                _PENDING_MIRAIDON_SWITCH = True
                return [index]
    return None


def _early_spidops_search_override(observation: dict):
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
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    if select.get("type") != MAIN:
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
    if not (target_meta.ex or getattr(target_meta, "megaEx", False)):
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
        if option.get("type") != 8:
            continue
        hand_index = option.get("index")
        if not isinstance(hand_index, int) or not 0 <= hand_index < len(hand):
            continue
        card = hand[hand_index]
        if not isinstance(card, dict) or card.get("id") != BRAVE_BANGLE:
            continue
        if option.get("inPlayArea") == ACTIVE and option.get("inPlayIndex") == 0:
            return [option_index]
    return None


def _fresh_agent() -> NNAgent:
    return NNAgent(
        _NET,
        greedy=True,
        deck_ids=_DECK,
        forbid_end_when_attack=True,
        opponent_deck_priors=_OPPONENT_DECK_PRIORS,
    )


def agent(observation: dict):
    global _AGENT, _STARMIE_SEEN, _PENDING_MIRAIDON_SWITCH
    if observation.get("select") is None:
        _AGENT = _fresh_agent()
        _STARMIE_SEEN = False
        _PENDING_MIRAIDON_SWITCH = False
        return _DECK
    route = _miraidon_starmie_override(observation)
    if route is not None:
        return route
    search = _early_spidops_search_override(observation)
    if search is not None:
        return search
    bangle = _spidops_bangle_lethal_override(observation)
    if bangle is not None:
        return bangle
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _AGENT(observation)
