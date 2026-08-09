"""Public Starmie conversion probe using Tera Pikachu ex and Sparkling Crystal.

Sparkling Crystal reduces Topaz Bolt to two attached Energy. Crispin plus the
manual attachment can therefore ready Pikachu in one turn, and its 300 damage
exploits Mega Starmie ex's Lightning Weakness without a bench-width condition.
The neural policy remains the default outside public Staryu/Starmie states.

This deterministic bridge is a mechanism probe and prospective teacher only.
It is not eligible for deployment as the final strategic policy.
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
_PENDING_KOKO_SWITCH = False


BASIC_GRASS = 1
BASIC_LIGHTNING = 4
BASIC_PSYCHIC = 5
ROCKET_ENERGY = 15

TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
MEWTWO_EX = 431
MIMIKYU = 434
TAPU_KOKO_EX = 210  # Legacy helper name; this route's attacker is Pikachu ex.
STARYU = 1030
MEGA_STARMIE_EX = 1031

BUDDY_POFFIN = 1086
BUG_CATCHING_SET = 1094
NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
SWITCH = 1123
TRANSCEIVER = 1134
POKE_PAD = 1152
MAXIMUM_BELT = 1165  # Legacy helper name; this route uses Sparkling Crystal.
CRISPIN = 1198
ARIANA = 1216
GIOVANNI = 1218
PETREL = 1219
PROTON = 1220
LILLIE = 1227
FACTORY = 1257

LINKED_LIGHTNING = 282  # Legacy helper name; this route uses Topaz Bolt.

MAIN = 0
PLAY = 7
ATTACH = 8
RETREAT = 12
ATTACK = 13
END_TURN = 14
ACTIVE = 4
BENCH = 5
DISCARD = 8
TO_ACTIVE = 3
TARGET_ACTIVE = 4
# Crispin's official-engine sequence uses context 22 to choose the searched
# Energy that will be attached, then context 21 to choose the target Pokemon.
# The generic enum names are easy to read backwards here, so keep route-local
# names that describe the actual prompt payload seen in official traces.
CRISPIN_TARGET = 21
CRISPIN_ENERGY_CHOICE = 22

_ROCKET_BASICS = (TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO)
_DISCARDABLE = (
    ARIANA,
    LILLIE,
    FACTORY,
    GIOVANNI,
    PROTON,
    TRANSCEIVER,
    BUG_CATCHING_SET,
    BUDDY_POFFIN,
)


def _view(observation: dict):
    current = observation.get("current") or {}
    me = current.get("yourIndex")
    players = current.get("players") or []
    if me not in (0, 1) or len(players) != 2:
        return current, None, {}, {}
    return current, int(me), players[me] or {}, players[1 - me] or {}


def _zone(player: dict, name: str) -> list[dict]:
    return [card for card in player.get(name) or [] if isinstance(card, dict)]


def _board(player: dict) -> list[dict]:
    return _zone(player, "active") + _zone(player, "bench")


def _observe_starmie(observation: dict) -> bool:
    global _STARMIE_SEEN
    _, me, _, theirs = _view(observation)
    if me is not None and any(
        card.get("id") in (STARYU, MEGA_STARMIE_EX) for card in _board(theirs)
    ):
        _STARMIE_SEEN = True
    return _STARMIE_SEEN


def _card_id(option: dict, select: dict, current: dict, me: int) -> int:
    return int(_opt_card_id(option, current, me, select) or 0)


def _choose_ids(
    select: dict,
    current: dict,
    me: int,
    wanted: list[int] | tuple[int, ...],
) -> list[int] | None:
    options = select.get("option") or []
    maximum = min(int(select.get("maxCount") or 0), len(options))
    minimum = min(int(select.get("minCount") or 0), maximum)
    selected: list[int] = []
    for wanted_id in wanted:
        for index, option in enumerate(options):
            if index not in selected and _card_id(option, select, current, me) == wanted_id:
                selected.append(index)
                break
        if len(selected) >= maximum:
            break
    return sorted(selected) if len(selected) >= minimum else None


def _slot_target(option: dict, mine: dict) -> dict | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    cards = (
        mine.get("active") or []
        if area == ACTIVE
        else mine.get("bench") or []
        if area == BENCH
        else []
    )
    if isinstance(index, int) and 0 <= index < len(cards):
        target = cards[index]
        return target if isinstance(target, dict) else None
    return None


def _koko_cards(mine: dict) -> list[dict]:
    return [card for card in _board(mine) if card.get("id") == TAPU_KOKO_EX]


def _has_lightning(card: dict) -> bool:
    return any(
        energy == BASIC_LIGHTNING
        or (isinstance(energy, dict) and energy.get("type") == BASIC_LIGHTNING)
        for energy in (card.get("energies") or [])
    )


def _has_grass(card: dict) -> bool:
    return any(
        energy == BASIC_GRASS
        or (isinstance(energy, dict) and energy.get("type") == BASIC_GRASS)
        for energy in (card.get("energies") or [])
    )


def _attack_energy_ready(card: dict) -> bool:
    return _has_lightning(card) and _has_grass(card)


def _koko_ready(card: dict) -> bool:
    return _attack_energy_ready(card) and _has_belt(card)


def _has_belt(card: dict) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("id") == MAXIMUM_BELT
        for tool in (card.get("tools") or [])
    )


def _play_option(observation: dict, wanted: list[int]) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    for wanted_id in wanted:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == PLAY and _card_id(option, select, current, me) == wanted_id:
                return [index]
    return None


def _attach_option(
    observation: dict,
    mine: dict,
    target: dict,
    wanted: tuple[int, ...],
) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    for wanted_id in wanted:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") != ATTACH:
                continue
            if _card_id(option, select, current, me) != wanted_id:
                continue
            if _slot_target(option, mine) is target:
                return [index]
    return None


def _setup_override(observation: dict) -> list[int] | None:
    select = observation.get("select") or {}
    if select.get("type") != 1 or select.get("context") not in (1, 2):
        return None
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    if select.get("context") == 1:
        return _choose_ids(
            select,
            current,
            me,
            [TAPU_KOKO_EX, MIMIKYU, TAROUNTULA, ARTICUNO, MEWTWO_EX],
        )
    return _choose_ids(
        select,
        current,
        me,
        [TAPU_KOKO_EX, TAROUNTULA, TAROUNTULA, MIMIKYU, MEWTWO_EX, ARTICUNO],
    )


def _ultra_discards(select: dict, current: dict, me: int) -> list[int] | None:
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    maximum = min(int(select.get("maxCount") or 0), len(options))
    ids = [_card_id(option, select, current, me) for option in options]
    extra_grass = max(0, ids.count(BASIC_GRASS) - 1)

    def rank(index: int):
        card_id = ids[index]
        expendable_grass = card_id == BASIC_GRASS and extra_grass > 0
        return (
            0 if expendable_grass else 1,
            0 if card_id in _DISCARDABLE else 1,
            0 if card_id == BASIC_PSYCHIC else 1,
            1 if card_id in (TAPU_KOKO_EX, BASIC_LIGHTNING, MAXIMUM_BELT, SWITCH, CRISPIN) else 0,
            1 if card_id in (SPIDOPS, TAROUNTULA, PETREL, NIGHT_STRETCHER) else 0,
            index,
        )

    ranked = sorted(range(len(options)), key=rank)
    return sorted(ranked[:maximum]) if len(ranked[:maximum]) >= minimum else None


def _prompt_override(observation: dict) -> list[int] | None:
    global _PENDING_KOKO_SWITCH
    select = observation.get("select") or {}
    current, me, mine, theirs = _view(observation)
    if me is None:
        return None
    effect = select.get("effect") or {}
    effect_id = effect.get("id")
    koko = _koko_cards(mine)
    target = _zone(theirs, "active")
    target_id = target[0].get("id") if target else 0

    if effect_id == CRISPIN and koko:
        primary = koko[0]
        options = select.get("option") or []
        deck_energy = [
            index
            for index, option in enumerate(options)
            if option.get("area") == 1
            and _card_id(option, select, current, me)
            in (BASIC_LIGHTNING, BASIC_GRASS)
        ]
        if deck_energy and int(select.get("maxCount") or 0) >= 2:
            return _choose_ids(
                select,
                current,
                me,
                [BASIC_LIGHTNING, BASIC_GRASS],
            )
        if select.get("context") == 7:
            # Leave Grass in hand while Lightning is still missing so the
            # Supporter attachment supplies the typed requirement.
            wanted = (
                [BASIC_GRASS, BASIC_LIGHTNING]
                if not _has_lightning(primary)
                else [BASIC_LIGHTNING, BASIC_GRASS]
            )
            return _choose_ids(select, current, me, wanted)
        if select.get("context") == CRISPIN_ENERGY_CHOICE:
            wanted = (
                [BASIC_LIGHTNING, BASIC_GRASS]
                if not _has_lightning(primary)
                else [BASIC_GRASS, BASIC_LIGHTNING]
            )
            return _choose_ids(select, current, me, wanted)
        if select.get("context") == CRISPIN_TARGET:
            for index, option in enumerate(options):
                if _slot_target(option, mine) is primary:
                    return [index]

    if effect_id == POKE_PAD and _STARMIE_SEEN:
        wanted = (
            [TAPU_KOKO_EX]
            if not koko
            else [TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO, SPIDOPS]
            if len(_board(mine)) < 5
            else [SPIDOPS, TAROUNTULA, MEWTWO_EX]
        )
        return _choose_ids(select, current, me, wanted)

    if effect_id == ULTRA_BALL:
        if select.get("context") == DISCARD:
            return _ultra_discards(select, current, me)
        wanted = (
            [TAPU_KOKO_EX]
            if not koko
            else [TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO, SPIDOPS]
            if len(_board(mine)) < 5
            else [SPIDOPS, TAROUNTULA, MEWTWO_EX]
        )
        return _choose_ids(select, current, me, wanted)

    if effect_id == NIGHT_STRETCHER and _STARMIE_SEEN:
        return _choose_ids(
            select,
            current,
            me,
            [TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO, TAPU_KOKO_EX,
             BASIC_LIGHTNING, BASIC_GRASS],
        )

    if effect_id == TRANSCEIVER and _STARMIE_SEEN:
        board = _board(mine)
        hand_ids = {card.get("id") for card in _zone(mine, "hand")}
        crispin_needed = bool(
            koko and not _attack_energy_ready(koko[0]) and CRISPIN not in hand_ids
        )
        belt_needed = bool(
            koko and not _has_belt(koko[0]) and PETREL not in hand_ids
        )
        active = _zone(mine, "active")
        switch_needed = bool(
            active
            and koko
            and _koko_ready(koko[0])
            and active[0] is not koko[0]
            and GIOVANNI not in hand_ids
        )
        priorities = []
        if crispin_needed:
            priorities.append(CRISPIN)
        if belt_needed:
            priorities.append(PETREL)
        if switch_needed:
            priorities.append(GIOVANNI)
        if len(board) < 5 and PROTON not in hand_ids:
            priorities.append(PROTON)
        for card_id in (CRISPIN, PETREL, PROTON, GIOVANNI, ARIANA):
            if card_id not in hand_ids and card_id not in priorities:
                priorities.append(card_id)
        return _choose_ids(select, current, me, priorities)

    if effect_id == PROTON:
        known = _board(mine) + _zone(mine, "hand")
        priorities = []
        priorities.extend([TAROUNTULA] * max(0, 2 - sum(c.get("id") == TAROUNTULA for c in known)))
        for card_id in (MEWTWO_EX, MIMIKYU, ARTICUNO):
            if not any(card.get("id") == card_id for card in known):
                priorities.append(card_id)
        priorities.extend((TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO))
        return _choose_ids(select, current, me, priorities)

    if effect_id == PETREL and _STARMIE_SEEN:
        priorities = []
        if koko and not _has_belt(koko[0]):
            priorities.append(MAXIMUM_BELT)
        if koko and _zone(mine, "active") and _zone(mine, "active")[0] is not koko[0]:
            priorities.append(SWITCH)
        if not koko:
            priorities.extend((POKE_PAD, ULTRA_BALL))
        priorities.extend((NIGHT_STRETCHER, TRANSCEIVER, POKE_PAD, ULTRA_BALL))
        return _choose_ids(select, current, me, priorities)

    if select.get("context") in (TO_ACTIVE, TARGET_ACTIVE):
        own: list[tuple[int, dict]] = []
        enemy: list[tuple[int, int]] = []
        for index, option in enumerate(select.get("option") or []):
            card = _slot_target(option, mine)
            if card is not None:
                own.append((index, card))
            elif option.get("playerIndex") == 1 - me:
                enemy.append((index, _card_id(option, select, current, me)))
        if own and (_PENDING_KOKO_SWITCH or not _zone(mine, "active")):
            for index, card in own:
                if card.get("id") == TAPU_KOKO_EX:
                    _PENDING_KOKO_SWITCH = False
                    return [index]
        for wanted in (STARYU, MEGA_STARMIE_EX):
            for index, card_id in enemy:
                if card_id == wanted:
                    return [index]
    return None


def _safe_ultra(mine: dict) -> bool:
    hand = _zone(mine, "hand")
    # If Ultra Ball is presented as a legal MAIN action, the engine has already
    # established that two other cards can be discarded.  Early Koko access is
    # the point of this mechanism probe, so do not require both costs to be
    # low-value trainers; the prompt ranker still protects Lightning, Belt,
    # Switch, and Koko as long as any alternative exists.
    return len(hand) >= 3


def _route_override(observation: dict) -> list[int] | None:
    global _PENDING_KOKO_SWITCH
    setup = _setup_override(observation)
    if setup is not None:
        return setup
    _observe_starmie(observation)
    prompt = _prompt_override(observation)
    if prompt is not None:
        return prompt
    if not _STARMIE_SEEN:
        return None

    select = observation.get("select") or {}
    if select.get("type") != MAIN:
        return None
    current, me, mine, theirs = _view(observation)
    if me is None:
        return None
    koko = _koko_cards(mine)
    board = _board(mine)
    active = _zone(mine, "active")
    target = _zone(theirs, "active")
    target_id = target[0].get("id") if target else 0

    if not koko:
        action = _play_option(observation, [TAPU_KOKO_EX])
        if action is not None:
            return action
        action = _play_option(observation, [POKE_PAD])
        if action is not None:
            return action
        if _safe_ultra(mine):
            action = _play_option(observation, [ULTRA_BALL])
            if action is not None:
                return action
        return None

    primary = koko[0]
    rocket_count = sum(card.get("id") in _ROCKET_BASICS + (SPIDOPS,) for card in board)
    hand_ids = {card.get("id") for card in _zone(mine, "hand")}

    if not _attack_energy_ready(primary) and not bool(current.get("supporterPlayed")):
        action = _play_option(observation, [CRISPIN])
        if action is not None:
            return action
        if CRISPIN not in hand_ids:
            action = _play_option(observation, [TRANSCEIVER])
            if action is not None:
                return action

    # Topaz Bolt discards every attached Energy when Crystal reduces its cost
    # to two. Re-establish the missing typed attachment before any optional
    # board search can spend or discard the Energy Crispin put into hand.
    if not bool(current.get("energyAttached")):
        if not _has_lightning(primary):
            action = _attach_option(observation, mine, primary, (BASIC_LIGHTNING,))
        elif not _has_grass(primary):
            action = _attach_option(observation, mine, primary, (BASIC_GRASS,))
        else:
            action = None
        if action is not None:
            return action

    if len(board) < 5 or rocket_count < 3:
        if not bool(current.get("supporterPlayed")):
            action = _play_option(observation, [PROTON, TRANSCEIVER])
            if action is not None:
                return action
        action = _play_option(
            observation,
            [TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO],
        )
        if action is not None:
            return action
        if len(board) < 5:
            action = _play_option(observation, [NIGHT_STRETCHER, POKE_PAD])
            if action is not None:
                return action
            if _safe_ultra(mine):
                action = _play_option(observation, [ULTRA_BALL])
                if action is not None:
                    return action

    if not _has_belt(primary):
        action = _attach_option(observation, mine, primary, (MAXIMUM_BELT,))
        if action is not None:
            return action
        if PETREL not in hand_ids:
            action = _play_option(observation, [TRANSCEIVER])
            if action is not None:
                return action
        if target_id == MEGA_STARMIE_EX and not bool(current.get("supporterPlayed")):
            action = _play_option(observation, [PETREL, TRANSCEIVER])
            if action is not None:
                return action

    if active and active[0] is not primary and _koko_ready(primary):
        action = _play_option(observation, [SWITCH])
        if action is not None:
            _PENDING_KOKO_SWITCH = True
            return action
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == RETREAT:
                _PENDING_KOKO_SWITCH = True
                return [index]
        if not bool(current.get("supporterPlayed")):
            action = _play_option(observation, [GIOVANNI, TRANSCEIVER])
            if action is not None:
                _PENDING_KOKO_SWITCH = True
                return action

    if active and active[0].get("id") == TAPU_KOKO_EX and _koko_ready(active[0]):
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == ATTACK and option.get("attackId") == LINKED_LIGHTNING:
                return [index]
    return None


def _guard_neural_fallback(observation: dict, action: list[int]) -> list[int]:
    if not action:
        return action
    select = observation.get("select") or {}
    current, me, mine, _ = _view(observation)
    options = select.get("option") or []
    index = action[0]
    if me is None or not isinstance(index, int) or not 0 <= index < len(options):
        return action
    option = options[index]
    unsafe_ultra = (
        _STARMIE_SEEN
        and select.get("type") == MAIN
        and option.get("type") == PLAY
        and _card_id(option, select, current, me) == ULTRA_BALL
        and not _safe_ultra(mine)
    )
    card_id = _card_id(option, select, current, me)
    target = _slot_target(option, mine)
    active = _zone(mine, "active")
    wastes_koko_tool = bool(
        _STARMIE_SEEN
        and option.get("type") == ATTACH
        and card_id in (MAXIMUM_BELT, 1175)
        and (
            target is None
            or target.get("id") != TAPU_KOKO_EX
            or card_id == 1175
        )
    )
    switches_koko_out = bool(
        _STARMIE_SEEN
        and option.get("type") == PLAY
        and card_id == SWITCH
        and active
        and active[0].get("id") == TAPU_KOKO_EX
    )
    retreats_pikachu = bool(
        _STARMIE_SEEN
        and option.get("type") == RETREAT
        and active
        and active[0].get("id") == TAPU_KOKO_EX
        and _has_belt(active[0])
    )
    if unsafe_ultra or wastes_koko_tool or switches_koko_out or retreats_pikachu:
        for fallback, candidate in enumerate(options):
            if candidate.get("type") == END_TURN:
                return [fallback]
    return action


def _fresh_agent() -> NNAgent:
    return NNAgent(
        _NET,
        greedy=True,
        deck_ids=_DECK,
        forbid_end_when_attack=True,
        opponent_deck_priors=_OPPONENT_DECK_PRIORS,
    )


def agent(observation: dict):
    global _AGENT, _STARMIE_SEEN, _PENDING_KOKO_SWITCH
    if observation.get("select") is None:
        _AGENT = _fresh_agent()
        _STARMIE_SEEN = False
        _PENDING_KOKO_SWITCH = False
        return _DECK
    route = _route_override(observation)
    if route is not None:
        return route
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _guard_neural_fallback(observation, _AGENT(observation))
