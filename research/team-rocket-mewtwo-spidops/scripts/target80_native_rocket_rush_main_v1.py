"""Public-state native Rocket opening probe for the zero-Wobbuffet deck.

The exact-deck-trained neural policy remains the baseline.  This research
wrapper executes one coherent opening plan when its legal pieces are public:

* prefer Tarountula as the opening Active;
* use first-turn Proton (directly or through Transceiver) to establish two
  Tarountula lines and enough Team Rocket Pokemon for Rocket Rush;
* reserve/search Spidops and attach Grass to the line or retreat pivot;
* on the second own turn, evolve the correct line, attach, retreat/switch,
  use Charging Up, and attack with Rocket Rush when legal; and
* begin charging a second Spidops before spending further energy elsewhere.

This is a mechanism probe and trajectory teacher, not the final strategic
policy.  It reads only the normal redacted observation and never reads an
opponent hidden zone.
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
_PENDING_PROMOTION = False


BASIC_GRASS = 1
BASIC_PSYCHIC = 5
ROCKET_ENERGY = 15
TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
MEWTWO_EX = 431
MIMIKYU = 434

BUDDY_POFFIN = 1086
BUG_CATCHING_SET = 1094
NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
TRANSCEIVER = 1134
POKE_PAD = 1152
BRAVE_BANGLE = 1175
ARIANA = 1216
GIOVANNI = 1218
PETREL = 1219
PROTON = 1220
LILLIE = 1227
FACTORY = 1257

ROCKET_RUSH = 560

MAIN = 0
SETUP_ACTIVE = 1
SETUP_BENCH = 2
TO_ACTIVE = 3
TARGET_ACTIVE = 4
TO_HAND = 7
DISCARD = 8
CHARGING_TARGET = 22
RETREAT_DISCARD = 30

ACTIVE = 4
BENCH = 5

PLAY = 7
ATTACH = 8
EVOLVE = 9
ABILITY = 10
RETREAT = 12
ATTACK = 13
END_TURN = 14

_ROCKET_BASICS = (TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO)
_ENERGY_IDS = (BASIC_GRASS, BASIC_PSYCHIC, ROCKET_ENERGY)
_DISCARDABLE_TRAINERS = (
    ARIANA,
    LILLIE,
    FACTORY,
    GIOVANNI,
    PETREL,
    PROTON,
    TRANSCEIVER,
    BUG_CATCHING_SET,
    POKE_PAD,
    BUDDY_POFFIN,
)


def _view(observation: dict):
    current = observation.get("current") or {}
    me = current.get("yourIndex")
    players = current.get("players") or []
    if me not in (0, 1) or len(players) != 2:
        return current, None, {}, {}
    return current, int(me), players[me] or {}, players[1 - me] or {}


def _board_slots(player: dict) -> list[tuple[int, int, dict]]:
    answer: list[tuple[int, int, dict]] = []
    for area, zone in ((ACTIVE, "active"), (BENCH, "bench")):
        for index, card in enumerate(player.get(zone) or []):
            if isinstance(card, dict):
                answer.append((area, index, card))
    return answer


def _cards(player: dict, zone: str) -> list[dict]:
    return [card for card in player.get(zone) or [] if isinstance(card, dict)]


def _option_card_id(option: dict, select: dict, current: dict, me: int) -> int:
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
    chosen: list[int] = []
    for wanted_id in wanted:
        for index, option in enumerate(options):
            if (
                index not in chosen
                and _option_card_id(option, select, current, me) == wanted_id
            ):
                chosen.append(index)
                break
        if len(chosen) >= maximum:
            break
    if len(chosen) < minimum:
        return None
    return sorted(chosen)


def _energy_units(card: dict) -> int:
    return len(card.get("energies") or [])


def _has_grass(card: dict) -> bool:
    return any(
        energy == BASIC_GRASS
        or (isinstance(energy, dict) and energy.get("type") == BASIC_GRASS)
        for energy in (card.get("energies") or [])
    )


def _ready_spidops(card: dict) -> bool:
    return card.get("id") == SPIDOPS and _has_grass(card) and _energy_units(card) >= 2


def _opening_turn(current: dict, me: int) -> int | None:
    first = current.get("firstPlayer")
    if first not in (0, 1):
        return None
    return 1 if int(first) == me else 2


def _target_card(option: dict, mine: dict) -> dict | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    zone = "active" if area == ACTIVE else "bench" if area == BENCH else None
    cards = mine.get(zone) or [] if zone else []
    if isinstance(index, int) and 0 <= index < len(cards):
        card = cards[index]
        return card if isinstance(card, dict) else None
    return None


def _target_option(
    select: dict,
    mine: dict,
    target: dict,
    option_type: int | None = None,
) -> int | None:
    for index, option in enumerate(select.get("option") or []):
        if option_type is not None and option.get("type") != option_type:
            continue
        if _target_card(option, mine) is target:
            return index
    return None


def _spidops_slots(mine: dict) -> list[tuple[int, int, dict]]:
    slots = [slot for slot in _board_slots(mine) if slot[2].get("id") == SPIDOPS]
    return sorted(
        slots,
        key=lambda slot: (
            0 if slot[0] == ACTIVE else 1,
            0 if _has_grass(slot[2]) else 1,
            -_energy_units(slot[2]),
            slot[1],
        ),
    )


def _setup_override(observation: dict) -> list[int] | None:
    select = observation.get("select") or {}
    context = select.get("context")
    if select.get("type") != 1 or context not in (SETUP_ACTIVE, SETUP_BENCH):
        return None
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    if context == SETUP_ACTIVE:
        return _choose_ids(
            select,
            current,
            me,
            [TAROUNTULA, MIMIKYU, ARTICUNO, MEWTWO_EX],
        )
    return _choose_ids(
        select,
        current,
        me,
        [TAROUNTULA, TAROUNTULA, MIMIKYU, MEWTWO_EX, ARTICUNO],
    )


def _need_spidops(mine: dict) -> bool:
    board = [card for _, _, card in _board_slots(mine)]
    tar_count = sum(card.get("id") == TAROUNTULA for card in board)
    spid_count = sum(card.get("id") == SPIDOPS for card in board)
    hand_spid = sum(card.get("id") == SPIDOPS for card in _cards(mine, "hand"))
    return tar_count > hand_spid and spid_count + hand_spid < 2


def _need_backup_spidops(mine: dict) -> bool:
    board = [card for _, _, card in _board_slots(mine)]
    tar_count = sum(card.get("id") == TAROUNTULA for card in board)
    spid_count = sum(card.get("id") == SPIDOPS for card in board)
    hand_spid = sum(card.get("id") == SPIDOPS for card in _cards(mine, "hand"))
    return tar_count > 0 and spid_count + hand_spid < 2


def _proton_wanted(mine: dict) -> list[int]:
    board = [card for _, _, card in _board_slots(mine)]
    hand = _cards(mine, "hand")
    known = board + hand
    tar_known = sum(card.get("id") == TAROUNTULA for card in known)
    wanted = [TAROUNTULA] * max(0, 2 - tar_known)
    if not any(card.get("id") == MEWTWO_EX for card in known):
        wanted.append(MEWTWO_EX)
    if not any(card.get("id") == MIMIKYU for card in known):
        wanted.append(MIMIKYU)
    if not any(card.get("id") == ARTICUNO for card in known):
        wanted.append(ARTICUNO)
    wanted.extend((TAROUNTULA, TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO))
    return wanted


def _ultra_discard_indices(
    select: dict,
    current: dict,
    me: int,
    mine: dict,
) -> list[int] | None:
    options = select.get("option") or []
    minimum = int(select.get("minCount") or 0)
    maximum = min(int(select.get("maxCount") or 0), len(options))
    if maximum < minimum:
        return None
    ids = [_option_card_id(option, select, current, me) for option in options]
    board = [card for _, _, card in _board_slots(mine)]
    grass_in_hand = sum(card_id == BASIC_GRASS for card_id in ids)
    line_has_grass = any(
        card.get("id") in (TAROUNTULA, SPIDOPS) and _has_grass(card)
        for card in board
    )

    def rank(index: int):
        card_id = ids[index]
        safe_grass = card_id == BASIC_GRASS and (grass_in_hand >= 2 or line_has_grass)
        return (
            0 if safe_grass else 1,
            0 if card_id in _DISCARDABLE_TRAINERS else 1,
            0 if card_id == BASIC_PSYCHIC else 1,
            1 if card_id in (SPIDOPS, TAROUNTULA, MEWTWO_EX, NIGHT_STRETCHER) else 0,
            1 if card_id in _ENERGY_IDS and not safe_grass else 0,
            index,
        )

    ranked = sorted(range(len(options)), key=rank)
    return sorted(ranked[:maximum]) if len(ranked[:maximum]) >= minimum else None


def _safe_ultra_available(observation: dict, mine: dict) -> bool:
    hand = _cards(mine, "hand")
    if len(hand) < 3 or any(card.get("id") == SPIDOPS for card in hand):
        return False
    expendable = sum(
        card.get("id") in _DISCARDABLE_TRAINERS
        or card.get("id") == BASIC_PSYCHIC
        for card in hand
    )
    extra_grass = max(0, sum(card.get("id") == BASIC_GRASS for card in hand) - 1)
    return expendable + extra_grass >= 2


def _prompt_override(observation: dict) -> list[int] | None:
    global _PENDING_PROMOTION
    select = observation.get("select") or {}
    effect = select.get("effect") or {}
    effect_id = effect.get("id")
    current, me, mine, theirs = _view(observation)
    if me is None:
        return None
    opening = _opening_turn(current, me)
    turn = int(current.get("turn") or 0)
    evolution = opening + 2 if opening is not None else None
    route_window = evolution is not None and turn <= evolution + 4
    early = evolution is not None and turn <= evolution

    if effect_id == TRANSCEIVER and early:
        active = _cards(mine, "active")
        stuck = bool(active and active[0].get("id") == MEWTWO_EX)
        bank_switch = stuck and (
            turn == evolution or bool(current.get("supporterPlayed"))
        )
        priorities = [GIOVANNI, PROTON] if bank_switch else [PROTON, GIOVANNI]
        priorities.extend((ARIANA, PETREL))
        return _choose_ids(select, current, me, priorities)

    if effect_id == PROTON:
        return _choose_ids(select, current, me, _proton_wanted(mine))

    if effect_id == POKE_PAD and route_window:
        wanted = [SPIDOPS, TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO]
        return _choose_ids(select, current, me, wanted)

    if effect_id == BUG_CATCHING_SET and route_window:
        return _choose_ids(
            select,
            current,
            me,
            [SPIDOPS, BASIC_GRASS, TAROUNTULA, SPIDOPS, BASIC_GRASS],
        )

    if effect_id == ULTRA_BALL:
        if select.get("context") == DISCARD:
            return _ultra_discard_indices(select, current, me, mine)
        return _choose_ids(
            select,
            current,
            me,
            [SPIDOPS, TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO],
        )

    if effect_id == NIGHT_STRETCHER and route_window:
        return _choose_ids(
            select,
            current,
            me,
            [BASIC_GRASS, SPIDOPS, TAROUNTULA, MEWTWO_EX],
        )

    if effect_id == SPIDOPS and select.get("context") == CHARGING_TARGET:
        return _choose_ids(select, current, me, [BASIC_GRASS])

    if select.get("context") == RETREAT_DISCARD:
        choice = _choose_ids(
            select,
            current,
            me,
            [BASIC_GRASS, BASIC_PSYCHIC, ROCKET_ENERGY],
        )
        if choice is not None:
            return choice

    if select.get("context") in (TO_ACTIVE, TARGET_ACTIVE):
        options = select.get("option") or []
        own_options: list[tuple[int, dict]] = []
        opponent_options: list[tuple[int, int]] = []
        for index, option in enumerate(options):
            target = _target_card(option, mine)
            if target is not None:
                own_options.append((index, target))
                continue
            card_id = _option_card_id(option, select, current, me)
            if option.get("playerIndex") == 1 - me:
                opponent_options.append((index, card_id))
        if own_options and (_PENDING_PROMOTION or not _cards(mine, "active")):
            own_options.sort(
                key=lambda item: (
                    0 if _ready_spidops(item[1]) else 1,
                    0 if item[1].get("id") == SPIDOPS else 1,
                    0 if _has_grass(item[1]) else 1,
                    -_energy_units(item[1]),
                    item[0],
                )
            )
            if own_options[0][1].get("id") == SPIDOPS:
                _PENDING_PROMOTION = False
                return [own_options[0][0]]
        for wanted in (1030, 1031):
            for index, card_id in opponent_options:
                if card_id == wanted:
                    return [index]
    return None


def _play_option(
    observation: dict,
    wanted_ids: list[int] | tuple[int, ...],
) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    for wanted in wanted_ids:
        for index, option in enumerate(select.get("option") or []):
            if (
                option.get("type") == PLAY
                and _option_card_id(option, select, current, me) == wanted
            ):
                return [index]
    return None


def _bench_option(observation: dict, wanted_ids: list[int]) -> list[int] | None:
    return _play_option(observation, wanted_ids)


def _evolution_option(observation: dict, mine: dict) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for index, option in enumerate(select.get("option") or []):
        if option.get("type") != EVOLVE:
            continue
        if _option_card_id(option, select, current, me) != SPIDOPS:
            continue
        target = _target_card(option, mine)
        if target is None or target.get("id") != TAROUNTULA:
            continue
        candidates.append(
            (
                (
                    0 if option.get("inPlayArea") == ACTIVE else 1,
                    0 if _has_grass(target) else 1,
                    -_energy_units(target),
                    index,
                ),
                index,
            )
        )
    return [min(candidates)[1]] if candidates else None


def _attachment_option(
    observation: dict,
    mine: dict,
    target: dict,
    energy_ids: tuple[int, ...] = (BASIC_GRASS, ROCKET_ENERGY, BASIC_PSYCHIC),
) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    for wanted in energy_ids:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") != ATTACH:
                continue
            if _option_card_id(option, select, current, me) != wanted:
                continue
            if _target_card(option, mine) is target:
                return [index]
    return None


def _opening_attachment(observation: dict, mine: dict) -> list[int] | None:
    active = _cards(mine, "active")
    bench = _cards(mine, "bench")
    active_card = active[0] if active else None
    tar_bench = [card for card in bench if card.get("id") == TAROUNTULA]
    if active_card and active_card.get("id") == TAROUNTULA:
        return _attachment_option(observation, mine, active_card, (BASIC_GRASS,))
    if not tar_bench:
        return None
    retreat_cost = 0
    if active_card and active_card.get("id") in CARDS:
        retreat_cost = int(getattr(CARDS[active_card["id"]], "retreatCost", 0) or 0)
    if active_card and retreat_cost == 1:
        action = _attachment_option(observation, mine, active_card, (BASIC_GRASS,))
        if action is not None:
            return action
    return _attachment_option(observation, mine, tar_bench[0], (BASIC_GRASS,))


def _evolution_attachment(observation: dict, mine: dict) -> list[int] | None:
    slots = _spidops_slots(mine)
    if not slots:
        return None
    lead = slots[0][2]
    active = _cards(mine, "active")
    active_card = active[0] if active else None
    if active_card is not lead and active_card is not None:
        retreat_cost = 0
        if active_card.get("id") in CARDS:
            retreat_cost = int(
                getattr(CARDS[active_card["id"]], "retreatCost", 0) or 0
            )
        if retreat_cost == 1 and _energy_units(active_card) == 0 and _has_grass(lead):
            action = _attachment_option(observation, mine, active_card, (BASIC_GRASS,))
            if action is not None:
                return action
    if not _has_grass(lead):
        return _attachment_option(observation, mine, lead, (BASIC_GRASS,))
    if _energy_units(lead) < 2:
        return _attachment_option(
            observation,
            mine,
            lead,
            (BASIC_GRASS, ROCKET_ENERGY, BASIC_PSYCHIC),
        )
    return None


def _charge_option(observation: dict, mine: dict, prefer_backup: bool) -> list[int] | None:
    select = observation.get("select") or {}
    slots = _spidops_slots(mine)
    if not slots:
        return None
    active = _cards(mine, "active")
    active_card = active[0] if active else None
    targets = [slot for slot in slots if not _ready_spidops(slot[2])]
    targets.sort(
        key=lambda slot: (
            0 if prefer_backup and slot[2] is not active_card else 1,
            0 if slot[2] is active_card else 1,
            0 if _has_grass(slot[2]) else 1,
            -_energy_units(slot[2]),
        )
    )
    for _, _, target in targets:
        index = _target_option(select, mine, target, option_type=ABILITY)
        if index is not None:
            return [index]
    return None


def _switch_or_attack(observation: dict, mine: dict, evolution_turn: int) -> list[int] | None:
    global _PENDING_PROMOTION
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None:
        return None
    active = _cards(mine, "active")
    active_card = active[0] if active else None
    slots = _spidops_slots(mine)
    if not slots:
        return None
    lead = slots[0][2]
    if active_card is not lead:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == RETREAT:
                _PENDING_PROMOTION = True
                return [index]
        if not bool(current.get("supporterPlayed")):
            action = _play_option(observation, [GIOVANNI])
            if action is not None:
                _PENDING_PROMOTION = True
                return action
            action = _play_option(observation, [TRANSCEIVER])
            if action is not None:
                _PENDING_PROMOTION = True
                return action
        return None
    if _ready_spidops(active_card):
        bangle = _spidops_bangle_lethal_override(observation)
        if bangle is not None:
            return bangle
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == ATTACK and option.get("attackId") == ROCKET_RUSH:
                return [index]
    return None


def _native_route_override(observation: dict) -> list[int] | None:
    setup = _setup_override(observation)
    if setup is not None:
        return setup
    prompt = _prompt_override(observation)
    if prompt is not None:
        return prompt

    select = observation.get("select") or {}
    if select.get("type") != MAIN:
        return None
    current, me, mine, _ = _view(observation)
    if me is None:
        return None
    opening = _opening_turn(current, me)
    if opening is None:
        return None
    evolution = opening + 2
    turn = int(current.get("turn") or 0)
    board = [card for _, _, card in _board_slots(mine)]
    tar_count = sum(card.get("id") == TAROUNTULA for card in board)
    rocket_count = sum(card.get("id") in _ROCKET_BASICS + (SPIDOPS,) for card in board)

    if turn == opening:
        if not bool(current.get("supporterPlayed")) and (tar_count < 2 or rocket_count < 3):
            action = _play_option(observation, [PROTON, TRANSCEIVER])
            if action is not None:
                return action
        if tar_count < 2:
            action = _bench_option(observation, [TAROUNTULA])
            if action is not None:
                return action
        if rocket_count < 4:
            action = _bench_option(
                observation,
                [MEWTWO_EX, MIMIKYU, ARTICUNO, TAROUNTULA],
            )
            if action is not None:
                return action
        if _need_spidops(mine):
            action = _play_option(observation, [POKE_PAD, BUG_CATCHING_SET])
            if action is not None:
                return action
            if _safe_ultra_available(observation, mine):
                action = _play_option(observation, [ULTRA_BALL])
                if action is not None:
                    return action
        if not bool(current.get("energyAttached")):
            action = _opening_attachment(observation, mine)
            if action is not None:
                return action
        active = _cards(mine, "active")
        active_card = active[0] if active else None
        stuck = bool(active_card and active_card.get("id") == MEWTWO_EX)
        has_giovanni = any(
            card.get("id") == GIOVANNI for card in _cards(mine, "hand")
        )
        if stuck and bool(current.get("supporterPlayed")) and not has_giovanni:
            action = _play_option(observation, [TRANSCEIVER])
            if action is not None:
                return action

    if turn == evolution:
        action = _evolution_option(observation, mine)
        if action is not None:
            return action
        if not _spidops_slots(mine) and _need_spidops(mine):
            action = _play_option(observation, [POKE_PAD, BUG_CATCHING_SET])
            if action is not None:
                return action
            if _safe_ultra_available(observation, mine):
                action = _play_option(observation, [ULTRA_BALL])
                if action is not None:
                    return action
        if _spidops_slots(mine) and _need_backup_spidops(mine):
            action = _play_option(
                observation,
                [POKE_PAD, BUG_CATCHING_SET, NIGHT_STRETCHER],
            )
            if action is not None:
                return action
            if _safe_ultra_available(observation, mine):
                action = _play_option(observation, [ULTRA_BALL])
                if action is not None:
                    return action
        if rocket_count < 3:
            action = _bench_option(
                observation,
                [TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO],
            )
            if action is not None:
                return action
        if not bool(current.get("energyAttached")):
            action = _evolution_attachment(observation, mine)
            if action is not None:
                return action
        action = _switch_or_attack(observation, mine, evolution)
        if action is not None and (
            (select.get("option") or [])[action[0]].get("type") != ATTACK
        ):
            return action
        charge = _charge_option(observation, mine, prefer_backup=False)
        if charge is not None:
            return charge
        if action is not None:
            return action

    # The immediate follow-through is part of the mechanism: evolve and charge
    # a second Spidops before the first one is returned to the discard pile.
    if evolution <= turn <= evolution + 4:
        active = _cards(mine, "active")
        active_ready = bool(active and _ready_spidops(active[0]))
        action = _evolution_option(observation, mine)
        if action is not None:
            return action
        if active_ready:
            if _need_backup_spidops(mine):
                action = _play_option(
                    observation,
                    [POKE_PAD, BUG_CATCHING_SET, NIGHT_STRETCHER],
                )
                if action is not None:
                    return action
                if _safe_ultra_available(observation, mine):
                    action = _play_option(observation, [ULTRA_BALL])
                    if action is not None:
                        return action
            charge = _charge_option(observation, mine, prefer_backup=True)
            if charge is not None:
                return charge
            if not bool(current.get("energyAttached")):
                backup = [
                    slot[2]
                    for slot in _spidops_slots(mine)
                    if slot[2] is not active[0] and not _ready_spidops(slot[2])
                ]
                if backup:
                    action = _attachment_option(
                        observation,
                        mine,
                        backup[0],
                        (BASIC_GRASS, ROCKET_ENERGY, BASIC_PSYCHIC),
                    )
                    if action is not None:
                        return action
            return _switch_or_attack(observation, mine, evolution)
    return None


def _spidops_bangle_lethal_override(observation: dict) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, mine, theirs = _view(observation)
    if select.get("type") != MAIN or me is None:
        return None
    active = _cards(mine, "active")
    target_zone = _cards(theirs, "active")
    if not active or active[0].get("id") != SPIDOPS or not target_zone:
        return None
    target = target_zone[0]
    if target.get("id") not in CARDS:
        return None
    meta = CARDS[target["id"]]
    if not (meta.ex or getattr(meta, "megaEx", False)):
        return None
    if not any(
        option.get("type") == ATTACK and option.get("attackId") == ROCKET_RUSH
        for option in select.get("option") or []
    ):
        return None
    rocket_count = sum(
        card.get("id") in CARDS
        and CARDS[card["id"]].name.startswith("Team Rocket's")
        for _, _, card in _board_slots(mine)
    )
    multiplier = 2 if meta.weakness == BASIC_GRASS else 1
    resistance = 30 if meta.resistance == BASIC_GRASS else 0
    base = max(0, 30 * rocket_count * multiplier - resistance)
    boosted = max(0, (30 * rocket_count + 30) * multiplier - resistance)
    remaining = int(target.get("hp") or 0)
    if not base < remaining <= boosted:
        return None
    for index, option in enumerate(select.get("option") or []):
        if option.get("type") != ATTACH:
            continue
        if _option_card_id(option, select, current, me) != BRAVE_BANGLE:
            continue
        if option.get("inPlayArea") == ACTIVE and option.get("inPlayIndex") == 0:
            return [index]
    return None


def _guard_unsafe_ultra(observation: dict, action: list[int]) -> list[int]:
    if not action:
        return action
    select = observation.get("select") or {}
    current, me, mine, _ = _view(observation)
    options = select.get("option") or []
    index = action[0]
    if me is None or not isinstance(index, int) or not 0 <= index < len(options):
        return action
    chosen = options[index]
    if (
        select.get("type") == MAIN
        and chosen.get("type") == PLAY
        and _option_card_id(chosen, select, current, me) == ULTRA_BALL
        and not _safe_ultra_available(observation, mine)
    ):
        for fallback, option in enumerate(options):
            if option.get("type") == END_TURN:
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
    global _AGENT, _PENDING_PROMOTION
    if observation.get("select") is None:
        _AGENT = _fresh_agent()
        _PENDING_PROMOTION = False
        return _DECK
    route = _native_route_override(observation)
    if route is not None:
        return route
    bangle = _spidops_bangle_lethal_override(observation)
    if bangle is not None:
        return bangle
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _guard_unsafe_ultra(observation, _AGENT(observation))
