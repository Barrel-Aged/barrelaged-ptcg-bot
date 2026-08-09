"""Team Rocket's Zapdos / Maximum Belt official-engine mechanism route.

The frozen neural policy remains the fallback. This deterministic development
wrapper verifies the native-Rocket search, Crispin plus special-Energy charge,
promotion, Belt breakpoint, and Wicked Thunder sequence. It is not the final
strategic policy or a deployment-strength claim.
"""
from __future__ import annotations

import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/kaggle_simulations/agent"
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import torch  # noqa: E402

torch.set_num_threads(1)

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
_PENDING_PROMOTION = False

BASIC_GRASS = 1
BASIC_LIGHTNING = 4
ROCKET_ENERGY = 15
TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
ZAPDOS = 425
MEWTWO_EX = 431
MIMIKYU = 434
STARYU = 1030
MEGA_STARMIE_EX = 1031

NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
SWITCH = 1123
TRANSCEIVER = 1134
POKE_PAD = 1152
MAXIMUM_BELT = 1158
BRAVE_BANGLE = 1175
CRISPIN = 1198
ARIANA = 1216
GIOVANNI = 1218
PETREL = 1219
PROTON = 1220

WICKED_THUNDER = 600

MAIN = 0
PLAY = 7
ATTACH = 8
RETREAT = 12
ATTACK = 13
ACTIVE = 4
BENCH = 5
DISCARD = 8
SETUP_ACTIVE = 1
SETUP_BENCH = 2
TO_HAND = 7
TARGET = 21
ENERGY_CHOICE = 22


def _view(observation: dict):
    current = observation.get("current") or {}
    me = current.get("yourIndex")
    players = current.get("players") or []
    if me not in (0, 1) or len(players) != 2:
        return current, None, {}, {}
    return current, int(me), players[me] or {}, players[1 - me] or {}


def _zone(player: dict, name: str) -> list[dict]:
    return [card for card in (player.get(name) or []) if isinstance(card, dict)]


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


def _choose_ids(select: dict, current: dict, me: int, wanted: list[int]) -> list[int] | None:
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
    return sorted(picked) if len(picked) >= minimum else None


def _slot_target(option: dict, mine: dict) -> dict | None:
    area = option.get("inPlayArea", option.get("area"))
    index = option.get("inPlayIndex", option.get("index"))
    zone = "active" if area == ACTIVE else "bench" if area == BENCH else None
    cards = mine.get(zone) or [] if zone else []
    if isinstance(index, int) and 0 <= index < len(cards) and isinstance(cards[index], dict):
        return cards[index]
    return None


def _zapdos_cards(mine: dict) -> list[dict]:
    active = _zone(mine, "active")
    return [card for card in active + _zone(mine, "bench") if card.get("id") == ZAPDOS]


def _has_energy(card: dict, card_id: int) -> bool:
    return any(
        isinstance(energy, dict) and energy.get("id") == card_id
        for energy in (card.get("energyCards") or [])
    )


def _has_tool(card: dict, card_id: int) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("id") == card_id
        for tool in (card.get("tools") or [])
    )


def _zapdos_ready(card: dict) -> bool:
    return card.get("id") == ZAPDOS and _has_energy(card, BASIC_LIGHTNING) and _has_energy(card, ROCKET_ENERGY)


def _play_option(observation: dict, wanted: list[int]) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None or select.get("type") != MAIN:
        return None
    for wanted_id in wanted:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == PLAY and _card_id(option, select, current, me) == wanted_id:
                return [index]
    return None


def _attach_option(observation: dict, mine: dict, target: dict, wanted: tuple[int, ...]) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None or select.get("type") != MAIN:
        return None
    hand = _zone(mine, "hand")
    for wanted_id in wanted:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") != ATTACH or _slot_target(option, mine) is not target:
                continue
            hand_index = option.get("index")
            if isinstance(hand_index, int) and 0 <= hand_index < len(hand) and hand[hand_index].get("id") == wanted_id:
                return [index]
    return None


def _setup_override(observation: dict) -> list[int] | None:
    select = observation.get("select") or {}
    if select.get("context") not in (SETUP_ACTIVE, SETUP_BENCH):
        return None
    current, me, mine, _ = _view(observation)
    if me is None:
        return None
    if select.get("context") == SETUP_ACTIVE:
        wanted = [ZAPDOS, TAROUNTULA, ARTICUNO, MIMIKYU, MEWTWO_EX]
    else:
        have_zapdos = any(card.get("id") == ZAPDOS for card in _board(mine))
        wanted = ([ZAPDOS] if not have_zapdos else []) + [TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO]
    return _choose_ids(select, current, me, wanted)


def _ultra_discards(select: dict, current: dict, me: int) -> list[int] | None:
    options = select.get("option") or []
    maximum = min(int(select.get("maxCount") or 0), len(options))
    minimum = int(select.get("minCount") or 0)
    protected = {ZAPDOS, BASIC_LIGHTNING, ROCKET_ENERGY, MAXIMUM_BELT, CRISPIN, SWITCH}
    ranked = sorted(
        range(len(options)),
        key=lambda index: (
            _card_id(options[index], select, current, me) in protected,
            _card_id(options[index], select, current, me) in (TAROUNTULA, SPIDOPS, MEWTWO_EX),
            index,
        ),
    )
    return sorted(ranked[:maximum]) if maximum >= minimum else None


def _prompt_override(observation: dict) -> list[int] | None:
    global _PENDING_PROMOTION
    select = observation.get("select") or {}
    current, me, mine, theirs = _view(observation)
    if me is None:
        return None
    effect_id = (select.get("effect") or {}).get("id")
    zapdos = _zapdos_cards(mine)
    hand_ids = {card.get("id") for card in _zone(mine, "hand")}
    opposing_active = _zone(theirs, "active")
    opposing_active_id = opposing_active[0].get("id") if opposing_active else 0

    if effect_id == CRISPIN and zapdos:
        primary = zapdos[0]
        options = select.get("option") or []
        if any(option.get("area") == 1 for option in options) and int(select.get("maxCount") or 0) >= 2:
            choice = _choose_ids(select, current, me, [BASIC_LIGHTNING, BASIC_GRASS])
            if choice is not None:
                return choice
        if select.get("context") == TO_HAND:
            return _choose_ids(select, current, me, [BASIC_GRASS, BASIC_LIGHTNING])
        if select.get("context") == ENERGY_CHOICE:
            return _choose_ids(select, current, me, [BASIC_LIGHTNING, BASIC_GRASS])
        if select.get("context") == TARGET:
            for index, option in enumerate(options):
                if _slot_target(option, mine) is primary:
                    return [index]

    if effect_id == POKE_PAD:
        wanted = ([ZAPDOS] if not zapdos else []) + [SPIDOPS, TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO]
        return _choose_ids(select, current, me, wanted)

    if effect_id == ULTRA_BALL:
        if select.get("context") == DISCARD:
            return _ultra_discards(select, current, me)
        wanted = ([ZAPDOS] if not zapdos else []) + [SPIDOPS, MEWTWO_EX, TAROUNTULA]
        return _choose_ids(select, current, me, wanted)

    if effect_id == PROTON:
        wanted = ([ZAPDOS] if not zapdos else []) + [TAROUNTULA, TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO, ZAPDOS]
        return _choose_ids(select, current, me, wanted)

    if effect_id == PETREL:
        needs_belt = bool(zapdos and not _has_tool(zapdos[0], MAXIMUM_BELT))
        wanted = ([MAXIMUM_BELT] if needs_belt else []) + [SWITCH, POKE_PAD, NIGHT_STRETCHER, ULTRA_BALL]
        return _choose_ids(select, current, me, wanted)

    if effect_id == TRANSCEIVER:
        wanted = []
        if not zapdos:
            wanted.append(PROTON)
        if zapdos and opposing_active_id == MEGA_STARMIE_EX and MAXIMUM_BELT not in hand_ids:
            wanted.append(PETREL)
        wanted.extend((PROTON, PETREL, GIOVANNI, ARIANA))
        return _choose_ids(select, current, me, wanted)

    if effect_id == NIGHT_STRETCHER:
        return _choose_ids(select, current, me, [ZAPDOS, ROCKET_ENERGY, BASIC_LIGHTNING, SPIDOPS, TAROUNTULA, BASIC_GRASS])

    if select.get("context") in (3, 4):
        ready = [card for card in zapdos if _zapdos_ready(card)]
        if ready and (_PENDING_PROMOTION or not _zone(mine, "active")):
            for index, option in enumerate(select.get("option") or []):
                if _slot_target(option, mine) is ready[0]:
                    _PENDING_PROMOTION = False
                    return [index]
    return None


def _safe_ultra(mine: dict) -> bool:
    return len(_zone(mine, "hand")) >= 3


def _route_override(observation: dict) -> list[int] | None:
    global _PENDING_PROMOTION
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
    zapdos = _zapdos_cards(mine)
    active = _zone(mine, "active")
    target = _zone(theirs, "active")
    target_id = target[0].get("id") if target else 0
    hand_ids = {card.get("id") for card in _zone(mine, "hand")}

    if active and active[0].get("id") == ZAPDOS:
        legal_attack = next(
            (
                [index]
                for index, option in enumerate(select.get("option") or [])
                if option.get("type") == ATTACK and option.get("attackId") == WICKED_THUNDER
            ),
            None,
        )
        if legal_attack is not None:
            if target_id == MEGA_STARMIE_EX and not _has_tool(active[0], MAXIMUM_BELT):
                action = _attach_option(observation, mine, active[0], (MAXIMUM_BELT,))
                if action is not None:
                    return action
                if not bool(current.get("supporterPlayed")):
                    action = _play_option(observation, [PETREL])
                    if action is not None:
                        return action
            return legal_attack

    if not zapdos:
        action = _play_option(observation, [ZAPDOS])
        if action is not None:
            return action
        if not bool(current.get("supporterPlayed")):
            action = _play_option(observation, [PROTON, TRANSCEIVER])
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

    primary = zapdos[0]
    if not _has_energy(primary, BASIC_LIGHTNING) and not bool(current.get("supporterPlayed")):
        action = _play_option(observation, [CRISPIN])
        if action is not None:
            return action

    if not bool(current.get("energyAttached")):
        wanted = (ROCKET_ENERGY,) if not _has_energy(primary, ROCKET_ENERGY) else (BASIC_LIGHTNING,)
        action = _attach_option(observation, mine, primary, wanted)
        if action is not None:
            return action

    if target_id == MEGA_STARMIE_EX and not _has_tool(primary, MAXIMUM_BELT):
        action = _attach_option(observation, mine, primary, (MAXIMUM_BELT,))
        if action is not None:
            return action
        if not bool(current.get("supporterPlayed")) and PETREL in hand_ids:
            action = _play_option(observation, [PETREL])
            if action is not None:
                return action

    if _zapdos_ready(primary) and (not active or active[0] is not primary):
        action = _play_option(observation, [SWITCH])
        if action is not None:
            _PENDING_PROMOTION = True
            return action
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == RETREAT:
                _PENDING_PROMOTION = True
                return [index]
        if not bool(current.get("supporterPlayed")):
            action = _play_option(observation, [GIOVANNI])
            if action is not None:
                _PENDING_PROMOTION = True
                return action

    if not _has_energy(primary, BASIC_LIGHTNING) and not bool(current.get("supporterPlayed")):
        action = _play_option(observation, [CRISPIN])
        if action is not None:
            return action
    return None


def _fresh_agent() -> NNAgent:
    return NNAgent(
        _NET,
        greedy=True,
        deck_ids=_DECK,
        forbid_end_when_attack=True,
        opponent_deck_priors=_OPPONENT_DECK_PRIORS,
    )


def _guard_neural_fallback(observation: dict, action: list[int]) -> list[int]:
    """Do not spend a switching card while the routed attacker is Active."""
    select = observation.get("select") or {}
    if select.get("type") != MAIN or len(action) != 1:
        return action
    index = action[0]
    options = select.get("option") or []
    if not isinstance(index, int) or not 0 <= index < len(options):
        return action
    current, me, mine, _ = _view(observation)
    if me is None:
        return action
    active = _zone(mine, "active")
    chosen = options[index]
    if (
        active
        and active[0].get("id") == ZAPDOS
        and chosen.get("type") == PLAY
        and _card_id(chosen, select, current, me) == SWITCH
    ):
        for fallback_index, option in enumerate(options):
            if option.get("type") == 14:
                return [fallback_index]
    return action


def agent(observation: dict):
    global _AGENT, _STARMIE_SEEN, _PENDING_PROMOTION
    if observation.get("select") is None:
        _AGENT = _fresh_agent()
        _STARMIE_SEEN = False
        _PENDING_PROMOTION = False
        return _DECK
    routed = _route_override(observation)
    if routed is not None:
        return routed
    if _AGENT is None:
        _AGENT = _fresh_agent()
    return _guard_neural_fallback(observation, _AGENT(observation))
