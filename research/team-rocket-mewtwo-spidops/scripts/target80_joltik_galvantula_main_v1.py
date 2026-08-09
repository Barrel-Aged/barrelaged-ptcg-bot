"""Joltik / Galvantula ex official-engine mechanism route.

This deterministic wrapper is restricted to a development smoke. The frozen
neural policy remains the fallback. It exists to verify search, acceleration,
evolution, promotion, and Charged Web in the official engine; it is not the
final strategic architecture or a deployment-strength claim.
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
JOLTIK = 160
GALVANTULA_EX = 161
TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
MEWTWO_EX = 431
MIMIKYU = 434
STARYU = 1030
MEGA_STARMIE_EX = 1031

BUDDY_POFFIN = 1086
NIGHT_STRETCHER = 1097
ULTRA_BALL = 1121
SWITCH = 1123
TRANSCEIVER = 1134
POKE_PAD = 1152
SPARKLING_CRYSTAL = 1165
ARIANA = 1216
PETREL = 1219
PROTON = 1220
HILDA = 1225
LILLIE = 1227

JOLTING_CHARGE = 208
CHARGED_WEB = 209

MAIN = 0
PLAY = 7
EVOLVE = 9
ATTACH = 8
RETREAT = 12
ATTACK = 13
ACTIVE = 4
BENCH = 5
DISCARD = 8
SETUP_ACTIVE = 1
SETUP_BENCH = 2
TARGET_OR_ATTACH = 21
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


def _lines(mine: dict) -> list[dict]:
    active = _zone(mine, "active")
    bench = _zone(mine, "bench")
    cards = [card for card in active + bench if card.get("id") in (JOLTIK, GALVANTULA_EX)]
    return sorted(cards, key=lambda card: (card not in active, card.get("id") != GALVANTULA_EX))


def _has_energy_card(card: dict, card_id: int) -> bool:
    return any(
        isinstance(energy, dict) and energy.get("id") == card_id
        for energy in (card.get("energyCards") or [])
    )


def _has_tool(card: dict, card_id: int) -> bool:
    return any(
        isinstance(tool, dict) and tool.get("id") == card_id
        for tool in (card.get("tools") or [])
    )


def _galv_ready(card: dict) -> bool:
    if card.get("id") != GALVANTULA_EX or not _has_energy_card(card, BASIC_LIGHTNING):
        return False
    return len(card.get("energies") or []) >= 2 or _has_tool(card, SPARKLING_CRYSTAL)


def _play_option(observation: dict, wanted: list[int]) -> list[int] | None:
    select = observation.get("select") or {}
    current, me, _, _ = _view(observation)
    if me is None or select.get("type") != MAIN:
        return None
    for wanted_id in wanted:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") in (PLAY, EVOLVE) and _card_id(option, select, current, me) == wanted_id:
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
        going_second = current.get("firstPlayer") in (0, 1) and int(current["firstPlayer"]) != me
        wanted = [JOLTIK, TAROUNTULA, ARTICUNO, MIMIKYU, MEWTWO_EX] if going_second else [TAROUNTULA, ARTICUNO, MIMIKYU, MEWTWO_EX, JOLTIK]
    else:
        have_joltik = any(card.get("id") == JOLTIK for card in _board(mine))
        wanted = ([JOLTIK] if not have_joltik else []) + [TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO]
    return _choose_ids(select, current, me, wanted)


def _ultra_discards(select: dict, current: dict, me: int) -> list[int] | None:
    options = select.get("option") or []
    maximum = min(int(select.get("maxCount") or 0), len(options))
    minimum = int(select.get("minCount") or 0)
    protected = {JOLTIK, GALVANTULA_EX, BASIC_LIGHTNING, SPARKLING_CRYSTAL, HILDA, SWITCH}
    ranked = sorted(
        range(len(options)),
        key=lambda index: (
            _card_id(options[index], select, current, me) in protected,
            _card_id(options[index], select, current, me) in (TAROUNTULA, SPIDOPS, MEWTWO_EX),
            index,
        ),
    )
    return sorted(ranked[:maximum]) if maximum >= minimum else None


def _energy_target(select: dict, mine: dict, energy_id: int) -> list[int] | None:
    options = select.get("option") or []
    candidates: list[dict] = []
    lines = _lines(mine)
    if energy_id == BASIC_LIGHTNING:
        candidates.extend(card for card in lines if not _has_energy_card(card, BASIC_LIGHTNING))
    else:
        candidates.extend(card for card in lines if len(card.get("energies") or []) < 2)
    candidates.extend(
        card for card in _board(mine)
        if card.get("id") == SPIDOPS and not card.get("energies")
    )
    candidates.extend(
        card for card in _board(mine)
        if card not in candidates and card.get("id") not in (JOLTIK, GALVANTULA_EX)
    )
    candidates.extend(card for card in lines if card not in candidates)
    for target in candidates:
        for index, option in enumerate(options):
            if _slot_target(option, mine) is target:
                return [index]
    return None


def _prompt_override(observation: dict) -> list[int] | None:
    global _PENDING_PROMOTION
    select = observation.get("select") or {}
    current, me, mine, _ = _view(observation)
    if me is None:
        return None
    effect_id = (select.get("effect") or {}).get("id")
    lines = _lines(mine)
    hand_ids = {card.get("id") for card in _zone(mine, "hand")}

    if effect_id == JOLTIK:
        if select.get("context") == ENERGY_CHOICE:
            ids = [_card_id(option, select, current, me) for option in (select.get("option") or [])]
            preferred = BASIC_LIGHTNING if BASIC_LIGHTNING in ids else BASIC_GRASS
            return _choose_ids(
                select,
                current,
                me,
                [preferred] * max(1, int(select.get("maxCount") or 0)),
            )
        if select.get("context") == TARGET_OR_ATTACH:
            context_card = select.get("contextCard") or {}
            return _energy_target(select, mine, int(context_card.get("id") or 0))

    if effect_id == HILDA:
        return _choose_ids(select, current, me, [GALVANTULA_EX, BASIC_LIGHTNING, BASIC_GRASS])

    if effect_id == BUDDY_POFFIN:
        wanted = ([] if lines else [JOLTIK]) + [TAROUNTULA, JOLTIK]
        return _choose_ids(select, current, me, wanted)

    if effect_id == POKE_PAD:
        wanted = ([JOLTIK] if not lines else []) + [SPIDOPS, TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO]
        return _choose_ids(select, current, me, wanted)

    if effect_id == ULTRA_BALL:
        if select.get("context") == DISCARD:
            return _ultra_discards(select, current, me)
        mature_joltik = any(card.get("id") == JOLTIK and not card.get("appearThisTurn") for card in lines)
        wanted = ([GALVANTULA_EX] if mature_joltik else []) + ([JOLTIK] if not lines else []) + [SPIDOPS, MEWTWO_EX, TAROUNTULA]
        return _choose_ids(select, current, me, wanted)

    if effect_id == PETREL:
        needs_crystal = bool(lines and not any(_has_tool(card, SPARKLING_CRYSTAL) for card in lines))
        wanted = ([SPARKLING_CRYSTAL] if needs_crystal else []) + [SWITCH, ULTRA_BALL, NIGHT_STRETCHER, BUDDY_POFFIN]
        return _choose_ids(select, current, me, wanted)

    if effect_id == TRANSCEIVER:
        wanted = []
        if lines and SPARKLING_CRYSTAL not in hand_ids:
            wanted.append(PETREL)
        if not lines:
            wanted.append(PROTON)
        wanted.extend((PETREL, PROTON, ARIANA))
        return _choose_ids(select, current, me, wanted)

    if effect_id == PROTON:
        return _choose_ids(select, current, me, [TAROUNTULA, TAROUNTULA, MEWTWO_EX, MIMIKYU, ARTICUNO])

    if effect_id == NIGHT_STRETCHER:
        return _choose_ids(select, current, me, [GALVANTULA_EX, JOLTIK, BASIC_LIGHTNING, SPIDOPS, TAROUNTULA, BASIC_GRASS])

    if select.get("context") in (3, 4):
        ready = [card for card in lines if _galv_ready(card)]
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
    current, me, mine, _ = _view(observation)
    if me is None:
        return None
    active = _zone(mine, "active")
    lines = _lines(mine)
    hand_ids = {card.get("id") for card in _zone(mine, "hand")}

    if active and active[0].get("id") == GALVANTULA_EX:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == ATTACK and option.get("attackId") == CHARGED_WEB:
                return [index]

    if not lines:
        action = _play_option(observation, [JOLTIK, BUDDY_POFFIN, POKE_PAD])
        if action is not None:
            return action
        if _safe_ultra(mine):
            action = _play_option(observation, [ULTRA_BALL])
            if action is not None:
                return action
        return None

    joltik = next((card for card in lines if card.get("id") == JOLTIK), None)
    galv = next((card for card in lines if card.get("id") == GALVANTULA_EX), None)

    if joltik is not None and GALVANTULA_EX in hand_ids:
        action = _play_option(observation, [GALVANTULA_EX])
        if action is not None:
            return action

    if joltik is not None and not joltik.get("appearThisTurn") and GALVANTULA_EX not in hand_ids:
        if not bool(current.get("supporterPlayed")):
            action = _play_option(observation, [HILDA])
            if action is not None:
                return action
        if _safe_ultra(mine):
            action = _play_option(observation, [ULTRA_BALL])
            if action is not None:
                return action

    primary = galv or joltik
    if primary is not None and not _has_tool(primary, SPARKLING_CRYSTAL):
        action = _attach_option(observation, mine, primary, (SPARKLING_CRYSTAL,))
        if action is not None:
            return action

    if primary is not None and not bool(current.get("energyAttached")):
        wanted = (BASIC_LIGHTNING,) if not _has_energy_card(primary, BASIC_LIGHTNING) else (BASIC_GRASS, BASIC_LIGHTNING)
        action = _attach_option(observation, mine, primary, wanted)
        if action is not None:
            return action

    if galv is not None and _galv_ready(galv) and (not active or active[0] is not galv):
        action = _play_option(observation, [SWITCH])
        if action is not None:
            _PENDING_PROMOTION = True
            return action
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == RETREAT:
                _PENDING_PROMOTION = True
                return [index]

    if active and active[0].get("id") == JOLTIK:
        for index, option in enumerate(select.get("option") or []):
            if option.get("type") == ATTACK and option.get("attackId") == JOLTING_CHARGE:
                return [index]

    if joltik is not None and GALVANTULA_EX not in hand_ids and not bool(current.get("supporterPlayed")):
        action = _play_option(observation, [HILDA, PETREL])
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
    return _AGENT(observation)
