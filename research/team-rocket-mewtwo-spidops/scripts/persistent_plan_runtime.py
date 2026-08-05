"""Persistent public-plan logit overlay for the frozen Rocket blueprint.

This module is deliberately copyable into an arena candidate directory.  It
uses only ordinary redacted observations and the legal options supplied by the
official engine.  The semantic features are bounded research abstractions; the
underlying game execution and legality remain the official engine's.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from soldier import features as F
from soldier.engine import random_legal_choice
from soldier.net import NNAgent, attack_safe_logits, feat_to_tensors


PLAN_SCHEMA = "rocket-persistent-public-plan/v1"

AXES: tuple[str, ...] = (
    "board_setup",
    "mewtwo_access",
    "spidops_engine",
    "mewtwo_charge",
    "second_attacker",
    "charging_fuel",
    "draw_engine",
    "tool_assignment",
    "attack_tempo",
    "bench_discipline",
    "fuel_preservation",
)
AXIS_INDEX = {name: index for index, name in enumerate(AXES)}


# Official 2026 competition card IDs used by the immutable target deck.
GRASS = 1
PSYCHIC = 5
ROCKET_ENERGY = 15
TAROUNTULA = 400
SPIDOPS = 401
ARTICUNO = 414
MEWTWO = 431
WOBBUFFET = 432
MURKROW = 463
BUG_SET = 1094
NIGHT_STRETCHER = 1097
ENERGY_SEARCH = 1119
TRANSCEIVER = 1134
POKE_PAD = 1152
HERO_CAPE = 1159
BRAVE_BANGLE = 1175
ARIANA = 1216
ARCHER = 1217
GIOVANNI = 1218
PETREL = 1219
PROTON = 1220
LILLIE = 1227
FACTORY = 1257

MAIN = 0
HAND = 2
DISCARD = 3
ACTIVE = 4
BENCH = 5
STADIUM = 7

CARD = 3
PLAY = 7
ATTACH = 8
EVOLVE = 9
ABILITY = 10
RETREAT = 12
ATTACK = 13
END = 14

BASIC_ROCKET = frozenset({TAROUNTULA, ARTICUNO, MEWTWO, WOBBUFFET, MURKROW})
ROCKET_POKEMON = frozenset({TAROUNTULA, SPIDOPS, ARTICUNO, MEWTWO, WOBBUFFET, MURKROW})
ATTACKERS = frozenset({TAROUNTULA, SPIDOPS, MEWTWO, WOBBUFFET, MURKROW})
ENERGY_IDS = frozenset({GRASS, PSYCHIC, ROCKET_ENERGY})
TOOLS = frozenset({HERO_CAPE, BRAVE_BANGLE})


def normalized_weights(values: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    """Validate and return one finite float32 weight per registered axis."""
    if isinstance(values, Mapping):
        unknown = set(values) - set(AXES)
        if unknown:
            raise ValueError(f"unknown persistent-plan axes: {sorted(unknown)}")
        result = np.asarray([float(values.get(name, 0.0)) for name in AXES], dtype=np.float32)
    else:
        result = np.asarray(list(values), dtype=np.float32)
        if result.shape != (len(AXES),):
            raise ValueError(f"expected {len(AXES)} plan weights, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("persistent-plan weights must be finite")
    if float(np.max(np.abs(result), initial=0.0)) > 3.0:
        raise ValueError("persistent-plan weights are bounded to [-3, 3]")
    return result


def plan_record(name: str, weights: Mapping[str, float] | Sequence[float]) -> dict[str, Any]:
    vector = normalized_weights(weights)
    return {
        "schema": PLAN_SCHEMA,
        "name": str(name),
        "weights": {axis: float(vector[index]) for index, axis in enumerate(AXES)},
    }


def _zone(player: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    return [item for item in player.get(name) or () if isinstance(item, Mapping)]


def _board(player: Mapping[str, Any]) -> list[tuple[int, int, Mapping[str, Any]]]:
    result: list[tuple[int, int, Mapping[str, Any]]] = []
    result.extend((ACTIVE, index, item) for index, item in enumerate(_zone(player, "active")))
    result.extend((BENCH, index, item) for index, item in enumerate(_zone(player, "bench")))
    return result


def _energy_card_ids(pokemon: Mapping[str, Any]) -> list[int]:
    cards = [
        int(item.get("id") or 0)
        for item in pokemon.get("energyCards") or ()
        if isinstance(item, Mapping)
    ]
    if cards:
        return cards
    # Older observations can expose only produced Energy types.  Preserve a
    # conservative unit count without pretending type 11 is a hidden card.
    return [ROCKET_ENERGY if int(value) == 11 else int(value) for value in pokemon.get("energies") or ()]


def _energy_profile(card_ids: Sequence[int]) -> tuple[int, int, int]:
    rocket = sum(int(card_id) == ROCKET_ENERGY for card_id in card_ids)
    grass = sum(int(card_id) == GRASS for card_id in card_ids)
    psychic = sum(int(card_id) == PSYCHIC for card_id in card_ids) + 2 * rocket
    total = len(card_ids) + rocket
    return total, grass, psychic


def _ready(card_id: int, energy_cards: Sequence[int], rocket_board_count: int) -> bool:
    total, grass, psychic = _energy_profile(energy_cards)
    if card_id == MEWTWO:
        return rocket_board_count >= 4 and psychic >= 2 and total >= 3
    if card_id == SPIDOPS:
        return grass >= 1 and total >= 2
    if card_id == TAROUNTULA:
        return grass >= 1
    if card_id == WOBBUFFET:
        return psychic >= 1 and total >= 2
    if card_id == MURKROW:
        return total >= 1
    # Articuno requires Water, which the immutable list does not contain.
    return False


def _target_pokemon(
    option: Mapping[str, Any],
    current: Mapping[str, Any],
    me: int,
) -> tuple[int, int, Mapping[str, Any]] | None:
    area = option.get("inPlayArea")
    index = option.get("inPlayIndex")
    if area is None and option.get("area") in (ACTIVE, BENCH):
        area = option.get("area")
        index = option.get("index")
    player_index = option.get("playerIndex")
    player_index = me if player_index is None else int(player_index)
    players = current.get("players") or ()
    if player_index != me or area not in (ACTIVE, BENCH) or not isinstance(index, int):
        return None
    if not (0 <= me < len(players)):
        return None
    zone = "active" if int(area) == ACTIVE else "bench"
    values = _zone(players[me] or {}, zone)
    if not (0 <= index < len(values)):
        return None
    return int(area), int(index), values[index]


def _card_id(
    option: Mapping[str, Any],
    current: Mapping[str, Any],
    me: int,
    select: Mapping[str, Any],
) -> int:
    return int(F._opt_card_id(option, current, me, select) or 0)


def _readiness_gain(
    pokemon: Mapping[str, Any],
    energy_card_id: int,
    rocket_board_count: int,
) -> float:
    pokemon_id = int(pokemon.get("id") or 0)
    before_cards = _energy_card_ids(pokemon)
    before = _ready(pokemon_id, before_cards, rocket_board_count)
    after = _ready(pokemon_id, [*before_cards, int(energy_card_id)], rocket_board_count)
    if before:
        return -0.35
    if after:
        return 1.0
    before_total = _energy_profile(before_cards)[0]
    after_total = _energy_profile([*before_cards, int(energy_card_id)])[0]
    return min(0.7, 0.2 * max(1, after_total - before_total))


def _remove_energy_value(
    pokemon: Mapping[str, Any],
    energy_index: int,
    rocket_board_count: int,
) -> float:
    cards = _energy_card_ids(pokemon)
    if not cards:
        return 0.0
    index = int(energy_index)
    if not (0 <= index < len(cards)):
        index = 0
    was_ready = _ready(int(pokemon.get("id") or 0), cards, rocket_board_count)
    remaining = cards[:index] + cards[index + 1 :]
    stays_ready = _ready(int(pokemon.get("id") or 0), remaining, rocket_board_count)
    if was_ready and not stays_ready:
        return -1.0
    if stays_ready:
        return 1.0
    return 0.25 if len(cards) >= 2 else -0.25


def semantic_option_features(
    observation: Mapping[str, Any],
    option: Mapping[str, Any],
    *,
    picked_card_ids: Sequence[int] = (),
) -> np.ndarray:
    """Return bounded strategic features using only the acting observation."""
    result = np.zeros(len(AXES), dtype=np.float32)
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    players = current.get("players") or ()
    me = int(current.get("yourIndex") or 0)
    if me not in (0, 1) or len(players) != 2:
        return result
    mine = players[me] or {}
    own_board = _board(mine)
    board_ids = [int(pokemon.get("id") or 0) for _, _, pokemon in own_board]
    counts = Counter(board_ids)
    picked = Counter(map(int, picked_card_ids))
    rocket_count = sum(card_id in ROCKET_POKEMON for card_id in board_ids)
    board_count_for_plan = rocket_count + sum(picked[card_id] for card_id in BASIC_ROCKET)
    turn = int(current.get("turn") or 0)
    option_type = int(option.get("type") or 0)
    effect_id = int((select.get("effect") or {}).get("id") or 0)
    card_id = _card_id(option, current, me, select)
    target = _target_pokemon(option, current, me)
    target_id = int(target[2].get("id") or 0) if target else 0
    target_area = int(target[0]) if target else 0
    hand_ids = [int(card.get("id") or 0) for card in _zone(mine, "hand")]
    hand_count = int(mine.get("handCount") or len(hand_ids))
    prize_count = len(mine.get("prize") or ())
    direct_card_action = option_type in (PLAY, ATTACH, EVOLVE)
    search_card_action = option_type == CARD and effect_id in {
        PROTON,
        POKE_PAD,
        TRANSCEIVER,
        BUG_SET,
        ENERGY_SEARCH,
        PETREL,
        NIGHT_STRETCHER,
        SPIDOPS,
    }

    # Establish the four-Rocket Power Saver threshold, with diminishing value
    # after the threshold and no generic bonus for late liabilities.
    if card_id in BASIC_ROCKET and (direct_card_action or search_card_action):
        if board_count_for_plan < 4:
            result[AXIS_INDEX["board_setup"]] = max(0.25, (4.0 - board_count_for_plan) / 4.0)

    if card_id == MEWTWO and (direct_card_action or search_card_action):
        known_mewtwo = counts[MEWTWO] + picked[MEWTWO]
        if known_mewtwo == 0:
            result[AXIS_INDEX["mewtwo_access"]] = 1.0 if turn <= 8 else 0.55
        elif known_mewtwo == 1 and turn >= 8:
            result[AXIS_INDEX["mewtwo_access"]] = 0.15

    tar_count = counts[TAROUNTULA] + picked[TAROUNTULA]
    spidops_count = counts[SPIDOPS] + picked[SPIDOPS]
    hand_spidops = hand_ids.count(SPIDOPS)
    if card_id == SPIDOPS and (direct_card_action or search_card_action):
        if tar_count > 0 and hand_spidops + picked[SPIDOPS] < tar_count:
            result[AXIS_INDEX["spidops_engine"]] = 1.0
        elif tar_count > 0:
            result[AXIS_INDEX["spidops_engine"]] = 0.45
    elif card_id == TAROUNTULA and (direct_card_action or search_card_action):
        if tar_count + spidops_count < 2 and turn <= 8:
            result[AXIS_INDEX["spidops_engine"]] = 0.8

    # Hand Energy and Tool attachments share option type 8; immutable card IDs
    # disambiguate them without consulting private opponent information.
    if option_type == ATTACH and target is not None and card_id in ENERGY_IDS:
        gain = _readiness_gain(target[2], card_id, max(rocket_count, 1))
        if target_id == MEWTWO:
            current_cards = _energy_card_ids(target[2])
            _, _, psychic = _energy_profile(current_cards)
            compatibility = 1.0 if card_id == ROCKET_ENERGY and psychic < 2 else 0.75
            if card_id == PSYCHIC and psychic < 2:
                compatibility = 0.8
            result[AXIS_INDEX["mewtwo_charge"]] = max(-0.5, compatibility + 0.55 * gain)
        elif target_id in ATTACKERS:
            result[AXIS_INDEX["second_attacker"]] = gain + (0.2 if target_area == BENCH else 0.0)

        ready_attackers = [
            (area, pokemon)
            for area, _, pokemon in own_board
            if int(pokemon.get("id") or 0) in ATTACKERS
            and _ready(int(pokemon.get("id") or 0), _energy_card_ids(pokemon), rocket_count)
        ]
        unready_bench = [
            pokemon
            for area, _, pokemon in own_board
            if area == BENCH
            and int(pokemon.get("id") or 0) in ATTACKERS
            and not _ready(int(pokemon.get("id") or 0), _energy_card_ids(pokemon), rocket_count)
        ]
        if target_area == ACTIVE and ready_attackers and unready_bench:
            result[AXIS_INDEX["second_attacker"]] -= 0.8

    if effect_id == ENERGY_SEARCH and card_id in (PSYCHIC, GRASS):
        if counts[MEWTWO] and card_id == PSYCHIC:
            result[AXIS_INDEX["mewtwo_charge"]] += 0.7
        if counts[SPIDOPS] and card_id == GRASS:
            result[AXIS_INDEX["second_attacker"]] += 0.45

    if option_type == ABILITY and target_id == SPIDOPS:
        basic_energy_in_discard = any(
            int(card.get("id") or 0) in (GRASS, PSYCHIC) for card in _zone(mine, "discard")
        )
        if basic_energy_in_discard:
            ready = _ready(SPIDOPS, _energy_card_ids(target[2]), rocket_count)
            result[AXIS_INDEX["charging_fuel"]] = 0.65 if ready and target_area == BENCH else 1.0
            if not ready:
                result[AXIS_INDEX["second_attacker"]] += 0.45
    if effect_id == SPIDOPS and card_id in (GRASS, PSYCHIC):
        # The effect itself attaches to the activating Spidops.  Grass is the
        # typed requirement; either Basic Energy can become Erasure Ball fuel.
        result[AXIS_INDEX["charging_fuel"]] = 1.0 if card_id == GRASS else 0.45

    # Public setup/draw sequencing.  These values do not predict a hidden hand;
    # they only use our own visible hand size and prize count.
    if option_type == PLAY:
        if card_id == FACTORY and turn <= 8:
            result[AXIS_INDEX["draw_engine"]] = 0.8
        elif card_id == ARIANA:
            result[AXIS_INDEX["draw_engine"]] = 1.0 if hand_count <= 5 else -0.35
        elif card_id == LILLIE:
            result[AXIS_INDEX["draw_engine"]] = 0.9 if prize_count == 6 and hand_count <= 6 else (0.45 if hand_count <= 4 else -0.2)
        elif card_id in (TRANSCEIVER, POKE_PAD, BUG_SET, ENERGY_SEARCH) and turn <= 8:
            result[AXIS_INDEX["draw_engine"]] = 0.4
        elif card_id == PROTON and board_count_for_plan < 4:
            result[AXIS_INDEX["draw_engine"]] = 0.6
    if option_type == ABILITY and option.get("area") == STADIUM:
        result[AXIS_INDEX["draw_engine"]] = 1.0
    if effect_id == TRANSCEIVER:
        if card_id == ARIANA and hand_count <= 5:
            result[AXIS_INDEX["draw_engine"]] = 1.0
        elif card_id == PROTON and board_count_for_plan < 4:
            result[AXIS_INDEX["draw_engine"]] = 0.8

    if option_type == ATTACH and target is not None and card_id in TOOLS:
        if card_id == HERO_CAPE:
            if target_id == MEWTWO:
                result[AXIS_INDEX["tool_assignment"]] = 1.0
            elif target_id == SPIDOPS:
                result[AXIS_INDEX["tool_assignment"]] = 0.5
        elif card_id == BRAVE_BANGLE:
            result[AXIS_INDEX["tool_assignment"]] = 1.0 if target_id == SPIDOPS else (0.35 if target_id in ATTACKERS - {MEWTWO} else -1.0)

    if option_type == ATTACK:
        result[AXIS_INDEX["attack_tempo"]] = 1.0

    if card_id in BASIC_ROCKET and option_type == PLAY and board_count_for_plan >= 4:
        strategically_missing = card_id == MEWTWO and counts[MEWTWO] == 0
        result[AXIS_INDEX["bench_discipline"]] = 0.0 if strategically_missing else -1.0
    elif option_type in (ATTACK, END) and board_count_for_plan >= 4:
        result[AXIS_INDEX["bench_discipline"]] = 0.2

    # Erasure Ball's optional discard targets own Benched Energy.  Favor an
    # Energy card that leaves its host ready, and penalize disabling the only
    # credible backup attacker.
    if effect_id == MEWTWO and option_type in (5, 6) and target is not None:
        result[AXIS_INDEX["fuel_preservation"]] = _remove_energy_value(
            target[2], int(option.get("energyIndex") or 0), rocket_count
        )

    return np.clip(result, -1.0, 1.0)


class PersistentPlanAgent:
    """Greedy frozen blueprint with one bounded plan vector for the whole game."""

    def __init__(
        self,
        net: torch.nn.Module,
        weights: Mapping[str, float] | Sequence[float],
        *,
        plan_name: str = "persistent_plan",
        device: str = "cpu",
        deck_ids: Sequence[int] | None = None,
        opponent_deck_priors: Mapping[str, Sequence[int]] | None = None,
        forbid_end_when_attack: bool = True,
        max_abs_bias: float = 2.5,
    ) -> None:
        if getattr(net, "mem", 0):
            raise ValueError("persistent-plan v1 requires the frozen memory-free blueprint")
        self.net = net
        self.device = device
        self.plan_name = str(plan_name)
        self.weights = normalized_weights(weights)
        self.max_abs_bias = float(max_abs_bias)
        if not math.isfinite(self.max_abs_bias) or not (0.0 < self.max_abs_bias <= 5.0):
            raise ValueError("max_abs_bias must be finite and in (0, 5]")
        self.blueprint = NNAgent(
            net,
            greedy=True,
            device=device,
            deck_ids=deck_ids,
            forbid_end_when_attack=forbid_end_when_attack,
            opponent_deck_priors=opponent_deck_priors,
        )
        self.belief = self.blueprint.belief
        self.slot_link = self.blueprint.slot_link
        self.forbid_end_when_attack = bool(forbid_end_when_attack)
        self.decision_count = 0
        self.intervention_count = 0
        self.exception_fallback_count = 0
        self.accumulated_abs_bias = 0.0
        self.last_decision: dict[str, Any] | None = None

    def _biases(
        self,
        observation: Mapping[str, Any],
        *,
        picked_card_ids: Sequence[int] = (),
    ) -> torch.Tensor:
        options = (observation.get("select") or {}).get("option") or ()
        features = np.stack(
            [
                semantic_option_features(
                    observation,
                    option,
                    picked_card_ids=picked_card_ids,
                )
                for option in options
            ],
            axis=0,
        )
        raw = features @ self.weights
        clipped = np.clip(raw, -self.max_abs_bias, self.max_abs_bias).astype(np.float32)
        return torch.as_tensor(clipped, dtype=torch.float32, device=self.device)

    def _record(self, baseline: int, chosen: int, bias: float, pick: int) -> None:
        changed = int(baseline) != int(chosen)
        self.decision_count += 1
        self.intervention_count += int(changed)
        self.accumulated_abs_bias += abs(float(bias))
        self.last_decision = {
            "schema": PLAN_SCHEMA,
            "plan": self.plan_name,
            "pick": int(pick),
            "baseline_index": int(baseline),
            "chosen_index": int(chosen),
            "changed": bool(changed),
            "chosen_bias": float(bias),
            "interventions": self.intervention_count,
        }

    @torch.no_grad()
    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        select = observation.get("select") or {}
        options = select.get("option") or ()
        count = len(options)
        low = int(select.get("minCount") or 0)
        high = min(int(select.get("maxCount") or 0), count)
        if high == 0:
            return []
        try:
            base = F.extract(
                observation,
                belief=self.belief,
                slot_link=self.slot_link,
            )
            if high == 1:
                tensors = feat_to_tensors(base, self.device)
                logits, _ = self.net(tensors)
                if self.forbid_end_when_attack:
                    logits = attack_safe_logits(logits, options)
                baseline = int(torch.argmax(logits))
                biases = self._biases(observation)
                adjusted = logits + biases
                if self.forbid_end_when_attack:
                    adjusted = attack_safe_logits(adjusted, options)
                chosen = int(torch.argmax(adjusted))
                self._record(baseline, chosen, float(biases[chosen]), 0)
                return [chosen]

            picked: list[int] = []
            picked_card_ids: list[int] = []
            for pick in range(high):
                conditional = F.mark_selection(base, picked, high)
                tensors = feat_to_tensors(conditional, self.device)
                logits, _ = self.net(tensors)
                baseline = int(torch.argmax(logits))
                biases = self._biases(
                    observation,
                    picked_card_ids=picked_card_ids,
                )
                if picked:
                    biases = biases.clone()
                    biases[picked] = 0.0
                adjusted = logits + biases
                chosen = int(torch.argmax(adjusted))
                self._record(baseline, chosen, float(biases[chosen]), pick)
                picked.append(chosen)
                picked_card_ids.append(
                    _card_id(options[chosen], observation.get("current") or {}, int((observation.get("current") or {}).get("yourIndex") or 0), select)
                )
            if len(picked) < low:
                raise RuntimeError("persistent plan produced fewer than minCount picks")
            return picked
        except Exception:
            self.exception_fallback_count += 1
            return random_legal_choice(select)

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "plan": self.plan_name,
            "decisions": self.decision_count,
            "interventions": self.intervention_count,
            "exception_fallbacks": self.exception_fallback_count,
            "mean_abs_chosen_bias": self.accumulated_abs_bias / max(1, self.decision_count),
            "weights": {axis: float(self.weights[index]) for index, axis in enumerate(AXES)},
        }
