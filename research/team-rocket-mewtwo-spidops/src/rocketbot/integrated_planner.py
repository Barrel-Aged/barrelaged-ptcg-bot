"""Dependency-free deployed belief-aware strategic planner.

The training side exports ordinary dense-layer tensors to JSON.  This module
implements the exact forward pass using only the Python standard library so the
official arena archive does not need PyTorch.  Native legality remains owned by
``SelectionContract``; the model only ranks contract-valid tokens.

The preserved Phase 6 heuristic is accepted as one tactical feature and as the
last exception fallback.  It is not the strategy provider.  The live provider
is the frozen public-belief plan blueprint plus its autoregressive neural action
head.  Safe re-solving is fail closed: without an admitted calibration
certificate it returns the blueprint distribution byte-for-byte.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

try:
    from .competition import (
        CompetitionAgentConfig,
        DeckResolution,
        OptionType,
        Phase6CompetitionAgent,
        SelectContext,
        SelectionContract,
        StrategicPlan,
        _card_id,
        _in_play,
        _player,
        _safe_int,
        _state,
        _your_index,
        option_card_id,
        option_target_pokemon_id,
    )
except ImportError:  # Flat imports in the generated arena archive.
    from rocketbot_competition import (  # type: ignore[no-redef]
        CompetitionAgentConfig,
        DeckResolution,
        OptionType,
        Phase6CompetitionAgent,
        SelectContext,
        SelectionContract,
        StrategicPlan,
        _card_id,
        _in_play,
        _player,
        _safe_int,
        _state,
        _your_index,
        option_card_id,
        option_target_pokemon_id,
    )


INTEGRATED_PLANNER_SCHEMA_VERSION = 1
PLAN_NAMES = (
    "board_setup",
    "energy_engine",
    "prize_push",
    "recovery",
    "control",
)
STRATEGIC_AXES = (
    "setup",
    "tempo",
    "resource_efficiency",
    "prize_control",
    "board_control",
    "optionality",
)
INTENT_NAMES = (
    "setup",
    "prize_race",
    "lock_disruption",
    "board_denial",
    "resource_conservation",
    "combo_assembly",
)
RESOURCE_NAMES = (
    "basic_pokemon",
    "evolution",
    "energy",
    "search",
    "draw",
    "switch_mobility",
    "recovery",
    "disruption",
    "tool",
    "stadium",
)
ARCHETYPE_NAMES = (
    "abomasnow",
    "alakazam",
    "crustle",
    "dragapult",
    "rocket_mewtwo",
    "starmie",
)
HISTORY_CATEGORIES = (
    "ability",
    "attach",
    "attack",
    "end",
    "evolve",
    "other",
    "play",
    "retreat",
)
HISTORY_FEATURE_NAMES = tuple(
    f"{owner}_main_action_{category}_count"
    for owner in ("own", "opponent")
    for category in HISTORY_CATEGORIES
)

_TYPE_TO_HISTORY = {
    int(OptionType.ABILITY): "ability",
    int(OptionType.ATTACH): "attach",
    int(OptionType.ATTACK): "attack",
    int(OptionType.END): "end",
    int(OptionType.EVOLVE): "evolve",
    int(OptionType.PLAY): "play",
    int(OptionType.RETREAT): "retreat",
}

_PLAN_TYPE_BIASES: Mapping[str, Mapping[int, float]] = {
    "board_setup": {
        int(OptionType.PLAY): 1.0,
        int(OptionType.EVOLVE): 1.0,
        int(OptionType.ABILITY): 0.55,
        int(OptionType.ATTACH): 0.35,
        int(OptionType.ATTACK): -0.35,
        int(OptionType.END): -0.7,
    },
    "energy_engine": {
        int(OptionType.ATTACH): 1.0,
        int(OptionType.ABILITY): 0.55,
        int(OptionType.PLAY): 0.4,
        int(OptionType.END): -0.7,
    },
    "prize_push": {
        int(OptionType.ATTACK): 1.0,
        int(OptionType.ABILITY): 0.3,
        int(OptionType.PLAY): 0.15,
        int(OptionType.RETREAT): -0.2,
        int(OptionType.END): -0.4,
    },
    "recovery": {
        int(OptionType.PLAY): 0.85,
        int(OptionType.RETREAT): 0.65,
        int(OptionType.ABILITY): 0.4,
        int(OptionType.ATTACK): -0.1,
    },
    "control": {
        int(OptionType.ABILITY): 0.9,
        int(OptionType.PLAY): 0.65,
        int(OptionType.RETREAT): 0.3,
        int(OptionType.ATTACK): 0.2,
    },
}

_AXIS_TYPE_BIASES: Mapping[str, Mapping[int, float]] = {
    "setup": {int(OptionType.PLAY): 1.0, int(OptionType.EVOLVE): 1.0},
    "tempo": {int(OptionType.ATTACK): 1.0, int(OptionType.ATTACH): 0.6},
    "resource_efficiency": {
        int(OptionType.ABILITY): 0.8,
        int(OptionType.END): 0.25,
        int(OptionType.PLAY): 0.2,
    },
    "prize_control": {int(OptionType.ATTACK): 1.0, int(OptionType.PLAY): 0.25},
    "board_control": {
        int(OptionType.PLAY): 0.55,
        int(OptionType.RETREAT): 0.5,
        int(OptionType.ABILITY): 0.45,
    },
    "optionality": {
        int(OptionType.ABILITY): 0.7,
        int(OptionType.PLAY): 0.45,
        int(OptionType.END): 0.2,
    },
}

_INTENT_ACTION_PROFILES: Mapping[str, Mapping[str, float]] = {
    "setup": {"ability": 0.14, "attach": 0.15, "attack": 0.06, "end": 0.05, "evolve": 0.23, "other": 0.03, "play": 0.31, "retreat": 0.03},
    "prize_race": {"ability": 0.10, "attach": 0.24, "attack": 0.32, "end": 0.03, "evolve": 0.10, "other": 0.02, "play": 0.15, "retreat": 0.04},
    "lock_disruption": {"ability": 0.21, "attach": 0.10, "attack": 0.10, "end": 0.04, "evolve": 0.09, "other": 0.04, "play": 0.34, "retreat": 0.08},
    "board_denial": {"ability": 0.19, "attach": 0.11, "attack": 0.18, "end": 0.03, "evolve": 0.11, "other": 0.03, "play": 0.27, "retreat": 0.08},
    "resource_conservation": {"ability": 0.16, "attach": 0.10, "attack": 0.11, "end": 0.16, "evolve": 0.09, "other": 0.04, "play": 0.23, "retreat": 0.11},
    "combo_assembly": {"ability": 0.19, "attach": 0.16, "attack": 0.07, "end": 0.04, "evolve": 0.18, "other": 0.03, "play": 0.29, "retreat": 0.04},
}

_INTENT_RESOURCE_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "setup": ("basic_pokemon", "evolution", "search", "draw"),
    "prize_race": ("energy", "switch_mobility", "tool"),
    "lock_disruption": ("disruption", "search", "draw"),
    "board_denial": ("disruption", "switch_mobility", "energy"),
    "resource_conservation": ("recovery", "draw", "energy"),
    "combo_assembly": ("search", "draw", "evolution", "energy"),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _stable_code(value: Any) -> float:
    digest = sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / float(2**32 - 1)


def _card_bucket(card_id: int) -> int:
    digest = sha256(f"official-card:{int(card_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 32


def _sigmoid(value: float) -> float:
    if value >= 0:
        term = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + term)
    term = math.exp(max(value, -60.0))
    return term / (1.0 + term)


def _softmax(values: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    if not values:
        return ()
    scale = max(float(temperature), 1e-6)
    maximum = max(values)
    exponentials = [math.exp(max(-60.0, min(60.0, (value - maximum) / scale))) for value in values]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities)


def behavior_intent_probabilities(features: Mapping[str, float]) -> tuple[float, ...]:
    """Likelihood of observed opponent actions under each intent alternative."""

    counts = {
        category: max(0.0, float(features.get(f"opponent_main_action_{category}_count", 0.0)))
        for category in HISTORY_CATEGORIES
    }
    if sum(counts.values()) <= 0:
        return tuple(1.0 / len(INTENT_NAMES) for _ in INTENT_NAMES)
    log_likelihoods = []
    for intent in INTENT_NAMES:
        profile = _INTENT_ACTION_PROFILES[intent]
        log_likelihoods.append(
            sum(count * math.log(max(profile[category], 1e-8)) for category, count in counts.items())
        )
    return _softmax(log_likelihoods)


def _fuse_intent_evidence(
    model_probabilities: Sequence[float],
    behavior_probabilities: Sequence[float],
    weight: float,
) -> tuple[float, ...]:
    weight = min(1.0, max(0.0, float(weight)))
    logs = [
        (1.0 - weight) * math.log(max(model, 1e-12))
        + weight * math.log(max(behavior, 1e-12))
        for model, behavior in zip(model_probabilities, behavior_probabilities, strict=True)
    ]
    return _softmax(logs)


def hidden_resource_hypotheses(evaluation: "PlannerEvaluation") -> tuple[dict[str, Any], ...]:
    """Materialize multiple public-consistent intent/resource/style hypotheses."""

    style_priors = (("rational", 0.72), ("imperfect", 0.20), ("deceptive", 0.08))
    candidates: list[dict[str, Any]] = []
    resource_map = dict(zip(RESOURCE_NAMES, evaluation.resource_probabilities, strict=True))
    for intent_index, intent in enumerate(INTENT_NAMES):
        required = _INTENT_RESOURCE_REQUIREMENTS[intent]
        enabling = tuple(
            sorted(required, key=lambda name: (-resource_map[name], name))[:3]
        )
        support = math.prod(max(0.05, resource_map[name]) for name in enabling) ** (1.0 / max(1, len(enabling)))
        for style, style_prior in style_priors:
            probability = evaluation.intent_probabilities[intent_index] * style_prior * support
            candidates.append(
                {
                    "hypothesis_id": f"{intent}:{style}",
                    "intent": intent,
                    "play_style": style,
                    "minimum_enabling_resource_categories": list(enabling),
                    "unnormalized_probability": probability,
                }
            )
    retained = sorted(candidates, key=lambda item: (-float(item["unnormalized_probability"]), str(item["hypothesis_id"])))[:8]
    total = sum(float(item["unnormalized_probability"]) for item in retained)
    for item in retained:
        item["probability"] = float(item.pop("unnormalized_probability")) / max(total, 1e-12)
    return tuple(retained)


def _dense(values: Sequence[float], layer: Mapping[str, Any]) -> list[float]:
    weights = layer["weight"]
    biases = layer["bias"]
    if len(weights) != len(biases):
        raise ValueError("frozen dense layer output shape mismatch")
    result = []
    for row, bias in zip(weights, biases, strict=True):
        if len(row) != len(values):
            raise ValueError("frozen dense layer input shape mismatch")
        result.append(float(bias) + sum(float(weight) * float(value) for weight, value in zip(row, values, strict=True)))
    return result


def _tanh_dense(values: Sequence[float], layer: Mapping[str, Any]) -> list[float]:
    return [math.tanh(value) for value in _dense(values, layer)]


def _normalise(
    values: Mapping[str, float],
    normalizer: Mapping[str, Any],
) -> list[float]:
    names = normalizer["names"]
    means = normalizer["means"]
    scales = normalizer["scales"]
    if not (len(names) == len(means) == len(scales)):
        raise ValueError("frozen normalizer shape mismatch")
    return [
        (float(values.get(str(name), 0.0)) - float(mean)) / max(float(scale), 1e-8)
        for name, mean, scale in zip(names, means, scales, strict=True)
    ]


def _card_id_from_any(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return _card_id(value)
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _public_player_view(player: Mapping[str, Any]) -> dict[str, Any]:
    def pokemon_view(raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, Mapping):
            return None
        return {
            "id": _card_id(raw),
            "hp": _safe_int(raw.get("hp"), 0),
            "max_hp": _safe_int(raw.get("maxHp"), 0),
            "appear_this_turn": bool(raw.get("appearThisTurn", False)),
            "energies": tuple(_safe_int(item, 0) for item in _sequence(raw.get("energies"))),
            "energy_cards": tuple(_card_id_from_any(item) for item in _sequence(raw.get("energyCards"))),
            "tools": tuple(_card_id_from_any(item) for item in _sequence(raw.get("tools"))),
            "pre_evolution": tuple(_card_id_from_any(item) for item in _sequence(raw.get("preEvolution"))),
        }

    return {
        "active": tuple(pokemon_view(item) for item in _sequence(player.get("active")) if pokemon_view(item) is not None),
        "bench": tuple(pokemon_view(item) for item in _sequence(player.get("bench")) if pokemon_view(item) is not None),
        "deck_count": _safe_int(player.get("deckCount"), 0),
        "discard": tuple(_card_id_from_any(item) for item in _sequence(player.get("discard"))),
        "prize_count": len(_sequence(player.get("prize"))),
        "hand_count": _safe_int(player.get("handCount"), len(_sequence(player.get("hand")))),
        "status": (
            bool(player.get("poisoned", False)),
            bool(player.get("burned", False)),
            bool(player.get("asleep", False)),
            bool(player.get("paralyzed", False)),
            bool(player.get("confused", False)),
        ),
    }


def _public_card_ids(player: Mapping[str, Any]) -> tuple[int, ...]:
    values: list[int] = []
    for raw in _sequence(player.get("discard")):
        card_id = _card_id_from_any(raw)
        if card_id is not None:
            values.append(card_id)
    for raw in tuple(_sequence(player.get("active"))) + tuple(_sequence(player.get("bench"))):
        if not isinstance(raw, Mapping):
            continue
        for candidate in (raw,) + tuple(_sequence(raw.get("energyCards"))) + tuple(_sequence(raw.get("tools"))) + tuple(_sequence(raw.get("preEvolution"))):
            card_id = _card_id_from_any(candidate)
            if card_id is not None:
                values.append(card_id)
    return tuple(values)


PUBLIC_BASE_NAMES = (
    "turn",
    "turn_action_count",
    "own_board_count",
    "own_board_hp",
    "own_hand_count",
    "own_prize_count",
    "own_deck_count",
    "opponent_board_count",
    "opponent_board_hp",
    "opponent_hand_count",
    "opponent_prize_count",
    "opponent_deck_count",
    "opponent_public_revealed_copies",
    "opponent_range_entropy",
    "opponent_range_effective_sample_size",
)
PUBLIC_FEATURE_NAMES = tuple(
    sorted(
        PUBLIC_BASE_NAMES
        + tuple(f"own_public_card_bucket_{index:02d}" for index in range(32))
        + tuple(f"opponent_public_card_bucket_{index:02d}" for index in range(32))
        + tuple(f"stadium_card_bucket_{index:02d}" for index in range(32))
    )
)
PRIVATE_BASE_NAMES = (
    "selection_type_code",
    "selection_context_code",
    "selection_min_count",
    "selection_max_count",
    "selection_option_count",
    "remain_damage_counter",
    "remain_energy_cost",
    "own_hand_count",
    "private_event_identity_count",
    "effect_card_code",
    "context_card_code",
)
PRIVATE_FEATURE_NAMES = tuple(
    sorted(
        PRIVATE_BASE_NAMES
        + tuple(f"own_hand_card_bucket_{index:02d}" for index in range(32))
        + tuple(f"visible_search_card_bucket_{index:02d}" for index in range(32))
    )
)
PLANNER_FEATURE_NAMES = PUBLIC_FEATURE_NAMES + PRIVATE_FEATURE_NAMES + HISTORY_FEATURE_NAMES

TOKEN_FEATURE_NAMES = (
    (
        "owner_self",
        "owner_opponent",
        "active",
        "bench",
        "position",
        "card_code",
        "card_bucket_sin",
        "card_bucket_cos",
    )
    + tuple(f"card_bucket_{index:02d}" for index in range(32))
    + (
        "hp",
        "max_hp",
        "damage",
        "energy_count",
        "grass_energy",
        "psychic_energy",
        "rocket_energy",
        "other_energy",
        "tool_count",
        "evolution_depth",
        "appeared_this_turn",
        "status_count",
    )
)

ACTION_FEATURE_NAMES = (
    tuple(f"option_type_{index:02d}" for index in range(16))
    + tuple(f"option_card_bucket_{index:02d}" for index in range(32))
    + tuple(f"option_attack_bucket_{index:02d}" for index in range(16))
    + tuple(f"option_target_bucket_{index:02d}" for index in range(32))
    + tuple(f"plan_alignment_{name}" for name in PLAN_NAMES)
    + tuple(f"axis_alignment_{name}" for name in STRATEGIC_AXES)
    + (
        "stop_token",
        "option_number",
        "option_area",
        "option_index",
        "option_player_index",
        "option_count",
        "option_in_play_area",
        "option_in_play_index",
        "option_card_code",
        "option_attack_code",
        "option_target_code",
        "baseline_score",
        "baseline_rank",
        "baseline_gap",
        "prefix_length",
        "remaining_fraction",
    )
)

PREFIX_FEATURE_NAMES = tuple(f"prefix_type_{index:02d}" for index in range(16)) + (
    "prefix_length",
    "prefix_card_code_mean",
)

SEARCH_GATE_FEATURE_NAMES = (
    "original_gate_probability",
    "uncertainty",
    "plan_entropy",
    "intent_entropy",
    "archetype_entropy",
    "plan_value_spread",
    "best_predicted_advantage",
    "strategic_disagreement",
    "low_policy_confidence",
    "prize_threshold",
    "turn",
    "selection_option_count",
    "board_hp_delta",
    "board_count_delta",
)


def search_gate_feature_items(
    features: Mapping[str, float],
    *,
    original_gate_probability: float,
    uncertainty: float,
    archetype_probabilities: Sequence[float],
    intent_probabilities: Sequence[float],
    plan_probabilities: Sequence[float],
    plan_values: Sequence[float],
    strategic_values: Sequence[float],
) -> dict[str, float]:
    baseline = sum(probability * value for probability, value in zip(plan_probabilities, plan_values, strict=True))
    return {
        "original_gate_probability": float(original_gate_probability),
        "uncertainty": float(uncertainty),
        "plan_entropy": _entropy(plan_probabilities),
        "intent_entropy": _entropy(intent_probabilities),
        "archetype_entropy": _entropy(archetype_probabilities),
        "plan_value_spread": max(plan_values) - min(plan_values),
        "best_predicted_advantage": max(plan_values) - baseline,
        "strategic_disagreement": statistics.pstdev(strategic_values),
        "low_policy_confidence": 1.0 - max(plan_probabilities),
        "prize_threshold": max(
            0.0,
            (3.0 - min(float(features.get("own_prize_count", 6.0)), float(features.get("opponent_prize_count", 6.0)))) / 3.0,
        ),
        "turn": float(features.get("turn", 0.0)),
        "selection_option_count": float(features.get("selection_option_count", 0.0)),
        "board_hp_delta": float(features.get("own_board_hp", 0.0)) - float(features.get("opponent_board_hp", 0.0)),
        "board_count_delta": float(features.get("own_board_count", 0.0)) - float(features.get("opponent_board_count", 0.0)),
    }


@dataclass(slots=True)
class PlannerMemory:
    plan_latent: list[float] = field(default_factory=lambda: [1.0 / len(PLAN_NAMES)] * len(PLAN_NAMES))
    archetype_posterior: list[float] = field(default_factory=lambda: [1.0 / len(ARCHETYPE_NAMES)] * len(ARCHETYPE_NAMES))
    intent_posterior: list[float] = field(default_factory=lambda: [1.0 / len(INTENT_NAMES)] * len(INTENT_NAMES))
    resource_posterior: list[float] = field(default_factory=lambda: [0.5] * len(RESOURCE_NAMES))
    own_history: Counter[str] = field(default_factory=Counter)
    opponent_history: Counter[str] = field(default_factory=Counter)
    last_observation_summary: dict[str, float] | None = None
    last_turn: int = -1
    last_plan_turn: int = -1
    decision_index: int = 0
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    hidden_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    exception_fallback_count: int = 0
    exception_types: Counter[str] = field(default_factory=Counter)
    tactical_teacher_deference_count: int = 0

    def reset(self) -> None:
        self.plan_latent = [1.0 / len(PLAN_NAMES)] * len(PLAN_NAMES)
        self.archetype_posterior = [1.0 / len(ARCHETYPE_NAMES)] * len(ARCHETYPE_NAMES)
        self.intent_posterior = [1.0 / len(INTENT_NAMES)] * len(INTENT_NAMES)
        self.resource_posterior = [0.5] * len(RESOURCE_NAMES)
        self.own_history.clear()
        self.opponent_history.clear()
        self.last_observation_summary = None
        self.last_turn = -1
        self.last_plan_turn = -1
        self.decision_index = 0
        self.telemetry.clear()
        self.hidden_hypotheses.clear()
        self.exception_fallback_count = 0
        self.exception_types.clear()
        self.tactical_teacher_deference_count = 0

    def history_items(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for owner, counts in (("own", self.own_history), ("opponent", self.opponent_history)):
            for category in HISTORY_CATEGORIES:
                result[f"{owner}_main_action_{category}_count"] = float(counts[category])
        return result

    def observe_selection(self, observation: Mapping[str, Any], selection: Sequence[int]) -> None:
        select = observation.get("select")
        if not isinstance(select, Mapping) or _safe_int(select.get("context"), -1) != int(SelectContext.MAIN):
            return
        options = _sequence(select.get("option"))
        for index in selection:
            if 0 <= int(index) < len(options) and isinstance(options[int(index)], Mapping):
                option_type = _safe_int(options[int(index)].get("type"), -1)
                self.own_history[_TYPE_TO_HISTORY.get(option_type, "other")] += 1

    def infer_opponent_history(self, observation: Mapping[str, Any]) -> None:
        state = _state(observation)
        turn = _safe_int(state.get("turn"), 0)
        if self.last_turn >= 0 and turn < self.last_turn:
            self.reset()
        self.last_turn = turn
        opponent = _player(observation, 1 - _your_index(observation))
        board = _in_play(opponent)
        summary = {
            "board": float(len(board)),
            "energy": float(sum(len(_sequence(card.get("energies"))) for card in board)),
            "discard": float(len(_sequence(opponent.get("discard")))),
            "prizes": float(len(_sequence(opponent.get("prize")))),
            "active": float(_card_id(board[0]) or -1) if board else -1.0,
        }
        previous = self.last_observation_summary
        if previous is not None:
            if summary["board"] > previous["board"]:
                self.opponent_history["play"] += int(summary["board"] - previous["board"])
            if summary["energy"] > previous["energy"]:
                self.opponent_history["attach"] += int(summary["energy"] - previous["energy"])
            if summary["prizes"] < previous["prizes"]:
                self.opponent_history["attack"] += 1
            if summary["active"] != previous["active"] and previous["active"] >= 0:
                self.opponent_history["retreat"] += 1
            if summary["discard"] > previous["discard"] and summary["board"] <= previous["board"]:
                self.opponent_history["play"] += 1
        self.last_observation_summary = summary


def planner_feature_items(
    observation: Mapping[str, Any],
    memory: PlannerMemory,
) -> dict[str, float]:
    state = _state(observation)
    actor = _your_index(observation)
    own_raw = _player(observation, actor)
    opponent_raw = _player(observation, 1 - actor)
    own = _public_player_view(own_raw)
    opponent = _public_player_view(opponent_raw)
    own_board = own["active"] + own["bench"]
    opponent_board = opponent["active"] + opponent["bench"]
    own_cards = _public_card_ids(own_raw)
    opponent_cards = _public_card_ids(opponent_raw)
    stadium_ids = tuple(
        card_id
        for card_id in (_card_id_from_any(item) for item in _sequence(state.get("stadium")))
        if card_id is not None
    )
    archetype_entropy = _entropy(memory.archetype_posterior)
    effective_sample = 1.0 / max(sum(value * value for value in memory.archetype_posterior), 1e-12)
    public: dict[str, float] = {
        "turn": float(_safe_int(state.get("turn"), 0)),
        "turn_action_count": float(_safe_int(state.get("turnActionCount"), 0)),
        "own_board_count": float(len(own_board)),
        "own_board_hp": float(sum(_safe_int(item.get("hp"), 0) for item in own_board)),
        "own_hand_count": float(own["hand_count"]),
        "own_prize_count": float(own["prize_count"]),
        "own_deck_count": float(own["deck_count"]),
        "opponent_board_count": float(len(opponent_board)),
        "opponent_board_hp": float(sum(_safe_int(item.get("hp"), 0) for item in opponent_board)),
        "opponent_hand_count": float(opponent["hand_count"]),
        "opponent_prize_count": float(opponent["prize_count"]),
        "opponent_deck_count": float(opponent["deck_count"]),
        "opponent_public_revealed_copies": float(len(opponent_cards)),
        "opponent_range_entropy": archetype_entropy,
        "opponent_range_effective_sample_size": effective_sample,
    }
    for prefix, cards in (
        ("own_public_card_bucket", own_cards),
        ("opponent_public_card_bucket", opponent_cards),
        ("stadium_card_bucket", stadium_ids),
    ):
        counts = Counter(_card_bucket(card_id) for card_id in cards)
        for index in range(32):
            public[f"{prefix}_{index:02d}"] = float(counts[index])

    select = observation.get("select")
    select = select if isinstance(select, Mapping) else {}
    own_hand = tuple(
        card_id
        for card_id in (_card_id_from_any(item) for item in _sequence(own_raw.get("hand")))
        if card_id is not None
    )
    visible = tuple(
        card_id
        for card_id in (
            _card_id_from_any(item)
            for item in tuple(_sequence(select.get("deck"))) + tuple(_sequence(state.get("looking")))
        )
        if card_id is not None
    )
    effect = select.get("effect")
    context_card = select.get("contextCard")
    private: dict[str, float] = {
        "selection_type_code": _stable_code(str(select.get("type", "Unknown"))),
        "selection_context_code": _stable_code(str(select.get("context", "Unknown"))),
        "selection_min_count": float(_safe_int(select.get("minCount"), 0)),
        "selection_max_count": float(_safe_int(select.get("maxCount"), 0)),
        "selection_option_count": float(len(_sequence(select.get("option")))),
        "remain_damage_counter": float(_safe_int(select.get("remainDamageCounter"), 0)),
        "remain_energy_cost": float(_safe_int(select.get("remainEnergyCost"), 0)),
        "own_hand_count": float(len(own_hand)),
        "private_event_identity_count": 0.0,
        "effect_card_code": 0.0 if _card_id_from_any(effect) is None else _stable_code(f"card:{_card_id_from_any(effect)}"),
        "context_card_code": 0.0 if _card_id_from_any(context_card) is None else _stable_code(f"card:{_card_id_from_any(context_card)}"),
    }
    for prefix, cards in (("own_hand_card_bucket", own_hand), ("visible_search_card_bucket", visible)):
        counts = Counter(_card_bucket(card_id) for card_id in cards)
        for index in range(32):
            private[f"{prefix}_{index:02d}"] = float(counts[index])
    return {**public, **private, **memory.history_items()}


def token_feature_rows(observation: Mapping[str, Any]) -> list[dict[str, float]]:
    actor = _your_index(observation)
    rows: list[dict[str, float]] = []
    for owner_index in (actor, 1 - actor):
        player = _player(observation, owner_index)
        statuses = sum(
            bool(player.get(name, False))
            for name in ("poisoned", "burned", "asleep", "paralyzed", "confused")
        )
        for zone_name, pokemon_values in (("active", _sequence(player.get("active"))), ("bench", _sequence(player.get("bench")))):
            for position, raw in enumerate(pokemon_values):
                if not isinstance(raw, Mapping):
                    continue
                card_id = _card_id(raw)
                if card_id is None:
                    continue
                energies = tuple(_card_id_from_any(item) for item in _sequence(raw.get("energyCards")))
                energy_types = tuple(_safe_int(item, -1) for item in _sequence(raw.get("energies")))
                hp = _safe_int(raw.get("hp"), 0)
                max_hp = max(_safe_int(raw.get("maxHp"), hp), hp, 1)
                angle = 2.0 * math.pi * _card_bucket(card_id) / 32.0
                rows.append(
                    {
                        "owner_self": float(owner_index == actor),
                        "owner_opponent": float(owner_index != actor),
                        "active": float(zone_name == "active"),
                        "bench": float(zone_name == "bench"),
                        "position": float(position) / 5.0,
                        "card_code": _stable_code(f"card:{card_id}"),
                        "card_bucket_sin": math.sin(angle),
                        "card_bucket_cos": math.cos(angle),
                        "hp": hp / 350.0,
                        "max_hp": max_hp / 350.0,
                        "damage": max(0, max_hp - hp) / 350.0,
                        "energy_count": len(energies or energy_types) / 5.0,
                        "grass_energy": float(1 in energies or 1 in energy_types),
                        "psychic_energy": float(5 in energies or 5 in energy_types),
                        "rocket_energy": float(15 in energies),
                        "other_energy": float(any(value not in (None, 1, 5, 15) for value in energies)),
                        "tool_count": len(_sequence(raw.get("tools"))) / 2.0,
                        "evolution_depth": len(_sequence(raw.get("preEvolution"))) / 2.0,
                        "appeared_this_turn": float(bool(raw.get("appearThisTurn", False))),
                        "status_count": float(statuses) / 5.0,
                        **{
                            f"card_bucket_{index:02d}": float(index == _card_bucket(card_id))
                            for index in range(32)
                        },
                    }
                )
    if not rows:
        rows.append({name: 0.0 for name in TOKEN_FEATURE_NAMES})
    return rows


def _numeric(option: Mapping[str, Any], name: str) -> float:
    value = option.get(name)
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _stable_code(value)
    return numeric if math.isfinite(numeric) else 0.0


def action_feature_rows(
    observation: Mapping[str, Any],
    baseline_scores: Sequence[float],
    plan_probabilities: Sequence[float],
    strategic_weights: Sequence[float],
    *,
    prefix: Sequence[int],
    remaining: Sequence[int],
    include_stop: bool,
) -> tuple[list[int | None], list[dict[str, float]]]:
    select = observation.get("select")
    select = select if isinstance(select, Mapping) else {}
    options = _sequence(select.get("option"))
    ordered = sorted(range(len(baseline_scores)), key=lambda index: (-float(baseline_scores[index]), index))
    ranks = {index: rank for rank, index in enumerate(ordered)}
    maximum = max((float(value) for value in baseline_scores), default=0.0)
    indices: list[int | None] = list(remaining)
    if include_stop:
        indices.append(None)
    rows: list[dict[str, float]] = []
    for index in indices:
        stop = index is None
        option: Mapping[str, Any] = {} if stop else options[int(index)]
        option_type = -1 if stop else _safe_int(option.get("type"), -1)
        card_id = None if stop else option_card_id(observation, option)
        target_id = None if stop else option_target_pokemon_id(observation, option)
        row = {name: 0.0 for name in ACTION_FEATURE_NAMES}
        if 0 <= option_type < 16:
            row[f"option_type_{option_type:02d}"] = 1.0
        if card_id is not None:
            row[f"option_card_bucket_{_card_bucket(card_id):02d}"] = 1.0
        if option.get("attackId") is not None:
            attack_bucket = int.from_bytes(
                sha256(f"official-attack:{option.get('attackId')}".encode("utf-8")).digest()[:4],
                "big",
            ) % 16
            row[f"option_attack_bucket_{attack_bucket:02d}"] = 1.0
        if target_id is not None:
            row[f"option_target_bucket_{_card_bucket(target_id):02d}"] = 1.0
        for plan_index, name in enumerate(PLAN_NAMES):
            row[f"plan_alignment_{name}"] = float(plan_probabilities[plan_index]) * _PLAN_TYPE_BIASES[name].get(option_type, 0.0)
        for axis_index, name in enumerate(STRATEGIC_AXES):
            row[f"axis_alignment_{name}"] = float(strategic_weights[axis_index]) * _AXIS_TYPE_BIASES[name].get(option_type, 0.0)
        score = 0.0 if stop else float(baseline_scores[int(index)])
        row.update(
            {
                "stop_token": float(stop),
                "option_number": _numeric(option, "number") / 10.0,
                "option_area": _numeric(option, "area") / 12.0,
                "option_index": _numeric(option, "index") / 60.0,
                "option_player_index": _numeric(option, "playerIndex"),
                "option_count": _numeric(option, "count") / 10.0,
                "option_in_play_area": _numeric(option, "inPlayArea") / 12.0,
                "option_in_play_index": _numeric(option, "inPlayIndex") / 6.0,
                "option_card_code": 0.0 if card_id is None else _stable_code(f"card:{card_id}"),
                "option_attack_code": 0.0 if option.get("attackId") is None else _stable_code(f"attack:{option.get('attackId')}"),
                "option_target_code": 0.0 if target_id is None else _stable_code(f"card:{target_id}"),
                "baseline_score": math.tanh(score / 220.0),
                "baseline_rank": 1.0 if stop else 1.0 - ranks[int(index)] / max(1, len(options) - 1),
                "baseline_gap": 0.0 if stop else math.tanh((score - maximum) / 220.0),
                "prefix_length": len(prefix) / max(1.0, float(_safe_int(select.get("maxCount"), 1))),
                "remaining_fraction": len(remaining) / max(1.0, float(len(options))),
            }
        )
        rows.append(row)
    return indices, rows


def prefix_feature_items(observation: Mapping[str, Any], prefix: Sequence[int]) -> dict[str, float]:
    select = observation.get("select")
    options = _sequence(select.get("option")) if isinstance(select, Mapping) else ()
    counts = Counter()
    codes: list[float] = []
    for index in prefix:
        if 0 <= int(index) < len(options) and isinstance(options[int(index)], Mapping):
            option = options[int(index)]
            option_type = _safe_int(option.get("type"), -1)
            if 0 <= option_type < 16:
                counts[option_type] += 1
            card_id = option_card_id(observation, option)
            if card_id is not None:
                codes.append(_stable_code(f"card:{card_id}"))
    result = {f"prefix_type_{index:02d}": float(counts[index]) for index in range(16)}
    result["prefix_length"] = float(len(prefix))
    result["prefix_card_code_mean"] = sum(codes) / len(codes) if codes else 0.0
    return result


@dataclass(frozen=True, slots=True)
class PlannerEvaluation:
    archetype_probabilities: tuple[float, ...]
    intent_probabilities: tuple[float, ...]
    resource_probabilities: tuple[float, ...]
    blueprint_plan_probabilities: tuple[float, ...]
    final_plan_probabilities: tuple[float, ...]
    plan_values: tuple[float, ...]
    strategic_values: tuple[float, ...]
    strategic_weights: tuple[float, ...]
    aggregate_value: float
    uncertainty: float
    search_gate_probability: float
    safe_resolve_attempted: bool
    safe_resolve_accepted: bool
    bounds_admitted: bool

    @property
    def selected_plan_index(self) -> int:
        return min(range(len(self.final_plan_probabilities)), key=lambda index: (-self.final_plan_probabilities[index], index))


@dataclass(frozen=True, slots=True)
class IntegratedPlannerConfig:
    """Controlled ablations used by the final held-out evaluation harness."""

    ablation: str = "full"

    def __post_init__(self) -> None:
        if self.ablation not in {"full", "no_beliefs", "no_plan", "no_relational", "no_safe_resolve", "no_meta"}:
            raise ValueError(f"unsupported integrated-planner ablation: {self.ablation}")


class FrozenStrategicPlanner:
    def __init__(self, artifact: Mapping[str, Any]) -> None:
        if int(artifact.get("schema_version", -1)) != INTEGRATED_PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported integrated planner artifact")
        if tuple(artifact.get("plan_names", ())) != PLAN_NAMES:
            raise ValueError("integrated planner plan vocabulary mismatch")
        if tuple(artifact.get("strategic_axes", ())) != STRATEGIC_AXES:
            raise ValueError("integrated planner strategic-axis mismatch")
        self.artifact = dict(artifact)
        supplied_fingerprint = str(self.artifact.pop("fingerprint", ""))
        expected = _fingerprint(self.artifact)
        self.artifact["fingerprint"] = supplied_fingerprint
        if supplied_fingerprint != expected:
            raise ValueError("integrated planner artifact fingerprint mismatch")
        self.layers = self.artifact["layers"]
        self.normalizers = self.artifact["normalizers"]
        self.calibration = self.artifact["calibration"]

    @property
    def fingerprint(self) -> str:
        return str(self.artifact["fingerprint"])

    def _planner_hidden(
        self,
        features: Mapping[str, float],
        previous_plan: Sequence[float],
    ) -> list[float]:
        values = _normalise(features, self.normalizers["planner"]) + [float(value) for value in previous_plan]
        hidden = _tanh_dense(values, self.layers["planner_hidden_1"])
        return _tanh_dense(hidden, self.layers["planner_hidden_2"])

    def evaluate(
        self,
        features: Mapping[str, float],
        previous_plan: Sequence[float],
        *,
        allow_safe_resolve: bool,
    ) -> PlannerEvaluation:
        hidden = self._planner_hidden(features, previous_plan)
        archetype_logits = _dense(hidden, self.layers["archetype_head"])
        intent_logits = _dense(hidden, self.layers["intent_head"])
        resource_logits = _dense(hidden, self.layers["resource_head"])
        archetype = _softmax(archetype_logits, float(self.calibration["archetype_temperature"]))
        intent = _softmax(intent_logits, float(self.calibration["intent_temperature"]))
        resources = tuple(_sigmoid(value) for value in resource_logits)
        plan_logits = _dense(hidden, self.layers["plan_policy_head"])
        if "belief_plan_bridge" in self.layers:
            bridge = _dense(
                list(_softmax(archetype_logits))
                + list(_softmax(intent_logits))
                + list(resources),
                self.layers["belief_plan_bridge"],
            )
            plan_logits = [base + residual for base, residual in zip(plan_logits, bridge, strict=True)]
        blueprint = _softmax(plan_logits, float(self.calibration["plan_temperature"]))
        plan_values = tuple(math.tanh(value) for value in _dense(hidden, self.layers["plan_value_head"]))
        strategic = tuple(math.tanh(value) for value in _dense(hidden, self.layers["strategic_head"]))
        meta_weights = _softmax(_dense(hidden, self.layers["meta_head"]))
        aggregate = sum(value * weight for value, weight in zip(strategic, meta_weights, strict=True))
        uncertainty = max(0.0, float(_dense(hidden, self.layers["uncertainty_head"])[0]))
        original_gate = _sigmoid(float(_dense(hidden, self.layers["search_gate_head"])[0]))
        gate = original_gate
        if "search_gate_direct_head" in self.layers and "search_gate" in self.normalizers:
            gate_features = search_gate_feature_items(
                features,
                original_gate_probability=original_gate,
                uncertainty=uncertainty,
                archetype_probabilities=archetype,
                intent_probabilities=intent,
                plan_probabilities=blueprint,
                plan_values=plan_values,
                strategic_values=strategic,
            )
            gate = _sigmoid(
                float(
                    _dense(
                        _normalise(gate_features, self.normalizers["search_gate"]),
                        self.layers["search_gate_direct_head"],
                    )[0]
                )
            )
        elif "search_gate_feature_hidden" in self.layers and "search_gate" in self.normalizers:
            gate_features = search_gate_feature_items(
                features,
                original_gate_probability=original_gate,
                uncertainty=uncertainty,
                archetype_probabilities=archetype,
                intent_probabilities=intent,
                plan_probabilities=blueprint,
                plan_values=plan_values,
                strategic_values=strategic,
            )
            gate_hidden = _tanh_dense(
                _normalise(gate_features, self.normalizers["search_gate"]),
                self.layers["search_gate_feature_hidden"],
            )
            gate = _sigmoid(float(_dense(gate_hidden, self.layers["search_gate_feature_head"])[0]))
        final = blueprint
        attempted = bool(allow_safe_resolve and gate >= float(self.calibration["search_gate_threshold"]))
        accepted = False
        admitted = bool(self.calibration.get("counterfactual_bounds_admitted", False))
        if attempted and admitted:
            radius = float(self.calibration["counterfactual_advantage_radius"])
            tolerance = float(self.calibration.get("safe_non_regression_tolerance", 0.0))
            baseline_value = sum(probability * value for probability, value in zip(blueprint, plan_values, strict=True))
            candidates = [
                index
                for index, value in enumerate(plan_values)
                if (value - baseline_value) - radius >= -tolerance
                and (value - baseline_value) >= float(self.calibration.get("safe_improvement_margin", 0.0))
            ]
            if candidates:
                selected = min(candidates, key=lambda index: (-plan_values[index], index))
                safe_mass = min(1.0, max(0.0, float(self.calibration.get("maximum_resolve_mass", 0.65))))
                materialized = [(1.0 - safe_mass) * value for value in blueprint]
                materialized[selected] += safe_mass
                final = tuple(materialized)
                accepted = selected != min(range(len(blueprint)), key=lambda index: (-blueprint[index], index))
        return PlannerEvaluation(
            archetype_probabilities=archetype,
            intent_probabilities=intent,
            resource_probabilities=resources,
            blueprint_plan_probabilities=blueprint,
            final_plan_probabilities=final,
            plan_values=plan_values,
            strategic_values=strategic,
            strategic_weights=meta_weights,
            aggregate_value=aggregate,
            uncertainty=uncertainty,
            search_gate_probability=gate,
            safe_resolve_attempted=attempted,
            safe_resolve_accepted=accepted,
            bounds_admitted=admitted,
        )

    def recondition_plan_for_beliefs(
        self,
        evaluation: PlannerEvaluation,
        archetype: Sequence[float],
        intent: Sequence[float],
        resources: Sequence[float],
    ) -> tuple[float, ...]:
        if "belief_plan_bridge" not in self.layers:
            return evaluation.blueprint_plan_probabilities
        old_bridge = _dense(
            list(evaluation.archetype_probabilities)
            + list(evaluation.intent_probabilities)
            + list(evaluation.resource_probabilities),
            self.layers["belief_plan_bridge"],
        )
        new_bridge = _dense(
            list(archetype) + list(intent) + list(resources),
            self.layers["belief_plan_bridge"],
        )
        temperature = max(float(self.calibration["plan_temperature"]), 1e-6)
        adjusted_logits = [
            math.log(max(probability, 1e-12)) + (new - old) / temperature
            for probability, new, old in zip(
                evaluation.blueprint_plan_probabilities, new_bridge, old_bridge, strict=True
            )
        ]
        return _softmax(adjusted_logits)

    def _token_pool(self, token_rows: Sequence[Mapping[str, float]], state_hidden: Sequence[float]) -> list[float]:
        token_normalizer = self.normalizers["token"]
        token_hidden = [
            _tanh_dense(_normalise(row, token_normalizer), self.layers["token_hidden"])
            for row in token_rows
        ]
        queries = [_dense(row, self.layers["token_query"]) for row in token_hidden]
        keys = [_dense(row, self.layers["token_key"]) for row in token_hidden]
        values = [_dense(row, self.layers["token_value"]) for row in token_hidden]
        contexts: list[list[float]] = []
        dimension = max(1, len(queries[0]))
        for query in queries:
            weights = _softmax([
                sum(left * right for left, right in zip(query, key, strict=True)) / math.sqrt(dimension)
                for key in keys
            ])
            contexts.append([
                sum(weight * value[index] for weight, value in zip(weights, values, strict=True))
                for index in range(len(values[0]))
            ])
        pooled = [sum(row[index] for row in contexts) / len(contexts) for index in range(len(contexts[0]))]
        state_query = _dense(state_hidden, self.layers["token_pool_query"])
        attention = _softmax([
            sum(left * right for left, right in zip(state_query, key, strict=True)) / math.sqrt(max(1, len(key)))
            for key in keys
        ])
        attended = [
            sum(weight * value[index] for weight, value in zip(attention, values, strict=True))
            for index in range(len(values[0]))
        ]
        return [math.tanh(value) for value in _dense(pooled + attended, self.layers["token_output"])]

    def action_scores(
        self,
        planner_features: Mapping[str, float],
        token_rows: Sequence[Mapping[str, float]],
        action_rows: Sequence[Mapping[str, float]],
        prefix_features: Mapping[str, float],
        evaluation: PlannerEvaluation,
    ) -> tuple[float, ...]:
        state_inputs = (
            _normalise(planner_features, self.normalizers["planner"])
            + list(evaluation.archetype_probabilities)
            + list(evaluation.intent_probabilities)
            + list(evaluation.resource_probabilities)
            + list(evaluation.final_plan_probabilities)
            + list(evaluation.strategic_values)
            + list(evaluation.strategic_weights)
            + [evaluation.aggregate_value, evaluation.uncertainty]
        )
        state_hidden = _tanh_dense(state_inputs, self.layers["action_state_hidden"])
        token_pool = self._token_pool(token_rows, state_hidden)
        prefix_values = [float(prefix_features.get(name, 0.0)) for name in PREFIX_FEATURE_NAMES]
        query = _tanh_dense(state_hidden + token_pool + prefix_values, self.layers["action_query"])
        exploit_query = _tanh_dense(state_hidden + list(evaluation.intent_probabilities), self.layers["exploit_query"])
        exploit_weight = (
            float(self.calibration.get("exploitation_weight", 0.0))
            if bool(self.calibration.get("opponent_exploitation_admitted", False))
            else 0.0
        )
        result = []
        baseline_feature_index = ACTION_FEATURE_NAMES.index("baseline_score")
        plan_feature_indices = [
            ACTION_FEATURE_NAMES.index(f"plan_alignment_{name}") for name in PLAN_NAMES
        ]
        axis_feature_indices = [
            ACTION_FEATURE_NAMES.index(f"axis_alignment_{name}") for name in STRATEGIC_AXES
        ]
        for row in action_rows:
            action_values = _normalise(row, self.normalizers["action"])
            action_hidden = _tanh_dense(action_values, self.layers["action_hidden"])
            base = sum(left * right for left, right in zip(query, action_hidden, strict=True)) / math.sqrt(max(1, len(query)))
            bias = _dense(action_hidden, self.layers["action_bias"])[0]
            exploit = sum(left * right for left, right in zip(exploit_query, action_hidden, strict=True)) / math.sqrt(max(1, len(exploit_query)))
            tactical_prior = float(self.calibration.get("tactical_prior_weight", 0.0)) * action_values[baseline_feature_index]
            plan_alignment = float(self.calibration.get("plan_alignment_weight", 0.0)) * sum(
                action_values[index] for index in plan_feature_indices
            )
            axis_alignment = float(self.calibration.get("axis_alignment_weight", 0.0)) * sum(
                action_values[index] for index in axis_feature_indices
            )
            result.append(
                base + bias + exploit_weight * exploit + tactical_prior + plan_alignment + axis_alignment
            )
        return tuple(result)


class IntegratedBeliefPlannerAgent:
    """Official-engine controller whose final policy is the frozen neural blueprint."""

    def __init__(
        self,
        deck: DeckResolution,
        model_path: str | Path,
        *,
        telemetry_limit: int = 128,
        planner_config: IntegratedPlannerConfig | None = None,
        tactical_teacher: Any | None = None,
    ) -> None:
        if not deck.playable:
            raise ValueError("integrated planner requires a playable deck")
        self.deck = deck
        self.model_path = Path(model_path)
        artifact = json.loads(self.model_path.read_text(encoding="utf-8"))
        self.planner = FrozenStrategicPlanner(artifact)
        self.memory = PlannerMemory()
        self.telemetry_limit = max(1, int(telemetry_limit))
        self.planner_config = planner_config or IntegratedPlannerConfig()
        self.tactical_teacher = tactical_teacher
        self._tactical_prior = Phase6CompetitionAgent(
            deck,
            config=CompetitionAgentConfig(
                decision_budget_ms=1000.0,
                search_budget_ms=0.0,
            ),
        )

    @property
    def config(self) -> CompetitionAgentConfig:
        return self._tactical_prior.config

    def reset(self) -> None:
        self.memory.reset()
        self._tactical_prior.reset()
        if self.tactical_teacher is not None and hasattr(self.tactical_teacher, "reset"):
            self.tactical_teacher.reset()

    def choose(self, observation: Mapping[str, Any]) -> list[int]:
        select = observation.get("select")
        if select is None:
            self.reset()
            return list(self.deck.card_ids)
        contract = SelectionContract.from_observation(observation)
        try:
            self.memory.infer_opponent_history(observation)
            # Tactical-prior memory receives exactly the visible observation.
            self._tactical_prior._update_memory(observation)
            options = _sequence(select.get("option"))
            teacher_selection: list[int] | None = None
            if self.tactical_teacher is not None:
                try:
                    teacher_raw = (
                        self.tactical_teacher.choose(observation)
                        if hasattr(self.tactical_teacher, "choose")
                        else self.tactical_teacher.act(observation)
                    )
                    teacher_selection = list(contract.validate(teacher_raw))
                except Exception:
                    teacher_selection = None
            baseline_scores = tuple(
                self._tactical_prior._score_option(observation, option, index)
                for index, option in enumerate(options)
            )
            features = planner_feature_items(observation, self.memory)
            is_main = _safe_int(select.get("context"), -1) == int(SelectContext.MAIN)
            allow_resolve = is_main and self.planner_config.ablation != "no_safe_resolve"
            evaluation = self.planner.evaluate(
                features,
                self.memory.plan_latent,
                allow_safe_resolve=allow_resolve,
            )
            behavior_probabilities = behavior_intent_probabilities(features)
            fused_intent = _fuse_intent_evidence(
                evaluation.intent_probabilities,
                behavior_probabilities,
                float(self.planner.calibration.get("behavior_evidence_weight", 0.0)),
            )
            evaluation = replace(evaluation, intent_probabilities=fused_intent)
            if self.planner_config.ablation == "no_beliefs":
                uniform_archetype = tuple(1.0 / len(ARCHETYPE_NAMES) for _ in ARCHETYPE_NAMES)
                uniform_intent = tuple(1.0 / len(INTENT_NAMES) for _ in INTENT_NAMES)
                neutral_resources = tuple(0.5 for _ in RESOURCE_NAMES)
                reconditioned_plan = self.planner.recondition_plan_for_beliefs(
                    evaluation,
                    uniform_archetype,
                    uniform_intent,
                    neutral_resources,
                )
                evaluation = replace(
                    evaluation,
                    archetype_probabilities=uniform_archetype,
                    intent_probabilities=uniform_intent,
                    resource_probabilities=neutral_resources,
                    blueprint_plan_probabilities=reconditioned_plan,
                    final_plan_probabilities=reconditioned_plan,
                    safe_resolve_attempted=False,
                    safe_resolve_accepted=False,
                )
            if self.planner_config.ablation == "no_plan":
                uniform_plan = tuple(1.0 / len(PLAN_NAMES) for _ in PLAN_NAMES)
                evaluation = replace(
                    evaluation,
                    blueprint_plan_probabilities=uniform_plan,
                    final_plan_probabilities=uniform_plan,
                    safe_resolve_attempted=False,
                    safe_resolve_accepted=False,
                )
            if self.planner_config.ablation == "no_meta":
                uniform_meta = tuple(1.0 / len(STRATEGIC_AXES) for _ in STRATEGIC_AXES)
                evaluation = replace(
                    evaluation,
                    strategic_weights=uniform_meta,
                    aggregate_value=sum(evaluation.strategic_values) / len(evaluation.strategic_values),
                )
            self.memory.archetype_posterior = list(evaluation.archetype_probabilities)
            self.memory.intent_posterior = list(evaluation.intent_probabilities)
            self.memory.resource_posterior = list(evaluation.resource_probabilities)
            turn = _safe_int(_state(observation).get("turn"), 0)
            if is_main and turn != self.memory.last_plan_turn:
                persistence = float(self.planner.calibration.get("plan_persistence", 0.35))
                self.memory.plan_latent = [
                    persistence * old + (1.0 - persistence) * new
                    for old, new in zip(self.memory.plan_latent, evaluation.final_plan_probabilities, strict=True)
                ]
                total = sum(self.memory.plan_latent)
                self.memory.plan_latent = [value / total for value in self.memory.plan_latent]
                self.memory.last_plan_turn = turn
            selected: list[int] = []
            prefix_confidences: list[float] = []
            remaining = list(range(contract.option_count))
            tokens = token_feature_rows(observation)
            if self.planner_config.ablation == "no_relational":
                token_normalizer = self.planner.normalizers["token"]
                neutral = dict(zip(token_normalizer["names"], token_normalizer["means"], strict=True))
                tokens = [neutral]
            while len(selected) < contract.max_count:
                include_stop = len(selected) >= contract.min_count
                indices, rows = action_feature_rows(
                    observation,
                    baseline_scores,
                    evaluation.final_plan_probabilities,
                    evaluation.strategic_weights,
                    prefix=selected,
                    remaining=remaining,
                    include_stop=include_stop,
                )
                scores = self.planner.action_scores(
                    features,
                    tokens,
                    rows,
                    prefix_feature_items(observation, selected),
                    evaluation,
                )
                probabilities = _softmax(scores)
                prefix_confidences.append(max(probabilities))
                chosen_position = min(range(len(scores)), key=lambda index: (-scores[index], index))
                chosen = indices[chosen_position]
                if chosen is None:
                    break
                selected.append(int(chosen))
                remaining.remove(int(chosen))
                if not remaining:
                    break
            neural_result = list(contract.validate(selected))
            confidence = min(prefix_confidences, default=1.0)
            deference_threshold = float(
                self.planner.calibration.get("tactical_teacher_deference_threshold", -1.0)
            )
            deferred = bool(
                teacher_selection is not None
                and bool(self.planner.calibration.get("tactical_teacher_deference_admitted", False))
                and confidence < deference_threshold
            )
            result = list(teacher_selection) if deferred else neural_result
            self.memory.tactical_teacher_deference_count += int(deferred)
            self.memory.observe_selection(observation, result)
            self.memory.hidden_hypotheses = list(hidden_resource_hypotheses(evaluation))
            self.memory.decision_index += 1
            objective_index = max(range(len(evaluation.strategic_weights)), key=lambda index: evaluation.strategic_weights[index])
            record = {
                "decision_index": self.memory.decision_index,
                "turn": turn,
                "context": _safe_int(select.get("context"), -1),
                "selection": result,
                "neural_selection": neural_result,
                "tactical_policy_minimum_prefix_confidence": confidence,
                "tactical_teacher_available": teacher_selection is not None,
                "tactical_teacher_deferred": deferred,
                "plan": PLAN_NAMES[evaluation.selected_plan_index],
                "persistent_plan": list(self.memory.plan_latent),
                "strategic_objective": STRATEGIC_AXES[objective_index],
                "strategic_values": dict(zip(STRATEGIC_AXES, evaluation.strategic_values, strict=True)),
                "meta_weights": dict(zip(STRATEGIC_AXES, evaluation.strategic_weights, strict=True)),
                "aggregate_value": evaluation.aggregate_value,
                "archetype_range": dict(zip(ARCHETYPE_NAMES, evaluation.archetype_probabilities, strict=True)),
                "intent_range": dict(zip(INTENT_NAMES, evaluation.intent_probabilities, strict=True)),
                "behavior_intent_likelihood": dict(zip(INTENT_NAMES, behavior_probabilities, strict=True)),
                "resource_range": dict(zip(RESOURCE_NAMES, evaluation.resource_probabilities, strict=True)),
                "hidden_resource_hypotheses": list(self.memory.hidden_hypotheses),
                "search_gate_probability": evaluation.search_gate_probability,
                "search_trigger_reasons": [
                    reason
                    for reason, active in (
                        ("low_policy_confidence", max(evaluation.final_plan_probabilities) < 0.45),
                        ("strategic_head_disagreement", statistics.pstdev(evaluation.strategic_values) > 0.30),
                        ("opponent_belief_entropy", _entropy(evaluation.intent_probabilities) > 1.45),
                        ("prize_threshold", min(len(_sequence(_player(observation, 0).get("prize"))), len(_sequence(_player(observation, 1).get("prize")))) <= 2),
                        ("close_plan_values", sorted(evaluation.plan_values, reverse=True)[0] - sorted(evaluation.plan_values, reverse=True)[1] < 0.12),
                    )
                    if active
                ],
                "safe_resolve_attempted": evaluation.safe_resolve_attempted,
                "safe_resolve_accepted": evaluation.safe_resolve_accepted,
                "bounds_admitted": evaluation.bounds_admitted,
                "model_fingerprint": self.planner.fingerprint,
            }
            self.memory.telemetry.append(record)
            if len(self.memory.telemetry) > self.telemetry_limit:
                del self.memory.telemetry[: len(self.memory.telemetry) - self.telemetry_limit]
            return result
        except Exception as exc:
            # The legality-preserving fallback is deliberately narrow.  It does
            # not silently relabel the heuristic as the deployed strategy.
            self.memory.exception_fallback_count += 1
            self.memory.exception_types[type(exc).__name__] += 1
            if self.tactical_teacher is not None:
                try:
                    teacher_raw = (
                        self.tactical_teacher.choose(observation)
                        if hasattr(self.tactical_teacher, "choose")
                        else self.tactical_teacher.act(observation)
                    )
                    return list(contract.validate(teacher_raw))
                except Exception:
                    pass
            try:
                return list(contract.validate(self._tactical_prior.choose(observation)))
            except Exception:
                return list(contract.fallback())

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": INTEGRATED_PLANNER_SCHEMA_VERSION,
            "model_fingerprint": self.planner.fingerprint,
            "decision_count": self.memory.decision_index,
            "persistent_plan": dict(zip(PLAN_NAMES, self.memory.plan_latent, strict=True)),
            "archetype_range": dict(zip(ARCHETYPE_NAMES, self.memory.archetype_posterior, strict=True)),
            "intent_range": dict(zip(INTENT_NAMES, self.memory.intent_posterior, strict=True)),
            "resource_range": dict(zip(RESOURCE_NAMES, self.memory.resource_posterior, strict=True)),
            "hidden_resource_hypotheses": list(self.memory.hidden_hypotheses),
            "exception_fallback_count": self.memory.exception_fallback_count,
            "exception_types": dict(self.memory.exception_types),
            "tactical_teacher_deference_count": self.memory.tactical_teacher_deference_count,
            "tactical_teacher_deference_admitted": bool(
                self.planner.calibration.get("tactical_teacher_deference_admitted", False)
            ),
            "telemetry": list(self.memory.telemetry),
            "ablation": self.planner_config.ablation,
            "strategy_provider": "frozen_public_belief_neural_blueprint",
            "heuristic_final_controller": False,
            "sampled_rollout_final_controller": False,
        }


__all__ = [
    "ACTION_FEATURE_NAMES",
    "ARCHETYPE_NAMES",
    "HISTORY_FEATURE_NAMES",
    "INTEGRATED_PLANNER_SCHEMA_VERSION",
    "INTENT_NAMES",
    "IntegratedPlannerConfig",
    "IntegratedBeliefPlannerAgent",
    "PLAN_NAMES",
    "PLANNER_FEATURE_NAMES",
    "PREFIX_FEATURE_NAMES",
    "PlannerEvaluation",
    "PlannerMemory",
    "RESOURCE_NAMES",
    "SEARCH_GATE_FEATURE_NAMES",
    "STRATEGIC_AXES",
    "TOKEN_FEATURE_NAMES",
    "action_feature_rows",
    "behavior_intent_probabilities",
    "hidden_resource_hypotheses",
    "planner_feature_items",
    "prefix_feature_items",
    "search_gate_feature_items",
    "token_feature_rows",
]
