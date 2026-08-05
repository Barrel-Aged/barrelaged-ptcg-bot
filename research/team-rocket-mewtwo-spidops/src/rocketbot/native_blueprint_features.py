"""Leakage-safe semantic features for the Phase 6 native blueprint path.

The native search state contains an opaque hidden-completion truth used by the
simulator.  This module never accepts that truth.  It derives policy inputs
only from the acting player's observation, the public belief object, and the
complete legal-action list.  Card identities are represented through the same
fixed hash buckets used by the Phase 6H/6I trajectory bootstrap; raw card IDs
are never returned or persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from .game_theory import (
    AbstractAction,
    InformationSet,
    OpponentRange,
    PublicBeliefState,
)
from .official_bridge import (
    official_private_observation_payload,
    sanitize_official_public_state,
)


NATIVE_BLUEPRINT_FEATURE_SCHEMA_VERSION = 1
NATIVE_CARD_HASH_BUCKETS = 32


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def stable_type_code(value: Any) -> float:
    digest = sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / float(2**32 - 1)


def card_bucket(card_id: int) -> int:
    digest = sha256(f"official-card:{int(card_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % NATIVE_CARD_HASH_BUCKETS


def bucketed_card_features(
    prefix: str,
    card_ids: Sequence[int | None],
) -> tuple[tuple[str, float], ...]:
    counts = [0] * NATIVE_CARD_HASH_BUCKETS
    for raw in card_ids:
        if raw is None:
            continue
        card_id = _integer(raw, -1)
        if card_id >= 0:
            counts[card_bucket(card_id)] += 1
    return tuple(
        (f"{prefix}_{index:02d}", float(count))
        for index, count in enumerate(counts)
    )


def _card_code(value: Any) -> float:
    if value is None:
        return 0.0
    card_id = _integer(value, -1)
    return 0.0 if card_id < 0 else stable_type_code(f"card:{card_id}")


def _option_numeric(option: Mapping[str, Any], name: str) -> float:
    value = option.get(name)
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return stable_type_code(value)


def native_action_feature_items(
    observation: Mapping[str, Any],
    selected_indices: Sequence[int],
) -> tuple[tuple[str, float], ...]:
    """Return the Phase 6I-compatible semantic encoding of one legal action."""

    private = official_private_observation_payload(observation)
    selection = private.get("selection", {})
    options = list(_sequence(selection.get("options"))) if isinstance(selection, Mapping) else []
    indices = tuple(_integer(value, -1) for value in selected_indices)
    if any(index < 0 or index >= len(options) for index in indices):
        raise ValueError("native action index is outside the visible option list")
    selected = tuple(options[index] for index in indices)
    if any(not isinstance(option, Mapping) for option in selected):
        raise ValueError("native action option must be a mapping")
    features = (
        ("selection_count", float(len(indices))),
        (
            "option_type_code_sum",
            float(sum(stable_type_code(option.get("type")) for option in selected)),
        ),
        ("option_number_sum", float(sum(_option_numeric(option, "number") for option in selected))),
        ("option_area_sum", float(sum(_option_numeric(option, "area") for option in selected))),
        ("option_index_sum", float(sum(_option_numeric(option, "index") for option in selected))),
        (
            "option_player_index_sum",
            float(sum(_option_numeric(option, "playerIndex") for option in selected)),
        ),
        ("option_count_sum", float(sum(_option_numeric(option, "count") for option in selected))),
        (
            "option_in_play_area_sum",
            float(sum(_option_numeric(option, "inPlayArea") for option in selected)),
        ),
        (
            "option_in_play_index_sum",
            float(sum(_option_numeric(option, "inPlayIndex") for option in selected)),
        ),
        (
            "option_card_code_sum",
            float(sum(_card_code(option.get("cardId")) for option in selected)),
        ),
        (
            "option_attack_code_sum",
            float(
                sum(
                    0.0
                    if option.get("attackId") is None
                    else stable_type_code(f"attack:{option['attackId']}")
                    for option in selected
                )
            ),
        ),
    )
    return tuple(sorted(features))


def _public_player_card_ids(player: Mapping[str, Any]) -> tuple[int | None, ...]:
    cards: list[int | None] = list(_sequence(player.get("discard")))
    for pokemon in tuple(_sequence(player.get("active"))) + tuple(
        _sequence(player.get("bench"))
    ):
        if not isinstance(pokemon, Mapping):
            continue
        cards.append(pokemon.get("id"))
        cards.extend(_sequence(pokemon.get("energy_cards")))
        cards.extend(_sequence(pokemon.get("tools")))
        cards.extend(_sequence(pokemon.get("pre_evolution")))
    return tuple(cards)


def native_observer_public_feature_items(
    observation: Mapping[str, Any],
    opponent_range: OpponentRange,
    *,
    observer_player: int,
) -> tuple[tuple[str, float], ...]:
    """Return public-only features oriented to an explicit observer seat.

    Official prompts are emitted for the player currently making a procedural
    selection.  Belief learning often needs the same public position from the
    other player's perspective.  This helper changes only the public feature
    orientation; it never reads that observer's hand, the acting player's hand,
    visualizer truth, or any supplied hidden completion.
    """

    public = sanitize_official_public_state(observation)
    players = list(_sequence(public.get("players")))
    if len(players) != 2 or observer_player not in (0, 1):
        raise ValueError("native public features require two players and a valid observer")
    own = players[observer_player]
    opponent = players[1 - observer_player]
    if not isinstance(own, Mapping) or not isinstance(opponent, Mapping):
        raise ValueError("native public player views must be mappings")
    own_board = tuple(_sequence(own.get("active"))) + tuple(_sequence(own.get("bench")))
    opponent_board = tuple(_sequence(opponent.get("active"))) + tuple(
        _sequence(opponent.get("bench"))
    )
    own_cards = _public_player_card_ids(own)
    opponent_cards = _public_player_card_ids(opponent)
    stadium_cards = tuple(_sequence(public.get("stadium")))
    features = (
        ("turn", float(_integer(public.get("turn")))),
        ("turn_action_count", float(_integer(public.get("turn_action_count")))),
        ("own_board_count", float(len(own_board))),
        (
            "own_board_hp",
            float(
                sum(
                    _integer(item.get("hp"))
                    for item in own_board
                    if isinstance(item, Mapping)
                )
            ),
        ),
        ("own_hand_count", float(_integer(own.get("hand_count")))),
        ("own_prize_count", float(_integer(own.get("prize_count")))),
        ("own_deck_count", float(_integer(own.get("deck_count")))),
        ("opponent_board_count", float(len(opponent_board))),
        (
            "opponent_board_hp",
            float(
                sum(
                    _integer(item.get("hp"))
                    for item in opponent_board
                    if isinstance(item, Mapping)
                )
            ),
        ),
        ("opponent_hand_count", float(_integer(opponent.get("hand_count")))),
        ("opponent_prize_count", float(_integer(opponent.get("prize_count")))),
        ("opponent_deck_count", float(_integer(opponent.get("deck_count")))),
        ("opponent_public_revealed_copies", float(len(opponent_cards))),
        ("opponent_range_entropy", float(opponent_range.entropy)),
        (
            "opponent_range_effective_sample_size",
            float(opponent_range.effective_sample_size),
        ),
    )
    return tuple(
        sorted(
            features
            + bucketed_card_features("own_public_card_bucket", own_cards)
            + bucketed_card_features("opponent_public_card_bucket", opponent_cards)
            + bucketed_card_features("stadium_card_bucket", stadium_cards)
        )
    )


def native_public_feature_items(
    observation: Mapping[str, Any],
    belief: PublicBeliefState,
) -> tuple[tuple[str, float], ...]:
    """Return acting-player public features compatible with Phase 6I."""

    public = sanitize_official_public_state(observation)
    actor = _integer(public.get("acting_player"), -1)
    if actor not in (0, 1):
        raise ValueError("native public features require a valid acting player")
    return native_observer_public_feature_items(
        observation,
        belief.opponent_range,
        observer_player=actor,
    )


def native_private_feature_items(
    observation: Mapping[str, Any],
) -> tuple[tuple[str, float], ...]:
    """Encode only the acting player's agent-visible private observation."""

    private = official_private_observation_payload(observation)
    selection = private.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("native private features require a selection mapping")
    own_hand = tuple(_sequence(private.get("own_hand")))
    visible_search = tuple(_sequence(selection.get("visible_deck"))) + tuple(
        _sequence(private.get("own_looking"))
    )
    features = (
        ("selection_type_code", stable_type_code(str(selection.get("type", "Unknown")))),
        ("selection_context_code", stable_type_code(str(selection.get("context", "Unknown")))),
        ("selection_min_count", float(_integer(selection.get("min_count")))),
        ("selection_max_count", float(_integer(selection.get("max_count")))),
        ("selection_option_count", float(len(_sequence(selection.get("options"))))),
        ("remain_damage_counter", float(_integer(selection.get("remain_damage_counter")))),
        ("remain_energy_cost", float(_integer(selection.get("remain_energy_cost")))),
        ("own_hand_count", float(len(own_hand))),
        # Live observations do not expose the replay event interval here.  Zero
        # is an explicit missing-event baseline, not a hidden-truth lookup.
        ("private_event_identity_count", 0.0),
        ("effect_card_code", _card_code(selection.get("effect_card"))),
        ("context_card_code", _card_code(selection.get("context_card"))),
    )
    return tuple(
        sorted(
            features
            + bucketed_card_features("own_hand_card_bucket", own_hand)
            + bucketed_card_features("visible_search_card_bucket", visible_search)
        )
    )


def _schema_names() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    public_base = (
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
    public = tuple(
        sorted(
            public_base
            + tuple(f"own_public_card_bucket_{index:02d}" for index in range(32))
            + tuple(f"opponent_public_card_bucket_{index:02d}" for index in range(32))
            + tuple(f"stadium_card_bucket_{index:02d}" for index in range(32))
        )
    )
    private_base = (
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
    private = tuple(
        sorted(
            private_base
            + tuple(f"own_hand_card_bucket_{index:02d}" for index in range(32))
            + tuple(f"visible_search_card_bucket_{index:02d}" for index in range(32))
        )
    )
    action = tuple(
        sorted(
            (
                "selection_count",
                "option_type_code_sum",
                "option_number_sum",
                "option_area_sum",
                "option_index_sum",
                "option_player_index_sum",
                "option_count_sum",
                "option_in_play_area_sum",
                "option_in_play_index_sum",
                "option_card_code_sum",
                "option_attack_code_sum",
            )
        )
    )
    return public, private, action


(
    NATIVE_PUBLIC_FEATURE_NAMES,
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_ACTION_FEATURE_NAMES,
) = _schema_names()


NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT = _fingerprint(
    {
        "schema_version": NATIVE_BLUEPRINT_FEATURE_SCHEMA_VERSION,
        "public": NATIVE_PUBLIC_FEATURE_NAMES,
        "private": NATIVE_PRIVATE_FEATURE_NAMES,
        "action": NATIVE_ACTION_FEATURE_NAMES,
        "card_hash_buckets": NATIVE_CARD_HASH_BUCKETS,
        "hidden_completion_truth_input": False,
    }
)


@dataclass(frozen=True, slots=True)
class NativeBlueprintView:
    information_set: InformationSet
    public_belief: PublicBeliefState
    feature_schema_fingerprint: str = NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT

    def __post_init__(self) -> None:
        if self.information_set.public_belief_fingerprint != self.public_belief.fingerprint:
            raise ValueError("native blueprint information set/belief mismatch")
        if tuple(name for name, _ in self.public_belief.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("native blueprint public feature schema mismatch")
        if tuple(name for name, _ in self.information_set.private_features) != NATIVE_PRIVATE_FEATURE_NAMES:
            raise ValueError("native blueprint private feature schema mismatch")
        for action in self.information_set.legal_actions:
            if tuple(name for name, _ in action.features) != NATIVE_ACTION_FEATURE_NAMES:
                raise ValueError("native blueprint action feature schema mismatch")


class NativeBlueprintFeatureBridge:
    """Enrich a Phase 6J state without exposing its opaque completion truth."""

    @property
    def fingerprint(self) -> str:
        return NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT

    def view(self, game: Any, state: Any) -> NativeBlueprintView:
        player = state.player_index
        if player not in (0, 1) or state.terminal:
            raise ValueError("native blueprint view requires an acting-player state")
        raw_belief = game.public_belief(state, player=player)
        public_features = native_public_feature_items(state.observation, raw_belief)
        belief = PublicBeliefState(
            public_state_fingerprint=raw_belief.public_state_fingerprint,
            public_history_fingerprint=raw_belief.public_history_fingerprint,
            turn=raw_belief.turn,
            acting_player=raw_belief.acting_player,
            opponent_range=raw_belief.opponent_range,
            public_features=public_features,
            updater_fingerprint=_fingerprint(
                {
                    "source_updater": raw_belief.updater_fingerprint,
                    "feature_schema": self.fingerprint,
                    "hidden_completion_truth_input": False,
                }
            ),
        )
        raw_information_set = game.information_set(state, player=player)
        actions = tuple(
            replace(
                action,
                features=native_action_feature_items(
                    state.observation,
                    action.official_option_indices,
                ),
            )
            for action in raw_information_set.legal_actions
        )
        private_features = native_private_feature_items(state.observation)
        identifier = _fingerprint(
            {
                "source_information_set": raw_information_set.info_set_id,
                "belief": belief.fingerprint,
                "private": raw_information_set.private_observation_fingerprint,
                "actions": [
                    (action.action_id, action.official_option_indices) for action in actions
                ],
                "feature_schema": self.fingerprint,
                "hidden_completion_truth_input": False,
            }
        )
        information_set = InformationSet(
            info_set_id=f"native-blueprint:{identifier}",
            public_belief_fingerprint=belief.fingerprint,
            actor=raw_information_set.actor,
            legal_actions=actions,
            depth=raw_information_set.depth,
            terminal=False,
            private_observation_fingerprint=(
                raw_information_set.private_observation_fingerprint
            ),
            private_features=private_features,
        )
        return NativeBlueprintView(
            information_set=information_set,
            public_belief=belief,
        )
