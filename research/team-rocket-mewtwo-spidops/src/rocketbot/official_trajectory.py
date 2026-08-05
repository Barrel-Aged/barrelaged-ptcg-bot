"""Leakage-safe logged information-set trajectories from official replays.

This module converts the perspective-isolated Phase 6A replay corpus into a
typed semi-Markov training substrate.  Source transitions genuinely occurred in
the official engine, but the resulting dataset is an offline abstraction: it
does not expose counterfactual branches and must not be presented as an
``AbstractInformationSetGame`` implementation or a trained blueprint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import permutations
import json
import math
from typing import Any, Mapping, Sequence

from .game_theory import (
    AbstractAction,
    GameBoundaryKind,
    GameBoundaryProvenance,
    InformationSet,
    InformationSetActor,
    OpponentRange,
    PublicBeliefState,
    RangeHypothesis,
)
from .public_replay import (
    AgentVisibleSelection,
    PublicEvent,
    PublicPlayerView,
    PublicPokemonView,
    PublicReplayDataset,
    PublicReplayEpisode,
    PublicReplayFrame,
    RevealTarget,
)


OFFICIAL_TRAJECTORY_SCHEMA_VERSION = 1
MAXIMUM_REPLAY_ACTIONS = 4_096
TRAJECTORY_CARD_HASH_BUCKETS = 32
_ZERO_DIGEST = "0" * 64
_COARSE_MAIN_ACTION_TOKENS = (
    "ability",
    "attach",
    "attack",
    "end",
    "evolve",
    "play",
    "retreat",
    "other",
)


def _coarse_action_updater_fingerprint(base_fingerprint: str) -> str:
    return _fingerprint(
        {
            "base_updater_fingerprint": base_fingerprint,
            "public_history_extension": "causal_coarse_main_action_counts_v1",
            "tokens": _COARSE_MAIN_ACTION_TOKENS,
            "private_target_or_card_fields": False,
        }
    )
_PRIVATE_EVENT_KEYS = frozenset(
    {
        "cardId",
        "cardIdActive",
        "cardIdAfter",
        "cardIdBefore",
        "cardIdBench",
        "cardIdTarget",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _finite(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _utility(result: int, observer_player: int) -> float:
    if result == 2:
        return 0.0
    if result == observer_player:
        return 1.0
    if result in (0, 1):
        return -1.0
    raise ValueError("official replay must have a terminal result")


@dataclass(frozen=True, slots=True)
class PublicRangeCandidate:
    """One catalog hypothesis available to the public-only range updater."""

    candidate_id: str
    archetype_id: str
    style_id: str
    card_ids: tuple[int, ...]
    deck_fingerprint: str
    prior: float
    source: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.archetype_id or not self.style_id:
            raise ValueError("range candidate identity must be non-empty")
        if not self.source:
            raise ValueError("range candidate source is required")
        if len(self.card_ids) != 60 or any(card_id <= 0 for card_id in self.card_ids):
            raise ValueError("range candidate must contain 60 positive card IDs")
        _validate_digest(self.deck_fingerprint, "deck_fingerprint")
        if self.prior <= 0 or not math.isfinite(self.prior):
            raise ValueError("range candidate prior must be positive and finite")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "archetype_id": self.archetype_id,
            "style_id": self.style_id,
            "card_ids": list(self.card_ids),
            "deck_fingerprint": self.deck_fingerprint,
            "prior": self.prior,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRangeCandidate":
        return cls(
            candidate_id=str(raw["candidate_id"]),
            archetype_id=str(raw["archetype_id"]),
            style_id=str(raw["style_id"]),
            card_ids=tuple(int(value) for value in raw["card_ids"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            prior=float(raw["prior"]),
            source=str(raw["source"]),
        )


def _pokemon_payload(pokemon: PublicPokemonView) -> dict[str, Any]:
    return {
        "card_id": pokemon.card_id,
        "owner": pokemon.owner,
        "zone": pokemon.zone,
        "hp": pokemon.hp,
        "max_hp": pokemon.max_hp,
        "appeared_this_turn": pokemon.appeared_this_turn,
        "energy_card_ids": list(pokemon.energy_card_ids),
        "tool_card_ids": list(pokemon.tool_card_ids),
        "pre_evolution_card_ids": list(pokemon.pre_evolution_card_ids),
    }


def _public_player_payload(player: PublicPlayerView) -> dict[str, Any]:
    # ``known_hand_card_ids`` is observer private and deliberately absent.
    return {
        "player_index": player.player_index,
        "active": [_pokemon_payload(item) for item in player.active],
        "bench": [_pokemon_payload(item) for item in player.bench],
        "discard_card_ids": list(player.discard_card_ids),
        "deck_count": player.deck_count,
        "hand_count": player.hand_count,
        "prize_count": player.prize_count,
        "bench_max": player.bench_max,
        "special_conditions": list(player.special_conditions),
    }


def _common_event_payload(event: PublicEvent) -> dict[str, Any]:
    # Both perspectives see the public event shape.  Card identity fields are
    # dropped here even when the acting player privately saw its own draw.
    fields = {
        name: value
        for name, value in event.fields
        if name not in _PRIVATE_EVENT_KEYS and "serial" not in name.lower()
    }
    return {"event_type": event.event_type, "fields": fields}


def public_frame_payload(frame: PublicReplayFrame) -> dict[str, Any]:
    """Return only common/public state; no hand, selection, or chosen identity."""

    return {
        "turn": frame.turn,
        "turn_action_count": frame.turn_action_count,
        "first_player": frame.first_player,
        "result": frame.result,
        "supporter_played": frame.supporter_played,
        "stadium_played": frame.stadium_played,
        "energy_attached": frame.energy_attached,
        "retreated": frame.retreated,
        "stadium_card_ids": list(frame.stadium_card_ids),
        "players": [_public_player_payload(player) for player in frame.players],
    }


def public_event_payload(frame: PublicReplayFrame) -> list[dict[str, Any]]:
    return [_common_event_payload(event) for event in frame.events]


def private_frame_payload(frame: PublicReplayFrame) -> dict[str, Any]:
    """Return agent-visible private context only for hashing, never persistence."""

    own = frame.players[frame.observer_player]
    private_events = []
    for event in frame.events:
        fields = dict(event.fields)
        if fields.get("playerIndex") == frame.observer_player:
            identities = {
                name: value for name, value in fields.items() if name in _PRIVATE_EVENT_KEYS
            }
            if identities:
                private_events.append(
                    {"event_type": event.event_type, "identities": identities}
                )
    return {
        "own_hand_card_ids": list(own.known_hand_card_ids),
        "selection": None if frame.selection is None else frame.selection.to_dict(),
        "own_private_events": private_events,
    }


def _public_cards(player: PublicPlayerView) -> Counter[int]:
    result: Counter[int] = Counter(player.discard_card_ids)
    for pokemon in player.active + player.bench:
        result[pokemon.card_id] += 1
        result.update(pokemon.energy_card_ids)
        result.update(pokemon.tool_card_ids)
        result.update(pokemon.pre_evolution_card_ids)
    return result


def _card_bucket(card_id: int) -> int:
    digest = sha256(f"official-card:{int(card_id)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % TRAJECTORY_CARD_HASH_BUCKETS


def _bucketed_card_features(
    prefix: str,
    card_ids: Sequence[int],
) -> tuple[tuple[str, float], ...]:
    counts = Counter(_card_bucket(card_id) for card_id in card_ids)
    return tuple(
        (f"{prefix}_{index:02d}", float(counts[index]))
        for index in range(TRAJECTORY_CARD_HASH_BUCKETS)
    )


def _player_public_card_ids(player: PublicPlayerView) -> tuple[int, ...]:
    cards = list(player.discard_card_ids)
    for pokemon in player.active + player.bench:
        cards.append(pokemon.card_id)
        cards.extend(pokemon.energy_card_ids)
        cards.extend(pokemon.tool_card_ids)
        cards.extend(pokemon.pre_evolution_card_ids)
    return tuple(cards)


def private_feature_items(
    frame: PublicReplayFrame,
) -> tuple[tuple[str, float], ...]:
    """Fixed-width learnable acting-player context with no raw card IDs."""

    if frame.selection is None:
        raise ValueError("private trajectory features require a visible selection")
    own = frame.players[frame.observer_player]
    selection = frame.selection
    base = (
        ("selection_type_code", _stable_type_code(selection.selection_type)),
        ("selection_context_code", _stable_type_code(selection.context)),
        ("selection_min_count", float(selection.min_count)),
        ("selection_max_count", float(selection.max_count)),
        ("selection_option_count", float(len(selection.options))),
        ("remain_damage_counter", float(selection.remain_damage_counter)),
        ("remain_energy_cost", float(selection.remain_energy_cost)),
        ("own_hand_count", float(own.hand_count)),
        ("private_event_identity_count", float(len(private_frame_payload(frame)["own_private_events"]))),
        (
            "effect_card_code",
            0.0
            if selection.effect_card_id is None
            else _stable_type_code(f"card:{selection.effect_card_id}"),
        ),
        (
            "context_card_code",
            0.0
            if selection.context_card_id is None
            else _stable_type_code(f"card:{selection.context_card_id}"),
        ),
    )
    return (
        base
        + _bucketed_card_features("own_hand_card_bucket", own.known_hand_card_ids)
        + _bucketed_card_features(
            "visible_search_card_bucket",
            selection.visible_search_card_ids,
        )
    )


class CausalPublicRangeUpdater:
    """Public revealed-card high-water pseudo-likelihood over deck hypotheses.

    The update consumes neither the episode's true deck fingerprint nor later
    reveal labels.  It is a deterministic baseline representation and data
    audit, not a calibrated neural hidden-hand model.
    """

    def __init__(
        self,
        candidates: Sequence[PublicRangeCandidate],
        *,
        smoothing: float = 1e-6,
    ) -> None:
        materialized = tuple(candidates)
        if not materialized:
            raise ValueError("public range updater requires candidates")
        identifiers = [item.candidate_id for item in materialized]
        fingerprints = [item.deck_fingerprint for item in materialized]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("range candidate IDs must be unique")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("range candidate deck fingerprints must be unique")
        if not 0 < smoothing < 1 or not math.isfinite(smoothing):
            raise ValueError("range smoothing must be finite and in (0, 1)")
        self.candidates = materialized
        self.smoothing = float(smoothing)
        self._counts = {
            item.candidate_id: Counter(item.card_ids) for item in self.candidates
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": OFFICIAL_TRAJECTORY_SCHEMA_VERSION,
                "algorithm": "public_revealed_high_water_combinatorial_pseudolikelihood",
                "smoothing": self.smoothing,
                "candidates": [item.to_dict() for item in self.candidates],
                "truth_label_input": False,
                "future_reveal_input": False,
            }
        )

    def range_from_revealed(self, revealed: Mapping[int, int]) -> OpponentRange:
        cleaned = {
            int(card_id): int(count)
            for card_id, count in revealed.items()
            if int(count) > 0
        }
        if any(card_id <= 0 or count <= 0 for card_id, count in cleaned.items()):
            raise ValueError("public revealed-card counts must be positive")
        raw_scores: list[float] = []
        for candidate in self.candidates:
            counts = self._counts[candidate.candidate_id]
            log_score = math.log(candidate.prior)
            for card_id, seen in cleaned.items():
                available = counts[card_id]
                supported = min(seen, available)
                if supported:
                    log_score += (
                        math.lgamma(available + 1)
                        - math.lgamma(supported + 1)
                        - math.lgamma(available - supported + 1)
                    )
                if seen > available:
                    log_score += (seen - available) * math.log(self.smoothing)
            raw_scores.append(log_score)
        maximum = max(raw_scores)
        masses = [math.exp(value - maximum) for value in raw_scores]
        total = sum(masses)
        hypotheses = []
        revealed_total = sum(cleaned.values())
        for candidate, mass in zip(self.candidates, masses, strict=True):
            counts = self._counts[candidate.candidate_id]
            incompatible = sum(
                max(0, seen - counts[card_id]) for card_id, seen in cleaned.items()
            )
            supported = sum(
                min(seen, counts[card_id]) for card_id, seen in cleaned.items()
            )
            hypotheses.append(
                RangeHypothesis(
                    hypothesis_id=candidate.candidate_id,
                    weight=mass / total,
                    hidden_resource_counts=(
                        ("public_revealed_copies", revealed_total),
                        ("candidate_supported_reveals", supported),
                        ("candidate_unseen_cards", max(0, 60 - supported)),
                        ("incompatible_public_reveals", incompatible),
                    ),
                    intent_id=candidate.archetype_id,
                    style_id=candidate.style_id,
                    source=(
                        "public-only high-water revealed-card pseudo-likelihood; "
                        "candidate catalog prior; no true deck or future reveal input"
                    ),
                )
            )
        return OpponentRange.normalized(hypotheses)

    def belief_for_frame(
        self,
        frame: PublicReplayFrame,
        *,
        revealed: Mapping[int, int],
        public_state_fingerprint: str,
        public_history_fingerprint: str,
    ) -> PublicBeliefState:
        opponent_range = self.range_from_revealed(revealed)
        own = frame.players[frame.observer_player]
        opponent = frame.players[1 - frame.observer_player]
        own_board = own.active + own.bench
        opponent_board = opponent.active + opponent.bench
        public_card_features = (
            _bucketed_card_features(
                "own_public_card_bucket",
                _player_public_card_ids(own),
            )
            + _bucketed_card_features(
                "opponent_public_card_bucket",
                _player_public_card_ids(opponent),
            )
            + _bucketed_card_features(
                "stadium_card_bucket",
                frame.stadium_card_ids,
            )
        )
        return PublicBeliefState(
            public_state_fingerprint=public_state_fingerprint,
            public_history_fingerprint=public_history_fingerprint,
            turn=max(0, frame.turn),
            acting_player=InformationSetActor.SELF,
            opponent_range=opponent_range,
            public_features=(
                ("turn", float(frame.turn)),
                ("turn_action_count", float(frame.turn_action_count)),
                ("own_board_count", float(len(own_board))),
                ("opponent_board_count", float(len(opponent_board))),
                ("own_board_hp", float(sum(item.hp for item in own_board))),
                ("opponent_board_hp", float(sum(item.hp for item in opponent_board))),
                ("own_hand_count", float(own.hand_count)),
                ("opponent_hand_count", float(opponent.hand_count)),
                ("own_prize_count", float(own.prize_count)),
                ("opponent_prize_count", float(opponent.prize_count)),
                ("own_deck_count", float(own.deck_count)),
                ("opponent_deck_count", float(opponent.deck_count)),
                ("opponent_public_revealed_copies", float(sum(revealed.values()))),
                ("opponent_range_entropy", float(opponent_range.entropy)),
                (
                    "opponent_range_effective_sample_size",
                    float(opponent_range.effective_sample_size),
                ),
            )
            + public_card_features,
            updater_fingerprint=self.fingerprint,
        )


def _stable_type_code(value: Any) -> float:
    digest = sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / float(2**32 - 1)


def _option_numeric(option: Mapping[str, Any], name: str) -> float:
    value = option.get(name)
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return _stable_type_code(value)


def complete_replay_actions(
    selection: AgentVisibleSelection,
    *,
    maximum_actions: int = MAXIMUM_REPLAY_ACTIONS,
) -> tuple[AbstractAction, ...]:
    option_count = len(selection.options)
    if selection.max_count > option_count:
        raise ValueError("selection max_count exceeds replay option count")
    total = sum(
        math.perm(option_count, count)
        for count in range(selection.min_count, selection.max_count + 1)
    )
    if total > maximum_actions:
        raise ValueError(
            f"complete replay action space {total} exceeds cap {maximum_actions}"
        )
    options = [dict(item) for item in selection.options]
    actions = []
    for count in range(selection.min_count, selection.max_count + 1):
        for indices in permutations(range(option_count), count):
            type_codes = tuple(_stable_type_code(options[index].get("type")) for index in indices)
            selected_options = tuple(options[index] for index in indices)
            action_id = (
                "native:none"
                if not indices
                else "native:" + ",".join(str(index) for index in indices)
            )
            actions.append(
                AbstractAction(
                    action_id=action_id,
                    label=(
                        "select []"
                        if not indices
                        else f"select {list(indices)}"
                    ),
                    official_option_indices=tuple(indices),
                    features=(
                        ("selection_count", float(len(indices))),
                        ("option_type_code_sum", float(sum(type_codes))),
                        (
                            "option_number_sum",
                            float(sum(_option_numeric(option, "number") for option in selected_options)),
                        ),
                        (
                            "option_area_sum",
                            float(sum(_option_numeric(option, "area") for option in selected_options)),
                        ),
                        (
                            "option_index_sum",
                            float(sum(_option_numeric(option, "index") for option in selected_options)),
                        ),
                        (
                            "option_player_index_sum",
                            float(sum(_option_numeric(option, "playerIndex") for option in selected_options)),
                        ),
                        (
                            "option_count_sum",
                            float(sum(_option_numeric(option, "count") for option in selected_options)),
                        ),
                        (
                            "option_in_play_area_sum",
                            float(sum(_option_numeric(option, "inPlayArea") for option in selected_options)),
                        ),
                        (
                            "option_in_play_index_sum",
                            float(sum(_option_numeric(option, "inPlayIndex") for option in selected_options)),
                        ),
                        (
                            "option_card_code_sum",
                            float(sum(
                                0.0
                                if option.get("cardId") is None
                                else _stable_type_code(f"card:{option['cardId']}")
                                for option in selected_options
                            )),
                        ),
                        (
                            "option_attack_code_sum",
                            float(sum(
                                0.0
                                if option.get("attackId") is None
                                else _stable_type_code(f"attack:{option['attackId']}")
                                for option in selected_options
                            )),
                        ),
                    ),
                )
            )
    if len(actions) != total:
        raise RuntimeError("complete replay action enumeration count mismatch")
    return tuple(actions)


def information_set_from_replay_frame(
    frame: PublicReplayFrame,
    belief: PublicBeliefState,
) -> InformationSet:
    if frame.selection is None or frame.acting_player != frame.observer_player:
        raise ValueError("replay information set requires an acting-observer selection")
    actions = complete_replay_actions(frame.selection)
    private_fingerprint = _fingerprint(private_frame_payload(frame))
    identity = _fingerprint(
        {
            "public_belief": belief.fingerprint,
            "private_observation": private_fingerprint,
            "selection": frame.selection.to_dict(),
            "actions": [
                (action.action_id, action.official_option_indices) for action in actions
            ],
        }
    )
    return InformationSet(
        info_set_id=f"official-replay:{identity}",
        public_belief_fingerprint=belief.fingerprint,
        actor=InformationSetActor.SELF,
        legal_actions=actions,
        depth=frame.frame_index,
        terminal=False,
        private_observation_fingerprint=private_fingerprint,
    )


def _action_to_dict(action: AbstractAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "label": action.label,
        "official_option_indices": list(action.official_option_indices),
        "features": dict(action.features),
    }


def _action_from_dict(raw: Mapping[str, Any]) -> AbstractAction:
    features = raw.get("features", {})
    items = features.items() if isinstance(features, Mapping) else features
    return AbstractAction(
        action_id=str(raw["action_id"]),
        label=str(raw["label"]),
        official_option_indices=tuple(int(value) for value in raw["official_option_indices"]),
        features=tuple((str(name), float(value)) for name, value in items),
    )


def _information_set_to_dict(information_set: InformationSet) -> dict[str, Any]:
    return {
        "info_set_id": information_set.info_set_id,
        "public_belief_fingerprint": information_set.public_belief_fingerprint,
        "actor": information_set.actor.value,
        "legal_actions": [
            _action_to_dict(action) for action in information_set.legal_actions
        ],
        "depth": information_set.depth,
        "terminal": information_set.terminal,
        "private_observation_fingerprint": (
            information_set.private_observation_fingerprint
        ),
    }


def _information_set_from_dict(raw: Mapping[str, Any]) -> InformationSet:
    return InformationSet(
        info_set_id=str(raw["info_set_id"]),
        public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
        actor=InformationSetActor(str(raw["actor"])),
        legal_actions=tuple(
            _action_from_dict(item) for item in raw.get("legal_actions", ())
        ),
        depth=int(raw["depth"]),
        terminal=bool(raw.get("terminal", False)),
        private_observation_fingerprint=(
            None
            if raw.get("private_observation_fingerprint") is None
            else str(raw["private_observation_fingerprint"])
        ),
    )


@dataclass(frozen=True, slots=True)
class LoggedInformationSetStep:
    """One observed acting-player decision and its next logged continuation."""

    step_index: int
    source_frame_index: int
    split: str
    public_belief: PublicBeliefState
    information_set: InformationSet
    private_features: tuple[tuple[str, float], ...]
    behavior_action_id: str | None
    behavior_indices: tuple[int, ...] | None
    next_public_belief_fingerprint: str
    next_public_features: tuple[tuple[str, float], ...]
    public_frames_elapsed: int
    opponent_frames_elapsed: int
    public_event_count: int
    terminal_after_transition: bool
    outcome_utility: float
    eligible_reveal_target_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.source_frame_index < 0:
            raise ValueError("trajectory step indices cannot be negative")
        if self.split not in {"train", "calibration", "validation"}:
            raise ValueError("trajectory split is invalid")
        if self.information_set.public_belief_fingerprint != self.public_belief.fingerprint:
            raise ValueError("information set and public belief fingerprints disagree")
        if self.information_set.actor is not InformationSetActor.SELF:
            raise ValueError("logged acting-observer information set must use SELF actor")
        private_names = [name for name, _ in self.private_features]
        if not private_names or len(private_names) != len(set(private_names)):
            raise ValueError("private feature names must be non-empty and unique")
        for name, value in self.private_features:
            if not name:
                raise ValueError("private feature names cannot be empty")
            _finite(value, f"private feature {name}")
        if (self.behavior_action_id is None) != (self.behavior_indices is None):
            raise ValueError("behavior action ID and indices must be present together")
        if self.behavior_action_id is not None:
            matches = [
                action
                for action in self.information_set.legal_actions
                if action.action_id == self.behavior_action_id
                and action.official_option_indices == self.behavior_indices
            ]
            if len(matches) != 1:
                raise ValueError("logged behavior must match exactly one legal action")
        _validate_digest(
            self.next_public_belief_fingerprint,
            "next_public_belief_fingerprint",
        )
        next_names = [name for name, _ in self.next_public_features]
        current_names = [name for name, _ in self.public_belief.public_features]
        if next_names != current_names:
            raise ValueError("next-public feature schema must match current schema")
        for name, value in self.next_public_features:
            _finite(value, f"next public feature {name}")
        if min(
            self.public_frames_elapsed,
            self.opponent_frames_elapsed,
            self.public_event_count,
        ) < 0:
            raise ValueError("logged transition counts cannot be negative")
        if self.opponent_frames_elapsed > self.public_frames_elapsed:
            raise ValueError("opponent frames cannot exceed elapsed public frames")
        _finite(self.outcome_utility, "outcome_utility")
        if not -1.0 <= self.outcome_utility <= 1.0:
            raise ValueError("outcome utility must be in [-1, 1]")
        if tuple(sorted(set(self.eligible_reveal_target_indices))) != (
            self.eligible_reveal_target_indices
        ):
            raise ValueError("eligible reveal target indices must be sorted and unique")
        if any(index < 0 for index in self.eligible_reveal_target_indices):
            raise ValueError("eligible reveal target indices cannot be negative")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "source_frame_index": self.source_frame_index,
            "split": self.split,
            "public_belief": self.public_belief.to_dict(),
            "information_set": _information_set_to_dict(self.information_set),
            "private_features": dict(self.private_features),
            "behavior_action_id": self.behavior_action_id,
            "behavior_indices": (
                None
                if self.behavior_indices is None
                else list(self.behavior_indices)
            ),
            "next_public_belief_fingerprint": self.next_public_belief_fingerprint,
            "next_public_features": dict(self.next_public_features),
            "public_frames_elapsed": self.public_frames_elapsed,
            "opponent_frames_elapsed": self.opponent_frames_elapsed,
            "public_event_count": self.public_event_count,
            "terminal_after_transition": self.terminal_after_transition,
            "outcome_utility": self.outcome_utility,
            "eligible_reveal_target_indices": list(
                self.eligible_reveal_target_indices
            ),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LoggedInformationSetStep":
        private_features = raw.get("private_features", {})
        private_items = (
            private_features.items()
            if isinstance(private_features, Mapping)
            else private_features
        )
        next_features = raw.get("next_public_features", {})
        next_items = (
            next_features.items()
            if isinstance(next_features, Mapping)
            else next_features
        )
        return cls(
            step_index=int(raw["step_index"]),
            source_frame_index=int(raw["source_frame_index"]),
            split=str(raw["split"]),
            public_belief=PublicBeliefState.from_dict(raw["public_belief"]),
            information_set=_information_set_from_dict(raw["information_set"]),
            private_features=tuple(
                (str(name), float(value)) for name, value in private_items
            ),
            behavior_action_id=(
                None
                if raw.get("behavior_action_id") is None
                else str(raw["behavior_action_id"])
            ),
            behavior_indices=(
                None
                if raw.get("behavior_indices") is None
                else tuple(int(value) for value in raw["behavior_indices"])
            ),
            next_public_belief_fingerprint=str(
                raw["next_public_belief_fingerprint"]
            ),
            next_public_features=tuple(
                (str(name), float(value)) for name, value in next_items
            ),
            public_frames_elapsed=int(raw["public_frames_elapsed"]),
            opponent_frames_elapsed=int(raw["opponent_frames_elapsed"]),
            public_event_count=int(raw["public_event_count"]),
            terminal_after_transition=bool(raw["terminal_after_transition"]),
            outcome_utility=float(raw["outcome_utility"]),
            eligible_reveal_target_indices=tuple(
                int(value)
                for value in raw.get("eligible_reveal_target_indices", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class LoggedTrajectoryEpisode:
    match_id: str
    observer_player: int
    split: str
    source_episode_fingerprint: str
    truth_candidate_id_supervision_only: str
    final_result: int
    final_utility: float
    supervision_only_reveal_targets: tuple[RevealTarget, ...]
    steps: tuple[LoggedInformationSetStep, ...]

    def __post_init__(self) -> None:
        if not self.match_id or self.observer_player not in (0, 1):
            raise ValueError("trajectory episode identity is invalid")
        if self.split not in {"train", "calibration", "validation"}:
            raise ValueError("trajectory episode split is invalid")
        _validate_digest(self.source_episode_fingerprint, "source_episode_fingerprint")
        if not self.truth_candidate_id_supervision_only:
            raise ValueError("episode truth candidate label is required")
        if self.final_result not in (0, 1, 2):
            raise ValueError("trajectory episode must be terminal")
        if not math.isclose(
            self.final_utility,
            _utility(self.final_result, self.observer_player),
            abs_tol=1e-12,
        ):
            raise ValueError("trajectory final utility disagrees with result")
        if not self.steps:
            raise ValueError("trajectory episode requires acting-observer steps")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("trajectory step indices must be contiguous")
        if any(step.split != self.split for step in self.steps):
            raise ValueError("trajectory step split must match episode split")
        if sum(step.terminal_after_transition for step in self.steps) != 1:
            raise ValueError("exactly the final logged step must be terminal")
        if not self.steps[-1].terminal_after_transition:
            raise ValueError("final trajectory step must link to terminal outcome")
        if any(
            index >= len(self.supervision_only_reveal_targets)
            for step in self.steps
            for index in step.eligible_reveal_target_indices
        ):
            raise ValueError("trajectory step references an unknown reveal target")
        for step in self.steps:
            for index in step.eligible_reveal_target_indices:
                target = self.supervision_only_reveal_targets[index]
                if target.revealed_at_frame <= step.source_frame_index:
                    raise ValueError("later-reveal target cannot supervise after reveal")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "observer_player": self.observer_player,
            "split": self.split,
            "source_episode_fingerprint": self.source_episode_fingerprint,
            "truth_candidate_id_supervision_only": (
                self.truth_candidate_id_supervision_only
            ),
            "final_result": self.final_result,
            "final_utility": self.final_utility,
            "supervision_only_reveal_targets": [
                asdict(target) for target in self.supervision_only_reveal_targets
            ],
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LoggedTrajectoryEpisode":
        return cls(
            match_id=str(raw["match_id"]),
            observer_player=int(raw["observer_player"]),
            split=str(raw["split"]),
            source_episode_fingerprint=str(raw["source_episode_fingerprint"]),
            truth_candidate_id_supervision_only=str(
                raw["truth_candidate_id_supervision_only"]
            ),
            final_result=int(raw["final_result"]),
            final_utility=float(raw["final_utility"]),
            supervision_only_reveal_targets=tuple(
                RevealTarget(**item)
                for item in raw.get("supervision_only_reveal_targets", ())
            ),
            steps=tuple(
                LoggedInformationSetStep.from_dict(item)
                for item in raw.get("steps", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryLeakageAudit:
    paired_public_frames: int
    public_state_mismatches: int
    public_history_mismatches: int
    future_reveal_feature_inputs: int
    truth_label_feature_inputs: int
    raw_private_identifiers_persisted: int
    cross_perspective_join_allowed: bool

    def __post_init__(self) -> None:
        for field_name in (
            "paired_public_frames",
            "public_state_mismatches",
            "public_history_mismatches",
            "future_reveal_feature_inputs",
            "truth_label_feature_inputs",
            "raw_private_identifiers_persisted",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.public_state_mismatches > self.paired_public_frames:
            raise ValueError("public-state mismatches exceed paired frames")
        if self.public_history_mismatches > self.paired_public_frames:
            raise ValueError("public-history mismatches exceed paired frames")
        if self.cross_perspective_join_allowed:
            raise ValueError("cross-perspective training joins are forbidden")

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.public_state_mismatches,
                self.public_history_mismatches,
                self.future_reveal_feature_inputs,
                self.truth_label_feature_inputs,
                self.raw_private_identifiers_persisted,
            )
        )


@dataclass(frozen=True, slots=True)
class OfficialTrajectoryDataset:
    source_public_replay_fingerprint: str
    provenance: GameBoundaryProvenance
    source_official_engine_executed: bool
    branchable_transition_game: bool
    range_updater_fingerprint: str
    candidates: tuple[PublicRangeCandidate, ...]
    episodes: tuple[LoggedTrajectoryEpisode, ...]
    leakage_audit: TrajectoryLeakageAudit

    def __post_init__(self) -> None:
        _validate_digest(
            self.source_public_replay_fingerprint,
            "source_public_replay_fingerprint",
        )
        _validate_digest(self.range_updater_fingerprint, "range_updater_fingerprint")
        if self.provenance.kind is not GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME:
            raise ValueError("logged trajectory dataset must be labeled abstract")
        if self.provenance.official_engine_executed:
            raise ValueError("derived trajectory abstraction cannot claim official execution")
        if not self.source_official_engine_executed:
            raise ValueError("official trajectory source must be genuine native replay")
        if self.branchable_transition_game:
            raise ValueError("logged trajectories are not a branchable transition game")
        if not self.candidates or not self.episodes:
            raise ValueError("trajectory dataset requires candidates and episodes")
        candidate_ids = {item.candidate_id for item in self.candidates}
        if len(candidate_ids) != len(self.candidates):
            raise ValueError("trajectory candidates must be unique")
        keys = [(item.match_id, item.observer_player) for item in self.episodes]
        if len(keys) != len(set(keys)):
            raise ValueError("trajectory match/observer pairs must be unique")
        if any(
            episode.truth_candidate_id_supervision_only not in candidate_ids
            for episode in self.episodes
        ):
            raise ValueError("trajectory truth label must reference a candidate")
        steps = tuple(step for episode in self.episodes for step in episode.steps)
        if len({len(step.public_belief.public_features) for step in steps}) != 1:
            raise ValueError("trajectory public feature width must be fixed")
        if len({len(step.private_features) for step in steps}) != 1:
            raise ValueError("trajectory private feature width must be fixed")
        if len(
            {
                len(action.features)
                for step in steps
                for action in step.information_set.legal_actions
            }
        ) != 1:
            raise ValueError("trajectory action feature width must be fixed")
        if not self.leakage_audit.passed:
            raise ValueError("trajectory leakage audit failed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_summary=False))

    @property
    def steps(self) -> tuple[LoggedInformationSetStep, ...]:
        return tuple(step for episode in self.episodes for step in episode.steps)

    @property
    def summary(self) -> dict[str, Any]:
        steps = self.steps
        labeled = [step for step in steps if step.behavior_action_id is not None]
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        nll = []
        brier = []
        correct = 0
        split_counts: Counter[str] = Counter()
        for episode in self.episodes:
            split_counts[episode.split] += len(episode.steps)
            truth = episode.truth_candidate_id_supervision_only
            for step in episode.steps:
                probabilities = {
                    item.hypothesis_id: item.weight
                    for item in step.public_belief.opponent_range.hypotheses
                }
                probability = max(probabilities.get(truth, 0.0), 1e-12)
                nll.append(-math.log(probability))
                brier.append(
                    sum(
                        (
                            probabilities.get(candidate_id, 0.0)
                            - float(candidate_id == truth)
                        )
                        ** 2
                        for candidate_id in candidate_ids
                    )
                )
                predicted = min(
                    probabilities,
                    key=lambda candidate_id: (-probabilities[candidate_id], candidate_id),
                )
                correct += int(predicted == truth)
        candidate_count = len(self.candidates)
        uniform_nll = math.log(candidate_count)
        uniform_brier = 1.0 - 1.0 / candidate_count
        return {
            "schema_version": OFFICIAL_TRAJECTORY_SCHEMA_VERSION,
            "matches": len({episode.match_id for episode in self.episodes}),
            "episodes": len(self.episodes),
            "steps": len(steps),
            "behavior_labeled_steps": len(labeled),
            "behavior_missing_steps": len(steps) - len(labeled),
            "observed_transition_links": len(steps),
            "terminal_transition_links": sum(
                step.terminal_after_transition for step in steps
            ),
            "candidate_ranges": candidate_count,
            "maximum_complete_action_count": max(
                len(step.information_set.legal_actions) for step in steps
            ),
            "mean_complete_action_count": sum(
                len(step.information_set.legal_actions) for step in steps
            )
            / len(steps),
            "public_feature_width": len(steps[0].public_belief.public_features),
            "private_feature_width": len(steps[0].private_features),
            "action_feature_width": len(
                steps[0].information_set.legal_actions[0].features
            ),
            "eligible_future_reveal_links": sum(
                len(step.eligible_reveal_target_indices) for step in steps
            ),
            "split_step_counts": dict(sorted(split_counts.items())),
            "public_range_nll": sum(nll) / len(nll),
            "uniform_range_nll": uniform_nll,
            "public_range_brier": sum(brier) / len(brier),
            "uniform_range_brier": uniform_brier,
            "public_range_top1_accuracy": correct / len(steps),
            "source_official_engine_executed": self.source_official_engine_executed,
            "dataset_boundary_kind": self.provenance.kind.value,
            "dataset_official_engine_executed": self.provenance.official_engine_executed,
            "branchable_transition_game": self.branchable_transition_game,
            "cross_perspective_join_allowed": False,
            "leakage_audit_passed": self.leakage_audit.passed,
            "fingerprint": self.fingerprint,
        }

    def to_dict(self, *, include_summary: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": OFFICIAL_TRAJECTORY_SCHEMA_VERSION,
            "source_public_replay_fingerprint": self.source_public_replay_fingerprint,
            "provenance": {
                **asdict(self.provenance),
                "kind": self.provenance.kind.value,
            },
            "source_official_engine_executed": self.source_official_engine_executed,
            "branchable_transition_game": self.branchable_transition_game,
            "range_updater_fingerprint": self.range_updater_fingerprint,
            "candidates": [item.to_dict() for item in self.candidates],
            "episodes": [item.to_dict() for item in self.episodes],
            "leakage_audit": {
                **asdict(self.leakage_audit),
                "passed": self.leakage_audit.passed,
            },
        }
        if include_summary:
            result["summary"] = self.summary
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(include_summary=True),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialTrajectoryDataset":
        provenance_raw = raw["provenance"]
        provenance = GameBoundaryProvenance(
            name=str(provenance_raw["name"]),
            kind=GameBoundaryKind(str(provenance_raw["kind"])),
            schema_version=int(provenance_raw["schema_version"]),
            ruleset_fingerprint=str(provenance_raw["ruleset_fingerprint"]),
            official_engine_executed=bool(
                provenance_raw["official_engine_executed"]
            ),
            fidelity_notes=str(provenance_raw["fidelity_notes"]),
        )
        audit_raw = raw["leakage_audit"]
        return cls(
            source_public_replay_fingerprint=str(
                raw["source_public_replay_fingerprint"]
            ),
            provenance=provenance,
            source_official_engine_executed=bool(
                raw["source_official_engine_executed"]
            ),
            branchable_transition_game=bool(raw["branchable_transition_game"]),
            range_updater_fingerprint=str(raw["range_updater_fingerprint"]),
            candidates=tuple(
                PublicRangeCandidate.from_dict(item)
                for item in raw.get("candidates", ())
            ),
            episodes=tuple(
                LoggedTrajectoryEpisode.from_dict(item)
                for item in raw.get("episodes", ())
            ),
            leakage_audit=TrajectoryLeakageAudit(
                paired_public_frames=int(audit_raw["paired_public_frames"]),
                public_state_mismatches=int(audit_raw["public_state_mismatches"]),
                public_history_mismatches=int(
                    audit_raw["public_history_mismatches"]
                ),
                future_reveal_feature_inputs=int(
                    audit_raw["future_reveal_feature_inputs"]
                ),
                truth_label_feature_inputs=int(
                    audit_raw["truth_label_feature_inputs"]
                ),
                raw_private_identifiers_persisted=int(
                    audit_raw["raw_private_identifiers_persisted"]
                ),
                cross_perspective_join_allowed=bool(
                    audit_raw["cross_perspective_join_allowed"]
                ),
            ),
        )


def _public_fingerprints(
    episode: PublicReplayEpisode,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    state_fingerprints = []
    history_fingerprints = []
    previous = _ZERO_DIGEST
    for frame in episode.frames:
        state_fingerprint = _fingerprint(public_frame_payload(frame))
        history_fingerprint = _fingerprint(
            {
                "previous": previous,
                "public_state": state_fingerprint,
                "public_events": public_event_payload(frame),
            }
        )
        state_fingerprints.append(state_fingerprint)
        history_fingerprints.append(history_fingerprint)
        previous = history_fingerprint
    return tuple(state_fingerprints), tuple(history_fingerprints)


def _episode_beliefs(
    episode: PublicReplayEpisode,
    updater: CausalPublicRangeUpdater,
) -> tuple[PublicBeliefState, ...]:
    state_fingerprints, history_fingerprints = _public_fingerprints(episode)
    opponent = 1 - episode.provenance.observer_player
    high_water: Counter[int] = Counter()
    has_coarse_history = any(
        event.event_type == "CoarseMainAction"
        for frame in episode.frames
        for event in frame.events
    )
    coarse_counts = {0: Counter(), 1: Counter()}
    enhanced_updater_fingerprint = _coarse_action_updater_fingerprint(
        updater.fingerprint
    )
    beliefs = []
    for frame, state_fingerprint, history_fingerprint in zip(
        episode.frames,
        state_fingerprints,
        history_fingerprints,
        strict=True,
    ):
        for event in frame.events:
            if event.event_type != "CoarseMainAction":
                continue
            fields = dict(event.fields)
            try:
                actor = int(fields.get("playerIndex", -1))
            except (TypeError, ValueError):
                actor = -1
            if actor not in (0, 1):
                raise ValueError("coarse action event has an invalid player")
            raw_token = str(fields.get("optionType", "Other")).lower()
            token = (
                raw_token
                if raw_token in _COARSE_MAIN_ACTION_TOKENS[:-1]
                else "other"
            )
            coarse_counts[actor][token] += 1
        visible = _public_cards(frame.players[opponent])
        for card_id, count in visible.items():
            high_water[card_id] = max(high_water[card_id], count)
        belief = updater.belief_for_frame(
            frame,
            revealed=high_water,
            public_state_fingerprint=state_fingerprint,
            public_history_fingerprint=history_fingerprint,
        )
        if has_coarse_history:
            observer = episode.provenance.observer_player
            additions = tuple(
                (
                    f"{role}_main_action_{token}_count",
                    float(coarse_counts[player][token]),
                )
                for role, player in (
                    ("own", observer),
                    ("opponent", 1 - observer),
                )
                for token in _COARSE_MAIN_ACTION_TOKENS
            )
            belief = replace(
                belief,
                public_features=tuple(
                    sorted(belief.public_features + additions)
                ),
                updater_fingerprint=enhanced_updater_fingerprint,
            )
        beliefs.append(belief)
    return tuple(beliefs)


def _split_by_match(
    dataset_fingerprint: str,
    match_ids: Sequence[str],
) -> dict[str, str]:
    unique = sorted(
        set(match_ids),
        key=lambda match_id: _fingerprint(
            {"dataset": dataset_fingerprint, "match_id": match_id}
        ),
    )
    if len(unique) < 3:
        return {match_id: "train" for match_id in unique}
    calibration_index = len(unique) - 2
    validation_index = len(unique) - 1
    return {
        match_id: (
            "calibration"
            if index == calibration_index
            else "validation"
            if index == validation_index
            else "train"
        )
        for index, match_id in enumerate(unique)
    }


def _behavior_for_information_set(
    frame: PublicReplayFrame,
    information_set: InformationSet,
) -> tuple[str | None, tuple[int, ...] | None]:
    if frame.chosen_indices is None:
        return None, None
    selected = tuple(int(value) for value in frame.chosen_indices)
    selection = frame.selection
    if selection is None:
        raise ValueError("chosen replay indices require visible selection")
    if not selection.min_count <= len(selected) <= selection.max_count:
        raise ValueError("replay behavior violates selection count contract")
    if len(selected) != len(set(selected)):
        raise ValueError("replay behavior repeats an option index")
    if any(index < 0 or index >= len(selection.options) for index in selected):
        raise ValueError("replay behavior option index is out of range")
    matches = [
        action
        for action in information_set.legal_actions
        if action.official_option_indices == selected
    ]
    if len(matches) != 1:
        raise ValueError("replay behavior does not lift to exactly one action")
    return matches[0].action_id, selected


def _build_logged_episode(
    episode: PublicReplayEpisode,
    *,
    updater: CausalPublicRangeUpdater,
    split: str,
    truth_candidate_id: str,
) -> LoggedTrajectoryEpisode:
    beliefs = _episode_beliefs(episode, updater)
    decision_indices = [
        index
        for index, frame in enumerate(episode.frames)
        if frame.selection is not None and frame.acting_player == frame.observer_player
    ]
    if not decision_indices:
        raise ValueError("public replay episode has no acting-observer decisions")
    final_result = episode.frames[-1].result
    final_utility = _utility(final_result, episode.provenance.observer_player)
    steps = []
    for step_index, frame_index in enumerate(decision_indices):
        frame = episode.frames[frame_index]
        belief = beliefs[frame_index]
        information_set = information_set_from_replay_frame(frame, belief)
        behavior_action_id, behavior_indices = _behavior_for_information_set(
            frame,
            information_set,
        )
        is_final = step_index == len(decision_indices) - 1
        next_frame_index = (
            len(episode.frames) - 1
            if is_final
            else decision_indices[step_index + 1]
        )
        interval = episode.frames[frame_index + 1 : next_frame_index + 1]
        reveal_indices = tuple(
            index
            for index, target in enumerate(episode.reveal_targets)
            if target.revealed_at_frame > frame_index
            and target.may_supervise_prefix_through >= frame_index
        )
        steps.append(
            LoggedInformationSetStep(
                step_index=step_index,
                source_frame_index=frame_index,
                split=split,
                public_belief=belief,
                information_set=information_set,
                private_features=private_feature_items(frame),
                behavior_action_id=behavior_action_id,
                behavior_indices=behavior_indices,
                next_public_belief_fingerprint=beliefs[next_frame_index].fingerprint,
                next_public_features=beliefs[next_frame_index].public_features,
                public_frames_elapsed=next_frame_index - frame_index,
                opponent_frames_elapsed=sum(
                    next_frame.acting_player != episode.provenance.observer_player
                    for next_frame in interval
                ),
                public_event_count=sum(len(next_frame.events) for next_frame in interval),
                terminal_after_transition=is_final,
                outcome_utility=final_utility,
                eligible_reveal_target_indices=reveal_indices,
            )
        )
    return LoggedTrajectoryEpisode(
        match_id=episode.provenance.match_id,
        observer_player=episode.provenance.observer_player,
        split=split,
        source_episode_fingerprint=episode.fingerprint,
        truth_candidate_id_supervision_only=truth_candidate_id,
        final_result=final_result,
        final_utility=final_utility,
        supervision_only_reveal_targets=episode.reveal_targets,
        steps=tuple(steps),
    )


def build_official_trajectory_dataset(
    public_replays: PublicReplayDataset,
    candidates: Sequence[PublicRangeCandidate],
    *,
    match_splits: Mapping[str, str] | None = None,
) -> OfficialTrajectoryDataset:
    """Build a deterministic logged corpus without future-label feature input."""

    if not public_replays.validated:
        raise ValueError("official trajectory build requires validated public replays")
    materialized_candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    updater = CausalPublicRangeUpdater(materialized_candidates)
    by_deck_fingerprint = {
        item.deck_fingerprint: item.candidate_id for item in materialized_candidates
    }
    match_ids = {
        episode.provenance.match_id for episode in public_replays.episodes
    }
    if match_splits is None:
        split_map = _split_by_match(public_replays.fingerprint, tuple(match_ids))
    else:
        split_map = {str(name): str(split) for name, split in match_splits.items()}
        missing = match_ids - set(split_map)
        extra = set(split_map) - match_ids
        if missing or extra:
            raise ValueError(
                "explicit trajectory match splits do not match replay IDs: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        invalid = {
            match_id: split
            for match_id, split in split_map.items()
            if split not in {"train", "calibration", "validation"}
        }
        if invalid:
            raise ValueError(f"explicit trajectory match splits are invalid: {invalid}")
        if set(split_map.values()) != {"train", "calibration", "validation"}:
            raise ValueError(
                "explicit trajectory match splits must populate train, calibration, "
                "and validation"
            )
    logged_episodes = []
    for episode in sorted(
        public_replays.episodes,
        key=lambda item: (item.provenance.match_id, item.provenance.observer_player),
    ):
        truth_candidate_id = by_deck_fingerprint.get(
            episode.provenance.opponent_deck_fingerprint
        )
        if truth_candidate_id is None:
            raise ValueError(
                "public replay opponent deck fingerprint is absent from range catalog"
            )
        logged_episodes.append(
            _build_logged_episode(
                episode,
                updater=updater,
                split=split_map[episode.provenance.match_id],
                truth_candidate_id=truth_candidate_id,
            )
        )

    paired_public_frames = 0
    public_state_mismatches = 0
    public_history_mismatches = 0
    by_match: dict[str, dict[int, PublicReplayEpisode]] = {}
    for episode in public_replays.episodes:
        by_match.setdefault(episode.provenance.match_id, {})[
            episode.provenance.observer_player
        ] = episode
    for match_id, perspectives in by_match.items():
        if set(perspectives) != {0, 1}:
            raise ValueError(f"match {match_id} lacks both isolated perspectives")
        zero_states, zero_histories = _public_fingerprints(perspectives[0])
        one_states, one_histories = _public_fingerprints(perspectives[1])
        if len(zero_states) != len(one_states):
            raise ValueError("isolated perspectives have different frame counts")
        paired_public_frames += len(zero_states)
        public_state_mismatches += sum(
            left != right for left, right in zip(zero_states, one_states, strict=True)
        )
        public_history_mismatches += sum(
            left != right
            for left, right in zip(zero_histories, one_histories, strict=True)
        )

    engine_fingerprints = sorted(
        {episode.provenance.engine_sha256 for episode in public_replays.episodes}
    )
    ruleset_fingerprint = (
        engine_fingerprints[0]
        if len(engine_fingerprints) == 1
        else _fingerprint({"engine_fingerprints": engine_fingerprints})
    )
    provenance = GameBoundaryProvenance(
        name="official_public_replay_logged_semi_markov_trajectory",
        kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
        schema_version=OFFICIAL_TRAJECTORY_SCHEMA_VERSION,
        ruleset_fingerprint=ruleset_fingerprint,
        official_engine_executed=False,
        fidelity_notes=(
            "Source transitions genuinely occurred in the official engine. "
            "This derived offline corpus records only observed continuations, "
            "has no counterfactual branches, and is not a native transition game."
        ),
    )
    return OfficialTrajectoryDataset(
        source_public_replay_fingerprint=public_replays.fingerprint,
        provenance=provenance,
        source_official_engine_executed=True,
        branchable_transition_game=False,
        range_updater_fingerprint=(
            _coarse_action_updater_fingerprint(updater.fingerprint)
            if any(
                event.event_type == "CoarseMainAction"
                for episode in public_replays.episodes
                for frame in episode.frames
                for event in frame.events
            )
            else updater.fingerprint
        ),
        candidates=materialized_candidates,
        episodes=tuple(logged_episodes),
        leakage_audit=TrajectoryLeakageAudit(
            paired_public_frames=paired_public_frames,
            public_state_mismatches=public_state_mismatches,
            public_history_mismatches=public_history_mismatches,
            future_reveal_feature_inputs=0,
            truth_label_feature_inputs=0,
            raw_private_identifiers_persisted=0,
            cross_perspective_join_allowed=False,
        ),
    )
