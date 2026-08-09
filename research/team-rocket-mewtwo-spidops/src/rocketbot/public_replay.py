"""Leakage-audited public replay conversion for Phase 6A.

Official visualizer replays contain truth-level hands, decks, prizes, serials,
and draw identities for both players. They are evidence sources, not legal
agent observations. This module converts them through an allow-list into one
agent-perspective episode per player. Private truth remains bound by the source
replay hash but is never copied into model-input frames.

The converter is intentionally conservative. Opponent hidden-zone identities
and choices are omitted; an opponent card identity is retained only after a
public transition (play, attack, attachment, evolution, damage, switch, or a
move touching a public zone). Own hand/search identities are retained because
they are part of that player's information set. Card serials are never retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence


PUBLIC_REPLAY_SCHEMA_VERSION = 2

_PUBLIC_AREAS = frozenset({3, 4, 5, 7, 8, 9, 10, 11})
_PUBLIC_CARD_EVENT_TYPES = frozenset(
    {"Attach", "Attack", "Evolve", "HpChange", "MoveAttached", "Play", "Switch"}
)
_KNOWN_EVENT_TYPES = _PUBLIC_CARD_EVENT_TYPES | frozenset(
    {"Draw", "HasBasicPokemon", "MoveCard", "Result", "Shuffle", "TurnEnd", "TurnStart"}
)
_EVENT_SCALAR_KEYS = frozenset(
    {
        "playerIndex",
        "fromArea",
        "toArea",
        "value",
        "putDamageCounter",
        "hasBasicPokemon",
        "result",
        "reason",
        "attackId",
    }
)
_EVENT_CARD_KEYS = frozenset(
    {
        "cardId",
        "cardIdTarget",
        "cardIdBefore",
        "cardIdAfter",
        "cardIdActive",
        "cardIdBench",
    }
)
_OPTION_KEYS = (
    "type",
    "area",
    "index",
    "playerIndex",
    "inPlayArea",
    "inPlayIndex",
    "energyIndex",
    "count",
    "number",
    "attackId",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _card_id(card: Any) -> int | None:
    if not isinstance(card, Mapping):
        return None
    value = card.get("id", card.get("cardId"))
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result >= 0 else None


def _scalar(value: Any) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class PublicPokemonView:
    card_id: int
    owner: int
    zone: str
    hp: int
    max_hp: int
    appeared_this_turn: bool
    energy_card_ids: tuple[int, ...]
    tool_card_ids: tuple[int, ...]
    pre_evolution_card_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.card_id < 0 or self.owner not in (0, 1):
            raise ValueError("public Pokemon identity/owner is invalid")
        if not self.zone or self.hp < 0 or self.max_hp < 0 or self.hp > self.max_hp:
            raise ValueError("public Pokemon zone/HP is invalid")
        for values in (
            self.energy_card_ids,
            self.tool_card_ids,
            self.pre_evolution_card_ids,
        ):
            if any(value < 0 for value in values):
                raise ValueError("attached public card IDs cannot be negative")


@dataclass(frozen=True, slots=True)
class PublicPlayerView:
    player_index: int
    active: tuple[PublicPokemonView, ...]
    bench: tuple[PublicPokemonView, ...]
    discard_card_ids: tuple[int, ...]
    deck_count: int
    hand_count: int
    prize_count: int
    known_hand_card_ids: tuple[int, ...]
    bench_max: int
    special_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.player_index not in (0, 1):
            raise ValueError("player index must be 0 or 1")
        if min(self.deck_count, self.hand_count, self.prize_count, self.bench_max) < 0:
            raise ValueError("public zone counts cannot be negative")
        if len(self.known_hand_card_ids) not in (0, self.hand_count):
            raise ValueError("known hand must be empty or cover the observer's hand")
        if any(value < 0 for value in self.discard_card_ids + self.known_hand_card_ids):
            raise ValueError("public/known card IDs cannot be negative")
        if len(set(self.special_conditions)) != len(self.special_conditions):
            raise ValueError("special conditions cannot repeat")


@dataclass(frozen=True, slots=True)
class PublicEvent:
    event_type: str
    fields: tuple[tuple[str, bool | int | float | str | None], ...]
    card_identity_redacted: bool

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event type must be non-empty")
        names = [name for name, _ in self.fields]
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("event fields must have unique non-empty names")
        if any("serial" in name.lower() for name in names):
            raise ValueError("card serials are forbidden in public events")
        if self.card_identity_redacted and any(name in _EVENT_CARD_KEYS for name in names):
            raise ValueError("redacted events cannot retain card identity fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "fields": dict(self.fields),
            "card_identity_redacted": self.card_identity_redacted,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicEvent":
        fields = raw.get("fields", {})
        items = fields.items() if isinstance(fields, Mapping) else fields
        return cls(
            event_type=str(raw["event_type"]),
            fields=tuple(sorted((str(name), _scalar(value)) for name, value in items)),
            card_identity_redacted=bool(raw["card_identity_redacted"]),
        )


@dataclass(frozen=True, slots=True)
class AgentVisibleSelection:
    selection_type: str
    context: str
    min_count: int
    max_count: int
    remain_damage_counter: int
    remain_energy_cost: int
    options: tuple[tuple[tuple[str, bool | int | float | str | None], ...], ...]
    visible_search_card_ids: tuple[int, ...]
    effect_card_id: int | None
    context_card_id: int | None

    def __post_init__(self) -> None:
        if not self.selection_type or not self.context:
            raise ValueError("selection type/context must be non-empty")
        if self.min_count < 0 or self.max_count < self.min_count:
            raise ValueError("selection bounds are invalid")
        if self.remain_damage_counter < 0 or self.remain_energy_cost < 0:
            raise ValueError("remaining selection costs cannot be negative")
        if any(value < 0 for value in self.visible_search_card_ids):
            raise ValueError("visible search card IDs cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_type": self.selection_type,
            "context": self.context,
            "min_count": self.min_count,
            "max_count": self.max_count,
            "remain_damage_counter": self.remain_damage_counter,
            "remain_energy_cost": self.remain_energy_cost,
            "options": [dict(option) for option in self.options],
            "visible_search_card_ids": list(self.visible_search_card_ids),
            "effect_card_id": self.effect_card_id,
            "context_card_id": self.context_card_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentVisibleSelection":
        return cls(
            selection_type=str(raw["selection_type"]),
            context=str(raw["context"]),
            min_count=int(raw["min_count"]),
            max_count=int(raw["max_count"]),
            remain_damage_counter=int(raw["remain_damage_counter"]),
            remain_energy_cost=int(raw["remain_energy_cost"]),
            options=tuple(
                tuple(sorted((str(name), _scalar(value)) for name, value in option.items()))
                for option in raw.get("options", ())
            ),
            visible_search_card_ids=tuple(
                int(value) for value in raw.get("visible_search_card_ids", ())
            ),
            effect_card_id=(
                None if raw.get("effect_card_id") is None else int(raw["effect_card_id"])
            ),
            context_card_id=(
                None
                if raw.get("context_card_id") is None
                else int(raw["context_card_id"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicReplayFrame:
    frame_index: int
    observer_player: int
    acting_player: int
    turn: int
    turn_action_count: int
    first_player: int
    result: int
    supporter_played: bool
    stadium_played: bool
    energy_attached: bool
    retreated: bool
    stadium_card_ids: tuple[int, ...]
    players: tuple[PublicPlayerView, PublicPlayerView]
    events: tuple[PublicEvent, ...]
    selection: AgentVisibleSelection | None
    chosen_indices: tuple[int, ...] | None
    public_history_fingerprint: str
    observation_fingerprint: str

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.observer_player not in (0, 1):
            raise ValueError("frame index/observer is invalid")
        if self.acting_player not in (0, 1) or self.turn < 0 or self.turn_action_count < 0:
            raise ValueError("acting player/turn fields are invalid")
        if self.first_player not in (-1, 0, 1) or self.result not in (-1, 0, 1, 2):
            raise ValueError("first player/result is invalid")
        if tuple(player.player_index for player in self.players) != (0, 1):
            raise ValueError("public frames require player views in 0,1 order")
        opponent = self.players[1 - self.observer_player]
        if opponent.known_hand_card_ids:
            raise ValueError("opponent hand identities cannot enter public replay frames")
        if self.selection is not None and self.acting_player != self.observer_player:
            raise ValueError("only the acting observer can receive legal selection details")
        if self.chosen_indices is not None and self.acting_player != self.observer_player:
            raise ValueError("only the acting observer can receive chosen private indices")
        if self.result != -1 and self.selection is not None:
            raise ValueError("terminal replay frames cannot expose a pending selection")
        if self.chosen_indices is not None:
            if self.selection is None:
                raise ValueError("chosen indices require their source selection")
            if not self.selection.min_count <= len(self.chosen_indices) <= self.selection.max_count:
                raise ValueError("chosen indices violate their selection count contract")
            if len(self.chosen_indices) != len(set(self.chosen_indices)):
                raise ValueError("chosen indices cannot repeat")
            if any(
                index < 0 or index >= len(self.selection.options)
                for index in self.chosen_indices
            ):
                raise ValueError("chosen indices must reference source selection options")
        _validate_digest(self.public_history_fingerprint, "public_history_fingerprint")
        _validate_digest(self.observation_fingerprint, "observation_fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "observer_player": self.observer_player,
            "acting_player": self.acting_player,
            "turn": self.turn,
            "turn_action_count": self.turn_action_count,
            "first_player": self.first_player,
            "result": self.result,
            "supporter_played": self.supporter_played,
            "stadium_played": self.stadium_played,
            "energy_attached": self.energy_attached,
            "retreated": self.retreated,
            "stadium_card_ids": list(self.stadium_card_ids),
            "players": [asdict(player) for player in self.players],
            "events": [event.to_dict() for event in self.events],
            "selection": None if self.selection is None else self.selection.to_dict(),
            "chosen_indices": (
                None if self.chosen_indices is None else list(self.chosen_indices)
            ),
            "public_history_fingerprint": self.public_history_fingerprint,
            "observation_fingerprint": self.observation_fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicReplayFrame":
        def pokemon(item: Mapping[str, Any]) -> PublicPokemonView:
            return PublicPokemonView(
                card_id=int(item["card_id"]),
                owner=int(item["owner"]),
                zone=str(item["zone"]),
                hp=int(item["hp"]),
                max_hp=int(item["max_hp"]),
                appeared_this_turn=bool(item["appeared_this_turn"]),
                energy_card_ids=tuple(int(value) for value in item["energy_card_ids"]),
                tool_card_ids=tuple(int(value) for value in item["tool_card_ids"]),
                pre_evolution_card_ids=tuple(
                    int(value) for value in item["pre_evolution_card_ids"]
                ),
            )

        def player(item: Mapping[str, Any]) -> PublicPlayerView:
            return PublicPlayerView(
                player_index=int(item["player_index"]),
                active=tuple(pokemon(value) for value in item["active"]),
                bench=tuple(pokemon(value) for value in item["bench"]),
                discard_card_ids=tuple(int(value) for value in item["discard_card_ids"]),
                deck_count=int(item["deck_count"]),
                hand_count=int(item["hand_count"]),
                prize_count=int(item["prize_count"]),
                known_hand_card_ids=tuple(
                    int(value) for value in item["known_hand_card_ids"]
                ),
                bench_max=int(item["bench_max"]),
                special_conditions=tuple(str(value) for value in item["special_conditions"]),
            )

        players = tuple(player(item) for item in raw["players"])
        if len(players) != 2:
            raise ValueError("public replay frame requires two players")
        return cls(
            frame_index=int(raw["frame_index"]),
            observer_player=int(raw["observer_player"]),
            acting_player=int(raw["acting_player"]),
            turn=int(raw["turn"]),
            turn_action_count=int(raw["turn_action_count"]),
            first_player=int(raw["first_player"]),
            result=int(raw["result"]),
            supporter_played=bool(raw["supporter_played"]),
            stadium_played=bool(raw["stadium_played"]),
            energy_attached=bool(raw["energy_attached"]),
            retreated=bool(raw["retreated"]),
            stadium_card_ids=tuple(int(value) for value in raw["stadium_card_ids"]),
            players=(players[0], players[1]),
            events=tuple(PublicEvent.from_dict(item) for item in raw["events"]),
            selection=(
                None
                if raw.get("selection") is None
                else AgentVisibleSelection.from_dict(raw["selection"])
            ),
            chosen_indices=(
                None
                if raw.get("chosen_indices") is None
                else tuple(int(value) for value in raw["chosen_indices"])
            ),
            public_history_fingerprint=str(raw["public_history_fingerprint"]),
            observation_fingerprint=str(raw["observation_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class PublicReplayProvenance:
    match_id: str
    observer_player: int
    source_kind: str
    replay_sha256: str
    engine_sha256: str
    card_pool_sha256: str
    environment_fingerprint: str
    observer_deck_fingerprint: str
    opponent_deck_fingerprint: str
    converter_fingerprint: str

    def __post_init__(self) -> None:
        if not self.match_id or self.observer_player not in (0, 1):
            raise ValueError("replay provenance identity is invalid")
        if self.source_kind != "official_engine_truth_replay":
            raise ValueError("source kind must explicitly identify truth-level replay input")
        for field_name in (
            "replay_sha256",
            "engine_sha256",
            "card_pool_sha256",
            "environment_fingerprint",
            "observer_deck_fingerprint",
            "opponent_deck_fingerprint",
            "converter_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class RevealTarget:
    """Supervision-only label that becomes available after a public reveal."""

    opaque_card_token: str
    opponent_player: int
    card_id: int
    revealed_at_frame: int
    may_supervise_prefix_through: int
    source_event_type: str

    def __post_init__(self) -> None:
        _validate_digest(self.opaque_card_token, "opaque_card_token")
        if self.opponent_player not in (0, 1) or self.card_id < 0:
            raise ValueError("reveal target opponent/card is invalid")
        if self.revealed_at_frame < 0 or self.may_supervise_prefix_through != (
            self.revealed_at_frame - 1
        ):
            raise ValueError("reveal label must only supervise prefixes before reveal")
        if not self.source_event_type:
            raise ValueError("reveal target source event must be described")


@dataclass(frozen=True, slots=True)
class LeakageAuditReport:
    passed: bool
    frame_count: int
    decision_frame_count: int
    redacted_event_count: int
    unknown_event_types: tuple[str, ...]
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        if min(self.frame_count, self.decision_frame_count, self.redacted_event_count) < 0:
            raise ValueError("leakage audit counts cannot be negative")
        if self.decision_frame_count > self.frame_count:
            raise ValueError("decision frames cannot exceed all frames")
        if self.passed != (not self.violations):
            raise ValueError("leakage audit pass flag must agree with violations")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class PublicReplayEpisode:
    provenance: PublicReplayProvenance
    frames: tuple[PublicReplayFrame, ...]
    reveal_targets: tuple[RevealTarget, ...]
    audit: LeakageAuditReport

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("public replay episode cannot be empty")
        if tuple(frame.frame_index for frame in self.frames) != tuple(range(len(self.frames))):
            raise ValueError("public replay frame indices must be contiguous")
        if any(
            frame.observer_player != self.provenance.observer_player for frame in self.frames
        ):
            raise ValueError("episode observer must match every frame")
        if self.audit.frame_count != len(self.frames):
            raise ValueError("leakage audit frame count must match episode")
        tokens = [target.opaque_card_token for target in self.reveal_targets]
        if len(tokens) != len(set(tokens)):
            raise ValueError("each revealed physical card can appear only once")
        if any(
            target.opponent_player == self.provenance.observer_player
            for target in self.reveal_targets
        ):
            raise ValueError("reveal targets must describe the opponent")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_audit=True))

    def to_dict(self, *, include_audit: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PUBLIC_REPLAY_SCHEMA_VERSION,
            "provenance": asdict(self.provenance),
            "frames": [frame.to_dict() for frame in self.frames],
            "supervision_only_reveal_targets": [
                asdict(target) for target in self.reveal_targets
            ],
        }
        if include_audit:
            result["audit"] = asdict(self.audit)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicReplayEpisode":
        provenance = PublicReplayProvenance(**raw["provenance"])
        frames = tuple(PublicReplayFrame.from_dict(item) for item in raw["frames"])
        reveal_targets = tuple(
            RevealTarget(**item)
            for item in raw.get("supervision_only_reveal_targets", ())
        )
        audit_raw = raw["audit"]
        audit = LeakageAuditReport(
            passed=bool(audit_raw["passed"]),
            frame_count=int(audit_raw["frame_count"]),
            decision_frame_count=int(audit_raw["decision_frame_count"]),
            redacted_event_count=int(audit_raw["redacted_event_count"]),
            unknown_event_types=tuple(audit_raw.get("unknown_event_types", ())),
            violations=tuple(audit_raw.get("violations", ())),
        )
        return cls(
            provenance=provenance,
            frames=frames,
            reveal_targets=reveal_targets,
            audit=audit,
        )


@dataclass(frozen=True, slots=True)
class PublicReplayDataset:
    episodes: tuple[PublicReplayEpisode, ...]

    def __post_init__(self) -> None:
        if not self.episodes:
            raise ValueError("public replay dataset cannot be empty")
        keys = [
            (episode.provenance.match_id, episode.provenance.observer_player)
            for episode in self.episodes
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("public replay match/observer pairs must be unique")

    @property
    def validated(self) -> bool:
        return all(episode.audit.passed for episode in self.episodes)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_episodes=True))

    @property
    def summary(self) -> dict[str, Any]:
        unknown = sorted(
            {
                item
                for episode in self.episodes
                for item in episode.audit.unknown_event_types
            }
        )
        return {
            "schema_version": PUBLIC_REPLAY_SCHEMA_VERSION,
            "episodes": len(self.episodes),
            "matches": len({episode.provenance.match_id for episode in self.episodes}),
            "frames": sum(len(episode.frames) for episode in self.episodes),
            "decision_frames": sum(
                episode.audit.decision_frame_count for episode in self.episodes
            ),
            "redacted_events": sum(
                episode.audit.redacted_event_count for episode in self.episodes
            ),
            "supervision_only_reveal_targets": sum(
                len(episode.reveal_targets) for episode in self.episodes
            ),
            "cross_perspective_join_allowed": False,
            "unknown_event_types": unknown,
            "validated": self.validated,
            "fingerprint": self.fingerprint,
        }

    def to_dict(self, *, include_episodes: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"schema_version": PUBLIC_REPLAY_SCHEMA_VERSION}
        if include_episodes:
            result["episodes"] = [episode.to_dict() for episode in self.episodes]
        return result

    def to_json(self, *, indent: int = 2) -> str:
        result = self.to_dict(include_episodes=True)
        result["fingerprint"] = self.fingerprint
        result["summary"] = self.summary
        return json.dumps(result, indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicReplayDataset":
        return cls(tuple(PublicReplayEpisode.from_dict(item) for item in raw["episodes"]))


class OfficialReplayConverter:
    """Allow-list converter from official truth replay to agent perspectives."""

    RULES = {
        "schema": PUBLIC_REPLAY_SCHEMA_VERSION,
        "private_zones": ["deck", "hand", "prize", "looking"],
        "serials": "drop_all",
        "opponent_hidden_events": "drop_card_identity",
        "selection": "acting_observer_only",
        "public_areas": sorted(_PUBLIC_AREAS),
        "known_event_types": sorted(_KNOWN_EVENT_TYPES),
        "selected_semantics": "next_frame_selected_labels_current_prompt",
        "behavior_label_in_feature_fingerprints": False,
        "public_history_card_identity": "drop_all_event_card_ids",
    }

    def __init__(
        self,
        *,
        extra_known_event_types: Sequence[str] = (),
        extra_public_scalar_keys: Sequence[str] = (),
        include_coarse_main_action_events: bool = False,
    ) -> None:
        self.extra_known_event_types = tuple(
            sorted({str(value) for value in extra_known_event_types})
        )
        self.extra_public_scalar_keys = tuple(
            sorted({str(value) for value in extra_public_scalar_keys})
        )
        self.include_coarse_main_action_events = bool(
            include_coarse_main_action_events
        )
        if any(not value for value in self.extra_known_event_types):
            raise ValueError("extra known replay event types cannot be empty")
        if any(not value for value in self.extra_public_scalar_keys):
            raise ValueError("extra public replay scalar keys cannot be empty")

    @property
    def known_event_types(self) -> frozenset[str]:
        return _KNOWN_EVENT_TYPES | frozenset(self.extra_known_event_types)

    @property
    def public_scalar_keys(self) -> frozenset[str]:
        return _EVENT_SCALAR_KEYS | frozenset(self.extra_public_scalar_keys)

    @property
    def fingerprint(self) -> str:
        if (
            not self.extra_known_event_types
            and not self.extra_public_scalar_keys
            and not self.include_coarse_main_action_events
        ):
            return _fingerprint(self.RULES)
        return _fingerprint(
            {
                **self.RULES,
                "extra_known_event_types": list(self.extra_known_event_types),
                "extra_public_scalar_keys": list(self.extra_public_scalar_keys),
                "include_coarse_main_action_events": (
                    self.include_coarse_main_action_events
                ),
            }
        )

    def convert_match(
        self,
        replay_data: str | Sequence[Mapping[str, Any]],
        *,
        match_id: str,
        engine_sha256: str,
        card_pool_sha256: str,
        environment_fingerprint: str,
        deck_fingerprints: tuple[str, str],
    ) -> tuple[PublicReplayEpisode, PublicReplayEpisode]:
        if isinstance(replay_data, str):
            frames_raw = json.loads(replay_data)
            replay_text = replay_data
        else:
            frames_raw = list(replay_data)
            replay_text = json.dumps(frames_raw, sort_keys=True, separators=(",", ":"))
        if not isinstance(frames_raw, list) or not frames_raw:
            raise ValueError("official replay must be a non-empty frame list")
        for digest, name in (
            (engine_sha256, "engine_sha256"),
            (card_pool_sha256, "card_pool_sha256"),
            (environment_fingerprint, "environment_fingerprint"),
            (deck_fingerprints[0], "deck_fingerprint_0"),
            (deck_fingerprints[1], "deck_fingerprint_1"),
        ):
            _validate_digest(digest, name)
        replay_sha = sha256(replay_text.encode("utf-8")).hexdigest()
        episodes = tuple(
            self._convert_observer(
                frames_raw,
                provenance=PublicReplayProvenance(
                    match_id=match_id,
                    observer_player=observer,
                    source_kind="official_engine_truth_replay",
                    replay_sha256=replay_sha,
                    engine_sha256=engine_sha256,
                    card_pool_sha256=card_pool_sha256,
                    environment_fingerprint=environment_fingerprint,
                    observer_deck_fingerprint=deck_fingerprints[observer],
                    opponent_deck_fingerprint=deck_fingerprints[1 - observer],
                    converter_fingerprint=self.fingerprint,
                ),
            )
            for observer in (0, 1)
        )
        return (episodes[0], episodes[1])

    def _convert_observer(
        self,
        source_frames: Sequence[Mapping[str, Any]],
        *,
        provenance: PublicReplayProvenance,
    ) -> PublicReplayEpisode:
        observer = provenance.observer_player
        previous_logs: Sequence[Mapping[str, Any]] = ()
        history_digest = "0" * 64
        frames: list[PublicReplayFrame] = []
        unknown_types: set[str] = set()
        reveal_targets: list[RevealTarget] = []
        revealed_serials: set[int] = set()
        for frame_index, source in enumerate(source_frames):
            current = source.get("current")
            if not isinstance(current, Mapping):
                raise ValueError(f"frame {frame_index} has no current state")
            raw_logs = source.get("logs", ())
            logs = tuple(item for item in raw_logs if isinstance(item, Mapping))
            incremental = self._incremental_logs(previous_logs, logs)
            previous_logs = logs
            events: list[PublicEvent] = []
            for raw_event in incremental:
                event = self._public_event(raw_event, observer)
                events.append(event)
                event_type = str(raw_event.get("type", "Unknown"))
                if event_type not in self.known_event_types:
                    unknown_types.add(event_type)
                reveal_targets.extend(
                    self._reveal_targets(
                        raw_event,
                        observer=observer,
                        frame_index=frame_index,
                        replay_sha256=provenance.replay_sha256,
                        already_revealed=revealed_serials,
                    )
                )
            if self.include_coarse_main_action_events and frame_index > 0:
                previous_source = source_frames[frame_index - 1]
                previous_select = previous_source.get("select")
                selected = source.get("selected")
                if (
                    isinstance(previous_select, Mapping)
                    and str(previous_select.get("context", "")) == "Main"
                    and isinstance(selected, Sequence)
                    and not isinstance(selected, (str, bytes, bytearray))
                ):
                    previous_current = previous_source.get("current")
                    if not isinstance(previous_current, Mapping):
                        raise ValueError("coarse action history lacks its source state")
                    actor = _safe_int(previous_current.get("yourIndex"), -1)
                    options = previous_select.get("option")
                    if actor not in (0, 1) or not isinstance(options, Sequence):
                        raise ValueError("coarse action history source is malformed")
                    indices = tuple(_safe_int(value, -1) for value in selected)
                    if len(indices) != 1 or indices[0] not in range(len(options)):
                        raise ValueError("MAIN action history must select one legal option")
                    option = options[indices[0]]
                    if not isinstance(option, Mapping):
                        raise ValueError("coarse action history option is malformed")
                    events.append(
                        PublicEvent(
                            event_type="CoarseMainAction",
                            fields=(
                                ("optionType", str(option.get("type", "Other"))),
                                ("playerIndex", actor),
                            ),
                            card_identity_redacted=False,
                        )
                    )
            acting_player = _safe_int(current.get("yourIndex"), 0)
            players_raw = current.get("players", ())
            if not isinstance(players_raw, Sequence) or len(players_raw) != 2:
                raise ValueError(f"frame {frame_index} must contain two players")
            players = tuple(
                self._public_player(player, player_index=index, observer=observer)
                for index, player in enumerate(players_raw)
            )
            if len(players) != 2:
                raise AssertionError("two-player conversion invariant failed")
            result = _safe_int(current.get("result"), -1)
            selection = (
                self._selection(source.get("select"))
                if result == -1
                and acting_player == observer
                and isinstance(source.get("select"), Mapping)
                else None
            )
            # The official visualizer records the selection that advanced frame
            # N on frame N+1.  Shift it back to its source prompt.  Attaching
            # ``source['selected']`` here would label the new prompt with the
            # previous actor's action and can violate the new prompt contract.
            next_source = (
                source_frames[frame_index + 1]
                if frame_index + 1 < len(source_frames)
                else None
            )
            chosen = (
                next_source.get("selected")
                if isinstance(next_source, Mapping)
                else None
            )
            chosen_indices = (
                tuple(_safe_int(value) for value in chosen)
                if selection is not None
                and isinstance(chosen, Sequence)
                and not isinstance(chosen, (str, bytes, bytearray))
                else None
            )
            stadium_ids = tuple(
                value
                for value in (_card_id(card) for card in (current.get("stadium") or ()))
                if value is not None
            )
            history_payload = {
                "previous": history_digest,
                "events": [self._common_history_event(event) for event in events],
                "acting_player": acting_player,
            }
            history_digest = _fingerprint(history_payload)
            observation_payload = {
                "observer_player": observer,
                "acting_player": acting_player,
                "turn": _safe_int(current.get("turn")),
                "turn_action_count": _safe_int(current.get("turnActionCount")),
                "first_player": _safe_int(current.get("firstPlayer"), -1),
                "result": result,
                "stadium_card_ids": stadium_ids,
                "players": [asdict(player) for player in players],
                "selection": None if selection is None else selection.to_dict(),
                "public_history_fingerprint": history_digest,
            }
            frames.append(
                PublicReplayFrame(
                    frame_index=frame_index,
                    observer_player=observer,
                    acting_player=acting_player,
                    turn=_safe_int(current.get("turn")),
                    turn_action_count=_safe_int(current.get("turnActionCount")),
                    first_player=_safe_int(current.get("firstPlayer"), -1),
                    result=result,
                    supporter_played=bool(current.get("supporterPlayed", False)),
                    stadium_played=bool(current.get("stadiumPlayed", False)),
                    energy_attached=bool(current.get("energyAttached", False)),
                    retreated=bool(current.get("retreated", False)),
                    stadium_card_ids=stadium_ids,
                    players=(players[0], players[1]),
                    events=tuple(events),
                    selection=selection,
                    chosen_indices=chosen_indices,
                    public_history_fingerprint=history_digest,
                    observation_fingerprint=_fingerprint(observation_payload),
                )
            )
        audit = self.audit(tuple(frames), unknown_event_types=tuple(sorted(unknown_types)))
        return PublicReplayEpisode(
            provenance=provenance,
            frames=tuple(frames),
            reveal_targets=tuple(reveal_targets),
            audit=audit,
        )

    @staticmethod
    def _common_history_event(event: PublicEvent) -> dict[str, Any]:
        """Common event shape with all private/self card identities removed."""

        return {
            "event_type": event.event_type,
            "fields": {
                name: value
                for name, value in event.fields
                if name not in _EVENT_CARD_KEYS and "serial" not in name.lower()
            },
        }

    @staticmethod
    def _incremental_logs(
        previous: Sequence[Mapping[str, Any]],
        current: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        common = 0
        limit = min(len(previous), len(current))
        while common < limit and previous[common] == current[common]:
            common += 1
        return tuple(current[common:])

    @staticmethod
    def _pokemon(card: Mapping[str, Any], *, owner: int, zone: str) -> PublicPokemonView:
        identifier = _card_id(card)
        if identifier is None:
            raise ValueError("public Pokemon is missing a card ID")
        return PublicPokemonView(
            card_id=identifier,
            owner=owner,
            zone=zone,
            hp=max(0, _safe_int(card.get("hp"))),
            max_hp=max(0, _safe_int(card.get("maxHp"))),
            appeared_this_turn=bool(card.get("appearThisTurn", False)),
            energy_card_ids=tuple(
                value
                for value in (_card_id(item) for item in (card.get("energyCards") or ()))
                if value is not None
            ),
            tool_card_ids=tuple(
                value
                for value in (_card_id(item) for item in (card.get("tools") or ()))
                if value is not None
            ),
            pre_evolution_card_ids=tuple(
                value
                for value in (
                    _card_id(item) for item in (card.get("preEvolution") or ())
                )
                if value is not None
            ),
        )

    def _public_player(
        self,
        raw: Any,
        *,
        player_index: int,
        observer: int,
    ) -> PublicPlayerView:
        if not isinstance(raw, Mapping):
            raise ValueError("player state must be a mapping")
        active = tuple(
            self._pokemon(item, owner=player_index, zone="active")
            for item in (raw.get("active") or ())
            if isinstance(item, Mapping)
        )
        bench = tuple(
            self._pokemon(item, owner=player_index, zone="bench")
            for item in (raw.get("bench") or ())
            if isinstance(item, Mapping)
        )
        discard_ids = tuple(
            value
            for value in (_card_id(item) for item in (raw.get("discard") or ()))
            if value is not None
        )
        hand_items = raw.get("hand") or ()
        deck_items = raw.get("deck") or ()
        prize_items = raw.get("prize") or ()
        hand_count = max(0, _safe_int(raw.get("handCount"), len(hand_items)))
        known_hand = (
            tuple(
                value
                for value in (_card_id(item) for item in hand_items)
                if value is not None
            )
            if player_index == observer
            else ()
        )
        conditions = tuple(
            name
            for name, key in (
                ("poisoned", "poisoned"),
                ("burned", "burned"),
                ("asleep", "asleep"),
                ("paralyzed", "paralyzed"),
                ("confused", "confused"),
            )
            if bool(raw.get(key, False))
        )
        return PublicPlayerView(
            player_index=player_index,
            active=active,
            bench=bench,
            discard_card_ids=discard_ids,
            deck_count=max(0, _safe_int(raw.get("deckCount"), len(deck_items))),
            hand_count=hand_count,
            prize_count=max(0, len(prize_items)),
            known_hand_card_ids=known_hand,
            bench_max=max(0, _safe_int(raw.get("benchMax"), 5)),
            special_conditions=conditions,
        )

    @staticmethod
    def _selection(raw: Any) -> AgentVisibleSelection:
        if not isinstance(raw, Mapping):
            raise ValueError("agent-visible selection must be a mapping")
        options: list[tuple[tuple[str, bool | int | float | str | None], ...]] = []
        for option in (raw.get("option") or ()):
            if not isinstance(option, Mapping):
                continue
            fields = tuple(
                (key, _scalar(option[key])) for key in _OPTION_KEYS if key in option
            )
            options.append(fields)
        search_ids = tuple(
            value
            for value in (_card_id(card) for card in (raw.get("deck") or ()))
            if value is not None
        )
        return AgentVisibleSelection(
            selection_type=str(raw.get("type", "Unknown")),
            context=str(raw.get("context", "Unknown")),
            min_count=max(0, _safe_int(raw.get("minCount"))),
            max_count=max(0, _safe_int(raw.get("maxCount"))),
            remain_damage_counter=max(0, _safe_int(raw.get("remainDamageCounter"))),
            remain_energy_cost=max(0, _safe_int(raw.get("remainEnergyCost"))),
            options=tuple(options),
            visible_search_card_ids=search_ids,
            effect_card_id=_card_id(raw.get("effect")),
            context_card_id=_card_id(raw.get("contextCard")),
        )

    def _public_event(self, raw: Mapping[str, Any], observer: int) -> PublicEvent:
        event_type = str(raw.get("type", "Unknown"))
        owner = _safe_int(raw.get("playerIndex"), -1)
        card_visible = OfficialReplayConverter._event_card_visible(raw, observer)
        fields: list[tuple[str, bool | int | float | str | None]] = []
        for name in sorted(self.public_scalar_keys):
            if name in raw:
                fields.append((name, _scalar(raw[name])))
        hidden_card_present = any(name in raw for name in _EVENT_CARD_KEYS)
        if card_visible:
            for name in sorted(_EVENT_CARD_KEYS):
                if name in raw:
                    fields.append((name, _scalar(raw[name])))
        return PublicEvent(
            event_type=event_type,
            fields=tuple(sorted(fields)),
            card_identity_redacted=hidden_card_present and not card_visible,
        )

    @staticmethod
    def _event_card_visible(raw: Mapping[str, Any], observer: int) -> bool:
        event_type = str(raw.get("type", "Unknown"))
        owner = _safe_int(raw.get("playerIndex"), -1)
        if event_type in _PUBLIC_CARD_EVENT_TYPES:
            return True
        if event_type == "Draw":
            return owner == observer
        if event_type == "MoveCard":
            from_area = _safe_int(raw.get("fromArea"), -1)
            to_area = _safe_int(raw.get("toArea"), -1)
            return owner == observer or from_area in _PUBLIC_AREAS or to_area in _PUBLIC_AREAS
        return owner == observer

    @staticmethod
    def _reveal_targets(
        raw: Mapping[str, Any],
        *,
        observer: int,
        frame_index: int,
        replay_sha256: str,
        already_revealed: set[int],
    ) -> tuple[RevealTarget, ...]:
        owner = _safe_int(raw.get("playerIndex"), -1)
        if owner != 1 - observer or not OfficialReplayConverter._event_card_visible(
            raw, observer
        ):
            return ()
        pairs = (
            ("cardId", "serial"),
            ("cardIdTarget", "serialTarget"),
            ("cardIdBefore", "serialBefore"),
            ("cardIdAfter", "serialAfter"),
            ("cardIdActive", "serialActive"),
            ("cardIdBench", "serialBench"),
        )
        result: list[RevealTarget] = []
        for card_key, serial_key in pairs:
            if card_key not in raw or serial_key not in raw:
                continue
            card_id = _safe_int(raw[card_key], -1)
            serial = _safe_int(raw[serial_key], -1)
            if card_id < 0 or serial < 0 or serial in already_revealed:
                continue
            already_revealed.add(serial)
            result.append(
                RevealTarget(
                    opaque_card_token=_fingerprint(
                        {
                            "replay": replay_sha256,
                            "owner": owner,
                            "private_serial": serial,
                        }
                    ),
                    opponent_player=owner,
                    card_id=card_id,
                    revealed_at_frame=frame_index,
                    may_supervise_prefix_through=frame_index - 1,
                    source_event_type=str(raw.get("type", "Unknown")),
                )
            )
        return tuple(result)

    @staticmethod
    def audit(
        frames: tuple[PublicReplayFrame, ...],
        *,
        unknown_event_types: tuple[str, ...] = (),
    ) -> LeakageAuditReport:
        violations: list[str] = []
        for frame in frames:
            if frame.players[1 - frame.observer_player].known_hand_card_ids:
                violations.append(f"frame {frame.frame_index}: opponent hand identity retained")
            if frame.acting_player != frame.observer_player and frame.selection is not None:
                violations.append(f"frame {frame.frame_index}: opponent selection retained")
            encoded = frame.to_dict()

            def walk(value: Any, path: str = "") -> None:
                if isinstance(value, Mapping):
                    for name, child in value.items():
                        child_path = f"{path}.{name}" if path else str(name)
                        if "serial" in str(name).lower():
                            violations.append(
                                f"frame {frame.frame_index}: forbidden serial at {child_path}"
                            )
                        if name in {"hand", "deck", "prize", "looking"}:
                            violations.append(
                                f"frame {frame.frame_index}: raw hidden zone at {child_path}"
                            )
                        walk(child, child_path)
                elif isinstance(value, (list, tuple)):
                    for index, child in enumerate(value):
                        walk(child, f"{path}[{index}]")

            walk(encoded)
        unique_violations = tuple(sorted(set(violations)))
        return LeakageAuditReport(
            passed=not unique_violations,
            frame_count=len(frames),
            decision_frame_count=sum(frame.selection is not None for frame in frames),
            redacted_event_count=sum(
                event.card_identity_redacted for frame in frames for event in frame.events
            ),
            unknown_event_types=tuple(sorted(set(unknown_event_types))),
            violations=unique_violations,
        )


def build_public_replay_dataset(
    converted: Iterable[PublicReplayEpisode],
) -> PublicReplayDataset:
    episodes = tuple(
        sorted(
            converted,
            key=lambda item: (
                item.provenance.match_id,
                item.provenance.observer_player,
            ),
        )
    )
    return PublicReplayDataset(episodes)
