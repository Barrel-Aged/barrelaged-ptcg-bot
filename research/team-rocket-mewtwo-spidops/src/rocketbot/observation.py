"""Agent-visible observations separated from omniscient simulator truth."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

import torch

from .state import (
    ActionKind,
    CardInstance,
    GameAction,
    GameState,
    MatchTrace,
    PlayerId,
    PlayerState,
    PokemonState,
)


def _hidden(card: CardInstance) -> CardInstance:
    return replace(card, card_name=None, revealed=False)


def _public(card: CardInstance) -> CardInstance:
    if card.card_name is None:
        return _hidden(card)
    return replace(card, revealed=True)


def _public_pokemon(pokemon: PokemonState) -> PokemonState:
    return replace(
        pokemon,
        top_card=_public(pokemon.top_card),
        evolution_stack=tuple(_public(card) for card in pokemon.evolution_stack),
        attached_energy=tuple(_public(card) for card in pokemon.attached_energy),
        tool=None if pokemon.tool is None else _public(pokemon.tool),
    )


def redact_state_for_observer(state: GameState, observer: PlayerId) -> GameState:
    """Return a structurally complete state with private identities removed.

    Both decks and prize cards are unknown by identity. Only the observer's hand
    is exposed; all in-play, discard, lost-zone, and stadium cards are public.
    Counts and instance IDs remain available so zone transitions can be tracked
    without claiming a hidden card identity.
    """

    redacted_players: list[PlayerState] = []
    for player in state.players:
        own_hand = player.player_id is observer
        redacted_players.append(
            replace(
                player,
                active=None if player.active is None else _public_pokemon(player.active),
                bench=tuple(_public_pokemon(pokemon) for pokemon in player.bench),
                hand=tuple(
                    _public(card) if own_hand else _hidden(card) for card in player.hand
                ),
                deck=tuple(_hidden(card) for card in player.deck),
                prizes=tuple(_hidden(card) for card in player.prizes),
                discard=tuple(_public(card) for card in player.discard),
                lost_zone=tuple(_public(card) for card in player.lost_zone),
            )
        )
    return replace(
        state,
        players=(redacted_players[0], redacted_players[1]),
        stadium=None if state.stadium is None else _public(state.stadium),
    )


@dataclass(frozen=True, slots=True)
class ObservedAction:
    """Public portion of an action, stripped of arbitrary simulator payloads."""

    turn_number: int
    sequence_index: int
    actor: PlayerId
    kind: ActionKind
    card_name: str | None = None
    source_instance_id: str | None = None
    target_instance_id: str | None = None

    @classmethod
    def from_game_action(
        cls,
        action: GameAction,
        *,
        turn_number: int,
        observer: PlayerId,
    ) -> "ObservedAction":
        private_draw = action.kind is ActionKind.DRAW and action.actor is not observer
        return cls(
            turn_number=turn_number,
            sequence_index=action.sequence_index,
            actor=action.actor,
            kind=action.kind,
            card_name=None if private_draw else action.card_name,
            source_instance_id=None if private_draw else action.source_instance_id,
            target_instance_id=None if private_draw else action.target_instance_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "sequence_index": self.sequence_index,
            "actor": self.actor.value,
            "kind": self.kind.value,
            "card_name": self.card_name,
            "source_instance_id": self.source_instance_id,
            "target_instance_id": self.target_instance_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObservedAction":
        return cls(
            turn_number=int(raw["turn_number"]),
            sequence_index=int(raw["sequence_index"]),
            actor=PlayerId(str(raw["actor"])),
            kind=ActionKind(str(raw["kind"])),
            card_name=None if raw.get("card_name") is None else str(raw["card_name"]),
            source_instance_id=(
                None
                if raw.get("source_instance_id") is None
                else str(raw["source_instance_id"])
            ),
            target_instance_id=(
                None
                if raw.get("target_instance_id") is None
                else str(raw["target_instance_id"])
            ),
        )


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """Versioned public state and action history from one player's view."""

    observer: PlayerId
    public_state: GameState
    actions: tuple[ObservedAction, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.public_state.player(self.observer).player_id is not self.observer:
            raise ValueError("observer must be present in public_state")
        opponent = self.public_state.player(self.observer.other)
        private_zones = opponent.hand + opponent.deck + opponent.prizes
        if any(card.card_name is not None or card.revealed for card in private_zones):
            raise ValueError("opponent private identities leaked into AgentObservation")
        own = self.public_state.player(self.observer)
        if any(card.card_name is not None or card.revealed for card in own.deck + own.prizes):
            raise ValueError("own deck or prize identities leaked into AgentObservation")

    @classmethod
    def from_trace(
        cls,
        trace: MatchTrace,
        *,
        observer: PlayerId = PlayerId.SELF,
    ) -> "AgentObservation":
        observed = tuple(
            ObservedAction.from_game_action(
                action,
                turn_number=turn.turn_number,
                observer=observer,
            )
            for turn in trace.turns
            for action in turn.actions
        )
        return cls(
            observer=observer,
            public_state=redact_state_for_observer(trace.current_state, observer),
            actions=observed,
        )

    @classmethod
    def from_state(
        cls,
        state: GameState,
        *,
        observer: PlayerId = PlayerId.SELF,
        actions: Iterable[ObservedAction] = (),
    ) -> "AgentObservation":
        return cls(
            observer=observer,
            public_state=redact_state_for_observer(state, observer),
            actions=tuple(actions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observer": self.observer.value,
            "public_state": self.public_state.to_dict(),
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentObservation":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            observer=PlayerId(str(raw["observer"])),
            public_state=GameState.from_dict(raw["public_state"]),
            actions=tuple(ObservedAction.from_dict(item) for item in raw.get("actions", [])),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, encoded: str) -> "AgentObservation":
        return cls.from_dict(json.loads(encoded))

    @property
    def digest(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()


ACTION_HISTORY_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"kind_{kind.value}" for kind in ActionKind
) + (
    "actor_is_observer",
    "actor_is_opponent",
    "card_name_known",
    "source_known",
    "target_known",
    "turn_progress",
    "turn_recency",
    "sequence_progress",
)


class ActionHistoryTensorizer:
    """Tensorize public action patterns as `[batch, time, features]`."""

    feature_names = ACTION_HISTORY_FEATURE_NAMES
    feature_dim = len(ACTION_HISTORY_FEATURE_NAMES)

    def tensorize_action(
        self,
        action: ObservedAction,
        *,
        observation: AgentObservation,
    ) -> tuple[float, ...]:
        kind_one_hot = tuple(float(action.kind is kind) for kind in ActionKind)
        current_turn = observation.public_state.turn_number
        extras = (
            float(action.actor is observation.observer),
            float(action.actor is observation.observer.other),
            float(action.card_name is not None),
            float(action.source_instance_id is not None),
            float(action.target_instance_id is not None),
            min(action.turn_number / 30.0, 1.0),
            min(max(current_turn - action.turn_number, 0) / 30.0, 1.0),
            min(action.sequence_index / 20.0, 1.0),
        )
        values = kind_one_hot + extras
        if len(values) != self.feature_dim:
            raise RuntimeError("action feature implementation does not match schema")
        return values

    def __call__(
        self,
        observations: Iterable[AgentObservation],
        *,
        max_events: int = 32,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        observation_rows = list(observations)
        if not observation_rows:
            raise ValueError("at least one observation is required")
        batch = torch.zeros(
            len(observation_rows),
            max_events,
            self.feature_dim,
            dtype=dtype,
            device=device,
        )
        mask = torch.zeros(
            len(observation_rows),
            max_events,
            dtype=torch.bool,
            device=device,
        )
        for batch_index, observation in enumerate(observation_rows):
            recent = observation.actions[-max_events:]
            for time_index, action in enumerate(recent):
                batch[batch_index, time_index] = torch.tensor(
                    self.tensorize_action(action, observation=observation),
                    dtype=dtype,
                    device=device,
                )
                mask[batch_index, time_index] = True
        return batch, mask
