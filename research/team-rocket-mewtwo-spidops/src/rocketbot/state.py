"""Immutable game-state and turn-history schemas.

These objects represent simulator truth. Phase 2 observations and probabilistic
beliefs should wrap this schema rather than inserting guesses into it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Iterator, Mapping, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None


class PlayerId(str, Enum):
    SELF = "self"
    OPPONENT = "opponent"

    @property
    def other(self) -> "PlayerId":
        return PlayerId.OPPONENT if self is PlayerId.SELF else PlayerId.SELF


class TurnPhase(str, Enum):
    SETUP = "setup"
    DRAW = "draw"
    MAIN = "main"
    ATTACK = "attack"
    BETWEEN_TURNS = "between_turns"
    GAME_OVER = "game_over"


class SpecialCondition(str, Enum):
    ASLEEP = "asleep"
    BURNED = "burned"
    CONFUSED = "confused"
    PARALYZED = "paralyzed"
    POISONED = "poisoned"


class ActionKind(str, Enum):
    DRAW = "draw"
    PLAY_POKEMON = "play_pokemon"
    EVOLVE = "evolve"
    ATTACH_ENERGY = "attach_energy"
    PLAY_TRAINER = "play_trainer"
    USE_ABILITY = "use_ability"
    RETREAT = "retreat"
    ATTACK = "attack"
    RESOLVE_EFFECT = "resolve_effect"
    END_TURN = "end_turn"
    CONCEDE = "concede"


@dataclass(frozen=True, slots=True)
class CardInstance:
    """One physical card. Hidden cards may have no public name."""

    instance_id: str
    card_name: str | None
    owner: PlayerId
    revealed: bool = True

    def __post_init__(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        if self.revealed and not self.card_name:
            raise ValueError("a revealed card must have a card_name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "card_name": self.card_name,
            "owner": self.owner.value,
            "revealed": self.revealed,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CardInstance":
        name = raw.get("card_name")
        return cls(
            instance_id=str(raw["instance_id"]),
            card_name=None if name is None else str(name),
            owner=PlayerId(str(raw["owner"])),
            revealed=bool(raw["revealed"]),
        )


@dataclass(frozen=True, slots=True)
class PokemonState:
    """A Pokemon in play, including its stack and attached resources."""

    top_card: CardInstance
    max_hp: int
    damage: int = 0
    evolution_stack: tuple[CardInstance, ...] = ()
    attached_energy: tuple[CardInstance, ...] = ()
    tool: CardInstance | None = None
    special_conditions: tuple[SpecialCondition, ...] = ()
    effect_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_hp <= 0:
            raise ValueError("max_hp must be positive")
        if self.damage < 0:
            raise ValueError("damage cannot be negative")
        if len(set(self.special_conditions)) != len(self.special_conditions):
            raise ValueError("special_conditions cannot contain duplicates")

        owner = self.top_card.owner
        if any(card.owner is not owner for card in self.iter_instances()):
            raise ValueError("all cards in a Pokemon stack must share an owner")

    @property
    def hp_remaining(self) -> int:
        return max(0, self.max_hp - self.damage)

    def iter_instances(self) -> Iterator[CardInstance]:
        yield self.top_card
        yield from self.evolution_stack
        yield from self.attached_energy
        if self.tool is not None:
            yield self.tool

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_card": self.top_card.to_dict(),
            "max_hp": self.max_hp,
            "damage": self.damage,
            "evolution_stack": [card.to_dict() for card in self.evolution_stack],
            "attached_energy": [card.to_dict() for card in self.attached_energy],
            "tool": None if self.tool is None else self.tool.to_dict(),
            "special_conditions": [condition.value for condition in self.special_conditions],
            "effect_tags": list(self.effect_tags),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PokemonState":
        tool = raw.get("tool")
        return cls(
            top_card=CardInstance.from_dict(raw["top_card"]),
            max_hp=int(raw["max_hp"]),
            damage=int(raw.get("damage", 0)),
            evolution_stack=tuple(
                CardInstance.from_dict(card) for card in raw.get("evolution_stack", [])
            ),
            attached_energy=tuple(
                CardInstance.from_dict(card) for card in raw.get("attached_energy", [])
            ),
            tool=None if tool is None else CardInstance.from_dict(tool),
            special_conditions=tuple(
                SpecialCondition(value) for value in raw.get("special_conditions", [])
            ),
            effect_tags=tuple(str(tag) for tag in raw.get("effect_tags", [])),
        )


@dataclass(frozen=True, slots=True)
class PlayerState:
    """All simulator zones and per-turn flags for one player."""

    player_id: PlayerId
    active: PokemonState | None
    bench: tuple[PokemonState, ...]
    hand: tuple[CardInstance, ...]
    deck: tuple[CardInstance, ...]
    prizes: tuple[CardInstance, ...]
    discard: tuple[CardInstance, ...] = ()
    lost_zone: tuple[CardInstance, ...] = ()
    supporter_used: bool = False
    energy_attachment_used: bool = False
    retreat_used: bool = False

    def __post_init__(self) -> None:
        if len(self.bench) > 5:
            raise ValueError("bench cannot contain more than five Pokemon")
        if len(self.prizes) > 6:
            raise ValueError("prize zone cannot contain more than six cards")
        if any(card.owner is not self.player_id for card in self.iter_instances()):
            raise ValueError("all cards in a PlayerState must belong to that player")

    @property
    def prizes_remaining(self) -> int:
        return len(self.prizes)

    def pokemon_in_play(self) -> tuple[PokemonState, ...]:
        return (() if self.active is None else (self.active,)) + self.bench

    def iter_instances(self) -> Iterator[CardInstance]:
        if self.active is not None:
            yield from self.active.iter_instances()
        for pokemon in self.bench:
            yield from pokemon.iter_instances()
        yield from self.hand
        yield from self.deck
        yield from self.prizes
        yield from self.discard
        yield from self.lost_zone

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id.value,
            "active": None if self.active is None else self.active.to_dict(),
            "bench": [pokemon.to_dict() for pokemon in self.bench],
            "hand": [card.to_dict() for card in self.hand],
            "deck": [card.to_dict() for card in self.deck],
            "prizes": [card.to_dict() for card in self.prizes],
            "discard": [card.to_dict() for card in self.discard],
            "lost_zone": [card.to_dict() for card in self.lost_zone],
            "supporter_used": self.supporter_used,
            "energy_attachment_used": self.energy_attachment_used,
            "retreat_used": self.retreat_used,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlayerState":
        active = raw.get("active")
        return cls(
            player_id=PlayerId(str(raw["player_id"])),
            active=None if active is None else PokemonState.from_dict(active),
            bench=tuple(PokemonState.from_dict(item) for item in raw.get("bench", [])),
            hand=tuple(CardInstance.from_dict(card) for card in raw.get("hand", [])),
            deck=tuple(CardInstance.from_dict(card) for card in raw.get("deck", [])),
            prizes=tuple(CardInstance.from_dict(card) for card in raw.get("prizes", [])),
            discard=tuple(CardInstance.from_dict(card) for card in raw.get("discard", [])),
            lost_zone=tuple(
                CardInstance.from_dict(card) for card in raw.get("lost_zone", [])
            ),
            supporter_used=bool(raw.get("supporter_used", False)),
            energy_attachment_used=bool(raw.get("energy_attachment_used", False)),
            retreat_used=bool(raw.get("retreat_used", False)),
        )


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete state at one decision boundary."""

    match_id: str
    turn_number: int
    acting_player: PlayerId
    phase: TurnPhase
    players: tuple[PlayerState, PlayerState]
    stadium: CardInstance | None = None
    winner: PlayerId | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id must be non-empty")
        if self.turn_number < 0:
            raise ValueError("turn_number cannot be negative")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if {player.player_id for player in self.players} != set(PlayerId):
            raise ValueError("GameState must contain exactly self and opponent")

        instances = [
            card
            for player in self.players
            for card in player.iter_instances()
        ]
        if self.stadium is not None:
            instances.append(self.stadium)
        instance_ids = [card.instance_id for card in instances]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("card instance IDs must be globally unique")

    def player(self, player_id: PlayerId) -> PlayerState:
        return next(player for player in self.players if player.player_id is player_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "turn_number": self.turn_number,
            "acting_player": self.acting_player.value,
            "phase": self.phase.value,
            "players": [player.to_dict() for player in self.players],
            "stadium": None if self.stadium is None else self.stadium.to_dict(),
            "winner": None if self.winner is None else self.winner.value,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GameState":
        stadium = raw.get("stadium")
        winner = raw.get("winner")
        players = tuple(PlayerState.from_dict(item) for item in raw["players"])
        if len(players) != 2:
            raise ValueError("serialized GameState must contain two players")
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            match_id=str(raw["match_id"]),
            turn_number=int(raw["turn_number"]),
            acting_player=PlayerId(str(raw["acting_player"])),
            phase=TurnPhase(str(raw["phase"])),
            players=(players[0], players[1]),
            stadium=None if stadium is None else CardInstance.from_dict(stadium),
            winner=None if winner is None else PlayerId(str(winner)),
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
    def from_json(cls, encoded: str) -> "GameState":
        return cls.from_dict(json.loads(encoded))

    @property
    def digest(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GameAction:
    """One resolved semantic action in a turn record."""

    sequence_index: int
    actor: PlayerId
    kind: ActionKind
    source_instance_id: str | None = None
    target_instance_id: str | None = None
    card_name: str | None = None
    payload: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        if self.sequence_index < 0:
            raise ValueError("sequence_index cannot be negative")
        keys = [key for key, _ in self.payload]
        if len(keys) != len(set(keys)):
            raise ValueError("action payload keys must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "actor": self.actor.value,
            "kind": self.kind.value,
            "source_instance_id": self.source_instance_id,
            "target_instance_id": self.target_instance_id,
            "card_name": self.card_name,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GameAction":
        payload = raw.get("payload", {})
        return cls(
            sequence_index=int(raw["sequence_index"]),
            actor=PlayerId(str(raw["actor"])),
            kind=ActionKind(str(raw["kind"])),
            source_instance_id=raw.get("source_instance_id"),
            target_instance_id=raw.get("target_instance_id"),
            card_name=raw.get("card_name"),
            payload=tuple(sorted((str(key), value) for key, value in payload.items())),
        )


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """Ordered actions plus digest links between two decision states."""

    turn_number: int
    acting_player: PlayerId
    state_before_digest: str
    actions: tuple[GameAction, ...]
    state_after_digest: str

    def __post_init__(self) -> None:
        expected = list(range(len(self.actions)))
        actual = [action.sequence_index for action in self.actions]
        if actual != expected:
            raise ValueError("turn action sequence indices must be contiguous from zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "acting_player": self.acting_player.value,
            "state_before_digest": self.state_before_digest,
            "actions": [action.to_dict() for action in self.actions],
            "state_after_digest": self.state_after_digest,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TurnRecord":
        return cls(
            turn_number=int(raw["turn_number"]),
            acting_player=PlayerId(str(raw["acting_player"])),
            state_before_digest=str(raw["state_before_digest"]),
            actions=tuple(GameAction.from_dict(item) for item in raw["actions"]),
            state_after_digest=str(raw["state_after_digest"]),
        )


@dataclass(frozen=True, slots=True)
class MatchTrace:
    """A digest-checked, serializable chain of complete turns."""

    initial_state: GameState
    current_state: GameState
    turns: tuple[TurnRecord, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.initial_state.match_id != self.current_state.match_id:
            raise ValueError("trace states must belong to the same match")
        expected_digest = self.initial_state.digest
        for turn in self.turns:
            if turn.state_before_digest != expected_digest:
                raise ValueError("turn history digest chain is broken")
            expected_digest = turn.state_after_digest
        if expected_digest != self.current_state.digest:
            raise ValueError("current state does not match the final trace digest")

    def append_turn(self, turn: TurnRecord, new_state: GameState) -> "MatchTrace":
        if turn.state_before_digest != self.current_state.digest:
            raise ValueError("turn does not start at the trace's current state")
        if turn.state_after_digest != new_state.digest:
            raise ValueError("turn does not end at the supplied new state")
        if new_state.match_id != self.current_state.match_id:
            raise ValueError("new state belongs to another match")
        return replace(self, current_state=new_state, turns=self.turns + (turn,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "initial_state": self.initial_state.to_dict(),
            "current_state": self.current_state.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MatchTrace":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            initial_state=GameState.from_dict(raw["initial_state"]),
            current_state=GameState.from_dict(raw["current_state"]),
            turns=tuple(TurnRecord.from_dict(item) for item in raw.get("turns", [])),
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
    def from_json(cls, encoded: str) -> "MatchTrace":
        return cls.from_dict(json.loads(encoded))
