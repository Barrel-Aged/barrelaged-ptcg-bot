"""Validated, reproducible deck configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_DECK_RESOURCE = "naic_2026_juho_kallama_mewtwo_spidops.json"


class CardCategory(str, Enum):
    POKEMON = "pokemon"
    TRAINER = "trainer"
    ENERGY = "energy"


@dataclass(frozen=True, slots=True)
class DeckEntry:
    """One named card and its multiplicity in a deck list."""

    name: str
    count: int
    category: CardCategory

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("deck entry name must be non-empty")
        if self.count <= 0:
            raise ValueError(f"count for {self.name!r} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "name": self.name,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DeckEntry":
        return cls(
            name=str(raw["name"]),
            count=int(raw["count"]),
            category=CardCategory(str(raw["category"])),
        )


@dataclass(frozen=True, slots=True)
class DeckConfig:
    """Versioned deck list with canonical serialization and integrity checks."""

    schema_version: int
    deck_name: str
    player: str
    event: str
    placing: int
    provenance: str
    entries: tuple[DeckEntry, ...]

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if not self.deck_name.strip():
            raise ValueError("deck_name must be non-empty")
        if self.placing <= 0:
            raise ValueError("placing must be positive")
        if not self.entries:
            raise ValueError("deck must contain entries")

        normalized_names = [entry.name.casefold() for entry in self.entries]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("deck contains duplicate card names")
        if self.total_cards != 60:
            raise ValueError(f"deck must contain exactly 60 cards, got {self.total_cards}")

    @property
    def total_cards(self) -> int:
        return sum(entry.count for entry in self.entries)

    @property
    def category_counts(self) -> dict[CardCategory, int]:
        return {
            category: sum(
                entry.count for entry in self.entries if entry.category is category
            )
            for category in CardCategory
        }

    @property
    def card_counts(self) -> dict[str, int]:
        return {entry.name: entry.count for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "deck_name": self.deck_name,
            "player": self.player,
            "event": self.event,
            "placing": self.placing,
            "provenance": self.provenance,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DeckConfig":
        return cls(
            schema_version=int(raw["schema_version"]),
            deck_name=str(raw["deck_name"]),
            player=str(raw["player"]),
            event=str(raw["event"]),
            placing=int(raw["placing"]),
            provenance=str(raw["provenance"]),
            entries=tuple(DeckEntry.from_dict(entry) for entry in raw["entries"]),
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
    def from_json(cls, encoded: str) -> "DeckConfig":
        return cls.from_dict(json.loads(encoded))

    @property
    def fingerprint(self) -> str:
        """Stable content fingerprint for trajectories and checkpoints."""

        return sha256(self.to_json().encode("utf-8")).hexdigest()


def load_deck(path: str | Path) -> DeckConfig:
    """Load and validate a deck from a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return DeckConfig.from_dict(json.load(handle))


def load_default_deck() -> DeckConfig:
    """Load the package's immutable NAIC 2026 starting list."""

    resource = files("rocketbot.data").joinpath(DEFAULT_DECK_RESOURCE)
    with resource.open("r", encoding="utf-8") as handle:
        return DeckConfig.from_dict(json.load(handle))
