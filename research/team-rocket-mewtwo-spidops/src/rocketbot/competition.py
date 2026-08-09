"""Phase 6 integration-baseline boundary for the official Kaggle simulator.

The official engine sends plain dictionaries and expects a ``list[int]`` of
option indices.  This module deliberately uses only the Python standard
library so it can be copied into a local compatibility archive without PyTorch.  It
keeps the research representation inspectable: every decision records a
persistent plan, six strategic values, state-conditioned weights, an opponent
archetype posterior, search advice, latency, and any failsafe use.

The heuristic scorer and sampled-rollout advisor are preserved baselines, not
the final strategic architecture. The neural game-theoretic target and its
official bridge are defined separately in ``game_theory.py``.

The exact NAIC 2026 list is immutable.  The current Kaggle card pool ends at
Perfect Order and does not contain Prism Tower (Chaos Rising 80), so exact
deck resolution fails closed.  A clearly labelled Watchtower compatibility
profile is available for local engine integration, but is never substituted
silently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from hashlib import sha256
import json
import math
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence


COMPETITION_SCHEMA_VERSION = 1


class UnsupportedCardPoolError(ValueError):
    """Raised when an exact deck cannot be represented by an engine catalog."""


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class SelectContext(IntEnum):
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    TO_FIELD = 6
    TO_HAND = 7
    DISCARD = 8
    TO_DECK = 9
    TO_DECK_BOTTOM = 10
    TO_PRIZE = 11
    NOT_MOVE = 12
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    EVOLVES_FROM = 18
    EVOLVES_TO = 19
    DEVOLVE = 20
    ATTACH_FROM = 21
    ATTACH_TO = 22
    DETACH_FROM = 23
    LOOK = 24
    EFFECT_TARGET = 25
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    SWITCH_ENERGY_CARD = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    TO_HAND_ENERGY = 31
    TO_DECK_ENERGY = 32
    SWITCH_ENERGY = 33
    SKILL_ORDER = 34
    ATTACK = 35
    DISABLE_ATTACK = 36
    EVOLVE = 37
    DRAW_COUNT = 38
    DAMAGE_COUNTER_COUNT = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43
    FIRST_EFFECT = 44
    MORE_DEVOLVE = 45
    COIN_HEAD = 46
    AFFECT_SPECIAL_CONDITION = 47
    RECOVER_SPECIAL_CONDITION = 48


class OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class AblationMode(str, Enum):
    """Competition policies used for matched component ablations."""

    POLICY_ONLY = "policy_only"
    STRATEGIC = "strategic"
    BELIEF_STRATEGIC = "belief_strategic"
    FULL = "full"


class StrategicPlan(str, Enum):
    BOARD_SETUP = "board_setup"
    ENERGY_ENGINE = "energy_engine"
    PRIZE_PUSH = "prize_push"
    RECOVERY = "recovery"
    CONTROL = "control"


STRATEGIC_AXIS_NAMES: tuple[str, ...] = (
    "setup",
    "tempo",
    "resource_efficiency",
    "prize_control",
    "board_control",
    "optionality",
)


# The name/count representation intentionally mirrors the immutable package
# deck rather than treating the compatibility profile as the source of truth.
EXACT_DECK_COUNTS: tuple[tuple[str, int], ...] = (
    ("Team Rocket's Mewtwo ex", 2),
    ("Team Rocket's Tarountula", 4),
    ("Team Rocket's Spidops", 4),
    ("Team Rocket's Articuno", 2),
    ("Team Rocket's Mimikyu", 2),
    ("Lillie's Clefairy ex", 1),
    ("Lillie's Determination", 4),
    ("Team Rocket's Ariana", 4),
    ("Team Rocket's Giovanni", 3),
    ("Team Rocket's Proton", 2),
    ("Team Rocket's Petrel", 1),
    ("Team Rocket's Transceiver", 4),
    ("Ultra Ball", 4),
    ("Night Stretcher", 3),
    ("Bug Catching Set", 2),
    ("Lucky Helmet", 2),
    ("Prism Tower", 2),
    ("Team Rocket's Factory", 1),
    ("Energy Switch", 1),
    ("Maximum Belt", 1),
    ("Team Rocket's Energy", 4),
    ("Basic Grass Energy", 6),
    ("Basic Psychic Energy", 1),
)


# Canonical IDs exposed by the 2026-07-15 competition engine.  Reprints with
# identical implemented rules share a canonical engine ID (for example Ultra
# Ball).  Prism Tower is deliberately None because it is absent from the pool.
OFFICIAL_CARD_IDS: Mapping[str, int | None] = {
    "Team Rocket's Mewtwo ex": 431,
    "Team Rocket's Tarountula": 400,
    "Team Rocket's Spidops": 401,
    "Team Rocket's Articuno": 414,
    "Team Rocket's Mimikyu": 434,
    "Lillie's Clefairy ex": 272,
    "Lillie's Determination": 1227,
    "Team Rocket's Ariana": 1216,
    "Team Rocket's Giovanni": 1218,
    "Team Rocket's Proton": 1220,
    "Team Rocket's Petrel": 1219,
    "Team Rocket's Transceiver": 1134,
    "Ultra Ball": 1121,
    "Night Stretcher": 1097,
    "Bug Catching Set": 1094,
    "Lucky Helmet": 1156,
    "Prism Tower": None,
    "Team Rocket's Factory": 1257,
    "Energy Switch": 1116,
    "Maximum Belt": 1158,
    "Team Rocket's Energy": 15,
    "Basic Grass Energy": 1,
    "Basic Psychic Energy": 5,
    "Team Rocket's Watchtower": 1256,
}


KAGGLE_WATCHTOWER_SUBSTITUTION: Mapping[str, str] = {
    "Prism Tower": "Team Rocket's Watchtower"
}


CARD_NAMES: Mapping[int, str] = {
    card_id: name
    for name, card_id in OFFICIAL_CARD_IDS.items()
    if card_id is not None
}


TEAM_ROCKET_POKEMON = frozenset({400, 401, 414, 431, 434})
BASIC_ROCKET_POKEMON = frozenset({400, 414, 431, 434})
BASIC_POKEMON = frozenset({272, 400, 414, 431, 434})
POKEMON = frozenset({272, 400, 401, 414, 431, 434})
ENERGY = frozenset({1, 5, 15})
SUPPORTERS = frozenset({1216, 1218, 1219, 1220, 1227})
STADIUMS = frozenset({1256, 1257})
TOOLS = frozenset({1156, 1158})


ATTACK_DAMAGE: Mapping[int, int] = {
    371: 20,   # Full Moon Rondo, plus 20 per Benched Pokemon.
    559: 30,   # Take Down.
    560: 0,    # Rocket Rush, 30 per Team Rocket Pokemon in play.
    583: 60,   # Dark Frost, plus 60 with Team Rocket Energy.
    608: 160,  # Erasure Ball, plus up to 120 from discarded Energy.
    612: 0,    # Gemstone Mimicry is matchup dependent.
}


ARCHETYPE_SIGNATURES: Mapping[str, frozenset[int]] = {
    "rocket_mewtwo": frozenset({400, 401, 414, 431, 434}),
    "abomasnow": frozenset({721, 722, 723}),
    "alakazam": frozenset({109, 245, 741, 742, 743}),
    "starmie": frozenset({360, 361, 1030, 1031}),
    "dragapult": frozenset({119, 120, 121}),
    "crustle": frozenset({344, 345, 532, 533}),
    "grimmsnarl": frozenset({646, 647, 648}),
    "lucario": frozenset({677, 678}),
}


def _canonical_name(name: str) -> str:
    return (
        name.replace("’", "'")
        .replace("‘", "'")
        .replace("{G}", "Grass")
        .replace("{P}", "Psychic")
        .casefold()
        .strip()
    )


@dataclass(frozen=True, slots=True)
class DeckSubstitution:
    original_name: str
    replacement_name: str
    count: int


@dataclass(frozen=True, slots=True)
class DeckResolution:
    """Auditable result of mapping the immutable deck into an engine pool."""

    card_ids: tuple[int, ...]
    unresolved: tuple[str, ...]
    substitutions: tuple[DeckSubstitution, ...]
    exact: bool
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.exact and (self.unresolved or self.substitutions):
            raise ValueError("an exact resolution cannot contain gaps or substitutions")
        if not self.unresolved and len(self.card_ids) != 60:
            raise ValueError("a fully resolved competition deck must contain 60 IDs")

    @property
    def playable(self) -> bool:
        return not self.unresolved and len(self.card_ids) == 60

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema": COMPETITION_SCHEMA_VERSION,
            "card_ids": self.card_ids,
            "unresolved": self.unresolved,
            "substitutions": [asdict(item) for item in self.substitutions],
            "exact": self.exact,
            "source_fingerprint": self.source_fingerprint,
        }
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COMPETITION_SCHEMA_VERSION,
            "card_ids": list(self.card_ids),
            "unresolved": list(self.unresolved),
            "substitutions": [asdict(item) for item in self.substitutions],
            "exact": self.exact,
            "playable": self.playable,
            "source_fingerprint": self.source_fingerprint,
            "fingerprint": self.fingerprint,
        }


def exact_deck_fingerprint() -> str:
    encoded = json.dumps(EXACT_DECK_COUNTS, separators=(",", ":"), ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def resolve_official_deck(
    *,
    substitutions: Mapping[str, str] | None = None,
    fail_on_unresolved: bool = False,
) -> DeckResolution:
    """Resolve the exact list, with substitutions only when explicitly supplied."""

    supplied = dict(substitutions or {})
    unknown_sources = set(supplied).difference(name for name, _ in EXACT_DECK_COUNTS)
    if unknown_sources:
        raise ValueError(f"substitution sources are not in the exact deck: {unknown_sources}")

    card_ids: list[int] = []
    unresolved: list[str] = []
    applied: list[DeckSubstitution] = []
    for original_name, count in EXACT_DECK_COUNTS:
        resolved_name = supplied.get(original_name, original_name)
        card_id = OFFICIAL_CARD_IDS.get(resolved_name)
        if card_id is None:
            unresolved.append(original_name)
            continue
        card_ids.extend([card_id] * count)
        if resolved_name != original_name:
            applied.append(DeckSubstitution(original_name, resolved_name, count))

    result = DeckResolution(
        card_ids=tuple(card_ids),
        unresolved=tuple(unresolved),
        substitutions=tuple(applied),
        exact=not unresolved and not applied,
        source_fingerprint=exact_deck_fingerprint(),
    )
    if fail_on_unresolved and not result.playable:
        missing = ", ".join(result.unresolved)
        raise UnsupportedCardPoolError(
            f"official Kaggle card pool cannot represent exact deck: {missing}"
        )
    return result


@dataclass(frozen=True, slots=True)
class SelectionContract:
    min_count: int
    max_count: int
    option_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.min_count <= self.max_count <= self.option_count:
            raise ValueError("invalid official selection bounds")

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any]) -> "SelectionContract":
        select = observation.get("select")
        if not isinstance(select, Mapping):
            raise ValueError("observation has no selection")
        options = select.get("option", ())
        return cls(
            min_count=int(select.get("minCount", 0)),
            max_count=int(select.get("maxCount", 0)),
            option_count=len(options),
        )

    def validate(self, selection: Sequence[int]) -> tuple[int, ...]:
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in selection):
            raise ValueError("selection must contain only integer option indices")
        normalized = tuple(selection)
        if not self.min_count <= len(normalized) <= self.max_count:
            raise ValueError("selection length violates official bounds")
        if len(set(normalized)) != len(normalized):
            raise ValueError("selection contains duplicate option indices")
        if any(value < 0 or value >= self.option_count for value in normalized):
            raise ValueError("selection contains an out-of-range option index")
        return normalized

    def fallback(self) -> tuple[int, ...]:
        """Return the lowest-index legal answer; always valid by construction."""

        return tuple(range(self.min_count))


@dataclass(frozen=True, slots=True)
class StrategicSnapshot:
    values: tuple[float, ...]
    weights: tuple[float, ...]
    aggregate: float

    def __post_init__(self) -> None:
        if len(self.values) != len(STRATEGIC_AXIS_NAMES):
            raise ValueError("strategic snapshot must have six values")
        if len(self.weights) != len(STRATEGIC_AXIS_NAMES):
            raise ValueError("strategic snapshot must have six weights")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-8):
            raise ValueError("strategic weights must sum to one")
        if any(not -1.0 <= value <= 1.0 for value in self.values):
            raise ValueError("strategic values must be in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(zip(STRATEGIC_AXIS_NAMES, self.values, strict=True)),
            "weights": dict(zip(STRATEGIC_AXIS_NAMES, self.weights, strict=True)),
            "aggregate": self.aggregate,
        }


@dataclass(frozen=True, slots=True)
class DecisionTelemetry:
    decision_index: int
    turn: int
    context: int
    plan: str
    selection: tuple[int, ...]
    policy_selection: tuple[int, ...]
    option_scores: tuple[float, ...]
    strategic: StrategicSnapshot
    archetype_posterior: tuple[tuple[str, float], ...]
    belief_entropy: float
    search_used: bool
    search_changed_action: bool
    search_bonus: tuple[tuple[int, float], ...]
    criticality: float
    latency_ms: float
    fallback_used: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["strategic"] = self.strategic.to_dict()
        result["archetype_posterior"] = dict(self.archetype_posterior)
        result["search_bonus"] = dict(self.search_bonus)
        return result


class SearchAdvisor(Protocol):
    """Bounded tactical look-ahead adapter implemented against a real engine."""

    def advise(
        self,
        observation: Mapping[str, Any],
        candidate_indices: Sequence[int],
        *,
        deadline: float,
        root_player: int,
    ) -> Mapping[int, float]: ...


@dataclass(slots=True)
class CompetitionAgentConfig:
    ablation: AblationMode = AblationMode.FULL
    decision_budget_ms: float = 75.0
    search_budget_ms: float = 35.0
    criticality_threshold: float = 0.55
    max_telemetry: int = 4096

    def __post_init__(self) -> None:
        if self.decision_budget_ms <= 0:
            raise ValueError("decision_budget_ms must be positive")
        if not 0 <= self.search_budget_ms <= self.decision_budget_ms:
            raise ValueError("search budget must fit within decision budget")
        if not 0 <= self.criticality_threshold <= 1:
            raise ValueError("criticality threshold must be in [0, 1]")
        if self.max_telemetry <= 0:
            raise ValueError("max_telemetry must be positive")


@dataclass(slots=True)
class CompetitionAgentMemory:
    match_index: int = 0
    decision_index: int = 0
    last_turn: int = -1
    plan: StrategicPlan = StrategicPlan.BOARD_SETUP
    revealed_opponent_ids: Counter[int] = field(default_factory=Counter)
    archetype_posterior: dict[str, float] = field(default_factory=dict)
    telemetry: list[DecisionTelemetry] = field(default_factory=list)
    fallback_count: int = 0
    search_count: int = 0
    search_change_count: int = 0
    max_latency_ms: float = 0.0

    def reset(self) -> None:
        self.match_index += 1
        self.decision_index = 0
        self.last_turn = -1
        self.plan = StrategicPlan.BOARD_SETUP
        self.revealed_opponent_ids.clear()
        count = len(ARCHETYPE_SIGNATURES)
        self.archetype_posterior = {
            name: 1.0 / count for name in ARCHETYPE_SIGNATURES
        }
        self.telemetry.clear()
        self.fallback_count = 0
        self.search_count = 0
        self.search_change_count = 0
        self.max_latency_ms = 0.0


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _state(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    current = observation.get("current")
    return current if isinstance(current, Mapping) else {}


def _players(observation: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    players = _state(observation).get("players", ())
    return players if isinstance(players, Sequence) else ()


def _player(observation: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    players = _players(observation)
    if 0 <= index < len(players) and isinstance(players[index], Mapping):
        return players[index]
    return {}


def _your_index(observation: Mapping[str, Any]) -> int:
    return _safe_int(_state(observation).get("yourIndex"), 0)


def _pokemon_cards(pokemon: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(pokemon, Mapping):
        return []
    cards: list[Mapping[str, Any]] = [pokemon]
    for key in ("preEvolution", "energyCards", "tools"):
        values = pokemon.get(key, ())
        if isinstance(values, Sequence):
            cards.extend(item for item in values if isinstance(item, Mapping))
    return cards


def _in_play(player: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    active = player.get("active", ())
    if isinstance(active, Sequence):
        result.extend(item for item in active if isinstance(item, Mapping))
    bench = player.get("bench", ())
    if isinstance(bench, Sequence):
        result.extend(item for item in bench if isinstance(item, Mapping))
    return result


def _card_id(card: Mapping[str, Any] | None) -> int | None:
    if not isinstance(card, Mapping):
        return None
    value = card.get("id")
    return _safe_int(value, -1) if value is not None else None


def _zone_card(
    observation: Mapping[str, Any],
    *,
    player_index: int,
    area: int | None,
    index: int | None,
) -> Mapping[str, Any] | None:
    if area is None or index is None or index < 0:
        return None
    select = observation.get("select", {})
    current = _state(observation)
    player = _player(observation, player_index)
    zone: Any = None
    if area == AreaType.DECK:
        zone = select.get("deck") if isinstance(select, Mapping) else None
    elif area == AreaType.HAND:
        zone = player.get("hand")
    elif area == AreaType.DISCARD:
        zone = player.get("discard")
    elif area == AreaType.ACTIVE:
        zone = player.get("active")
    elif area == AreaType.BENCH:
        zone = player.get("bench")
    elif area == AreaType.PRIZE:
        zone = player.get("prize")
    elif area == AreaType.STADIUM:
        zone = current.get("stadium")
    elif area == AreaType.LOOKING:
        zone = current.get("looking")
    if isinstance(zone, Sequence) and 0 <= index < len(zone):
        card = zone[index]
        return card if isinstance(card, Mapping) else None
    return None


def option_card_id(
    observation: Mapping[str, Any], option: Mapping[str, Any]
) -> int | None:
    """Resolve the card an official option acts with or points at."""

    option_type = _safe_int(option.get("type"), -1)
    your_index = _your_index(observation)
    if option_type == OptionType.PLAY:
        card = _zone_card(
            observation,
            player_index=your_index,
            area=AreaType.HAND,
            index=_safe_int(option.get("index"), -1),
        )
        return _card_id(card)
    if option_type in (OptionType.ATTACH, OptionType.EVOLVE):
        card = _zone_card(
            observation,
            player_index=your_index,
            area=_safe_int(option.get("area"), -1),
            index=_safe_int(option.get("index"), -1),
        )
        return _card_id(card)
    if option_type in (
        OptionType.CARD,
        OptionType.TOOL_CARD,
        OptionType.ENERGY_CARD,
        OptionType.ENERGY,
        OptionType.ABILITY,
        OptionType.DISCARD,
    ):
        player_index = _safe_int(option.get("playerIndex"), your_index)
        card = _zone_card(
            observation,
            player_index=player_index,
            area=_safe_int(option.get("area"), -1),
            index=_safe_int(option.get("index"), -1),
        )
        return _card_id(card)
    value = option.get("cardId")
    return _safe_int(value, -1) if value is not None else None


def option_target_pokemon_id(
    observation: Mapping[str, Any], option: Mapping[str, Any]
) -> int | None:
    your_index = _your_index(observation)
    if option.get("inPlayArea") is not None:
        card = _zone_card(
            observation,
            player_index=your_index,
            area=_safe_int(option.get("inPlayArea"), -1),
            index=_safe_int(option.get("inPlayIndex"), -1),
        )
        return _card_id(card)
    if _safe_int(option.get("type"), -1) == OptionType.CARD:
        card = _zone_card(
            observation,
            player_index=_safe_int(option.get("playerIndex"), your_index),
            area=_safe_int(option.get("area"), -1),
            index=_safe_int(option.get("index"), -1),
        )
        return _card_id(card)
    return None


def visible_card_ids(player: Mapping[str, Any]) -> Counter[int]:
    result: Counter[int] = Counter()
    for key in ("hand", "discard", "prize"):
        cards = player.get(key, ())
        if isinstance(cards, Sequence):
            for card in cards:
                card_id = _card_id(card if isinstance(card, Mapping) else None)
                if card_id is not None and card_id >= 0:
                    result[card_id] += 1
    for pokemon in _in_play(player):
        for card in _pokemon_cards(pokemon):
            card_id = _card_id(card)
            if card_id is not None and card_id >= 0:
                result[card_id] += 1
    return result


class Phase6CompetitionAgent:
    """Deterministic, auditable policy over the official simulator contract."""

    def __init__(
        self,
        deck: DeckResolution,
        *,
        config: CompetitionAgentConfig | None = None,
        search_advisor: SearchAdvisor | None = None,
    ) -> None:
        if not deck.playable:
            raise UnsupportedCardPoolError(
                "competition agent requires a fully resolved 60-card deck"
            )
        self.deck = deck
        self.config = config or CompetitionAgentConfig()
        self.search_advisor = search_advisor
        self.memory = CompetitionAgentMemory()
        self.memory.reset()

    def reset(self) -> None:
        self.memory.reset()

    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        return self.choose(observation)

    def choose(self, observation: Mapping[str, Any]) -> list[int]:
        started = perf_counter()
        select = observation.get("select")
        if select is None:
            self.reset()
            return list(self.deck.card_ids)

        contract = SelectionContract.from_observation(observation)
        fallback = contract.fallback()
        fallback_used = False
        error: str | None = None
        policy_selection = fallback
        final_selection = fallback
        scores: tuple[float, ...] = ()
        search_bonus: dict[int, float] = {}
        criticality = 0.0
        search_used = False
        try:
            self._update_memory(observation)
            snapshot = self._strategic_snapshot(observation)
            options = select.get("option", ())
            scores = tuple(
                self._score_option(observation, option, index)
                for index, option in enumerate(options)
            )
            policy_selection = self._ranked_selection(observation, scores)
            final_selection = policy_selection
            criticality = self._criticality(observation, scores)

            should_search = (
                self.config.ablation is AblationMode.FULL
                and self.search_advisor is not None
                and criticality >= self.config.criticality_threshold
                and perf_counter() - started
                < (self.config.decision_budget_ms - self.config.search_budget_ms) / 1000.0
            )
            if should_search:
                search_used = True
                search_deadline = min(
                    started + self.config.decision_budget_ms / 1000.0,
                    perf_counter() + self.config.search_budget_ms / 1000.0,
                )
                candidates = tuple(
                    sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:6]
                )
                search_bonus = {
                    int(index): float(value)
                    for index, value in self.search_advisor.advise(
                        observation,
                        candidates,
                        deadline=search_deadline,
                        root_player=_your_index(observation),
                    ).items()
                    if index in candidates and math.isfinite(float(value))
                }
                if search_bonus:
                    combined = tuple(
                        score + search_bonus.get(index, 0.0)
                        for index, score in enumerate(scores)
                    )
                    final_selection = self._ranked_selection(observation, combined)

            final_selection = contract.validate(final_selection)
        except Exception as exc:  # Competition safety boundary is intentionally broad.
            fallback_used = True
            error = f"{type(exc).__name__}: {exc}"
            final_selection = contract.validate(fallback)
            if not scores:
                scores = tuple(0.0 for _ in range(contract.option_count))
            snapshot = self._strategic_snapshot(observation)

        latency_ms = (perf_counter() - started) * 1000.0
        if latency_ms > self.config.decision_budget_ms:
            # A completed legal answer is still returned, but the overrun remains
            # visible in evidence instead of being hidden.
            error = error or (
                f"soft decision budget exceeded: {latency_ms:.3f} ms > "
                f"{self.config.decision_budget_ms:.3f} ms"
            )

        changed = tuple(final_selection) != tuple(policy_selection)
        self.memory.decision_index += 1
        self.memory.fallback_count += int(fallback_used)
        self.memory.search_count += int(search_used)
        self.memory.search_change_count += int(search_used and changed)
        self.memory.max_latency_ms = max(self.memory.max_latency_ms, latency_ms)
        telemetry = DecisionTelemetry(
            decision_index=self.memory.decision_index,
            turn=_safe_int(_state(observation).get("turn"), 0),
            context=_safe_int(select.get("context"), -1),
            plan=self.memory.plan.value,
            selection=tuple(final_selection),
            policy_selection=tuple(policy_selection),
            option_scores=scores,
            strategic=snapshot,
            archetype_posterior=tuple(sorted(self.memory.archetype_posterior.items())),
            belief_entropy=self._belief_entropy(),
            search_used=search_used,
            search_changed_action=search_used and changed,
            search_bonus=tuple(sorted(search_bonus.items())),
            criticality=criticality,
            latency_ms=latency_ms,
            fallback_used=fallback_used,
            error=error,
        )
        self.memory.telemetry.append(telemetry)
        if len(self.memory.telemetry) > self.config.max_telemetry:
            del self.memory.telemetry[: len(self.memory.telemetry) - self.config.max_telemetry]
        return list(final_selection)

    def _update_memory(self, observation: Mapping[str, Any]) -> None:
        state = _state(observation)
        turn = _safe_int(state.get("turn"), 0)
        if self.memory.last_turn >= 0 and turn < self.memory.last_turn:
            self.reset()
        self.memory.last_turn = turn

        opponent = _player(observation, 1 - _your_index(observation))
        visible = visible_card_ids(opponent)
        self.memory.revealed_opponent_ids |= visible
        evidence_scores: dict[str, float] = {}
        for name, signature in ARCHETYPE_SIGNATURES.items():
            hits = sum(self.memory.revealed_opponent_ids[card_id] for card_id in signature)
            misses = sum(
                count
                for card_id, count in self.memory.revealed_opponent_ids.items()
                if card_id in set().union(*ARCHETYPE_SIGNATURES.values())
                and card_id not in signature
            )
            evidence_scores[name] = math.exp(min(20.0, 1.35 * hits - 0.18 * misses))
        total = sum(evidence_scores.values())
        self.memory.archetype_posterior = {
            name: score / total for name, score in evidence_scores.items()
        }
        if self.config.ablation is AblationMode.POLICY_ONLY:
            self.memory.plan = StrategicPlan.BOARD_SETUP
        else:
            self.memory.plan = self._infer_plan(observation)

    def _infer_plan(self, observation: Mapping[str, Any]) -> StrategicPlan:
        your = _player(observation, _your_index(observation))
        opponent = _player(observation, 1 - _your_index(observation))
        own_board = _in_play(your)
        own_ids = [_card_id(card) for card in own_board]
        rocket_count = sum(card_id in TEAM_ROCKET_POKEMON for card_id in own_ids)
        own_prizes = len(your.get("prize", ()))
        opponent_prizes = len(opponent.get("prize", ()))
        if len(own_board) <= 2 and _safe_int(_state(observation).get("turn"), 0) > 4:
            return StrategicPlan.RECOVERY
        if rocket_count < 4 or 431 not in own_ids:
            return StrategicPlan.BOARD_SETUP
        powered = any(len(card.get("energies", ())) >= 2 for card in own_board)
        if not powered:
            return StrategicPlan.ENERGY_ENGINE
        if opponent_prizes <= 3 or own_prizes <= 2:
            return StrategicPlan.PRIZE_PUSH
        if len(opponent.get("bench", ())) >= 3 and 1218 in [
            _card_id(card) for card in your.get("hand", ()) if isinstance(card, Mapping)
        ]:
            return StrategicPlan.CONTROL
        return StrategicPlan.PRIZE_PUSH

    def _strategic_snapshot(self, observation: Mapping[str, Any]) -> StrategicSnapshot:
        your = _player(observation, _your_index(observation))
        opponent = _player(observation, 1 - _your_index(observation))
        own_board = _in_play(your)
        opposing_board = _in_play(opponent)
        rocket_count = sum(_card_id(card) in TEAM_ROCKET_POKEMON for card in own_board)
        spidops_count = sum(_card_id(card) == 401 for card in own_board)
        own_energy = sum(len(card.get("energies", ())) for card in own_board)
        opposing_energy = sum(len(card.get("energies", ())) for card in opposing_board)
        own_hp = sum(_safe_int(card.get("hp"), 0) for card in own_board)
        opposing_hp = sum(_safe_int(card.get("hp"), 0) for card in opposing_board)
        own_hand = _safe_int(your.get("handCount"), len(your.get("hand", ())))
        opposing_hand = _safe_int(opponent.get("handCount"), 0)
        own_prizes = len(your.get("prize", ()))
        opposing_prizes = len(opponent.get("prize", ()))
        select = observation.get("select", {})
        option_count = len(select.get("option", ())) if isinstance(select, Mapping) else 0

        values = (
            _clamp((rocket_count - 2.0) / 3.0 + 0.12 * spidops_count),
            _clamp((own_energy - opposing_energy) / 5.0),
            _clamp((own_hand - opposing_hand) / 8.0 + own_energy / 12.0),
            _clamp((opposing_prizes - own_prizes) / 4.0),
            _clamp((own_hp - opposing_hp) / 600.0 + (len(own_board) - len(opposing_board)) / 5.0),
            _clamp(option_count / 18.0 - 0.35),
        )
        weight_by_plan: Mapping[StrategicPlan, tuple[float, ...]] = {
            StrategicPlan.BOARD_SETUP: (0.32, 0.14, 0.16, 0.08, 0.16, 0.14),
            StrategicPlan.ENERGY_ENGINE: (0.16, 0.24, 0.25, 0.08, 0.12, 0.15),
            StrategicPlan.PRIZE_PUSH: (0.08, 0.24, 0.10, 0.30, 0.18, 0.10),
            StrategicPlan.RECOVERY: (0.22, 0.10, 0.22, 0.08, 0.23, 0.15),
            StrategicPlan.CONTROL: (0.10, 0.18, 0.12, 0.18, 0.28, 0.14),
        }
        if self.config.ablation is AblationMode.POLICY_ONLY:
            weights = (1.0 / 6.0,) * 6
        else:
            weights = weight_by_plan[self.memory.plan]
        aggregate = sum(value * weight for value, weight in zip(values, weights, strict=True))
        return StrategicSnapshot(values=values, weights=weights, aggregate=aggregate)

    def _belief_entropy(self) -> float:
        probabilities = self.memory.archetype_posterior.values()
        return -sum(p * math.log(max(p, 1e-12)) for p in probabilities)

    def _score_option(
        self,
        observation: Mapping[str, Any],
        option: Mapping[str, Any],
        index: int,
    ) -> float:
        select = observation.get("select", {})
        context = _safe_int(select.get("context"), -1)
        option_type = _safe_int(option.get("type"), -1)
        card_id = option_card_id(observation, option)
        target_id = option_target_pokemon_id(observation, option)
        score = -0.001 * index  # deterministic stable tie break

        if context == SelectContext.MAIN:
            score += self._score_main(observation, option, option_type, card_id, target_id)
        elif option_type == OptionType.NUMBER:
            number = _safe_int(option.get("number"), 0)
            score += number * (10.0 if context != SelectContext.DAMAGE_COUNTER_COUNT else 3.0)
        elif option_type in (OptionType.YES, OptionType.NO):
            yes = option_type == OptionType.YES
            score += 80.0 if yes else 5.0
            if context == SelectContext.COIN_HEAD:
                score = 10.0 if yes else 9.9
        elif option_type == OptionType.ATTACK:
            score += self._attack_score(observation, _safe_int(option.get("attackId"), -1))
        elif context == SelectContext.SETUP_ACTIVE_POKEMON:
            score += {414: 135, 434: 112, 400: 92, 431: 68, 272: 40}.get(card_id, 20)
        elif context == SelectContext.SETUP_BENCH_POKEMON:
            score += {431: 170, 400: 155, 414: 118, 434: 72, 272: 12}.get(card_id, 20)
        elif context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
            score += self._active_target_score(observation, target_id or card_id)
        elif context == SelectContext.TO_HAND:
            score += self._search_target_score(observation, card_id)
        elif context in (
            SelectContext.DISCARD,
            SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
            SelectContext.DISCARD_ENERGY_CARD,
            SelectContext.DISCARD_ENERGY,
            SelectContext.DISCARD_TOOL_CARD,
        ):
            score += self._discard_score(observation, card_id)
        elif context in (SelectContext.ATTACH_FROM, SelectContext.ATTACH_TO):
            score += self._attach_target_score(observation, card_id, target_id or card_id)
        elif context in (SelectContext.EVOLVE, SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO):
            score += 150.0 if card_id in (400, 401) else 40.0
        elif context in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER, SelectContext.DAMAGE_COUNTER_ANY):
            score += self._damage_target_score(observation, option)
        elif context in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
            score += self._heal_target_score(observation, option)
        elif context == SelectContext.SKILL_ORDER:
            score += 20.0 if card_id in (401, 1257) else 10.0
        else:
            score += self._generic_card_value(card_id)

        if self.config.ablation in (AblationMode.BELIEF_STRATEGIC, AblationMode.FULL):
            score += self._belief_adjustment(observation, option, card_id, target_id)
        return float(score)

    def _score_main(
        self,
        observation: Mapping[str, Any],
        option: Mapping[str, Any],
        option_type: int,
        card_id: int | None,
        target_id: int | None,
    ) -> float:
        your = _player(observation, _your_index(observation))
        hand_count = _safe_int(your.get("handCount"), len(your.get("hand", ())))
        own_board_ids = [_card_id(card) for card in _in_play(your)]
        turn = _safe_int(_state(observation).get("turn"), 0)
        if option_type == OptionType.PLAY:
            base = self._play_card_score(observation, card_id, hand_count, turn, own_board_ids)
            return base
        if option_type == OptionType.ATTACH:
            return self._attach_target_score(observation, card_id, target_id)
        if option_type == OptionType.EVOLVE:
            return 178.0 if card_id == 401 else 100.0
        if option_type == OptionType.ABILITY:
            if card_id == 401:
                grass_in_discard = sum(
                    _card_id(card) == 1
                    for card in your.get("discard", ())
                    if isinstance(card, Mapping)
                )
                return 182.0 if grass_in_discard else 30.0
            if card_id == 1257:
                return 145.0
            return 85.0
        if option_type == OptionType.RETREAT:
            has_attack = self._has_option_type(observation, OptionType.ATTACK)
            active = _in_play(your)[:1]
            active_id = _card_id(active[0]) if active else None
            rocket_count = sum(card_id in TEAM_ROCKET_POKEMON for card_id in own_board_ids)
            stranded_mewtwo = active_id == 431 and rocket_count < 4
            return 125.0 if stranded_mewtwo or not has_attack else -35.0
        if option_type == OptionType.ATTACK:
            return self._attack_score(observation, _safe_int(option.get("attackId"), -1))
        if option_type == OptionType.END:
            return -120.0 if self._has_option_type(observation, OptionType.ATTACK) else 1.0
        return 0.0

    def _play_card_score(
        self,
        observation: Mapping[str, Any],
        card_id: int | None,
        hand_count: int,
        turn: int,
        own_board_ids: Sequence[int | None],
    ) -> float:
        rocket_count = sum(value in TEAM_ROCKET_POKEMON for value in own_board_ids)
        bench_count = len(_player(observation, _your_index(observation)).get("bench", ()))
        if card_id == 431:
            return 176.0 if 431 not in own_board_ids else 105.0
        if card_id == 400:
            return 171.0 if 400 not in own_board_ids else 138.0
        if card_id == 414:
            return 132.0
        if card_id == 434:
            return 78.0 if bench_count < 4 else 20.0
        if card_id == 272:
            dragon = self.memory.archetype_posterior.get("dragapult", 0.0)
            return 90.0 * dragon + 8.0
        if card_id == 1220:
            return 215.0 if turn <= 2 and rocket_count < 4 else 48.0
        if card_id == 1216:
            return 172.0 + max(0, 6 - hand_count) * 8.0
        if card_id == 1227:
            return 164.0 + max(0, 6 - hand_count) * 7.0
        if card_id == 1219:
            return 118.0
        if card_id == 1218:
            opponent = _player(observation, 1 - _your_index(observation))
            return 188.0 if len(opponent.get("bench", ())) >= 2 else 62.0
        if card_id == 1134:
            return 157.0
        if card_id == 1121:
            return 148.0 if hand_count >= 4 else 40.0
        if card_id == 1094:
            return 151.0 if self.memory.plan in (
                StrategicPlan.BOARD_SETUP,
                StrategicPlan.ENERGY_ENGINE,
            ) else 86.0
        if card_id == 1097:
            discard_ids = visible_card_ids(
                {"discard": _player(observation, _your_index(observation)).get("discard", ())}
            )
            return 154.0 if any(discard_ids[value] for value in (1, 5, 400, 401, 431)) else 25.0
        if card_id == 1116:
            return 122.0
        if card_id == 1158:
            return 164.0
        if card_id == 1156:
            return 112.0
        if card_id == 1257:
            return 169.0 if turn <= 6 else 90.0
        if card_id == 1256:
            return 96.0
        return self._generic_card_value(card_id)

    def _attack_score(self, observation: Mapping[str, Any], attack_id: int) -> float:
        damage = self._expected_attack_damage(observation, attack_id)
        opponent = _player(observation, 1 - _your_index(observation))
        active = _in_play(opponent)[:1]
        hp = _safe_int(active[0].get("hp"), 9999) if active else 9999
        lethal = bool(active and damage >= hp)
        score = 205.0 + min(140.0, damage * 0.35)
        if lethal:
            score += 260.0
            active_id = _card_id(active[0])
            if active_id in (272, 431, 648, 678, 723, 1031, 121):
                score += 85.0
        return score

    def _expected_attack_damage(self, observation: Mapping[str, Any], attack_id: int) -> int:
        damage = ATTACK_DAMAGE.get(attack_id, 50)
        your = _player(observation, _your_index(observation))
        opponent = _player(observation, 1 - _your_index(observation))
        active = _in_play(your)[:1]
        if attack_id == 560:
            return 30 * sum(
                _card_id(card) in TEAM_ROCKET_POKEMON for card in _in_play(your)
            )
        if attack_id == 583 and active:
            energy_ids = [_card_id(card) for card in active[0].get("energyCards", ())]
            return 120 if 15 in energy_ids else 60
        if attack_id == 608:
            bench_energy = sum(
                len(card.get("energyCards", ())) for card in your.get("bench", ())
            )
            return 160 + 60 * min(2, bench_energy)
        if attack_id == 371:
            return 20 + 20 * (
                len(your.get("bench", ())) + len(opponent.get("bench", ()))
            )
        return damage

    def _attach_target_score(
        self,
        observation: Mapping[str, Any],
        attached_card_id: int | None,
        target_id: int | None,
    ) -> float:
        if attached_card_id == 15:
            return {431: 196, 414: 156, 401: 104, 400: 70, 434: 58}.get(target_id, -50)
        if attached_card_id == 5:
            return {431: 188, 272: 126, 434: 98, 401: 20}.get(target_id, 10)
        if attached_card_id == 1:
            return {401: 184, 400: 142, 431: 72, 414: 35}.get(target_id, 12)
        if attached_card_id == 1158:
            return {431: 196, 401: 178, 414: 104}.get(target_id, 25)
        if attached_card_id == 1156:
            own_active = _in_play(_player(observation, _your_index(observation)))[:1]
            active_id = _card_id(own_active[0]) if own_active else None
            return 156.0 if target_id == active_id else 86.0
        if attached_card_id in ENERGY:
            return 80.0
        return 50.0

    def _active_target_score(self, observation: Mapping[str, Any], card_id: int | None) -> float:
        your = _player(observation, _your_index(observation))
        candidate = next((card for card in _in_play(your) if _card_id(card) == card_id), None)
        energy_count = len(candidate.get("energies", ())) if candidate else 0
        rocket_count = sum(_card_id(card) in TEAM_ROCKET_POKEMON for card in _in_play(your))
        base = {431: 145 if rocket_count >= 4 else 30, 401: 138, 414: 105, 434: 25, 400: 50, 272: 80}.get(card_id, 20)
        return base + 24.0 * energy_count

    def _search_target_score(self, observation: Mapping[str, Any], card_id: int | None) -> float:
        select = observation.get("select", {})
        effect = select.get("effect") or select.get("contextCard") or {}
        effect_id = _card_id(effect if isinstance(effect, Mapping) else None)
        your = _player(observation, _your_index(observation))
        own_ids = [_card_id(card) for card in _in_play(your)]
        hand_ids = [
            _card_id(card) for card in your.get("hand", ()) if isinstance(card, Mapping)
        ]
        if effect_id == 1220:
            return {431: 210, 400: 195, 414: 150, 434: 75}.get(card_id, -50)
        if effect_id == 1134:
            if self.memory.plan == StrategicPlan.BOARD_SETUP:
                return {1220: 210, 1216: 160, 1219: 125, 1218: 85}.get(card_id, 30)
            if self.memory.plan in (StrategicPlan.PRIZE_PUSH, StrategicPlan.CONTROL):
                return {1218: 215, 1216: 140, 1219: 115, 1220: 40}.get(card_id, 30)
            return {1216: 190, 1219: 140, 1218: 110, 1220: 80}.get(card_id, 30)
        if effect_id == 1121:
            if 400 in own_ids and 401 not in own_ids and 401 not in hand_ids:
                return 225.0 if card_id == 401 else self._generic_card_value(card_id)
            if 431 not in own_ids and 431 not in hand_ids:
                return 215.0 if card_id == 431 else self._generic_card_value(card_id)
            return {401: 175, 400: 160, 431: 155, 414: 100, 434: 55, 272: 20}.get(card_id, 10)
        if effect_id == 1094:
            return {401: 205, 400: 185, 1: 168}.get(card_id, 20)
        if effect_id == 1097:
            return {431: 210, 401: 196, 400: 158, 5: 148, 1: 140, 414: 110}.get(card_id, 20)
        if effect_id == 1219:
            return {1158: 220, 1134: 180, 1121: 165, 1257: 155, 1097: 145}.get(card_id, self._generic_card_value(card_id))
        return self._generic_card_value(card_id)

    def _discard_score(self, observation: Mapping[str, Any], card_id: int | None) -> float:
        your = _player(observation, _your_index(observation))
        own_ids = [_card_id(card) for card in _in_play(your)]
        hand_ids = [
            _card_id(card) for card in your.get("hand", ()) if isinstance(card, Mapping)
        ]
        duplicate = hand_ids.count(card_id) > 1 if card_id is not None else False
        if card_id == 1 and 401 in own_ids:
            return 185.0  # Charging Up converts this discard into acceleration.
        if card_id in (1256, 1257) and duplicate:
            return 170.0
        if card_id in (272, 434):
            return 135.0
        if card_id in (1219, 1220) and self.memory.plan != StrategicPlan.BOARD_SETUP:
            return 120.0
        if duplicate:
            return 105.0
        if card_id in (431, 401, 15, 5, 1158):
            return -80.0
        if card_id in ENERGY:
            return 70.0
        return 65.0

    def _damage_target_score(
        self, observation: Mapping[str, Any], option: Mapping[str, Any]
    ) -> float:
        target = _zone_card(
            observation,
            player_index=_safe_int(option.get("playerIndex"), 1 - _your_index(observation)),
            area=_safe_int(option.get("area"), -1),
            index=_safe_int(option.get("index"), -1),
        )
        if not target:
            return 20.0
        hp = _safe_int(target.get("hp"), 0)
        max_hp = _safe_int(target.get("maxHp"), hp)
        damage_taken = max(0, max_hp - hp)
        ex_bonus = 50.0 if _card_id(target) in (272, 431, 648, 678, 723, 1031, 121) else 0.0
        return 100.0 + damage_taken * 0.5 + ex_bonus

    def _heal_target_score(
        self, observation: Mapping[str, Any], option: Mapping[str, Any]
    ) -> float:
        target = _zone_card(
            observation,
            player_index=_safe_int(option.get("playerIndex"), _your_index(observation)),
            area=_safe_int(option.get("area"), -1),
            index=_safe_int(option.get("index"), -1),
        )
        if not target:
            return 20.0
        return float(_safe_int(target.get("maxHp"), 0) - _safe_int(target.get("hp"), 0))

    def _generic_card_value(self, card_id: int | None) -> float:
        if card_id in POKEMON:
            return {431: 180, 401: 175, 400: 155, 414: 120, 434: 60, 272: 35}.get(card_id, 80)
        if card_id in SUPPORTERS:
            return 125.0
        if card_id in ENERGY:
            return 110.0
        if card_id in TOOLS:
            return 105.0
        if card_id in STADIUMS:
            return 95.0
        return 50.0

    def _belief_adjustment(
        self,
        observation: Mapping[str, Any],
        option: Mapping[str, Any],
        card_id: int | None,
        target_id: int | None,
    ) -> float:
        posterior = self.memory.archetype_posterior
        adjustment = 0.0
        if card_id == 272:
            adjustment += 85.0 * posterior.get("dragapult", 0.0)
        if card_id == 1256:
            adjustment += 60.0 * (
                posterior.get("abomasnow", 0.0)
                + posterior.get("starmie", 0.0)
                + posterior.get("lucario", 0.0)
            )
        if target_id in (723, 1031, 648, 678, 121):
            adjustment += 45.0
        if _safe_int(option.get("type"), -1) == OptionType.ATTACK:
            adjustment += 12.0 * (1.0 - self._belief_entropy() / math.log(len(ARCHETYPE_SIGNATURES)))
        return adjustment

    def _ranked_selection(
        self, observation: Mapping[str, Any], scores: Sequence[float]
    ) -> tuple[int, ...]:
        contract = SelectionContract.from_observation(observation)
        select = observation.get("select", {})
        context = _safe_int(select.get("context"), -1)
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        count = self._selection_count(observation, ranked, scores, contract)
        return tuple(ranked[:count])

    def _selection_count(
        self,
        observation: Mapping[str, Any],
        ranked: Sequence[int],
        scores: Sequence[float],
        contract: SelectionContract,
    ) -> int:
        if contract.min_count == contract.max_count:
            return contract.max_count
        select = observation.get("select", {})
        context = _safe_int(select.get("context"), -1)
        if context == SelectContext.DISCARD_ENERGY_CARD:
            # Erasure Ball should spend only the Energy required for the knockout.
            opponent = _player(observation, 1 - _your_index(observation))
            active = _in_play(opponent)[:1]
            hp = _safe_int(active[0].get("hp"), 9999) if active else 9999
            required = max(0, math.ceil((hp - 160) / 60))
            return max(contract.min_count, min(contract.max_count, required))
        if context == SelectContext.SETUP_BENCH_POKEMON:
            positive = sum(scores[index] > 30 for index in ranked)
            return max(contract.min_count, min(contract.max_count, positive))
        if context in (
            SelectContext.TO_HAND,
            SelectContext.DISCARD,
            SelectContext.DAMAGE_COUNTER,
            SelectContext.DAMAGE_COUNTER_ANY,
        ):
            return contract.max_count
        positive = sum(scores[index] > 0 for index in ranked)
        return max(contract.min_count, min(contract.max_count, positive))

    def _criticality(self, observation: Mapping[str, Any], scores: Sequence[float]) -> float:
        if len(scores) <= 1:
            return 0.0
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1]
        close_score = math.exp(-abs(margin) / 35.0)
        select = observation.get("select", {})
        main = _safe_int(select.get("context"), -1) == SelectContext.MAIN
        attack = self._has_option_type(observation, OptionType.ATTACK)
        opponent = _player(observation, 1 - _your_index(observation))
        own = _player(observation, _your_index(observation))
        prize_pressure = max(0.0, (4 - min(len(own.get("prize", ())), len(opponent.get("prize", ())))) / 4.0)
        entropy = self._belief_entropy() / math.log(len(ARCHETYPE_SIGNATURES))
        value = (
            0.28 * close_score
            + 0.24 * float(main and attack)
            + 0.20 * prize_pressure
            + 0.18 * entropy
            + 0.10 * min(1.0, len(scores) / 12.0)
        )
        return max(0.0, min(1.0, value))

    @staticmethod
    def _has_option_type(observation: Mapping[str, Any], option_type: int) -> bool:
        select = observation.get("select", {})
        options = select.get("option", ()) if isinstance(select, Mapping) else ()
        return any(_safe_int(option.get("type"), -1) == option_type for option in options)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": COMPETITION_SCHEMA_VERSION,
            "deck": self.deck.to_dict(),
            "ablation": self.config.ablation.value,
            "match_index": self.memory.match_index,
            "decision_count": self.memory.decision_index,
            "fallback_count": self.memory.fallback_count,
            "search_count": self.memory.search_count,
            "search_change_count": self.memory.search_change_count,
            "max_latency_ms": self.memory.max_latency_ms,
            "plan": self.memory.plan.value,
            "archetype_posterior": dict(sorted(self.memory.archetype_posterior.items())),
            "telemetry": [item.to_dict() for item in self.memory.telemetry],
        }
