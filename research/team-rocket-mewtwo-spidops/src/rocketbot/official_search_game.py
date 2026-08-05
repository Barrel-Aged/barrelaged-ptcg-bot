"""Branchable information-set boundary over the licensed native search API.

The competition runtime can reconstruct a native state from an exact agent
observation plus caller-supplied hidden-zone completions.  Retained native
search IDs are immutable branch roots: applying different legal selections to
the same ID creates independent children.  This module exposes that capability
behind :class:`~rocketbot.game_theory.AbstractInformationSetGame`.

The distinction is important:

* transitions after a completion is chosen are genuine native-engine search
  transitions;
* the chance distribution over hidden completions is a research abstraction
  supplied by this project, not an official belief distribution; and
* native search IDs and the selected completion remain opaque solver state and
  never enter public-belief or information-set policy payloads.

The adapter is production-disabled, process-local, and competition-use-only.
It does not by itself train an equilibrium blueprint or certify safe resolving.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from .competition import (
    OptionType,
    SelectContext,
    SelectionContract,
    _card_id,
    _player,
    _safe_int,
    _state,
    _your_index,
    visible_card_ids,
)
from .game_theory import (
    AbstractAction,
    ChanceOutcome,
    GameBoundaryKind,
    GameBoundaryProvenance,
    InformationSet,
    InformationSetActor,
    OpponentRange,
    PublicBeliefState,
    RangeHypothesis,
)
from .official_bridge import (
    complete_official_actions,
    official_private_observation_payload,
    sanitize_official_public_state,
)
from .phase6_bridge import official_runtime_fingerprint


OFFICIAL_SEARCH_GAME_SCHEMA_VERSION = 1
_ACTIVE_GAME_TOKEN: str | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _common_log_payload(observation: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Conservative public log shape with all card/serial identities absent."""

    allowed = (
        "type",
        "playerIndex",
        "targetPlayerIndex",
        "number",
        "count",
        "result",
        "reason",
        "head",
    )
    result = []
    for raw in _sequence(observation.get("logs", ())):
        if not isinstance(raw, Mapping):
            continue
        result.append(
            {
                name: raw[name]
                for name in allowed
                if name in raw and raw[name] is not None
            }
        )
    return tuple(result)


def _root_public_history(observation: Mapping[str, Any]) -> str:
    return _fingerprint(
        {
            "public_state": sanitize_official_public_state(observation),
            "common_logs": _common_log_payload(observation),
        }
    )


def _next_public_history(
    previous: str,
    observation: Mapping[str, Any],
) -> str:
    _validate_digest(previous, "previous_public_history")
    return _fingerprint(
        {
            "previous": previous,
            "public_state": sanitize_official_public_state(observation),
            "common_logs": _common_log_payload(observation),
        }
    )


@dataclass(frozen=True, slots=True)
class HiddenZoneCompletion:
    """One opaque joint hidden-state particle supplied to ``search_begin``."""

    hypothesis_id: str
    probability: float
    root_player_index: int
    your_candidate_id: str
    opponent_candidate_id: str
    your_archetype_id: str
    opponent_archetype_id: str
    your_style_id: str
    opponent_style_id: str
    your_deck_fingerprint: str
    opponent_deck_fingerprint: str
    your_deck: tuple[int, ...]
    your_prize: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    opponent_prize: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    opponent_active: tuple[int, ...]
    manual_coin: bool
    public_evidence_adjustments: int
    source: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.hypothesis_id,
                self.your_candidate_id,
                self.opponent_candidate_id,
                self.your_archetype_id,
                self.opponent_archetype_id,
                self.your_style_id,
                self.opponent_style_id,
                self.source,
            )
        ):
            raise ValueError("hidden completion identity/source fields are required")
        if self.root_player_index not in (0, 1):
            raise ValueError("hidden completion root player must be 0 or 1")
        if not math.isfinite(self.probability) or self.probability <= 0:
            raise ValueError("hidden completion probability must be positive and finite")
        for field_name in ("your_deck_fingerprint", "opponent_deck_fingerprint"):
            _validate_digest(getattr(self, field_name), field_name)
        for field_name in (
            "your_deck",
            "your_prize",
            "opponent_deck",
            "opponent_prize",
            "opponent_hand",
            "opponent_active",
        ):
            values = getattr(self, field_name)
            if any(not isinstance(value, int) or value <= 0 for value in values):
                raise ValueError(f"{field_name} must contain positive card IDs")
        if len(self.opponent_active) > 1:
            raise ValueError("hidden completion can supply at most one Active card")
        if self.public_evidence_adjustments < 0:
            raise ValueError("public-evidence adjustment count cannot be negative")

    @property
    def fingerprint(self) -> str:
        # Raw private identities bind the opaque particle fingerprint but are
        # never emitted by ``manifest`` or policy-facing methods.
        return _fingerprint(
            {
                "schema": OFFICIAL_SEARCH_GAME_SCHEMA_VERSION,
                **asdict(self),
            }
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "probability": self.probability,
            "root_player_index": self.root_player_index,
            "your_candidate_id": self.your_candidate_id,
            "opponent_candidate_id": self.opponent_candidate_id,
            "your_archetype_id": self.your_archetype_id,
            "opponent_archetype_id": self.opponent_archetype_id,
            "your_style_id": self.your_style_id,
            "opponent_style_id": self.opponent_style_id,
            "your_deck_fingerprint": self.your_deck_fingerprint,
            "opponent_deck_fingerprint": self.opponent_deck_fingerprint,
            "zone_counts": {
                "your_deck": len(self.your_deck),
                "your_prize": len(self.your_prize),
                "opponent_deck": len(self.opponent_deck),
                "opponent_prize": len(self.opponent_prize),
                "opponent_hand": len(self.opponent_hand),
                "opponent_active": len(self.opponent_active),
            },
            "manual_coin": self.manual_coin,
            "public_evidence_adjustments": self.public_evidence_adjustments,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "raw_card_ids_persisted": False,
        }

    def candidate_for_player(self, player: int) -> tuple[str, str, str]:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        if player == self.root_player_index:
            return self.your_candidate_id, self.your_archetype_id, self.your_style_id
        return (
            self.opponent_candidate_id,
            self.opponent_archetype_id,
            self.opponent_style_id,
        )

    def search_arguments(self) -> dict[str, Any]:
        return {
            "your_deck": list(self.your_deck),
            "your_prize": list(self.your_prize),
            "opponent_deck": list(self.opponent_deck),
            "opponent_prize": list(self.opponent_prize),
            "opponent_hand": list(self.opponent_hand),
            "opponent_active": list(self.opponent_active),
            "manual_coin": self.manual_coin,
        }


def permute_hidden_zone_completion(
    completion: HiddenZoneCompletion,
    *,
    hypothesis_id: str,
    probability: float,
    permutation_seed: int,
    source: str | None = None,
) -> HiddenZoneCompletion:
    """Create a held-out particle by permuting order within hidden zones.

    Card allocation between zones is deliberately unchanged. This creates
    unseen deck/prize/hand order particles without inventing cards or violating
    the public-consistent completion built at the root. It is a narrow
    generalization probe, not a calibrated posterior over hidden states.
    """

    if not hypothesis_id:
        raise ValueError("permuted completion requires a hypothesis ID")
    if permutation_seed < 0:
        raise ValueError("permutation seed must be non-negative")

    def permuted(values: tuple[int, ...], zone: str) -> tuple[int, ...]:
        materialized = list(values)
        seed = int.from_bytes(
            sha256(
                _canonical_json(
                    {
                        "completion": completion.fingerprint,
                        "permutation_seed": permutation_seed,
                        "zone": zone,
                    }
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        random.Random(seed).shuffle(materialized)
        return tuple(materialized)

    return HiddenZoneCompletion(
        hypothesis_id=hypothesis_id,
        probability=probability,
        root_player_index=completion.root_player_index,
        your_candidate_id=completion.your_candidate_id,
        opponent_candidate_id=completion.opponent_candidate_id,
        your_archetype_id=completion.your_archetype_id,
        opponent_archetype_id=completion.opponent_archetype_id,
        your_style_id=completion.your_style_id,
        opponent_style_id=completion.opponent_style_id,
        your_deck_fingerprint=completion.your_deck_fingerprint,
        opponent_deck_fingerprint=completion.opponent_deck_fingerprint,
        your_deck=permuted(completion.your_deck, "your_deck"),
        your_prize=permuted(completion.your_prize, "your_prize"),
        opponent_deck=permuted(completion.opponent_deck, "opponent_deck"),
        opponent_prize=permuted(completion.opponent_prize, "opponent_prize"),
        opponent_hand=permuted(completion.opponent_hand, "opponent_hand"),
        opponent_active=completion.opponent_active,
        manual_coin=completion.manual_coin,
        public_evidence_adjustments=completion.public_evidence_adjustments,
        source=(
            source
            or (
                "seeded within-zone hidden-order permutation of a "
                "public-consistent completion; allocation unchanged"
            )
        ),
    )


def _visible_without_prize(player: Mapping[str, Any]) -> Counter[int]:
    proxy = dict(player)
    proxy["prize"] = ()
    return visible_card_ids(proxy)


def _drop_nonpositive(pool: Counter[int]) -> None:
    for card_id in tuple(pool):
        if pool[card_id] <= 0:
            del pool[card_id]


def _make_prior_cover(pool: Counter[int], required: Counter[int]) -> int:
    adjustments = 0
    for card_id, count in required.items():
        deficit = count - pool[card_id]
        while deficit > 0:
            donor = max(pool, key=lambda value: (pool[value], -value), default=None)
            if donor is None or pool[donor] <= 0:
                raise ValueError("candidate deck cannot cover public evidence")
            pool[donor] -= 1
            pool[card_id] += 1
            deficit -= 1
            adjustments += 1
    _drop_nonpositive(pool)
    return adjustments


def _take_cards(pool: Counter[int], count: int) -> list[int]:
    if count < 0 or count > sum(pool.values()):
        raise ValueError("hidden completion requests an impossible zone size")
    result: list[int] = []
    for card_id in sorted(pool, key=lambda value: (-pool[value], value)):
        take = min(pool[card_id], count - len(result))
        result.extend([card_id] * take)
        pool[card_id] -= take
        if len(result) == count:
            break
    _drop_nonpositive(pool)
    if len(result) != count:
        raise RuntimeError("hidden completion failed to fill a zone")
    return result


def _complete_prizes(
    player: Mapping[str, Any],
    pool: Counter[int],
) -> list[int]:
    result: list[int] = []
    unknown = 0
    for prize in _sequence(player.get("prize", ())):
        card_id = _card_id(prize if isinstance(prize, Mapping) else None)
        if card_id is None or card_id < 0:
            unknown += 1
        else:
            result.append(card_id)
            pool[card_id] -= 1
    _drop_nonpositive(pool)
    result.extend(_take_cards(pool, unknown))
    return result


def _known_prize_cards(player: Mapping[str, Any]) -> Counter[int]:
    """Return only prize identities already visible in the observation."""

    result: Counter[int] = Counter()
    for prize in _sequence(player.get("prize", ())):
        card_id = _card_id(prize if isinstance(prize, Mapping) else None)
        if card_id is not None and card_id >= 0:
            result[card_id] += 1
    return result


def _take_sampled_cards(
    pool: Counter[int],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Sample without replacement from a card multiset deterministically by RNG."""

    if count < 0 or count > sum(pool.values()):
        raise ValueError("hidden completion requests an impossible sampled zone size")
    population = [
        card_id
        for card_id in sorted(pool)
        for _ in range(pool[card_id])
    ]
    rng.shuffle(population)
    result = population[:count]
    pool.subtract(result)
    _drop_nonpositive(pool)
    if len(result) != count:
        raise RuntimeError("sampled hidden completion failed to fill a zone")
    return result


def _complete_sampled_prizes(
    player: Mapping[str, Any],
    pool: Counter[int],
    rng: random.Random,
) -> list[int]:
    """Preserve revealed prizes and sample identities for unknown prize slots."""

    result: list[int] = []
    unknown = 0
    for prize in _sequence(player.get("prize", ())):
        card_id = _card_id(prize if isinstance(prize, Mapping) else None)
        if card_id is None or card_id < 0:
            unknown += 1
        else:
            if pool[card_id] <= 0:
                raise ValueError("candidate deck cannot cover a revealed prize")
            result.append(card_id)
            pool[card_id] -= 1
    _drop_nonpositive(pool)
    result.extend(_take_sampled_cards(pool, unknown, rng))
    rng.shuffle(result)
    return result


def build_hidden_zone_completion(
    observation: Mapping[str, Any],
    *,
    hypothesis_id: str,
    probability: float,
    your_candidate_id: str,
    opponent_candidate_id: str,
    your_archetype_id: str,
    opponent_archetype_id: str,
    your_style_id: str,
    opponent_style_id: str,
    your_card_ids: Sequence[int],
    opponent_card_ids: Sequence[int],
    your_deck_fingerprint: str,
    opponent_deck_fingerprint: str,
    basic_card_ids: Sequence[int] = (),
    manual_coin: bool = True,
    source: str,
) -> HiddenZoneCompletion:
    """Build a deterministic hidden completion consistent with visible counts.

    Candidate composition is minimally adjusted when public cards contradict a
    deck hypothesis. The adjustment count is explicit and must be used as an
    abstraction/OOD signal by later training or resolving code.
    """

    if len(your_card_ids) != 60 or len(opponent_card_ids) != 60:
        raise ValueError("hidden completion candidates must each contain 60 cards")
    root_player = _your_index(observation)
    your = _player(observation, root_player)
    opponent = _player(observation, 1 - root_player)

    your_pool = Counter(int(value) for value in your_card_ids)
    your_visible = _visible_without_prize(your)
    your_required = your_visible + _known_prize_cards(your)
    your_adjustments = _make_prior_cover(your_pool, your_required)
    your_pool.subtract(your_visible)
    _drop_nonpositive(your_pool)
    your_prize = _complete_prizes(your, your_pool)
    select = observation.get("select")
    visible_deck = select.get("deck") if isinstance(select, Mapping) else None
    your_deck = (
        []
        if visible_deck is not None
        else _take_cards(your_pool, _safe_int(your.get("deckCount"), 0))
    )

    opponent_pool = Counter(int(value) for value in opponent_card_ids)
    opponent_visible = _visible_without_prize(opponent)
    opponent_required = opponent_visible + _known_prize_cards(opponent)
    opponent_adjustments = _make_prior_cover(opponent_pool, opponent_required)
    opponent_pool.subtract(opponent_visible)
    _drop_nonpositive(opponent_pool)
    opponent_active: list[int] = []
    active = _sequence(opponent.get("active", ()))
    if active and active[0] is None:
        basic_set = {int(value) for value in basic_card_ids}
        basic = next(
            (card_id for card_id in opponent_card_ids if card_id in basic_set and opponent_pool[card_id] > 0),
            None,
        )
        if basic is None:
            raise ValueError("face-down opponent Active requires a candidate Basic card")
        opponent_active.append(basic)
        opponent_pool[basic] -= 1
        _drop_nonpositive(opponent_pool)
    opponent_prize = _complete_prizes(opponent, opponent_pool)
    opponent_hand = _take_cards(
        opponent_pool,
        _safe_int(opponent.get("handCount"), 0),
    )
    opponent_deck = _take_cards(
        opponent_pool,
        _safe_int(opponent.get("deckCount"), 0),
    )
    return HiddenZoneCompletion(
        hypothesis_id=hypothesis_id,
        probability=probability,
        root_player_index=root_player,
        your_candidate_id=your_candidate_id,
        opponent_candidate_id=opponent_candidate_id,
        your_archetype_id=your_archetype_id,
        opponent_archetype_id=opponent_archetype_id,
        your_style_id=your_style_id,
        opponent_style_id=opponent_style_id,
        your_deck_fingerprint=your_deck_fingerprint,
        opponent_deck_fingerprint=opponent_deck_fingerprint,
        your_deck=tuple(your_deck),
        your_prize=tuple(your_prize),
        opponent_deck=tuple(opponent_deck),
        opponent_prize=tuple(opponent_prize),
        opponent_hand=tuple(opponent_hand),
        opponent_active=tuple(opponent_active),
        manual_coin=manual_coin,
        public_evidence_adjustments=your_adjustments + opponent_adjustments,
        source=source,
    )


def sample_hidden_zone_completion(
    observation: Mapping[str, Any],
    *,
    hypothesis_id: str,
    probability: float,
    allocation_seed: int,
    your_candidate_id: str,
    opponent_candidate_id: str,
    your_archetype_id: str,
    opponent_archetype_id: str,
    your_style_id: str,
    opponent_style_id: str,
    your_card_ids: Sequence[int],
    opponent_card_ids: Sequence[int],
    your_deck_fingerprint: str,
    opponent_deck_fingerprint: str,
    basic_card_ids: Sequence[int] = (),
    manual_coin: bool = True,
    source: str,
) -> HiddenZoneCompletion:
    """Sample a public-consistent hidden-zone *allocation* particle.

    Unlike :func:`permute_hidden_zone_completion`, this function resamples
    which still-unseen cards occupy deck, prize, hand, and a face-down Active.
    It preserves every visible card identity and every public zone count.  The
    selected particle remains caller-supplied research state: neither the seed
    nor any raw hidden identity is exposed by the public manifest or policy
    payload.
    """

    if allocation_seed < 0:
        raise ValueError("allocation seed must be non-negative")
    if len(your_card_ids) != 60 or len(opponent_card_ids) != 60:
        raise ValueError("hidden completion candidates must each contain 60 cards")
    root_player = _your_index(observation)
    your = _player(observation, root_player)
    opponent = _player(observation, 1 - root_player)

    def zone_rng(owner: str) -> random.Random:
        seed = int.from_bytes(
            sha256(
                _canonical_json(
                    {
                        "allocation_seed": allocation_seed,
                        "hypothesis_id": hypothesis_id,
                        "owner": owner,
                        "public_root": _root_public_history(observation),
                    }
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        return random.Random(seed)

    your_rng = zone_rng("your")
    your_pool = Counter(int(value) for value in your_card_ids)
    your_visible = _visible_without_prize(your)
    your_required = your_visible + _known_prize_cards(your)
    your_adjustments = _make_prior_cover(your_pool, your_required)
    your_pool.subtract(your_visible)
    _drop_nonpositive(your_pool)
    your_prize = _complete_sampled_prizes(your, your_pool, your_rng)
    select = observation.get("select")
    visible_deck = select.get("deck") if isinstance(select, Mapping) else None
    your_deck = (
        []
        if visible_deck is not None
        else _take_sampled_cards(
            your_pool,
            _safe_int(your.get("deckCount"), 0),
            your_rng,
        )
    )

    opponent_rng = zone_rng("opponent")
    opponent_pool = Counter(int(value) for value in opponent_card_ids)
    opponent_visible = _visible_without_prize(opponent)
    opponent_required = opponent_visible + _known_prize_cards(opponent)
    opponent_adjustments = _make_prior_cover(opponent_pool, opponent_required)
    opponent_pool.subtract(opponent_visible)
    _drop_nonpositive(opponent_pool)
    opponent_active: list[int] = []
    active = _sequence(opponent.get("active", ()))
    if active and active[0] is None:
        basic_set = {int(value) for value in basic_card_ids}
        eligible = [
            card_id
            for card_id in sorted(opponent_pool)
            for _ in range(opponent_pool[card_id])
            if card_id in basic_set
        ]
        if not eligible:
            raise ValueError("face-down opponent Active requires a candidate Basic card")
        opponent_rng.shuffle(eligible)
        basic = eligible[0]
        opponent_active.append(basic)
        opponent_pool[basic] -= 1
        _drop_nonpositive(opponent_pool)
    opponent_prize = _complete_sampled_prizes(
        opponent,
        opponent_pool,
        opponent_rng,
    )
    opponent_hand = _take_sampled_cards(
        opponent_pool,
        _safe_int(opponent.get("handCount"), 0),
        opponent_rng,
    )
    opponent_deck = _take_sampled_cards(
        opponent_pool,
        _safe_int(opponent.get("deckCount"), 0),
        opponent_rng,
    )
    return HiddenZoneCompletion(
        hypothesis_id=hypothesis_id,
        probability=probability,
        root_player_index=root_player,
        your_candidate_id=your_candidate_id,
        opponent_candidate_id=opponent_candidate_id,
        your_archetype_id=your_archetype_id,
        opponent_archetype_id=opponent_archetype_id,
        your_style_id=your_style_id,
        opponent_style_id=opponent_style_id,
        your_deck_fingerprint=your_deck_fingerprint,
        opponent_deck_fingerprint=opponent_deck_fingerprint,
        your_deck=tuple(your_deck),
        your_prize=tuple(your_prize),
        opponent_deck=tuple(opponent_deck),
        opponent_prize=tuple(opponent_prize),
        opponent_hand=tuple(opponent_hand),
        opponent_active=tuple(opponent_active),
        manual_coin=manual_coin,
        public_evidence_adjustments=your_adjustments + opponent_adjustments,
        source=source,
    )


@dataclass(frozen=True, slots=True)
class OfficialSearchGameState:
    """Opaque native search state. Only game methods may consume truth fields."""

    game_fingerprint: str
    state_id: str
    actor: InformationSetActor
    depth: int
    terminal: bool
    player_index: int | None
    hypothesis_id: str | None
    hypothesis_fingerprint: str | None
    path: tuple[str, ...]
    public_history_fingerprint: str
    observation_json: str | None
    search_id: int | None

    def __post_init__(self) -> None:
        _validate_digest(self.game_fingerprint, "game_fingerprint")
        _validate_digest(self.state_id, "state_id")
        _validate_digest(self.public_history_fingerprint, "public_history_fingerprint")
        if self.depth < 0:
            raise ValueError("official search state depth cannot be negative")
        if self.terminal != (self.actor is InformationSetActor.TERMINAL):
            raise ValueError("official search terminal/actor flags disagree")
        if self.actor in (InformationSetActor.SELF, InformationSetActor.OPPONENT):
            if self.player_index not in (0, 1) or self.search_id is None:
                raise ValueError("player search state requires player and native ID")
        elif self.actor is InformationSetActor.CHANCE and self.depth > 0:
            if self.search_id is None:
                raise ValueError("native chance state requires a search ID")
        if self.depth == 0:
            if self.hypothesis_id is not None or self.search_id is not None:
                raise ValueError("chance root cannot select hidden truth")
        else:
            if not self.hypothesis_id or not self.hypothesis_fingerprint:
                raise ValueError("native descendant requires an opaque completion")
            _validate_digest(self.hypothesis_fingerprint, "hypothesis_fingerprint")
        if self.observation_json is None and self.depth > 0:
            raise ValueError("native descendants require an observation")

    @property
    def observation(self) -> dict[str, Any]:
        if self.observation_json is None:
            raise ValueError("chance root has no native observation")
        return json.loads(self.observation_json)


class OfficialNativeSearchGame:
    """Live branchable native transition game under an abstract chance prior."""

    def __init__(
        self,
        observation: Mapping[str, Any],
        completions: Sequence[HiddenZoneCompletion],
        *,
        runtime_directory: str | Path,
        maximum_complete_actions: int = 4_096,
    ) -> None:
        global _ACTIVE_GAME_TOKEN
        runtime = Path(runtime_directory).resolve()
        if not (runtime / "cg").is_dir():
            raise FileNotFoundError(f"official runtime must contain cg/: {runtime}")
        if maximum_complete_actions <= 0:
            raise ValueError("maximum_complete_actions must be positive")
        materialized = tuple(completions)
        if not materialized:
            raise ValueError("official search game requires hidden completions")
        ids = [item.hypothesis_id for item in materialized]
        fingerprints = [item.fingerprint for item in materialized]
        if len(ids) != len(set(ids)) or len(fingerprints) != len(set(fingerprints)):
            raise ValueError("hidden completion IDs/fingerprints must be unique")
        root_player = _your_index(observation)
        if any(item.root_player_index != root_player for item in materialized):
            raise ValueError("hidden completion root-player mismatch")
        if observation.get("search_begin_input") is None:
            raise ValueError("official search game requires an exact agent observation")
        total_probability = sum(item.probability for item in materialized)
        if not math.isfinite(total_probability) or total_probability <= 0:
            raise ValueError("hidden completion mass must be positive and finite")
        self.runtime_directory = runtime
        self.maximum_complete_actions = maximum_complete_actions
        self.completions = materialized
        self._probabilities = {
            item.hypothesis_id: item.probability / total_probability for item in materialized
        }
        self._completion_by_id = {item.hypothesis_id: item for item in materialized}
        self.root_observation = deepcopy(dict(observation))
        self.root_player_index = root_player
        self.ruleset_fingerprint = official_runtime_fingerprint(runtime)
        self._provenance = GameBoundaryProvenance(
            name="official_native_search_with_supplied_hidden_completion_chance",
            kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            schema_version=OFFICIAL_SEARCH_GAME_SCHEMA_VERSION,
            ruleset_fingerprint=self.ruleset_fingerprint,
            official_engine_executed=False,
            fidelity_notes=(
                "Descendant transitions execute the licensed official native search "
                "engine. The hidden-completion chance prior is supplied by the "
                "research adapter, is not an official belief distribution, and "
                "keeps this game boundary abstract."
            ),
        )
        self._fingerprint = _fingerprint(
            {
                "schema": OFFICIAL_SEARCH_GAME_SCHEMA_VERSION,
                "provenance": {
                    **asdict(self._provenance),
                    "kind": self._provenance.kind.value,
                },
                "root_public": sanitize_official_public_state(observation),
                "root_private_fingerprint": _fingerprint(
                    official_private_observation_payload(observation)
                ),
                "search_begin_input_fingerprint": _fingerprint(
                    str(observation["search_begin_input"])
                ),
                "completions": [item.fingerprint for item in materialized],
                "probabilities": self._probabilities,
                "maximum_complete_actions": maximum_complete_actions,
                "official_descendant_transitions": True,
                "hidden_prior_official": False,
            }
        )
        if _ACTIVE_GAME_TOKEN is not None:
            raise RuntimeError("only one native search game may be active per process")
        _ACTIVE_GAME_TOKEN = self._fingerprint
        self._closed = False
        self._transition_count = 0
        self._created_state_count = 0
        path_text = str(runtime)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
        try:
            from cg.api import (  # type: ignore[import-not-found]
                search_begin,
                search_end,
                search_step,
                to_observation_class,
            )

            self._search_begin = search_begin
            self._search_end = search_end
            self._search_step = search_step
            self._to_observation_class = to_observation_class
            self._scenario_roots: dict[str, OfficialSearchGameState] = {}
            root_history = _root_public_history(observation)
            for completion in materialized:
                native = self._search_begin(
                    self._to_observation_class(deepcopy(dict(observation))),
                    **completion.search_arguments(),
                )
                self._scenario_roots[completion.hypothesis_id] = self._state_from_native(
                    native,
                    completion=completion,
                    depth=1,
                    path=(f"completion:{completion.fingerprint}",),
                    public_history_fingerprint=root_history,
                )
        except Exception:
            try:
                if hasattr(self, "_search_end"):
                    self._search_end()
            finally:
                _ACTIVE_GAME_TOKEN = None
                self._closed = True
            raise
        self._root = OfficialSearchGameState(
            game_fingerprint=self.fingerprint,
            state_id=_fingerprint({"game": self.fingerprint, "node": "chance_root"}),
            actor=InformationSetActor.CHANCE,
            depth=0,
            terminal=False,
            player_index=None,
            hypothesis_id=None,
            hypothesis_fingerprint=None,
            path=(),
            public_history_fingerprint=root_history,
            observation_json=None,
            search_id=None,
        )

    @property
    def provenance(self) -> GameBoundaryProvenance:
        return self._provenance

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def source_official_engine_executed(self) -> bool:
        return True

    @property
    def branchable_native_transitions(self) -> bool:
        return True

    @property
    def hidden_completion_prior_official(self) -> bool:
        return False

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def created_state_count(self) -> int:
        return self._created_state_count

    def completion(self, hypothesis_id: str) -> HiddenZoneCompletion:
        """Return opaque solver truth for evaluation code, never policy code."""

        try:
            return self._completion_by_id[str(hypothesis_id)]
        except KeyError as exc:
            raise ValueError("unknown hidden completion hypothesis") from exc

    def __enter__(self) -> "OfficialNativeSearchGame":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        global _ACTIVE_GAME_TOKEN
        if self._closed:
            return
        try:
            self._search_end()
        finally:
            self._closed = True
            if _ACTIVE_GAME_TOKEN == self.fingerprint:
                _ACTIVE_GAME_TOKEN = None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("official native search game is closed")

    def _validate_state(self, state: OfficialSearchGameState) -> None:
        self._require_open()
        if not isinstance(state, OfficialSearchGameState):
            raise TypeError("state does not belong to the official search game")
        if state.game_fingerprint != self.fingerprint:
            raise ValueError("official search state/game fingerprint mismatch")

    def _state_from_native(
        self,
        native: Any,
        *,
        completion: HiddenZoneCompletion,
        depth: int,
        path: tuple[str, ...],
        public_history_fingerprint: str,
    ) -> OfficialSearchGameState:
        observation = asdict(native.observation)
        current = _state(observation)
        result = _safe_int(current.get("result"), -1)
        terminal = result != -1
        select = observation.get("select")
        if terminal:
            actor = InformationSetActor.TERMINAL
            player_index = None
        else:
            if not isinstance(select, Mapping):
                raise RuntimeError("nonterminal native search state lacks a selection")
            player_index = _your_index(observation)
            actor = (
                InformationSetActor.CHANCE
                if _safe_int(select.get("context"), -1) == int(SelectContext.COIN_HEAD)
                else (
                    InformationSetActor.SELF
                    if player_index == 0
                    else InformationSetActor.OPPONENT
                )
            )
        observation_json = _canonical_json(observation)
        state_id = _fingerprint(
            {
                "game": self.fingerprint,
                "hypothesis": completion.fingerprint,
                "path": path,
                "observation": _fingerprint(observation),
                "depth": depth,
            }
        )
        self._created_state_count += 1
        return OfficialSearchGameState(
            game_fingerprint=self.fingerprint,
            state_id=state_id,
            actor=actor,
            depth=depth,
            terminal=terminal,
            player_index=player_index,
            hypothesis_id=completion.hypothesis_id,
            hypothesis_fingerprint=completion.fingerprint,
            path=path,
            public_history_fingerprint=public_history_fingerprint,
            observation_json=observation_json,
            search_id=int(native.searchId),
        )

    def root_state(self) -> OfficialSearchGameState:
        self._require_open()
        return self._root

    def player_to_act(self, state: OfficialSearchGameState) -> int | None:
        """Expose only turn ownership; do not expose the selected completion."""

        self._validate_state(state)
        return state.player_index

    def _completion_range_for_player(self, player: int) -> OpponentRange:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        grouped: dict[tuple[str, str, str], float] = {}
        for completion in self.completions:
            opponent = completion.candidate_for_player(1 - player)
            grouped[opponent] = grouped.get(opponent, 0.0) + self._probabilities[
                completion.hypothesis_id
            ]
        return OpponentRange.normalized(
            tuple(
                RangeHypothesis(
                    hypothesis_id=candidate_id,
                    weight=weight,
                    hidden_resource_counts=(),
                    intent_id=archetype_id,
                    style_id=style_id,
                    source=(
                        "static joint hidden-completion prior over native search "
                        "roots; selected completion truth is not an input"
                    ),
                )
                for (candidate_id, archetype_id, style_id), weight in sorted(grouped.items())
            )
        )

    def supplied_completion_range(self, *, player: int) -> OpponentRange:
        """Expose the static supplied range without exposing selected truth.

        This is useful for offline belief diagnostics from either player's
        perspective. It remains an uncalibrated research prior and is not tied
        to the selected opaque completion.
        """

        self._require_open()
        return self._completion_range_for_player(player)

    def public_belief(
        self,
        state: OfficialSearchGameState,
        *,
        player: int,
    ) -> PublicBeliefState:
        self._validate_state(state)
        if state.terminal or state.actor is InformationSetActor.CHANCE:
            raise ValueError("public belief is available only at player decisions")
        if state.player_index != player:
            raise ValueError("public belief requires the acting player's perspective")
        observation = state.observation
        public = sanitize_official_public_state(observation)
        players = public["players"]
        own = players[player] if 0 <= player < len(players) else {}
        opponent = players[1 - player] if len(players) == 2 else {}
        opponent_range = self._completion_range_for_player(player)
        public_features = (
            ("turn", float(public["turn"])),
            ("turn_action_count", float(public["turn_action_count"])),
            ("own_board_count", float(len(own.get("active", ())) + len(own.get("bench", ())))),
            ("opponent_board_count", float(len(opponent.get("active", ())) + len(opponent.get("bench", ())))),
            ("own_hand_count", float(own.get("hand_count", 0))),
            ("opponent_hand_count", float(opponent.get("hand_count", 0))),
            ("own_prize_count", float(own.get("prize_count", 0))),
            ("opponent_prize_count", float(opponent.get("prize_count", 0))),
            ("own_deck_count", float(own.get("deck_count", 0))),
            ("opponent_deck_count", float(opponent.get("deck_count", 0))),
            ("completion_range_entropy", float(opponent_range.entropy)),
            (
                "completion_range_effective_sample_size",
                float(opponent_range.effective_sample_size),
            ),
        )
        return PublicBeliefState(
            public_state_fingerprint=_fingerprint(public),
            public_history_fingerprint=state.public_history_fingerprint,
            turn=max(0, _safe_int(public.get("turn"), 0)),
            acting_player=InformationSetActor.SELF,
            opponent_range=opponent_range,
            public_features=public_features,
            updater_fingerprint=_fingerprint(
                {
                    "game": self.fingerprint,
                    "updater": "static_joint_completion_prior",
                    "selected_truth_input": False,
                }
            ),
        )

    def legal_actions(
        self,
        state: OfficialSearchGameState,
    ) -> tuple[AbstractAction, ...]:
        self._validate_state(state)
        if state.terminal or state.actor is InformationSetActor.CHANCE:
            return ()
        return complete_official_actions(
            state.observation,
            maximum_complete_actions=self.maximum_complete_actions,
        )

    def information_set(
        self,
        state: OfficialSearchGameState,
        *,
        player: int,
    ) -> InformationSet:
        self._validate_state(state)
        if state.terminal or state.actor is InformationSetActor.CHANCE:
            raise ValueError("information set is available only at player decisions")
        if state.player_index != player:
            raise ValueError("information set requires the acting player")
        observation = state.observation
        belief = self.public_belief(state, player=player)
        actions = self.legal_actions(state)
        private_fingerprint = _fingerprint(
            official_private_observation_payload(observation)
        )
        contract = SelectionContract.from_observation(observation)
        identity = _fingerprint(
            {
                "public_belief": belief.fingerprint,
                "private_observation": private_fingerprint,
                "contract": asdict(contract),
                "actions": [
                    (action.action_id, action.official_option_indices) for action in actions
                ],
                "selected_completion_input": False,
            }
        )
        return InformationSet(
            info_set_id=f"official-search:{identity}",
            public_belief_fingerprint=belief.fingerprint,
            actor=InformationSetActor.SELF,
            legal_actions=actions,
            depth=state.depth,
            terminal=False,
            private_observation_fingerprint=private_fingerprint,
        )

    def chance_outcomes(
        self,
        state: OfficialSearchGameState,
    ) -> tuple[ChanceOutcome, ...]:
        self._validate_state(state)
        if state.actor is not InformationSetActor.CHANCE:
            return ()
        if state.depth == 0:
            return tuple(
                ChanceOutcome(
                    outcome_id=f"completion:{completion.fingerprint}",
                    probability=self._probabilities[completion.hypothesis_id],
                    public_label=f"opaque hidden completion particle {index}",
                )
                for index, completion in enumerate(self.completions)
            )
        observation = state.observation
        options = _sequence(observation.get("select", {}).get("option", ()))
        outcomes = []
        for index, option in enumerate(options):
            option_type = _safe_int(option.get("type"), -1) if isinstance(option, Mapping) else -1
            if option_type not in (int(OptionType.YES), int(OptionType.NO)):
                continue
            head = option_type == int(OptionType.YES)
            outcomes.append(
                ChanceOutcome(
                    outcome_id=f"coin:{'head' if head else 'tail'}:{index}",
                    probability=0.5,
                    public_label="coin head" if head else "coin tail",
                )
            )
        if len(outcomes) != 2:
            raise RuntimeError("manual coin prompt did not expose head/tail choices")
        return tuple(outcomes)

    def transition(
        self,
        state: OfficialSearchGameState,
        action: AbstractAction | None,
        *,
        chance_outcome: ChanceOutcome | None = None,
    ) -> OfficialSearchGameState:
        self._validate_state(state)
        if state.terminal:
            raise ValueError("cannot transition a terminal native state")
        if state.depth == 0:
            if action is not None or chance_outcome is None:
                raise ValueError("hidden-completion root requires only a chance outcome")
            matches = [
                completion
                for completion in self.completions
                if chance_outcome.outcome_id == f"completion:{completion.fingerprint}"
            ]
            if len(matches) != 1:
                raise ValueError("unknown hidden-completion chance outcome")
            expected = next(
                item
                for item in self.chance_outcomes(state)
                if item.outcome_id == chance_outcome.outcome_id
            )
            if chance_outcome != expected:
                raise ValueError("hidden-completion chance probability/label mismatch")
            return self._scenario_roots[matches[0].hypothesis_id]
        if state.actor is InformationSetActor.CHANCE:
            if action is not None or chance_outcome is None:
                raise ValueError("native chance node requires only a chance outcome")
            expected = {
                item.outcome_id: item for item in self.chance_outcomes(state)
            }.get(chance_outcome.outcome_id)
            if expected is None or expected != chance_outcome:
                raise ValueError("unknown or altered native chance outcome")
            index = int(chance_outcome.outcome_id.rsplit(":", 1)[1])
            selected_indices = (index,)
            path_item = chance_outcome.outcome_id
        else:
            if action is None or chance_outcome is not None:
                raise ValueError("player node requires exactly one abstract action")
            matches = [
                legal
                for legal in self.legal_actions(state)
                if legal.action_id == action.action_id
                and legal.official_option_indices == action.official_option_indices
            ]
            if len(matches) != 1 or matches[0] != action:
                raise ValueError("abstract action is not exactly legal at this native state")
            selected_indices = action.official_option_indices
            path_item = action.action_id
        completion = self._completion_by_id[state.hypothesis_id or ""]
        native = self._search_step(int(state.search_id), list(selected_indices))
        observation = asdict(native.observation)
        history = _next_public_history(state.public_history_fingerprint, observation)
        self._transition_count += 1
        return self._state_from_native(
            native,
            completion=completion,
            depth=state.depth + 1,
            path=state.path + (path_item,),
            public_history_fingerprint=history,
        )

    def terminal_utility(self, state: OfficialSearchGameState, player: int) -> float:
        self._validate_state(state)
        if not state.terminal or player not in (0, 1):
            raise ValueError("terminal utility requires terminal state and player")
        result = _safe_int(_state(state.observation).get("result"), -1)
        if result == 2:
            return 0.0
        if result not in (0, 1):
            raise RuntimeError("terminal native state has invalid result")
        return 1.0 if result == player else -1.0
