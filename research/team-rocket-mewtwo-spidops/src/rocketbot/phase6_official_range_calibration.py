"""Phase 6AD observer-oriented official hidden-card range calibration.

Phase 6AC proved that a realized official hidden state can be joined to its
complete behavior prefix and native value surface at one exact root.  Its 18
roots were too small for range calibration.  This module collects a larger,
split-isolated corpus at the same later-game boundary and views every public
root from player 0 (the Rocket Mewtwo seat), even though the procedural prompt
belongs to player 1.

Only public state enters the range model.  Visualizer truth is transient
supervision used to produce 32-bucket hidden hand/prize/deck count labels; raw
card identities are never serialized or passed to the provider.  A public-
consistent allocation sampler supplies per-candidate hidden-resource marginals.
The held-out audit is scoped to six declared decks, three baseline styles, and
the player-1 second-MAIN collection boundary.  It cannot calibrate opponent
intent, prove abstraction fidelity, supply counterfactual values, authorize a
resolver, or control the official agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .competition import (
    ARCHETYPE_SIGNATURES,
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    _safe_int,
    _state,
    resolve_official_deck,
)
from .game_theory import OpponentRange, RangeHypothesis
from .native_blueprint_features import (
    NATIVE_CARD_HASH_BUCKETS,
    NATIVE_PUBLIC_FEATURE_NAMES,
    card_bucket,
    native_observer_public_feature_items,
)
from .official_bridge import (
    official_private_observation_payload,
    sanitize_official_public_state,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    rocket_compatibility_profile,
)
from .official_search_game import sample_hidden_zone_completion
from .phase6_aligned_counterfactual_values import ALIGNED_VALUE_SPLITS
from .phase6_aligned_public_range import (
    PUBLIC_RANGE_MINIMUM_CALIBRATION_ROOTS,
    PUBLIC_RANGE_MINIMUM_VALIDATION_ROOTS,
    PUBLIC_RANGE_TEMPERATURE_GRID,
    PUBLIC_RANGE_UNIFORM_MIX_GRID,
)
from .phase6_joint_official_values import (
    JointOfficialValueDataset,
    _official_truth_completion,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_official_prefix_reach import (
    OFFICIAL_PREFIX_CAPTURE_CONFIG,
    OfficialPrefixReachTrace,
    _capture_official_prefix,
)
from .official_search_game import _root_public_history
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_OFFICIAL_RANGE_SCHEMA_VERSION = 1
PHASE6_OFFICIAL_RANGE_SEED = 6_036
OFFICIAL_RANGE_OBSERVER_PLAYER = 0
OFFICIAL_RANGE_ROOT_ACTOR = 1
OFFICIAL_RANGE_ROOTS_PER_CANDIDATE_PER_SPLIT = 10
OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE = 32
OFFICIAL_RANGE_ADJUSTMENT_PENALTY_GRID = (0.0, 0.25, 1.0, 4.0, 16.0)
OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT = 0.15
OFFICIAL_RANGE_RESOURCE_ECE_LIMIT = 0.10
OFFICIAL_RANGE_ZONES = ("hand", "prize", "deck")
OFFICIAL_RANGE_RESOURCE_NAMES = tuple(
    f"{zone}_bucket_{bucket:02d}"
    for zone in OFFICIAL_RANGE_ZONES
    for bucket in range(NATIVE_CARD_HASH_BUCKETS)
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8],
        "big",
    )


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _uniform_candidate_range(candidate_ids: Sequence[str]) -> OpponentRange:
    probability = 1.0 / len(candidate_ids)
    return OpponentRange(
        tuple(
            RangeHypothesis(
                hypothesis_id=f"candidate:{candidate}",
                weight=probability,
                hidden_resource_counts=(),
                intent_id="unknown",
                style_id="unknown",
                source="uniform public feature-extraction prior",
            )
            for candidate in candidate_ids
        )
    )


def _resource_counts(
    *,
    hand: Sequence[int],
    prize: Sequence[int],
    deck: Sequence[int],
) -> tuple[int, ...]:
    values = []
    for cards in (hand, prize, deck):
        counts = [0] * NATIVE_CARD_HASH_BUCKETS
        for card_id in cards:
            counts[card_bucket(int(card_id))] += 1
        values.extend(counts)
    return tuple(values)


def _zone_sums(values: Sequence[int | float]) -> tuple[float, ...]:
    return tuple(
        float(sum(values[index * NATIVE_CARD_HASH_BUCKETS : (index + 1) * NATIVE_CARD_HASH_BUCKETS]))
        for index in range(len(OFFICIAL_RANGE_ZONES))
    )


def _observer_proxy(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return a public-orientation copy suitable for player-0 allocation sampling."""

    proxy = copy.deepcopy(dict(observation))
    current = proxy.get("current")
    if not isinstance(current, dict):
        raise ValueError("official range root has no mutable current state")
    current["yourIndex"] = OFFICIAL_RANGE_OBSERVER_PLAYER
    players = current.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise ValueError("official range observer proxy requires two mutable players")
    hidden_opponent = players[1 - OFFICIAL_RANGE_OBSERVER_PLAYER]
    if not isinstance(hidden_opponent, dict):
        raise ValueError("official range observer proxy opponent is malformed")
    # The source prompt belongs to player 1 and therefore exposes player 1's
    # hand in its private payload.  That hand is hidden from observer/player 0.
    # Retain the public count while removing every identity before sampling.
    hidden_opponent["hand"] = [
        None for _ in range(_safe_int(hidden_opponent.get("handCount"), 0))
    ]
    current["looking"] = []
    # The procedural prompt belongs to player 1 and can contain their private
    # search payload.  Allocation sampling needs no selection payload here.
    proxy["select"] = {}
    return proxy


@dataclass(frozen=True, slots=True)
class CandidateParticleSummary:
    candidate_id: str
    candidate_deck_fingerprint: str
    particle_count: int
    expected_resource_counts: tuple[float, ...]
    resource_presence_probabilities: tuple[float, ...]
    mean_public_evidence_adjustments: float
    maximum_public_evidence_adjustments: int
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id or self.particle_count <= 0:
            raise ValueError("candidate particle summary identity/count is invalid")
        _validate_digest(
            self.candidate_deck_fingerprint,
            "candidate_deck_fingerprint",
        )
        if len(self.expected_resource_counts) != len(OFFICIAL_RANGE_RESOURCE_NAMES):
            raise ValueError("candidate expected-resource width is invalid")
        if len(self.resource_presence_probabilities) != len(OFFICIAL_RANGE_RESOURCE_NAMES):
            raise ValueError("candidate resource-presence width is invalid")
        if any(value < 0 or not math.isfinite(value) for value in self.expected_resource_counts):
            raise ValueError("candidate expected-resource values are invalid")
        if any(
            value < 0 or value > 1 or not math.isfinite(value)
            for value in self.resource_presence_probabilities
        ):
            raise ValueError("candidate resource-presence probabilities are invalid")
        if (
            self.mean_public_evidence_adjustments < 0
            or not math.isfinite(self.mean_public_evidence_adjustments)
            or self.maximum_public_evidence_adjustments < 0
        ):
            raise ValueError("candidate public-evidence adjustments are invalid")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("candidate particle summary persisted raw private IDs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_deck_fingerprint": self.candidate_deck_fingerprint,
            "particle_count": self.particle_count,
            "expected_resource_counts": [
                [name, value]
                for name, value in zip(
                    OFFICIAL_RANGE_RESOURCE_NAMES,
                    self.expected_resource_counts,
                    strict=True,
                )
            ],
            "resource_presence_probabilities": [
                [name, value]
                for name, value in zip(
                    OFFICIAL_RANGE_RESOURCE_NAMES,
                    self.resource_presence_probabilities,
                    strict=True,
                )
            ],
            "public_evidence_adjustments": {
                "mean": self.mean_public_evidence_adjustments,
                "maximum": self.maximum_public_evidence_adjustments,
            },
            "claims": {
                "public_consistent_allocation_sampler": True,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateParticleSummary":
        if not bool(raw["claims"]["public_consistent_allocation_sampler"]):
            raise ValueError("candidate particle sampler provenance changed")
        expected = tuple(raw["expected_resource_counts"])
        presence = tuple(raw["resource_presence_probabilities"])
        if tuple(str(item[0]) for item in expected) != OFFICIAL_RANGE_RESOURCE_NAMES:
            raise ValueError("candidate expected-resource names changed")
        if tuple(str(item[0]) for item in presence) != OFFICIAL_RANGE_RESOURCE_NAMES:
            raise ValueError("candidate resource-presence names changed")
        adjustments = raw["public_evidence_adjustments"]
        return cls(
            candidate_id=str(raw["candidate_id"]),
            candidate_deck_fingerprint=str(raw["candidate_deck_fingerprint"]),
            particle_count=int(raw["particle_count"]),
            expected_resource_counts=tuple(float(item[1]) for item in expected),
            resource_presence_probabilities=tuple(float(item[1]) for item in presence),
            mean_public_evidence_adjustments=float(adjustments["mean"]),
            maximum_public_evidence_adjustments=int(adjustments["maximum"]),
            raw_private_card_ids_persisted=int(
                raw["claims"]["raw_private_card_ids_persisted"]
            ),
        )


@dataclass(frozen=True, slots=True)
class OfficialRangeTruthExample:
    example_id: str
    source_root_id: str
    split: str
    candidate_id_supervision_only: str
    collection_ordinal: int
    prefix_trace: OfficialPrefixReachTrace
    public_state_fingerprint: str
    public_history_fingerprint: str
    public_abstraction_fingerprint: str
    public_features: tuple[tuple[str, float], ...]
    truth_sample_fingerprint: str
    truth_resource_counts: tuple[int, ...]
    truth_zone_counts: tuple[int, int, int]
    candidate_particle_summaries: tuple[CandidateParticleSummary, ...]
    observer_player: int = OFFICIAL_RANGE_OBSERVER_PLAYER
    root_actor: int = OFFICIAL_RANGE_ROOT_ACTOR
    visualizer_public_parity: bool = True
    visualizer_acting_private_parity: bool = True
    target_candidate_identifier_in_features: bool = False
    truth_resource_counts_selected_for_model_input: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "source_root_id",
            "public_state_fingerprint",
            "public_history_fingerprint",
            "public_abstraction_fingerprint",
            "truth_sample_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.split not in ALIGNED_VALUE_SPLITS or not self.candidate_id_supervision_only:
            raise ValueError("official range example split/candidate is invalid")
        if self.collection_ordinal < 0:
            raise ValueError("official range collection ordinal is invalid")
        if self.prefix_trace.split != self.split:
            raise ValueError("official range prefix split mismatch")
        if self.prefix_trace.candidate_id != self.candidate_id_supervision_only:
            raise ValueError("official range prefix candidate mismatch")
        if self.prefix_trace.root_public_state_fingerprint != self.public_state_fingerprint:
            raise ValueError("official range prefix public-state mismatch")
        if self.prefix_trace.root_public_history_fingerprint != self.public_history_fingerprint:
            raise ValueError("official range prefix public-history mismatch")
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("official range public feature schema mismatch")
        if _fingerprint(self.public_features) != self.public_abstraction_fingerprint:
            raise ValueError("official range public abstraction fingerprint mismatch")
        if len(self.truth_resource_counts) != len(OFFICIAL_RANGE_RESOURCE_NAMES):
            raise ValueError("official range truth-resource width is invalid")
        if any(count < 0 for count in self.truth_resource_counts):
            raise ValueError("official range truth-resource count is invalid")
        if tuple(int(value) for value in _zone_sums(self.truth_resource_counts)) != self.truth_zone_counts:
            raise ValueError("official range truth zone/resource accounting mismatch")
        identifiers = tuple(item.candidate_id for item in self.candidate_particle_summaries)
        if len(identifiers) != len(set(identifiers)) or self.candidate_id_supervision_only not in identifiers:
            raise ValueError("official range candidate particle support is invalid")
        expected_zone_sums = tuple(float(value) for value in self.truth_zone_counts)
        for summary in self.candidate_particle_summaries:
            if any(
                not math.isclose(observed, expected, abs_tol=1e-8)
                for observed, expected in zip(
                    _zone_sums(summary.expected_resource_counts),
                    expected_zone_sums,
                    strict=True,
                )
            ):
                raise ValueError("candidate particle zone counts do not match public root")
        true_summary = next(
            item
            for item in self.candidate_particle_summaries
            if item.candidate_id == self.candidate_id_supervision_only
        )
        if true_summary.maximum_public_evidence_adjustments != 0:
            raise ValueError("true candidate required a public-evidence deck adjustment")
        if (self.observer_player, self.root_actor) != (
            OFFICIAL_RANGE_OBSERVER_PLAYER,
            OFFICIAL_RANGE_ROOT_ACTOR,
        ):
            raise ValueError("official range observer/root orientation changed")
        if not self.visualizer_public_parity or not self.visualizer_acting_private_parity:
            raise ValueError("official range visualizer parity failed")
        if any(
            (
                self.target_candidate_identifier_in_features,
                self.truth_resource_counts_selected_for_model_input,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official range example promoted hidden supervision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_root_id": self.source_root_id,
            "split": self.split,
            "candidate_id_supervision_only": self.candidate_id_supervision_only,
            "collection_ordinal": self.collection_ordinal,
            "prefix_trace": self.prefix_trace.to_dict(),
            "public": {
                "state_fingerprint": self.public_state_fingerprint,
                "history_fingerprint": self.public_history_fingerprint,
                "abstraction_fingerprint": self.public_abstraction_fingerprint,
                "features": [[name, value] for name, value in self.public_features],
                "observer_player": self.observer_player,
                "root_actor": self.root_actor,
            },
            "truth_supervision": {
                "sample_fingerprint": self.truth_sample_fingerprint,
                "resource_counts": [
                    [name, count]
                    for name, count in zip(
                        OFFICIAL_RANGE_RESOURCE_NAMES,
                        self.truth_resource_counts,
                        strict=True,
                    )
                ],
                "zone_counts": dict(zip(OFFICIAL_RANGE_ZONES, self.truth_zone_counts, strict=True)),
            },
            "candidate_particle_summaries": [
                item.to_dict() for item in self.candidate_particle_summaries
            ],
            "gates": {
                "visualizer_public_parity": self.visualizer_public_parity,
                "visualizer_acting_private_parity": self.visualizer_acting_private_parity,
            },
            "claims": {
                "public_features_only": True,
                "target_candidate_identifier_in_features": self.target_candidate_identifier_in_features,
                "truth_resource_counts_selected_for_model_input": (
                    self.truth_resource_counts_selected_for_model_input
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
                "realized_hidden_state_official_sample": True,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialRangeTruthExample":
        public = raw["public"]
        truth = raw["truth_supervision"]
        claims = raw["claims"]
        gates = raw["gates"]
        resources = tuple(truth["resource_counts"])
        if tuple(str(item[0]) for item in resources) != OFFICIAL_RANGE_RESOURCE_NAMES:
            raise ValueError("official range truth-resource names changed")
        if not bool(claims["public_features_only"]):
            raise ValueError("official range public-only claim changed")
        return cls(
            example_id=str(raw["example_id"]),
            source_root_id=str(raw["source_root_id"]),
            split=str(raw["split"]),
            candidate_id_supervision_only=str(raw["candidate_id_supervision_only"]),
            collection_ordinal=int(raw["collection_ordinal"]),
            prefix_trace=OfficialPrefixReachTrace.from_dict(raw["prefix_trace"]),
            public_state_fingerprint=str(public["state_fingerprint"]),
            public_history_fingerprint=str(public["history_fingerprint"]),
            public_abstraction_fingerprint=str(public["abstraction_fingerprint"]),
            public_features=tuple(
                (str(name), float(value)) for name, value in public["features"]
            ),
            truth_sample_fingerprint=str(truth["sample_fingerprint"]),
            truth_resource_counts=tuple(int(item[1]) for item in resources),
            truth_zone_counts=tuple(
                int(truth["zone_counts"][zone]) for zone in OFFICIAL_RANGE_ZONES
            ),
            candidate_particle_summaries=tuple(
                CandidateParticleSummary.from_dict(item)
                for item in raw["candidate_particle_summaries"]
            ),
            observer_player=int(public["observer_player"]),
            root_actor=int(public["root_actor"]),
            visualizer_public_parity=bool(gates["visualizer_public_parity"]),
            visualizer_acting_private_parity=bool(
                gates["visualizer_acting_private_parity"]
            ),
            target_candidate_identifier_in_features=bool(
                claims["target_candidate_identifier_in_features"]
            ),
            truth_resource_counts_selected_for_model_input=bool(
                claims["truth_resource_counts_selected_for_model_input"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )


@dataclass(frozen=True, slots=True)
class OfficialRangeTruthDataset:
    source_phase6ac_dataset_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    roots_per_candidate_per_split: int
    particles_per_candidate: int
    examples: tuple[OfficialRangeTruthExample, ...]
    collection_split_disjoint: bool
    observer_orientation_validated: bool
    official_battle_roots_executed: bool = True
    visualizer_truth_supervision_only: bool = True
    model_transformation_official_engine_executed: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_phase6ac_dataset_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("official range candidate catalog is invalid")
        if tuple(candidate for candidate, _ in self.candidate_deck_fingerprints) != self.candidate_ids:
            raise ValueError("official range candidate deck catalog is misaligned")
        for _, fingerprint in self.candidate_deck_fingerprints:
            _validate_digest(fingerprint, "candidate_deck_fingerprint")
        if self.roots_per_candidate_per_split <= 0 or self.particles_per_candidate <= 1:
            raise ValueError("official range collection sizes are invalid")
        expected_count = (
            len(ALIGNED_VALUE_SPLITS)
            * len(self.candidate_ids)
            * self.roots_per_candidate_per_split
        )
        if len(self.examples) != expected_count:
            raise ValueError("official range example count is incomplete")
        expected_support = {
            (split, candidate, ordinal)
            for split in ALIGNED_VALUE_SPLITS
            for candidate in self.candidate_ids
            for ordinal in range(self.roots_per_candidate_per_split)
        }
        observed_support = {
            (item.split, item.candidate_id_supervision_only, item.collection_ordinal)
            for item in self.examples
        }
        if observed_support != expected_support:
            raise ValueError("official range split/candidate/ordinal support is incomplete")
        if len({item.example_id for item in self.examples}) != len(self.examples):
            raise ValueError("official range example IDs are duplicated")
        if len({item.source_root_id for item in self.examples}) != len(self.examples):
            raise ValueError("official range source root IDs are duplicated")
        split_roots = {
            split: {
                item.source_root_id for item in self.examples if item.split == split
            }
            for split in ALIGNED_VALUE_SPLITS
        }
        observed_disjoint = all(
            split_roots[left].isdisjoint(split_roots[right])
            for index, left in enumerate(ALIGNED_VALUE_SPLITS)
            for right in ALIGNED_VALUE_SPLITS[index + 1 :]
        )
        if self.collection_split_disjoint != observed_disjoint or not observed_disjoint:
            raise ValueError("official range collection splits overlap")
        if not all(
            (
                self.observer_orientation_validated,
                self.official_battle_roots_executed,
                self.visualizer_truth_supervision_only,
            )
        ):
            raise ValueError("official range genuine/source gates failed")
        if self.model_transformation_official_engine_executed or self.raw_private_card_ids_persisted != 0:
            raise ValueError("official range dataset promoted an invalid claim")
        if any(
            tuple(summary.candidate_id for summary in item.candidate_particle_summaries)
            != self.candidate_ids
            or any(
                summary.particle_count != self.particles_per_candidate
                for summary in item.candidate_particle_summaries
            )
            for item in self.examples
        ):
            raise ValueError("official range candidate particle grid is incomplete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def live_selection_count(self) -> int:
        return sum(len(item.prefix_trace.steps) for item in self.examples)

    @property
    def roots_per_split(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (split, sum(item.split == split for item in self.examples))
            for split in ALIGNED_VALUE_SPLITS
        )

    @property
    def repeated_public_abstraction_groups(self) -> int:
        counts: dict[str, int] = {}
        for item in self.examples:
            counts[item.public_abstraction_fingerprint] = (
                counts.get(item.public_abstraction_fingerprint, 0) + 1
            )
        return sum(count > 1 for count in counts.values())

    @property
    def maximum_samples_per_public_abstraction(self) -> int:
        counts: dict[str, int] = {}
        for item in self.examples:
            counts[item.public_abstraction_fingerprint] = (
                counts.get(item.public_abstraction_fingerprint, 0) + 1
            )
        return max(counts.values())

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_RANGE_SCHEMA_VERSION,
            "source_phase6ac_dataset_fingerprint": self.source_phase6ac_dataset_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": [
                {"candidate_id": candidate, "fingerprint": fingerprint}
                for candidate, fingerprint in self.candidate_deck_fingerprints
            ],
            "collection": {
                "roots_per_candidate_per_split": self.roots_per_candidate_per_split,
                "particles_per_candidate": self.particles_per_candidate,
                "root_count": len(self.examples),
                "roots_per_split": dict(self.roots_per_split),
                "live_selection_count": self.live_selection_count,
                "repeated_public_abstraction_groups": self.repeated_public_abstraction_groups,
                "maximum_samples_per_public_abstraction": (
                    self.maximum_samples_per_public_abstraction
                ),
            },
            "examples": [item.to_dict() for item in self.examples],
            "gates": {
                "collection_split_disjoint": self.collection_split_disjoint,
                "observer_orientation_validated": self.observer_orientation_validated,
            },
            "claims": {
                "official_battle_roots_executed": self.official_battle_roots_executed,
                "visualizer_truth_supervision_only": self.visualizer_truth_supervision_only,
                "model_transformation_official_engine_executed": (
                    self.model_transformation_official_engine_executed
                ),
                "public_model_inputs_only": True,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialRangeTruthDataset":
        collection = raw["collection"]
        gates = raw["gates"]
        claims = raw["claims"]
        if not bool(claims["public_model_inputs_only"]):
            raise ValueError("official range public-only model claim changed")
        result = cls(
            source_phase6ac_dataset_fingerprint=str(
                raw["source_phase6ac_dataset_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            candidate_deck_fingerprints=tuple(
                (str(item["candidate_id"]), str(item["fingerprint"]))
                for item in raw["candidate_deck_fingerprints"]
            ),
            roots_per_candidate_per_split=int(
                collection["roots_per_candidate_per_split"]
            ),
            particles_per_candidate=int(collection["particles_per_candidate"]),
            examples=tuple(
                OfficialRangeTruthExample.from_dict(item) for item in raw["examples"]
            ),
            collection_split_disjoint=bool(gates["collection_split_disjoint"]),
            observer_orientation_validated=bool(gates["observer_orientation_validated"]),
            official_battle_roots_executed=bool(
                claims["official_battle_roots_executed"]
            ),
            visualizer_truth_supervision_only=bool(
                claims["visualizer_truth_supervision_only"]
            ),
            model_transformation_official_engine_executed=bool(
                claims["model_transformation_official_engine_executed"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected_collection = {
            "roots_per_candidate_per_split": result.roots_per_candidate_per_split,
            "particles_per_candidate": result.particles_per_candidate,
            "root_count": len(result.examples),
            "roots_per_split": dict(result.roots_per_split),
            "live_selection_count": result.live_selection_count,
            "repeated_public_abstraction_groups": (
                result.repeated_public_abstraction_groups
            ),
            "maximum_samples_per_public_abstraction": (
                result.maximum_samples_per_public_abstraction
            ),
        }
        if dict(collection) != expected_collection:
            raise ValueError("official range collection accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official range dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OfficialRangeSelectionMetric:
    temperature: float
    uniform_mix: float
    adjustment_penalty: float
    calibration_nll: float
    calibration_brier: float
    calibration_top1_accuracy: float

    def __post_init__(self) -> None:
        if self.temperature not in PUBLIC_RANGE_TEMPERATURE_GRID:
            raise ValueError("official range temperature is unsupported")
        if self.uniform_mix not in PUBLIC_RANGE_UNIFORM_MIX_GRID:
            raise ValueError("official range uniform mixture is unsupported")
        if self.adjustment_penalty not in OFFICIAL_RANGE_ADJUSTMENT_PENALTY_GRID:
            raise ValueError("official range adjustment penalty is unsupported")
        if any(
            value < 0 or not math.isfinite(value)
            for value in (self.calibration_nll, self.calibration_brier)
        ):
            raise ValueError("official range selection loss is invalid")
        if not 0 <= self.calibration_top1_accuracy <= 1:
            raise ValueError("official range selection accuracy is invalid")


@dataclass(frozen=True, slots=True)
class OfficialPublicRangeModel:
    source_dataset_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    normalizer: NamedFeatureNormalizer
    prototypes: tuple[tuple[str, tuple[float, ...]], ...]
    selection_metrics: tuple[OfficialRangeSelectionMetric, ...]
    selected_temperature: float
    selected_uniform_mix: float
    selected_adjustment_penalty: float
    train_root_ids: tuple[str, ...]
    calibration_root_ids: tuple[str, ...]
    train_rows_fingerprint: str
    observer_player: int = OFFICIAL_RANGE_OBSERVER_PLAYER
    public_features_only: bool = True
    selected_truth_identifier_input: bool = False
    hidden_resource_particle_marginals: bool = True
    opponent_intent_modeled: bool = False
    official_engine_executed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_dataset_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "train_rows_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if tuple(candidate for candidate, _ in self.candidate_deck_fingerprints) != self.candidate_ids:
            raise ValueError("official range model candidate catalog is misaligned")
        if self.normalizer.names != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("official range model feature schema mismatch")
        if self.normalizer.fit_rows_fingerprint != self.train_rows_fingerprint:
            raise ValueError("official range normalizer provenance mismatch")
        if tuple(candidate for candidate, _ in self.prototypes) != self.candidate_ids:
            raise ValueError("official range prototype support is invalid")
        if any(len(values) != len(NATIVE_PUBLIC_FEATURE_NAMES) for _, values in self.prototypes):
            raise ValueError("official range prototype width is invalid")
        expected_grid = tuple(
            (temperature, uniform_mix, penalty)
            for temperature in PUBLIC_RANGE_TEMPERATURE_GRID
            for uniform_mix in PUBLIC_RANGE_UNIFORM_MIX_GRID
            for penalty in OFFICIAL_RANGE_ADJUSTMENT_PENALTY_GRID
        )
        if tuple(
            (item.temperature, item.uniform_mix, item.adjustment_penalty)
            for item in self.selection_metrics
        ) != expected_grid:
            raise ValueError("official range selection grid is incomplete")
        selected = min(
            self.selection_metrics,
            key=lambda item: (
                item.calibration_nll,
                item.calibration_brier,
                item.temperature,
                item.uniform_mix,
                item.adjustment_penalty,
            ),
        )
        if (
            self.selected_temperature,
            self.selected_uniform_mix,
            self.selected_adjustment_penalty,
        ) != (
            selected.temperature,
            selected.uniform_mix,
            selected.adjustment_penalty,
        ):
            raise ValueError("official range selected configuration is not optimal")
        if not self.train_root_ids or not self.calibration_root_ids:
            raise ValueError("official range model split roots are missing")
        if set(self.train_root_ids).intersection(self.calibration_root_ids):
            raise ValueError("official range model train/calibration roots overlap")
        if self.observer_player != OFFICIAL_RANGE_OBSERVER_PLAYER:
            raise ValueError("official range model observer orientation changed")
        if not self.public_features_only or not self.hidden_resource_particle_marginals:
            raise ValueError("official range model scope claims changed")
        if any(
            (
                self.selected_truth_identifier_input,
                self.opponent_intent_modeled,
                self.official_engine_executed,
            )
        ):
            raise ValueError("official range model promoted an invalid claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def candidate_probabilities(
        self,
        public_features: Sequence[tuple[str, float]],
        summaries: Sequence[CandidateParticleSummary],
    ) -> tuple[tuple[str, float], ...]:
        if tuple(item.candidate_id for item in summaries) != self.candidate_ids:
            raise ValueError("official range prediction candidate summaries are misaligned")
        values = self.normalizer.transform(public_features)
        logits = []
        for (candidate, prototype), summary in zip(
            self.prototypes,
            summaries,
            strict=True,
        ):
            distance = sum(
                (value - reference) ** 2
                for value, reference in zip(values, prototype, strict=True)
            )
            logits.append(
                (
                    candidate,
                    -distance / self.selected_temperature
                    - self.selected_adjustment_penalty
                    * summary.mean_public_evidence_adjustments,
                )
            )
        maximum = max(value for _, value in logits)
        exponentials = [(candidate, math.exp(value - maximum)) for candidate, value in logits]
        total = sum(value for _, value in exponentials)
        uniform = 1.0 / len(exponentials)
        return tuple(
            (
                candidate,
                (1.0 - self.selected_uniform_mix) * value / total
                + self.selected_uniform_mix * uniform,
            )
            for candidate, value in exponentials
        )

    def opponent_range(
        self,
        public_features: Sequence[tuple[str, float]],
        summaries: Sequence[CandidateParticleSummary],
    ) -> OpponentRange:
        probabilities = dict(self.candidate_probabilities(public_features, summaries))
        return OpponentRange(
            tuple(
                RangeHypothesis(
                    hypothesis_id=f"official-range:{summary.candidate_id}",
                    weight=probabilities[summary.candidate_id],
                    hidden_resource_counts=tuple(
                        (name, max(0, int(round(value))))
                        for name, value in zip(
                            OFFICIAL_RANGE_RESOURCE_NAMES,
                            summary.expected_resource_counts,
                            strict=True,
                        )
                    ),
                    intent_id="unknown",
                    style_id="unknown",
                    source=(
                        "public-only candidate posterior with public-consistent "
                        "hidden-allocation particle marginals"
                    ),
                )
                for summary in summaries
            )
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_RANGE_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": [
                {"candidate_id": candidate, "fingerprint": fingerprint}
                for candidate, fingerprint in self.candidate_deck_fingerprints
            ],
            "normalizer": self.normalizer.to_dict(),
            "prototypes": [
                {"candidate_id": candidate, "values": list(values)}
                for candidate, values in self.prototypes
            ],
            "selection_grid": {
                "temperatures": list(PUBLIC_RANGE_TEMPERATURE_GRID),
                "uniform_mixes": list(PUBLIC_RANGE_UNIFORM_MIX_GRID),
                "adjustment_penalties": list(OFFICIAL_RANGE_ADJUSTMENT_PENALTY_GRID),
                "metrics": [asdict(item) for item in self.selection_metrics],
                "selected": {
                    "temperature": self.selected_temperature,
                    "uniform_mix": self.selected_uniform_mix,
                    "adjustment_penalty": self.selected_adjustment_penalty,
                },
            },
            "split_provenance": {
                "train_root_ids": list(self.train_root_ids),
                "calibration_root_ids": list(self.calibration_root_ids),
                "train_rows_fingerprint": self.train_rows_fingerprint,
            },
            "observer_player": self.observer_player,
            "claims": {
                "public_features_only": self.public_features_only,
                "selected_truth_identifier_input": self.selected_truth_identifier_input,
                "hidden_resource_particle_marginals": self.hidden_resource_particle_marginals,
                "opponent_intent_modeled": self.opponent_intent_modeled,
                "official_engine_executed": self.official_engine_executed,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPublicRangeModel":
        grid = raw["selection_grid"]
        selected = grid["selected"]
        split = raw["split_provenance"]
        claims = raw["claims"]
        if tuple(float(value) for value in grid["temperatures"]) != PUBLIC_RANGE_TEMPERATURE_GRID:
            raise ValueError("official range temperature grid changed")
        if tuple(float(value) for value in grid["uniform_mixes"]) != PUBLIC_RANGE_UNIFORM_MIX_GRID:
            raise ValueError("official range uniform mixture grid changed")
        if tuple(float(value) for value in grid["adjustment_penalties"]) != OFFICIAL_RANGE_ADJUSTMENT_PENALTY_GRID:
            raise ValueError("official range adjustment penalty grid changed")
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            candidate_deck_fingerprints=tuple(
                (str(item["candidate_id"]), str(item["fingerprint"]))
                for item in raw["candidate_deck_fingerprints"]
            ),
            normalizer=NamedFeatureNormalizer.from_dict(raw["normalizer"]),
            prototypes=tuple(
                (
                    str(item["candidate_id"]),
                    tuple(float(value) for value in item["values"]),
                )
                for item in raw["prototypes"]
            ),
            selection_metrics=tuple(
                OfficialRangeSelectionMetric(**dict(item))
                for item in grid["metrics"]
            ),
            selected_temperature=float(selected["temperature"]),
            selected_uniform_mix=float(selected["uniform_mix"]),
            selected_adjustment_penalty=float(selected["adjustment_penalty"]),
            train_root_ids=tuple(str(value) for value in split["train_root_ids"]),
            calibration_root_ids=tuple(
                str(value) for value in split["calibration_root_ids"]
            ),
            train_rows_fingerprint=str(split["train_rows_fingerprint"]),
            observer_player=int(raw["observer_player"]),
            public_features_only=bool(claims["public_features_only"]),
            selected_truth_identifier_input=bool(
                claims["selected_truth_identifier_input"]
            ),
            hidden_resource_particle_marginals=bool(
                claims["hidden_resource_particle_marginals"]
            ),
            opponent_intent_modeled=bool(claims["opponent_intent_modeled"]),
            official_engine_executed=bool(claims["official_engine_executed"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official range model fingerprint mismatch")
        return result


def _mixed_resource_predictions(
    probabilities: Sequence[tuple[str, float]],
    summaries: Sequence[CandidateParticleSummary],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    weights = dict(probabilities)
    expected = []
    presence = []
    for index in range(len(OFFICIAL_RANGE_RESOURCE_NAMES)):
        expected.append(
            sum(
                weights[item.candidate_id] * item.expected_resource_counts[index]
                for item in summaries
            )
        )
        # A convex mixture is mathematically in [0, 1], but the six-term
        # floating-point sum can land at 1.0 + one ulp.  Canonicalize that
        # representation before the checked probability artifact is built.
        mixed_presence = sum(
            weights[item.candidate_id]
            * item.resource_presence_probabilities[index]
            for item in summaries
        )
        presence.append(min(1.0, max(0.0, mixed_presence)))
    return tuple(expected), tuple(presence)


@dataclass(frozen=True, slots=True)
class OfficialRangeEvaluationRow:
    example_id: str
    source_root_id: str
    split: str
    target_candidate_id_supervision_only: str
    candidate_probabilities: tuple[tuple[str, float], ...]
    predicted_resource_counts: tuple[float, ...]
    resource_presence_probabilities: tuple[float, ...]
    uniform_candidate_resource_presence_probabilities: tuple[float, ...]
    oracle_candidate_resource_presence_probabilities: tuple[float, ...]
    truth_resource_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.example_id, "example_id")
        _validate_digest(self.source_root_id, "source_root_id")
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("official range evaluation split is invalid")
        identifiers = tuple(candidate for candidate, _ in self.candidate_probabilities)
        if self.target_candidate_id_supervision_only not in identifiers:
            raise ValueError("official range evaluation target is outside support")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("official range evaluation candidates are duplicated")
        if not math.isclose(sum(value for _, value in self.candidate_probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("official range candidate probabilities are not normalized")
        width = len(OFFICIAL_RANGE_RESOURCE_NAMES)
        if any(
            len(values) != width
            for values in (
                self.predicted_resource_counts,
                self.resource_presence_probabilities,
                self.uniform_candidate_resource_presence_probabilities,
                self.oracle_candidate_resource_presence_probabilities,
                self.truth_resource_counts,
            )
        ):
            raise ValueError("official range evaluation resource width is invalid")
        if any(value < 0 or not math.isfinite(value) for value in self.predicted_resource_counts):
            raise ValueError("official range predicted resource count is invalid")
        for values in (
            self.resource_presence_probabilities,
            self.uniform_candidate_resource_presence_probabilities,
            self.oracle_candidate_resource_presence_probabilities,
        ):
            if any(value < 0 or value > 1 or not math.isfinite(value) for value in values):
                raise ValueError("official range resource probability is invalid")

    @property
    def nll(self) -> float:
        probability = dict(self.candidate_probabilities)[
            self.target_candidate_id_supervision_only
        ]
        return -math.log(max(probability, 1e-15))

    @property
    def candidate_brier(self) -> float:
        return sum(
            (probability - float(candidate == self.target_candidate_id_supervision_only)) ** 2
            for candidate, probability in self.candidate_probabilities
        )

    @property
    def confidence(self) -> float:
        return max(value for _, value in self.candidate_probabilities)

    @property
    def top1_correct(self) -> bool:
        predicted = min(
            self.candidate_probabilities,
            key=lambda item: (-item[1], item[0]),
        )[0]
        return predicted == self.target_candidate_id_supervision_only

    @property
    def truth_presence(self) -> tuple[float, ...]:
        return tuple(float(value > 0) for value in self.truth_resource_counts)

    @staticmethod
    def _brier(predicted: Sequence[float], truth: Sequence[float]) -> float:
        return sum((left - right) ** 2 for left, right in zip(predicted, truth, strict=True)) / len(truth)

    @property
    def resource_presence_brier(self) -> float:
        return self._brier(self.resource_presence_probabilities, self.truth_presence)

    @property
    def uniform_candidate_resource_brier(self) -> float:
        return self._brier(
            self.uniform_candidate_resource_presence_probabilities,
            self.truth_presence,
        )

    @property
    def oracle_candidate_resource_brier(self) -> float:
        return self._brier(
            self.oracle_candidate_resource_presence_probabilities,
            self.truth_presence,
        )

    @property
    def resource_count_mae(self) -> float:
        return sum(
            abs(predicted - truth)
            for predicted, truth in zip(
                self.predicted_resource_counts,
                self.truth_resource_counts,
                strict=True,
            )
        ) / len(self.truth_resource_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_root_id": self.source_root_id,
            "split": self.split,
            "target_candidate_id_supervision_only": self.target_candidate_id_supervision_only,
            "candidate_probabilities": [list(item) for item in self.candidate_probabilities],
            "predicted_resource_counts": list(self.predicted_resource_counts),
            "resource_presence_probabilities": list(self.resource_presence_probabilities),
            "uniform_candidate_resource_presence_probabilities": list(
                self.uniform_candidate_resource_presence_probabilities
            ),
            "oracle_candidate_resource_presence_probabilities": list(
                self.oracle_candidate_resource_presence_probabilities
            ),
            "truth_resource_counts": list(self.truth_resource_counts),
            "metrics": {
                "candidate_nll": self.nll,
                "candidate_brier": self.candidate_brier,
                "candidate_confidence": self.confidence,
                "candidate_top1_correct": self.top1_correct,
                "resource_presence_brier": self.resource_presence_brier,
                "uniform_candidate_resource_brier": self.uniform_candidate_resource_brier,
                "oracle_candidate_resource_brier": self.oracle_candidate_resource_brier,
                "resource_count_mae": self.resource_count_mae,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialRangeEvaluationRow":
        result = cls(
            example_id=str(raw["example_id"]),
            source_root_id=str(raw["source_root_id"]),
            split=str(raw["split"]),
            target_candidate_id_supervision_only=str(
                raw["target_candidate_id_supervision_only"]
            ),
            candidate_probabilities=tuple(
                (str(item[0]), float(item[1]))
                for item in raw["candidate_probabilities"]
            ),
            predicted_resource_counts=tuple(
                float(value) for value in raw["predicted_resource_counts"]
            ),
            resource_presence_probabilities=tuple(
                float(value) for value in raw["resource_presence_probabilities"]
            ),
            uniform_candidate_resource_presence_probabilities=tuple(
                float(value)
                for value in raw["uniform_candidate_resource_presence_probabilities"]
            ),
            oracle_candidate_resource_presence_probabilities=tuple(
                float(value)
                for value in raw["oracle_candidate_resource_presence_probabilities"]
            ),
            truth_resource_counts=tuple(
                int(value) for value in raw["truth_resource_counts"]
            ),
        )
        metrics = raw["metrics"]
        expected = {
            "candidate_nll": result.nll,
            "candidate_brier": result.candidate_brier,
            "candidate_confidence": result.confidence,
            "candidate_top1_correct": result.top1_correct,
            "resource_presence_brier": result.resource_presence_brier,
            "uniform_candidate_resource_brier": result.uniform_candidate_resource_brier,
            "oracle_candidate_resource_brier": result.oracle_candidate_resource_brier,
            "resource_count_mae": result.resource_count_mae,
        }
        for name, value in expected.items():
            observed = metrics[name]
            if isinstance(value, bool):
                if bool(observed) != value:
                    raise ValueError("official range evaluation boolean metric mismatch")
            elif not math.isclose(float(observed), value, abs_tol=1e-12):
                raise ValueError("official range evaluation numeric metric mismatch")
        return result


def _ece(pairs: Sequence[tuple[float, bool]], *, bins: int = 10) -> float:
    if not pairs:
        raise ValueError("ECE requires observations")
    result = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = tuple(
            (confidence, correct)
            for confidence, correct in pairs
            if lower <= confidence < upper
            or (index == bins - 1 and confidence <= upper)
        )
        if bucket:
            accuracy = sum(correct for _, correct in bucket) / len(bucket)
            confidence = sum(value for value, _ in bucket) / len(bucket)
            result += len(bucket) / len(pairs) * abs(accuracy - confidence)
    return result


@dataclass(frozen=True, slots=True)
class OfficialRangeSplitMetrics:
    split: str
    roots: int
    candidate_strata: int
    candidate_nll: float
    candidate_brier: float
    candidate_top1_accuracy: float
    candidate_expected_calibration_error: float
    resource_presence_brier: float
    uniform_candidate_resource_brier: float
    oracle_candidate_resource_brier: float
    resource_expected_calibration_error: float
    resource_count_mae: float

    def __post_init__(self) -> None:
        if self.split not in ALIGNED_VALUE_SPLITS or self.roots <= 0 or self.candidate_strata <= 0:
            raise ValueError("official range split metric identity is invalid")
        for name in (
            "candidate_nll",
            "candidate_brier",
            "candidate_expected_calibration_error",
            "resource_presence_brier",
            "uniform_candidate_resource_brier",
            "oracle_candidate_resource_brier",
            "resource_expected_calibration_error",
            "resource_count_mae",
        ):
            if getattr(self, name) < 0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"official range {name} is invalid")
        if not 0 <= self.candidate_top1_accuracy <= 1:
            raise ValueError("official range top-1 accuracy is invalid")


@dataclass(frozen=True, slots=True)
class OfficialRangeCalibration:
    dataset_fingerprint: str
    model_fingerprint: str
    evaluation_rows: tuple[OfficialRangeEvaluationRow, ...]
    train: OfficialRangeSplitMetrics
    calibration: OfficialRangeSplitMetrics
    validation: OfficialRangeSplitMetrics
    uniform_candidate_nll: float
    uniform_candidate_brier: float
    minimum_calibration_roots: int
    minimum_validation_roots: int
    candidate_ece_limit: float
    resource_ece_limit: float
    root_split_gate_passed: bool
    sufficient_calibration_roots_gate_passed: bool
    sufficient_validation_roots_gate_passed: bool
    observer_orientation_gate_passed: bool
    heldout_candidate_calibration_gate_passed: bool
    heldout_hidden_resource_calibration_gate_passed: bool
    in_scope_official_hidden_card_range_calibrated: bool
    opponent_intent_range_calibrated: bool = False
    full_official_public_belief_calibrated: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        _validate_digest(self.model_fingerprint, "model_fingerprint")
        if tuple(item.split for item in (self.train, self.calibration, self.validation)) != ALIGNED_VALUE_SPLITS:
            raise ValueError("official range split metric order is invalid")
        if len(self.evaluation_rows) != self.train.roots + self.calibration.roots + self.validation.roots:
            raise ValueError("official range evaluation row accounting is inconsistent")
        expected_in_scope = all(
            (
                self.root_split_gate_passed,
                self.sufficient_calibration_roots_gate_passed,
                self.sufficient_validation_roots_gate_passed,
                self.observer_orientation_gate_passed,
                self.heldout_candidate_calibration_gate_passed,
                self.heldout_hidden_resource_calibration_gate_passed,
            )
        )
        if self.in_scope_official_hidden_card_range_calibrated != expected_in_scope:
            raise ValueError("official range in-scope calibration gate is inconsistent")
        if any(
            (
                self.opponent_intent_range_calibrated,
                self.full_official_public_belief_calibrated,
                self.counterfactual_value_compatible,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("official range calibration promoted a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_RANGE_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "evaluation_rows": [item.to_dict() for item in self.evaluation_rows],
            "metrics": {
                "train": asdict(self.train),
                "calibration": asdict(self.calibration),
                "validation": asdict(self.validation),
                "uniform_candidate_nll": self.uniform_candidate_nll,
                "uniform_candidate_brier": self.uniform_candidate_brier,
            },
            "thresholds": {
                "minimum_calibration_roots": self.minimum_calibration_roots,
                "minimum_validation_roots": self.minimum_validation_roots,
                "candidate_ece_limit": self.candidate_ece_limit,
                "resource_ece_limit": self.resource_ece_limit,
            },
            "gates": {
                "root_split": self.root_split_gate_passed,
                "sufficient_calibration_roots": self.sufficient_calibration_roots_gate_passed,
                "sufficient_validation_roots": self.sufficient_validation_roots_gate_passed,
                "observer_orientation": self.observer_orientation_gate_passed,
                "heldout_candidate_calibration": self.heldout_candidate_calibration_gate_passed,
                "heldout_hidden_resource_calibration": (
                    self.heldout_hidden_resource_calibration_gate_passed
                ),
                "in_scope_official_hidden_card_range_calibrated": (
                    self.in_scope_official_hidden_card_range_calibrated
                ),
                "opponent_intent_range_calibrated": self.opponent_intent_range_calibrated,
                "full_official_public_belief_calibrated": (
                    self.full_official_public_belief_calibrated
                ),
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialRangeCalibration":
        metrics = raw["metrics"]
        thresholds = raw["thresholds"]
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            evaluation_rows=tuple(
                OfficialRangeEvaluationRow.from_dict(item)
                for item in raw["evaluation_rows"]
            ),
            train=OfficialRangeSplitMetrics(**dict(metrics["train"])),
            calibration=OfficialRangeSplitMetrics(**dict(metrics["calibration"])),
            validation=OfficialRangeSplitMetrics(**dict(metrics["validation"])),
            uniform_candidate_nll=float(metrics["uniform_candidate_nll"]),
            uniform_candidate_brier=float(metrics["uniform_candidate_brier"]),
            minimum_calibration_roots=int(thresholds["minimum_calibration_roots"]),
            minimum_validation_roots=int(thresholds["minimum_validation_roots"]),
            candidate_ece_limit=float(thresholds["candidate_ece_limit"]),
            resource_ece_limit=float(thresholds["resource_ece_limit"]),
            root_split_gate_passed=bool(gates["root_split"]),
            sufficient_calibration_roots_gate_passed=bool(
                gates["sufficient_calibration_roots"]
            ),
            sufficient_validation_roots_gate_passed=bool(
                gates["sufficient_validation_roots"]
            ),
            observer_orientation_gate_passed=bool(gates["observer_orientation"]),
            heldout_candidate_calibration_gate_passed=bool(
                gates["heldout_candidate_calibration"]
            ),
            heldout_hidden_resource_calibration_gate_passed=bool(
                gates["heldout_hidden_resource_calibration"]
            ),
            in_scope_official_hidden_card_range_calibrated=bool(
                gates["in_scope_official_hidden_card_range_calibrated"]
            ),
            opponent_intent_range_calibrated=bool(
                gates["opponent_intent_range_calibrated"]
            ),
            full_official_public_belief_calibrated=bool(
                gates["full_official_public_belief_calibrated"]
            ),
            counterfactual_value_compatible=bool(
                gates["counterfactual_value_compatible"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official range calibration fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialRangeEvidence:
    benchmark_id: str
    status: str
    source_phase6ac_dataset_fingerprint: str
    dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    roots_per_split: int
    candidate_strata: int
    live_selection_count: int
    particles_per_candidate: int
    total_public_consistent_particles: int
    repeated_public_abstraction_groups: int
    maximum_samples_per_public_abstraction: int
    selected_temperature: float
    selected_uniform_mix: float
    selected_adjustment_penalty: float
    validation_candidate_nll: float
    uniform_candidate_nll: float
    validation_candidate_brier: float
    uniform_candidate_brier: float
    validation_candidate_top1_accuracy: float
    validation_candidate_ece: float
    validation_resource_presence_brier: float
    validation_uniform_resource_brier: float
    validation_oracle_resource_brier: float
    validation_resource_ece: float
    validation_resource_count_mae: float
    in_scope_official_hidden_card_range_calibrated: bool
    opponent_intent_range_calibrated: bool
    full_official_public_belief_calibrated: bool
    counterfactual_value_compatible: bool
    safe_resolver_enabled: bool
    opponent_exploitation_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AD evidence identity is missing")
        for name in (
            "source_phase6ac_dataset_fingerprint",
            "dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if min(
            self.root_count,
            self.roots_per_split,
            self.candidate_strata,
            self.live_selection_count,
            self.particles_per_candidate,
            self.total_public_consistent_particles,
            self.maximum_samples_per_public_abstraction,
        ) <= 0:
            raise ValueError("Phase 6AD evidence accounting is invalid")
        if any(
            (
                self.opponent_intent_range_calibrated,
                self.full_official_public_belief_calibrated,
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AD evidence enabled a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_RANGE_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AD observer-oriented official range evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Fresh official truth roots | {self.root_count} |
| Train / calibration / validation roots | {self.roots_per_split} / {self.roots_per_split} / {self.roots_per_split} |
| Candidate strata | {self.candidate_strata} |
| Genuine live selections | {self.live_selection_count} |
| Particles per candidate/root | {self.particles_per_candidate} |
| Public-consistent allocation particles | {self.total_public_consistent_particles} |
| Repeated public-feature groups / maximum samples | {self.repeated_public_abstraction_groups} / {self.maximum_samples_per_public_abstraction} |
| Validation candidate NLL / uniform | `{self.validation_candidate_nll:.6f}` / `{self.uniform_candidate_nll:.6f}` |
| Validation candidate Brier / uniform | `{self.validation_candidate_brier:.6f}` / `{self.uniform_candidate_brier:.6f}` |
| Validation candidate top-1 / ECE | `{self.validation_candidate_top1_accuracy:.6f}` / `{self.validation_candidate_ece:.6f}` |
| Validation resource Brier / uniform-candidate / oracle-candidate | `{self.validation_resource_presence_brier:.6f}` / `{self.validation_uniform_resource_brier:.6f}` / `{self.validation_oracle_resource_brier:.6f}` |
| Validation resource ECE / count MAE | `{self.validation_resource_ece:.6f}` / `{self.validation_resource_count_mae:.6f}` |
| In-scope official hidden-card range calibrated | **{str(self.in_scope_official_hidden_card_range_calibrated).lower()}** |
| Opponent intent / full official belief calibrated | **false / false** |
| Counterfactual values / safe resolver / official control | **false / false / false** |

The roots and transient truth labels come from genuine official battles. The
observer adapter, prototype classifier, public-consistent allocation sampler,
hash-bucket resources, calibration metrics, and range interpretation are
research abstractions. Even an in-scope pass covers only this six-deck,
baseline-policy, later-root distribution; intent, counterfactual values,
resolving, exploitation, and official strategy control remain disabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6OfficialRangeEvidence":
        fields = {
            name: raw[name]
            for name in cls.__dataclass_fields__
            if name != "claims"
        }
        result = cls(
            **fields,
            claims=tuple(
                (str(key), str(value)) for key, value in raw["claims"].items()
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AD evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialRangeArtifacts:
    dataset: OfficialRangeTruthDataset
    model: OfficialPublicRangeModel
    calibration: OfficialRangeCalibration
    evidence: Phase6OfficialRangeEvidence


def _candidate_particle_summaries(
    observation: Mapping[str, Any],
    *,
    target: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    particles_per_candidate: int,
    root_token: str,
    style: str,
    seed: int,
) -> tuple[CandidateParticleSummary, ...]:
    proxy = _observer_proxy(observation)
    summaries = []
    for candidate in candidates:
        counts = []
        adjustments = []
        for particle_index in range(particles_per_candidate):
            completion = sample_hidden_zone_completion(
                proxy,
                hypothesis_id=(
                    f"phase6ad:{candidate.archetype}:{particle_index:03d}"
                ),
                probability=1.0 / particles_per_candidate,
                allocation_seed=_seed(
                    seed,
                    "phase6ad-public-particle",
                    root_token,
                    candidate.archetype,
                    particle_index,
                ),
                your_candidate_id=target.archetype,
                opponent_candidate_id=candidate.archetype,
                your_archetype_id=target.archetype,
                opponent_archetype_id=candidate.archetype,
                your_style_id=style,
                opponent_style_id=style,
                your_card_ids=target.card_ids,
                opponent_card_ids=candidate.card_ids,
                your_deck_fingerprint=target.fingerprint,
                opponent_deck_fingerprint=candidate.fingerprint,
                basic_card_ids=tuple(
                    ARCHETYPE_SIGNATURES.get(candidate.archetype, ())
                ),
                source=(
                    "public-only observer-oriented allocation particle; "
                    "candidate deck and uniform hidden allocation are abstractions"
                ),
            )
            counts.append(
                _resource_counts(
                    hand=completion.opponent_hand,
                    prize=completion.opponent_prize,
                    deck=completion.opponent_deck,
                )
            )
            adjustments.append(completion.public_evidence_adjustments)
        expected = tuple(
            sum(row[index] for row in counts) / len(counts)
            for index in range(len(OFFICIAL_RANGE_RESOURCE_NAMES))
        )
        presence = tuple(
            sum(row[index] > 0 for row in counts) / len(counts)
            for index in range(len(OFFICIAL_RANGE_RESOURCE_NAMES))
        )
        summaries.append(
            CandidateParticleSummary(
                candidate_id=candidate.archetype,
                candidate_deck_fingerprint=candidate.fingerprint,
                particle_count=particles_per_candidate,
                expected_resource_counts=expected,
                resource_presence_probabilities=presence,
                mean_public_evidence_adjustments=(
                    sum(adjustments) / len(adjustments)
                ),
                maximum_public_evidence_adjustments=max(adjustments),
            )
        )
    return tuple(summaries)


def _build_truth_example(
    observation: Mapping[str, Any],
    prefix_trace: OfficialPrefixReachTrace,
    *,
    visualize_data: Any,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    split: str,
    style: str,
    collection_ordinal: int,
    particles_per_candidate: int,
    seed: int,
) -> OfficialRangeTruthExample:
    frames = json.loads(visualize_data())
    if not isinstance(frames, Sequence) or not frames or not isinstance(frames[-1], Mapping):
        raise RuntimeError("Phase 6AD visualizer returned no truth frame")
    truth_frame = frames[-1]
    visualizer_public_parity = (
        sanitize_official_public_state(truth_frame)
        == sanitize_official_public_state(observation)
    )
    truth_private = official_private_observation_payload(truth_frame)
    live_private = official_private_observation_payload(observation)
    visualizer_private_parity = all(
        truth_private[name] == live_private[name]
        for name in ("own_hand", "own_looking")
    )
    if not visualizer_public_parity or not visualizer_private_parity:
        raise RuntimeError("Phase 6AD visualizer/live root parity failed")
    completion = _official_truth_completion(
        observation,
        truth_frame,
        target=target,
        candidate=candidate,
        style=style,
    )
    acting_hand = tuple(int(value) for value in live_private["own_hand"])
    truth_counts = _resource_counts(
        hand=acting_hand,
        prize=completion.your_prize,
        deck=completion.your_deck,
    )
    truth_zone_counts = (
        len(acting_hand),
        len(completion.your_prize),
        len(completion.your_deck),
    )
    public_state_fingerprint = _fingerprint(
        sanitize_official_public_state(observation)
    )
    public_history_fingerprint = _root_public_history(observation)
    uniform_range = _uniform_candidate_range(
        tuple(item.archetype for item in candidates)
    )
    public_features = native_observer_public_feature_items(
        observation,
        uniform_range,
        observer_player=OFFICIAL_RANGE_OBSERVER_PLAYER,
    )
    abstraction_fingerprint = _fingerprint(public_features)
    truth_sample_fingerprint = _fingerprint(truth_frame)
    source_root_id = _fingerprint(
        {
            "split": split,
            "candidate": candidate.archetype,
            "collection_ordinal": collection_ordinal,
            "public_state": public_state_fingerprint,
            "public_history": public_history_fingerprint,
            "truth_sample": truth_sample_fingerprint,
        }
    )
    summaries = _candidate_particle_summaries(
        observation,
        target=target,
        candidates=candidates,
        particles_per_candidate=particles_per_candidate,
        root_token=source_root_id,
        style=style,
        seed=seed,
    )
    example_id = _fingerprint(
        {
            "source_root": source_root_id,
            "public_features": public_features,
            "candidate_label_supervision_only": candidate.archetype,
            "truth_resources_supervision_only": truth_counts,
        }
    )
    return OfficialRangeTruthExample(
        example_id=example_id,
        source_root_id=source_root_id,
        split=split,
        candidate_id_supervision_only=candidate.archetype,
        collection_ordinal=collection_ordinal,
        prefix_trace=prefix_trace,
        public_state_fingerprint=public_state_fingerprint,
        public_history_fingerprint=public_history_fingerprint,
        public_abstraction_fingerprint=abstraction_fingerprint,
        public_features=public_features,
        truth_sample_fingerprint=truth_sample_fingerprint,
        truth_resource_counts=truth_counts,
        truth_zone_counts=truth_zone_counts,
        candidate_particle_summaries=summaries,
        visualizer_public_parity=visualizer_public_parity,
        visualizer_acting_private_parity=visualizer_private_parity,
    )


def collect_official_range_truth_dataset(
    *,
    source: JointOfficialValueDataset,
    target: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    visualize_data: Any,
    roots_per_candidate_per_split: int,
    particles_per_candidate: int,
    maximum_live_selections: int,
    maximum_capture_attempts: int,
    seed: int,
) -> OfficialRangeTruthDataset:
    if roots_per_candidate_per_split <= 0 or particles_per_candidate <= 1:
        raise ValueError("Phase 6AD collection sizes are invalid")
    examples = []
    for split in ALIGNED_VALUE_SPLITS:
        style, ordinal, base_reverse = OFFICIAL_PREFIX_CAPTURE_CONFIG[split]
        for candidate_index, candidate in enumerate(candidates):
            for collection_ordinal in range(roots_per_candidate_per_split):
                capture_error: RuntimeError | None = None
                for attempt in range(1, maximum_capture_attempts + 1):
                    reverse = (
                        base_reverse
                        ^ bool(candidate_index % 2)
                        ^ bool(collection_ordinal % 2)
                        ^ bool((attempt - 1) % 2)
                    )
                    holder: list[OfficialRangeTruthExample] = []

                    def at_root(
                        observation: Mapping[str, Any],
                        trace: OfficialPrefixReachTrace,
                    ) -> None:
                        holder.append(
                            _build_truth_example(
                                observation,
                                trace,
                                visualize_data=visualize_data,
                                target=target,
                                candidate=candidate,
                                candidates=candidates,
                                split=split,
                                style=style,
                                collection_ordinal=collection_ordinal,
                                particles_per_candidate=particles_per_candidate,
                                seed=seed,
                            )
                        )

                    try:
                        _capture_official_prefix(
                            battle_start,
                            battle_select,
                            battle_finish,
                            target=target,
                            candidate=candidate,
                            split=split,
                            style=style,
                            ordinal=ordinal,
                            reverse=reverse,
                            capture_attempt=(
                                collection_ordinal * maximum_capture_attempts + attempt
                            ),
                            maximum_live_selections=maximum_live_selections,
                            root_callback=at_root,
                        )
                        if len(holder) != 1:
                            raise RuntimeError("Phase 6AD root callback did not emit one example")
                        examples.append(holder[0])
                        break
                    except RuntimeError as exc:
                        capture_error = exc
                else:
                    raise RuntimeError(
                        "Phase 6AD exhausted capture attempts for "
                        f"{split}/{candidate.archetype}/{collection_ordinal}"
                    ) from capture_error
    dataset = OfficialRangeTruthDataset(
        source_phase6ac_dataset_fingerprint=source.fingerprint,
        runtime_fingerprint=source.runtime_fingerprint,
        deck_fingerprint=target.fingerprint,
        candidate_ids=tuple(item.archetype for item in candidates),
        candidate_deck_fingerprints=tuple(
            (item.archetype, item.fingerprint) for item in candidates
        ),
        roots_per_candidate_per_split=roots_per_candidate_per_split,
        particles_per_candidate=particles_per_candidate,
        examples=tuple(examples),
        collection_split_disjoint=True,
        observer_orientation_validated=True,
    )
    if _raw_private_key_count(dataset.to_dict()) != 0:
        raise RuntimeError("Phase 6AD serialized a forbidden raw private key")
    restored = OfficialRangeTruthDataset.from_dict(dataset.to_dict())
    if restored != dataset:
        raise RuntimeError("Phase 6AD dataset roundtrip changed values")
    return dataset


def _probabilities_from_components(
    example: OfficialRangeTruthExample,
    *,
    normalizer: NamedFeatureNormalizer,
    prototypes: Sequence[tuple[str, tuple[float, ...]]],
    temperature: float,
    uniform_mix: float,
    adjustment_penalty: float,
) -> tuple[tuple[str, float], ...]:
    values = normalizer.transform(example.public_features)
    logits = []
    for (candidate, prototype), summary in zip(
        prototypes,
        example.candidate_particle_summaries,
        strict=True,
    ):
        distance = sum(
            (value - reference) ** 2
            for value, reference in zip(values, prototype, strict=True)
        )
        logits.append(
            (
                candidate,
                -distance / temperature
                - adjustment_penalty * summary.mean_public_evidence_adjustments,
            )
        )
    maximum = max(value for _, value in logits)
    exponentials = [(candidate, math.exp(value - maximum)) for candidate, value in logits]
    total = sum(value for _, value in exponentials)
    uniform = 1.0 / len(exponentials)
    return tuple(
        (
            candidate,
            (1.0 - uniform_mix) * value / total + uniform_mix * uniform,
        )
        for candidate, value in exponentials
    )


def _candidate_summary_metrics(
    examples: Sequence[OfficialRangeTruthExample],
    *,
    normalizer: NamedFeatureNormalizer,
    prototypes: Sequence[tuple[str, tuple[float, ...]]],
    temperature: float,
    uniform_mix: float,
    adjustment_penalty: float,
) -> tuple[float, float, float]:
    nll = []
    brier = []
    correct = []
    for example in examples:
        probabilities = _probabilities_from_components(
            example,
            normalizer=normalizer,
            prototypes=prototypes,
            temperature=temperature,
            uniform_mix=uniform_mix,
            adjustment_penalty=adjustment_penalty,
        )
        distribution = dict(probabilities)
        target = example.candidate_id_supervision_only
        nll.append(-math.log(max(distribution[target], 1e-15)))
        brier.append(
            sum(
                (probability - float(candidate == target)) ** 2
                for candidate, probability in probabilities
            )
        )
        predicted = min(probabilities, key=lambda item: (-item[1], item[0]))[0]
        correct.append(predicted == target)
    return (
        sum(nll) / len(nll),
        sum(brier) / len(brier),
        sum(correct) / len(correct),
    )


def train_official_public_range(
    dataset: OfficialRangeTruthDataset,
) -> OfficialPublicRangeModel:
    train = tuple(item for item in dataset.examples if item.split == "train")
    calibration = tuple(
        item for item in dataset.examples if item.split == "calibration"
    )
    train_rows_fingerprint = _fingerprint(
        [(item.example_id, item.public_features) for item in train]
    )
    normalizer = NamedFeatureNormalizer.fit(
        [item.public_features for item in train],
        fit_rows_fingerprint=train_rows_fingerprint,
    )
    prototypes = []
    for candidate in dataset.candidate_ids:
        rows = tuple(
            item
            for item in train
            if item.candidate_id_supervision_only == candidate
        )
        if len(rows) != dataset.roots_per_candidate_per_split:
            raise ValueError("Phase 6AD candidate training root count is incomplete")
        vectors = tuple(normalizer.transform(item.public_features) for item in rows)
        prototypes.append(
            (
                candidate,
                tuple(
                    sum(vector[index] for vector in vectors) / len(vectors)
                    for index in range(len(NATIVE_PUBLIC_FEATURE_NAMES))
                ),
            )
        )
    metrics = []
    for temperature in PUBLIC_RANGE_TEMPERATURE_GRID:
        for uniform_mix in PUBLIC_RANGE_UNIFORM_MIX_GRID:
            for penalty in OFFICIAL_RANGE_ADJUSTMENT_PENALTY_GRID:
                nll, brier, accuracy = _candidate_summary_metrics(
                    calibration,
                    normalizer=normalizer,
                    prototypes=prototypes,
                    temperature=temperature,
                    uniform_mix=uniform_mix,
                    adjustment_penalty=penalty,
                )
                metrics.append(
                    OfficialRangeSelectionMetric(
                        temperature=temperature,
                        uniform_mix=uniform_mix,
                        adjustment_penalty=penalty,
                        calibration_nll=nll,
                        calibration_brier=brier,
                        calibration_top1_accuracy=accuracy,
                    )
                )
    selected = min(
        metrics,
        key=lambda item: (
            item.calibration_nll,
            item.calibration_brier,
            item.temperature,
            item.uniform_mix,
            item.adjustment_penalty,
        ),
    )
    return OfficialPublicRangeModel(
        source_dataset_fingerprint=dataset.fingerprint,
        runtime_fingerprint=dataset.runtime_fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        candidate_ids=dataset.candidate_ids,
        candidate_deck_fingerprints=dataset.candidate_deck_fingerprints,
        normalizer=normalizer,
        prototypes=tuple(prototypes),
        selection_metrics=tuple(metrics),
        selected_temperature=selected.temperature,
        selected_uniform_mix=selected.uniform_mix,
        selected_adjustment_penalty=selected.adjustment_penalty,
        train_root_ids=tuple(item.source_root_id for item in train),
        calibration_root_ids=tuple(item.source_root_id for item in calibration),
        train_rows_fingerprint=train_rows_fingerprint,
    )


def _evaluation_row(
    example: OfficialRangeTruthExample,
    model: OfficialPublicRangeModel,
) -> OfficialRangeEvaluationRow:
    probabilities = model.candidate_probabilities(
        example.public_features,
        example.candidate_particle_summaries,
    )
    expected, presence = _mixed_resource_predictions(
        probabilities,
        example.candidate_particle_summaries,
    )
    uniform_probabilities = tuple(
        (candidate, 1.0 / len(model.candidate_ids))
        for candidate in model.candidate_ids
    )
    _, uniform_presence = _mixed_resource_predictions(
        uniform_probabilities,
        example.candidate_particle_summaries,
    )
    oracle = next(
        item
        for item in example.candidate_particle_summaries
        if item.candidate_id == example.candidate_id_supervision_only
    )
    return OfficialRangeEvaluationRow(
        example_id=example.example_id,
        source_root_id=example.source_root_id,
        split=example.split,
        target_candidate_id_supervision_only=example.candidate_id_supervision_only,
        candidate_probabilities=probabilities,
        predicted_resource_counts=expected,
        resource_presence_probabilities=presence,
        uniform_candidate_resource_presence_probabilities=uniform_presence,
        oracle_candidate_resource_presence_probabilities=(
            oracle.resource_presence_probabilities
        ),
        truth_resource_counts=example.truth_resource_counts,
    )


def _split_metrics(
    rows: Sequence[OfficialRangeEvaluationRow],
    split: str,
) -> OfficialRangeSplitMetrics:
    selected = tuple(item for item in rows if item.split == split)
    candidate_pairs = tuple((item.confidence, item.top1_correct) for item in selected)
    resource_pairs = tuple(
        (probability, truth > 0)
        for item in selected
        for probability, truth in zip(
            item.resource_presence_probabilities,
            item.truth_resource_counts,
            strict=True,
        )
    )
    return OfficialRangeSplitMetrics(
        split=split,
        roots=len(selected),
        candidate_strata=len(
            {item.target_candidate_id_supervision_only for item in selected}
        ),
        candidate_nll=sum(item.nll for item in selected) / len(selected),
        candidate_brier=sum(item.candidate_brier for item in selected) / len(selected),
        candidate_top1_accuracy=sum(item.top1_correct for item in selected) / len(selected),
        candidate_expected_calibration_error=_ece(candidate_pairs),
        resource_presence_brier=(
            sum(item.resource_presence_brier for item in selected) / len(selected)
        ),
        uniform_candidate_resource_brier=(
            sum(item.uniform_candidate_resource_brier for item in selected)
            / len(selected)
        ),
        oracle_candidate_resource_brier=(
            sum(item.oracle_candidate_resource_brier for item in selected)
            / len(selected)
        ),
        resource_expected_calibration_error=_ece(resource_pairs),
        resource_count_mae=sum(item.resource_count_mae for item in selected) / len(selected),
    )


def evaluate_official_public_range(
    dataset: OfficialRangeTruthDataset,
    model: OfficialPublicRangeModel,
) -> OfficialRangeCalibration:
    if model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("Phase 6AD model/dataset fingerprint mismatch")
    rows = tuple(_evaluation_row(item, model) for item in dataset.examples)
    train = _split_metrics(rows, "train")
    calibration = _split_metrics(rows, "calibration")
    validation = _split_metrics(rows, "validation")
    uniform_nll = math.log(len(dataset.candidate_ids))
    uniform_brier = 1.0 - 1.0 / len(dataset.candidate_ids)
    candidate_gate = all(
        (
            calibration.candidate_nll < uniform_nll,
            validation.candidate_nll < uniform_nll,
            calibration.candidate_brier < uniform_brier,
            validation.candidate_brier < uniform_brier,
            calibration.candidate_expected_calibration_error
            <= OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT,
            validation.candidate_expected_calibration_error
            <= OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT,
        )
    )
    resource_gate = all(
        (
            calibration.resource_presence_brier
            <= calibration.uniform_candidate_resource_brier,
            validation.resource_presence_brier
            <= validation.uniform_candidate_resource_brier,
            calibration.resource_expected_calibration_error
            <= OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
            validation.resource_expected_calibration_error
            <= OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
        )
    )
    enough_calibration = calibration.roots >= PUBLIC_RANGE_MINIMUM_CALIBRATION_ROOTS
    enough_validation = validation.roots >= PUBLIC_RANGE_MINIMUM_VALIDATION_ROOTS
    in_scope = all(
        (
            dataset.collection_split_disjoint,
            enough_calibration,
            enough_validation,
            dataset.observer_orientation_validated,
            candidate_gate,
            resource_gate,
        )
    )
    return OfficialRangeCalibration(
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        evaluation_rows=rows,
        train=train,
        calibration=calibration,
        validation=validation,
        uniform_candidate_nll=uniform_nll,
        uniform_candidate_brier=uniform_brier,
        minimum_calibration_roots=PUBLIC_RANGE_MINIMUM_CALIBRATION_ROOTS,
        minimum_validation_roots=PUBLIC_RANGE_MINIMUM_VALIDATION_ROOTS,
        candidate_ece_limit=OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT,
        resource_ece_limit=OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
        root_split_gate_passed=dataset.collection_split_disjoint,
        sufficient_calibration_roots_gate_passed=enough_calibration,
        sufficient_validation_roots_gate_passed=enough_validation,
        observer_orientation_gate_passed=dataset.observer_orientation_validated,
        heldout_candidate_calibration_gate_passed=candidate_gate,
        heldout_hidden_resource_calibration_gate_passed=resource_gate,
        in_scope_official_hidden_card_range_calibrated=in_scope,
    )


def run_phase6_official_range_calibration(
    *,
    project_root: str | Path | None = None,
    source_phase6ac_dataset_path: str | Path | None = None,
    roots_per_candidate_per_split: int = OFFICIAL_RANGE_ROOTS_PER_CANDIDATE_PER_SPLIT,
    particles_per_candidate: int = OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE,
    maximum_live_selections: int = 900,
    maximum_capture_attempts: int = 8,
    seed: int = PHASE6_OFFICIAL_RANGE_SEED,
) -> Phase6OfficialRangeArtifacts:
    if min(
        roots_per_candidate_per_split,
        particles_per_candidate,
        maximum_live_selections,
        maximum_capture_attempts,
    ) <= 0:
        raise ValueError("Phase 6AD limits must be positive")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    source_path = Path(
        source_phase6ac_dataset_path
        or root
        / "artifacts"
        / "phase6_joint_official_values"
        / "joint_official_value_dataset.json"
    ).resolve()
    source = JointOfficialValueDataset.from_dict(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    if set(profile_by_id) != set(source.candidate_ids):
        raise ValueError("Phase 6AD candidate catalog/source mismatch")
    candidates = tuple(profile_by_id[name] for name in source.candidate_ids)
    if target.fingerprint != source.deck_fingerprint:
        raise ValueError("Phase 6AD target deck/source mismatch")

    runtime_text = str(runtime)
    import sys

    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
        visualize_data,
    )

    dataset = collect_official_range_truth_dataset(
        source=source,
        target=target,
        candidates=candidates,
        battle_start=battle_start,
        battle_select=battle_select,
        battle_finish=battle_finish,
        visualize_data=visualize_data,
        roots_per_candidate_per_split=roots_per_candidate_per_split,
        particles_per_candidate=particles_per_candidate,
        maximum_live_selections=maximum_live_selections,
        maximum_capture_attempts=maximum_capture_attempts,
        seed=seed,
    )
    model = train_official_public_range(dataset)
    restored_model = OfficialPublicRangeModel.from_dict(model.to_dict())
    if restored_model != model:
        raise RuntimeError("Phase 6AD model roundtrip changed values")
    calibration = evaluate_official_public_range(dataset, model)
    restored_calibration = OfficialRangeCalibration.from_dict(calibration.to_dict())
    if restored_calibration != calibration:
        raise RuntimeError("Phase 6AD calibration roundtrip changed values")
    evidence = Phase6OfficialRangeEvidence(
        benchmark_id="phase6ad-observer-official-hidden-card-range-v1",
        status=(
            "in_scope_hidden_card_range_calibrated_full_belief_fail_closed"
            if calibration.in_scope_official_hidden_card_range_calibrated
            else "official_hidden_card_range_diagnostic_rejected_fail_closed"
        ),
        source_phase6ac_dataset_fingerprint=source.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=len(dataset.examples),
        roots_per_split=dict(dataset.roots_per_split)["train"],
        candidate_strata=len(dataset.candidate_ids),
        live_selection_count=dataset.live_selection_count,
        particles_per_candidate=dataset.particles_per_candidate,
        total_public_consistent_particles=(
            len(dataset.examples)
            * len(dataset.candidate_ids)
            * dataset.particles_per_candidate
        ),
        repeated_public_abstraction_groups=(
            dataset.repeated_public_abstraction_groups
        ),
        maximum_samples_per_public_abstraction=(
            dataset.maximum_samples_per_public_abstraction
        ),
        selected_temperature=model.selected_temperature,
        selected_uniform_mix=model.selected_uniform_mix,
        selected_adjustment_penalty=model.selected_adjustment_penalty,
        validation_candidate_nll=calibration.validation.candidate_nll,
        uniform_candidate_nll=calibration.uniform_candidate_nll,
        validation_candidate_brier=calibration.validation.candidate_brier,
        uniform_candidate_brier=calibration.uniform_candidate_brier,
        validation_candidate_top1_accuracy=(
            calibration.validation.candidate_top1_accuracy
        ),
        validation_candidate_ece=(
            calibration.validation.candidate_expected_calibration_error
        ),
        validation_resource_presence_brier=(
            calibration.validation.resource_presence_brier
        ),
        validation_uniform_resource_brier=(
            calibration.validation.uniform_candidate_resource_brier
        ),
        validation_oracle_resource_brier=(
            calibration.validation.oracle_candidate_resource_brier
        ),
        validation_resource_ece=(
            calibration.validation.resource_expected_calibration_error
        ),
        validation_resource_count_mae=calibration.validation.resource_count_mae,
        in_scope_official_hidden_card_range_calibrated=(
            calibration.in_scope_official_hidden_card_range_calibrated
        ),
        opponent_intent_range_calibrated=False,
        full_official_public_belief_calibrated=False,
        counterfactual_value_compatible=False,
        safe_resolver_enabled=False,
        opponent_exploitation_enabled=False,
        official_strategy_control_enabled=False,
        claims=(
            ("genuine", "fresh official roots, behavior prefixes, and transient truth labels"),
            ("model_input", "player-0-oriented public state only"),
            ("range_scope", "six declared decks and hidden hand/prize/deck hash buckets"),
            ("abstract", "prototype classifier, allocation prior, hash buckets, and calibration"),
            ("unavailable", "opponent intent, full belief, values, safety, exploitation, and control"),
            ("kaggle_submission", "not_attempted"),
            ("pokemon_strength", "not_claimed"),
            ("phase6_complete", "false"),
        ),
    )
    restored_evidence = Phase6OfficialRangeEvidence.from_dict(evidence.to_dict())
    if restored_evidence != evidence:
        raise RuntimeError("Phase 6AD evidence roundtrip changed values")
    return Phase6OfficialRangeArtifacts(dataset, model, calibration, evidence)


def main() -> None:
    print(run_phase6_official_range_calibration().evidence.to_json())


if __name__ == "__main__":
    main()
