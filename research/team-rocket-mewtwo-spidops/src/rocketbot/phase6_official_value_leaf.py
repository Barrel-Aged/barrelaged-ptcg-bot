"""Phase 6AG implementation-heldout official full-game value-leaf evidence.

This module deliberately targets the failed leaf prerequisite identified by
Phases 6R and 6U--6W.  It collects complete official-engine games while holding
an opponent-policy implementation out, converts truth replays through the
existing leakage-safe perspective converter, and trains a match-balanced neural
outcome leaf on public plus acting-player-private features.

Official battle transitions and replay frames are genuine.  Perspective
conversion, hashed features, policy families, the neural model, conformal
intervals, and every downstream safety interpretation are research
abstractions.  A passing leaf gate is not a counterfactual-value bound, an
equilibrium blueprint, safe resolving, official strategy control, or a Pokemon
strength claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .beliefs import OPPONENT_INTENTS
from .competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    SelectionContract,
    _safe_int,
    _state,
    resolve_official_deck,
)
from .neural_blueprint import TensorArtifact
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    PriorityBaseline,
    rocket_compatibility_profile,
)
from .official_trajectory import (
    LoggedInformationSetStep,
    OfficialTrajectoryDataset,
    PublicRangeCandidate,
    build_official_trajectory_dataset,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_official_intent_calibration import (
    OFFICIAL_INTENT_POLICY_VARIANT_IDS,
    StrategicIntentPriorityBaseline,
    official_intent_policy_variant_fingerprint,
)
from .public_replay import (
    OfficialReplayConverter,
    PublicReplayDataset,
    build_public_replay_dataset,
)
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_OFFICIAL_VALUE_LEAF_SCHEMA_VERSION = 1
PHASE6_OFFICIAL_VALUE_LEAF_SEED = 6_047
PHASE6_OFFICIAL_VALUE_LEAF_REPLICAS = 2
PHASE6_OFFICIAL_VALUE_LEAF_MAX_SELECTIONS = 2_000
PHASE6_OFFICIAL_VALUE_LEAF_POLICY_SPLITS = (
    ("canonical", "train"),
    ("engine_first", "train"),
    ("hand_first", "calibration"),
    ("action_first", "validation"),
)
PHASE6_OFFICIAL_VALUE_LEAF_PROGRESS_BUCKETS = (
    "opening",
    "midgame",
    "endgame",
)
PHASE6_OFFICIAL_VALUE_LEAF_HIDDEN_GRID = (32, 64)
PHASE6_OFFICIAL_VALUE_LEAF_WEIGHT_DECAY_GRID = (0.0, 1e-4)
PHASE6_OFFICIAL_VALUE_LEAF_BLEND_GRID = (0.50, 0.75, 1.0)
PHASE6_OFFICIAL_VALUE_LEAF_EPOCHS = 40
PHASE6_OFFICIAL_VALUE_LEAF_LEARNING_RATE = 0.01
PHASE6_OFFICIAL_VALUE_LEAF_COVERAGE = 0.90
PHASE6_OFFICIAL_VALUE_LEAF_MINIMUM_BUCKET_COVERAGE = 0.90
PHASE6_OFFICIAL_VALUE_LEAF_MAXIMUM_OVERALL_WIDTH = 1.90
PHASE6_OFFICIAL_VALUE_LEAF_MAXIMUM_ENDGAME_WIDTH = 1.50


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _policy_split_map() -> dict[str, str]:
    result = dict(PHASE6_OFFICIAL_VALUE_LEAF_POLICY_SPLITS)
    if tuple(result) != OFFICIAL_INTENT_POLICY_VARIANT_IDS:
        raise RuntimeError("Phase 6AG policy variant ordering drifted")
    return result


def _calibration_partition(
    split: str,
    replica: int,
    *,
    candidate_index: int = 0,
    intent_index: int = 0,
) -> str:
    if split == "train":
        return "train"
    if split == "calibration":
        return (
            "selection"
            if (replica + candidate_index + intent_index) % 2 == 0
            else "bounds"
        )
    if split == "validation":
        return "untouched"
    raise ValueError(f"unsupported Phase 6AG split: {split}")


@dataclass(frozen=True, slots=True)
class OfficialValueLeafMatch:
    match_id: str
    split: str
    calibration_partition: str
    policy_variant_id: str
    policy_variant_fingerprint: str
    candidate_id: str
    intent_id: str
    replica: int
    target_reverse: bool
    opponent_reverse: bool
    winner: int
    turns: int
    selection_count: int
    replay_sha256: str
    observer_episode_fingerprints: tuple[str, str]

    def __post_init__(self) -> None:
        if not self.match_id or self.split not in {"train", "calibration", "validation"}:
            raise ValueError("Phase 6AG match identity/split is invalid")
        if self.policy_variant_id not in OFFICIAL_INTENT_POLICY_VARIANT_IDS:
            raise ValueError("Phase 6AG match policy variant is invalid")
        _validate_digest(self.policy_variant_fingerprint, "policy_variant_fingerprint")
        if self.policy_variant_fingerprint != official_intent_policy_variant_fingerprint(
            self.policy_variant_id
        ):
            raise ValueError("Phase 6AG policy variant fingerprint mismatch")
        if self.replica not in range(PHASE6_OFFICIAL_VALUE_LEAF_REPLICAS):
            raise ValueError("Phase 6AG match replica is invalid")
        allowed_partitions = {
            "train": {"train"},
            "calibration": {"selection", "bounds"},
            "validation": {"untouched"},
        }
        if self.calibration_partition not in allowed_partitions[self.split]:
            raise ValueError("Phase 6AG calibration partition mismatch")
        if self.opponent_reverse == self.target_reverse:
            raise ValueError("Phase 6AG tie-break directions must be opposed")
        if self.winner not in (0, 1, 2) or min(self.turns, self.selection_count) <= 0:
            raise ValueError("Phase 6AG match did not finish normally")
        _validate_digest(self.replay_sha256, "replay_sha256")
        if len(self.observer_episode_fingerprints) != 2:
            raise ValueError("Phase 6AG match requires both replay perspectives")
        for value in self.observer_episode_fingerprints:
            _validate_digest(value, "observer_episode_fingerprint")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observer_episode_fingerprints": list(self.observer_episode_fingerprints),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialValueLeafMatch":
        return cls(
            match_id=str(raw["match_id"]),
            split=str(raw["split"]),
            calibration_partition=str(raw["calibration_partition"]),
            policy_variant_id=str(raw["policy_variant_id"]),
            policy_variant_fingerprint=str(raw["policy_variant_fingerprint"]),
            candidate_id=str(raw["candidate_id"]),
            intent_id=str(raw["intent_id"]),
            replica=int(raw["replica"]),
            target_reverse=bool(raw["target_reverse"]),
            opponent_reverse=bool(raw["opponent_reverse"]),
            winner=int(raw["winner"]),
            turns=int(raw["turns"]),
            selection_count=int(raw["selection_count"]),
            replay_sha256=str(raw["replay_sha256"]),
            observer_episode_fingerprints=tuple(
                str(value) for value in raw["observer_episode_fingerprints"]
            ),
        )


@dataclass(frozen=True, slots=True)
class OfficialValueLeafManifest:
    trajectory_dataset_fingerprint: str
    public_replay_dataset_fingerprint: str
    runtime_fingerprint: str
    engine_sha256: str
    card_pool_sha256: str
    environment_fingerprint: str
    deck_fingerprint: str
    target_policy_id: str
    policy_variant_splits: tuple[tuple[str, str], ...]
    policy_variant_fingerprints: tuple[tuple[str, str], ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    intent_ids: tuple[str, ...]
    replicas_per_stratum: int
    matches: tuple[OfficialValueLeafMatch, ...]
    official_engine_matches_executed: bool = True
    truth_replays_converted_in_memory: bool = True
    raw_truth_replays_persisted: bool = False
    model_inputs_public_plus_acting_private_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "trajectory_dataset_fingerprint",
            "public_replay_dataset_fingerprint",
            "runtime_fingerprint",
            "engine_sha256",
            "card_pool_sha256",
            "environment_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.target_policy_id != "fixed_player0_tempo":
            raise ValueError("Phase 6AG target policy drifted")
        if self.policy_variant_splits != PHASE6_OFFICIAL_VALUE_LEAF_POLICY_SPLITS:
            raise ValueError("Phase 6AG policy split protocol drifted")
        expected_variant_fingerprints = tuple(
            (variant, official_intent_policy_variant_fingerprint(variant))
            for variant, _ in self.policy_variant_splits
        )
        if self.policy_variant_fingerprints != expected_variant_fingerprints:
            raise ValueError("Phase 6AG policy fingerprint catalog drifted")
        if self.replicas_per_stratum != PHASE6_OFFICIAL_VALUE_LEAF_REPLICAS:
            raise ValueError("Phase 6AG replica count drifted")
        if not self.candidate_deck_fingerprints or not self.intent_ids:
            raise ValueError("Phase 6AG candidate/intent catalog is empty")
        if len(dict(self.candidate_deck_fingerprints)) != len(
            self.candidate_deck_fingerprints
        ):
            raise ValueError("Phase 6AG candidate catalog contains duplicates")
        expected = {
            (variant, candidate, intent, replica)
            for variant, _ in self.policy_variant_splits
            for candidate, _ in self.candidate_deck_fingerprints
            for intent in self.intent_ids
            for replica in range(self.replicas_per_stratum)
        }
        observed = {
            (
                item.policy_variant_id,
                item.candidate_id,
                item.intent_id,
                item.replica,
            )
            for item in self.matches
        }
        if observed != expected or len(self.matches) != len(expected):
            raise ValueError("Phase 6AG collection grid is incomplete or duplicated")
        split_by_variant = dict(self.policy_variant_splits)
        if any(item.split != split_by_variant[item.policy_variant_id] for item in self.matches):
            raise ValueError("Phase 6AG policy implementation crossed a split")
        candidate_indices = {
            candidate: index
            for index, (candidate, _) in enumerate(self.candidate_deck_fingerprints)
        }
        intent_indices = {
            intent: index for index, intent in enumerate(self.intent_ids)
        }
        if any(
            item.calibration_partition
            != _calibration_partition(
                item.split,
                item.replica,
                candidate_index=candidate_indices[item.candidate_id],
                intent_index=intent_indices[item.intent_id],
            )
            for item in self.matches
        ):
            raise ValueError("Phase 6AG calibration roles are not stratum-balanced")
        if len({item.match_id for item in self.matches}) != len(self.matches):
            raise ValueError("Phase 6AG match IDs are not unique")
        if not all(
            (
                self.official_engine_matches_executed,
                self.truth_replays_converted_in_memory,
                self.model_inputs_public_plus_acting_private_only,
                not self.raw_truth_replays_persisted,
            )
        ):
            raise ValueError("Phase 6AG provenance/retention boundary changed")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def split_match_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (split, sum(item.split == split for item in self.matches))
            for split in ("train", "calibration", "validation")
        )

    @property
    def live_selection_count(self) -> int:
        return sum(item.selection_count for item in self.matches)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_VALUE_LEAF_SCHEMA_VERSION,
            "trajectory_dataset_fingerprint": self.trajectory_dataset_fingerprint,
            "public_replay_dataset_fingerprint": self.public_replay_dataset_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "engine_sha256": self.engine_sha256,
            "card_pool_sha256": self.card_pool_sha256,
            "environment_fingerprint": self.environment_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "target_policy_id": self.target_policy_id,
            "policy_variant_splits": [list(item) for item in self.policy_variant_splits],
            "policy_variant_fingerprints": [
                list(item) for item in self.policy_variant_fingerprints
            ],
            "candidate_deck_fingerprints": [
                list(item) for item in self.candidate_deck_fingerprints
            ],
            "intent_ids": list(self.intent_ids),
            "replicas_per_stratum": self.replicas_per_stratum,
            "collection": {
                "match_count": len(self.matches),
                "split_match_counts": dict(self.split_match_counts),
                "live_selection_count": self.live_selection_count,
            },
            "matches": [item.to_dict() for item in self.matches],
            "claims": {
                "official_engine_matches_executed": (
                    self.official_engine_matches_executed
                ),
                "truth_replays_converted_in_memory": (
                    self.truth_replays_converted_in_memory
                ),
                "raw_truth_replays_persisted": self.raw_truth_replays_persisted,
                "model_inputs_public_plus_acting_private_only": (
                    self.model_inputs_public_plus_acting_private_only
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialValueLeafManifest":
        claims = raw["claims"]
        result = cls(
            trajectory_dataset_fingerprint=str(raw["trajectory_dataset_fingerprint"]),
            public_replay_dataset_fingerprint=str(
                raw["public_replay_dataset_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            engine_sha256=str(raw["engine_sha256"]),
            card_pool_sha256=str(raw["card_pool_sha256"]),
            environment_fingerprint=str(raw["environment_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            target_policy_id=str(raw["target_policy_id"]),
            policy_variant_splits=tuple(
                (str(item[0]), str(item[1])) for item in raw["policy_variant_splits"]
            ),
            policy_variant_fingerprints=tuple(
                (str(item[0]), str(item[1]))
                for item in raw["policy_variant_fingerprints"]
            ),
            candidate_deck_fingerprints=tuple(
                (str(item[0]), str(item[1]))
                for item in raw["candidate_deck_fingerprints"]
            ),
            intent_ids=tuple(str(value) for value in raw["intent_ids"]),
            replicas_per_stratum=int(raw["replicas_per_stratum"]),
            matches=tuple(
                OfficialValueLeafMatch.from_dict(item) for item in raw["matches"]
            ),
            official_engine_matches_executed=bool(
                claims["official_engine_matches_executed"]
            ),
            truth_replays_converted_in_memory=bool(
                claims["truth_replays_converted_in_memory"]
            ),
            raw_truth_replays_persisted=bool(
                claims["raw_truth_replays_persisted"]
            ),
            model_inputs_public_plus_acting_private_only=bool(
                claims["model_inputs_public_plus_acting_private_only"]
            ),
        )
        collection = raw["collection"]
        expected_collection = {
            "match_count": len(result.matches),
            "split_match_counts": dict(result.split_match_counts),
            "live_selection_count": result.live_selection_count,
        }
        if dict(collection) != expected_collection:
            raise ValueError("Phase 6AG manifest accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AG manifest fingerprint mismatch")
        return result


def rebalance_official_value_leaf_manifest(
    raw: Mapping[str, Any],
    dataset: OfficialTrajectoryDataset,
) -> OfficialValueLeafManifest:
    """Migrate the rejected replica-confounded source manifest exactly once."""

    recorded_fingerprint = str(raw.get("fingerprint", ""))
    unsigned = dict(raw)
    unsigned.pop("fingerprint", None)
    if recorded_fingerprint != _fingerprint(unsigned):
        raise ValueError("Phase 6AG source manifest is stale or corrupt")
    candidate_catalog = tuple(
        (str(item[0]), str(item[1])) for item in raw["candidate_deck_fingerprints"]
    )
    intent_ids = tuple(str(value) for value in raw["intent_ids"])
    candidate_indices = {
        candidate: index for index, (candidate, _) in enumerate(candidate_catalog)
    }
    intent_indices = {intent: index for index, intent in enumerate(intent_ids)}
    matches = []
    changed = 0
    for item_raw in raw["matches"]:
        item = OfficialValueLeafMatch.from_dict(item_raw)
        partition = _calibration_partition(
            item.split,
            item.replica,
            candidate_index=candidate_indices[item.candidate_id],
            intent_index=intent_indices[item.intent_id],
        )
        changed += int(partition != item.calibration_partition)
        matches.append(replace(item, calibration_partition=partition))
    if changed == 0:
        raise ValueError("Phase 6AG source manifest is already stratum-balanced")
    claims = raw["claims"]
    result = OfficialValueLeafManifest(
        trajectory_dataset_fingerprint=str(raw["trajectory_dataset_fingerprint"]),
        public_replay_dataset_fingerprint=str(raw["public_replay_dataset_fingerprint"]),
        runtime_fingerprint=str(raw["runtime_fingerprint"]),
        engine_sha256=str(raw["engine_sha256"]),
        card_pool_sha256=str(raw["card_pool_sha256"]),
        environment_fingerprint=str(raw["environment_fingerprint"]),
        deck_fingerprint=str(raw["deck_fingerprint"]),
        target_policy_id=str(raw["target_policy_id"]),
        policy_variant_splits=tuple(
            (str(item[0]), str(item[1])) for item in raw["policy_variant_splits"]
        ),
        policy_variant_fingerprints=tuple(
            (str(item[0]), str(item[1]))
            for item in raw["policy_variant_fingerprints"]
        ),
        candidate_deck_fingerprints=candidate_catalog,
        intent_ids=intent_ids,
        replicas_per_stratum=int(raw["replicas_per_stratum"]),
        matches=tuple(matches),
        official_engine_matches_executed=bool(
            claims["official_engine_matches_executed"]
        ),
        truth_replays_converted_in_memory=bool(
            claims["truth_replays_converted_in_memory"]
        ),
        raw_truth_replays_persisted=bool(claims["raw_truth_replays_persisted"]),
        model_inputs_public_plus_acting_private_only=bool(
            claims["model_inputs_public_plus_acting_private_only"]
        ),
    )
    if result.trajectory_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("Phase 6AG migrated manifest does not bind the frozen dataset")
    return result


@dataclass(frozen=True, slots=True)
class OfficialValueLeafRow:
    row_id: str
    match_id: str
    observer_player: int
    split: str
    calibration_partition: str
    progress_bucket: str
    features: tuple[tuple[str, float], ...]
    heuristic_value: float
    outcome_utility: float

    def __post_init__(self) -> None:
        _validate_digest(self.row_id, "row_id")
        if not self.match_id or self.observer_player not in (0, 1):
            raise ValueError("Phase 6AG row identity is invalid")
        if self.split not in {"train", "calibration", "validation"}:
            raise ValueError("Phase 6AG row split is invalid")
        if self.progress_bucket not in PHASE6_OFFICIAL_VALUE_LEAF_PROGRESS_BUCKETS:
            raise ValueError("Phase 6AG row progress bucket is invalid")
        if not self.features or len({name for name, _ in self.features}) != len(
            self.features
        ):
            raise ValueError("Phase 6AG row feature schema is invalid")
        forbidden = ("policy_variant", "intent_id", "truth_candidate", "outcome")
        if any(any(token in name for token in forbidden) for name, _ in self.features):
            raise ValueError("Phase 6AG row leaked a supervision-only label")
        for name, value in self.features:
            if not name:
                raise ValueError("Phase 6AG feature name is empty")
            _finite(value, name)
        if not -1 <= _finite(self.heuristic_value, "heuristic_value") <= 1:
            raise ValueError("Phase 6AG heuristic value is outside [-1, 1]")
        if self.outcome_utility not in (-1.0, 0.0, 1.0):
            raise ValueError("Phase 6AG outcome is not a terminal utility")


def _progress_bucket(step: LoggedInformationSetStep) -> str:
    public = dict(step.public_belief.public_features)
    minimum_prizes = min(
        int(round(public["own_prize_count"])),
        int(round(public["opponent_prize_count"])),
    )
    if minimum_prizes <= 2:
        return "endgame"
    if minimum_prizes <= 4:
        return "midgame"
    return "opening"


def _heuristic_value(step: LoggedInformationSetStep) -> float:
    public = dict(step.public_belief.public_features)
    prize_advantage = (
        public["opponent_prize_count"] - public["own_prize_count"]
    ) / 6.0
    hp_advantage = (
        public["own_board_hp"] - public["opponent_board_hp"]
    ) / 600.0
    board_advantage = (
        public["own_board_count"] - public["opponent_board_count"]
    ) / 6.0
    return math.tanh(2.0 * prize_advantage + 0.35 * hp_advantage + 0.20 * board_advantage)


def official_value_leaf_feature_items(
    step: LoggedInformationSetStep,
    candidate_ids: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    range_weights = {
        item.hypothesis_id: item.weight
        for item in step.public_belief.opponent_range.hypotheses
    }
    if set(range_weights) != set(candidate_ids):
        raise ValueError("Phase 6AG public range support drifted")
    return (
        tuple((f"public::{name}", float(value)) for name, value in step.public_belief.public_features)
        + tuple((f"private::{name}", float(value)) for name, value in step.private_features)
        + tuple(
            (f"range_probability::{candidate_id}", float(range_weights[candidate_id]))
            for candidate_id in candidate_ids
        )
    )


def build_official_value_leaf_rows(
    dataset: OfficialTrajectoryDataset,
    manifest: OfficialValueLeafManifest,
) -> tuple[OfficialValueLeafRow, ...]:
    if dataset.fingerprint != manifest.trajectory_dataset_fingerprint:
        raise ValueError("Phase 6AG dataset/manifest fingerprint mismatch")
    records = {item.match_id: item for item in manifest.matches}
    candidate_ids = tuple(item.candidate_id for item in dataset.candidates)
    rows = []
    for episode in dataset.episodes:
        record = records.get(episode.match_id)
        if record is None or record.split != episode.split:
            raise ValueError("Phase 6AG episode is absent from its collection manifest")
        if record.observer_episode_fingerprints[episode.observer_player] != (
            episode.source_episode_fingerprint
        ):
            raise ValueError("Phase 6AG source episode fingerprint mismatch")
        for step in episode.steps:
            features = official_value_leaf_feature_items(step, candidate_ids)
            row_id = _fingerprint(
                {
                    "match": episode.match_id,
                    "observer": episode.observer_player,
                    "step": step.fingerprint,
                    "features": features,
                }
            )
            rows.append(
                OfficialValueLeafRow(
                    row_id=row_id,
                    match_id=episode.match_id,
                    observer_player=episode.observer_player,
                    split=episode.split,
                    calibration_partition=record.calibration_partition,
                    progress_bucket=_progress_bucket(step),
                    features=features,
                    heuristic_value=_heuristic_value(step),
                    outcome_utility=step.outcome_utility,
                )
            )
    if len({item.row_id for item in rows}) != len(rows):
        raise ValueError("Phase 6AG row identities collided")
    schemas = {tuple(name for name, _ in item.features) for item in rows}
    if len(schemas) != 1:
        raise ValueError("Phase 6AG row feature schemas differ")
    return tuple(rows)


class _OfficialValueLeafNetwork(nn.Module):
    def __init__(self, input_width: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_width = input_width
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Linear(input_width, hidden_dim, dtype=torch.float64)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64)
        self.output_layer = nn.Linear(hidden_dim, 1, dtype=torch.float64)

    def forward(self, features: Tensor) -> Tensor:
        hidden = torch.tanh(self.input_layer(features))
        hidden = torch.tanh(self.hidden_layer(hidden))
        return torch.tanh(self.output_layer(hidden).squeeze(-1))


def _tensor_artifacts(network: _OfficialValueLeafNetwork) -> tuple[TensorArtifact, ...]:
    return tuple(
        TensorArtifact(
            name=name,
            shape=tuple(tensor.shape),
            values=tuple(float(value) for value in tensor.detach().reshape(-1).tolist()),
        )
        for name, tensor in network.state_dict().items()
    )


def _restore_network(
    *,
    normalizer: NamedFeatureNormalizer,
    hidden_dim: int,
    tensors: Sequence[TensorArtifact],
) -> _OfficialValueLeafNetwork:
    network = _OfficialValueLeafNetwork(len(normalizer.names), hidden_dim)
    expected = network.state_dict()
    supplied = {item.name: item for item in tensors}
    if set(expected) != set(supplied):
        raise ValueError("Phase 6AG tensor names do not match the architecture")
    restored = {}
    for name, template in expected.items():
        item = supplied[name]
        if tuple(template.shape) != item.shape:
            raise ValueError(f"Phase 6AG tensor shape mismatch: {name}")
        restored[name] = torch.tensor(item.values, dtype=torch.float64).reshape(
            item.shape
        )
    network.load_state_dict(restored, strict=True)
    network.eval()
    return network


@dataclass(frozen=True, slots=True)
class OfficialValueLeafSelectionMetric:
    hidden_dim: int
    weight_decay: float
    blend_weight: float
    initial_training_loss: float
    final_training_loss: float
    selection_state_mse: float
    selection_episode_mse: float
    selection_zero_episode_mse: float
    selection_heuristic_episode_mse: float
    selection_endgame_mse: float
    selection_endgame_reference_mse: float
    selection_score: float

    def __post_init__(self) -> None:
        if self.hidden_dim not in PHASE6_OFFICIAL_VALUE_LEAF_HIDDEN_GRID:
            raise ValueError("Phase 6AG selected an unsupported hidden width")
        if self.weight_decay not in PHASE6_OFFICIAL_VALUE_LEAF_WEIGHT_DECAY_GRID:
            raise ValueError("Phase 6AG selected unsupported weight decay")
        if self.blend_weight not in PHASE6_OFFICIAL_VALUE_LEAF_BLEND_GRID:
            raise ValueError("Phase 6AG selected unsupported blend weight")
        for name, value in asdict(self).items():
            if name == "hidden_dim":
                continue
            if float(value) < 0 or not math.isfinite(float(value)):
                raise ValueError(f"Phase 6AG selection metric {name} is invalid")
        if self.final_training_loss >= self.initial_training_loss:
            raise ValueError("Phase 6AG neural training did not reduce loss")


@dataclass(frozen=True, slots=True)
class OfficialValueLeafModel:
    source_dataset_fingerprint: str
    source_manifest_fingerprint: str
    normalizer: NamedFeatureNormalizer
    seed: int
    epochs: int
    learning_rate: float
    selected_hidden_dim: int
    selected_weight_decay: float
    selected_blend_weight: float
    selection_metrics: tuple[OfficialValueLeafSelectionMetric, ...]
    train_match_ids: tuple[str, ...]
    calibration_selection_match_ids: tuple[str, ...]
    tensors: tuple[TensorArtifact, ...]
    public_plus_acting_private_features_only: bool = True
    supervision_identifiers_as_features: bool = False
    official_engine_executed: bool = False
    equilibrium_trained: bool = False
    counterfactual_value_provider: bool = False
    deployable_blueprint: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.source_dataset_fingerprint, "source_dataset_fingerprint")
        _validate_digest(self.source_manifest_fingerprint, "source_manifest_fingerprint")
        if min(self.seed, self.epochs, self.selected_hidden_dim) <= 0:
            raise ValueError("Phase 6AG model configuration is invalid")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("Phase 6AG learning rate is invalid")
        if self.selected_hidden_dim not in PHASE6_OFFICIAL_VALUE_LEAF_HIDDEN_GRID:
            raise ValueError("Phase 6AG model hidden width is unsupported")
        if self.selected_weight_decay not in PHASE6_OFFICIAL_VALUE_LEAF_WEIGHT_DECAY_GRID:
            raise ValueError("Phase 6AG model weight decay is unsupported")
        if self.selected_blend_weight not in PHASE6_OFFICIAL_VALUE_LEAF_BLEND_GRID:
            raise ValueError("Phase 6AG model blend is unsupported")
        if not self.selection_metrics or not self.tensors:
            raise ValueError("Phase 6AG model diagnostics/tensors are empty")
        selected = min(
            self.selection_metrics,
            key=lambda item: (
                item.selection_score,
                item.selection_episode_mse,
                item.hidden_dim,
                item.weight_decay,
                -item.blend_weight,
            ),
        )
        if (
            self.selected_hidden_dim,
            self.selected_weight_decay,
            self.selected_blend_weight,
        ) != (selected.hidden_dim, selected.weight_decay, selected.blend_weight):
            raise ValueError("Phase 6AG model selection record is inconsistent")
        if not self.train_match_ids or not self.calibration_selection_match_ids:
            raise ValueError("Phase 6AG model split identities are empty")
        if set(self.train_match_ids) & set(self.calibration_selection_match_ids):
            raise ValueError("Phase 6AG model train/selection matches overlap")
        if not self.public_plus_acting_private_features_only or any(
            (
                self.supervision_identifiers_as_features,
                self.official_engine_executed,
                self.equilibrium_trained,
                self.counterfactual_value_provider,
                self.deployable_blueprint,
            )
        ):
            raise ValueError("Phase 6AG model promoted its abstraction boundary")
        _restore_network(
            normalizer=self.normalizer,
            hidden_dim=self.selected_hidden_dim,
            tensors=self.tensors,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_VALUE_LEAF_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "normalizer": self.normalizer.to_dict(),
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "selected_hidden_dim": self.selected_hidden_dim,
            "selected_weight_decay": self.selected_weight_decay,
            "selected_blend_weight": self.selected_blend_weight,
            "selection_metrics": [asdict(item) for item in self.selection_metrics],
            "train_match_ids": list(self.train_match_ids),
            "calibration_selection_match_ids": list(
                self.calibration_selection_match_ids
            ),
            "tensors": [
                {
                    "name": item.name,
                    "shape": list(item.shape),
                    "values": list(item.values),
                }
                for item in self.tensors
            ],
            "claims": {
                "public_plus_acting_private_features_only": (
                    self.public_plus_acting_private_features_only
                ),
                "supervision_identifiers_as_features": (
                    self.supervision_identifiers_as_features
                ),
                "official_engine_executed": self.official_engine_executed,
                "equilibrium_trained": self.equilibrium_trained,
                "counterfactual_value_provider": self.counterfactual_value_provider,
                "deployable_blueprint": self.deployable_blueprint,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialValueLeafModel":
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            source_manifest_fingerprint=str(raw["source_manifest_fingerprint"]),
            normalizer=NamedFeatureNormalizer.from_dict(raw["normalizer"]),
            seed=int(raw["seed"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            selected_hidden_dim=int(raw["selected_hidden_dim"]),
            selected_weight_decay=float(raw["selected_weight_decay"]),
            selected_blend_weight=float(raw["selected_blend_weight"]),
            selection_metrics=tuple(
                OfficialValueLeafSelectionMetric(**dict(item))
                for item in raw["selection_metrics"]
            ),
            train_match_ids=tuple(str(value) for value in raw["train_match_ids"]),
            calibration_selection_match_ids=tuple(
                str(value) for value in raw["calibration_selection_match_ids"]
            ),
            tensors=tuple(
                TensorArtifact(
                    name=str(item["name"]),
                    shape=tuple(int(value) for value in item["shape"]),
                    values=tuple(float(value) for value in item["values"]),
                )
                for item in raw["tensors"]
            ),
            public_plus_acting_private_features_only=bool(
                claims["public_plus_acting_private_features_only"]
            ),
            supervision_identifiers_as_features=bool(
                claims["supervision_identifiers_as_features"]
            ),
            official_engine_executed=bool(claims["official_engine_executed"]),
            equilibrium_trained=bool(claims["equilibrium_trained"]),
            counterfactual_value_provider=bool(
                claims["counterfactual_value_provider"]
            ),
            deployable_blueprint=bool(claims["deployable_blueprint"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AG model fingerprint mismatch")
        return result


class OfficialValueLeafProvider:
    def __init__(self, model: OfficialValueLeafModel) -> None:
        self.model = model
        self.network = _restore_network(
            normalizer=model.normalizer,
            hidden_dim=model.selected_hidden_dim,
            tensors=model.tensors,
        )

    def predict_features(
        self,
        features: Sequence[tuple[str, float]],
        *,
        heuristic_value: float,
    ) -> float:
        normalized = self.model.normalizer.transform(features)
        with torch.no_grad():
            neural = float(
                self.network(torch.tensor([normalized], dtype=torch.float64))[0]
            )
        blend = self.model.selected_blend_weight
        return max(-1.0, min(1.0, blend * neural + (1.0 - blend) * heuristic_value))

    def predict_row(self, row: OfficialValueLeafRow) -> float:
        return self.predict_features(
            row.features,
            heuristic_value=row.heuristic_value,
        )


def restore_official_value_leaf(model: OfficialValueLeafModel) -> OfficialValueLeafProvider:
    return OfficialValueLeafProvider(model)


def _episode_mse(
    rows: Sequence[OfficialValueLeafRow],
    predictions: Sequence[float],
) -> float:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        grouped[(row.match_id, row.observer_player)].append(
            (prediction - row.outcome_utility) ** 2
        )
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def _point_metrics(
    rows: Sequence[OfficialValueLeafRow],
    predictions: Sequence[float],
) -> tuple[float, float, float, float]:
    state_mse = sum(
        (prediction - row.outcome_utility) ** 2
        for row, prediction in zip(rows, predictions, strict=True)
    ) / len(rows)
    episode_mse = _episode_mse(rows, predictions)
    zero_episode_mse = _episode_mse(rows, [0.0] * len(rows))
    heuristic_episode_mse = _episode_mse(
        rows, [row.heuristic_value for row in rows]
    )
    return state_mse, episode_mse, zero_episode_mse, heuristic_episode_mse


def _training_tensors(
    rows: Sequence[OfficialValueLeafRow],
    normalizer: NamedFeatureNormalizer,
) -> tuple[Tensor, Tensor, Tensor]:
    features = torch.tensor(
        [normalizer.transform(row.features) for row in rows], dtype=torch.float64
    )
    targets = torch.tensor([row.outcome_utility for row in rows], dtype=torch.float64)
    counts = Counter((row.match_id, row.observer_player) for row in rows)
    raw_weights = [1.0 / counts[(row.match_id, row.observer_player)] for row in rows]
    scale = len(raw_weights) / sum(raw_weights)
    weights = torch.tensor([value * scale for value in raw_weights], dtype=torch.float64)
    return features, targets, weights


def _train_network(
    rows: Sequence[OfficialValueLeafRow],
    normalizer: NamedFeatureNormalizer,
    *,
    hidden_dim: int,
    weight_decay: float,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> tuple[_OfficialValueLeafNetwork, float, float]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = _OfficialValueLeafNetwork(len(normalizer.names), hidden_dim)
    features, targets, weights = _training_tensors(rows, normalizer)
    optimizer = torch.optim.Adam(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def loss_value() -> Tensor:
        predictions = network(features)
        return torch.sum(weights * (predictions - targets) ** 2) / torch.sum(weights)

    with torch.no_grad():
        initial = float(loss_value())
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for _ in range(epochs):
            loss = loss_value()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
            optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)
    with torch.no_grad():
        final = float(loss_value())
    if not final < initial:
        raise RuntimeError("Phase 6AG neural value loss did not decrease")
    network.eval()
    return network, initial, final


def train_official_value_leaf(
    dataset: OfficialTrajectoryDataset,
    manifest: OfficialValueLeafManifest,
    *,
    seed: int = PHASE6_OFFICIAL_VALUE_LEAF_SEED,
    epochs: int = PHASE6_OFFICIAL_VALUE_LEAF_EPOCHS,
    learning_rate: float = PHASE6_OFFICIAL_VALUE_LEAF_LEARNING_RATE,
) -> OfficialValueLeafModel:
    rows = build_official_value_leaf_rows(dataset, manifest)
    train_rows = tuple(row for row in rows if row.split == "train")
    selection_rows = tuple(
        row
        for row in rows
        if row.split == "calibration" and row.calibration_partition == "selection"
    )
    if not train_rows or not selection_rows:
        raise ValueError("Phase 6AG train/selection rows are empty")
    train_rows_fingerprint = _fingerprint([row.row_id for row in train_rows])
    normalizer = NamedFeatureNormalizer.fit(
        [row.features for row in train_rows],
        fit_rows_fingerprint=train_rows_fingerprint,
    )
    candidates: list[
        tuple[OfficialValueLeafSelectionMetric, _OfficialValueLeafNetwork]
    ] = []
    endgame_rows = tuple(
        row for row in selection_rows if row.progress_bucket == "endgame"
    )
    if not endgame_rows:
        raise ValueError("Phase 6AG selection split contains no endgame rows")
    for hidden_dim in PHASE6_OFFICIAL_VALUE_LEAF_HIDDEN_GRID:
        for weight_decay in PHASE6_OFFICIAL_VALUE_LEAF_WEIGHT_DECAY_GRID:
            network, initial, final = _train_network(
                train_rows,
                normalizer,
                hidden_dim=hidden_dim,
                weight_decay=weight_decay,
                seed=seed + hidden_dim + int(weight_decay * 1e8),
                epochs=epochs,
                learning_rate=learning_rate,
            )
            with torch.no_grad():
                neural = network(
                    torch.tensor(
                        [normalizer.transform(row.features) for row in selection_rows],
                        dtype=torch.float64,
                    )
                ).tolist()
            for blend in PHASE6_OFFICIAL_VALUE_LEAF_BLEND_GRID:
                predictions = [
                    max(
                        -1.0,
                        min(
                            1.0,
                            blend * float(value)
                            + (1.0 - blend) * row.heuristic_value,
                        ),
                    )
                    for row, value in zip(selection_rows, neural, strict=True)
                ]
                state_mse, episode_mse, zero_mse, heuristic_mse = _point_metrics(
                    selection_rows, predictions
                )
                endgame_predictions = [
                    prediction
                    for row, prediction in zip(selection_rows, predictions, strict=True)
                    if row.progress_bucket == "endgame"
                ]
                endgame_mse = sum(
                    (prediction - row.outcome_utility) ** 2
                    for row, prediction in zip(
                        endgame_rows, endgame_predictions, strict=True
                    )
                ) / len(endgame_rows)
                endgame_reference = min(
                    sum(row.outcome_utility**2 for row in endgame_rows)
                    / len(endgame_rows),
                    sum(
                        (row.heuristic_value - row.outcome_utility) ** 2
                        for row in endgame_rows
                    )
                    / len(endgame_rows),
                )
                reference = max(min(zero_mse, heuristic_mse), 1e-12)
                end_reference = max(endgame_reference, 1e-12)
                score = max(episode_mse / reference, endgame_mse / end_reference)
                metric = OfficialValueLeafSelectionMetric(
                    hidden_dim=hidden_dim,
                    weight_decay=weight_decay,
                    blend_weight=blend,
                    initial_training_loss=initial,
                    final_training_loss=final,
                    selection_state_mse=state_mse,
                    selection_episode_mse=episode_mse,
                    selection_zero_episode_mse=zero_mse,
                    selection_heuristic_episode_mse=heuristic_mse,
                    selection_endgame_mse=endgame_mse,
                    selection_endgame_reference_mse=endgame_reference,
                    selection_score=score,
                )
                candidates.append((metric, network))
    selected_metric, selected_network = min(
        candidates,
        key=lambda item: (
            item[0].selection_score,
            item[0].selection_episode_mse,
            item[0].hidden_dim,
            item[0].weight_decay,
            -item[0].blend_weight,
        ),
    )
    model = OfficialValueLeafModel(
        source_dataset_fingerprint=dataset.fingerprint,
        source_manifest_fingerprint=manifest.fingerprint,
        normalizer=normalizer,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        selected_hidden_dim=selected_metric.hidden_dim,
        selected_weight_decay=selected_metric.weight_decay,
        selected_blend_weight=selected_metric.blend_weight,
        selection_metrics=tuple(metric for metric, _ in candidates),
        train_match_ids=tuple(
            sorted({row.match_id for row in train_rows})
        ),
        calibration_selection_match_ids=tuple(
            sorted({row.match_id for row in selection_rows})
        ),
        tensors=_tensor_artifacts(selected_network),
    )
    restored = OfficialValueLeafModel.from_dict(
        json.loads(model.to_json())
    )
    if restored.fingerprint != model.fingerprint:
        raise RuntimeError("Phase 6AG model roundtrip drifted")
    return model


@dataclass(frozen=True, slots=True)
class OfficialValueLeafPointMetric:
    role: str
    row_count: int
    match_count: int
    state_mse: float
    episode_mse: float
    zero_episode_mse: float
    heuristic_episode_mse: float

    def __post_init__(self) -> None:
        if not self.role or min(self.row_count, self.match_count) <= 0:
            raise ValueError("Phase 6AG point metric identity is invalid")
        for name in (
            "state_mse",
            "episode_mse",
            "zero_episode_mse",
            "heuristic_episode_mse",
        ):
            if getattr(self, name) < 0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"Phase 6AG {name} is invalid")


@dataclass(frozen=True, slots=True)
class OfficialValueLeafBucketMetric:
    progress_bucket: str
    calibration_row_count: int
    validation_row_count: int
    residual_radius: float
    validation_mse: float
    validation_coverage: float
    validation_mean_width: float

    def __post_init__(self) -> None:
        if self.progress_bucket not in PHASE6_OFFICIAL_VALUE_LEAF_PROGRESS_BUCKETS:
            raise ValueError("Phase 6AG bucket metric name is invalid")
        if min(self.calibration_row_count, self.validation_row_count) <= 0:
            raise ValueError("Phase 6AG bucket metric is empty")
        if not 0 <= self.residual_radius <= 2:
            raise ValueError("Phase 6AG conformal radius is invalid")
        if self.validation_mse < 0 or not math.isfinite(self.validation_mse):
            raise ValueError("Phase 6AG bucket MSE is invalid")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("Phase 6AG bucket coverage is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AG bucket interval width is invalid")


def _conformal_radius(residuals: Sequence[float], coverage: float) -> float:
    materialized = sorted(float(value) for value in residuals)
    if not materialized:
        raise ValueError("Phase 6AG conformal residual set is empty")
    rank = min(len(materialized), math.ceil((len(materialized) + 1) * coverage))
    return min(2.0, materialized[rank - 1])


@dataclass(frozen=True, slots=True)
class OfficialValueLeafCalibration:
    source_dataset_fingerprint: str
    source_manifest_fingerprint: str
    source_model_fingerprint: str
    requested_coverage: float
    calibration_bound_match_ids: tuple[str, ...]
    validation_match_ids: tuple[str, ...]
    point_metrics: tuple[OfficialValueLeafPointMetric, ...]
    bucket_metrics: tuple[OfficialValueLeafBucketMetric, ...]
    validation_coverage: float
    validation_mean_width: float
    point_model_gate_passed: bool
    progress_interval_gate_passed: bool
    implementation_holdout_gate_passed: bool
    implementation_heldout_value_leaf_calibrated: bool
    counterfactual_value_compatible: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_dataset_fingerprint",
            "source_manifest_fingerprint",
            "source_model_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.requested_coverage != PHASE6_OFFICIAL_VALUE_LEAF_COVERAGE:
            raise ValueError("Phase 6AG requested coverage drifted")
        if not self.calibration_bound_match_ids or not self.validation_match_ids:
            raise ValueError("Phase 6AG calibration/validation identities are empty")
        if set(self.calibration_bound_match_ids) & set(self.validation_match_ids):
            raise ValueError("Phase 6AG calibration/validation matches overlap")
        if tuple(item.progress_bucket for item in self.bucket_metrics) != (
            PHASE6_OFFICIAL_VALUE_LEAF_PROGRESS_BUCKETS
        ):
            raise ValueError("Phase 6AG bucket metrics are incomplete")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("Phase 6AG validation coverage is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AG validation width is invalid")
        expected = all(
            (
                self.point_model_gate_passed,
                self.progress_interval_gate_passed,
                self.implementation_holdout_gate_passed,
            )
        )
        if self.implementation_heldout_value_leaf_calibrated != expected:
            raise ValueError("Phase 6AG leaf admission gate is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AG calibration promoted a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_VALUE_LEAF_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "requested_coverage": self.requested_coverage,
            "calibration_bound_match_ids": list(self.calibration_bound_match_ids),
            "validation_match_ids": list(self.validation_match_ids),
            "point_metrics": [asdict(item) for item in self.point_metrics],
            "bucket_metrics": [asdict(item) for item in self.bucket_metrics],
            "validation_coverage": self.validation_coverage,
            "validation_mean_width": self.validation_mean_width,
            "gates": {
                "point_model_gate_passed": self.point_model_gate_passed,
                "progress_interval_gate_passed": self.progress_interval_gate_passed,
                "implementation_holdout_gate_passed": (
                    self.implementation_holdout_gate_passed
                ),
                "implementation_heldout_value_leaf_calibrated": (
                    self.implementation_heldout_value_leaf_calibrated
                ),
                "counterfactual_value_compatible": (
                    self.counterfactual_value_compatible
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialValueLeafCalibration":
        gates = raw["gates"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            source_manifest_fingerprint=str(raw["source_manifest_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            requested_coverage=float(raw["requested_coverage"]),
            calibration_bound_match_ids=tuple(
                str(value) for value in raw["calibration_bound_match_ids"]
            ),
            validation_match_ids=tuple(
                str(value) for value in raw["validation_match_ids"]
            ),
            point_metrics=tuple(
                OfficialValueLeafPointMetric(**dict(item))
                for item in raw["point_metrics"]
            ),
            bucket_metrics=tuple(
                OfficialValueLeafBucketMetric(**dict(item))
                for item in raw["bucket_metrics"]
            ),
            validation_coverage=float(raw["validation_coverage"]),
            validation_mean_width=float(raw["validation_mean_width"]),
            point_model_gate_passed=bool(gates["point_model_gate_passed"]),
            progress_interval_gate_passed=bool(
                gates["progress_interval_gate_passed"]
            ),
            implementation_holdout_gate_passed=bool(
                gates["implementation_holdout_gate_passed"]
            ),
            implementation_heldout_value_leaf_calibrated=bool(
                gates["implementation_heldout_value_leaf_calibrated"]
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
            raise ValueError("Phase 6AG calibration fingerprint mismatch")
        return result


def _point_metric(
    role: str,
    rows: Sequence[OfficialValueLeafRow],
    predictions: Sequence[float],
) -> OfficialValueLeafPointMetric:
    state, episode, zero, heuristic = _point_metrics(rows, predictions)
    return OfficialValueLeafPointMetric(
        role=role,
        row_count=len(rows),
        match_count=len({row.match_id for row in rows}),
        state_mse=state,
        episode_mse=episode,
        zero_episode_mse=zero,
        heuristic_episode_mse=heuristic,
    )


def calibrate_official_value_leaf(
    dataset: OfficialTrajectoryDataset,
    manifest: OfficialValueLeafManifest,
    model: OfficialValueLeafModel,
) -> OfficialValueLeafCalibration:
    if model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("Phase 6AG model/dataset mismatch")
    if model.source_manifest_fingerprint != manifest.fingerprint:
        raise ValueError("Phase 6AG model/manifest mismatch")
    rows = build_official_value_leaf_rows(dataset, manifest)
    selection_rows = tuple(
        row
        for row in rows
        if row.split == "calibration" and row.calibration_partition == "selection"
    )
    bound_rows = tuple(
        row
        for row in rows
        if row.split == "calibration" and row.calibration_partition == "bounds"
    )
    validation_rows = tuple(row for row in rows if row.split == "validation")
    provider = restore_official_value_leaf(model)
    selection_predictions = [provider.predict_row(row) for row in selection_rows]
    bound_predictions = [provider.predict_row(row) for row in bound_rows]
    validation_predictions = [provider.predict_row(row) for row in validation_rows]
    radii = {}
    for bucket in PHASE6_OFFICIAL_VALUE_LEAF_PROGRESS_BUCKETS:
        residuals = [
            abs(prediction - row.outcome_utility)
            for row, prediction in zip(bound_rows, bound_predictions, strict=True)
            if row.progress_bucket == bucket
        ]
        radii[bucket] = _conformal_radius(
            residuals, PHASE6_OFFICIAL_VALUE_LEAF_COVERAGE
        )
    bucket_metrics = []
    covered = 0
    widths = []
    for bucket in PHASE6_OFFICIAL_VALUE_LEAF_PROGRESS_BUCKETS:
        pairs = [
            (row, prediction)
            for row, prediction in zip(
                validation_rows, validation_predictions, strict=True
            )
            if row.progress_bucket == bucket
        ]
        radius = radii[bucket]
        bucket_covered = 0
        bucket_widths = []
        bucket_errors = []
        for row, prediction in pairs:
            lower = max(-1.0, prediction - radius)
            upper = min(1.0, prediction + radius)
            is_covered = lower - 1e-12 <= row.outcome_utility <= upper + 1e-12
            bucket_covered += int(is_covered)
            covered += int(is_covered)
            width = upper - lower
            bucket_widths.append(width)
            widths.append(width)
            bucket_errors.append((prediction - row.outcome_utility) ** 2)
        calibration_count = sum(
            row.progress_bucket == bucket for row in bound_rows
        )
        bucket_metrics.append(
            OfficialValueLeafBucketMetric(
                progress_bucket=bucket,
                calibration_row_count=calibration_count,
                validation_row_count=len(pairs),
                residual_radius=radius,
                validation_mse=sum(bucket_errors) / len(bucket_errors),
                validation_coverage=bucket_covered / len(pairs),
                validation_mean_width=sum(bucket_widths) / len(bucket_widths),
            )
        )
    point_metrics = (
        _point_metric("calibration_selection", selection_rows, selection_predictions),
        _point_metric("calibration_bounds", bound_rows, bound_predictions),
        _point_metric("validation_untouched", validation_rows, validation_predictions),
    )
    validation_metric = point_metrics[-1]
    point_gate = validation_metric.episode_mse < min(
        validation_metric.zero_episode_mse,
        validation_metric.heuristic_episode_mse,
    )
    bucket_by_name = {item.progress_bucket: item for item in bucket_metrics}
    interval_gate = (
        all(
            item.validation_coverage
            >= PHASE6_OFFICIAL_VALUE_LEAF_MINIMUM_BUCKET_COVERAGE
            for item in bucket_metrics
        )
        and sum(widths) / len(widths)
        <= PHASE6_OFFICIAL_VALUE_LEAF_MAXIMUM_OVERALL_WIDTH
        and bucket_by_name["endgame"].validation_mean_width
        <= PHASE6_OFFICIAL_VALUE_LEAF_MAXIMUM_ENDGAME_WIDTH
    )
    split_by_variant = dict(manifest.policy_variant_splits)
    implementation_gate = (
        set(split_by_variant) == set(OFFICIAL_INTENT_POLICY_VARIANT_IDS)
        and len({item.policy_variant_id for item in manifest.matches if item.split == "validation"})
        == 1
        and not (
            set(model.train_match_ids)
            | set(model.calibration_selection_match_ids)
            | {row.match_id for row in bound_rows}
        )
        & {row.match_id for row in validation_rows}
    )
    calibration = OfficialValueLeafCalibration(
        source_dataset_fingerprint=dataset.fingerprint,
        source_manifest_fingerprint=manifest.fingerprint,
        source_model_fingerprint=model.fingerprint,
        requested_coverage=PHASE6_OFFICIAL_VALUE_LEAF_COVERAGE,
        calibration_bound_match_ids=tuple(
            sorted({row.match_id for row in bound_rows})
        ),
        validation_match_ids=tuple(
            sorted({row.match_id for row in validation_rows})
        ),
        point_metrics=point_metrics,
        bucket_metrics=tuple(bucket_metrics),
        validation_coverage=covered / len(validation_rows),
        validation_mean_width=sum(widths) / len(widths),
        point_model_gate_passed=point_gate,
        progress_interval_gate_passed=interval_gate,
        implementation_holdout_gate_passed=implementation_gate,
        implementation_heldout_value_leaf_calibrated=(
            point_gate and interval_gate and implementation_gate
        ),
    )
    restored = OfficialValueLeafCalibration.from_dict(
        json.loads(calibration.to_json())
    )
    if restored.fingerprint != calibration.fingerprint:
        raise RuntimeError("Phase 6AG calibration roundtrip drifted")
    return calibration


@dataclass(frozen=True, slots=True)
class Phase6OfficialValueLeafEvidence:
    benchmark_id: str
    status: str
    manifest_fingerprint: str
    dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    match_count: int
    episode_count: int
    trajectory_step_count: int
    live_selection_count: int
    train_match_count: int
    calibration_match_count: int
    validation_match_count: int
    validation_episode_mse: float
    validation_zero_episode_mse: float
    validation_heuristic_episode_mse: float
    validation_coverage: float
    validation_mean_width: float
    validation_endgame_coverage: float
    validation_endgame_mean_width: float
    implementation_heldout_point_leaf_validated: bool
    implementation_heldout_value_leaf_calibrated: bool
    counterfactual_value_compatible: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False
    claims: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AG evidence identity is invalid")
        for name in (
            "manifest_fingerprint",
            "dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        for name in (
            "match_count",
            "episode_count",
            "trajectory_step_count",
            "live_selection_count",
            "train_match_count",
            "calibration_match_count",
            "validation_match_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Phase 6AG evidence {name} is invalid")
        for name in (
            "validation_episode_mse",
            "validation_zero_episode_mse",
            "validation_heuristic_episode_mse",
            "validation_mean_width",
            "validation_endgame_mean_width",
        ):
            if getattr(self, name) < 0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"Phase 6AG evidence {name} is invalid")
        for name in ("validation_coverage", "validation_endgame_coverage"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Phase 6AG evidence {name} is invalid")
        expected_point_gate = self.validation_episode_mse < min(
            self.validation_zero_episode_mse,
            self.validation_heuristic_episode_mse,
        )
        if self.implementation_heldout_point_leaf_validated != expected_point_gate:
            raise ValueError("Phase 6AG point-leaf evidence gate is inconsistent")
        if (
            self.implementation_heldout_value_leaf_calibrated
            and not self.implementation_heldout_point_leaf_validated
        ):
            raise ValueError("Phase 6AG interval gate cannot bypass its point gate")
        expected_status = (
            "implementation_heldout_official_value_leaf_calibrated_downstream_fail_closed"
            if self.implementation_heldout_value_leaf_calibrated
            else (
                "implementation_heldout_official_point_leaf_validated_intervals_rejected_downstream_fail_closed"
                if self.implementation_heldout_point_leaf_validated
                else "implementation_heldout_official_value_leaf_rejected_downstream_fail_closed"
            )
        )
        if self.status != expected_status:
            raise ValueError("Phase 6AG evidence status is inconsistent with its gates")
        if any(
            (
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AG evidence promoted a downstream capability")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6AG evidence claims contain duplicates")
        claim_map = dict(self.claims)
        expected_point_claim = (
            "implementation_heldout_validation_passed"
            if self.implementation_heldout_point_leaf_validated
            else "rejected_against_point_baselines"
        )
        expected_interval_claim = (
            "terminal_outcome_intervals_passed_not_expected_value"
            if self.implementation_heldout_value_leaf_calibrated
            else "rejected_as_too_wide_or_undercovered"
        )
        if claim_map.get("point_leaf") != expected_point_claim:
            raise ValueError("Phase 6AG point-leaf claim is inconsistent")
        if claim_map.get("expected_value_intervals") != expected_interval_claim:
            raise ValueError("Phase 6AG interval claim is inconsistent")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_VALUE_LEAF_SCHEMA_VERSION,
            **asdict(self),
        }
        result["claims"] = dict(self.claims)
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AG implementation-heldout official value leaf

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Genuine official matches | {self.match_count} |
| Perspective-isolated episodes | {self.episode_count} |
| Leakage-safe trajectory steps | {self.trajectory_step_count} |
| Genuine live selections | {self.live_selection_count} |
| Train / calibration / validation matches | {self.train_match_count} / {self.calibration_match_count} / {self.validation_match_count} |
| Validation neural MSE | `{self.validation_episode_mse:.6f}` |
| Validation zero / heuristic MSE | `{self.validation_zero_episode_mse:.6f}` / `{self.validation_heuristic_episode_mse:.6f}` |
| Validation coverage / mean width | `{self.validation_coverage:.6f}` / `{self.validation_mean_width:.6f}` |
| Endgame coverage / mean width | `{self.validation_endgame_coverage:.6f}` / `{self.validation_endgame_mean_width:.6f}` |
| Implementation-heldout point leaf validated | **{str(self.implementation_heldout_point_leaf_validated).lower()}** |
| Implementation-heldout value leaf admitted | **{str(self.implementation_heldout_value_leaf_calibrated).lower()}** |
| Counterfactual values / safe resolver / official control | **false / false / false** |

Official matches, selections, and truth replay frames genuinely execute in the
licensed engine. Perspective conversion, hashed features, authored policy
families, the neural leaf, and conformal intervals are abstractions. No
counterfactual, equilibrium, safety, control, submission, or strength claim is
enabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6OfficialValueLeafEvidence":
        values = dict(raw)
        values.pop("schema_version", None)
        fingerprint = values.pop("fingerprint", None)
        values["claims"] = tuple(
            (str(name), str(value)) for name, value in values["claims"].items()
        )
        result = cls(**values)
        if fingerprint not in (None, result.fingerprint):
            raise ValueError("Phase 6AG evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialValueLeafArtifacts:
    manifest: OfficialValueLeafManifest
    dataset: OfficialTrajectoryDataset
    model: OfficialValueLeafModel
    calibration: OfficialValueLeafCalibration
    evidence: Phase6OfficialValueLeafEvidence


def _runtime_fingerprint(runtime: Path) -> str:
    names = (
        "cg/cg.dll",
        "cg/libcg.so",
        "cg/libcg-arm64.so",
        "cg/libcg.dylib",
        "cg/api.py",
        "cg/game.py",
        "cg/sim.py",
    )
    return _fingerprint(
        {
            name: _file_sha256(runtime / name)
            for name in names
            if (runtime / name).is_file()
        }
    )


def _range_candidates(
    profiles: Sequence[KaggleDeckProfile],
) -> tuple[PublicRangeCandidate, ...]:
    return tuple(
        PublicRangeCandidate(
            candidate_id=profile.archetype,
            archetype_id=profile.archetype,
            style_id="unknown",
            card_ids=profile.card_ids,
            deck_fingerprint=profile.fingerprint,
            prior=1.0,
            source=profile.provenance,
        )
        for profile in profiles
    )


def collect_official_value_leaf_dataset(
    *,
    project_root: str | Path | None = None,
    maximum_selections: int = PHASE6_OFFICIAL_VALUE_LEAF_MAX_SELECTIONS,
) -> tuple[OfficialTrajectoryDataset, OfficialValueLeafManifest]:
    if maximum_selections <= 0:
        raise ValueError("Phase 6AG maximum selections must be positive")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    path_text = str(runtime)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
        visualize_data,
    )

    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profiles = OFFICIAL_PANEL_DECKS + (target,)
    if len({profile.archetype for profile in profiles}) != len(profiles):
        raise RuntimeError("Phase 6AG candidate catalog contains duplicate archetypes")
    engine_sha = _file_sha256(runtime / "cg" / "cg.dll")
    card_pool_sha = _file_sha256(root / "work" / "official_kaggle" / "EN_Card_Data.csv")
    runtime_fingerprint = _runtime_fingerprint(runtime)
    environment_fingerprint = _fingerprint(
        {
            "runtime": runtime_fingerprint,
            "engine": engine_sha,
            "card_pool": card_pool_sha,
            "deck": target.fingerprint,
            "candidate_decks": {
                profile.archetype: profile.fingerprint for profile in profiles
            },
            "policy_variants": {
                variant: official_intent_policy_variant_fingerprint(variant)
                for variant in OFFICIAL_INTENT_POLICY_VARIANT_IDS
            },
            "protocol": "phase6ag-full-game-implementation-heldout-v1",
        }
    )
    converter = OfficialReplayConverter(
        extra_known_event_types=("Coin",),
        extra_public_scalar_keys=("head",),
        include_coarse_main_action_events=True,
    )
    episodes = []
    provisional_matches = []
    split_by_variant = _policy_split_map()
    for variant in OFFICIAL_INTENT_POLICY_VARIANT_IDS:
        split = split_by_variant[variant]
        for candidate_index, candidate in enumerate(profiles):
            for intent_index, intent in enumerate(OPPONENT_INTENTS):
                for replica in range(PHASE6_OFFICIAL_VALUE_LEAF_REPLICAS):
                    target_reverse = bool(replica % 2)
                    opponent_reverse = not target_reverse
                    match_id = (
                        f"phase6ag-{split}-{variant}-{candidate.archetype}-"
                        f"{intent.value}-r{replica}"
                    )
                    observation, start = battle_start(
                        list(target.card_ids), list(candidate.card_ids)
                    )
                    if observation is None:
                        raise RuntimeError(
                            "official engine rejected Phase 6AG decks: "
                            f"player={start.errorPlayer}, error={start.errorType}"
                        )
                    policies = (
                        PriorityBaseline(
                            target.card_ids,
                            style="tempo",
                            reverse=target_reverse,
                        ),
                        StrategicIntentPriorityBaseline(
                            candidate.card_ids,
                            intent=intent,
                            reverse=opponent_reverse,
                            variant_id=variant,
                        ),
                    )
                    selections = 0
                    replay_text = ""
                    try:
                        while (
                            observation is not None
                            and _safe_int(_state(observation).get("result"), -1) == -1
                            and selections < maximum_selections
                        ):
                            actor = _safe_int(_state(observation).get("yourIndex"), -1)
                            if actor not in (0, 1):
                                raise RuntimeError("Phase 6AG official actor is invalid")
                            contract = SelectionContract.from_observation(observation)
                            selection = contract.validate(policies[actor].choose(observation))
                            observation = battle_select(list(selection))
                            selections += 1
                        if observation is None:
                            raise RuntimeError("Phase 6AG battle ended without an observation")
                        state = _state(observation)
                        winner = _safe_int(state.get("result"), -1)
                        turns = _safe_int(state.get("turn"), 0)
                        if winner not in (0, 1, 2):
                            raise RuntimeError(
                                f"Phase 6AG match failed to terminate within {maximum_selections} selections"
                            )
                        replay_text = visualize_data()
                    finally:
                        battle_finish()
                    converted = converter.convert_match(
                        replay_text,
                        match_id=match_id,
                        engine_sha256=engine_sha,
                        card_pool_sha256=card_pool_sha,
                        environment_fingerprint=environment_fingerprint,
                        deck_fingerprints=(target.fingerprint, candidate.fingerprint),
                    )
                    episodes.extend(converted)
                    provisional_matches.append(
                        OfficialValueLeafMatch(
                            match_id=match_id,
                            split=split,
                            calibration_partition=_calibration_partition(
                                split,
                                replica,
                                candidate_index=candidate_index,
                                intent_index=intent_index,
                            ),
                            policy_variant_id=variant,
                            policy_variant_fingerprint=(
                                official_intent_policy_variant_fingerprint(variant)
                            ),
                            candidate_id=candidate.archetype,
                            intent_id=intent.value,
                            replica=replica,
                            target_reverse=target_reverse,
                            opponent_reverse=opponent_reverse,
                            winner=winner,
                            turns=turns,
                            selection_count=selections,
                            replay_sha256=converted[0].provenance.replay_sha256,
                            observer_episode_fingerprints=(
                                converted[0].fingerprint,
                                converted[1].fingerprint,
                            ),
                        )
                    )
    public_replays = build_public_replay_dataset(episodes)
    if not public_replays.validated or public_replays.summary["unknown_event_types"]:
        raise RuntimeError(
            "Phase 6AG public replay conversion failed its audit: "
            f"validated={public_replays.validated}, "
            f"summary={public_replays.summary}"
        )
    match_splits = {item.match_id: item.split for item in provisional_matches}
    dataset = build_official_trajectory_dataset(
        public_replays,
        _range_candidates(profiles),
        match_splits=match_splits,
    )
    if _raw_private_key_count(dataset.to_dict()) != 0:
        raise RuntimeError("Phase 6AG serialized a forbidden raw private key")
    manifest = OfficialValueLeafManifest(
        trajectory_dataset_fingerprint=dataset.fingerprint,
        public_replay_dataset_fingerprint=public_replays.fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        engine_sha256=engine_sha,
        card_pool_sha256=card_pool_sha,
        environment_fingerprint=environment_fingerprint,
        deck_fingerprint=target.fingerprint,
        target_policy_id="fixed_player0_tempo",
        policy_variant_splits=PHASE6_OFFICIAL_VALUE_LEAF_POLICY_SPLITS,
        policy_variant_fingerprints=tuple(
            (variant, official_intent_policy_variant_fingerprint(variant))
            for variant in OFFICIAL_INTENT_POLICY_VARIANT_IDS
        ),
        candidate_deck_fingerprints=tuple(
            (profile.archetype, profile.fingerprint) for profile in profiles
        ),
        intent_ids=tuple(intent.value for intent in OPPONENT_INTENTS),
        replicas_per_stratum=PHASE6_OFFICIAL_VALUE_LEAF_REPLICAS,
        matches=tuple(provisional_matches),
    )
    restored_manifest = OfficialValueLeafManifest.from_dict(
        json.loads(manifest.to_json())
    )
    if restored_manifest.fingerprint != manifest.fingerprint:
        raise RuntimeError("Phase 6AG source manifest roundtrip drifted")
    return dataset, manifest


def run_phase6_official_value_leaf(
    *,
    project_root: str | Path | None = None,
    source_dataset_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    seed: int = PHASE6_OFFICIAL_VALUE_LEAF_SEED,
    epochs: int = PHASE6_OFFICIAL_VALUE_LEAF_EPOCHS,
    learning_rate: float = PHASE6_OFFICIAL_VALUE_LEAF_LEARNING_RATE,
) -> Phase6OfficialValueLeafArtifacts:
    if (source_dataset_path is None) != (source_manifest_path is None):
        raise ValueError("Phase 6AG source dataset and manifest must be supplied together")
    if source_dataset_path is None:
        dataset, manifest = collect_official_value_leaf_dataset(
            project_root=project_root
        )
    else:
        dataset = OfficialTrajectoryDataset.from_dict(
            json.loads(Path(source_dataset_path).read_text(encoding="utf-8"))
        )
        manifest = OfficialValueLeafManifest.from_dict(
            json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))
        )
    model = train_official_value_leaf(
        dataset,
        manifest,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    calibration = calibrate_official_value_leaf(dataset, manifest, model)
    split_counts = dict(manifest.split_match_counts)
    validation_metric = next(
        item for item in calibration.point_metrics if item.role == "validation_untouched"
    )
    endgame_metric = next(
        item
        for item in calibration.bucket_metrics
        if item.progress_bucket == "endgame"
    )
    admitted = calibration.implementation_heldout_value_leaf_calibrated
    evidence = Phase6OfficialValueLeafEvidence(
        benchmark_id="phase6ag-implementation-heldout-official-value-leaf-v1",
        status=(
            "implementation_heldout_official_value_leaf_calibrated_downstream_fail_closed"
            if admitted
            else (
                "implementation_heldout_official_point_leaf_validated_intervals_rejected_downstream_fail_closed"
                if calibration.point_model_gate_passed
                else "implementation_heldout_official_value_leaf_rejected_downstream_fail_closed"
            )
        ),
        manifest_fingerprint=manifest.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        match_count=len(manifest.matches),
        episode_count=len(dataset.episodes),
        trajectory_step_count=len(dataset.steps),
        live_selection_count=manifest.live_selection_count,
        train_match_count=split_counts["train"],
        calibration_match_count=split_counts["calibration"],
        validation_match_count=split_counts["validation"],
        validation_episode_mse=validation_metric.episode_mse,
        validation_zero_episode_mse=validation_metric.zero_episode_mse,
        validation_heuristic_episode_mse=(
            validation_metric.heuristic_episode_mse
        ),
        validation_coverage=calibration.validation_coverage,
        validation_mean_width=calibration.validation_mean_width,
        validation_endgame_coverage=endgame_metric.validation_coverage,
        validation_endgame_mean_width=endgame_metric.validation_mean_width,
        implementation_heldout_point_leaf_validated=(
            calibration.point_model_gate_passed
        ),
        implementation_heldout_value_leaf_calibrated=admitted,
        claims=(
            ("official_execution", "matches_selections_and_truth_replay_frames_only"),
            ("derived_dataset", "leakage_safe_offline_information_set_abstraction"),
            ("policy_labels", "controlled_authored_families_not_human_truth"),
            ("leaf_target", "on_policy_terminal_outcome_not_counterfactual_value"),
            (
                "point_leaf",
                "implementation_heldout_validation_passed"
                if calibration.point_model_gate_passed
                else "rejected_against_point_baselines",
            ),
            (
                "expected_value_intervals",
                "terminal_outcome_intervals_passed_not_expected_value"
                if admitted
                else "rejected_as_too_wide_or_undercovered",
            ),
            ("counterfactual_bounds", "not_available"),
            ("equilibrium_blueprint", "not_established"),
            ("safe_resolver", "disabled"),
            ("official_strategy_control", "disabled"),
            ("kaggle_submission", "not_attempted"),
            ("pokemon_strength", "not_claimed"),
            ("phase6_complete", "false"),
        ),
    )
    restored_evidence = Phase6OfficialValueLeafEvidence.from_dict(
        json.loads(evidence.to_json())
    )
    if restored_evidence.fingerprint != evidence.fingerprint:
        raise RuntimeError("Phase 6AG evidence roundtrip drifted")
    return Phase6OfficialValueLeafArtifacts(
        manifest=manifest,
        dataset=dataset,
        model=model,
        calibration=calibration,
        evidence=evidence,
    )


def main() -> None:
    print(run_phase6_official_value_leaf().evidence.to_json())


if __name__ == "__main__":
    main()
