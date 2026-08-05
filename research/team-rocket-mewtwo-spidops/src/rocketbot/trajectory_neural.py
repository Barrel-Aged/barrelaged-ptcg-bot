"""Offline neural bootstrap over logged official-engine trajectories.

The source transitions in :mod:`rocketbot.official_trajectory` genuinely
occurred in the native competition engine.  This module trains only on the
derived, non-branchable trajectory abstraction.  Its policy head imitates the
logged behavior policy; it is deliberately *not* a game-theoretic blueprint,
an equilibrium solver, a counterfactual-value provider, or an official-engine
execution path.

The bootstrap is useful for three narrower purposes:

* proving that the fixed-width public/private/action schemas are learnable;
* establishing split-safe, restorable representation pretraining; and
* supplying an initialization point for a later equilibrium-oriented trainer
  once a branchable information-set game exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .game_theory import AbstractAction
from .neural_blueprint import TensorArtifact
from .official_trajectory import (
    LoggedInformationSetStep,
    OfficialTrajectoryDataset,
)


TRAJECTORY_NEURAL_SCHEMA_VERSION = 1
DEFAULT_TRAJECTORY_NEURAL_SEED = 20260726
TEMPERATURE_GRID = (0.50, 0.67, 0.80, 1.00, 1.25, 1.50, 2.00, 3.00)


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


@dataclass(frozen=True, slots=True)
class NamedFeatureNormalizer:
    """Named affine transform fitted on one declared split only."""

    names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    fit_split: str
    fit_rows_fingerprint: str

    def __post_init__(self) -> None:
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("normalizer feature names must be non-empty and unique")
        if len(self.means) != len(self.names) or len(self.scales) != len(self.names):
            raise ValueError("normalizer vectors must match feature width")
        if self.fit_split != "train":
            raise ValueError("trajectory normalizers must be fitted on train only")
        _validate_digest(self.fit_rows_fingerprint, "fit_rows_fingerprint")
        if not all(math.isfinite(value) for value in self.means):
            raise ValueError("normalizer means must be finite")
        if not all(math.isfinite(value) and value > 0 for value in self.scales):
            raise ValueError("normalizer scales must be positive and finite")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "names": list(self.names),
            "means": list(self.means),
            "scales": list(self.scales),
            "fit_split": self.fit_split,
            "fit_rows_fingerprint": self.fit_rows_fingerprint,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NamedFeatureNormalizer":
        result = cls(
            names=tuple(str(value) for value in raw["names"]),
            means=tuple(float(value) for value in raw["means"]),
            scales=tuple(float(value) for value in raw["scales"]),
            fit_split=str(raw["fit_split"]),
            fit_rows_fingerprint=str(raw["fit_rows_fingerprint"]),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("normalizer fingerprint mismatch")
        return result

    @classmethod
    def fit(
        cls,
        rows: Sequence[Sequence[tuple[str, float]]],
        *,
        fit_rows_fingerprint: str,
    ) -> "NamedFeatureNormalizer":
        materialized = tuple(tuple(row) for row in rows)
        if not materialized:
            raise ValueError("normalizer requires training rows")
        names = tuple(name for name, _ in materialized[0])
        if any(tuple(name for name, _ in row) != names for row in materialized):
            raise ValueError("normalizer rows have inconsistent named schemas")
        columns = tuple(
            tuple(_finite(row[index][1], names[index]) for row in materialized)
            for index in range(len(names))
        )
        means = tuple(sum(column) / len(column) for column in columns)
        scales = []
        for column, mean in zip(columns, means, strict=True):
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            deviation = math.sqrt(variance)
            scales.append(deviation if deviation >= 1e-8 else 1.0)
        return cls(
            names=names,
            means=means,
            scales=tuple(scales),
            fit_split="train",
            fit_rows_fingerprint=fit_rows_fingerprint,
        )

    def transform(self, items: Sequence[tuple[str, float]]) -> tuple[float, ...]:
        materialized = tuple(items)
        if tuple(name for name, _ in materialized) != self.names:
            raise ValueError("feature schema does not match normalizer")
        return tuple(
            (_finite(value, name) - mean) / scale
            for (name, value), mean, scale in zip(
                materialized,
                self.means,
                self.scales,
                strict=True,
            )
        )

    def inverse(self, values: Sequence[float]) -> tuple[tuple[str, float], ...]:
        materialized = tuple(float(value) for value in values)
        if len(materialized) != len(self.names):
            raise ValueError("normalized feature width mismatch")
        return tuple(
            (name, _finite(value * scale + mean, name))
            for name, value, mean, scale in zip(
                self.names,
                materialized,
                self.means,
                self.scales,
                strict=True,
            )
        )


class LoggedTrajectoryBootstrapNetwork(nn.Module):
    """Multi-task encoder with a variable-cardinality legal-action head."""

    def __init__(
        self,
        *,
        public_width: int,
        private_width: int,
        action_width: int,
        candidate_count: int,
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        if min(public_width, private_width, action_width, candidate_count, hidden_dim) <= 0:
            raise ValueError("trajectory network widths must be positive")
        self.public_width = int(public_width)
        self.private_width = int(private_width)
        self.action_width = int(action_width)
        self.candidate_count = int(candidate_count)
        self.hidden_dim = int(hidden_dim)
        private_dim = max(8, hidden_dim // 2)
        self.public_encoder = nn.Sequential(
            nn.Linear(public_width, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.private_encoder = nn.Sequential(
            nn.Linear(private_width, private_dim),
            nn.Tanh(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim + private_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_width, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_query = nn.Linear(hidden_dim, hidden_dim)
        self.policy_action_bias = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        # The range head is public-only by construction.
        self.range_head = nn.Linear(hidden_dim, candidate_count)
        self.next_public_delta_head = nn.Linear(hidden_dim, public_width)
        # Identity is a strong and auditable baseline for a semi-Markov public
        # transition.  Start there instead of injecting an arbitrary random
        # delta; training must earn any departure from it.
        nn.init.zeros_(self.next_public_delta_head.weight)
        nn.init.zeros_(self.next_public_delta_head.bias)
        self.double()

    def forward(
        self,
        public_features: Tensor,
        private_features: Tensor,
        action_features: Tensor,
        action_counts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if public_features.ndim != 2 or public_features.shape[1] != self.public_width:
            raise ValueError("public feature tensor shape mismatch")
        if private_features.ndim != 2 or private_features.shape[1] != self.private_width:
            raise ValueError("private feature tensor shape mismatch")
        if action_features.ndim != 2 or action_features.shape[1] != self.action_width:
            raise ValueError("action feature tensor shape mismatch")
        if action_counts.ndim != 1 or len(action_counts) != len(public_features):
            raise ValueError("action-count tensor shape mismatch")
        if int(action_counts.sum()) != len(action_features) or bool((action_counts <= 0).any()):
            raise ValueError("action counts do not partition action rows")
        public_hidden = self.public_encoder(public_features)
        private_hidden = self.private_encoder(private_features)
        state_hidden = self.state_encoder(torch.cat((public_hidden, private_hidden), dim=-1))
        action_hidden = self.action_encoder(action_features)
        state_indices = torch.repeat_interleave(
            torch.arange(len(public_features), device=public_features.device),
            action_counts,
        )
        queries = self.policy_query(state_hidden)[state_indices]
        policy_logits = (
            (queries * action_hidden).sum(dim=-1) / math.sqrt(self.hidden_dim)
            + self.policy_action_bias(action_hidden).squeeze(-1)
        )
        outcome_values = torch.tanh(self.value_head(state_hidden).squeeze(-1))
        range_logits = self.range_head(public_hidden)
        next_public = public_features + self.next_public_delta_head(state_hidden)
        return policy_logits, outcome_values, range_logits, next_public, state_hidden


@dataclass(frozen=True, slots=True)
class TrajectoryBootstrapArtifact:
    dataset_fingerprint: str
    source_public_replay_fingerprint: str
    architecture_fingerprint: str
    candidate_ids: tuple[str, ...]
    public_normalizer: NamedFeatureNormalizer
    private_normalizer: NamedFeatureNormalizer
    action_normalizer: NamedFeatureNormalizer
    hidden_dim: int
    seed: int
    epochs: int
    learning_rate: float
    policy_loss_weight: float
    value_loss_weight: float
    range_loss_weight: float
    transition_loss_weight: float
    initial_training_loss: float
    final_training_loss: float
    policy_temperature: float
    range_temperature: float
    train_match_ids: tuple[str, ...]
    calibration_match_ids: tuple[str, ...]
    validation_match_ids: tuple[str, ...]
    train_batch_fingerprint: str
    calibration_batch_fingerprint: str
    validation_batch_fingerprint: str
    source_official_engine_executed: bool
    model_official_engine_executed: bool
    equilibrium_trained: bool
    deployable_blueprint: bool
    counterfactual_value_provider: bool
    tensors: tuple[TensorArtifact, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_fingerprint",
            "source_public_replay_fingerprint",
            "architecture_fingerprint",
            "train_batch_fingerprint",
            "calibration_batch_fingerprint",
            "validation_batch_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("trajectory candidate IDs must be non-empty and unique")
        if self.hidden_dim <= 0 or self.epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("trajectory bootstrap hyperparameters must be positive")
        weights = (
            self.policy_loss_weight,
            self.value_loss_weight,
            self.range_loss_weight,
            self.transition_loss_weight,
        )
        if not all(math.isfinite(value) and value > 0 for value in weights):
            raise ValueError("trajectory bootstrap loss weights must be positive")
        for field_name in (
            "initial_training_loss",
            "final_training_loss",
            "policy_temperature",
            "range_temperature",
        ):
            if _finite(getattr(self, field_name), field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        split_sets = tuple(
            set(values)
            for values in (
                self.train_match_ids,
                self.calibration_match_ids,
                self.validation_match_ids,
            )
        )
        if any(not values for values in split_sets):
            raise ValueError("trajectory bootstrap requires all three match splits")
        if any(split_sets[left] & split_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("trajectory bootstrap match splits overlap")
        if not self.source_official_engine_executed:
            raise ValueError("trajectory bootstrap source must be official execution")
        if any(
            (
                self.model_official_engine_executed,
                self.equilibrium_trained,
                self.deployable_blueprint,
                self.counterfactual_value_provider,
            )
        ):
            raise ValueError("offline behavior bootstrap cannot claim game-theoretic deployment")
        names = [tensor.name for tensor in self.tensors]
        if not names or len(names) != len(set(names)):
            raise ValueError("trajectory bootstrap tensors must be unique and non-empty")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": TRAJECTORY_NEURAL_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_public_replay_fingerprint": self.source_public_replay_fingerprint,
            "architecture_fingerprint": self.architecture_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "normalizers": {
                "public": self.public_normalizer.to_dict(),
                "private": self.private_normalizer.to_dict(),
                "action": self.action_normalizer.to_dict(),
            },
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "loss_weights": {
                "policy": self.policy_loss_weight,
                "value": self.value_loss_weight,
                "range": self.range_loss_weight,
                "transition": self.transition_loss_weight,
            },
            "initial_training_loss": self.initial_training_loss,
            "final_training_loss": self.final_training_loss,
            "temperatures": {
                "policy": self.policy_temperature,
                "range": self.range_temperature,
            },
            "split_match_ids": {
                "train": list(self.train_match_ids),
                "calibration": list(self.calibration_match_ids),
                "validation": list(self.validation_match_ids),
            },
            "split_batch_fingerprints": {
                "train": self.train_batch_fingerprint,
                "calibration": self.calibration_batch_fingerprint,
                "validation": self.validation_batch_fingerprint,
            },
            "claims": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "model_official_engine_executed": self.model_official_engine_executed,
                "equilibrium_trained": self.equilibrium_trained,
                "deployable_blueprint": self.deployable_blueprint,
                "counterfactual_value_provider": self.counterfactual_value_provider,
            },
            "tensors": [asdict(tensor) for tensor in self.tensors],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TrajectoryBootstrapArtifact":
        normalizers = raw["normalizers"]
        weights = raw["loss_weights"]
        temperatures = raw["temperatures"]
        split_ids = raw["split_match_ids"]
        split_fingerprints = raw["split_batch_fingerprints"]
        claims = raw["claims"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            source_public_replay_fingerprint=str(raw["source_public_replay_fingerprint"]),
            architecture_fingerprint=str(raw["architecture_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            public_normalizer=NamedFeatureNormalizer.from_dict(normalizers["public"]),
            private_normalizer=NamedFeatureNormalizer.from_dict(normalizers["private"]),
            action_normalizer=NamedFeatureNormalizer.from_dict(normalizers["action"]),
            hidden_dim=int(raw["hidden_dim"]),
            seed=int(raw["seed"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            policy_loss_weight=float(weights["policy"]),
            value_loss_weight=float(weights["value"]),
            range_loss_weight=float(weights["range"]),
            transition_loss_weight=float(weights["transition"]),
            initial_training_loss=float(raw["initial_training_loss"]),
            final_training_loss=float(raw["final_training_loss"]),
            policy_temperature=float(temperatures["policy"]),
            range_temperature=float(temperatures["range"]),
            train_match_ids=tuple(str(value) for value in split_ids["train"]),
            calibration_match_ids=tuple(str(value) for value in split_ids["calibration"]),
            validation_match_ids=tuple(str(value) for value in split_ids["validation"]),
            train_batch_fingerprint=str(split_fingerprints["train"]),
            calibration_batch_fingerprint=str(split_fingerprints["calibration"]),
            validation_batch_fingerprint=str(split_fingerprints["validation"]),
            source_official_engine_executed=bool(claims["source_official_engine_executed"]),
            model_official_engine_executed=bool(claims["model_official_engine_executed"]),
            equilibrium_trained=bool(claims["equilibrium_trained"]),
            deployable_blueprint=bool(claims["deployable_blueprint"]),
            counterfactual_value_provider=bool(claims["counterfactual_value_provider"]),
            tensors=tuple(
                TensorArtifact(
                    name=str(item["name"]),
                    shape=tuple(int(value) for value in item["shape"]),
                    values=tuple(float(value) for value in item["values"]),
                )
                for item in raw["tensors"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("trajectory bootstrap artifact fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class TrajectorySplitMetrics:
    split: str
    steps: int
    policy_nll: float
    uniform_policy_nll: float
    policy_top1_accuracy: float
    outcome_mse: float
    constant_outcome_mse: float
    range_nll: float
    uniform_range_nll: float
    range_top1_accuracy: float
    transition_mse: float
    identity_transition_mse: float

    def __post_init__(self) -> None:
        if self.split not in {"train", "calibration", "validation"} or self.steps <= 0:
            raise ValueError("trajectory split metric identity is invalid")
        for field_name in (
            "policy_nll",
            "uniform_policy_nll",
            "outcome_mse",
            "constant_outcome_mse",
            "range_nll",
            "uniform_range_nll",
            "transition_mse",
            "identity_transition_mse",
        ):
            if _finite(getattr(self, field_name), field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in ("policy_top1_accuracy", "range_top1_accuracy"):
            value = _finite(getattr(self, field_name), field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrajectoryBootstrapPrediction:
    action_probabilities: tuple[tuple[str, float], ...]
    outcome_value: float
    opponent_range: tuple[tuple[str, float], ...]
    next_public_features: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        action_ids = [name for name, _ in self.action_probabilities]
        candidate_ids = [name for name, _ in self.opponent_range]
        if not action_ids or len(action_ids) != len(set(action_ids)):
            raise ValueError("prediction action support must be non-empty and unique")
        if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("prediction range support must be non-empty and unique")
        for label, values in (
            ("action", self.action_probabilities),
            ("range", self.opponent_range),
        ):
            probabilities = [value for _, value in values]
            if any(not math.isfinite(value) or value < 0 for value in probabilities):
                raise ValueError(f"prediction {label} probabilities are invalid")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
                raise ValueError(f"prediction {label} probabilities must sum to one")
        if not -1.0 <= _finite(self.outcome_value, "outcome_value") <= 1.0:
            raise ValueError("prediction outcome value must be in [-1, 1]")
        if not self.next_public_features:
            raise ValueError("prediction must contain next-public features")


@dataclass(frozen=True, slots=True)
class TrajectoryBootstrapTrainingReport:
    artifact_fingerprint: str
    initial_training_loss: float
    final_training_loss: float
    gradient_parameter_names: tuple[str, ...]
    policy_temperature: float
    range_temperature: float
    metrics: tuple[TrajectorySplitMetrics, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.artifact_fingerprint, "artifact_fingerprint")
        if not self.gradient_parameter_names:
            raise ValueError("trajectory bootstrap requires non-zero gradient evidence")
        if self.final_training_loss >= self.initial_training_loss:
            raise ValueError("trajectory bootstrap training loss did not decrease")
        if tuple(metric.split for metric in self.metrics) != (
            "train",
            "calibration",
            "validation",
        ):
            raise ValueError("trajectory bootstrap metrics must cover ordered splits")


@dataclass(slots=True)
class TrajectoryBootstrapTrainingRun:
    provider: "LoggedTrajectoryBootstrapProvider"
    artifact: TrajectoryBootstrapArtifact
    report: TrajectoryBootstrapTrainingReport


@dataclass(frozen=True, slots=True)
class _TrajectoryBatch:
    split: str
    public_features: Tensor
    private_features: Tensor
    action_features: Tensor
    action_counts: Tensor
    action_offsets: tuple[int, ...]
    behavior_targets: Tensor
    outcome_targets: Tensor
    range_targets: Tensor
    next_public_targets: Tensor
    steps: tuple[LoggedInformationSetStep, ...]
    fingerprint: str


def _normalizer_fit_fingerprint(
    dataset: OfficialTrajectoryDataset,
    *,
    feature_family: str,
) -> str:
    return _fingerprint(
        {
            "dataset": dataset.fingerprint,
            "split": "train",
            "feature_family": feature_family,
            "steps": [
                step.fingerprint for step in dataset.steps if step.split == "train"
            ],
        }
    )


def _fit_normalizers(
    dataset: OfficialTrajectoryDataset,
) -> tuple[NamedFeatureNormalizer, NamedFeatureNormalizer, NamedFeatureNormalizer]:
    train_steps = tuple(step for step in dataset.steps if step.split == "train")
    public = NamedFeatureNormalizer.fit(
        [step.public_belief.public_features for step in train_steps],
        fit_rows_fingerprint=_normalizer_fit_fingerprint(
            dataset,
            feature_family="public",
        ),
    )
    private = NamedFeatureNormalizer.fit(
        [step.private_features for step in train_steps],
        fit_rows_fingerprint=_normalizer_fit_fingerprint(
            dataset,
            feature_family="private",
        ),
    )
    action = NamedFeatureNormalizer.fit(
        [
            item.features
            for step in train_steps
            for item in step.information_set.legal_actions
        ],
        fit_rows_fingerprint=_normalizer_fit_fingerprint(
            dataset,
            feature_family="action",
        ),
    )
    return public, private, action


def _split_match_ids(
    dataset: OfficialTrajectoryDataset,
    split: str,
) -> tuple[str, ...]:
    return tuple(
        sorted({episode.match_id for episode in dataset.episodes if episode.split == split})
    )


def _trajectory_batch(
    dataset: OfficialTrajectoryDataset,
    split: str,
    *,
    public_normalizer: NamedFeatureNormalizer,
    private_normalizer: NamedFeatureNormalizer,
    action_normalizer: NamedFeatureNormalizer,
) -> _TrajectoryBatch:
    if split not in {"train", "calibration", "validation"}:
        raise ValueError("unknown trajectory split")
    candidate_index = {
        candidate.candidate_id: index for index, candidate in enumerate(dataset.candidates)
    }
    records = [
        (episode, step)
        for episode in dataset.episodes
        if episode.split == split
        for step in episode.steps
    ]
    if not records:
        raise ValueError(f"trajectory split {split} is empty")
    public_rows = []
    private_rows = []
    action_rows = []
    action_counts = []
    behavior_targets = []
    outcome_targets = []
    range_targets = []
    next_public_targets = []
    steps = []
    payload = []
    offsets = [0]
    for episode, step in records:
        if step.behavior_action_id is None or step.behavior_indices is None:
            raise ValueError("neural trajectory bootstrap requires behavior labels")
        actions = step.information_set.legal_actions
        matching = [
            index
            for index, action in enumerate(actions)
            if action.action_id == step.behavior_action_id
            and action.official_option_indices == step.behavior_indices
        ]
        if len(matching) != 1:
            raise ValueError("behavior target does not identify exactly one legal action")
        public_rows.append(public_normalizer.transform(step.public_belief.public_features))
        private_rows.append(private_normalizer.transform(step.private_features))
        action_rows.extend(action_normalizer.transform(action.features) for action in actions)
        action_counts.append(len(actions))
        behavior_targets.append(matching[0])
        outcome_targets.append(step.outcome_utility)
        range_targets.append(candidate_index[episode.truth_candidate_id_supervision_only])
        next_public_targets.append(public_normalizer.transform(step.next_public_features))
        offsets.append(offsets[-1] + len(actions))
        steps.append(step)
        payload.append(
            {
                "match_id": episode.match_id,
                "observer_player": episode.observer_player,
                "step": step.fingerprint,
                "behavior_target": matching[0],
                "range_target": range_targets[-1],
            }
        )
    return _TrajectoryBatch(
        split=split,
        public_features=torch.tensor(public_rows, dtype=torch.float64),
        private_features=torch.tensor(private_rows, dtype=torch.float64),
        action_features=torch.tensor(action_rows, dtype=torch.float64),
        action_counts=torch.tensor(action_counts, dtype=torch.long),
        action_offsets=tuple(offsets),
        behavior_targets=torch.tensor(behavior_targets, dtype=torch.long),
        outcome_targets=torch.tensor(outcome_targets, dtype=torch.float64),
        range_targets=torch.tensor(range_targets, dtype=torch.long),
        next_public_targets=torch.tensor(next_public_targets, dtype=torch.float64),
        steps=tuple(steps),
        fingerprint=_fingerprint(
            {
                "dataset": dataset.fingerprint,
                "split": split,
                "public_normalizer": public_normalizer.fingerprint,
                "private_normalizer": private_normalizer.fingerprint,
                "action_normalizer": action_normalizer.fingerprint,
                "records": payload,
            }
        ),
    )


def _segmented_policy_loss(
    logits: Tensor,
    offsets: Sequence[int],
    targets: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("policy temperature must be positive and finite")
    losses = []
    for index, target in enumerate(targets.tolist()):
        start, end = offsets[index], offsets[index + 1]
        if not start <= target + start < end:
            raise ValueError("segmented policy target is outside legal support")
        log_probabilities = torch.log_softmax(logits[start:end] / temperature, dim=0)
        losses.append(-log_probabilities[target])
    return torch.stack(losses).mean()


def _loss(
    network: LoggedTrajectoryBootstrapNetwork,
    batch: _TrajectoryBatch,
    *,
    policy_weight: float,
    value_weight: float,
    range_weight: float,
    transition_weight: float,
) -> Tensor:
    policy, value, range_logits, next_public, _ = network(
        batch.public_features,
        batch.private_features,
        batch.action_features,
        batch.action_counts,
    )
    policy_loss = _segmented_policy_loss(
        policy,
        batch.action_offsets,
        batch.behavior_targets,
    )
    value_loss = torch.mean((value - batch.outcome_targets) ** 2)
    range_loss = nn.functional.cross_entropy(range_logits, batch.range_targets)
    transition_loss = torch.mean((next_public - batch.next_public_targets) ** 2)
    return (
        policy_weight * policy_loss
        + value_weight * value_loss
        + range_weight * range_loss
        + transition_weight * transition_loss
    )


def _architecture_fingerprint(
    *,
    public_width: int,
    private_width: int,
    action_width: int,
    candidate_count: int,
    hidden_dim: int,
) -> str:
    return _fingerprint(
        {
            "schema": TRAJECTORY_NEURAL_SCHEMA_VERSION,
            "network": "public_private_variable_action_multitask_bootstrap",
            "public_width": public_width,
            "private_width": private_width,
            "action_width": action_width,
            "candidate_count": candidate_count,
            "hidden_dim": hidden_dim,
            "range_head_private_input": False,
            "policy_target": "logged_behavior_action",
            "equilibrium_objective": False,
        }
    )


def _tensor_artifacts(
    network: LoggedTrajectoryBootstrapNetwork,
) -> tuple[TensorArtifact, ...]:
    return tuple(
        TensorArtifact(
            name=name,
            shape=tuple(tensor.shape),
            values=tuple(
                float(value)
                for value in tensor.detach().cpu().to(torch.float64).reshape(-1).tolist()
            ),
        )
        for name, tensor in network.state_dict().items()
    )


def restore_logged_trajectory_bootstrap(
    artifact: TrajectoryBootstrapArtifact,
) -> "LoggedTrajectoryBootstrapProvider":
    network = LoggedTrajectoryBootstrapNetwork(
        public_width=len(artifact.public_normalizer.names),
        private_width=len(artifact.private_normalizer.names),
        action_width=len(artifact.action_normalizer.names),
        candidate_count=len(artifact.candidate_ids),
        hidden_dim=artifact.hidden_dim,
    )
    expected_architecture = _architecture_fingerprint(
        public_width=network.public_width,
        private_width=network.private_width,
        action_width=network.action_width,
        candidate_count=network.candidate_count,
        hidden_dim=network.hidden_dim,
    )
    if expected_architecture != artifact.architecture_fingerprint:
        raise ValueError("trajectory bootstrap architecture fingerprint mismatch")
    expected = network.state_dict()
    supplied = {tensor.name: tensor for tensor in artifact.tensors}
    if set(expected) != set(supplied):
        raise ValueError("trajectory bootstrap tensor names do not match architecture")
    restored = {}
    for name, template in expected.items():
        tensor = supplied[name]
        if tuple(template.shape) != tensor.shape:
            raise ValueError(f"trajectory bootstrap tensor shape mismatch for {name}")
        restored[name] = torch.tensor(tensor.values, dtype=torch.float64).reshape(tensor.shape)
    network.load_state_dict(restored, strict=True)
    return LoggedTrajectoryBootstrapProvider(network, artifact)


class LoggedTrajectoryBootstrapProvider:
    """Restorable logged-data predictor, intentionally not a blueprint provider."""

    def __init__(
        self,
        network: LoggedTrajectoryBootstrapNetwork,
        artifact: TrajectoryBootstrapArtifact,
    ) -> None:
        self.network = network.eval()
        self.artifact = artifact

    @property
    def claim_boundary(self) -> Mapping[str, bool]:
        return {
            "official_engine_execution": False,
            "equilibrium_strategy": False,
            "counterfactual_values": False,
            "safe_resolving": False,
            "behavior_imitation_only": True,
        }

    def predict(
        self,
        *,
        public_features: Sequence[tuple[str, float]],
        private_features: Sequence[tuple[str, float]],
        legal_actions: Sequence[AbstractAction],
    ) -> TrajectoryBootstrapPrediction:
        actions = tuple(legal_actions)
        if not actions:
            raise ValueError("trajectory bootstrap prediction requires legal actions")
        public_values = self.artifact.public_normalizer.transform(public_features)
        private_values = self.artifact.private_normalizer.transform(private_features)
        action_values = tuple(
            self.artifact.action_normalizer.transform(action.features) for action in actions
        )
        with torch.no_grad():
            policy, value, ranges, next_public, _ = self.network(
                torch.tensor([public_values], dtype=torch.float64),
                torch.tensor([private_values], dtype=torch.float64),
                torch.tensor(action_values, dtype=torch.float64),
                torch.tensor([len(actions)], dtype=torch.long),
            )
            action_probabilities = torch.softmax(
                policy / self.artifact.policy_temperature,
                dim=0,
            ).tolist()
            range_probabilities = torch.softmax(
                ranges[0] / self.artifact.range_temperature,
                dim=0,
            ).tolist()
        return TrajectoryBootstrapPrediction(
            action_probabilities=tuple(
                (action.action_id, float(probability))
                for action, probability in zip(actions, action_probabilities, strict=True)
            ),
            outcome_value=float(value[0]),
            opponent_range=tuple(
                (candidate_id, float(probability))
                for candidate_id, probability in zip(
                    self.artifact.candidate_ids,
                    range_probabilities,
                    strict=True,
                )
            ),
            next_public_features=self.artifact.public_normalizer.inverse(
                next_public[0].tolist()
            ),
        )

    def predict_public_range(
        self,
        public_features: Sequence[tuple[str, float]],
    ) -> tuple[tuple[str, float], ...]:
        """Predict only the opponent range from public information.

        This is the permitted integration surface for native belief
        diagnostics.  It executes only ``public_encoder`` and ``range_head``;
        no private features, legal actions, hidden-completion identity, or
        simulator truth are accepted by the method.
        """

        public_values = self.artifact.public_normalizer.transform(public_features)
        with torch.no_grad():
            encoded = self.network.public_encoder(
                torch.tensor([public_values], dtype=torch.float64)
            )
            logits = self.network.range_head(encoded)[0]
            probabilities = torch.softmax(
                logits / self.artifact.range_temperature,
                dim=0,
            ).tolist()
        return tuple(
            (candidate_id, float(probability))
            for candidate_id, probability in zip(
                self.artifact.candidate_ids,
                probabilities,
                strict=True,
            )
        )

    def predict_step(
        self,
        step: LoggedInformationSetStep,
    ) -> TrajectoryBootstrapPrediction:
        return self.predict(
            public_features=step.public_belief.public_features,
            private_features=step.private_features,
            legal_actions=step.information_set.legal_actions,
        )


def _evaluate(
    network: LoggedTrajectoryBootstrapNetwork,
    batch: _TrajectoryBatch,
    *,
    candidate_count: int,
    outcome_constant: float,
    policy_temperature: float,
    range_temperature: float,
) -> TrajectorySplitMetrics:
    with torch.no_grad():
        policy, values, ranges, next_public, _ = network(
            batch.public_features,
            batch.private_features,
            batch.action_features,
            batch.action_counts,
        )
        policy_losses = []
        uniform_policy_losses = []
        policy_correct = 0
        for index, target in enumerate(batch.behavior_targets.tolist()):
            start, end = batch.action_offsets[index], batch.action_offsets[index + 1]
            logits = policy[start:end] / policy_temperature
            log_probabilities = torch.log_softmax(logits, dim=0)
            policy_losses.append(float(-log_probabilities[target]))
            uniform_policy_losses.append(math.log(end - start))
            policy_correct += int(int(torch.argmax(logits)) == target)
        range_log_probabilities = torch.log_softmax(ranges / range_temperature, dim=-1)
        range_nll = float(
            -range_log_probabilities[
                torch.arange(len(batch.range_targets)),
                batch.range_targets,
            ].mean()
        )
        range_correct = int((ranges.argmax(dim=-1) == batch.range_targets).sum())
        value_mse = float(torch.mean((values - batch.outcome_targets) ** 2))
        constant_mse = float(
            torch.mean(
                (torch.full_like(batch.outcome_targets, outcome_constant) - batch.outcome_targets)
                ** 2
            )
        )
        transition_mse = float(
            torch.mean((next_public - batch.next_public_targets) ** 2)
        )
        identity_transition_mse = float(
            torch.mean((batch.public_features - batch.next_public_targets) ** 2)
        )
    return TrajectorySplitMetrics(
        split=batch.split,
        steps=len(batch.steps),
        policy_nll=sum(policy_losses) / len(policy_losses),
        uniform_policy_nll=sum(uniform_policy_losses) / len(uniform_policy_losses),
        policy_top1_accuracy=policy_correct / len(batch.steps),
        outcome_mse=value_mse,
        constant_outcome_mse=constant_mse,
        range_nll=range_nll,
        uniform_range_nll=math.log(candidate_count),
        range_top1_accuracy=range_correct / len(batch.steps),
        transition_mse=transition_mse,
        identity_transition_mse=identity_transition_mse,
    )


def _calibrated_temperatures(
    network: LoggedTrajectoryBootstrapNetwork,
    calibration: _TrajectoryBatch,
) -> tuple[float, float]:
    with torch.no_grad():
        policy, _, ranges, _, _ = network(
            calibration.public_features,
            calibration.private_features,
            calibration.action_features,
            calibration.action_counts,
        )
        policy_scores = []
        range_scores = []
        for temperature in TEMPERATURE_GRID:
            policy_scores.append(
                (
                    float(
                        _segmented_policy_loss(
                            policy,
                            calibration.action_offsets,
                            calibration.behavior_targets,
                            temperature=temperature,
                        )
                    ),
                    temperature,
                )
            )
            range_scores.append(
                (
                    float(
                        nn.functional.cross_entropy(
                            ranges / temperature,
                            calibration.range_targets,
                        )
                    ),
                    temperature,
                )
            )
    return min(policy_scores)[1], min(range_scores)[1]


def train_logged_trajectory_bootstrap(
    dataset: OfficialTrajectoryDataset,
    *,
    seed: int = DEFAULT_TRAJECTORY_NEURAL_SEED,
    epochs: int = 10,
    hidden_dim: int = 32,
    learning_rate: float = 0.01,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 0.25,
    range_loss_weight: float = 0.20,
    transition_loss_weight: float = 0.20,
) -> TrajectoryBootstrapTrainingRun:
    """Train deterministic, split-safe representation/behavior pretraining."""

    if dataset.branchable_transition_game or dataset.provenance.official_engine_executed:
        raise ValueError("trajectory bootstrap expects a derived non-branchable abstraction")
    if epochs <= 0 or hidden_dim <= 0 or learning_rate <= 0:
        raise ValueError("trajectory bootstrap hyperparameters must be positive")
    public_normalizer, private_normalizer, action_normalizer = _fit_normalizers(dataset)
    batches = {
        split: _trajectory_batch(
            dataset,
            split,
            public_normalizer=public_normalizer,
            private_normalizer=private_normalizer,
            action_normalizer=action_normalizer,
        )
        for split in ("train", "calibration", "validation")
    }
    train = batches["train"]
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = LoggedTrajectoryBootstrapNetwork(
        public_width=len(public_normalizer.names),
        private_width=len(private_normalizer.names),
        action_width=len(action_normalizer.names),
        candidate_count=len(dataset.candidates),
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_arguments = {
        "policy_weight": policy_loss_weight,
        "value_weight": value_loss_weight,
        "range_weight": range_loss_weight,
        "transition_weight": transition_loss_weight,
    }
    with torch.no_grad():
        initial_loss = float(_loss(network, train, **loss_arguments))
    gradient_names: tuple[str, ...] = ()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for epoch in range(epochs):
            loss = _loss(network, train, **loss_arguments)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 0:
                gradient_names = tuple(
                    name
                    for name, parameter in network.named_parameters()
                    if parameter.grad is not None
                    and float(parameter.grad.detach().abs().sum()) > 0
                )
            optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)
    with torch.no_grad():
        final_loss = float(_loss(network, train, **loss_arguments))
    policy_temperature, range_temperature = _calibrated_temperatures(
        network,
        batches["calibration"],
    )
    candidate_ids = tuple(candidate.candidate_id for candidate in dataset.candidates)
    architecture_fingerprint = _architecture_fingerprint(
        public_width=len(public_normalizer.names),
        private_width=len(private_normalizer.names),
        action_width=len(action_normalizer.names),
        candidate_count=len(candidate_ids),
        hidden_dim=hidden_dim,
    )
    artifact = TrajectoryBootstrapArtifact(
        dataset_fingerprint=dataset.fingerprint,
        source_public_replay_fingerprint=dataset.source_public_replay_fingerprint,
        architecture_fingerprint=architecture_fingerprint,
        candidate_ids=candidate_ids,
        public_normalizer=public_normalizer,
        private_normalizer=private_normalizer,
        action_normalizer=action_normalizer,
        hidden_dim=hidden_dim,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
        range_loss_weight=range_loss_weight,
        transition_loss_weight=transition_loss_weight,
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        policy_temperature=policy_temperature,
        range_temperature=range_temperature,
        train_match_ids=_split_match_ids(dataset, "train"),
        calibration_match_ids=_split_match_ids(dataset, "calibration"),
        validation_match_ids=_split_match_ids(dataset, "validation"),
        train_batch_fingerprint=batches["train"].fingerprint,
        calibration_batch_fingerprint=batches["calibration"].fingerprint,
        validation_batch_fingerprint=batches["validation"].fingerprint,
        source_official_engine_executed=dataset.source_official_engine_executed,
        model_official_engine_executed=False,
        equilibrium_trained=False,
        deployable_blueprint=False,
        counterfactual_value_provider=False,
        tensors=_tensor_artifacts(network),
    )
    outcome_constant = float(train.outcome_targets.mean())
    metrics = tuple(
        _evaluate(
            network,
            batches[split],
            candidate_count=len(candidate_ids),
            outcome_constant=outcome_constant,
            policy_temperature=policy_temperature,
            range_temperature=range_temperature,
        )
        for split in ("train", "calibration", "validation")
    )
    provider = LoggedTrajectoryBootstrapProvider(network, artifact)
    report = TrajectoryBootstrapTrainingReport(
        artifact_fingerprint=artifact.fingerprint,
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        gradient_parameter_names=gradient_names,
        policy_temperature=policy_temperature,
        range_temperature=range_temperature,
        metrics=metrics,
    )
    return TrajectoryBootstrapTrainingRun(
        provider=provider,
        artifact=artifact,
        report=report,
    )
