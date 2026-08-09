"""Neural regret-matching candidate over branchable native-search states.

This module is intentionally narrower than a claim of Deep CFR, Nash
convergence, or safe deployment.  It consumes terminal-valued action branches
from :mod:`rocketbot.official_search_game`, fits cumulative-regret and policy
targets, and exposes a restorable full-game blueprint *candidate*.  The source
descendant transitions may be genuine licensed-engine executions while the
dataset, hidden-completion prior, model, and resulting strategy remain research
abstractions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .game_theory import (
    AbstractAction,
    ActionDistribution,
    BlueprintCheckpoint,
    CounterfactualValues,
    InformationSet,
    InformationSetActor,
    OpponentRange,
    PublicBeliefState,
    RangeHypothesis,
)
from .native_blueprint_features import (
    NATIVE_ACTION_FEATURE_NAMES,
    NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT,
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
)
from .neural_blueprint import TensorArtifact
from .trajectory_neural import (
    LoggedTrajectoryBootstrapProvider,
    NamedFeatureNormalizer,
    TrajectoryBootstrapArtifact,
    restore_logged_trajectory_bootstrap,
)


NATIVE_REGRET_BLUEPRINT_SCHEMA_VERSION = 1
DEFAULT_NATIVE_REGRET_SEED = 20260727


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


def _distribution(
    items: Sequence[tuple[str, float]],
    *,
    identifiers: Sequence[str],
    field_name: str,
) -> tuple[tuple[str, float], ...]:
    materialized = tuple((str(name), _finite(value, field_name)) for name, value in items)
    if tuple(name for name, _ in materialized) != tuple(identifiers):
        raise ValueError(f"{field_name} support/order mismatch")
    if any(value < 0 for _, value in materialized):
        raise ValueError(f"{field_name} cannot contain negative mass")
    if not math.isclose(sum(value for _, value in materialized), 1.0, abs_tol=1e-9):
        raise ValueError(f"{field_name} must sum to one")
    return materialized


def _action_to_dict(action: AbstractAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "label": action.label,
        "official_option_indices": list(action.official_option_indices),
        "features": dict(action.features),
    }


def _action_from_dict(raw: Mapping[str, Any]) -> AbstractAction:
    features = raw.get("features", {})
    feature_items = features.items() if isinstance(features, Mapping) else features
    return AbstractAction(
        action_id=str(raw["action_id"]),
        label=str(raw["label"]),
        official_option_indices=tuple(int(value) for value in raw["official_option_indices"]),
        features=tuple((str(name), float(value)) for name, value in feature_items),
    )


@dataclass(frozen=True, slots=True)
class NativeRegretTrainingSample:
    """One information set with terminal-valued legal-action branches."""

    round_index: int
    source_candidate_id_evaluation_only: str
    player: int
    depth: int
    split: str
    information_set_id: str
    public_belief_fingerprint: str
    private_observation_fingerprint: str
    public_features: tuple[tuple[str, float], ...]
    private_features: tuple[tuple[str, float], ...]
    actions: tuple[AbstractAction, ...]
    behavior_strategy: tuple[tuple[str, float], ...]
    action_values: tuple[tuple[str, float], ...]
    instantaneous_regrets: tuple[tuple[str, float], ...]
    cumulative_regrets: tuple[tuple[str, float], ...]
    target_strategy: tuple[tuple[str, float], ...]
    baseline_value: float
    rollouts_per_action: int
    native_transition_count: int
    source_official_engine_executed: bool = True
    selected_hidden_truth_in_policy_features: bool = False

    def __post_init__(self) -> None:
        if self.round_index <= 0 or self.depth <= 0:
            raise ValueError("native regret sample round/depth must be positive")
        if self.player not in (0, 1):
            raise ValueError("native regret sample player must be 0 or 1")
        if self.split not in {"train", "calibration", "validation"}:
            raise ValueError("unknown native regret sample split")
        if not self.source_candidate_id_evaluation_only or not self.information_set_id:
            raise ValueError("native regret sample identifiers must be non-empty")
        _validate_digest(self.public_belief_fingerprint, "public_belief_fingerprint")
        _validate_digest(
            self.private_observation_fingerprint,
            "private_observation_fingerprint",
        )
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("native regret public feature schema mismatch")
        if tuple(name for name, _ in self.private_features) != NATIVE_PRIVATE_FEATURE_NAMES:
            raise ValueError("native regret private feature schema mismatch")
        if not self.actions:
            raise ValueError("native regret sample requires legal actions")
        identifiers = tuple(action.action_id for action in self.actions)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("native regret sample action IDs must be unique")
        if any(
            tuple(name for name, _ in action.features) != NATIVE_ACTION_FEATURE_NAMES
            for action in self.actions
        ):
            raise ValueError("native regret action feature schema mismatch")
        _distribution(
            self.behavior_strategy,
            identifiers=identifiers,
            field_name="behavior_strategy",
        )
        _distribution(
            self.target_strategy,
            identifiers=identifiers,
            field_name="target_strategy",
        )
        for field_name in (
            "action_values",
            "instantaneous_regrets",
            "cumulative_regrets",
        ):
            items = getattr(self, field_name)
            if tuple(name for name, _ in items) != identifiers:
                raise ValueError(f"{field_name} support/order mismatch")
            for _, value in items:
                _finite(value, field_name)
        if any(not -1.0 <= value <= 1.0 for _, value in self.action_values):
            raise ValueError("terminal action values must be in [-1, 1]")
        if not -1.0 <= _finite(self.baseline_value, "baseline_value") <= 1.0:
            raise ValueError("native regret baseline value must be in [-1, 1]")
        if self.rollouts_per_action <= 0 or self.native_transition_count <= 0:
            raise ValueError("native regret rollout accounting must be positive")
        if not self.source_official_engine_executed:
            raise ValueError("native regret source branches must be official execution")
        if self.selected_hidden_truth_in_policy_features:
            raise ValueError("selected hidden truth cannot enter policy features")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "round_index": self.round_index,
            "source_candidate_id_evaluation_only": (
                self.source_candidate_id_evaluation_only
            ),
            "player": self.player,
            "depth": self.depth,
            "split": self.split,
            "information_set_id": self.information_set_id,
            "public_belief_fingerprint": self.public_belief_fingerprint,
            "private_observation_fingerprint": self.private_observation_fingerprint,
            "public_features": dict(self.public_features),
            "private_features": dict(self.private_features),
            "actions": [_action_to_dict(action) for action in self.actions],
            "behavior_strategy": [list(item) for item in self.behavior_strategy],
            "action_values": [list(item) for item in self.action_values],
            "instantaneous_regrets": [list(item) for item in self.instantaneous_regrets],
            "cumulative_regrets": [list(item) for item in self.cumulative_regrets],
            "target_strategy": [list(item) for item in self.target_strategy],
            "baseline_value": self.baseline_value,
            "rollouts_per_action": self.rollouts_per_action,
            "native_transition_count": self.native_transition_count,
            "provenance": {
                "source_official_engine_executed": (
                    self.source_official_engine_executed
                ),
                "selected_hidden_truth_in_policy_features": (
                    self.selected_hidden_truth_in_policy_features
                ),
                "raw_private_card_ids_persisted": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeRegretTrainingSample":
        def items(name: str) -> tuple[tuple[str, float], ...]:
            value = raw[name]
            source = value.items() if isinstance(value, Mapping) else value
            return tuple((str(key), float(item)) for key, item in source)

        provenance = raw["provenance"]
        result = cls(
            round_index=int(raw["round_index"]),
            source_candidate_id_evaluation_only=str(
                raw["source_candidate_id_evaluation_only"]
            ),
            player=int(raw["player"]),
            depth=int(raw["depth"]),
            split=str(raw["split"]),
            information_set_id=str(raw["information_set_id"]),
            public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
            private_observation_fingerprint=str(
                raw["private_observation_fingerprint"]
            ),
            public_features=items("public_features"),
            private_features=items("private_features"),
            actions=tuple(_action_from_dict(item) for item in raw["actions"]),
            behavior_strategy=items("behavior_strategy"),
            action_values=items("action_values"),
            instantaneous_regrets=items("instantaneous_regrets"),
            cumulative_regrets=items("cumulative_regrets"),
            target_strategy=items("target_strategy"),
            baseline_value=float(raw["baseline_value"]),
            rollouts_per_action=int(raw["rollouts_per_action"]),
            native_transition_count=int(raw["native_transition_count"]),
            source_official_engine_executed=bool(
                provenance["source_official_engine_executed"]
            ),
            selected_hidden_truth_in_policy_features=bool(
                provenance["selected_hidden_truth_in_policy_features"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native regret sample fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativeRegretDataset:
    game_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    source_bootstrap_fingerprint: str
    feature_schema_fingerprint: str
    sampling_seed: int
    training_rounds: int
    samples: tuple[NativeRegretTrainingSample, ...]
    source_official_engine_executed: bool = True
    dataset_official_engine_executed: bool = False
    hidden_completion_prior_official: bool = False
    equilibrium_convergence_validated: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "game_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_bootstrap_fingerprint",
            "feature_schema_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.feature_schema_fingerprint != NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT:
            raise ValueError("native regret dataset feature schema mismatch")
        if self.sampling_seed < 0 or self.training_rounds <= 0 or not self.samples:
            raise ValueError("native regret dataset sampling metadata is invalid")
        if max(sample.round_index for sample in self.samples) > self.training_rounds:
            raise ValueError("native regret sample exceeds declared training rounds")
        if not self.source_official_engine_executed:
            raise ValueError("native regret dataset requires genuine source transitions")
        if any(
            (
                self.dataset_official_engine_executed,
                self.hidden_completion_prior_official,
                self.equilibrium_convergence_validated,
            )
        ):
            raise ValueError("native regret abstraction cannot promote official/equilibrium claims")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("native regret dataset cannot persist raw private card IDs")
        by_information_set: dict[str, str] = {}
        for sample in self.samples:
            previous = by_information_set.setdefault(sample.information_set_id, sample.split)
            if previous != sample.split:
                raise ValueError("information set crosses native regret data splits")
        if {sample.split for sample in self.samples} != {
            "train",
            "calibration",
            "validation",
        }:
            raise ValueError("native regret dataset must contain all three splits")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": NATIVE_REGRET_BLUEPRINT_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_bootstrap_fingerprint": self.source_bootstrap_fingerprint,
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "sampling_seed": self.sampling_seed,
            "training_rounds": self.training_rounds,
            "samples": [sample.to_dict() for sample in self.samples],
            "claims": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "dataset_official_engine_executed": self.dataset_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "equilibrium_convergence_validated": (
                    self.equilibrium_convergence_validated
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
                "selected_hidden_truth_in_policy_features": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeRegretDataset":
        claims = raw["claims"]
        result = cls(
            game_fingerprint=str(raw["game_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_bootstrap_fingerprint=str(raw["source_bootstrap_fingerprint"]),
            feature_schema_fingerprint=str(raw["feature_schema_fingerprint"]),
            sampling_seed=int(raw["sampling_seed"]),
            training_rounds=int(raw["training_rounds"]),
            samples=tuple(
                NativeRegretTrainingSample.from_dict(item) for item in raw["samples"]
            ),
            source_official_engine_executed=bool(
                claims["source_official_engine_executed"]
            ),
            dataset_official_engine_executed=bool(
                claims["dataset_official_engine_executed"]
            ),
            hidden_completion_prior_official=bool(
                claims["hidden_completion_prior_official"]
            ),
            equilibrium_convergence_validated=bool(
                claims["equilibrium_convergence_validated"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native regret dataset fingerprint mismatch")
        return result


class NativeRegretBlueprintNetwork(nn.Module):
    """Shared semantic encoders with separate regret and average-policy heads."""

    def __init__(
        self,
        *,
        public_width: int,
        private_width: int,
        action_width: int,
        candidate_count: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(public_width, private_width, action_width, candidate_count, hidden_dim) <= 0:
            raise ValueError("native regret network widths must be positive")
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
        self.average_policy_query = nn.Linear(hidden_dim, hidden_dim)
        self.average_policy_action_bias = nn.Linear(hidden_dim, 1)
        self.advantage_query = nn.Linear(hidden_dim, hidden_dim)
        self.advantage_action_bias = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.range_head = nn.Linear(hidden_dim, candidate_count)
        self.double()

    def forward(
        self,
        public_features: Tensor,
        private_features: Tensor,
        action_features: Tensor,
        action_counts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if public_features.ndim != 2 or public_features.shape[1] != self.public_width:
            raise ValueError("native regret public tensor shape mismatch")
        if private_features.ndim != 2 or private_features.shape[1] != self.private_width:
            raise ValueError("native regret private tensor shape mismatch")
        if action_features.ndim != 2 or action_features.shape[1] != self.action_width:
            raise ValueError("native regret action tensor shape mismatch")
        if action_counts.ndim != 1 or len(action_counts) != len(public_features):
            raise ValueError("native regret action-count shape mismatch")
        if int(action_counts.sum()) != len(action_features) or bool((action_counts <= 0).any()):
            raise ValueError("native regret action counts do not partition actions")
        public_hidden = self.public_encoder(public_features)
        private_hidden = self.private_encoder(private_features)
        state_hidden = self.state_encoder(torch.cat((public_hidden, private_hidden), dim=-1))
        action_hidden = self.action_encoder(action_features)
        state_indices = torch.repeat_interleave(
            torch.arange(len(public_features), device=public_features.device),
            action_counts,
        )
        average_queries = self.average_policy_query(state_hidden)[state_indices]
        advantage_queries = self.advantage_query(state_hidden)[state_indices]
        scale = math.sqrt(self.hidden_dim)
        policy_logits = (
            (average_queries * action_hidden).sum(dim=-1) / scale
            + self.average_policy_action_bias(action_hidden).squeeze(-1)
        )
        advantages = 2.0 * torch.tanh(
            (advantage_queries * action_hidden).sum(dim=-1) / scale
            + self.advantage_action_bias(action_hidden).squeeze(-1)
        )
        state_values = torch.tanh(self.value_head(state_hidden).squeeze(-1))
        range_logits = self.range_head(public_hidden)
        return policy_logits, advantages, state_values, range_logits


def initialized_native_regret_network(
    bootstrap: LoggedTrajectoryBootstrapProvider,
) -> tuple[NativeRegretBlueprintNetwork, tuple[str, ...]]:
    """Copy every compatible Phase 6I encoder/head into the new candidate."""

    source = bootstrap.network
    network = NativeRegretBlueprintNetwork(
        public_width=source.public_width,
        private_width=source.private_width,
        action_width=source.action_width,
        candidate_count=source.candidate_count,
        hidden_dim=source.hidden_dim,
    )
    copied: list[str] = []

    def copy_module(target_name: str, source_name: str) -> None:
        target = getattr(network, target_name)
        source_module = getattr(source, source_name)
        target.load_state_dict(source_module.state_dict(), strict=True)
        copied.extend(f"{target_name}.{name}" for name in target.state_dict())

    for target_name, source_name in (
        ("public_encoder", "public_encoder"),
        ("private_encoder", "private_encoder"),
        ("state_encoder", "state_encoder"),
        ("action_encoder", "action_encoder"),
        ("average_policy_query", "policy_query"),
        ("average_policy_action_bias", "policy_action_bias"),
        ("advantage_query", "policy_query"),
        ("advantage_action_bias", "policy_action_bias"),
        ("value_head", "value_head"),
        ("range_head", "range_head"),
    ):
        copy_module(target_name, source_name)
    return network, tuple(copied)


@dataclass(frozen=True, slots=True)
class NativeRegretBlueprintArtifact:
    game_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    dataset_fingerprint: str
    source_bootstrap_fingerprint: str
    feature_schema_fingerprint: str
    architecture_fingerprint: str
    public_normalizer: NamedFeatureNormalizer
    private_normalizer: NamedFeatureNormalizer
    action_normalizer: NamedFeatureNormalizer
    candidate_ids: tuple[str, ...]
    hidden_dim: int
    seed: int
    training_rounds: int
    epochs: int
    learning_rate: float
    initial_training_loss: float
    final_training_loss: float
    gradient_parameter_names: tuple[str, ...]
    initialized_parameter_names: tuple[str, ...]
    tensors: tuple[TensorArtifact, ...]
    source_official_engine_executed: bool = True
    model_official_engine_executed: bool = False
    equilibrium_objective_trained: bool = True
    equilibrium_convergence_validated: bool = False
    deployable_blueprint: bool = False
    calibrated_counterfactual_values: bool = False
    safe_resolver_compatible: bool = False
    opponent_exploitation_enabled: bool = False
    exact_deck: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "game_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "dataset_fingerprint",
            "source_bootstrap_fingerprint",
            "feature_schema_fingerprint",
            "architecture_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.feature_schema_fingerprint != NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT:
            raise ValueError("native regret artifact feature schema mismatch")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("native regret candidate IDs must be unique and non-empty")
        if min(self.hidden_dim, self.training_rounds, self.epochs) <= 0:
            raise ValueError("native regret training dimensions must be positive")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("native regret learning rate must be positive and finite")
        for field_name in ("initial_training_loss", "final_training_loss"):
            if _finite(getattr(self, field_name), field_name) < 0:
                raise ValueError("native regret losses cannot be negative")
        if not self.gradient_parameter_names or not self.initialized_parameter_names:
            raise ValueError("native regret training/initialization evidence is required")
        tensor_names = [tensor.name for tensor in self.tensors]
        if not tensor_names or len(tensor_names) != len(set(tensor_names)):
            raise ValueError("native regret tensors must be unique and non-empty")
        if not self.source_official_engine_executed:
            raise ValueError("native regret artifact requires genuine source transitions")
        if any(
            (
                self.model_official_engine_executed,
                self.equilibrium_convergence_validated,
                self.deployable_blueprint,
                self.calibrated_counterfactual_values,
                self.safe_resolver_compatible,
                self.opponent_exploitation_enabled,
                self.exact_deck,
            )
        ):
            raise ValueError("native regret candidate cannot promote unvalidated capabilities")
        if not self.equilibrium_objective_trained:
            raise ValueError("native regret artifact must record its regret objective")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": NATIVE_REGRET_BLUEPRINT_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_bootstrap_fingerprint": self.source_bootstrap_fingerprint,
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "architecture_fingerprint": self.architecture_fingerprint,
            "normalizers": {
                "public": self.public_normalizer.to_dict(),
                "private": self.private_normalizer.to_dict(),
                "action": self.action_normalizer.to_dict(),
            },
            "candidate_ids": list(self.candidate_ids),
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "training_rounds": self.training_rounds,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "initial_training_loss": self.initial_training_loss,
            "final_training_loss": self.final_training_loss,
            "gradient_parameter_names": list(self.gradient_parameter_names),
            "initialized_parameter_names": list(self.initialized_parameter_names),
            "claims": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "model_official_engine_executed": self.model_official_engine_executed,
                "equilibrium_objective_trained": self.equilibrium_objective_trained,
                "equilibrium_convergence_validated": (
                    self.equilibrium_convergence_validated
                ),
                "deployable_blueprint": self.deployable_blueprint,
                "calibrated_counterfactual_values": (
                    self.calibrated_counterfactual_values
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "opponent_exploitation_enabled": self.opponent_exploitation_enabled,
                "exact_deck": self.exact_deck,
            },
            "tensors": [asdict(tensor) for tensor in self.tensors],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeRegretBlueprintArtifact":
        normalizers = raw["normalizers"]
        claims = raw["claims"]
        result = cls(
            game_fingerprint=str(raw["game_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            source_bootstrap_fingerprint=str(raw["source_bootstrap_fingerprint"]),
            feature_schema_fingerprint=str(raw["feature_schema_fingerprint"]),
            architecture_fingerprint=str(raw["architecture_fingerprint"]),
            public_normalizer=NamedFeatureNormalizer.from_dict(normalizers["public"]),
            private_normalizer=NamedFeatureNormalizer.from_dict(normalizers["private"]),
            action_normalizer=NamedFeatureNormalizer.from_dict(normalizers["action"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            hidden_dim=int(raw["hidden_dim"]),
            seed=int(raw["seed"]),
            training_rounds=int(raw["training_rounds"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            initial_training_loss=float(raw["initial_training_loss"]),
            final_training_loss=float(raw["final_training_loss"]),
            gradient_parameter_names=tuple(
                str(value) for value in raw["gradient_parameter_names"]
            ),
            initialized_parameter_names=tuple(
                str(value) for value in raw["initialized_parameter_names"]
            ),
            tensors=tuple(
                TensorArtifact(
                    name=str(item["name"]),
                    shape=tuple(int(value) for value in item["shape"]),
                    values=tuple(float(value) for value in item["values"]),
                )
                for item in raw["tensors"]
            ),
            source_official_engine_executed=bool(
                claims["source_official_engine_executed"]
            ),
            model_official_engine_executed=bool(
                claims["model_official_engine_executed"]
            ),
            equilibrium_objective_trained=bool(
                claims["equilibrium_objective_trained"]
            ),
            equilibrium_convergence_validated=bool(
                claims["equilibrium_convergence_validated"]
            ),
            deployable_blueprint=bool(claims["deployable_blueprint"]),
            calibrated_counterfactual_values=bool(
                claims["calibrated_counterfactual_values"]
            ),
            safe_resolver_compatible=bool(claims["safe_resolver_compatible"]),
            opponent_exploitation_enabled=bool(
                claims["opponent_exploitation_enabled"]
            ),
            exact_deck=bool(claims["exact_deck"]),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native regret blueprint fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativeRegretPrediction:
    strategy: ActionDistribution
    advantages: tuple[tuple[str, float], ...]
    state_value: float
    opponent_range: tuple[tuple[str, float], ...]


class NativeRegretBlueprintProvider:
    """Restorable full-game candidate; explicitly unsafe for deployment."""

    def __init__(
        self,
        network: NativeRegretBlueprintNetwork,
        artifact: NativeRegretBlueprintArtifact,
    ) -> None:
        self.network = network.eval()
        self.artifact = artifact
        self._checkpoint = BlueprintCheckpoint(
            provider_name="native_sampled_regret_blueprint_candidate",
            algorithm=(
                "terminal_native_counterfactual_branch_regret_matching_with_"
                "phase6i_encoder_initialization"
            ),
            model_fingerprint=artifact.fingerprint,
            game_fingerprint=artifact.game_fingerprint,
            deck_fingerprint=artifact.deck_fingerprint,
            training_steps=artifact.training_rounds * artifact.epochs,
            trained=True,
            evaluation_notes=(
                "Trained on terminal-valued native branches under a supplied hidden "
                "prior. Equilibrium convergence, counterfactual-value calibration, "
                "safety, official strategy control, and strength are not validated."
            ),
        )

    @property
    def checkpoint(self) -> BlueprintCheckpoint:
        return self._checkpoint

    @property
    def claim_boundary(self) -> Mapping[str, bool]:
        return {
            "source_official_engine_executed": True,
            "model_official_engine_executed": False,
            "equilibrium_objective_trained": True,
            "equilibrium_convergence_validated": False,
            "deployable_blueprint": False,
            "calibrated_counterfactual_values": False,
            "safe_resolving": False,
        }

    def predict(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> NativeRegretPrediction:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("native regret information set/belief mismatch")
        actions = information_set.legal_actions
        if not actions:
            raise ValueError("native regret prediction requires legal actions")
        public = self.artifact.public_normalizer.transform(belief.public_features)
        private = self.artifact.private_normalizer.transform(
            information_set.private_features
        )
        action_rows = tuple(
            self.artifact.action_normalizer.transform(action.features)
            for action in actions
        )
        with torch.no_grad():
            logits, advantages, state_values, range_logits = self.network(
                torch.tensor([public], dtype=torch.float64),
                torch.tensor([private], dtype=torch.float64),
                torch.tensor(action_rows, dtype=torch.float64),
                torch.tensor([len(actions)], dtype=torch.long),
            )
            probabilities = torch.softmax(logits, dim=0).tolist()
            ranges = torch.softmax(range_logits[0], dim=0).tolist()
        strategy = ActionDistribution.normalized(
            {
                action.action_id: float(probability)
                for action, probability in zip(actions, probabilities, strict=True)
            }
        )
        strategy.validate_support(information_set)
        return NativeRegretPrediction(
            strategy=strategy,
            advantages=tuple(
                (action.action_id, float(value))
                for action, value in zip(actions, advantages.tolist(), strict=True)
            ),
            state_value=float(state_values[0]),
            opponent_range=tuple(
                (candidate_id, float(probability))
                for candidate_id, probability in zip(
                    self.artifact.candidate_ids,
                    ranges,
                    strict=True,
                )
            ),
        )

    def strategy(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        return self.predict(information_set, belief).strategy

    def counterfactual_values(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        player: int,
    ) -> CounterfactualValues:
        del information_set, belief, player
        raise RuntimeError(
            "Phase 6K has no calibrated per-hypothesis counterfactual values; "
            "its scalar rollout value cannot be used for safe resolving"
        )


@dataclass(frozen=True, slots=True)
class NativeRegretSplitMetrics:
    split: str
    sample_count: int
    information_set_count: int
    policy_cross_entropy: float
    advantage_rmse: float
    value_rmse: float
    mean_candidate_deviation_incentive: float
    mean_behavior_deviation_incentive: float
    mean_target_deviation_incentive: float
    exact_legal_support_rate: float

    def __post_init__(self) -> None:
        if self.split not in {"train", "calibration", "validation"}:
            raise ValueError("unknown native regret metric split")
        if self.sample_count <= 0 or self.information_set_count <= 0:
            raise ValueError("native regret split metrics require samples")
        for field_name in (
            "policy_cross_entropy",
            "advantage_rmse",
            "value_rmse",
            "mean_candidate_deviation_incentive",
            "mean_behavior_deviation_incentive",
            "mean_target_deviation_incentive",
        ):
            if _finite(getattr(self, field_name), field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if not 0 <= self.exact_legal_support_rate <= 1:
            raise ValueError("legal support rate must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class NativeRegretTrainingReport:
    initial_training_loss: float
    final_training_loss: float
    gradient_parameter_names: tuple[str, ...]
    initialized_parameter_names: tuple[str, ...]
    split_metrics: tuple[NativeRegretSplitMetrics, ...]
    strategy_changed_from_behavior_samples: int
    artifact_fingerprint: str

    def __post_init__(self) -> None:
        if min(self.initial_training_loss, self.final_training_loss) < 0:
            raise ValueError("native regret report losses cannot be negative")
        if not self.gradient_parameter_names or not self.initialized_parameter_names:
            raise ValueError("native regret report needs gradient/initialization evidence")
        if tuple(item.split for item in self.split_metrics) != (
            "train",
            "calibration",
            "validation",
        ):
            raise ValueError("native regret metrics must cover ordered splits")
        if self.strategy_changed_from_behavior_samples < 0:
            raise ValueError("strategy change count cannot be negative")
        _validate_digest(self.artifact_fingerprint, "artifact_fingerprint")


@dataclass(slots=True)
class NativeRegretTrainingRun:
    provider: NativeRegretBlueprintProvider
    artifact: NativeRegretBlueprintArtifact
    report: NativeRegretTrainingReport


def native_regret_sample_view(
    sample: NativeRegretTrainingSample,
) -> tuple[InformationSet, PublicBeliefState]:
    """Reconstruct the exact persisted neural inputs without simulator truth."""

    belief = PublicBeliefState(
        public_state_fingerprint=_fingerprint(
            {"sample": sample.fingerprint, "kind": "metric_public_state"}
        ),
        public_history_fingerprint=_fingerprint(
            {"sample": sample.fingerprint, "kind": "metric_public_history"}
        ),
        turn=max(0, int(dict(sample.public_features).get("turn", 0.0))),
        acting_player=InformationSetActor.SELF,
        opponent_range=OpponentRange.normalized(
            (
                RangeHypothesis(
                    hypothesis_id="metric_range",
                    weight=1.0,
                    hidden_resource_counts=(),
                    intent_id="unknown",
                    style_id="unknown",
                    source="persisted-input reconstruction only",
                ),
            )
        ),
        public_features=sample.public_features,
        updater_fingerprint=_fingerprint(
            {"sample": sample.fingerprint, "kind": "metric_updater"}
        ),
    )
    information_set = InformationSet(
        info_set_id=sample.information_set_id,
        public_belief_fingerprint=belief.fingerprint,
        actor=InformationSetActor.SELF,
        legal_actions=sample.actions,
        depth=sample.depth,
        private_observation_fingerprint=sample.private_observation_fingerprint,
        private_features=sample.private_features,
    )
    return information_set, belief


@dataclass(frozen=True, slots=True)
class _NativeRegretBatch:
    split: str
    public_features: Tensor
    private_features: Tensor
    action_features: Tensor
    action_counts: Tensor
    action_offsets: tuple[int, ...]
    policy_targets: Tensor
    advantage_targets: Tensor
    value_targets: Tensor
    samples: tuple[NativeRegretTrainingSample, ...]


def _batch(
    dataset: NativeRegretDataset,
    split: str,
    *,
    public_normalizer: NamedFeatureNormalizer,
    private_normalizer: NamedFeatureNormalizer,
    action_normalizer: NamedFeatureNormalizer,
) -> _NativeRegretBatch:
    samples = tuple(sample for sample in dataset.samples if sample.split == split)
    if not samples:
        raise ValueError(f"native regret split {split} is empty")
    public_rows = []
    private_rows = []
    action_rows = []
    policy_targets = []
    advantage_targets = []
    action_counts = []
    value_targets = []
    offsets = [0]
    for sample in samples:
        public_rows.append(public_normalizer.transform(sample.public_features))
        private_rows.append(private_normalizer.transform(sample.private_features))
        action_rows.extend(
            action_normalizer.transform(action.features) for action in sample.actions
        )
        policy_targets.extend(value for _, value in sample.target_strategy)
        regrets = tuple(value for _, value in sample.cumulative_regrets)
        scale = max(1.0, max(abs(value) for value in regrets))
        advantage_targets.extend(value / scale for value in regrets)
        action_counts.append(len(sample.actions))
        offsets.append(offsets[-1] + len(sample.actions))
        value_targets.append(sample.baseline_value)
    return _NativeRegretBatch(
        split=split,
        public_features=torch.tensor(public_rows, dtype=torch.float64),
        private_features=torch.tensor(private_rows, dtype=torch.float64),
        action_features=torch.tensor(action_rows, dtype=torch.float64),
        action_counts=torch.tensor(action_counts, dtype=torch.long),
        action_offsets=tuple(offsets),
        policy_targets=torch.tensor(policy_targets, dtype=torch.float64),
        advantage_targets=torch.tensor(advantage_targets, dtype=torch.float64),
        value_targets=torch.tensor(value_targets, dtype=torch.float64),
        samples=samples,
    )


def _loss(network: NativeRegretBlueprintNetwork, batch: _NativeRegretBatch) -> Tensor:
    logits, advantages, values, _ = network(
        batch.public_features,
        batch.private_features,
        batch.action_features,
        batch.action_counts,
    )
    policy_losses = []
    for start, end in zip(batch.action_offsets[:-1], batch.action_offsets[1:], strict=True):
        policy_losses.append(
            -(batch.policy_targets[start:end] * torch.log_softmax(logits[start:end], dim=0)).sum()
        )
    policy_loss = torch.stack(policy_losses).mean()
    advantage_loss = torch.mean((advantages - batch.advantage_targets) ** 2)
    value_loss = torch.mean((values - batch.value_targets) ** 2)
    consistency = []
    for start, end in zip(batch.action_offsets[:-1], batch.action_offsets[1:], strict=True):
        probabilities = torch.softmax(logits[start:end], dim=0)
        consistency.append(torch.sum(probabilities * advantages[start:end]) ** 2)
    return policy_loss + 0.50 * advantage_loss + 0.25 * value_loss + 0.05 * torch.stack(consistency).mean()


def _tensor_artifacts(network: NativeRegretBlueprintNetwork) -> tuple[TensorArtifact, ...]:
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


def _architecture_fingerprint(network: NativeRegretBlueprintNetwork) -> str:
    return _fingerprint(
        {
            "schema": NATIVE_REGRET_BLUEPRINT_SCHEMA_VERSION,
            "public_width": network.public_width,
            "private_width": network.private_width,
            "action_width": network.action_width,
            "candidate_count": network.candidate_count,
            "hidden_dim": network.hidden_dim,
            "heads": ("average_policy", "cumulative_regret", "state_value", "range"),
            "feature_schema": NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT,
        }
    )


def restore_native_regret_blueprint(
    artifact: NativeRegretBlueprintArtifact,
) -> NativeRegretBlueprintProvider:
    network = NativeRegretBlueprintNetwork(
        public_width=len(artifact.public_normalizer.names),
        private_width=len(artifact.private_normalizer.names),
        action_width=len(artifact.action_normalizer.names),
        candidate_count=len(artifact.candidate_ids),
        hidden_dim=artifact.hidden_dim,
    )
    if _architecture_fingerprint(network) != artifact.architecture_fingerprint:
        raise ValueError("native regret architecture fingerprint mismatch")
    templates = network.state_dict()
    supplied = {tensor.name: tensor for tensor in artifact.tensors}
    if set(templates) != set(supplied):
        raise ValueError("native regret tensor names do not match architecture")
    state = {}
    for name, template in templates.items():
        tensor = supplied[name]
        if tuple(template.shape) != tensor.shape:
            raise ValueError(f"native regret tensor shape mismatch for {name}")
        state[name] = torch.tensor(tensor.values, dtype=torch.float64).reshape(tensor.shape)
    network.load_state_dict(state, strict=True)
    return NativeRegretBlueprintProvider(network, artifact)


def _split_metrics(
    provider: NativeRegretBlueprintProvider,
    dataset: NativeRegretDataset,
    split: str,
) -> NativeRegretSplitMetrics:
    samples = tuple(sample for sample in dataset.samples if sample.split == split)
    policy_loss = 0.0
    advantage_squared = []
    value_squared = []
    candidate_incentives = []
    behavior_incentives = []
    target_incentives = []
    exact_support = 0
    for sample in samples:
        information_set, belief = native_regret_sample_view(sample)
        prediction = provider.predict(information_set, belief)
        predicted = prediction.strategy.as_dict()
        targets = dict(sample.target_strategy)
        behaviors = dict(sample.behavior_strategy)
        values = dict(sample.action_values)
        regrets = dict(sample.cumulative_regrets)
        scale = max(1.0, max(abs(value) for value in regrets.values()))
        predicted_advantages = dict(prediction.advantages)
        policy_loss += -sum(
            probability * math.log(max(1e-15, predicted[action_id]))
            for action_id, probability in targets.items()
        )
        advantage_squared.extend(
            (predicted_advantages[action_id] - regrets[action_id] / scale) ** 2
            for action_id in values
        )
        value_squared.append((prediction.state_value - sample.baseline_value) ** 2)
        best = max(values.values())
        candidate_incentives.append(
            max(0.0, best - sum(predicted[key] * value for key, value in values.items()))
        )
        behavior_incentives.append(
            max(0.0, best - sum(behaviors[key] * value for key, value in values.items()))
        )
        target_incentives.append(
            max(0.0, best - sum(targets[key] * value for key, value in values.items()))
        )
        exact_support += int(set(predicted) == set(values))
    return NativeRegretSplitMetrics(
        split=split,
        sample_count=len(samples),
        information_set_count=len({sample.information_set_id for sample in samples}),
        policy_cross_entropy=policy_loss / len(samples),
        advantage_rmse=math.sqrt(sum(advantage_squared) / len(advantage_squared)),
        value_rmse=math.sqrt(sum(value_squared) / len(value_squared)),
        mean_candidate_deviation_incentive=sum(candidate_incentives) / len(samples),
        mean_behavior_deviation_incentive=sum(behavior_incentives) / len(samples),
        mean_target_deviation_incentive=sum(target_incentives) / len(samples),
        exact_legal_support_rate=exact_support / len(samples),
    )


def train_native_regret_blueprint(
    dataset: NativeRegretDataset,
    bootstrap_artifact: TrajectoryBootstrapArtifact,
    *,
    seed: int = DEFAULT_NATIVE_REGRET_SEED,
    epochs: int = 40,
    learning_rate: float = 0.003,
) -> NativeRegretTrainingRun:
    """Fit a regret-matching candidate; this is not a convergence certificate."""

    if epochs <= 0 or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("native regret training hyperparameters must be positive")
    if dataset.source_bootstrap_fingerprint != bootstrap_artifact.fingerprint:
        raise ValueError("native regret dataset/bootstrap fingerprint mismatch")
    if tuple(bootstrap_artifact.public_normalizer.names) != NATIVE_PUBLIC_FEATURE_NAMES:
        raise ValueError("Phase 6I public normalizer does not match native schema")
    if tuple(bootstrap_artifact.private_normalizer.names) != NATIVE_PRIVATE_FEATURE_NAMES:
        raise ValueError("Phase 6I private normalizer does not match native schema")
    if tuple(bootstrap_artifact.action_normalizer.names) != NATIVE_ACTION_FEATURE_NAMES:
        raise ValueError("Phase 6I action normalizer does not match native schema")
    bootstrap = restore_logged_trajectory_bootstrap(bootstrap_artifact)
    network, initialized_names = initialized_native_regret_network(bootstrap)
    train = _batch(
        dataset,
        "train",
        public_normalizer=bootstrap_artifact.public_normalizer,
        private_normalizer=bootstrap_artifact.private_normalizer,
        action_normalizer=bootstrap_artifact.action_normalizer,
    )
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial_loss = float(_loss(network, train))
    gradient_names: tuple[str, ...] = ()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        network.train()
        for epoch in range(epochs):
            loss = _loss(network, train)
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
    network.eval()
    with torch.no_grad():
        final_loss = float(_loss(network, train))
    architecture_fingerprint = _architecture_fingerprint(network)
    artifact = NativeRegretBlueprintArtifact(
        game_fingerprint=dataset.game_fingerprint,
        runtime_fingerprint=dataset.runtime_fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        source_bootstrap_fingerprint=bootstrap_artifact.fingerprint,
        feature_schema_fingerprint=dataset.feature_schema_fingerprint,
        architecture_fingerprint=architecture_fingerprint,
        public_normalizer=bootstrap_artifact.public_normalizer,
        private_normalizer=bootstrap_artifact.private_normalizer,
        action_normalizer=bootstrap_artifact.action_normalizer,
        candidate_ids=bootstrap_artifact.candidate_ids,
        hidden_dim=network.hidden_dim,
        seed=seed,
        training_rounds=dataset.training_rounds,
        epochs=epochs,
        learning_rate=learning_rate,
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        gradient_parameter_names=gradient_names,
        initialized_parameter_names=initialized_names,
        tensors=_tensor_artifacts(network),
    )
    provider = NativeRegretBlueprintProvider(network, artifact)
    metrics = tuple(
        _split_metrics(provider, dataset, split)
        for split in ("train", "calibration", "validation")
    )
    changed = 0
    for sample in dataset.samples:
        predicted = dict(sample.target_strategy)
        behavior = dict(sample.behavior_strategy)
        changed += int(max(predicted, key=predicted.get) != max(behavior, key=behavior.get))
    report = NativeRegretTrainingReport(
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        gradient_parameter_names=gradient_names,
        initialized_parameter_names=initialized_names,
        split_metrics=metrics,
        strategy_changed_from_behavior_samples=changed,
        artifact_fingerprint=artifact.fingerprint,
    )
    return NativeRegretTrainingRun(provider=provider, artifact=artifact, report=report)
