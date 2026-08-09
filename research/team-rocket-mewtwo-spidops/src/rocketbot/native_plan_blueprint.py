"""Separate neural regret and average-plan models for native plan trajectories.

The raw supervision is importance-weighted outcome-sampling regret from the
turn-level plan abstraction.  Because native horizons still create a heavy-
tailed estimator, the neural target uses a train-split-only ``asinh`` transform.
That transform preserves sign and ordering but changes the regret magnitude;
the resulting provider is a research candidate, not an equilibrium certificate
or a safe/deployable blueprint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import statistics
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
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
)
from .native_plan_game import NATIVE_PLAN_ACTION_FEATURE_NAMES, NATIVE_PLAN_ACTIONS
from .native_regret_blueprint import (
    NativeRegretBlueprintProvider,
)
from .neural_blueprint import TensorArtifact
from .outcome_sampling import (
    OutcomeRegretReservoir,
    OutcomeSamplingRegretSample,
    OutcomeSamplingStrategySample,
    OutcomeStrategyReservoir,
)
from .trajectory_neural import NamedFeatureNormalizer


NATIVE_PLAN_BLUEPRINT_SCHEMA_VERSION = 1
DEFAULT_NATIVE_PLAN_BLUEPRINT_SEED = 20_260_728


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


def _split_for_information_set(identifier: str) -> str:
    bucket = int(sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "calibration"
    return "validation"


def _sample_feature_contract(
    sample: OutcomeSamplingRegretSample | OutcomeSamplingStrategySample,
) -> None:
    if tuple(name for name, _ in sample.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
        raise ValueError("native plan public feature schema mismatch")
    if tuple(name for name, _ in sample.private_features) != NATIVE_PRIVATE_FEATURE_NAMES:
        raise ValueError("native plan private feature schema mismatch")
    expected_ids = tuple(action.action_id for action in NATIVE_PLAN_ACTIONS)
    if tuple(action_id for action_id, _ in sample.action_features) != expected_ids:
        raise ValueError("native plan action support mismatch")
    if any(
        tuple(name for name, _ in features) != NATIVE_PLAN_ACTION_FEATURE_NAMES
        for _, features in sample.action_features
    ):
        raise ValueError("native plan action feature schema mismatch")
    if not sample.source_official_engine_executed:
        raise ValueError("native plan samples require genuine descendant execution")
    if sample.hidden_completion_prior_official:
        raise ValueError("native plan samples cannot promote the supplied prior")


@dataclass(frozen=True, slots=True)
class NativePlanOutcomeDataset:
    native_game_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    macro_game_fingerprints: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    iterations: int
    exploration: float
    player_0_regrets: OutcomeRegretReservoir
    player_1_regrets: OutcomeRegretReservoir
    average_strategy: OutcomeStrategyReservoir
    native_transition_count: int
    macro_transition_count: int
    tactical_transition_count: int
    source_official_engine_executed: bool = True
    dataset_official_engine_executed: bool = False
    hidden_completion_prior_official: bool = False
    equilibrium_convergence_validated: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "native_game_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_tactical_blueprint_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.macro_game_fingerprints or not self.candidate_ids:
            raise ValueError("native plan dataset needs macro games/candidates")
        if len(self.macro_game_fingerprints) != len(self.candidate_ids):
            raise ValueError("native plan macro games/candidates must align")
        for value in self.macro_game_fingerprints:
            _validate_digest(value, "macro_game_fingerprint")
        if len(set(self.macro_game_fingerprints)) != len(self.macro_game_fingerprints):
            raise ValueError("native plan macro-game fingerprints must be unique")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("native plan candidate IDs must be unique")
        if self.iterations <= 0 or not 0 < self.exploration <= 1:
            raise ValueError("native plan sampling metadata is invalid")
        if min(
            self.player_0_regrets.seen,
            self.player_1_regrets.seen,
            self.average_strategy.seen,
            self.native_transition_count,
            self.macro_transition_count,
            self.tactical_transition_count,
        ) <= 0:
            raise ValueError("native plan dataset accounting must be positive")
        for sample in (
            self.player_0_regrets.samples
            + self.player_1_regrets.samples
            + self.average_strategy.samples
        ):
            _sample_feature_contract(sample)
            if sample.game_fingerprint not in set(self.macro_game_fingerprints):
                raise ValueError("native plan sample belongs to an unknown macro game")
        if any(
            (
                not self.source_official_engine_executed,
                self.dataset_official_engine_executed,
                self.hidden_completion_prior_official,
                self.equilibrium_convergence_validated,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("native plan dataset promotes an invalid capability")
        split_sets = {
            _split_for_information_set(sample.information_set_id)
            for sample in self.player_0_regrets.samples
            + self.player_1_regrets.samples
            + self.average_strategy.samples
        }
        if split_sets != {"train", "calibration", "validation"}:
            raise ValueError("native plan dataset must populate all three splits")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": NATIVE_PLAN_BLUEPRINT_SCHEMA_VERSION,
            "native_game_fingerprint": self.native_game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "macro_game_fingerprints": list(self.macro_game_fingerprints),
            "candidate_ids": list(self.candidate_ids),
            "iterations": self.iterations,
            "exploration": self.exploration,
            "player_0_regrets": self.player_0_regrets.to_dict(),
            "player_1_regrets": self.player_1_regrets.to_dict(),
            "average_strategy": self.average_strategy.to_dict(),
            "accounting": {
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "tactical_transition_count": self.tactical_transition_count,
            },
            "claims": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "dataset_official_engine_executed": self.dataset_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "equilibrium_convergence_validated": (
                    self.equilibrium_convergence_validated
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
                "turn_plan_actions_official": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativePlanOutcomeDataset":
        accounting = raw["accounting"]
        claims = raw["claims"]
        result = cls(
            native_game_fingerprint=str(raw["native_game_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            macro_game_fingerprints=tuple(
                str(value) for value in raw["macro_game_fingerprints"]
            ),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            iterations=int(raw["iterations"]),
            exploration=float(raw["exploration"]),
            player_0_regrets=OutcomeRegretReservoir.from_dict(raw["player_0_regrets"]),
            player_1_regrets=OutcomeRegretReservoir.from_dict(raw["player_1_regrets"]),
            average_strategy=OutcomeStrategyReservoir.from_dict(
                raw["average_strategy"]
            ),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            tactical_transition_count=int(accounting["tactical_transition_count"]),
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
            raise ValueError("native plan dataset fingerprint mismatch")
        return result


class NativePlanHeadNetwork(nn.Module):
    """One independently trained plan-value or average-policy head."""

    def __init__(
        self,
        *,
        public_width: int,
        private_width: int,
        action_width: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(public_width, private_width, action_width, hidden_dim) <= 0:
            raise ValueError("native plan network widths must be positive")
        self.public_width = int(public_width)
        self.private_width = int(private_width)
        self.action_width = int(action_width)
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
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.action_bias = nn.Linear(hidden_dim, 1)
        self.double()

    def forward(
        self,
        public_features: Tensor,
        private_features: Tensor,
        action_features: Tensor,
        action_counts: Tensor,
    ) -> Tensor:
        if public_features.ndim != 2 or public_features.shape[1] != self.public_width:
            raise ValueError("native plan public tensor shape mismatch")
        if private_features.ndim != 2 or private_features.shape[1] != self.private_width:
            raise ValueError("native plan private tensor shape mismatch")
        if action_features.ndim != 2 or action_features.shape[1] != self.action_width:
            raise ValueError("native plan action tensor shape mismatch")
        if action_counts.ndim != 1 or len(action_counts) != len(public_features):
            raise ValueError("native plan action-count tensor shape mismatch")
        if int(action_counts.sum()) != len(action_features):
            raise ValueError("native plan action counts do not partition actions")
        public_hidden = self.public_encoder(public_features)
        private_hidden = self.private_encoder(private_features)
        states = self.state_encoder(torch.cat((public_hidden, private_hidden), dim=-1))
        actions = self.action_encoder(action_features)
        indices = torch.repeat_interleave(
            torch.arange(len(states), device=states.device),
            action_counts,
        )
        queries = self.query(states)[indices]
        return (
            (queries * actions).sum(dim=-1) / math.sqrt(self.hidden_dim)
            + self.action_bias(actions).squeeze(-1)
        )


def _architecture_fingerprint(network: NativePlanHeadNetwork) -> str:
    return _fingerprint(
        {
            "schema": NATIVE_PLAN_BLUEPRINT_SCHEMA_VERSION,
            "public_width": network.public_width,
            "private_width": network.private_width,
            "action_width": network.action_width,
            "hidden_dim": network.hidden_dim,
            "action_features": NATIVE_PLAN_ACTION_FEATURE_NAMES,
            "heads": ("player_0_regret", "player_1_regret", "average_strategy"),
        }
    )


def _initialize_network(
    source: NativeRegretBlueprintProvider,
    *,
    mode: str,
) -> tuple[NativePlanHeadNetwork, tuple[str, ...]]:
    if mode not in {"advantage", "average"}:
        raise ValueError("native plan initialization mode is invalid")
    source_network = source.network
    network = NativePlanHeadNetwork(
        public_width=source_network.public_width,
        private_width=source_network.private_width,
        action_width=len(NATIVE_PLAN_ACTION_FEATURE_NAMES),
        hidden_dim=source_network.hidden_dim,
    )
    copied = []
    for target_name, source_name in (
        ("public_encoder", "public_encoder"),
        ("private_encoder", "private_encoder"),
        ("state_encoder", "state_encoder"),
        (
            "query",
            "advantage_query" if mode == "advantage" else "average_policy_query",
        ),
        (
            "action_bias",
            "advantage_action_bias"
            if mode == "advantage"
            else "average_policy_action_bias",
        ),
    ):
        target = getattr(network, target_name)
        source_module = getattr(source_network, source_name)
        target.load_state_dict(source_module.state_dict(), strict=True)
        copied.extend(f"{target_name}.{name}" for name in target.state_dict())
    return network, tuple(copied)


@dataclass(frozen=True, slots=True)
class NativePlanBlueprintArtifact:
    native_game_fingerprint: str
    deck_fingerprint: str
    dataset_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    architecture_fingerprint: str
    public_normalizer: NamedFeatureNormalizer
    private_normalizer: NamedFeatureNormalizer
    action_feature_names: tuple[str, ...]
    hidden_dim: int
    seed: int
    epochs: int
    learning_rate: float
    player_0_target_scale: float
    player_1_target_scale: float
    initial_training_loss: float
    final_training_loss: float
    gradient_parameter_names: tuple[str, ...]
    initialized_parameter_names: tuple[str, ...]
    tensors: tuple[TensorArtifact, ...]
    source_official_engine_executed: bool = True
    model_official_engine_executed: bool = False
    outcome_regret_objective_trained: bool = True
    equilibrium_convergence_validated: bool = False
    deployable_blueprint: bool = False
    calibrated_counterfactual_values: bool = False
    safe_resolver_compatible: bool = False
    exact_deck: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "native_game_fingerprint",
            "deck_fingerprint",
            "dataset_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "architecture_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.action_feature_names != NATIVE_PLAN_ACTION_FEATURE_NAMES:
            raise ValueError("native plan artifact action schema mismatch")
        if min(self.hidden_dim, self.epochs) <= 0 or self.learning_rate <= 0:
            raise ValueError("native plan artifact training metadata is invalid")
        if min(self.player_0_target_scale, self.player_1_target_scale) <= 0:
            raise ValueError("native plan target scales must be positive")
        if min(self.initial_training_loss, self.final_training_loss) < 0:
            raise ValueError("native plan losses cannot be negative")
        if not self.gradient_parameter_names or not self.initialized_parameter_names:
            raise ValueError("native plan artifact needs gradient/initialization evidence")
        names = [item.name for item in self.tensors]
        if not names or len(names) != len(set(names)):
            raise ValueError("native plan artifact tensors must be unique")
        if any(
            (
                not self.source_official_engine_executed,
                self.model_official_engine_executed,
                not self.outcome_regret_objective_trained,
                self.equilibrium_convergence_validated,
                self.deployable_blueprint,
                self.calibrated_counterfactual_values,
                self.safe_resolver_compatible,
                self.exact_deck,
            )
        ):
            raise ValueError("native plan artifact promotes an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": NATIVE_PLAN_BLUEPRINT_SCHEMA_VERSION,
            "native_game_fingerprint": self.native_game_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "architecture_fingerprint": self.architecture_fingerprint,
            "normalizers": {
                "public": self.public_normalizer.to_dict(),
                "private": self.private_normalizer.to_dict(),
            },
            "action_feature_names": list(self.action_feature_names),
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "target_scales": {
                "player_0": self.player_0_target_scale,
                "player_1": self.player_1_target_scale,
            },
            "initial_training_loss": self.initial_training_loss,
            "final_training_loss": self.final_training_loss,
            "gradient_parameter_names": list(self.gradient_parameter_names),
            "initialized_parameter_names": list(self.initialized_parameter_names),
            "claims": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "model_official_engine_executed": self.model_official_engine_executed,
                "outcome_regret_objective_trained": self.outcome_regret_objective_trained,
                "equilibrium_convergence_validated": (
                    self.equilibrium_convergence_validated
                ),
                "deployable_blueprint": self.deployable_blueprint,
                "calibrated_counterfactual_values": (
                    self.calibrated_counterfactual_values
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "exact_deck": self.exact_deck,
            },
            "tensors": [asdict(item) for item in self.tensors],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativePlanBlueprintArtifact":
        normalizers = raw["normalizers"]
        scales = raw["target_scales"]
        claims = raw["claims"]
        result = cls(
            native_game_fingerprint=str(raw["native_game_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            architecture_fingerprint=str(raw["architecture_fingerprint"]),
            public_normalizer=NamedFeatureNormalizer.from_dict(normalizers["public"]),
            private_normalizer=NamedFeatureNormalizer.from_dict(normalizers["private"]),
            action_feature_names=tuple(str(value) for value in raw["action_feature_names"]),
            hidden_dim=int(raw["hidden_dim"]),
            seed=int(raw["seed"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            player_0_target_scale=float(scales["player_0"]),
            player_1_target_scale=float(scales["player_1"]),
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
            outcome_regret_objective_trained=bool(
                claims["outcome_regret_objective_trained"]
            ),
            equilibrium_convergence_validated=bool(
                claims["equilibrium_convergence_validated"]
            ),
            deployable_blueprint=bool(claims["deployable_blueprint"]),
            calibrated_counterfactual_values=bool(
                claims["calibrated_counterfactual_values"]
            ),
            safe_resolver_compatible=bool(claims["safe_resolver_compatible"]),
            exact_deck=bool(claims["exact_deck"]),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native plan artifact fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativePlanSplitMetrics:
    split: str
    player_0_samples: int
    player_1_samples: int
    strategy_samples: int
    player_0_transformed_rmse: float
    player_1_transformed_rmse: float
    strategy_cross_entropy: float


@dataclass(frozen=True, slots=True)
class NativePlanTrainingReport:
    initial_training_loss: float
    final_training_loss: float
    raw_regret_median_absolute: float
    raw_regret_p95_absolute: float
    raw_regret_maximum_absolute: float
    strategy_weight_p95: float
    gradient_parameter_names: tuple[str, ...]
    initialized_parameter_names: tuple[str, ...]
    split_metrics: tuple[NativePlanSplitMetrics, ...]
    maximum_restore_drift: float
    artifact_fingerprint: str


@dataclass(slots=True)
class NativePlanTrainingRun:
    provider: "NativePlanBlueprintProvider"
    artifact: NativePlanBlueprintArtifact
    report: NativePlanTrainingReport


def native_plan_sample_view(
    sample: OutcomeSamplingRegretSample | OutcomeSamplingStrategySample,
) -> tuple[InformationSet, PublicBeliefState]:
    """Reconstruct persisted plan-model inputs without hidden simulator truth.

    The synthetic one-particle range is metadata for provider-shape validation;
    it is not the sampled native hidden completion and is never a belief claim.
    """

    _sample_feature_contract(sample)
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
    actions = tuple(
        AbstractAction(
            action_id=action_id,
            label=next(
                action.label
                for action in NATIVE_PLAN_ACTIONS
                if action.action_id == action_id
            ),
            official_option_indices=(),
            features=features,
        )
        for action_id, features in sample.action_features
    )
    information_set = InformationSet(
        info_set_id=sample.information_set_id,
        public_belief_fingerprint=belief.fingerprint,
        actor=InformationSetActor.SELF,
        legal_actions=actions,
        depth=sample.depth,
        private_observation_fingerprint=sample.private_observation_fingerprint,
        private_features=sample.private_features,
    )
    return information_set, belief


def _feature_rows(
    samples: Sequence[OutcomeSamplingRegretSample | OutcomeSamplingStrategySample],
    public_normalizer: NamedFeatureNormalizer,
    private_normalizer: NamedFeatureNormalizer,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    public = [public_normalizer.transform(sample.public_features) for sample in samples]
    private = [private_normalizer.transform(sample.private_features) for sample in samples]
    action = [
        tuple(float(value) for _, value in features)
        for sample in samples
        for _, features in sample.action_features
    ]
    counts = [len(sample.action_features) for sample in samples]
    return (
        torch.tensor(public, dtype=torch.float64),
        torch.tensor(private, dtype=torch.float64),
        torch.tensor(action, dtype=torch.float64),
        torch.tensor(counts, dtype=torch.long),
    )


def _target_scale(samples: Sequence[OutcomeSamplingRegretSample]) -> float:
    values = [
        abs(value)
        for sample in samples
        if _split_for_information_set(sample.information_set_id) == "train"
        for _, value in sample.sampled_regrets
        if abs(value) > 1e-15
    ]
    return max(1e-6, statistics.median(values) if values else 1.0)


def _regret_loss(
    network: NativePlanHeadNetwork,
    samples: Sequence[OutcomeSamplingRegretSample],
    scale: float,
    public_normalizer: NamedFeatureNormalizer,
    private_normalizer: NamedFeatureNormalizer,
) -> Tensor:
    features = _feature_rows(samples, public_normalizer, private_normalizer)
    predictions = network(*features)
    targets = torch.tensor(
        [
            math.asinh(value / scale)
            for sample in samples
            for _, value in sample.sampled_regrets
        ],
        dtype=torch.float64,
    )
    return torch.mean(torch.nn.functional.smooth_l1_loss(
        predictions, targets, reduction="none"
    ))


def _strategy_loss(
    network: NativePlanHeadNetwork,
    samples: Sequence[OutcomeSamplingStrategySample],
    public_normalizer: NamedFeatureNormalizer,
    private_normalizer: NamedFeatureNormalizer,
    weight_cap: float,
) -> Tensor:
    features = _feature_rows(samples, public_normalizer, private_normalizer)
    logits = network(*features)
    losses = []
    weights = []
    offset = 0
    for sample in samples:
        count = len(sample.strategy)
        target = torch.tensor(
            [value for _, value in sample.strategy], dtype=torch.float64
        )
        losses.append(
            -(target * torch.log_softmax(logits[offset : offset + count], dim=0)).sum()
        )
        weights.append(min(sample.importance_weight, weight_cap))
        offset += count
    weight_tensor = torch.tensor(weights, dtype=torch.float64)
    return torch.sum(torch.stack(losses) * weight_tensor) / torch.clamp(
        weight_tensor.sum(), min=1e-12
    )


def _all_tensors(
    networks: Mapping[str, NativePlanHeadNetwork],
) -> tuple[TensorArtifact, ...]:
    return tuple(
        TensorArtifact(
            name=f"{prefix}.{name}",
            shape=tuple(tensor.shape),
            values=tuple(
                float(value)
                for value in tensor.detach().cpu().reshape(-1).tolist()
            ),
        )
        for prefix, network in sorted(networks.items())
        for name, tensor in network.state_dict().items()
    )


def _restore_networks(
    artifact: NativePlanBlueprintArtifact,
) -> dict[str, NativePlanHeadNetwork]:
    networks = {
        prefix: NativePlanHeadNetwork(
            public_width=len(artifact.public_normalizer.names),
            private_width=len(artifact.private_normalizer.names),
            action_width=len(artifact.action_feature_names),
            hidden_dim=artifact.hidden_dim,
        )
        for prefix in ("average", "player_0", "player_1")
    }
    if any(
        _architecture_fingerprint(network) != artifact.architecture_fingerprint
        for network in networks.values()
    ):
        raise ValueError("native plan architecture fingerprint mismatch")
    supplied = {item.name: item for item in artifact.tensors}
    expected_names = {
        f"{prefix}.{name}"
        for prefix, network in networks.items()
        for name in network.state_dict()
    }
    if set(supplied) != expected_names:
        raise ValueError("native plan tensor names do not match architecture")
    for prefix, network in networks.items():
        state = {}
        for name, template in network.state_dict().items():
            item = supplied[f"{prefix}.{name}"]
            if item.shape != tuple(template.shape):
                raise ValueError(f"native plan tensor shape mismatch for {prefix}.{name}")
            state[name] = torch.tensor(item.values, dtype=torch.float64).reshape(
                item.shape
            )
        network.load_state_dict(state, strict=True)
        network.eval()
    return networks


class NativePlanBlueprintProvider:
    """Restorable average-plan candidate with separate player regret heads."""

    def __init__(
        self,
        networks: Mapping[str, NativePlanHeadNetwork],
        artifact: NativePlanBlueprintArtifact,
    ) -> None:
        self.networks = dict(networks)
        self.artifact = artifact
        self._checkpoint = BlueprintCheckpoint(
            provider_name="native_outcome_sampled_turn_plan_candidate",
            algorithm="outcome_sampling_mccfr_plan_abstraction_asinh_regression",
            model_fingerprint=artifact.fingerprint,
            game_fingerprint=artifact.native_game_fingerprint,
            deck_fingerprint=artifact.deck_fingerprint,
            training_steps=artifact.epochs,
            trained=True,
            evaluation_notes=(
                "Separate player regret heads and average-plan head trained on a "
                "heavy-tailed, transformed outcome-sampling dataset. No equilibrium, "
                "counterfactual-bound, safety, deployment, or strength certificate."
            ),
        )

    @property
    def checkpoint(self) -> BlueprintCheckpoint:
        return self._checkpoint

    def _scores(
        self,
        network: NativePlanHeadNetwork,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> tuple[float, ...]:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("native plan provider received mismatched belief")
        if tuple(action.action_id for action in information_set.legal_actions) != tuple(
            action.action_id for action in NATIVE_PLAN_ACTIONS
        ):
            raise ValueError("native plan provider support mismatch")
        sample_public = self.artifact.public_normalizer.transform(belief.public_features)
        sample_private = self.artifact.private_normalizer.transform(
            information_set.private_features
        )
        actions = tuple(
            tuple(float(value) for _, value in action.features)
            for action in information_set.legal_actions
        )
        with torch.no_grad():
            values = network(
                torch.tensor([sample_public], dtype=torch.float64),
                torch.tensor([sample_private], dtype=torch.float64),
                torch.tensor(actions, dtype=torch.float64),
                torch.tensor([len(actions)], dtype=torch.long),
            )
        return tuple(float(value) for value in values.tolist())

    def strategy(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        scores = self._scores(self.networks["average"], information_set, belief)
        probabilities = torch.softmax(
            torch.tensor(scores, dtype=torch.float64), dim=0
        ).tolist()
        result = ActionDistribution(
            tuple(
                (action.action_id, float(probability))
                for action, probability in zip(
                    information_set.legal_actions, probabilities, strict=True
                )
            )
        )
        result.validate_support(information_set)
        return result

    def current_strategy(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        player: int,
    ) -> ActionDistribution:
        if player not in (0, 1):
            raise ValueError("native plan current-strategy player is invalid")
        transformed = self._scores(
            self.networks[f"player_{player}"], information_set, belief
        )
        positive = [max(0.0, value) for value in transformed]
        if sum(positive) <= 1e-12:
            positive = [1.0] * len(positive)
        result = ActionDistribution.normalized(
            {
                action.action_id: value
                for action, value in zip(
                    information_set.legal_actions, positive, strict=True
                )
            }
        )
        result.validate_support(information_set)
        return result

    def counterfactual_values(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        player: int,
    ) -> CounterfactualValues:
        del information_set, belief, player
        raise RuntimeError(
            "native plan outcome sampling has no calibrated per-hypothesis "
            "counterfactual values and cannot be used for safe resolving"
        )


def restore_native_plan_blueprint(
    artifact: NativePlanBlueprintArtifact,
) -> NativePlanBlueprintProvider:
    return NativePlanBlueprintProvider(_restore_networks(artifact), artifact)


def train_native_plan_blueprint(
    dataset: NativePlanOutcomeDataset,
    source: NativeRegretBlueprintProvider,
    *,
    epochs: int = 30,
    learning_rate: float = 2e-3,
    seed: int = DEFAULT_NATIVE_PLAN_BLUEPRINT_SEED,
) -> NativePlanTrainingRun:
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("native plan training limits must be positive")
    torch.manual_seed(seed)
    player_0, initialized_advantage = _initialize_network(source, mode="advantage")
    torch.manual_seed(seed + 1)
    player_1, initialized_advantage_1 = _initialize_network(source, mode="advantage")
    torch.manual_seed(seed + 2)
    average, initialized_average = _initialize_network(source, mode="average")
    networks = {"player_0": player_0, "player_1": player_1, "average": average}
    train_p0 = tuple(
        sample
        for sample in dataset.player_0_regrets.samples
        if _split_for_information_set(sample.information_set_id) == "train"
    )
    train_p1 = tuple(
        sample
        for sample in dataset.player_1_regrets.samples
        if _split_for_information_set(sample.information_set_id) == "train"
    )
    train_strategy = tuple(
        sample
        for sample in dataset.average_strategy.samples
        if _split_for_information_set(sample.information_set_id) == "train"
    )
    if min(len(train_p0), len(train_p1), len(train_strategy)) <= 0:
        raise ValueError("native plan training split is empty")
    scale_p0 = _target_scale(dataset.player_0_regrets.samples)
    scale_p1 = _target_scale(dataset.player_1_regrets.samples)
    weights = sorted(sample.importance_weight for sample in train_strategy)
    weight_cap = weights[min(len(weights) - 1, int(0.95 * (len(weights) - 1)))]
    parameters = [
        parameter
        for network in networks.values()
        for parameter in network.parameters()
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    def objective() -> Tensor:
        return (
            _regret_loss(
                player_0,
                train_p0,
                scale_p0,
                source.artifact.public_normalizer,
                source.artifact.private_normalizer,
            )
            + _regret_loss(
                player_1,
                train_p1,
                scale_p1,
                source.artifact.public_normalizer,
                source.artifact.private_normalizer,
            )
            + _strategy_loss(
                average,
                train_strategy,
                source.artifact.public_normalizer,
                source.artifact.private_normalizer,
                weight_cap,
            )
        )

    with torch.no_grad():
        initial_loss = float(objective())
    gradient_names: set[str] = set()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        for prefix, network in networks.items():
            for name, parameter in network.named_parameters():
                if parameter.grad is not None and bool((parameter.grad != 0).any()):
                    gradient_names.add(f"{prefix}.{name}")
        optimizer.step()
    with torch.no_grad():
        final_loss = float(objective())
    initialized = tuple(
        sorted(
            {f"player_0.{name}" for name in initialized_advantage}
            | {f"player_1.{name}" for name in initialized_advantage_1}
            | {f"average.{name}" for name in initialized_average}
        )
    )
    artifact = NativePlanBlueprintArtifact(
        native_game_fingerprint=dataset.native_game_fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        source_tactical_blueprint_fingerprint=(
            dataset.source_tactical_blueprint_fingerprint
        ),
        architecture_fingerprint=_architecture_fingerprint(player_0),
        public_normalizer=source.artifact.public_normalizer,
        private_normalizer=source.artifact.private_normalizer,
        action_feature_names=NATIVE_PLAN_ACTION_FEATURE_NAMES,
        hidden_dim=player_0.hidden_dim,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        player_0_target_scale=scale_p0,
        player_1_target_scale=scale_p1,
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        gradient_parameter_names=tuple(sorted(gradient_names)),
        initialized_parameter_names=initialized,
        tensors=_all_tensors(networks),
    )
    provider = restore_native_plan_blueprint(artifact)

    def metrics(split: str) -> NativePlanSplitMetrics:
        p0 = tuple(
            sample
            for sample in dataset.player_0_regrets.samples
            if _split_for_information_set(sample.information_set_id) == split
        )
        p1 = tuple(
            sample
            for sample in dataset.player_1_regrets.samples
            if _split_for_information_set(sample.information_set_id) == split
        )
        strategies = tuple(
            sample
            for sample in dataset.average_strategy.samples
            if _split_for_information_set(sample.information_set_id) == split
        )

        def rmse(
            network: NativePlanHeadNetwork,
            samples: Sequence[OutcomeSamplingRegretSample],
            scale: float,
        ) -> float:
            if not samples:
                return 0.0
            with torch.no_grad():
                predictions = network(
                    *_feature_rows(
                        samples,
                        source.artifact.public_normalizer,
                        source.artifact.private_normalizer,
                    )
                )
            targets = torch.tensor(
                [
                    math.asinh(value / scale)
                    for sample in samples
                    for _, value in sample.sampled_regrets
                ],
                dtype=torch.float64,
            )
            return float(torch.sqrt(torch.mean((predictions - targets) ** 2)))

        ce = 0.0
        if strategies:
            with torch.no_grad():
                ce = float(
                    _strategy_loss(
                        average,
                        strategies,
                        source.artifact.public_normalizer,
                        source.artifact.private_normalizer,
                        weight_cap,
                    )
                )
        return NativePlanSplitMetrics(
            split=split,
            player_0_samples=len(p0),
            player_1_samples=len(p1),
            strategy_samples=len(strategies),
            player_0_transformed_rmse=rmse(player_0, p0, scale_p0),
            player_1_transformed_rmse=rmse(player_1, p1, scale_p1),
            strategy_cross_entropy=ce,
        )

    raw_values = sorted(
        abs(value)
        for sample in dataset.player_0_regrets.samples + dataset.player_1_regrets.samples
        for _, value in sample.sampled_regrets
    )
    strategy_weights = sorted(
        sample.importance_weight for sample in dataset.average_strategy.samples
    )
    p95_index = min(len(raw_values) - 1, int(0.95 * (len(raw_values) - 1)))
    strategy_p95_index = min(
        len(strategy_weights) - 1,
        int(0.95 * (len(strategy_weights) - 1)),
    )
    restored_networks = provider.networks
    max_drift = 0.0
    for prefix, network in networks.items():
        for name, tensor in network.state_dict().items():
            delta = torch.max(
                torch.abs(tensor - restored_networks[prefix].state_dict()[name])
            )
            max_drift = max(max_drift, float(delta))
    report = NativePlanTrainingReport(
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        raw_regret_median_absolute=statistics.median(raw_values),
        raw_regret_p95_absolute=raw_values[p95_index],
        raw_regret_maximum_absolute=raw_values[-1],
        strategy_weight_p95=strategy_weights[strategy_p95_index],
        gradient_parameter_names=artifact.gradient_parameter_names,
        initialized_parameter_names=artifact.initialized_parameter_names,
        split_metrics=tuple(metrics(split) for split in ("train", "calibration", "validation")),
        maximum_restore_drift=max_drift,
        artifact_fingerprint=artifact.fingerprint,
    )
    return NativePlanTrainingRun(provider, artifact, report)
