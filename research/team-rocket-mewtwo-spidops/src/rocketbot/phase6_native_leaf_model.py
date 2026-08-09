"""Phase 6U low-noise, allocation-heldout native leaf comparison.

The target remains a repeated fixed-profile terminal return, not an exact
counterfactual value.  This module increases each leaf to ten independent
continuations, compares three pre-existing value baselines on a calibration
split, and evaluates the selected baseline on disjoint hidden allocations.
All native descendants are genuine licensed-engine search transitions; the
particles, cutoff, fixed profile, predictions, and interval are abstractions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .depth_limited_game import DepthLimitedInformationSetGame
from .native_blueprint_features import (
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeBlueprintFeatureBridge,
)
from .native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    restore_native_plan_blueprint,
)
from .native_plan_game import NativePlanGameState, OfficialNativePlanGame
from .native_plan_leaf import UncalibratedNativePlanLeafValueProvider
from .native_public_range import LoggedTrajectoryNativePublicRangeProvider
from .native_regret_blueprint import (
    NativeRegretBlueprintArtifact,
    restore_native_regret_blueprint,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    rocket_compatibility_profile,
)
from .official_search_game import (
    HiddenZoneCompletion,
    OfficialNativeSearchGame,
    sample_hidden_zone_completion,
)
from .official_trajectory import OfficialTrajectoryDataset
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _runtime_basics
from .phase6_native_leaf_calibration import (
    _rollout_to_terminal,
    _sample_cutoff_state,
)
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence
from .trajectory_neural import (
    TrajectoryBootstrapArtifact,
    restore_logged_trajectory_bootstrap,
)


PHASE6_NATIVE_LEAF_MODEL_SCHEMA_VERSION = 1
PHASE6_NATIVE_LEAF_MODEL_SEED = 6_027
LEAF_MODEL_NAMES = (
    "phase6i_logged_value",
    "phase6k_tactical_value",
    "zero_baseline",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big"
    )


@dataclass(frozen=True, slots=True)
class NativeLeafModelConfig:
    target_coverage: float = 0.90
    minimum_calibration_examples: int = 18
    minimum_validation_examples: int = 6
    minimum_candidate_strata: int = 6
    minimum_rollouts_per_leaf: int = 10
    maximum_mean_interval_width: float = 1.25
    maximum_rollout_standard_error: float = 0.35
    numerical_margin: float = 1e-9

    def __post_init__(self) -> None:
        if not 0 < self.target_coverage <= 1:
            raise ValueError("leaf-model target coverage must lie in (0, 1]")
        if min(
            self.minimum_calibration_examples,
            self.minimum_validation_examples,
            self.minimum_candidate_strata,
            self.minimum_rollouts_per_leaf,
        ) <= 0:
            raise ValueError("leaf-model minimums must be positive")
        for field_name in (
            "maximum_mean_interval_width",
            "maximum_rollout_standard_error",
            "numerical_margin",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NativeLeafModelExample:
    example_id: str
    source_game_fingerprint: str
    source_state_id: str
    information_set_id: str
    public_belief_fingerprint: str
    public_history_fingerprint: str
    private_observation_fingerprint: str
    particle_id: str
    particle_fingerprint: str
    candidate_id_supervision_only: str
    split: str
    variant: int
    acting_player: int
    value_player: int
    predictions: tuple[tuple[str, float], ...]
    terminal_returns: tuple[float, ...]
    empirical_return: float
    rollout_standard_error: float
    rollout_macro_transitions: int
    terminal_native_depth_minimum: int
    terminal_native_depth_maximum: int
    public_features: tuple[tuple[str, float], ...]
    private_features: tuple[tuple[str, float], ...]
    source_descendant_official_engine_executed: bool = True
    target_exact_counterfactual_value: bool = False
    selected_hidden_truth_identifier_input: bool = False
    raw_private_card_ids_persisted: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "example_id",
            "source_game_fingerprint",
            "source_state_id",
            "public_belief_fingerprint",
            "public_history_fingerprint",
            "private_observation_fingerprint",
            "particle_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not all(
            (
                self.information_set_id,
                self.particle_id,
                self.candidate_id_supervision_only,
            )
        ):
            raise ValueError("leaf-model example identities are required")
        if self.split not in {"calibration", "validation"}:
            raise ValueError("leaf-model split is invalid")
        if self.variant < 0 or self.acting_player not in (0, 1) or self.value_player not in (0, 1):
            raise ValueError("leaf-model particle/player metadata is invalid")
        prediction_names = tuple(name for name, _ in self.predictions)
        if prediction_names != LEAF_MODEL_NAMES:
            raise ValueError("leaf-model predictions have invalid support/order")
        values = tuple(value for _, value in self.predictions) + self.terminal_returns + (
            self.empirical_return,
        )
        if len(self.terminal_returns) < 2 or any(
            not math.isfinite(value) or not -1 <= value <= 1 for value in values
        ):
            raise ValueError("leaf-model utilities are invalid")
        if not math.isclose(
            self.empirical_return,
            statistics.mean(self.terminal_returns),
            abs_tol=1e-12,
        ):
            raise ValueError("leaf-model empirical target must equal rollout mean")
        if not math.isfinite(self.rollout_standard_error) or self.rollout_standard_error < 0:
            raise ValueError("leaf-model rollout standard error is invalid")
        if min(
            self.rollout_macro_transitions,
            self.terminal_native_depth_minimum,
            self.terminal_native_depth_maximum,
        ) <= 0 or self.terminal_native_depth_minimum > self.terminal_native_depth_maximum:
            raise ValueError("leaf-model rollout accounting is invalid")
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("leaf-model public feature schema mismatch")
        if tuple(name for name, _ in self.private_features) != NATIVE_PRIVATE_FEATURE_NAMES:
            raise ValueError("leaf-model private feature schema mismatch")
        if any(
            (
                not self.source_descendant_official_engine_executed,
                self.target_exact_counterfactual_value,
                self.selected_hidden_truth_identifier_input,
                self.raw_private_card_ids_persisted,
            )
        ):
            raise ValueError("leaf-model example promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def prediction(self, model_name: str) -> float:
        try:
            return dict(self.predictions)[model_name]
        except KeyError as exc:
            raise ValueError("unknown leaf-model prediction") from exc

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_MODEL_SCHEMA_VERSION,
            "example_id": self.example_id,
            "source": {
                "game_fingerprint": self.source_game_fingerprint,
                "state_id": self.source_state_id,
                "information_set_id": self.information_set_id,
                "public_belief_fingerprint": self.public_belief_fingerprint,
                "public_history_fingerprint": self.public_history_fingerprint,
                "private_observation_fingerprint": self.private_observation_fingerprint,
                "source_descendant_official_engine_executed": (
                    self.source_descendant_official_engine_executed
                ),
            },
            "particle": {
                "particle_id": self.particle_id,
                "particle_fingerprint": self.particle_fingerprint,
                "candidate_id_supervision_only": self.candidate_id_supervision_only,
                "split": self.split,
                "variant": self.variant,
                "allocation_heldout": True,
                "selected_hidden_truth_identifier_input": (
                    self.selected_hidden_truth_identifier_input
                ),
            },
            "value_target": {
                "acting_player": self.acting_player,
                "value_player": self.value_player,
                "predictions": dict(self.predictions),
                "terminal_returns": list(self.terminal_returns),
                "empirical_return": self.empirical_return,
                "rollout_standard_error": self.rollout_standard_error,
                "rollout_macro_transitions": self.rollout_macro_transitions,
                "terminal_native_depth_minimum": self.terminal_native_depth_minimum,
                "terminal_native_depth_maximum": self.terminal_native_depth_maximum,
                "target_exact_counterfactual_value": self.target_exact_counterfactual_value,
            },
            "features": {
                "public": [list(item) for item in self.public_features],
                "actor_private_hashed": [list(item) for item in self.private_features],
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeLeafModelExample":
        source = raw["source"]
        particle = raw["particle"]
        target = raw["value_target"]
        features = raw["features"]
        predictions = target["predictions"]
        result = cls(
            example_id=str(raw["example_id"]),
            source_game_fingerprint=str(source["game_fingerprint"]),
            source_state_id=str(source["state_id"]),
            information_set_id=str(source["information_set_id"]),
            public_belief_fingerprint=str(source["public_belief_fingerprint"]),
            public_history_fingerprint=str(source["public_history_fingerprint"]),
            private_observation_fingerprint=str(source["private_observation_fingerprint"]),
            particle_id=str(particle["particle_id"]),
            particle_fingerprint=str(particle["particle_fingerprint"]),
            candidate_id_supervision_only=str(particle["candidate_id_supervision_only"]),
            split=str(particle["split"]),
            variant=int(particle["variant"]),
            acting_player=int(target["acting_player"]),
            value_player=int(target["value_player"]),
            predictions=tuple((name, float(predictions[name])) for name in LEAF_MODEL_NAMES),
            terminal_returns=tuple(float(value) for value in target["terminal_returns"]),
            empirical_return=float(target["empirical_return"]),
            rollout_standard_error=float(target["rollout_standard_error"]),
            rollout_macro_transitions=int(target["rollout_macro_transitions"]),
            terminal_native_depth_minimum=int(target["terminal_native_depth_minimum"]),
            terminal_native_depth_maximum=int(target["terminal_native_depth_maximum"]),
            public_features=tuple(
                (str(name), float(value)) for name, value in features["public"]
            ),
            private_features=tuple(
                (str(name), float(value))
                for name, value in features["actor_private_hashed"]
            ),
            source_descendant_official_engine_executed=bool(
                source["source_descendant_official_engine_executed"]
            ),
            target_exact_counterfactual_value=bool(target["target_exact_counterfactual_value"]),
            selected_hidden_truth_identifier_input=bool(
                particle["selected_hidden_truth_identifier_input"]
            ),
            raw_private_card_ids_persisted=bool(features["raw_private_card_ids_persisted"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("leaf-model example fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativeLeafModelDataset:
    runtime_fingerprint: str
    deck_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    source_trajectory_model_fingerprint: str
    source_replay_dataset_fingerprint: str
    range_provider_fingerprint: str
    candidate_ids: tuple[str, ...]
    particle_family_fingerprint: str
    maximum_source_depth_delta: int
    rollouts_per_leaf: int
    native_transition_count: int
    macro_transition_count: int
    tactical_transition_count: int
    examples: tuple[NativeLeafModelExample, ...]
    allocation_heldout: bool = True
    exact_counterfactual_targets: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
            "source_trajectory_model_fingerprint",
            "source_replay_dataset_fingerprint",
            "range_provider_fingerprint",
            "particle_family_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("leaf-model candidate IDs must be unique")
        if min(self.maximum_source_depth_delta, self.rollouts_per_leaf) <= 0:
            raise ValueError("leaf-model limits must be positive")
        if min(
            self.native_transition_count,
            self.macro_transition_count,
            self.tactical_transition_count,
        ) < 0 or not self.examples:
            raise ValueError("leaf-model dataset accounting is invalid")
        if any(
            item.candidate_id_supervision_only not in self.candidate_ids
            for item in self.examples
        ):
            raise ValueError("leaf-model example candidate is outside support")
        if not self.allocation_heldout or self.exact_counterfactual_targets:
            raise ValueError("leaf-model dataset scope was promoted incorrectly")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("leaf-model dataset cannot persist raw private IDs")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_MODEL_SCHEMA_VERSION,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "source_plan_blueprint_fingerprint": self.source_plan_blueprint_fingerprint,
            "source_trajectory_model_fingerprint": self.source_trajectory_model_fingerprint,
            "source_replay_dataset_fingerprint": self.source_replay_dataset_fingerprint,
            "range_provider_fingerprint": self.range_provider_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "particle_family_fingerprint": self.particle_family_fingerprint,
            "maximum_source_depth_delta": self.maximum_source_depth_delta,
            "rollouts_per_leaf": self.rollouts_per_leaf,
            "accounting": {
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "tactical_transition_count": self.tactical_transition_count,
            },
            "examples": [item.to_dict() for item in self.examples],
            "claims": {
                "allocation_heldout": self.allocation_heldout,
                "exact_counterfactual_targets": self.exact_counterfactual_targets,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeLeafModelDataset":
        accounting = raw["accounting"]
        claims = raw["claims"]
        result = cls(
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(raw["source_plan_blueprint_fingerprint"]),
            source_trajectory_model_fingerprint=str(
                raw["source_trajectory_model_fingerprint"]
            ),
            source_replay_dataset_fingerprint=str(raw["source_replay_dataset_fingerprint"]),
            range_provider_fingerprint=str(raw["range_provider_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            particle_family_fingerprint=str(raw["particle_family_fingerprint"]),
            maximum_source_depth_delta=int(raw["maximum_source_depth_delta"]),
            rollouts_per_leaf=int(raw["rollouts_per_leaf"]),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            tactical_transition_count=int(accounting["tactical_transition_count"]),
            examples=tuple(NativeLeafModelExample.from_dict(item) for item in raw["examples"]),
            allocation_heldout=bool(claims["allocation_heldout"]),
            exact_counterfactual_targets=bool(claims["exact_counterfactual_targets"]),
            raw_private_card_ids_persisted=int(claims["raw_private_card_ids_persisted"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("leaf-model dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class LeafModelMetrics:
    model_name: str
    calibration_mean_absolute_error: float
    calibration_root_mean_square_error: float
    validation_mean_absolute_error: float
    validation_root_mean_square_error: float

    def __post_init__(self) -> None:
        if self.model_name not in LEAF_MODEL_NAMES:
            raise ValueError("leaf-model metric name is invalid")
        for field_name in (
            "calibration_mean_absolute_error",
            "calibration_root_mean_square_error",
            "validation_mean_absolute_error",
            "validation_root_mean_square_error",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("leaf-model error metric is invalid")


@dataclass(frozen=True, slots=True)
class NativeLeafModelCalibration:
    dataset_fingerprint: str
    config: NativeLeafModelConfig
    selected_model: str
    selection_split: str
    conformal_rank: int
    conformal_radius: float
    calibration_coverage: float
    validation_coverage: float
    validation_mean_absolute_error: float
    validation_root_mean_square_error: float
    mean_interval_width: float
    informative_interval_fraction: float
    maximum_rollout_standard_error: float
    model_metrics: tuple[LeafModelMetrics, ...]
    empirical_fixed_profile_leaf_gate_passed: bool
    allocation_heldout_validation_passed: bool
    counterfactual_value_gate_passed: bool = False
    public_belief_gate_passed: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        if self.selected_model not in LEAF_MODEL_NAMES or self.selection_split != "calibration":
            raise ValueError("leaf-model selection metadata is invalid")
        if self.conformal_rank <= 0 or self.conformal_radius < 0:
            raise ValueError("leaf-model conformal metadata is invalid")
        if tuple(item.model_name for item in self.model_metrics) != LEAF_MODEL_NAMES:
            raise ValueError("leaf-model metrics have invalid support/order")
        for field_name in (
            "calibration_coverage",
            "validation_coverage",
            "informative_interval_fraction",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("leaf-model probability metric is invalid")
        if self.counterfactual_value_gate_passed or self.public_belief_gate_passed:
            raise ValueError("empirical leaf comparison cannot promote CF/belief gates")
        if self.allocation_heldout_validation_passed != self.empirical_fixed_profile_leaf_gate_passed:
            raise ValueError("leaf-model heldout/gate flags disagree")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_MODEL_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config": asdict(self.config),
            "selection": {
                "selected_model": self.selected_model,
                "selection_split": self.selection_split,
            },
            "calibration": {
                "conformal_rank": self.conformal_rank,
                "conformal_radius": self.conformal_radius,
                "calibration_coverage": self.calibration_coverage,
                "validation_coverage": self.validation_coverage,
                "validation_mean_absolute_error": self.validation_mean_absolute_error,
                "validation_root_mean_square_error": self.validation_root_mean_square_error,
                "mean_interval_width": self.mean_interval_width,
                "informative_interval_fraction": self.informative_interval_fraction,
                "maximum_rollout_standard_error": self.maximum_rollout_standard_error,
                "model_metrics": [asdict(item) for item in self.model_metrics],
            },
            "gates": {
                "empirical_fixed_profile_leaf": self.empirical_fixed_profile_leaf_gate_passed,
                "allocation_heldout_validation": self.allocation_heldout_validation_passed,
                "counterfactual_value": self.counterfactual_value_gate_passed,
                "public_belief": self.public_belief_gate_passed,
            },
            "scope": {
                "exact_counterfactual_value": False,
                "fixed_profile_empirical_return": True,
                "safe_resolver_compatible": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeLeafModelCalibration":
        selection = raw["selection"]
        values = raw["calibration"]
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            config=NativeLeafModelConfig(**dict(raw["config"])),
            selected_model=str(selection["selected_model"]),
            selection_split=str(selection["selection_split"]),
            conformal_rank=int(values["conformal_rank"]),
            conformal_radius=float(values["conformal_radius"]),
            calibration_coverage=float(values["calibration_coverage"]),
            validation_coverage=float(values["validation_coverage"]),
            validation_mean_absolute_error=float(values["validation_mean_absolute_error"]),
            validation_root_mean_square_error=float(values["validation_root_mean_square_error"]),
            mean_interval_width=float(values["mean_interval_width"]),
            informative_interval_fraction=float(values["informative_interval_fraction"]),
            maximum_rollout_standard_error=float(values["maximum_rollout_standard_error"]),
            model_metrics=tuple(LeafModelMetrics(**dict(item)) for item in values["model_metrics"]),
            empirical_fixed_profile_leaf_gate_passed=bool(
                gates["empirical_fixed_profile_leaf"]
            ),
            allocation_heldout_validation_passed=bool(
                gates["allocation_heldout_validation"]
            ),
            counterfactual_value_gate_passed=bool(gates["counterfactual_value"]),
            public_belief_gate_passed=bool(gates["public_belief"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("leaf-model calibration fingerprint mismatch")
        return result


def calibrate_native_leaf_models(
    dataset: NativeLeafModelDataset,
    config: NativeLeafModelConfig | None = None,
) -> NativeLeafModelCalibration:
    config = config or NativeLeafModelConfig()
    calibration = tuple(item for item in dataset.examples if item.split == "calibration")
    validation = tuple(item for item in dataset.examples if item.split == "validation")
    if len(calibration) < config.minimum_calibration_examples:
        raise ValueError("leaf-model calibration split is too small")
    if len(validation) < config.minimum_validation_examples:
        raise ValueError("leaf-model validation split is too small")
    if dataset.rollouts_per_leaf < config.minimum_rollouts_per_leaf:
        raise ValueError("leaf-model rollout count is below the precision gate")

    def errors(examples: Sequence[NativeLeafModelExample], model: str) -> tuple[float, ...]:
        return tuple(abs(item.prediction(model) - item.empirical_return) for item in examples)

    metrics = []
    for model in LEAF_MODEL_NAMES:
        calibration_errors = errors(calibration, model)
        validation_errors = errors(validation, model)
        metrics.append(
            LeafModelMetrics(
                model_name=model,
                calibration_mean_absolute_error=statistics.mean(calibration_errors),
                calibration_root_mean_square_error=math.sqrt(
                    statistics.mean(value * value for value in calibration_errors)
                ),
                validation_mean_absolute_error=statistics.mean(validation_errors),
                validation_root_mean_square_error=math.sqrt(
                    statistics.mean(value * value for value in validation_errors)
                ),
            )
        )
    selected = min(
        metrics,
        key=lambda item: (
            item.calibration_root_mean_square_error,
            item.calibration_mean_absolute_error,
            LEAF_MODEL_NAMES.index(item.model_name),
        ),
    ).model_name
    calibration_errors = sorted(errors(calibration, selected))
    rank = min(
        len(calibration_errors),
        max(1, math.ceil((len(calibration_errors) + 1) * config.target_coverage)),
    )
    radius = calibration_errors[rank - 1] + config.numerical_margin

    def interval(item: NativeLeafModelExample) -> tuple[float, float]:
        center = item.prediction(selected)
        return max(-1.0, center - radius), min(1.0, center + radius)

    def covered(item: NativeLeafModelExample) -> bool:
        lower, upper = interval(item)
        return lower - 1e-12 <= item.empirical_return <= upper + 1e-12

    widths = [interval(item)[1] - interval(item)[0] for item in validation]
    validation_errors = errors(validation, selected)
    candidate_strata = {
        item.candidate_id_supervision_only for item in validation
    }
    maximum_standard_error = max(
        item.rollout_standard_error for item in dataset.examples
    )
    calibration_coverage = sum(covered(item) for item in calibration) / len(calibration)
    validation_coverage = sum(covered(item) for item in validation) / len(validation)
    gate = all(
        (
            len(candidate_strata) >= config.minimum_candidate_strata,
            validation_coverage >= config.target_coverage,
            statistics.mean(widths) <= config.maximum_mean_interval_width,
            maximum_standard_error <= config.maximum_rollout_standard_error,
        )
    )
    return NativeLeafModelCalibration(
        dataset_fingerprint=dataset.fingerprint,
        config=config,
        selected_model=selected,
        selection_split="calibration",
        conformal_rank=rank,
        conformal_radius=radius,
        calibration_coverage=calibration_coverage,
        validation_coverage=validation_coverage,
        validation_mean_absolute_error=statistics.mean(validation_errors),
        validation_root_mean_square_error=math.sqrt(
            statistics.mean(value * value for value in validation_errors)
        ),
        mean_interval_width=statistics.mean(widths),
        informative_interval_fraction=sum(value < 2.0 - 1e-12 for value in widths)
        / len(widths),
        maximum_rollout_standard_error=maximum_standard_error,
        model_metrics=tuple(metrics),
        empirical_fixed_profile_leaf_gate_passed=gate,
        allocation_heldout_validation_passed=gate,
    )


@dataclass(frozen=True, slots=True)
class Phase6NativeLeafModelEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    calibration_fingerprint: str
    example_count: int
    terminal_rollout_count: int
    candidate_strata: int
    calibration_particles: int
    validation_particles: int
    native_transition_count: int
    selected_model: str
    conformal_radius: float
    validation_coverage: float
    validation_mean_absolute_error: float
    mean_interval_width: float
    informative_interval_fraction: float
    maximum_rollout_standard_error: float
    empirical_fixed_profile_leaf_gate_passed: bool
    allocation_heldout: bool
    counterfactual_values_calibrated: bool
    public_belief_calibrated: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    raw_private_card_ids_persisted: int
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in ("dataset_fingerprint", "calibration_fingerprint"):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.benchmark_id or not self.status or self.selected_model not in LEAF_MODEL_NAMES:
            raise ValueError("Phase 6U evidence identity/selection is invalid")
        if min(
            self.example_count,
            self.terminal_rollout_count,
            self.candidate_strata,
            self.calibration_particles,
            self.validation_particles,
        ) <= 0 or self.native_transition_count < 0:
            raise ValueError("Phase 6U evidence accounting is invalid")
        if not self.allocation_heldout:
            raise ValueError("Phase 6U evidence requires held-out allocations")
        if any(
            (
                self.counterfactual_values_calibrated,
                self.public_belief_calibrated,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6U empirical diagnostic cannot enable safety")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("Phase 6U cannot persist raw private IDs")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_MODEL_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6U low-noise native leaf evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Allocation-heldout examples | {self.example_count} |
| Genuine terminal continuations | {self.terminal_rollout_count} |
| Genuine native transitions | {self.native_transition_count} |
| Calibration-selected model | `{self.selected_model}` |
| Conformal radius | {self.conformal_radius:.6f} |
| Validation coverage | {self.validation_coverage:.3%} |
| Validation MAE | {self.validation_mean_absolute_error:.6f} |
| Mean interval width | {self.mean_interval_width:.6f} |
| Informative interval fraction | {self.informative_interval_fraction:.3%} |
| Maximum rollout standard error | {self.maximum_rollout_standard_error:.6f} |
| Empirical fixed-profile leaf gate | **{str(self.empirical_fixed_profile_leaf_gate_passed).lower()}** |
| Counterfactual values calibrated | **false** |

The ten-rollout targets improve sampling precision, but they remain returns of
one fixed abstract plan/tactical profile. Even a passing empirical gate cannot
be used as an opponent counterfactual-value certificate. No resolver,
exploitation layer, official strategy control, or Kaggle submission is enabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6NativeLeafModelEvidence":
        fields = {
            field_name: raw[field_name]
            for field_name in cls.__dataclass_fields__
            if field_name != "claims"
        }
        for field_name in (
            "example_count",
            "terminal_rollout_count",
            "candidate_strata",
            "calibration_particles",
            "validation_particles",
            "native_transition_count",
            "raw_private_card_ids_persisted",
        ):
            fields[field_name] = int(fields[field_name])
        for field_name in (
            "conformal_radius",
            "validation_coverage",
            "validation_mean_absolute_error",
            "mean_interval_width",
            "informative_interval_fraction",
            "maximum_rollout_standard_error",
        ):
            fields[field_name] = float(fields[field_name])
        for field_name in (
            "empirical_fixed_profile_leaf_gate_passed",
            "allocation_heldout",
            "counterfactual_values_calibrated",
            "public_belief_calibrated",
            "safe_resolver_enabled",
            "official_strategy_control_enabled",
        ):
            fields[field_name] = bool(fields[field_name])
        result = cls(
            **fields,
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6U evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativeLeafModelArtifacts:
    dataset: NativeLeafModelDataset
    calibration: NativeLeafModelCalibration
    evidence: Phase6NativeLeafModelEvidence


def _allocation_particles(
    observation: Mapping[str, Any],
    *,
    target: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    basic_card_ids: Sequence[int],
    variants: int,
    seed: int,
) -> tuple[tuple[HiddenZoneCompletion, ...], dict[str, tuple[str, int, str]]]:
    if variants < 4:
        raise ValueError("Phase 6U requires at least four allocation variants")
    probability = 1.0 / (len(candidates) * variants)
    particles = []
    metadata: dict[str, tuple[str, int, str]] = {}
    for candidate in candidates:
        for variant in range(variants):
            split = "validation" if variant == variants - 1 else "calibration"
            identifier = f"{candidate.archetype}:leaf-allocation-{variant}:{split}"
            particle = sample_hidden_zone_completion(
                observation,
                hypothesis_id=identifier,
                probability=probability,
                allocation_seed=_seed(seed, candidate.fingerprint, variant),
                your_candidate_id=target.archetype,
                opponent_candidate_id=candidate.archetype,
                your_archetype_id=target.archetype,
                opponent_archetype_id=candidate.archetype,
                your_style_id="unknown",
                opponent_style_id="unknown",
                your_card_ids=target.card_ids,
                opponent_card_ids=candidate.card_ids,
                your_deck_fingerprint=target.fingerprint,
                opponent_deck_fingerprint=candidate.fingerprint,
                basic_card_ids=basic_card_ids,
                source=(
                    "Phase 6U seeded public-consistent hidden allocation; "
                    "fixed-profile leaf research particle"
                ),
            )
            particles.append(particle)
            metadata[identifier] = (candidate.archetype, variant, split)
    if len({item.fingerprint for item in particles}) != len(particles):
        raise RuntimeError("Phase 6U allocation particles are not distinct")
    return tuple(particles), metadata


def run_phase6_native_leaf_model(
    *,
    project_root: str | Path | None = None,
    source_tactical_blueprint_path: str | Path | None = None,
    source_plan_blueprint_path: str | Path | None = None,
    source_phase6p_evidence_path: str | Path | None = None,
    source_replay_dataset_path: str | Path | None = None,
    source_trajectory_model_path: str | Path | None = None,
    maximum_source_depth_delta: int = 2,
    particle_variants: int = 4,
    rollouts_per_leaf: int = 10,
    maximum_macro_transitions: int = 512,
    maximum_tactical_transitions: int = 512,
    config: NativeLeafModelConfig | None = None,
    seed: int = PHASE6_NATIVE_LEAF_MODEL_SEED,
) -> Phase6NativeLeafModelArtifacts:
    if min(
        maximum_source_depth_delta,
        particle_variants,
        rollouts_per_leaf,
        maximum_macro_transitions,
        maximum_tactical_transitions,
    ) <= 0:
        raise ValueError("Phase 6U limits must be positive")
    if particle_variants < 4 or rollouts_per_leaf < 2:
        raise ValueError("Phase 6U requires disjoint allocations and repeated rollouts")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    tactical_path = Path(
        source_tactical_blueprint_path
        or root / "artifacts" / "phase6_native_blueprint" / "native_regret_blueprint.json"
    ).resolve()
    plan_path = Path(
        source_plan_blueprint_path
        or root
        / "artifacts"
        / "phase6_native_plan_iterations"
        / "iterative_native_plan_blueprint.json"
    ).resolve()
    phase6p_path = Path(
        source_phase6p_evidence_path
        or root / "artifacts" / "phase6_native_plan_iterations" / "phase6p_evidence.json"
    ).resolve()
    replay_path = Path(
        source_replay_dataset_path
        or root
        / "artifacts"
        / "phase6_official_trajectory"
        / "official_trajectory_dataset.json"
    ).resolve()
    trajectory_path = Path(
        source_trajectory_model_path
        or root
        / "artifacts"
        / "phase6_trajectory_neural"
        / "logged_trajectory_bootstrap.json"
    ).resolve()
    tactical_artifact = NativeRegretBlueprintArtifact.from_dict(
        json.loads(tactical_path.read_text(encoding="utf-8"))
    )
    tactical_provider = restore_native_regret_blueprint(tactical_artifact)
    plan_artifact = NativePlanBlueprintArtifact.from_dict(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    plan_provider = restore_native_plan_blueprint(plan_artifact)
    phase6p = Phase6NativePlanIterationsEvidence.from_dict(
        json.loads(phase6p_path.read_text(encoding="utf-8"))
    )
    replay_dataset = OfficialTrajectoryDataset.from_dict(
        json.loads(replay_path.read_text(encoding="utf-8"))
    )
    trajectory_artifact = TrajectoryBootstrapArtifact.from_dict(
        json.loads(trajectory_path.read_text(encoding="utf-8"))
    )
    trajectory_provider = restore_logged_trajectory_bootstrap(trajectory_artifact)
    range_provider = LoggedTrajectoryNativePublicRangeProvider(
        trajectory_provider,
        replay_dataset.candidates,
    )
    if phase6p.final_model_fingerprint != plan_artifact.fingerprint:
        raise ValueError("Phase 6U plan model/evidence mismatch")
    if plan_artifact.source_tactical_blueprint_fingerprint != tactical_artifact.fingerprint:
        raise ValueError("Phase 6U plan/tactical provenance mismatch")
    if trajectory_artifact.dataset_fingerprint != replay_dataset.fingerprint:
        raise ValueError("Phase 6U trajectory model/dataset mismatch")
    bridge = NativeBlueprintFeatureBridge()
    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_start  # type: ignore[import-not-found]

    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    if set(trajectory_artifact.candidate_ids) != set(profile_by_id):
        raise ValueError("Phase 6U candidate catalog/native profiles mismatch")
    candidates = tuple(profile_by_id[name] for name in trajectory_artifact.candidate_ids)
    if tactical_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6U tactical blueprint/deck mismatch")
    if plan_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6U plan blueprint/deck mismatch")
    observation, start = battle_start(list(target.card_ids), list(candidates[0].card_ids))
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6U decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        particles, metadata = _allocation_particles(
            observation,
            target=target,
            candidates=candidates,
            basic_card_ids=_runtime_basics(runtime),
            variants=particle_variants,
            seed=seed,
        )
        particle_family_fingerprint = _fingerprint(
            {
                "particles": [item.fingerprint for item in particles],
                "metadata": metadata,
                "allocation_heldout": True,
                "order_only": False,
            }
        )
        with OfficialNativeSearchGame(
            observation,
            particles,
            runtime_directory=runtime,
        ) as native_game:
            examples = []
            macro_games = []
            for particle_index, outcome in enumerate(
                native_game.chance_outcomes(native_game.root_state())
            ):
                scenario = native_game.transition(
                    native_game.root_state(),
                    None,
                    chance_outcome=outcome,
                )
                completion = native_game.completion(scenario.hypothesis_id or "")
                candidate_id, variant, split = metadata[completion.hypothesis_id]
                macro = OfficialNativePlanGame(
                    native_game,
                    scenario,
                    bridge=bridge,
                    tactical_provider=tactical_provider,
                    maximum_tactical_transitions=maximum_tactical_transitions,
                )
                macro_games.append(macro)
                phase6k_leaf = UncalibratedNativePlanLeafValueProvider(macro)
                wrapper = DepthLimitedInformationSetGame(
                    macro,
                    phase6k_leaf,
                    maximum_source_depth_delta=maximum_source_depth_delta,
                )
                cutoff = _sample_cutoff_state(
                    wrapper,
                    macro,
                    plan_provider,
                    rng=random.Random(_seed(seed, "cutoff", particle_index)),
                )
                source_state = cutoff.source_state
                if not isinstance(source_state, NativePlanGameState):
                    raise RuntimeError("Phase 6U cutoff did not retain a native plan state")
                acting_player = macro.player_to_act(source_state)
                if acting_player not in (0, 1):
                    raise RuntimeError("Phase 6U cutoff lacks an acting player")
                information_set = macro.information_set(source_state, player=acting_player)
                belief = macro.public_belief(source_state, player=acting_player)
                phase6k_prediction = phase6k_leaf.estimate(
                    macro,
                    source_state,
                    player=0,
                ).value
                native_view = bridge.view(native_game, source_state.native_state)
                range_prediction = range_provider.predict(
                    source_state.native_state.observation,
                    native_view.public_belief,
                )
                trajectory_prediction = trajectory_provider.predict(
                    public_features=range_prediction.model_input_public_features,
                    private_features=native_view.information_set.private_features,
                    legal_actions=native_view.information_set.legal_actions,
                ).outcome_value
                if acting_player != 0:
                    trajectory_prediction = -trajectory_prediction
                predictions = (
                    ("phase6i_logged_value", max(-1.0, min(1.0, trajectory_prediction))),
                    ("phase6k_tactical_value", max(-1.0, min(1.0, phase6k_prediction))),
                    ("zero_baseline", 0.0),
                )
                returns = []
                depths = []
                rollout_transitions = 0
                for rollout in range(rollouts_per_leaf):
                    value, transitions, terminal_depth = _rollout_to_terminal(
                        macro,
                        source_state,
                        plan_provider,
                        value_player=0,
                        rng=random.Random(_seed(seed, "rollout", particle_index, rollout)),
                        maximum_macro_transitions=maximum_macro_transitions,
                    )
                    returns.append(value)
                    depths.append(terminal_depth)
                    rollout_transitions += transitions
                empirical = statistics.mean(returns)
                standard_error = statistics.stdev(returns) / math.sqrt(len(returns))
                example_id = _fingerprint(
                    {
                        "state": source_state.state_id,
                        "particle": completion.fingerprint,
                        "predictions": predictions,
                        "rollout_seeds": [
                            _seed(seed, "rollout", particle_index, rollout)
                            for rollout in range(rollouts_per_leaf)
                        ],
                    }
                )
                examples.append(
                    NativeLeafModelExample(
                        example_id=example_id,
                        source_game_fingerprint=macro.fingerprint,
                        source_state_id=source_state.state_id,
                        information_set_id=information_set.info_set_id,
                        public_belief_fingerprint=belief.fingerprint,
                        public_history_fingerprint=belief.public_history_fingerprint,
                        private_observation_fingerprint=(
                            information_set.private_observation_fingerprint or ""
                        ),
                        particle_id=completion.hypothesis_id,
                        particle_fingerprint=completion.fingerprint,
                        candidate_id_supervision_only=candidate_id,
                        split=split,
                        variant=variant,
                        acting_player=acting_player,
                        value_player=0,
                        predictions=predictions,
                        terminal_returns=tuple(returns),
                        empirical_return=empirical,
                        rollout_standard_error=standard_error,
                        rollout_macro_transitions=rollout_transitions,
                        terminal_native_depth_minimum=min(depths),
                        terminal_native_depth_maximum=max(depths),
                        public_features=range_prediction.model_input_public_features,
                        private_features=native_view.information_set.private_features,
                    )
                )
            provisional = NativeLeafModelDataset(
                runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                deck_fingerprint=target.fingerprint,
                source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
                source_plan_blueprint_fingerprint=plan_artifact.fingerprint,
                source_trajectory_model_fingerprint=trajectory_artifact.fingerprint,
                source_replay_dataset_fingerprint=replay_dataset.fingerprint,
                range_provider_fingerprint=range_provider.fingerprint,
                candidate_ids=trajectory_artifact.candidate_ids,
                particle_family_fingerprint=particle_family_fingerprint,
                maximum_source_depth_delta=maximum_source_depth_delta,
                rollouts_per_leaf=rollouts_per_leaf,
                native_transition_count=native_game.transition_count,
                macro_transition_count=sum(game.macro_transition_count for game in macro_games),
                tactical_transition_count=sum(
                    game.tactical_transition_count for game in macro_games
                ),
                examples=tuple(examples),
            )
            serialized = provisional.to_dict()
            raw_private = _raw_private_key_count(serialized)
            if raw_private != 0:
                raise RuntimeError("Phase 6U serialized a forbidden raw private key")
            dataset = provisional
            restored = NativeLeafModelDataset.from_dict(json.loads(json.dumps(serialized)))
            if restored.fingerprint != dataset.fingerprint:
                raise RuntimeError("Phase 6U dataset roundtrip drifted")
            calibration = calibrate_native_leaf_models(dataset, config)
            calibration_particles = len(
                {item.particle_id for item in dataset.examples if item.split == "calibration"}
            )
            validation_particles = len(
                {item.particle_id for item in dataset.examples if item.split == "validation"}
            )
            claims = (
                ("source_terminal_continuations", "genuine_official_search_execution"),
                ("cutoff_and_fixed_profile", "research_abstraction"),
                ("hidden_allocations", "heldout_supplied_particles"),
                ("selected_model", calibration.selected_model),
                ("value_target", "fixed_profile_empirical_return_not_counterfactual"),
                (
                    "empirical_fixed_profile_leaf_gate",
                    str(calibration.empirical_fixed_profile_leaf_gate_passed).lower(),
                ),
                ("counterfactual_value_bounds", "not_built"),
                ("safe_resolver", "not_enabled"),
                ("official_strategy_control", "disabled"),
                ("pokemon_strength", "not_claimed"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            evidence = Phase6NativeLeafModelEvidence(
                benchmark_id="phase6u-native-low-noise-leaf-v1",
                status="phase6u_native_low_noise_leaf_diagnostic_validated_partial",
                dataset_fingerprint=dataset.fingerprint,
                calibration_fingerprint=calibration.fingerprint,
                example_count=len(dataset.examples),
                terminal_rollout_count=len(dataset.examples) * rollouts_per_leaf,
                candidate_strata=len(dataset.candidate_ids),
                calibration_particles=calibration_particles,
                validation_particles=validation_particles,
                native_transition_count=dataset.native_transition_count,
                selected_model=calibration.selected_model,
                conformal_radius=calibration.conformal_radius,
                validation_coverage=calibration.validation_coverage,
                validation_mean_absolute_error=(
                    calibration.validation_mean_absolute_error
                ),
                mean_interval_width=calibration.mean_interval_width,
                informative_interval_fraction=(
                    calibration.informative_interval_fraction
                ),
                maximum_rollout_standard_error=(
                    calibration.maximum_rollout_standard_error
                ),
                empirical_fixed_profile_leaf_gate_passed=(
                    calibration.empirical_fixed_profile_leaf_gate_passed
                ),
                allocation_heldout=True,
                counterfactual_values_calibrated=False,
                public_belief_calibrated=False,
                safe_resolver_enabled=False,
                official_strategy_control_enabled=False,
                raw_private_card_ids_persisted=raw_private,
                claims=claims,
            )
            return Phase6NativeLeafModelArtifacts(dataset, calibration, evidence)
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_leaf_model().evidence.to_json())


if __name__ == "__main__":
    main()
