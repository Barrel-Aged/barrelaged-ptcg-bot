"""Phase 6S: held-out native continuation and leaf-calibration diagnostic.

Phase 6R controlled outcome-sampling horizon variance but intentionally used an
uncalibrated Phase 6K scalar at cutoff leaves.  This module builds the next
evidence layer: unseen within-zone hidden-order particles, abstract cutoff
states, repeated genuine native terminal continuations, and disjoint
calibration/validation residual reporting.

The target is the empirical return of a fixed stochastic plan/tactical profile,
not an exact counterfactual value.  Card allocation between hidden zones is not
held out, the public range remains a static supplied prior, and only a few
rollouts are observed per leaf.  Consequently every safety/deployment
capability remains disabled regardless of empirical interval coverage.
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
from .depth_limited_game import DepthLimitedGameState, DepthLimitedInformationSetGame
from .game_theory import InformationSetActor
from .native_blueprint_features import (
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeBlueprintFeatureBridge,
)
from .native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanBlueprintProvider,
    restore_native_plan_blueprint,
)
from .native_plan_game import NativePlanGameState, OfficialNativePlanGame
from .native_plan_leaf import UncalibratedNativePlanLeafValueProvider
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
    permute_hidden_zone_completion,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _completions, _runtime_basics
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence


PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION = 1
PHASE6_NATIVE_LEAF_CALIBRATION_SEED = 6_025


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


def _finite(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _sample_identifier(
    probabilities: Sequence[tuple[str, float]], rng: random.Random
) -> str:
    if not probabilities:
        raise ValueError("cannot sample an empty distribution")
    threshold = rng.random()
    cumulative = 0.0
    for identifier, probability in probabilities:
        cumulative += probability
        if threshold <= cumulative + 1e-15:
            return identifier
    return probabilities[-1][0]


@dataclass(frozen=True, slots=True)
class NativeLeafCalibrationConfig:
    target_coverage: float = 0.90
    minimum_calibration_examples: int = 12
    minimum_validation_examples: int = 6
    minimum_candidate_strata: int = 6
    minimum_rollouts_per_leaf: int = 3
    maximum_mean_interval_width: float = 1.25
    maximum_rollout_standard_error: float = 0.35
    numerical_margin: float = 1e-9

    def __post_init__(self) -> None:
        if not 0 < self.target_coverage <= 1 or not math.isfinite(
            self.target_coverage
        ):
            raise ValueError("native leaf target coverage must be in (0, 1]")
        if min(
            self.minimum_calibration_examples,
            self.minimum_validation_examples,
            self.minimum_candidate_strata,
            self.minimum_rollouts_per_leaf,
        ) <= 0:
            raise ValueError("native leaf calibration minimums must be positive")
        for field_name in (
            "maximum_mean_interval_width",
            "maximum_rollout_standard_error",
            "numerical_margin",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True, slots=True)
class NativeLeafContinuationExample:
    example_id: str
    source_game_fingerprint: str
    source_state_id: str
    information_set_id: str
    public_belief_fingerprint: str
    public_history_fingerprint: str
    private_observation_fingerprint: str
    particle_id: str
    particle_candidate_id: str
    particle_split: str
    particle_variant: int
    acting_player: int
    value_player: int
    leaf_prediction: float
    terminal_returns: tuple[float, ...]
    empirical_return: float
    absolute_error: float
    rollout_standard_error: float
    rollout_macro_transitions: int
    terminal_native_depth_minimum: int
    terminal_native_depth_maximum: int
    range_truth_candidate_id: str
    range_distribution: tuple[tuple[str, float], ...]
    public_features: tuple[tuple[str, float], ...]
    private_features: tuple[tuple[str, float], ...]
    source_descendant_official_engine_executed: bool = True
    target_exact_counterfactual_value: bool = False
    selected_hidden_truth_in_policy_features: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "example_id",
            "source_game_fingerprint",
            "source_state_id",
            "public_belief_fingerprint",
            "public_history_fingerprint",
            "private_observation_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not all(
            (
                self.information_set_id,
                self.particle_id,
                self.particle_candidate_id,
                self.range_truth_candidate_id,
            )
        ):
            raise ValueError("native leaf continuation identities are required")
        if self.particle_split not in {"calibration", "validation"}:
            raise ValueError("native leaf particle split is invalid")
        if self.particle_variant < 0 or self.acting_player not in (0, 1):
            raise ValueError("native leaf particle/actor metadata is invalid")
        if self.value_player not in (0, 1):
            raise ValueError("native leaf value player is invalid")
        if len(self.terminal_returns) < 2:
            raise ValueError("native leaf continuation requires repeated rollouts")
        for value in (self.leaf_prediction, self.empirical_return) + self.terminal_returns:
            if not -1.0 <= _finite(value, "native leaf utility") <= 1.0:
                raise ValueError("native leaf utilities must be in [-1, 1]")
        if not math.isclose(
            self.empirical_return,
            statistics.mean(self.terminal_returns),
            abs_tol=1e-12,
        ):
            raise ValueError("native leaf empirical return must equal rollout mean")
        if not math.isclose(
            self.absolute_error,
            abs(self.leaf_prediction - self.empirical_return),
            abs_tol=1e-12,
        ):
            raise ValueError("native leaf absolute error is inconsistent")
        if self.rollout_standard_error < 0 or not math.isfinite(
            self.rollout_standard_error
        ):
            raise ValueError("native leaf standard error is invalid")
        if min(
            self.rollout_macro_transitions,
            self.terminal_native_depth_minimum,
            self.terminal_native_depth_maximum,
        ) <= 0 or self.terminal_native_depth_minimum > self.terminal_native_depth_maximum:
            raise ValueError("native leaf rollout depth/accounting is invalid")
        identifiers = tuple(identifier for identifier, _ in self.range_distribution)
        probabilities = tuple(float(value) for _, value in self.range_distribution)
        if (
            not identifiers
            or len(identifiers) != len(set(identifiers))
            or self.range_truth_candidate_id not in identifiers
            or any(value < 0 or not math.isfinite(value) for value in probabilities)
            or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("native leaf public range distribution is invalid")
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("native leaf public feature schema mismatch")
        if tuple(name for name, _ in self.private_features) != NATIVE_PRIVATE_FEATURE_NAMES:
            raise ValueError("native leaf private feature schema mismatch")
        if any(
            (
                not self.source_descendant_official_engine_executed,
                self.target_exact_counterfactual_value,
                self.selected_hidden_truth_in_policy_features,
            )
        ):
            raise ValueError("native leaf example promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def range_truth_probability(self) -> float:
        return dict(self.range_distribution)[self.range_truth_candidate_id]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION,
            "example_id": self.example_id,
            "source": {
                "game_fingerprint": self.source_game_fingerprint,
                "state_id": self.source_state_id,
                "information_set_id": self.information_set_id,
                "public_belief_fingerprint": self.public_belief_fingerprint,
                "public_history_fingerprint": self.public_history_fingerprint,
                "private_observation_fingerprint": (
                    self.private_observation_fingerprint
                ),
                "source_descendant_official_engine_executed": (
                    self.source_descendant_official_engine_executed
                ),
            },
            "particle": {
                "particle_id": self.particle_id,
                "candidate_id": self.particle_candidate_id,
                "split": self.particle_split,
                "variant": self.particle_variant,
                "within_zone_order_held_out": True,
                "hidden_zone_allocation_held_out": False,
                "raw_private_card_ids_persisted": False,
            },
            "value_target": {
                "acting_player": self.acting_player,
                "value_player": self.value_player,
                "leaf_prediction": self.leaf_prediction,
                "terminal_returns": list(self.terminal_returns),
                "empirical_return": self.empirical_return,
                "absolute_error": self.absolute_error,
                "rollout_standard_error": self.rollout_standard_error,
                "rollout_macro_transitions": self.rollout_macro_transitions,
                "terminal_native_depth_minimum": (
                    self.terminal_native_depth_minimum
                ),
                "terminal_native_depth_maximum": (
                    self.terminal_native_depth_maximum
                ),
                "target_exact_counterfactual_value": (
                    self.target_exact_counterfactual_value
                ),
            },
            "public_range_evaluation": {
                "observer_player": 0,
                "evaluation_root": "static_supplied_completion_prior",
                "truth_candidate_id": self.range_truth_candidate_id,
                "distribution": [list(item) for item in self.range_distribution],
                "truth_probability": self.range_truth_probability,
                "truth_is_evaluation_only": True,
                "selected_hidden_truth_in_policy_features": (
                    self.selected_hidden_truth_in_policy_features
                ),
            },
            "policy_features": {
                "public": [list(item) for item in self.public_features],
                "private": [list(item) for item in self.private_features],
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeLeafContinuationExample":
        source = raw["source"]
        particle = raw["particle"]
        target = raw["value_target"]
        range_evaluation = raw["public_range_evaluation"]
        features = raw["policy_features"]
        result = cls(
            example_id=str(raw["example_id"]),
            source_game_fingerprint=str(source["game_fingerprint"]),
            source_state_id=str(source["state_id"]),
            information_set_id=str(source["information_set_id"]),
            public_belief_fingerprint=str(source["public_belief_fingerprint"]),
            public_history_fingerprint=str(source["public_history_fingerprint"]),
            private_observation_fingerprint=str(
                source["private_observation_fingerprint"]
            ),
            particle_id=str(particle["particle_id"]),
            particle_candidate_id=str(particle["candidate_id"]),
            particle_split=str(particle["split"]),
            particle_variant=int(particle["variant"]),
            acting_player=int(target["acting_player"]),
            value_player=int(target["value_player"]),
            leaf_prediction=float(target["leaf_prediction"]),
            terminal_returns=tuple(float(value) for value in target["terminal_returns"]),
            empirical_return=float(target["empirical_return"]),
            absolute_error=float(target["absolute_error"]),
            rollout_standard_error=float(target["rollout_standard_error"]),
            rollout_macro_transitions=int(target["rollout_macro_transitions"]),
            terminal_native_depth_minimum=int(
                target["terminal_native_depth_minimum"]
            ),
            terminal_native_depth_maximum=int(
                target["terminal_native_depth_maximum"]
            ),
            range_truth_candidate_id=str(range_evaluation["truth_candidate_id"]),
            range_distribution=tuple(
                (str(identifier), float(probability))
                for identifier, probability in range_evaluation["distribution"]
            ),
            public_features=tuple(
                (str(name), float(value)) for name, value in features["public"]
            ),
            private_features=tuple(
                (str(name), float(value)) for name, value in features["private"]
            ),
            source_descendant_official_engine_executed=bool(
                source["source_descendant_official_engine_executed"]
            ),
            target_exact_counterfactual_value=bool(
                target["target_exact_counterfactual_value"]
            ),
            selected_hidden_truth_in_policy_features=bool(
                range_evaluation["selected_hidden_truth_in_policy_features"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native leaf continuation fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativeLeafContinuationDataset:
    native_game_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    candidate_ids: tuple[str, ...]
    particle_family_fingerprint: str
    maximum_source_depth_delta: int
    rollouts_per_leaf: int
    examples: tuple[NativeLeafContinuationExample, ...]
    native_transition_count: int
    macro_transition_count: int
    tactical_transition_count: int
    source_descendant_official_engine_executed: bool = True
    dataset_official_engine_executed: bool = False
    supplied_public_range_calibrated: bool = False
    hidden_zone_order_heldout: bool = True
    hidden_zone_allocation_heldout: bool = False
    exact_counterfactual_targets: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "native_game_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
            "particle_family_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(
            self.candidate_ids
        ):
            raise ValueError("native leaf candidates must be unique and nonempty")
        if self.maximum_source_depth_delta <= 0 or self.rollouts_per_leaf < 2:
            raise ValueError("native leaf horizon/rollout count is invalid")
        if not self.examples or len({item.example_id for item in self.examples}) != len(
            self.examples
        ):
            raise ValueError("native leaf examples must be unique and nonempty")
        if any(len(item.terminal_returns) != self.rollouts_per_leaf for item in self.examples):
            raise ValueError("native leaf rollout counts disagree")
        split_candidates = {
            split: {item.particle_candidate_id for item in self.examples if item.particle_split == split}
            for split in ("calibration", "validation")
        }
        required = set(self.candidate_ids)
        if any(values != required for values in split_candidates.values()):
            raise ValueError("each native leaf split must cover every candidate stratum")
        particle_sets = {
            split: {item.particle_id for item in self.examples if item.particle_split == split}
            for split in ("calibration", "validation")
        }
        if not all(particle_sets.values()) or particle_sets["calibration"] & particle_sets["validation"]:
            raise ValueError("native leaf particle splits must be disjoint")
        if min(
            self.native_transition_count,
            self.macro_transition_count,
            self.tactical_transition_count,
        ) <= 0:
            raise ValueError("native leaf transition accounting must be positive")
        if any(
            (
                not self.source_descendant_official_engine_executed,
                self.dataset_official_engine_executed,
                self.supplied_public_range_calibrated,
                not self.hidden_zone_order_heldout,
                self.hidden_zone_allocation_heldout,
                self.exact_counterfactual_targets,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("native leaf dataset promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION,
            "native_game_fingerprint": self.native_game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "source_plan_blueprint_fingerprint": self.source_plan_blueprint_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "particle_family_fingerprint": self.particle_family_fingerprint,
            "maximum_source_depth_delta": self.maximum_source_depth_delta,
            "rollouts_per_leaf": self.rollouts_per_leaf,
            "examples": [item.to_dict() for item in self.examples],
            "accounting": {
                "examples": len(self.examples),
                "calibration_examples": sum(
                    item.particle_split == "calibration" for item in self.examples
                ),
                "validation_examples": sum(
                    item.particle_split == "validation" for item in self.examples
                ),
                "terminal_rollouts": len(self.examples) * self.rollouts_per_leaf,
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "tactical_transition_count": self.tactical_transition_count,
            },
            "claims": {
                "source_descendant_official_engine_executed": (
                    self.source_descendant_official_engine_executed
                ),
                "dataset_official_engine_executed": (
                    self.dataset_official_engine_executed
                ),
                "supplied_public_range_calibrated": (
                    self.supplied_public_range_calibrated
                ),
                "hidden_zone_order_heldout": self.hidden_zone_order_heldout,
                "hidden_zone_allocation_heldout": (
                    self.hidden_zone_allocation_heldout
                ),
                "exact_counterfactual_targets": self.exact_counterfactual_targets,
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeLeafContinuationDataset":
        if int(raw.get("schema_version", -1)) != PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported native leaf dataset schema")
        accounting = raw["accounting"]
        claims = raw["claims"]
        result = cls(
            native_game_fingerprint=str(raw["native_game_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(
                raw["source_plan_blueprint_fingerprint"]
            ),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            particle_family_fingerprint=str(raw["particle_family_fingerprint"]),
            maximum_source_depth_delta=int(raw["maximum_source_depth_delta"]),
            rollouts_per_leaf=int(raw["rollouts_per_leaf"]),
            examples=tuple(
                NativeLeafContinuationExample.from_dict(item)
                for item in raw["examples"]
            ),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            tactical_transition_count=int(accounting["tactical_transition_count"]),
            source_descendant_official_engine_executed=bool(
                claims["source_descendant_official_engine_executed"]
            ),
            dataset_official_engine_executed=bool(
                claims["dataset_official_engine_executed"]
            ),
            supplied_public_range_calibrated=bool(
                claims["supplied_public_range_calibrated"]
            ),
            hidden_zone_order_heldout=bool(claims["hidden_zone_order_heldout"]),
            hidden_zone_allocation_heldout=bool(
                claims["hidden_zone_allocation_heldout"]
            ),
            exact_counterfactual_targets=bool(
                claims["exact_counterfactual_targets"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        if int(accounting["examples"]) != len(result.examples):
            raise ValueError("native leaf serialized example count mismatch")
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native leaf dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativeLeafCandidateMetrics:
    candidate_id: str
    calibration_examples: int
    validation_examples: int
    validation_coverage: float
    validation_mean_absolute_error: float

    def __post_init__(self) -> None:
        if not self.candidate_id or min(
            self.calibration_examples, self.validation_examples
        ) <= 0:
            raise ValueError("native leaf candidate metric identity/count is invalid")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("native leaf candidate coverage is invalid")
        if self.validation_mean_absolute_error < 0 or not math.isfinite(
            self.validation_mean_absolute_error
        ):
            raise ValueError("native leaf candidate error is invalid")


@dataclass(frozen=True, slots=True)
class NativeLeafCalibrationArtifact:
    dataset_fingerprint: str
    source_plan_blueprint_fingerprint: str
    config: NativeLeafCalibrationConfig
    conformal_rank: int
    conformal_radius: float
    calibration_coverage: float
    validation_coverage: float
    maximum_validation_miss: float
    validation_mean_absolute_error: float
    validation_root_mean_square_error: float
    mean_interval_width: float
    informative_interval_fraction: float
    maximum_rollout_standard_error: float
    candidate_metrics: tuple[NativeLeafCandidateMetrics, ...]
    nontrivial_range_examples: int
    public_range_nll: float
    uniform_range_nll: float
    public_range_brier: float
    public_range_top1_accuracy: float
    empirical_rollout_leaf_gate_passed: bool
    counterfactual_value_gate_passed: bool = False
    public_belief_calibration_gate_passed: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        _validate_digest(
            self.source_plan_blueprint_fingerprint,
            "source_plan_blueprint_fingerprint",
        )
        if self.conformal_rank <= 0 or self.conformal_radius < 0:
            raise ValueError("native leaf conformal rank/radius is invalid")
        for field_name in (
            "calibration_coverage",
            "validation_coverage",
            "informative_interval_fraction",
            "public_range_top1_accuracy",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be in [0, 1]")
        for field_name in (
            "maximum_validation_miss",
            "validation_mean_absolute_error",
            "validation_root_mean_square_error",
            "mean_interval_width",
            "maximum_rollout_standard_error",
            "public_range_nll",
            "uniform_range_nll",
            "public_range_brier",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if not self.candidate_metrics or self.nontrivial_range_examples <= 0:
            raise ValueError("native leaf calibration needs candidate/range evidence")
        if self.counterfactual_value_gate_passed or self.public_belief_calibration_gate_passed:
            raise ValueError("rollout diagnostics cannot promote CF/belief calibration")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_plan_blueprint_fingerprint": (
                self.source_plan_blueprint_fingerprint
            ),
            "config": asdict(self.config),
            "leaf_metrics": {
                "conformal_rank": self.conformal_rank,
                "conformal_radius": self.conformal_radius,
                "calibration_coverage": self.calibration_coverage,
                "validation_coverage": self.validation_coverage,
                "maximum_validation_miss": self.maximum_validation_miss,
                "validation_mean_absolute_error": (
                    self.validation_mean_absolute_error
                ),
                "validation_root_mean_square_error": (
                    self.validation_root_mean_square_error
                ),
                "mean_interval_width": self.mean_interval_width,
                "informative_interval_fraction": (
                    self.informative_interval_fraction
                ),
                "maximum_rollout_standard_error": (
                    self.maximum_rollout_standard_error
                ),
                "candidate_metrics": [
                    asdict(item) for item in self.candidate_metrics
                ],
            },
            "public_range_metrics": {
                "nontrivial_range_examples": self.nontrivial_range_examples,
                "nll": self.public_range_nll,
                "uniform_nll": self.uniform_range_nll,
                "brier": self.public_range_brier,
                "top1_accuracy": self.public_range_top1_accuracy,
                "range_source": "static_supplied_completion_prior",
            },
            "gates": {
                "empirical_rollout_leaf": self.empirical_rollout_leaf_gate_passed,
                "counterfactual_value": self.counterfactual_value_gate_passed,
                "public_belief_calibration": (
                    self.public_belief_calibration_gate_passed
                ),
            },
            "scope": {
                "target": "fixed_profile_empirical_terminal_return",
                "exact_counterfactual_value": False,
                "within_zone_order_heldout": True,
                "hidden_zone_allocation_heldout": False,
                "safety_certificate": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeLeafCalibrationArtifact":
        if int(raw.get("schema_version", -1)) != PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported native leaf calibration schema")
        leaf = raw["leaf_metrics"]
        ranges = raw["public_range_metrics"]
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            source_plan_blueprint_fingerprint=str(
                raw["source_plan_blueprint_fingerprint"]
            ),
            config=NativeLeafCalibrationConfig(**dict(raw["config"])),
            conformal_rank=int(leaf["conformal_rank"]),
            conformal_radius=float(leaf["conformal_radius"]),
            calibration_coverage=float(leaf["calibration_coverage"]),
            validation_coverage=float(leaf["validation_coverage"]),
            maximum_validation_miss=float(leaf["maximum_validation_miss"]),
            validation_mean_absolute_error=float(
                leaf["validation_mean_absolute_error"]
            ),
            validation_root_mean_square_error=float(
                leaf["validation_root_mean_square_error"]
            ),
            mean_interval_width=float(leaf["mean_interval_width"]),
            informative_interval_fraction=float(
                leaf["informative_interval_fraction"]
            ),
            maximum_rollout_standard_error=float(
                leaf["maximum_rollout_standard_error"]
            ),
            candidate_metrics=tuple(
                NativeLeafCandidateMetrics(**dict(item))
                for item in leaf["candidate_metrics"]
            ),
            nontrivial_range_examples=int(ranges["nontrivial_range_examples"]),
            public_range_nll=float(ranges["nll"]),
            uniform_range_nll=float(ranges["uniform_nll"]),
            public_range_brier=float(ranges["brier"]),
            public_range_top1_accuracy=float(ranges["top1_accuracy"]),
            empirical_rollout_leaf_gate_passed=bool(gates["empirical_rollout_leaf"]),
            counterfactual_value_gate_passed=bool(gates["counterfactual_value"]),
            public_belief_calibration_gate_passed=bool(
                gates["public_belief_calibration"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("native leaf calibration fingerprint mismatch")
        return result


def calibrate_native_leaf_rollout_residuals(
    dataset: NativeLeafContinuationDataset,
    config: NativeLeafCalibrationConfig | None = None,
) -> NativeLeafCalibrationArtifact:
    """Calibrate empirical rollout residuals without promoting CF values."""

    config = config or NativeLeafCalibrationConfig()
    calibration = tuple(
        item for item in dataset.examples if item.particle_split == "calibration"
    )
    validation = tuple(
        item for item in dataset.examples if item.particle_split == "validation"
    )
    if len(calibration) < config.minimum_calibration_examples:
        raise ValueError("native leaf calibration split is too small")
    if len(validation) < config.minimum_validation_examples:
        raise ValueError("native leaf validation split is too small")
    if dataset.rollouts_per_leaf < config.minimum_rollouts_per_leaf:
        raise ValueError("native leaf rollouts per state are below the declared minimum")
    errors = sorted(item.absolute_error for item in calibration)
    rank = min(
        len(errors),
        max(1, math.ceil((len(errors) + 1) * config.target_coverage)),
    )
    radius = errors[rank - 1] + config.numerical_margin

    def interval(item: NativeLeafContinuationExample) -> tuple[float, float]:
        return (
            max(-1.0, item.leaf_prediction - radius),
            min(1.0, item.leaf_prediction + radius),
        )

    def covered(item: NativeLeafContinuationExample) -> bool:
        lower, upper = interval(item)
        return lower - 1e-12 <= item.empirical_return <= upper + 1e-12

    calibration_coverage = sum(covered(item) for item in calibration) / len(calibration)
    validation_coverage = sum(covered(item) for item in validation) / len(validation)
    validation_misses = []
    widths = []
    for item in validation:
        lower, upper = interval(item)
        widths.append(upper - lower)
        validation_misses.append(
            max(0.0, lower - item.empirical_return, item.empirical_return - upper)
        )
    candidate_metrics = []
    for candidate_id in dataset.candidate_ids:
        candidate_calibration = tuple(
            item for item in calibration if item.particle_candidate_id == candidate_id
        )
        candidate_validation = tuple(
            item for item in validation if item.particle_candidate_id == candidate_id
        )
        if not candidate_calibration or not candidate_validation:
            raise ValueError("native leaf candidate stratum is incomplete")
        candidate_metrics.append(
            NativeLeafCandidateMetrics(
                candidate_id=candidate_id,
                calibration_examples=len(candidate_calibration),
                validation_examples=len(candidate_validation),
                validation_coverage=(
                    sum(covered(item) for item in candidate_validation)
                    / len(candidate_validation)
                ),
                validation_mean_absolute_error=statistics.mean(
                    item.absolute_error for item in candidate_validation
                ),
            )
        )

    nontrivial = tuple(item for item in dataset.examples if len(item.range_distribution) > 1)
    if not nontrivial:
        raise ValueError("native leaf range audit has no nontrivial examples")
    range_nll = statistics.mean(
        -math.log(max(item.range_truth_probability, 1e-12)) for item in nontrivial
    )
    uniform_nll = statistics.mean(math.log(len(item.range_distribution)) for item in nontrivial)
    range_brier = statistics.mean(
        sum(
            (probability - (1.0 if identifier == item.range_truth_candidate_id else 0.0))
            ** 2
            for identifier, probability in item.range_distribution
        )
        for item in nontrivial
    )
    range_top1 = statistics.mean(
        float(
            max(item.range_distribution, key=lambda pair: (pair[1], pair[0]))[0]
            == item.range_truth_candidate_id
        )
        for item in nontrivial
    )
    mean_width = statistics.mean(widths)
    maximum_standard_error = max(
        item.rollout_standard_error for item in dataset.examples
    )
    gate = (
        len(candidate_metrics) >= config.minimum_candidate_strata
        and validation_coverage >= config.target_coverage
        and max(validation_misses, default=0.0) <= 1e-12
        and mean_width <= config.maximum_mean_interval_width
        and maximum_standard_error <= config.maximum_rollout_standard_error
    )
    return NativeLeafCalibrationArtifact(
        dataset_fingerprint=dataset.fingerprint,
        source_plan_blueprint_fingerprint=(
            dataset.source_plan_blueprint_fingerprint
        ),
        config=config,
        conformal_rank=rank,
        conformal_radius=radius,
        calibration_coverage=calibration_coverage,
        validation_coverage=validation_coverage,
        maximum_validation_miss=max(validation_misses, default=0.0),
        validation_mean_absolute_error=statistics.mean(
            item.absolute_error for item in validation
        ),
        validation_root_mean_square_error=math.sqrt(
            statistics.mean(item.absolute_error**2 for item in validation)
        ),
        mean_interval_width=mean_width,
        informative_interval_fraction=(
            sum(width < 2.0 - 1e-12 for width in widths) / len(widths)
        ),
        maximum_rollout_standard_error=maximum_standard_error,
        candidate_metrics=tuple(candidate_metrics),
        nontrivial_range_examples=len(nontrivial),
        public_range_nll=range_nll,
        uniform_range_nll=uniform_nll,
        public_range_brier=range_brier,
        public_range_top1_accuracy=range_top1,
        empirical_rollout_leaf_gate_passed=gate,
    )


@dataclass(frozen=True, slots=True)
class Phase6NativeLeafCalibrationEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    calibration_fingerprint: str
    source_phase6p_evidence_fingerprint: str
    example_count: int
    terminal_rollout_count: int
    candidate_strata: int
    calibration_particles: int
    validation_particles: int
    native_transition_count: int
    conformal_radius: float
    validation_coverage: float
    mean_interval_width: float
    maximum_rollout_standard_error: float
    empirical_rollout_leaf_gate_passed: bool
    public_belief_calibrated: bool
    heldout_particle_order_validated: bool
    heldout_particle_allocation_validated: bool
    counterfactual_values_independently_calibrated: bool
    counterfactual_bounds_built: bool
    safe_resolver_executed: bool
    exploitation_executed: bool
    official_strategy_control_enabled: bool
    raw_private_card_ids_persisted: int
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6S evidence identity is invalid")
        for field_name in (
            "dataset_fingerprint",
            "calibration_fingerprint",
            "source_phase6p_evidence_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if min(
            self.example_count,
            self.terminal_rollout_count,
            self.candidate_strata,
            self.calibration_particles,
            self.validation_particles,
            self.native_transition_count,
        ) <= 0:
            raise ValueError("Phase 6S accounting must be positive")
        for field_name in (
            "conformal_radius",
            "validation_coverage",
            "mean_interval_width",
            "maximum_rollout_standard_error",
        ):
            value = _finite(getattr(self, field_name), field_name)
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if not self.heldout_particle_order_validated:
            raise ValueError("Phase 6S must validate its declared held-out order split")
        if any(
            (
                self.public_belief_calibrated,
                self.heldout_particle_allocation_validated,
                self.counterfactual_values_independently_calibrated,
                self.counterfactual_bounds_built,
                self.safe_resolver_executed,
                self.exploitation_executed,
                self.official_strategy_control_enabled,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6S promoted an invalid safety/deployment claim")
        if dict(self.claims).get("phase6_complete") != "false":
            raise ValueError("Phase 6S cannot mark Phase 6 complete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "dataset_fingerprint": self.dataset_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "source_phase6p_evidence_fingerprint": (
                self.source_phase6p_evidence_fingerprint
            ),
            "accounting": {
                "examples": self.example_count,
                "terminal_rollouts": self.terminal_rollout_count,
                "candidate_strata": self.candidate_strata,
                "calibration_particles": self.calibration_particles,
                "validation_particles": self.validation_particles,
                "native_transition_count": self.native_transition_count,
            },
            "diagnostics": {
                "conformal_radius": self.conformal_radius,
                "validation_coverage": self.validation_coverage,
                "mean_interval_width": self.mean_interval_width,
                "maximum_rollout_standard_error": (
                    self.maximum_rollout_standard_error
                ),
                "empirical_rollout_leaf_gate_passed": (
                    self.empirical_rollout_leaf_gate_passed
                ),
            },
            "admission": {
                "public_belief_calibrated": self.public_belief_calibrated,
                "heldout_particle_order_validated": (
                    self.heldout_particle_order_validated
                ),
                "heldout_particle_allocation_validated": (
                    self.heldout_particle_allocation_validated
                ),
                "counterfactual_values_independently_calibrated": (
                    self.counterfactual_values_independently_calibrated
                ),
            },
            "execution": {
                "counterfactual_bounds_built": self.counterfactual_bounds_built,
                "safe_resolver_executed": self.safe_resolver_executed,
                "exploitation_executed": self.exploitation_executed,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6NativeLeafCalibrationEvidence":
        if int(raw.get("schema_version", -1)) != PHASE6_NATIVE_LEAF_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported Phase 6S evidence schema")
        accounting = raw["accounting"]
        diagnostics = raw["diagnostics"]
        admission = raw["admission"]
        execution = raw["execution"]
        result = cls(
            benchmark_id=str(raw["benchmark_id"]),
            status=str(raw["status"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            calibration_fingerprint=str(raw["calibration_fingerprint"]),
            source_phase6p_evidence_fingerprint=str(
                raw["source_phase6p_evidence_fingerprint"]
            ),
            example_count=int(accounting["examples"]),
            terminal_rollout_count=int(accounting["terminal_rollouts"]),
            candidate_strata=int(accounting["candidate_strata"]),
            calibration_particles=int(accounting["calibration_particles"]),
            validation_particles=int(accounting["validation_particles"]),
            native_transition_count=int(accounting["native_transition_count"]),
            conformal_radius=float(diagnostics["conformal_radius"]),
            validation_coverage=float(diagnostics["validation_coverage"]),
            mean_interval_width=float(diagnostics["mean_interval_width"]),
            maximum_rollout_standard_error=float(
                diagnostics["maximum_rollout_standard_error"]
            ),
            empirical_rollout_leaf_gate_passed=bool(
                diagnostics["empirical_rollout_leaf_gate_passed"]
            ),
            public_belief_calibrated=bool(admission["public_belief_calibrated"]),
            heldout_particle_order_validated=bool(
                admission["heldout_particle_order_validated"]
            ),
            heldout_particle_allocation_validated=bool(
                admission["heldout_particle_allocation_validated"]
            ),
            counterfactual_values_independently_calibrated=bool(
                admission["counterfactual_values_independently_calibrated"]
            ),
            counterfactual_bounds_built=bool(
                execution["counterfactual_bounds_built"]
            ),
            safe_resolver_executed=bool(execution["safe_resolver_executed"]),
            exploitation_executed=bool(execution["exploitation_executed"]),
            official_strategy_control_enabled=bool(
                execution["official_strategy_control_enabled"]
            ),
            raw_private_card_ids_persisted=int(
                execution["raw_private_card_ids_persisted"]
            ),
            claims=tuple(
                (str(name), str(value)) for name, value in raw["claims"].items()
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("Phase 6S evidence fingerprint mismatch")
        return result

    def to_markdown(self) -> str:
        return f"""# Phase 6S native leaf continuation diagnostic

Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

The dataset contains {self.example_count} cutoff leaves across
{self.candidate_strata} candidate strata and {self.terminal_rollout_count}
genuine native terminal continuations. Calibration and validation use disjoint
within-zone hidden-order particles ({self.calibration_particles} / 
{self.validation_particles}). Card allocation between zones is not held out.

The empirical residual radius is `{self.conformal_radius:.6g}`, validation
coverage is `{self.validation_coverage:.3f}`, mean clipped interval width is
`{self.mean_interval_width:.6g}`, and maximum rollout standard error is
`{self.maximum_rollout_standard_error:.6g}`. Empirical rollout leaf gate:
**{str(self.empirical_rollout_leaf_gate_passed).lower()}**.

These targets are repeated returns of one fixed stochastic plan/tactical
profile, not exact opponent counterfactual values. The public range is still a
static supplied prior. Public-belief calibration, held-out allocation
validation, counterfactual calibration, bounds, safe resolving, exploitation,
official strategy control, and Kaggle submission all remain disabled.
"""


@dataclass(frozen=True, slots=True)
class Phase6NativeLeafCalibrationArtifacts:
    dataset: NativeLeafContinuationDataset
    calibration: NativeLeafCalibrationArtifact
    evidence: Phase6NativeLeafCalibrationEvidence


def _strategy(provider: NativePlanBlueprintProvider, information_set, belief):
    player = 0 if information_set.actor is InformationSetActor.SELF else 1
    return provider.current_strategy(information_set, belief, player=player)


def _sample_plan_action(
    game: OfficialNativePlanGame,
    state: NativePlanGameState,
    provider: NativePlanBlueprintProvider,
    rng: random.Random,
):
    player = game.player_to_act(state)
    if player not in (0, 1):
        raise RuntimeError("native plan sampling state lacks a player")
    information_set = game.information_set(state, player=player)
    belief = game.public_belief(state, player=player)
    distribution = _strategy(provider, information_set, belief)
    distribution.validate_support(information_set)
    selected = _sample_identifier(
        tuple(
            (action.action_id, distribution.probability(action.action_id))
            for action in information_set.legal_actions
        ),
        rng,
    )
    return next(
        action for action in information_set.legal_actions if action.action_id == selected
    )


def _sample_chance(game, state, rng: random.Random):
    outcomes = game.chance_outcomes(state)
    selected = _sample_identifier(
        tuple((item.outcome_id, item.probability) for item in outcomes), rng
    )
    return next(item for item in outcomes if item.outcome_id == selected)


def _sample_cutoff_state(
    wrapper: DepthLimitedInformationSetGame,
    macro_game: OfficialNativePlanGame,
    provider: NativePlanBlueprintProvider,
    *,
    rng: random.Random,
) -> DepthLimitedGameState:
    state = wrapper.root_state()
    for _ in range(wrapper.maximum_source_depth_delta + 2):
        if state.terminal:
            if not state.cutoff or state.source_state.terminal:
                raise RuntimeError("native leaf path reached an official terminal before cutoff")
            wrapper.terminal_utility(state, 0)
            return state
        if state.actor is InformationSetActor.CHANCE:
            state = wrapper.transition(
                state,
                None,
                chance_outcome=_sample_chance(wrapper, state, rng),
            )
        else:
            state = wrapper.transition(
                state,
                _sample_plan_action(macro_game, state.source_state, provider, rng),
            )
    raise RuntimeError("native leaf path did not reach the declared cutoff")


def _rollout_to_terminal(
    game: OfficialNativePlanGame,
    start: NativePlanGameState,
    provider: NativePlanBlueprintProvider,
    *,
    value_player: int,
    rng: random.Random,
    maximum_macro_transitions: int,
) -> tuple[float, int, int]:
    state = start
    transitions = 0
    while not state.terminal:
        if transitions >= maximum_macro_transitions:
            raise RuntimeError("native leaf terminal continuation exceeded its budget")
        if state.actor is InformationSetActor.CHANCE:
            state = game.transition(
                state,
                None,
                chance_outcome=_sample_chance(game, state, rng),
            )
        else:
            state = game.transition(
                state,
                _sample_plan_action(game, state, provider, rng),
            )
        transitions += 1
    return game.terminal_utility(state, value_player), transitions, state.native_state.depth


def _particle_family(
    base: tuple[HiddenZoneCompletion, ...],
    *,
    variants: int,
    seed: int,
) -> tuple[
    tuple[HiddenZoneCompletion, ...],
    dict[str, tuple[str, int, str]],
]:
    if variants < 3:
        raise ValueError("native leaf calibration needs at least three order variants")
    probability = 1.0 / (len(base) * variants)
    particles = []
    metadata: dict[str, tuple[str, int, str]] = {}
    for completion in base:
        for variant in range(variants):
            split = "validation" if variant == variants - 1 else "calibration"
            identifier = f"{completion.opponent_candidate_id}:order-{variant}:{split}"
            particle = permute_hidden_zone_completion(
                completion,
                hypothesis_id=identifier,
                probability=probability,
                permutation_seed=_seed(seed, completion.fingerprint, variant),
                source=(
                    "Phase 6S held-out within-zone order particle; public-consistent "
                    "card allocation inherited unchanged from Phase 6J"
                ),
            )
            particles.append(particle)
            metadata[identifier] = (
                completion.opponent_candidate_id,
                variant,
                split,
            )
    return tuple(particles), metadata


def run_phase6_native_leaf_calibration(
    *,
    project_root: str | Path | None = None,
    source_tactical_blueprint_path: str | Path | None = None,
    source_plan_blueprint_path: str | Path | None = None,
    source_phase6p_evidence_path: str | Path | None = None,
    maximum_source_depth_delta: int = 2,
    particle_variants: int = 3,
    rollouts_per_leaf: int = 3,
    maximum_macro_transitions: int = 512,
    maximum_tactical_transitions: int = 512,
    calibration_config: NativeLeafCalibrationConfig | None = None,
    seed: int = PHASE6_NATIVE_LEAF_CALIBRATION_SEED,
) -> Phase6NativeLeafCalibrationArtifacts:
    if min(
        maximum_source_depth_delta,
        particle_variants,
        rollouts_per_leaf,
        maximum_macro_transitions,
        maximum_tactical_transitions,
    ) <= 0:
        raise ValueError("Phase 6S limits must be positive")
    if particle_variants < 3 or rollouts_per_leaf < 2:
        raise ValueError("Phase 6S needs disjoint particles and repeated rollouts")
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
    if phase6p.final_model_fingerprint != plan_artifact.fingerprint:
        raise ValueError("Phase 6S plan model does not match Phase 6P evidence")
    if (
        plan_artifact.source_tactical_blueprint_fingerprint
        != tactical_artifact.fingerprint
    ):
        raise ValueError("Phase 6S plan/tactical provenance mismatch")
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
    if tactical_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6S tactical blueprint/deck mismatch")
    if plan_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6S plan blueprint/deck mismatch")
    candidates: tuple[KaggleDeckProfile, ...] = OFFICIAL_PANEL_DECKS + (target,)
    observation, start = battle_start(
        list(target.card_ids), list(OFFICIAL_PANEL_DECKS[0].card_ids)
    )
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6S decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        base = _completions(
            observation,
            target=target,
            candidates=candidates,
            basic_card_ids=_runtime_basics(runtime),
        )
        particles, particle_metadata = _particle_family(
            base, variants=particle_variants, seed=seed
        )
        particle_family_fingerprint = _fingerprint(
            {
                "particles": [item.fingerprint for item in particles],
                "metadata": particle_metadata,
                "allocation_heldout": False,
                "order_heldout": True,
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
                    native_game.root_state(), None, chance_outcome=outcome
                )
                completion = native_game.completion(scenario.hypothesis_id or "")
                candidate_id, variant, split = particle_metadata[completion.hypothesis_id]
                macro = OfficialNativePlanGame(
                    native_game,
                    scenario,
                    bridge=bridge,
                    tactical_provider=tactical_provider,
                    maximum_tactical_transitions=maximum_tactical_transitions,
                )
                macro_games.append(macro)
                root_range = native_game.supplied_completion_range(player=0)
                root_range_truth = completion.candidate_for_player(1)[0]
                root_range_distribution = tuple(
                    (item.hypothesis_id, item.weight)
                    for item in root_range.hypotheses
                )
                if root_range_truth not in dict(root_range_distribution):
                    raise RuntimeError(
                        "Phase 6S root evaluation truth is absent from public range"
                    )
                leaf_provider = UncalibratedNativePlanLeafValueProvider(macro)
                wrapper = DepthLimitedInformationSetGame(
                    macro,
                    leaf_provider,
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
                    raise RuntimeError("Phase 6S cutoff did not retain a native plan state")
                acting_player = macro.player_to_act(source_state)
                if acting_player not in (0, 1):
                    raise RuntimeError("Phase 6S cutoff lacks an acting player")
                information_set = macro.information_set(
                    source_state, player=acting_player
                )
                belief = macro.public_belief(source_state, player=acting_player)
                estimate = leaf_provider.estimate(macro, source_state, player=0)
                if estimate.calibrated or estimate.confidence != 0:
                    raise RuntimeError("Phase 6S source leaf unexpectedly claims calibration")
                returns = []
                depths = []
                rollout_transitions = 0
                for rollout in range(rollouts_per_leaf):
                    value, transitions, terminal_depth = _rollout_to_terminal(
                        macro,
                        source_state,
                        plan_provider,
                        value_player=0,
                        rng=random.Random(
                            _seed(seed, "rollout", particle_index, rollout)
                        ),
                        maximum_macro_transitions=maximum_macro_transitions,
                    )
                    returns.append(value)
                    depths.append(terminal_depth)
                    rollout_transitions += transitions
                empirical = statistics.mean(returns)
                standard_error = (
                    statistics.stdev(returns) / math.sqrt(len(returns))
                    if len(returns) > 1
                    else 0.0
                )
                example_id = _fingerprint(
                    {
                        "source_state": source_state.state_id,
                        "particle": completion.fingerprint,
                        "split": split,
                        "variant": variant,
                        "rollout_seeds": [
                            _seed(seed, "rollout", particle_index, rollout)
                            for rollout in range(rollouts_per_leaf)
                        ],
                        "plan_model": plan_artifact.fingerprint,
                    }
                )
                examples.append(
                    NativeLeafContinuationExample(
                        example_id=example_id,
                        source_game_fingerprint=macro.fingerprint,
                        source_state_id=source_state.state_id,
                        information_set_id=information_set.info_set_id,
                        public_belief_fingerprint=belief.fingerprint,
                        public_history_fingerprint=(
                            belief.public_history_fingerprint
                        ),
                        private_observation_fingerprint=(
                            information_set.private_observation_fingerprint or ""
                        ),
                        particle_id=completion.hypothesis_id,
                        particle_candidate_id=candidate_id,
                        particle_split=split,
                        particle_variant=variant,
                        acting_player=acting_player,
                        value_player=0,
                        leaf_prediction=estimate.value,
                        terminal_returns=tuple(returns),
                        empirical_return=empirical,
                        absolute_error=abs(estimate.value - empirical),
                        rollout_standard_error=standard_error,
                        rollout_macro_transitions=rollout_transitions,
                        terminal_native_depth_minimum=min(depths),
                        terminal_native_depth_maximum=max(depths),
                        range_truth_candidate_id=root_range_truth,
                        range_distribution=root_range_distribution,
                        public_features=tuple(belief.public_features),
                        private_features=tuple(information_set.private_features),
                    )
                )
            provisional = NativeLeafContinuationDataset(
                native_game_fingerprint=native_game.fingerprint,
                runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                deck_fingerprint=target.fingerprint,
                source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
                source_plan_blueprint_fingerprint=plan_artifact.fingerprint,
                candidate_ids=tuple(candidate.archetype for candidate in candidates),
                particle_family_fingerprint=particle_family_fingerprint,
                maximum_source_depth_delta=maximum_source_depth_delta,
                rollouts_per_leaf=rollouts_per_leaf,
                examples=tuple(examples),
                native_transition_count=native_game.transition_count,
                macro_transition_count=sum(
                    game.macro_transition_count for game in macro_games
                ),
                tactical_transition_count=sum(
                    game.tactical_transition_count for game in macro_games
                ),
            )
            serialized = provisional.to_dict()
            raw_private = _raw_private_key_count(serialized)
            if raw_private != 0:
                raise RuntimeError("Phase 6S serialized a forbidden raw private key")
            dataset = provisional
            restored = NativeLeafContinuationDataset.from_dict(
                json.loads(json.dumps(serialized))
            )
            if restored.fingerprint != dataset.fingerprint:
                raise RuntimeError("Phase 6S dataset roundtrip drifted")
            calibration = calibrate_native_leaf_rollout_residuals(
                dataset, calibration_config
            )
            calibration_particles = len(
                {
                    item.particle_id
                    for item in dataset.examples
                    if item.particle_split == "calibration"
                }
            )
            validation_particles = len(
                {
                    item.particle_id
                    for item in dataset.examples
                    if item.particle_split == "validation"
                }
            )
            claims = (
                ("source_terminal_continuations", "genuine_official_search_execution"),
                ("cutoff_game_and_plan_actions", "research_abstraction"),
                ("public_range", "static_supplied_prior_not_calibrated"),
                ("heldout_particle_scope", "within_zone_order_only_allocation_not_heldout"),
                ("value_target", "fixed_profile_empirical_return_not_counterfactual"),
                ("empirical_leaf_gate", str(calibration.empirical_rollout_leaf_gate_passed).lower()),
                ("counterfactual_value_bounds", "not_built"),
                ("safe_resolver", "not_enabled"),
                ("exploitation_overlay", "not_enabled"),
                ("official_strategy_control", "disabled"),
                ("equilibrium_convergence", "not_validated"),
                ("pokemon_strength", "not_claimed"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            evidence = Phase6NativeLeafCalibrationEvidence(
                benchmark_id="phase6s-native-leaf-calibration-v1",
                status="phase6s_native_leaf_continuation_diagnostic_validated_partial",
                dataset_fingerprint=dataset.fingerprint,
                calibration_fingerprint=calibration.fingerprint,
                source_phase6p_evidence_fingerprint=phase6p.fingerprint,
                example_count=len(dataset.examples),
                terminal_rollout_count=len(dataset.examples) * rollouts_per_leaf,
                candidate_strata=len(dataset.candidate_ids),
                calibration_particles=calibration_particles,
                validation_particles=validation_particles,
                native_transition_count=dataset.native_transition_count,
                conformal_radius=calibration.conformal_radius,
                validation_coverage=calibration.validation_coverage,
                mean_interval_width=calibration.mean_interval_width,
                maximum_rollout_standard_error=(
                    calibration.maximum_rollout_standard_error
                ),
                empirical_rollout_leaf_gate_passed=(
                    calibration.empirical_rollout_leaf_gate_passed
                ),
                public_belief_calibrated=False,
                heldout_particle_order_validated=True,
                heldout_particle_allocation_validated=False,
                counterfactual_values_independently_calibrated=False,
                counterfactual_bounds_built=False,
                safe_resolver_executed=False,
                exploitation_executed=False,
                official_strategy_control_enabled=False,
                raw_private_card_ids_persisted=raw_private,
                claims=claims,
            )
            return Phase6NativeLeafCalibrationArtifacts(
                dataset, calibration, evidence
            )
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_leaf_calibration().evidence.to_json())
