"""Phase 6Y exact reach audit and native counterfactual-reach ledger.

Phase 6X produced useful action-conditioned native returns but did not persist
the reach terms required to distinguish a conditional rollout value from a
counterfactual value.  Phase 6Y records those terms while recollecting the same
five-plan experiment and validates the importance-ratio convention exactly on
the enumerable microgame.

The native result deliberately remains inadmissible for safe resolving.  Its
hidden-completion prior is supplied, every public history is represented by a
single sampled root, and there is no collection-disjoint counterfactual model
or bound calibration.  Genuine native descendant execution and abstract reach
semantics are reported separately throughout the artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .counterfactual_reach import (
    CounterfactualReachExpectationAudit,
    CounterfactualReachTrace,
    audit_counterfactual_reach_expectation,
)
from .exact_game import RocketPrizeDuel, uniform_profile
from .phase6_opponent_deviation_values import (
    OPPONENT_DEVIATION_ACTION_IDS,
    OpponentDeviationCalibration,
    OpponentDeviationDataset,
    OpponentDeviationSurface,
    run_phase6_opponent_deviation_values,
)
from .phase6_native_external_sampling import _raw_private_key_count


PHASE6_NATIVE_COUNTERFACTUAL_REACH_SCHEMA_VERSION = 1
PHASE6_NATIVE_COUNTERFACTUAL_REACH_SEED = 6_031


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class NativeReachCorrectedPlanValue:
    action_id: str
    empirical_conditional_value: float
    rollout_standard_error: float
    counterfactual_importance_weight: float
    weighted_counterfactual_numerator_contribution: float
    weighted_counterfactual_mass_contribution: float
    self_normalized_counterfactual_value: float

    def __post_init__(self) -> None:
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("native reach value action is unsupported")
        value = _finite(self.empirical_conditional_value, "empirical conditional value")
        standard_error = _finite(self.rollout_standard_error, "rollout standard error")
        weight = _finite(
            self.counterfactual_importance_weight,
            "counterfactual importance weight",
        )
        if standard_error < 0 or weight <= 0:
            raise ValueError("native reach value precision/weight is invalid")
        if not math.isclose(
            self.weighted_counterfactual_numerator_contribution,
            weight * value,
            abs_tol=1e-12,
        ):
            raise ValueError("native reach numerator contribution is inconsistent")
        if not math.isclose(
            self.weighted_counterfactual_mass_contribution,
            weight,
            abs_tol=1e-12,
        ):
            raise ValueError("native reach mass contribution is inconsistent")
        if not math.isclose(
            self.self_normalized_counterfactual_value,
            value,
            abs_tol=1e-12,
        ):
            raise ValueError("single-root self-normalized value is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeReachCorrectedPlanValue":
        return cls(
            action_id=str(raw["action_id"]),
            empirical_conditional_value=float(raw["empirical_conditional_value"]),
            rollout_standard_error=float(raw["rollout_standard_error"]),
            counterfactual_importance_weight=float(
                raw["counterfactual_importance_weight"]
            ),
            weighted_counterfactual_numerator_contribution=float(
                raw["weighted_counterfactual_numerator_contribution"]
            ),
            weighted_counterfactual_mass_contribution=float(
                raw["weighted_counterfactual_mass_contribution"]
            ),
            self_normalized_counterfactual_value=float(
                raw["self_normalized_counterfactual_value"]
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeCounterfactualReachSurface:
    source_surface_id: str
    source_surface_fingerprint: str
    source_state_id: str
    information_set_id: str
    public_history_fingerprint: str
    particle_fingerprint: str
    candidate_id_supervision_only: str
    split: str
    variant: int
    particle_prior_probability: float
    reach_trace: CounterfactualReachTrace
    action_values: tuple[NativeReachCorrectedPlanValue, ...]
    source_descendant_official_engine_executed: bool = True
    hidden_completion_prior_official: bool = False
    counterfactual_value_calibrated: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "source_surface_id",
            "source_surface_fingerprint",
            "source_state_id",
            "public_history_fingerprint",
            "particle_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.information_set_id or not self.candidate_id_supervision_only:
            raise ValueError("native reach surface identity is incomplete")
        if self.split not in {"calibration", "validation"} or self.variant not in (0, 1):
            raise ValueError("native reach surface split/variant is invalid")
        if not 0 < self.particle_prior_probability <= 1:
            raise ValueError("native reach particle prior must be positive")
        if self.reach_trace.target_player != 1:
            raise ValueError("native reach surface must target opponent player 1")
        if self.reach_trace.decision_state_fingerprint != self.source_state_id:
            raise ValueError("native reach trace does not end at its source state")
        if not math.isclose(self.reach_trace.player_1_reach, 1.0, abs_tol=1e-12):
            raise ValueError("first opponent decision must have unit opponent self-reach")
        if tuple(value.action_id for value in self.action_values) != (
            OPPONENT_DEVIATION_ACTION_IDS
        ):
            raise ValueError("native reach action support/order is invalid")
        if any(
            not math.isclose(
                value.counterfactual_importance_weight,
                self.reach_trace.importance_weight,
                abs_tol=1e-12,
            )
            for value in self.action_values
        ):
            raise ValueError("native reach action weights disagree with their trace")
        if any(
            (
                not self.source_descendant_official_engine_executed,
                not self.reach_trace.source_descendants_official_engine_executed,
                self.reach_trace.upstream_official_reach_complete,
                self.hidden_completion_prior_official,
                self.counterfactual_value_calibrated,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("native reach surface promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_COUNTERFACTUAL_REACH_SCHEMA_VERSION,
            "source": {
                "surface_id": self.source_surface_id,
                "surface_fingerprint": self.source_surface_fingerprint,
                "state_id": self.source_state_id,
                "information_set_id": self.information_set_id,
                "public_history_fingerprint": self.public_history_fingerprint,
                "source_descendant_official_engine_executed": (
                    self.source_descendant_official_engine_executed
                ),
            },
            "hypothesis": {
                "particle_fingerprint": self.particle_fingerprint,
                "candidate_id_supervision_only": self.candidate_id_supervision_only,
                "split": self.split,
                "variant": self.variant,
                "particle_prior_probability": self.particle_prior_probability,
                "hidden_completion_prior_official": (
                    self.hidden_completion_prior_official
                ),
            },
            "reach_trace": self.reach_trace.to_dict(),
            "action_values": [value.to_dict() for value in self.action_values],
            "claims": {
                "root_relative_reach": True,
                "first_opponent_decision": True,
                "opponent_self_reach_excluded": True,
                "counterfactual_value_calibrated": (
                    self.counterfactual_value_calibrated
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeCounterfactualReachSurface":
        source = raw["source"]
        hypothesis = raw["hypothesis"]
        claims = raw["claims"]
        result = cls(
            source_surface_id=str(source["surface_id"]),
            source_surface_fingerprint=str(source["surface_fingerprint"]),
            source_state_id=str(source["state_id"]),
            information_set_id=str(source["information_set_id"]),
            public_history_fingerprint=str(source["public_history_fingerprint"]),
            particle_fingerprint=str(hypothesis["particle_fingerprint"]),
            candidate_id_supervision_only=str(
                hypothesis["candidate_id_supervision_only"]
            ),
            split=str(hypothesis["split"]),
            variant=int(hypothesis["variant"]),
            particle_prior_probability=float(
                hypothesis["particle_prior_probability"]
            ),
            reach_trace=CounterfactualReachTrace.from_dict(raw["reach_trace"]),
            action_values=tuple(
                NativeReachCorrectedPlanValue.from_dict(value)
                for value in raw["action_values"]
            ),
            source_descendant_official_engine_executed=bool(
                source["source_descendant_official_engine_executed"]
            ),
            hidden_completion_prior_official=bool(
                hypothesis["hidden_completion_prior_official"]
            ),
            counterfactual_value_calibrated=bool(
                claims["counterfactual_value_calibrated"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        if not all(
            bool(claims[name])
            for name in (
                "root_relative_reach",
                "first_opponent_decision",
                "opponent_self_reach_excluded",
            )
        ):
            raise ValueError("native reach surface semantic claim changed")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("native reach surface fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativeCounterfactualReachDataset:
    source_opponent_deviation_dataset_fingerprint: str
    source_opponent_deviation_calibration_fingerprint: str
    exact_reach_audit_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    candidate_ids: tuple[str, ...]
    rollouts_per_action: int
    native_transition_count: int
    surfaces: tuple[NativeCounterfactualReachSurface, ...]
    exact_reach_semantics_audited: bool
    native_reach_ledger_complete: bool
    hidden_completion_prior_official: bool = False
    upstream_official_reach_complete: bool = False
    collection_disjoint_counterfactual_calibration: bool = False
    native_counterfactual_values_calibrated: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "source_opponent_deviation_dataset_fingerprint",
            "source_opponent_deviation_calibration_fingerprint",
            "exact_reach_audit_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("native reach candidates must be unique")
        if self.rollouts_per_action < 2 or self.native_transition_count <= 0:
            raise ValueError("native reach dataset accounting is invalid")
        expected = {
            (candidate, variant)
            for candidate in self.candidate_ids
            for variant in (0, 1)
        }
        observed = {
            (surface.candidate_id_supervision_only, surface.variant)
            for surface in self.surfaces
        }
        if observed != expected or len(self.surfaces) != len(expected):
            raise ValueError("native reach hypothesis support is incomplete")
        if len({surface.source_surface_id for surface in self.surfaces}) != len(
            self.surfaces
        ):
            raise ValueError("native reach source surfaces must be unique")
        if not math.isclose(
            sum(surface.particle_prior_probability for surface in self.surfaces),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("native reach particle priors must sum to one")
        expected_ledger = all(
            (
                len(surface.action_values) == len(OPPONENT_DEVIATION_ACTION_IDS),
                surface.reach_trace.target_and_sampling_strategy_match,
                math.isclose(surface.reach_trace.player_1_reach, 1.0, abs_tol=1e-12),
                math.isfinite(surface.reach_trace.importance_weight),
                surface.reach_trace.importance_weight > 0,
            )
            for surface in self.surfaces
        )
        if self.native_reach_ledger_complete != expected_ledger:
            raise ValueError("native reach ledger gate is inconsistent")
        if not self.exact_reach_semantics_audited:
            raise ValueError("native reach dataset requires the exact semantic audit")
        if any(
            (
                self.hidden_completion_prior_official,
                self.upstream_official_reach_complete,
                self.collection_disjoint_counterfactual_calibration,
                self.native_counterfactual_values_calibrated,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("native reach dataset promoted an unavailable capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def action_target_count(self) -> int:
        return len(self.surfaces) * len(OPPONENT_DEVIATION_ACTION_IDS)

    @property
    def terminal_rollout_count(self) -> int:
        return self.action_target_count * self.rollouts_per_action

    @property
    def minimum_importance_weight(self) -> float:
        return min(surface.reach_trace.importance_weight for surface in self.surfaces)

    @property
    def maximum_importance_weight(self) -> float:
        return max(surface.reach_trace.importance_weight for surface in self.surfaces)

    @property
    def unique_public_history_count(self) -> int:
        return len({surface.public_history_fingerprint for surface in self.surfaces})

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_COUNTERFACTUAL_REACH_SCHEMA_VERSION,
            "source_opponent_deviation_dataset_fingerprint": (
                self.source_opponent_deviation_dataset_fingerprint
            ),
            "source_opponent_deviation_calibration_fingerprint": (
                self.source_opponent_deviation_calibration_fingerprint
            ),
            "exact_reach_audit_fingerprint": self.exact_reach_audit_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "source_plan_blueprint_fingerprint": (
                self.source_plan_blueprint_fingerprint
            ),
            "candidate_ids": list(self.candidate_ids),
            "rollouts_per_action": self.rollouts_per_action,
            "accounting": {
                "surface_count": len(self.surfaces),
                "action_target_count": self.action_target_count,
                "terminal_rollout_count": self.terminal_rollout_count,
                "native_transition_count": self.native_transition_count,
                "unique_public_history_count": self.unique_public_history_count,
                "minimum_importance_weight": self.minimum_importance_weight,
                "maximum_importance_weight": self.maximum_importance_weight,
            },
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "gates": {
                "exact_reach_semantics_audited": self.exact_reach_semantics_audited,
                "native_reach_ledger_complete": self.native_reach_ledger_complete,
                "native_counterfactual_values_calibrated": (
                    self.native_counterfactual_values_calibrated
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
            "claims": {
                "source_descendants_official_engine_executed": True,
                "plan_and_reach_game_abstract": True,
                "hidden_completion_prior_official": (
                    self.hidden_completion_prior_official
                ),
                "upstream_official_reach_complete": (
                    self.upstream_official_reach_complete
                ),
                "collection_disjoint_counterfactual_calibration": (
                    self.collection_disjoint_counterfactual_calibration
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeCounterfactualReachDataset":
        accounting = raw["accounting"]
        gates = raw["gates"]
        claims = raw["claims"]
        result = cls(
            source_opponent_deviation_dataset_fingerprint=str(
                raw["source_opponent_deviation_dataset_fingerprint"]
            ),
            source_opponent_deviation_calibration_fingerprint=str(
                raw["source_opponent_deviation_calibration_fingerprint"]
            ),
            exact_reach_audit_fingerprint=str(raw["exact_reach_audit_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(
                raw["source_plan_blueprint_fingerprint"]
            ),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            rollouts_per_action=int(raw["rollouts_per_action"]),
            native_transition_count=int(accounting["native_transition_count"]),
            surfaces=tuple(
                NativeCounterfactualReachSurface.from_dict(surface)
                for surface in raw["surfaces"]
            ),
            exact_reach_semantics_audited=bool(
                gates["exact_reach_semantics_audited"]
            ),
            native_reach_ledger_complete=bool(gates["native_reach_ledger_complete"]),
            hidden_completion_prior_official=bool(
                claims["hidden_completion_prior_official"]
            ),
            upstream_official_reach_complete=bool(
                claims["upstream_official_reach_complete"]
            ),
            collection_disjoint_counterfactual_calibration=bool(
                claims["collection_disjoint_counterfactual_calibration"]
            ),
            native_counterfactual_values_calibrated=bool(
                gates["native_counterfactual_values_calibrated"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected_accounting = {
            "surface_count": len(result.surfaces),
            "action_target_count": result.action_target_count,
            "terminal_rollout_count": result.terminal_rollout_count,
            "native_transition_count": result.native_transition_count,
            "unique_public_history_count": result.unique_public_history_count,
            "minimum_importance_weight": result.minimum_importance_weight,
            "maximum_importance_weight": result.maximum_importance_weight,
        }
        if any(
            not math.isclose(float(accounting[name]), float(value), abs_tol=1e-12)
            for name, value in expected_accounting.items()
        ):
            raise ValueError("native reach dataset accounting mismatch")
        if not all(
            bool(claims[name])
            for name in (
                "source_descendants_official_engine_executed",
                "plan_and_reach_game_abstract",
            )
        ):
            raise ValueError("native reach provenance claim changed")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("native counterfactual reach dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class NativeCounterfactualReachAdmission:
    dataset_fingerprint: str
    exact_reach_audit_fingerprint: str
    exact_maximum_absolute_error: float
    exact_error_threshold: float
    trace_coverage_fraction: float
    unit_opponent_self_reach_fraction: float
    finite_importance_weight_fraction: float
    unique_public_history_count: int
    maximum_histories_per_public_information_set: int
    exact_reach_semantics_gate_passed: bool
    native_reach_ledger_gate_passed: bool
    native_counterfactual_value_gate_passed: bool
    safe_resolver_compatible: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        _validate_digest(self.exact_reach_audit_fingerprint, "exact_reach_audit_fingerprint")
        if min(self.exact_maximum_absolute_error, self.exact_error_threshold) < 0:
            raise ValueError("native reach exact audit threshold is invalid")
        for field_name in (
            "trace_coverage_fraction",
            "unit_opponent_self_reach_fraction",
            "finite_importance_weight_fraction",
        ):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite fraction")
        if min(
            self.unique_public_history_count,
            self.maximum_histories_per_public_information_set,
        ) <= 0:
            raise ValueError("native reach public-history accounting is invalid")
        exact_gate = self.exact_maximum_absolute_error <= self.exact_error_threshold
        ledger_gate = all(
            math.isclose(value, 1.0, abs_tol=1e-12)
            for value in (
                self.trace_coverage_fraction,
                self.unit_opponent_self_reach_fraction,
                self.finite_importance_weight_fraction,
            )
        )
        if self.exact_reach_semantics_gate_passed != exact_gate:
            raise ValueError("native reach exact semantic gate is inconsistent")
        if self.native_reach_ledger_gate_passed != ledger_gate:
            raise ValueError("native reach ledger gate is inconsistent")
        if self.native_counterfactual_value_gate_passed or self.safe_resolver_compatible:
            raise ValueError("root-relative reach ledger cannot promote safe values")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_COUNTERFACTUAL_REACH_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "exact_reach_audit_fingerprint": self.exact_reach_audit_fingerprint,
            "metrics": {
                "exact_maximum_absolute_error": self.exact_maximum_absolute_error,
                "exact_error_threshold": self.exact_error_threshold,
                "trace_coverage_fraction": self.trace_coverage_fraction,
                "unit_opponent_self_reach_fraction": (
                    self.unit_opponent_self_reach_fraction
                ),
                "finite_importance_weight_fraction": (
                    self.finite_importance_weight_fraction
                ),
                "unique_public_history_count": self.unique_public_history_count,
                "maximum_histories_per_public_information_set": (
                    self.maximum_histories_per_public_information_set
                ),
            },
            "gates": {
                "exact_reach_semantics": self.exact_reach_semantics_gate_passed,
                "native_reach_ledger": self.native_reach_ledger_gate_passed,
                "native_counterfactual_value": (
                    self.native_counterfactual_value_gate_passed
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
            },
            "blockers": [
                "supplied hidden-completion prior is not an official posterior",
                "root-relative traces do not prove upstream official reach",
                "one sampled history per public information set cannot estimate a conditional range",
                "no collection-disjoint counterfactual model or interval calibration",
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeCounterfactualReachAdmission":
        metrics = raw["metrics"]
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            exact_reach_audit_fingerprint=str(raw["exact_reach_audit_fingerprint"]),
            exact_maximum_absolute_error=float(
                metrics["exact_maximum_absolute_error"]
            ),
            exact_error_threshold=float(metrics["exact_error_threshold"]),
            trace_coverage_fraction=float(metrics["trace_coverage_fraction"]),
            unit_opponent_self_reach_fraction=float(
                metrics["unit_opponent_self_reach_fraction"]
            ),
            finite_importance_weight_fraction=float(
                metrics["finite_importance_weight_fraction"]
            ),
            unique_public_history_count=int(metrics["unique_public_history_count"]),
            maximum_histories_per_public_information_set=int(
                metrics["maximum_histories_per_public_information_set"]
            ),
            exact_reach_semantics_gate_passed=bool(
                gates["exact_reach_semantics"]
            ),
            native_reach_ledger_gate_passed=bool(gates["native_reach_ledger"]),
            native_counterfactual_value_gate_passed=bool(
                gates["native_counterfactual_value"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
        )
        if len(raw.get("blockers", ())) != 4:
            raise ValueError("native reach admission blockers changed")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("native reach admission fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativeCounterfactualReachEvidence:
    benchmark_id: str
    status: str
    exact_reach_audit_fingerprint: str
    source_opponent_deviation_dataset_fingerprint: str
    dataset_fingerprint: str
    admission_fingerprint: str
    surface_count: int
    action_target_count: int
    terminal_rollout_count: int
    native_transition_count: int
    exact_maximum_absolute_error: float
    exact_maximum_importance_weight: float
    native_minimum_importance_weight: float
    native_maximum_importance_weight: float
    unique_public_history_count: int
    exact_reach_semantics_gate_passed: bool
    native_reach_ledger_gate_passed: bool
    native_counterfactual_values_calibrated: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6Y evidence identity is missing")
        for field_name in (
            "exact_reach_audit_fingerprint",
            "source_opponent_deviation_dataset_fingerprint",
            "dataset_fingerprint",
            "admission_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if min(
            self.surface_count,
            self.action_target_count,
            self.terminal_rollout_count,
            self.native_transition_count,
            self.unique_public_history_count,
        ) <= 0:
            raise ValueError("Phase 6Y evidence accounting is invalid")
        for field_name in (
            "exact_maximum_absolute_error",
            "exact_maximum_importance_weight",
            "native_minimum_importance_weight",
            "native_maximum_importance_weight",
        ):
            value = float(getattr(self, field_name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not self.exact_reach_semantics_gate_passed or not self.native_reach_ledger_gate_passed:
            raise ValueError("Phase 6Y evidence requires both substrate gates")
        if any(
            (
                self.native_counterfactual_values_calibrated,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6Y evidence cannot enable downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_COUNTERFACTUAL_REACH_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6Y counterfactual-reach evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Exact audit maximum absolute error | `{self.exact_maximum_absolute_error:.3e}` |
| Exact audit maximum importance weight | `{self.exact_maximum_importance_weight:.6f}` |
| Native reach-annotated surfaces | {self.surface_count} |
| Native forced plan-action targets | {self.action_target_count} |
| Genuine terminal continuations | {self.terminal_rollout_count} |
| Genuine native transitions | {self.native_transition_count} |
| Native importance-weight range | `{self.native_minimum_importance_weight:.6f}`–`{self.native_maximum_importance_weight:.6f}` |
| Unique sampled public histories | {self.unique_public_history_count} |
| Exact reach-semantics gate | **{str(self.exact_reach_semantics_gate_passed).lower()}** |
| Native reach-ledger gate | **{str(self.native_reach_ledger_gate_passed).lower()}** |
| Native counterfactual values calibrated | **false** |

The microgame audit proves the reach estimator only on an exact abstraction.
The native ledger is root-relative and its hidden-state prior is supplied, not
official. Each recorded continuation executes through the licensed engine, but
the plan abstraction, prior, rollout policy, estimator, and gates are research
constructs. No resolver or official strategy control is enabled.
"""

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
    ) -> "Phase6NativeCounterfactualReachEvidence":
        fields = {
            name: raw[name]
            for name in cls.__dataclass_fields__
            if name != "claims"
        }
        result = cls(
            **fields,
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6Y evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativeCounterfactualReachArtifacts:
    source_dataset: OpponentDeviationDataset
    source_calibration: OpponentDeviationCalibration
    exact_audit: CounterfactualReachExpectationAudit
    dataset: NativeCounterfactualReachDataset
    admission: NativeCounterfactualReachAdmission
    evidence: Phase6NativeCounterfactualReachEvidence


def _surface_record(
    surface: OpponentDeviationSurface,
    trace: CounterfactualReachTrace,
    particle_probability: float,
) -> NativeCounterfactualReachSurface:
    weight = trace.importance_weight
    return NativeCounterfactualReachSurface(
        source_surface_id=surface.surface_id,
        source_surface_fingerprint=surface.fingerprint,
        source_state_id=surface.source_state_id,
        information_set_id=surface.information_set_id,
        public_history_fingerprint=surface.public_history_fingerprint,
        particle_fingerprint=surface.particle_fingerprint,
        candidate_id_supervision_only=surface.candidate_id_supervision_only,
        split=surface.split,
        variant=surface.variant,
        particle_prior_probability=particle_probability,
        reach_trace=trace,
        action_values=tuple(
            NativeReachCorrectedPlanValue(
                action_id=value.action_id,
                empirical_conditional_value=value.empirical_opponent_value,
                rollout_standard_error=value.rollout_standard_error,
                counterfactual_importance_weight=weight,
                weighted_counterfactual_numerator_contribution=(
                    weight * value.empirical_opponent_value
                ),
                weighted_counterfactual_mass_contribution=weight,
                self_normalized_counterfactual_value=value.empirical_opponent_value,
            )
            for value in surface.action_values
        ),
    )


def run_phase6_native_counterfactual_reach(
    *,
    project_root: str | Path | None = None,
    particle_variants: int = 2,
    rollouts_per_action: int = 10,
    maximum_reach_transitions: int = 16,
    maximum_rollout_transitions: int = 512,
    maximum_tactical_transitions: int = 512,
    exact_error_threshold: float = 1e-12,
    seed: int = PHASE6_NATIVE_COUNTERFACTUAL_REACH_SEED,
) -> Phase6NativeCounterfactualReachArtifacts:
    if exact_error_threshold < 0 or not math.isfinite(exact_error_threshold):
        raise ValueError("Phase 6Y exact error threshold is invalid")
    game = RocketPrizeDuel()
    exact_audit = audit_counterfactual_reach_expectation(game, uniform_profile(game))
    captured: dict[str, tuple[CounterfactualReachTrace, float]] = {}

    def observe(
        surface: OpponentDeviationSurface,
        trace: CounterfactualReachTrace,
        particle_probability: float,
    ) -> None:
        if surface.fingerprint in captured:
            raise RuntimeError("Phase 6Y captured one native surface twice")
        captured[surface.fingerprint] = (trace, particle_probability)

    source = run_phase6_opponent_deviation_values(
        project_root=project_root,
        particle_variants=particle_variants,
        rollouts_per_action=rollouts_per_action,
        maximum_reach_transitions=maximum_reach_transitions,
        maximum_rollout_transitions=maximum_rollout_transitions,
        maximum_tactical_transitions=maximum_tactical_transitions,
        seed=seed,
        reach_observer=observe,
    )
    if set(captured) != {surface.fingerprint for surface in source.dataset.surfaces}:
        raise RuntimeError("Phase 6Y native reach capture is incomplete")
    surfaces = tuple(
        _surface_record(surface, *captured[surface.fingerprint])
        for surface in source.dataset.surfaces
    )
    trace_coverage = len(surfaces) / len(source.dataset.surfaces)
    unit_self_reach = sum(
        math.isclose(surface.reach_trace.player_1_reach, 1.0, abs_tol=1e-12)
        for surface in surfaces
    ) / len(surfaces)
    finite_weights = sum(
        math.isfinite(surface.reach_trace.importance_weight)
        and surface.reach_trace.importance_weight > 0
        for surface in surfaces
    ) / len(surfaces)
    history_counts: dict[str, int] = {}
    for surface in surfaces:
        history_counts[surface.information_set_id] = (
            history_counts.get(surface.information_set_id, 0) + 1
        )
    exact_gate = exact_audit.maximum_absolute_error <= exact_error_threshold
    ledger_gate = all(
        math.isclose(value, 1.0, abs_tol=1e-12)
        for value in (trace_coverage, unit_self_reach, finite_weights)
    )
    dataset = NativeCounterfactualReachDataset(
        source_opponent_deviation_dataset_fingerprint=source.dataset.fingerprint,
        source_opponent_deviation_calibration_fingerprint=source.calibration.fingerprint,
        exact_reach_audit_fingerprint=exact_audit.fingerprint,
        runtime_fingerprint=source.dataset.runtime_fingerprint,
        deck_fingerprint=source.dataset.deck_fingerprint,
        source_tactical_blueprint_fingerprint=(
            source.dataset.source_tactical_blueprint_fingerprint
        ),
        source_plan_blueprint_fingerprint=(
            source.dataset.source_plan_blueprint_fingerprint
        ),
        candidate_ids=source.dataset.candidate_ids,
        rollouts_per_action=source.dataset.rollouts_per_action,
        native_transition_count=source.dataset.native_transition_count,
        surfaces=surfaces,
        exact_reach_semantics_audited=exact_gate,
        native_reach_ledger_complete=ledger_gate,
    )
    admission = NativeCounterfactualReachAdmission(
        dataset_fingerprint=dataset.fingerprint,
        exact_reach_audit_fingerprint=exact_audit.fingerprint,
        exact_maximum_absolute_error=exact_audit.maximum_absolute_error,
        exact_error_threshold=exact_error_threshold,
        trace_coverage_fraction=trace_coverage,
        unit_opponent_self_reach_fraction=unit_self_reach,
        finite_importance_weight_fraction=finite_weights,
        unique_public_history_count=dataset.unique_public_history_count,
        maximum_histories_per_public_information_set=max(history_counts.values()),
        exact_reach_semantics_gate_passed=exact_gate,
        native_reach_ledger_gate_passed=ledger_gate,
        native_counterfactual_value_gate_passed=False,
    )
    claims = (
        ("exact_reach_estimator", "validated_on_abstract_microgame_only"),
        ("source_descendants", "genuine_official_search_execution"),
        ("native_reach", "root_relative_redacted_probability_ledger"),
        ("hidden_completion_prior", "supplied_research_abstraction"),
        ("plan_actions", "turn_level_research_abstraction"),
        ("native_counterfactual_values", "not_calibrated"),
        ("safe_resolver", "not_enabled"),
        ("official_strategy_control", "disabled"),
        ("pokemon_strength", "not_claimed"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6NativeCounterfactualReachEvidence(
        benchmark_id="phase6y-native-counterfactual-reach-v1",
        status="phase6y_reach_semantics_validated_counterfactual_values_rejected_partial",
        exact_reach_audit_fingerprint=exact_audit.fingerprint,
        source_opponent_deviation_dataset_fingerprint=source.dataset.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        admission_fingerprint=admission.fingerprint,
        surface_count=len(dataset.surfaces),
        action_target_count=dataset.action_target_count,
        terminal_rollout_count=dataset.terminal_rollout_count,
        native_transition_count=dataset.native_transition_count,
        exact_maximum_absolute_error=exact_audit.maximum_absolute_error,
        exact_maximum_importance_weight=exact_audit.maximum_importance_weight,
        native_minimum_importance_weight=dataset.minimum_importance_weight,
        native_maximum_importance_weight=dataset.maximum_importance_weight,
        unique_public_history_count=dataset.unique_public_history_count,
        exact_reach_semantics_gate_passed=admission.exact_reach_semantics_gate_passed,
        native_reach_ledger_gate_passed=admission.native_reach_ledger_gate_passed,
        native_counterfactual_values_calibrated=False,
        safe_resolver_enabled=False,
        official_strategy_control_enabled=False,
        claims=claims,
    )
    serialized = dataset.to_dict()
    if _raw_private_key_count(serialized) != 0:
        raise RuntimeError("Phase 6Y serialized a forbidden raw private key")
    restored = NativeCounterfactualReachDataset.from_dict(
        json.loads(json.dumps(serialized))
    )
    if restored.fingerprint != dataset.fingerprint:
        raise RuntimeError("Phase 6Y dataset roundtrip drifted")
    return Phase6NativeCounterfactualReachArtifacts(
        source.dataset,
        source.calibration,
        exact_audit,
        dataset,
        admission,
        evidence,
    )


def main() -> None:
    print(run_phase6_native_counterfactual_reach().evidence.to_json())


if __name__ == "__main__":
    main()
