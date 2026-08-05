"""Fail-closed admission gate for native plan safety work.

This module deliberately does not construct counterfactual-value bounds from
Phase 6P's unstable samples. It makes every necessary precondition explicit and
machine-checkable so an abstract training checkpoint cannot silently enter the
safe resolver or official strategy path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .native_plan_blueprint import NativePlanBlueprintArtifact, NativePlanOutcomeDataset
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence


NATIVE_PLAN_SAFETY_ADMISSION_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NativePlanSafetyThresholds:
    """Necessary engineering gates, never a sufficient safety certificate."""

    minimum_sampling_iterations: int = 20
    minimum_player_regret_samples: int = 5_000
    minimum_strategy_samples: int = 5_000
    maximum_raw_regret_p95_absolute: float = 100_000.0
    maximum_raw_regret_absolute: float = 100_000_000.0
    maximum_restore_drift: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.minimum_sampling_iterations,
            self.minimum_player_regret_samples,
            self.minimum_strategy_samples,
        ) <= 0:
            raise ValueError("native safety minimums must be positive")
        for name in (
            "maximum_raw_regret_p95_absolute",
            "maximum_raw_regret_absolute",
            "maximum_restore_drift",
        ):
            value = float(getattr(self, name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class NativePlanSafetyPrerequisites:
    """Independent evidence that cannot be inferred from a training loss."""

    public_belief_calibrated: bool = False
    heldout_particles_validated: bool = False
    full_game_regret_proxy_validated: bool = False
    counterfactual_values_independently_calibrated: bool = False


@dataclass(frozen=True, slots=True)
class NativePlanSafetyCheck:
    check_id: str
    passed: bool
    observed: str
    required: str
    reason: str

    def __post_init__(self) -> None:
        if not all((self.check_id, self.observed, self.required, self.reason)):
            raise ValueError("native safety checks must be fully described")


class NativePlanSafetyAdmissionError(RuntimeError):
    """Raised when a caller tries to bypass a rejected admission decision."""


@dataclass(frozen=True, slots=True)
class NativePlanSafetyAdmission:
    dataset_fingerprint: str
    model_fingerprint: str
    evidence_fingerprint: str
    thresholds: NativePlanSafetyThresholds
    prerequisites: NativePlanSafetyPrerequisites
    checks: tuple[NativePlanSafetyCheck, ...]
    admitted: bool
    counterfactual_bounds_enabled: bool
    safe_resolver_enabled: bool
    exploitation_enabled: bool
    official_strategy_control_enabled: bool

    def __post_init__(self) -> None:
        for name in (
            "dataset_fingerprint",
            "model_fingerprint",
            "evidence_fingerprint",
        ):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not self.checks or len({item.check_id for item in self.checks}) != len(
            self.checks
        ):
            raise ValueError("native safety checks must be nonempty and unique")
        if self.admitted != all(item.passed for item in self.checks):
            raise ValueError("native safety admission must equal all required checks")
        enabled = (
            self.counterfactual_bounds_enabled,
            self.safe_resolver_enabled,
            self.exploitation_enabled,
            self.official_strategy_control_enabled,
        )
        if any(enabled) and not self.admitted:
            raise ValueError("rejected native safety admission cannot enable capabilities")
        if self.safe_resolver_enabled and not self.counterfactual_bounds_enabled:
            raise ValueError("native safe resolver requires counterfactual bounds")
        if self.exploitation_enabled and not self.safe_resolver_enabled:
            raise ValueError("native exploitation requires an admitted safe resolver")

    @property
    def failed_checks(self) -> tuple[NativePlanSafetyCheck, ...]:
        return tuple(item for item in self.checks if not item.passed)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def require_admitted(self) -> None:
        if not self.admitted:
            identifiers = ", ".join(item.check_id for item in self.failed_checks)
            raise NativePlanSafetyAdmissionError(
                f"native plan safety admission rejected: {identifiers}"
            )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": NATIVE_PLAN_SAFETY_ADMISSION_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "thresholds": asdict(self.thresholds),
            "prerequisites": asdict(self.prerequisites),
            "checks": [asdict(item) for item in self.checks],
            "admitted": self.admitted,
            "enabled": {
                "counterfactual_bounds": self.counterfactual_bounds_enabled,
                "safe_resolver": self.safe_resolver_enabled,
                "exploitation": self.exploitation_enabled,
                "official_strategy_control": self.official_strategy_control_enabled,
            },
            "failed_check_ids": [item.check_id for item in self.failed_checks],
            "claim_boundary": {
                "necessary_not_sufficient_gate": True,
                "safety_certificate": False,
                "official_engine_strategy": False,
                "pokemon_strength": "not_established",
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativePlanSafetyAdmission":
        if int(raw.get("schema_version", -1)) != NATIVE_PLAN_SAFETY_ADMISSION_SCHEMA_VERSION:
            raise ValueError("unsupported native safety admission schema")
        enabled = raw["enabled"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            evidence_fingerprint=str(raw["evidence_fingerprint"]),
            thresholds=NativePlanSafetyThresholds(**dict(raw["thresholds"])),
            prerequisites=NativePlanSafetyPrerequisites(**dict(raw["prerequisites"])),
            checks=tuple(NativePlanSafetyCheck(**dict(item)) for item in raw["checks"]),
            admitted=bool(raw["admitted"]),
            counterfactual_bounds_enabled=bool(enabled["counterfactual_bounds"]),
            safe_resolver_enabled=bool(enabled["safe_resolver"]),
            exploitation_enabled=bool(enabled["exploitation"]),
            official_strategy_control_enabled=bool(
                enabled["official_strategy_control"]
            ),
        )
        supplied = raw.get("fingerprint")
        if supplied is not None and str(supplied) != result.fingerprint:
            raise ValueError("native safety admission fingerprint mismatch")
        if tuple(raw.get("failed_check_ids", ())) != tuple(
            item.check_id for item in result.failed_checks
        ):
            raise ValueError("native safety failed-check list mismatch")
        return result


def evaluate_native_plan_safety_admission(
    dataset: NativePlanOutcomeDataset,
    model: NativePlanBlueprintArtifact,
    evidence: Phase6NativePlanIterationsEvidence,
    *,
    thresholds: NativePlanSafetyThresholds | None = None,
    prerequisites: NativePlanSafetyPrerequisites | None = None,
) -> NativePlanSafetyAdmission:
    """Evaluate necessary gates and enable nothing unless every one passes."""

    thresholds = thresholds or NativePlanSafetyThresholds()
    prerequisites = prerequisites or NativePlanSafetyPrerequisites()
    if model.dataset_fingerprint != dataset.fingerprint:
        raise ValueError("native safety model/dataset fingerprint mismatch")
    if evidence.final_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("native safety evidence/dataset fingerprint mismatch")
    if evidence.final_model_fingerprint != model.fingerprint:
        raise ValueError("native safety evidence/model fingerprint mismatch")

    def check(
        identifier: str,
        passed: bool,
        observed: Any,
        required: Any,
        reason: str,
    ) -> NativePlanSafetyCheck:
        return NativePlanSafetyCheck(
            identifier,
            bool(passed),
            str(observed).lower() if isinstance(observed, bool) else str(observed),
            str(required).lower() if isinstance(required, bool) else str(required),
            reason,
        )

    checks = (
        check(
            "genuine_native_descendant_source",
            evidence.source_official_engine_executed,
            evidence.source_official_engine_executed,
            True,
            "training labels must trace to genuine native descendant execution",
        ),
        check(
            "private_data_redaction",
            dataset.raw_private_card_ids_persisted == 0,
            dataset.raw_private_card_ids_persisted,
            0,
            "serialized training data must contain no raw hidden-zone identities",
        ),
        check(
            "checkpoint_restore_drift",
            evidence.maximum_restore_drift <= thresholds.maximum_restore_drift,
            evidence.maximum_restore_drift,
            f"<= {thresholds.maximum_restore_drift}",
            "restored policy must be identical before any safety audit",
        ),
        check(
            "sampling_iterations",
            dataset.iterations >= thresholds.minimum_sampling_iterations,
            dataset.iterations,
            f">= {thresholds.minimum_sampling_iterations}",
            "three development rounds are insufficient for a convergence trend",
        ),
        check(
            "player_0_sample_count",
            dataset.player_0_regrets.seen
            >= thresholds.minimum_player_regret_samples,
            dataset.player_0_regrets.seen,
            f">= {thresholds.minimum_player_regret_samples}",
            "player-0 counterfactual supervision is too small",
        ),
        check(
            "player_1_sample_count",
            dataset.player_1_regrets.seen
            >= thresholds.minimum_player_regret_samples,
            dataset.player_1_regrets.seen,
            f">= {thresholds.minimum_player_regret_samples}",
            "player-1 counterfactual supervision is too small",
        ),
        check(
            "average_strategy_sample_count",
            dataset.average_strategy.seen >= thresholds.minimum_strategy_samples,
            dataset.average_strategy.seen,
            f">= {thresholds.minimum_strategy_samples}",
            "average-strategy supervision is too small",
        ),
        check(
            "regret_p95_variance",
            evidence.raw_regret_p95_absolute
            <= thresholds.maximum_raw_regret_p95_absolute,
            evidence.raw_regret_p95_absolute,
            f"<= {thresholds.maximum_raw_regret_p95_absolute}",
            "importance-weight p95 is too large for value calibration",
        ),
        check(
            "regret_maximum_variance",
            evidence.raw_regret_maximum_absolute
            <= thresholds.maximum_raw_regret_absolute,
            evidence.raw_regret_maximum_absolute,
            f"<= {thresholds.maximum_raw_regret_absolute}",
            "extreme importance weights dominate the raw target scale",
        ),
        check(
            "public_belief_calibration",
            prerequisites.public_belief_calibrated,
            prerequisites.public_belief_calibrated,
            True,
            "the supplied six-particle prior is not history-conditioned/calibrated",
        ),
        check(
            "heldout_particle_validation",
            prerequisites.heldout_particles_validated,
            prerequisites.heldout_particles_validated,
            True,
            "training particles cannot double as the abstraction-fidelity test",
        ),
        check(
            "full_game_regret_proxy",
            prerequisites.full_game_regret_proxy_validated,
            prerequisites.full_game_regret_proxy_validated,
            True,
            "no defensible full-game regret, NashConv, or exploitability proxy exists",
        ),
        check(
            "independent_counterfactual_calibration",
            prerequisites.counterfactual_values_independently_calibrated,
            prerequisites.counterfactual_values_independently_calibrated,
            True,
            "counterfactual bounds require independent held-out value targets",
        ),
        check(
            "model_counterfactual_value_capability",
            model.calibrated_counterfactual_values,
            model.calibrated_counterfactual_values,
            True,
            "the native plan provider intentionally refuses counterfactual values",
        ),
        check(
            "model_safe_resolver_compatibility",
            model.safe_resolver_compatible,
            model.safe_resolver_compatible,
            True,
            "the checkpoint capability flag is fail-closed",
        ),
        check(
            "equilibrium_convergence_validation",
            model.equilibrium_convergence_validated,
            model.equilibrium_convergence_validated,
            True,
            "transformed-loss improvement is not an equilibrium certificate",
        ),
    )
    admitted = all(item.passed for item in checks)
    return NativePlanSafetyAdmission(
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        evidence_fingerprint=evidence.fingerprint,
        thresholds=thresholds,
        prerequisites=prerequisites,
        checks=checks,
        admitted=admitted,
        counterfactual_bounds_enabled=False,
        safe_resolver_enabled=False,
        exploitation_enabled=False,
        official_strategy_control_enabled=False,
    )
