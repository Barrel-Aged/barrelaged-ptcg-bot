"""Phase 6W collection-heldout public leaf calibration.

The model is trained on one complete native allocation collection, conformal
calibration uses a second complete collection, and validation uses a third
collection generated only after the protocol is frozen. Targets remain
fixed-profile empirical returns rather than opponent counterfactual values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .native_public_leaf import (
    PublicRidgeLeafArtifact,
    PublicRidgeLeafProvider,
    train_public_ridge_leaf,
)
from .phase6_native_leaf_model import (
    LEAF_MODEL_NAMES,
    NativeLeafModelConfig,
    NativeLeafModelDataset,
    NativeLeafModelExample,
    run_phase6_native_leaf_model,
)
from .phase6_native_public_leaf import (
    PUBLIC_LEAF_EVALUATION_NAMES,
    PUBLIC_LEAF_MAXIMUM_MEAN_INTERVAL_WIDTH,
    PUBLIC_LEAF_MAXIMUM_ROLLOUT_STANDARD_ERROR,
    PUBLIC_LEAF_TARGET_COVERAGE,
    Phase6NativePublicLeafEvidence,
    PublicLeafEvaluationMetric,
)


PHASE6_ROOT_HELDOUT_LEAF_SCHEMA_VERSION = 1
PHASE6_ROOT_HELDOUT_LEAF_SEED = 6_029
ROOT_HELDOUT_TRAINING_VARIANTS = (0, 1, 2, 3)
ROOT_HELDOUT_MINIMUM_CALIBRATION_EXAMPLES = 20


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _particle_fingerprints(dataset: NativeLeafModelDataset) -> set[str]:
    return {item.particle_fingerprint for item in dataset.examples}


def _state_fingerprints(dataset: NativeLeafModelDataset) -> set[str]:
    return {item.source_state_id for item in dataset.examples}


def _history_fingerprints(dataset: NativeLeafModelDataset) -> set[str]:
    return {item.public_history_fingerprint for item in dataset.examples}


def _pairwise_overlap_count(
    left: NativeLeafModelDataset,
    right: NativeLeafModelDataset,
) -> int:
    return sum(
        (
            len(_particle_fingerprints(left) & _particle_fingerprints(right)),
            len(_state_fingerprints(left) & _state_fingerprints(right)),
            len(_history_fingerprints(left) & _history_fingerprints(right)),
        )
    )


def _validate_complete_support(dataset: NativeLeafModelDataset) -> None:
    expected = {
        (candidate_id, variant)
        for candidate_id in dataset.candidate_ids
        for variant in ROOT_HELDOUT_TRAINING_VARIANTS
    }
    observed = {
        (item.candidate_id_supervision_only, item.variant)
        for item in dataset.examples
    }
    if observed != expected or len(dataset.examples) != len(expected):
        raise ValueError("root-heldout dataset lacks complete candidate/variant support")


@dataclass(frozen=True, slots=True)
class RootHeldoutLeafCalibration:
    training_dataset_fingerprint: str
    calibration_dataset_fingerprint: str
    validation_dataset_fingerprint: str
    model_fingerprint: str
    source_phase6v_evidence_fingerprint: str
    training_particle_family_fingerprint: str
    calibration_particle_family_fingerprint: str
    validation_particle_family_fingerprint: str
    training_calibration_overlap_count: int
    training_validation_overlap_count: int
    calibration_validation_overlap_count: int
    calibration_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    target_coverage: float
    maximum_mean_interval_width_threshold: float
    maximum_rollout_standard_error_threshold: float
    conformal_rank: int
    conformal_radius: float
    calibration_coverage: float
    validation_coverage: float
    validation_mean_absolute_error: float
    validation_root_mean_square_error: float
    mean_interval_width: float
    informative_interval_fraction: float
    maximum_rollout_standard_error: float
    evaluation_metrics: tuple[PublicLeafEvaluationMetric, ...]
    best_reference_model: str
    best_reference_root_mean_square_error: float
    outperformed_best_reference: bool
    collection_disjoint_validation: bool
    empirical_fixed_profile_leaf_gate_passed: bool
    counterfactual_value_gate_passed: bool = False
    safe_resolver_compatible: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "training_dataset_fingerprint",
            "calibration_dataset_fingerprint",
            "validation_dataset_fingerprint",
            "model_fingerprint",
            "source_phase6v_evidence_fingerprint",
            "training_particle_family_fingerprint",
            "calibration_particle_family_fingerprint",
            "validation_particle_family_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if len(
            {
                self.training_dataset_fingerprint,
                self.calibration_dataset_fingerprint,
                self.validation_dataset_fingerprint,
            }
        ) != 3:
            raise ValueError("root-heldout datasets must be distinct")
        if len(
            {
                self.training_particle_family_fingerprint,
                self.calibration_particle_family_fingerprint,
                self.validation_particle_family_fingerprint,
            }
        ) != 3:
            raise ValueError("root-heldout particle families must be distinct")
        if any(
            value != 0
            for value in (
                self.training_calibration_overlap_count,
                self.training_validation_overlap_count,
                self.calibration_validation_overlap_count,
            )
        ) or not self.collection_disjoint_validation:
            raise ValueError("root-heldout collections overlap")
        if len(self.calibration_example_ids) < ROOT_HELDOUT_MINIMUM_CALIBRATION_EXAMPLES:
            raise ValueError("root-heldout conformal split is too small")
        if not self.validation_example_ids:
            raise ValueError("root-heldout validation split is empty")
        if set(self.calibration_example_ids) & set(self.validation_example_ids):
            raise ValueError("root-heldout calibration and validation IDs overlap")
        if not 0 < self.target_coverage <= 1:
            raise ValueError("root-heldout target coverage is invalid")
        if not 0 < self.conformal_rank < len(self.calibration_example_ids):
            raise ValueError("root-heldout conformal rank must exclude at least one tail row")
        if tuple(item.model_name for item in self.evaluation_metrics) != (
            PUBLIC_LEAF_EVALUATION_NAMES
        ):
            raise ValueError("root-heldout evaluation support/order is invalid")
        reference = min(
            self.evaluation_metrics[1:],
            key=lambda item: (
                item.validation_root_mean_square_error,
                item.model_name,
            ),
        )
        if self.best_reference_model != reference.model_name or not math.isclose(
            self.best_reference_root_mean_square_error,
            reference.validation_root_mean_square_error,
            abs_tol=1e-12,
        ):
            raise ValueError("root-heldout best-reference summary is inconsistent")
        public_rmse = self.evaluation_metrics[0].validation_root_mean_square_error
        if self.outperformed_best_reference != (
            public_rmse < self.best_reference_root_mean_square_error
        ):
            raise ValueError("root-heldout reference comparison is inconsistent")
        for field_name in (
            "target_coverage",
            "maximum_mean_interval_width_threshold",
            "maximum_rollout_standard_error_threshold",
            "conformal_radius",
            "calibration_coverage",
            "validation_coverage",
            "validation_mean_absolute_error",
            "validation_root_mean_square_error",
            "mean_interval_width",
            "informative_interval_fraction",
            "maximum_rollout_standard_error",
            "best_reference_root_mean_square_error",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if any(
            value > 1
            for value in (
                self.target_coverage,
                self.calibration_coverage,
                self.validation_coverage,
                self.informative_interval_fraction,
            )
        ):
            raise ValueError("root-heldout probability metric exceeds one")
        expected_gate = all(
            (
                self.collection_disjoint_validation,
                self.calibration_coverage >= self.target_coverage,
                self.validation_coverage >= self.target_coverage,
                self.mean_interval_width <= self.maximum_mean_interval_width_threshold,
                self.maximum_rollout_standard_error
                <= self.maximum_rollout_standard_error_threshold,
                self.outperformed_best_reference,
            )
        )
        if self.empirical_fixed_profile_leaf_gate_passed != expected_gate:
            raise ValueError("root-heldout empirical gate is inconsistent")
        if self.counterfactual_value_gate_passed or self.safe_resolver_compatible:
            raise ValueError("root-heldout empirical leaf cannot promote safety")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ROOT_HELDOUT_LEAF_SCHEMA_VERSION,
            "datasets": {
                "training": self.training_dataset_fingerprint,
                "calibration": self.calibration_dataset_fingerprint,
                "validation": self.validation_dataset_fingerprint,
                "training_particle_family": self.training_particle_family_fingerprint,
                "calibration_particle_family": self.calibration_particle_family_fingerprint,
                "validation_particle_family": self.validation_particle_family_fingerprint,
                "source_phase6v_evidence": self.source_phase6v_evidence_fingerprint,
            },
            "model_fingerprint": self.model_fingerprint,
            "overlap_counts": {
                "training_calibration": self.training_calibration_overlap_count,
                "training_validation": self.training_validation_overlap_count,
                "calibration_validation": self.calibration_validation_overlap_count,
            },
            "calibration_example_ids": list(self.calibration_example_ids),
            "validation_example_ids": list(self.validation_example_ids),
            "thresholds": {
                "target_coverage": self.target_coverage,
                "maximum_mean_interval_width": self.maximum_mean_interval_width_threshold,
                "maximum_rollout_standard_error": (
                    self.maximum_rollout_standard_error_threshold
                ),
            },
            "metrics": {
                "conformal_rank": self.conformal_rank,
                "conformal_radius": self.conformal_radius,
                "calibration_coverage": self.calibration_coverage,
                "validation_coverage": self.validation_coverage,
                "validation_mean_absolute_error": self.validation_mean_absolute_error,
                "validation_root_mean_square_error": (
                    self.validation_root_mean_square_error
                ),
                "mean_interval_width": self.mean_interval_width,
                "informative_interval_fraction": self.informative_interval_fraction,
                "maximum_rollout_standard_error": self.maximum_rollout_standard_error,
                "evaluation_metrics": [asdict(item) for item in self.evaluation_metrics],
                "best_reference_model": self.best_reference_model,
                "best_reference_root_mean_square_error": (
                    self.best_reference_root_mean_square_error
                ),
                "outperformed_best_reference": self.outperformed_best_reference,
            },
            "gates": {
                "collection_disjoint_validation": self.collection_disjoint_validation,
                "empirical_fixed_profile_leaf": (
                    self.empirical_fixed_profile_leaf_gate_passed
                ),
                "counterfactual_value": self.counterfactual_value_gate_passed,
                "safe_resolver_compatible": self.safe_resolver_compatible,
            },
            "scope": {
                "public_features_only": True,
                "fixed_profile_empirical_return": True,
                "exact_counterfactual_value": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RootHeldoutLeafCalibration":
        datasets = raw["datasets"]
        overlaps = raw["overlap_counts"]
        thresholds = raw["thresholds"]
        metrics = raw["metrics"]
        gates = raw["gates"]
        result = cls(
            training_dataset_fingerprint=str(datasets["training"]),
            calibration_dataset_fingerprint=str(datasets["calibration"]),
            validation_dataset_fingerprint=str(datasets["validation"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            source_phase6v_evidence_fingerprint=str(datasets["source_phase6v_evidence"]),
            training_particle_family_fingerprint=str(datasets["training_particle_family"]),
            calibration_particle_family_fingerprint=str(
                datasets["calibration_particle_family"]
            ),
            validation_particle_family_fingerprint=str(
                datasets["validation_particle_family"]
            ),
            training_calibration_overlap_count=int(overlaps["training_calibration"]),
            training_validation_overlap_count=int(overlaps["training_validation"]),
            calibration_validation_overlap_count=int(overlaps["calibration_validation"]),
            calibration_example_ids=tuple(str(value) for value in raw["calibration_example_ids"]),
            validation_example_ids=tuple(str(value) for value in raw["validation_example_ids"]),
            target_coverage=float(thresholds["target_coverage"]),
            maximum_mean_interval_width_threshold=float(
                thresholds["maximum_mean_interval_width"]
            ),
            maximum_rollout_standard_error_threshold=float(
                thresholds["maximum_rollout_standard_error"]
            ),
            conformal_rank=int(metrics["conformal_rank"]),
            conformal_radius=float(metrics["conformal_radius"]),
            calibration_coverage=float(metrics["calibration_coverage"]),
            validation_coverage=float(metrics["validation_coverage"]),
            validation_mean_absolute_error=float(metrics["validation_mean_absolute_error"]),
            validation_root_mean_square_error=float(
                metrics["validation_root_mean_square_error"]
            ),
            mean_interval_width=float(metrics["mean_interval_width"]),
            informative_interval_fraction=float(metrics["informative_interval_fraction"]),
            maximum_rollout_standard_error=float(metrics["maximum_rollout_standard_error"]),
            evaluation_metrics=tuple(
                PublicLeafEvaluationMetric(**dict(item))
                for item in metrics["evaluation_metrics"]
            ),
            best_reference_model=str(metrics["best_reference_model"]),
            best_reference_root_mean_square_error=float(
                metrics["best_reference_root_mean_square_error"]
            ),
            outperformed_best_reference=bool(metrics["outperformed_best_reference"]),
            collection_disjoint_validation=bool(gates["collection_disjoint_validation"]),
            empirical_fixed_profile_leaf_gate_passed=bool(
                gates["empirical_fixed_profile_leaf"]
            ),
            counterfactual_value_gate_passed=bool(gates["counterfactual_value"]),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("root-heldout leaf calibration fingerprint mismatch")
        return result


def _prediction(
    provider: PublicRidgeLeafProvider,
    item: NativeLeafModelExample,
    model_name: str,
) -> float:
    return (
        provider.predict(item.public_features)
        if model_name == "public_ridge_leaf"
        else item.prediction(model_name)
    )


def evaluate_root_heldout_leaf(
    training_dataset: NativeLeafModelDataset,
    calibration_dataset: NativeLeafModelDataset,
    validation_dataset: NativeLeafModelDataset,
    provider: PublicRidgeLeafProvider,
    artifact: PublicRidgeLeafArtifact,
    *,
    source_phase6v_evidence_fingerprint: str,
    target_coverage: float = PUBLIC_LEAF_TARGET_COVERAGE,
    maximum_mean_interval_width: float = PUBLIC_LEAF_MAXIMUM_MEAN_INTERVAL_WIDTH,
    maximum_rollout_standard_error: float = PUBLIC_LEAF_MAXIMUM_ROLLOUT_STANDARD_ERROR,
    numerical_margin: float = 1e-9,
) -> RootHeldoutLeafCalibration:
    datasets = (training_dataset, calibration_dataset, validation_dataset)
    if artifact.source_dataset_fingerprint != training_dataset.fingerprint:
        raise ValueError("root-heldout model/training mismatch")
    if artifact.trained_on_variants != ROOT_HELDOUT_TRAINING_VARIANTS:
        raise ValueError("root-heldout model did not use the complete training collection")
    if len({item.fingerprint for item in datasets}) != 3:
        raise ValueError("root-heldout datasets are not distinct")
    for dataset in datasets:
        _validate_complete_support(dataset)
    for dataset in datasets[1:]:
        if dataset.deck_fingerprint != training_dataset.deck_fingerprint:
            raise ValueError("root-heldout deck mismatch")
        if dataset.candidate_ids != training_dataset.candidate_ids:
            raise ValueError("root-heldout candidate support mismatch")
        if dataset.runtime_fingerprint != training_dataset.runtime_fingerprint:
            raise ValueError("root-heldout runtime mismatch")
    overlaps = (
        _pairwise_overlap_count(training_dataset, calibration_dataset),
        _pairwise_overlap_count(training_dataset, validation_dataset),
        _pairwise_overlap_count(calibration_dataset, validation_dataset),
    )
    if any(overlaps):
        raise ValueError("root-heldout collection identity overlap")
    calibration = calibration_dataset.examples
    validation = validation_dataset.examples
    errors = sorted(
        abs(provider.predict(item.public_features) - item.empirical_return)
        for item in calibration
    )
    rank = min(len(errors), max(1, math.ceil((len(errors) + 1) * target_coverage)))
    radius = errors[rank - 1] + numerical_margin

    def interval(item: NativeLeafModelExample) -> tuple[float, float]:
        center = provider.predict(item.public_features)
        return max(-1.0, center - radius), min(1.0, center + radius)

    def covered(item: NativeLeafModelExample) -> bool:
        lower, upper = interval(item)
        return lower - 1e-12 <= item.empirical_return <= upper + 1e-12

    widths = [interval(item)[1] - interval(item)[0] for item in validation]
    evaluation_metrics = []
    for name in PUBLIC_LEAF_EVALUATION_NAMES:
        residuals = [
            _prediction(provider, item, name) - item.empirical_return
            for item in validation
        ]
        evaluation_metrics.append(
            PublicLeafEvaluationMetric(
                model_name=name,
                validation_mean_absolute_error=statistics.mean(map(abs, residuals)),
                validation_root_mean_square_error=math.sqrt(
                    statistics.mean(value * value for value in residuals)
                ),
            )
        )
    public_metric = evaluation_metrics[0]
    best_reference = min(
        evaluation_metrics[1:],
        key=lambda item: (
            item.validation_root_mean_square_error,
            item.model_name,
        ),
    )
    outperformed = (
        public_metric.validation_root_mean_square_error
        < best_reference.validation_root_mean_square_error
    )
    calibration_coverage = sum(covered(item) for item in calibration) / len(calibration)
    validation_coverage = sum(covered(item) for item in validation) / len(validation)
    mean_width = statistics.mean(widths)
    maximum_standard_error = max(
        item.rollout_standard_error for item in calibration + validation
    )
    gate = all(
        (
            rank < len(calibration),
            calibration_coverage >= target_coverage,
            validation_coverage >= target_coverage,
            mean_width <= maximum_mean_interval_width,
            maximum_standard_error <= maximum_rollout_standard_error,
            outperformed,
        )
    )
    return RootHeldoutLeafCalibration(
        training_dataset_fingerprint=training_dataset.fingerprint,
        calibration_dataset_fingerprint=calibration_dataset.fingerprint,
        validation_dataset_fingerprint=validation_dataset.fingerprint,
        model_fingerprint=artifact.fingerprint,
        source_phase6v_evidence_fingerprint=source_phase6v_evidence_fingerprint,
        training_particle_family_fingerprint=training_dataset.particle_family_fingerprint,
        calibration_particle_family_fingerprint=(
            calibration_dataset.particle_family_fingerprint
        ),
        validation_particle_family_fingerprint=(
            validation_dataset.particle_family_fingerprint
        ),
        training_calibration_overlap_count=overlaps[0],
        training_validation_overlap_count=overlaps[1],
        calibration_validation_overlap_count=overlaps[2],
        calibration_example_ids=tuple(item.example_id for item in calibration),
        validation_example_ids=tuple(item.example_id for item in validation),
        target_coverage=target_coverage,
        maximum_mean_interval_width_threshold=maximum_mean_interval_width,
        maximum_rollout_standard_error_threshold=maximum_rollout_standard_error,
        conformal_rank=rank,
        conformal_radius=radius,
        calibration_coverage=calibration_coverage,
        validation_coverage=validation_coverage,
        validation_mean_absolute_error=public_metric.validation_mean_absolute_error,
        validation_root_mean_square_error=public_metric.validation_root_mean_square_error,
        mean_interval_width=mean_width,
        informative_interval_fraction=sum(value < 2.0 - 1e-12 for value in widths)
        / len(widths),
        maximum_rollout_standard_error=maximum_standard_error,
        evaluation_metrics=tuple(evaluation_metrics),
        best_reference_model=best_reference.model_name,
        best_reference_root_mean_square_error=(
            best_reference.validation_root_mean_square_error
        ),
        outperformed_best_reference=outperformed,
        collection_disjoint_validation=True,
        empirical_fixed_profile_leaf_gate_passed=gate,
    )


@dataclass(frozen=True, slots=True)
class Phase6RootHeldoutLeafEvidence:
    benchmark_id: str
    status: str
    model_fingerprint: str
    calibration_fingerprint: str
    training_dataset_fingerprint: str
    calibration_dataset_fingerprint: str
    validation_dataset_fingerprint: str
    selected_regularization: float
    training_examples: int
    calibration_examples: int
    validation_examples: int
    fresh_terminal_rollout_count: int
    fresh_native_transition_count: int
    target_coverage: float
    maximum_mean_interval_width_threshold: float
    maximum_rollout_standard_error_threshold: float
    conformal_rank: int
    conformal_radius: float
    calibration_coverage: float
    validation_coverage: float
    validation_mean_absolute_error: float
    validation_root_mean_square_error: float
    best_reference_model: str
    best_reference_root_mean_square_error: float
    mean_interval_width: float
    informative_interval_fraction: float
    maximum_rollout_standard_error: float
    collection_disjoint_validation: bool
    outperformed_best_reference: bool
    empirical_fixed_profile_leaf_gate_passed: bool
    counterfactual_values_calibrated: bool
    public_belief_calibrated: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "model_fingerprint",
            "calibration_fingerprint",
            "training_dataset_fingerprint",
            "calibration_dataset_fingerprint",
            "validation_dataset_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.benchmark_id or not self.status or not self.best_reference_model:
            raise ValueError("Phase 6W evidence identity is incomplete")
        if min(
            self.training_examples,
            self.calibration_examples,
            self.validation_examples,
            self.fresh_terminal_rollout_count,
            self.fresh_native_transition_count,
            self.conformal_rank,
        ) <= 0:
            raise ValueError("Phase 6W evidence accounting is invalid")
        if len(
            {
                self.training_dataset_fingerprint,
                self.calibration_dataset_fingerprint,
                self.validation_dataset_fingerprint,
            }
        ) != 3 or not self.collection_disjoint_validation:
            raise ValueError("Phase 6W evidence lacks collection-disjoint validation")
        if self.outperformed_best_reference != (
            self.validation_root_mean_square_error
            < self.best_reference_root_mean_square_error
        ):
            raise ValueError("Phase 6W evidence reference comparison is inconsistent")
        for field_name in (
            "selected_regularization",
            "target_coverage",
            "maximum_mean_interval_width_threshold",
            "maximum_rollout_standard_error_threshold",
            "conformal_radius",
            "calibration_coverage",
            "validation_coverage",
            "validation_mean_absolute_error",
            "validation_root_mean_square_error",
            "best_reference_root_mean_square_error",
            "mean_interval_width",
            "informative_interval_fraction",
            "maximum_rollout_standard_error",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if any(
            value > 1
            for value in (
                self.target_coverage,
                self.calibration_coverage,
                self.validation_coverage,
                self.informative_interval_fraction,
            )
        ):
            raise ValueError("Phase 6W evidence probability metric exceeds one")
        expected_gate = all(
            (
                self.collection_disjoint_validation,
                self.conformal_rank < self.calibration_examples,
                self.calibration_coverage >= self.target_coverage,
                self.validation_coverage >= self.target_coverage,
                self.mean_interval_width <= self.maximum_mean_interval_width_threshold,
                self.maximum_rollout_standard_error
                <= self.maximum_rollout_standard_error_threshold,
                self.outperformed_best_reference,
            )
        )
        if self.empirical_fixed_profile_leaf_gate_passed != expected_gate:
            raise ValueError("Phase 6W evidence empirical gate is inconsistent")
        if any(
            (
                self.counterfactual_values_calibrated,
                self.public_belief_calibrated,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6W empirical evidence cannot enable safety")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ROOT_HELDOUT_LEAF_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6W collection-heldout public leaf evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Training / calibration / validation rows | {self.training_examples} / {self.calibration_examples} / {self.validation_examples} |
| Selected ridge regularization | {self.selected_regularization:.1f} |
| Fresh validation terminal continuations | {self.fresh_terminal_rollout_count} |
| Fresh validation native transitions | {self.fresh_native_transition_count} |
| Conformal rank / radius | {self.conformal_rank} / {self.conformal_radius:.6f} |
| Validation coverage | {self.validation_coverage:.3%} |
| Validation MAE / RMSE | {self.validation_mean_absolute_error:.6f} / {self.validation_root_mean_square_error:.6f} |
| Best reference / RMSE | {self.best_reference_model} / {self.best_reference_root_mean_square_error:.6f} |
| Beat best reference RMSE | **{str(self.outperformed_best_reference).lower()}** |
| Mean interval width | {self.mean_interval_width:.6f} |
| Informative interval fraction | {self.informative_interval_fraction:.3%} |
| Empirical fixed-profile leaf gate | **{str(self.empirical_fixed_profile_leaf_gate_passed).lower()}** |
| Counterfactual values calibrated | **false** |

The three complete allocation collections are identity-disjoint. Native
descendant execution is genuine; the fitted scalar and fixed-profile targets
remain abstractions and cannot enable safe resolving.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6RootHeldoutLeafEvidence":
        kwargs = {name: raw[name] for name in cls.__dataclass_fields__ if name != "claims"}
        kwargs["claims"] = tuple(
            (str(key), str(value)) for key, value in raw["claims"].items()
        )
        result = cls(**kwargs)
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6W evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6RootHeldoutLeafArtifacts:
    validation_dataset: NativeLeafModelDataset
    model: PublicRidgeLeafArtifact
    calibration: RootHeldoutLeafCalibration
    evidence: Phase6RootHeldoutLeafEvidence


def _load_dataset(path: Path) -> NativeLeafModelDataset:
    return NativeLeafModelDataset.from_dict(json.loads(path.read_text(encoding="utf-8")))


def run_phase6_root_heldout_leaf(
    *,
    project_root: str | Path | None = None,
    training_dataset_path: str | Path | None = None,
    calibration_dataset_path: str | Path | None = None,
    source_phase6v_evidence_path: str | Path | None = None,
    validation_dataset_path: str | Path | None = None,
    seed: int = PHASE6_ROOT_HELDOUT_LEAF_SEED,
) -> Phase6RootHeldoutLeafArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    training_path = Path(
        training_dataset_path
        or root
        / "artifacts"
        / "phase6_native_leaf_model"
        / "native_low_noise_leaf_dataset.json"
    ).resolve()
    calibration_path = Path(
        calibration_dataset_path
        or root
        / "artifacts"
        / "phase6_native_public_leaf"
        / "fresh_native_leaf_dataset.json"
    ).resolve()
    phase6v_path = Path(
        source_phase6v_evidence_path
        or root
        / "artifacts"
        / "phase6_native_public_leaf"
        / "phase6v_evidence.json"
    ).resolve()
    training_dataset = _load_dataset(training_path)
    calibration_dataset = _load_dataset(calibration_path)
    phase6v = Phase6NativePublicLeafEvidence.from_dict(
        json.loads(phase6v_path.read_text(encoding="utf-8"))
    )
    if phase6v.fresh_holdout_dataset_fingerprint != calibration_dataset.fingerprint:
        raise ValueError("Phase 6W calibration collection/Phase 6V evidence mismatch")
    provider, model = train_public_ridge_leaf(
        training_dataset,
        trained_variants=ROOT_HELDOUT_TRAINING_VARIANTS,
    )
    if validation_dataset_path is None:
        validation_dataset = run_phase6_native_leaf_model(
            project_root=root,
            seed=seed,
            config=NativeLeafModelConfig(),
        ).dataset
    else:
        validation_dataset = _load_dataset(Path(validation_dataset_path).resolve())
    calibration = evaluate_root_heldout_leaf(
        training_dataset,
        calibration_dataset,
        validation_dataset,
        provider,
        model,
        source_phase6v_evidence_fingerprint=phase6v.fingerprint,
    )
    claims = (
        ("model", "public_only_ridge_fixed_profile_leaf"),
        ("training", "complete_phase6u_collection"),
        ("calibration", "complete_phase6v_collection"),
        ("validation", "fresh_complete_collection"),
        ("collection_disjoint", "true"),
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
    evidence = Phase6RootHeldoutLeafEvidence(
        benchmark_id="phase6w-root-heldout-public-leaf-v1",
        status=(
            "phase6w_root_heldout_public_leaf_passed_empirical_only"
            if calibration.empirical_fixed_profile_leaf_gate_passed
            else "phase6w_root_heldout_public_leaf_rejected_validated_partial"
        ),
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        training_dataset_fingerprint=training_dataset.fingerprint,
        calibration_dataset_fingerprint=calibration_dataset.fingerprint,
        validation_dataset_fingerprint=validation_dataset.fingerprint,
        selected_regularization=model.selected_regularization,
        training_examples=len(training_dataset.examples),
        calibration_examples=len(calibration_dataset.examples),
        validation_examples=len(validation_dataset.examples),
        fresh_terminal_rollout_count=(
            len(validation_dataset.examples) * validation_dataset.rollouts_per_leaf
        ),
        fresh_native_transition_count=validation_dataset.native_transition_count,
        target_coverage=calibration.target_coverage,
        maximum_mean_interval_width_threshold=(
            calibration.maximum_mean_interval_width_threshold
        ),
        maximum_rollout_standard_error_threshold=(
            calibration.maximum_rollout_standard_error_threshold
        ),
        conformal_rank=calibration.conformal_rank,
        conformal_radius=calibration.conformal_radius,
        calibration_coverage=calibration.calibration_coverage,
        validation_coverage=calibration.validation_coverage,
        validation_mean_absolute_error=calibration.validation_mean_absolute_error,
        validation_root_mean_square_error=(
            calibration.validation_root_mean_square_error
        ),
        best_reference_model=calibration.best_reference_model,
        best_reference_root_mean_square_error=(
            calibration.best_reference_root_mean_square_error
        ),
        mean_interval_width=calibration.mean_interval_width,
        informative_interval_fraction=calibration.informative_interval_fraction,
        maximum_rollout_standard_error=calibration.maximum_rollout_standard_error,
        collection_disjoint_validation=calibration.collection_disjoint_validation,
        outperformed_best_reference=calibration.outperformed_best_reference,
        empirical_fixed_profile_leaf_gate_passed=(
            calibration.empirical_fixed_profile_leaf_gate_passed
        ),
        counterfactual_values_calibrated=False,
        public_belief_calibrated=False,
        safe_resolver_enabled=False,
        official_strategy_control_enabled=False,
        claims=claims,
    )
    return Phase6RootHeldoutLeafArtifacts(validation_dataset, model, calibration, evidence)


def main() -> None:
    print(run_phase6_root_heldout_leaf().evidence.to_json())


if __name__ == "__main__":
    main()
