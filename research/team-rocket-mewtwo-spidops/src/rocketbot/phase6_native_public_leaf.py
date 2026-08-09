"""Phase 6V purpose-trained public leaf with fresh-root validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from .native_public_leaf import (
    PublicRidgeLeafArtifact,
    PublicRidgeLeafProvider,
    train_public_ridge_leaf,
)
from .phase6_native_leaf_model import (
    LEAF_MODEL_NAMES,
    NativeLeafModelDataset,
    NativeLeafModelExample,
    NativeLeafModelConfig,
    run_phase6_native_leaf_model,
)


PHASE6_NATIVE_PUBLIC_LEAF_SCHEMA_VERSION = 1
PHASE6_NATIVE_PUBLIC_LEAF_SEED = 6_028
PUBLIC_LEAF_EVALUATION_NAMES = ("public_ridge_leaf",) + LEAF_MODEL_NAMES
PUBLIC_LEAF_TARGET_COVERAGE = 0.90
PUBLIC_LEAF_MAXIMUM_MEAN_INTERVAL_WIDTH = 1.25
PUBLIC_LEAF_MAXIMUM_ROLLOUT_STANDARD_ERROR = 0.35


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PublicLeafEvaluationMetric:
    model_name: str
    validation_mean_absolute_error: float
    validation_root_mean_square_error: float

    def __post_init__(self) -> None:
        if self.model_name not in PUBLIC_LEAF_EVALUATION_NAMES:
            raise ValueError("public leaf evaluation model is invalid")
        for field_name in (
            "validation_mean_absolute_error",
            "validation_root_mean_square_error",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("public leaf evaluation metric is invalid")


@dataclass(frozen=True, slots=True)
class NativePublicLeafCalibration:
    training_dataset_fingerprint: str
    fresh_holdout_dataset_fingerprint: str
    model_fingerprint: str
    calibration_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    calibration_particle_family_fingerprint: str
    training_holdout_overlap_count: int
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
    empirical_fixed_profile_leaf_gate_passed: bool
    fresh_root_validation: bool
    counterfactual_value_gate_passed: bool = False
    safe_resolver_compatible: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "training_dataset_fingerprint",
            "fresh_holdout_dataset_fingerprint",
            "model_fingerprint",
            "calibration_particle_family_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.calibration_example_ids or not self.validation_example_ids:
            raise ValueError("public leaf requires calibration and validation examples")
        if set(self.calibration_example_ids) & set(self.validation_example_ids):
            raise ValueError("public leaf calibration/validation overlap")
        if self.training_holdout_overlap_count != 0 or not self.fresh_root_validation:
            raise ValueError("public leaf requires a disjoint fresh-root holdout")
        if not 0 < self.target_coverage <= 1 or self.conformal_rank <= 0:
            raise ValueError("public leaf coverage/rank metadata is invalid")
        if tuple(item.model_name for item in self.evaluation_metrics) != PUBLIC_LEAF_EVALUATION_NAMES:
            raise ValueError("public leaf evaluation metrics have invalid support/order")
        reference_metrics = self.evaluation_metrics[1:]
        best_reference = min(
            reference_metrics,
            key=lambda item: (
                item.validation_root_mean_square_error,
                item.model_name,
            ),
        )
        if self.best_reference_model != best_reference.model_name or not math.isclose(
            self.best_reference_root_mean_square_error,
            best_reference.validation_root_mean_square_error,
            abs_tol=1e-12,
        ):
            raise ValueError("public leaf best-reference summary is inconsistent")
        public_rmse = self.evaluation_metrics[0].validation_root_mean_square_error
        if self.outperformed_best_reference != (
            public_rmse < self.best_reference_root_mean_square_error
        ):
            raise ValueError("public leaf reference-comparison flag is inconsistent")
        for field_name in (
            "conformal_radius",
            "calibration_coverage",
            "validation_coverage",
            "validation_mean_absolute_error",
            "validation_root_mean_square_error",
            "mean_interval_width",
            "informative_interval_fraction",
            "maximum_rollout_standard_error",
            "best_reference_root_mean_square_error",
            "maximum_mean_interval_width_threshold",
            "maximum_rollout_standard_error_threshold",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if any(
            value > 1
            for value in (
                self.calibration_coverage,
                self.validation_coverage,
                self.informative_interval_fraction,
            )
        ):
            raise ValueError("public leaf probability metric exceeds one")
        expected_gate = all(
            (
                self.training_holdout_overlap_count == 0,
                self.fresh_root_validation,
                self.calibration_coverage >= self.target_coverage,
                self.validation_coverage >= self.target_coverage,
                self.mean_interval_width <= self.maximum_mean_interval_width_threshold,
                self.maximum_rollout_standard_error
                <= self.maximum_rollout_standard_error_threshold,
                self.outperformed_best_reference,
            )
        )
        if self.empirical_fixed_profile_leaf_gate_passed != expected_gate:
            raise ValueError("public leaf gate is inconsistent with its declared thresholds")
        if self.counterfactual_value_gate_passed or self.safe_resolver_compatible:
            raise ValueError("fixed-profile public leaf cannot promote safety capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_PUBLIC_LEAF_SCHEMA_VERSION,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "fresh_holdout_dataset_fingerprint": self.fresh_holdout_dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "calibration_example_ids": list(self.calibration_example_ids),
            "validation_example_ids": list(self.validation_example_ids),
            "calibration_particle_family_fingerprint": (
                self.calibration_particle_family_fingerprint
            ),
            "training_holdout_overlap_count": self.training_holdout_overlap_count,
            "target_coverage": self.target_coverage,
            "thresholds": {
                "maximum_mean_interval_width": (
                    self.maximum_mean_interval_width_threshold
                ),
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
                "validation_root_mean_square_error": self.validation_root_mean_square_error,
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
                "empirical_fixed_profile_leaf": self.empirical_fixed_profile_leaf_gate_passed,
                "fresh_root_validation": self.fresh_root_validation,
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativePublicLeafCalibration":
        metrics = raw["metrics"]
        gates = raw["gates"]
        thresholds = raw["thresholds"]
        result = cls(
            training_dataset_fingerprint=str(raw["training_dataset_fingerprint"]),
            fresh_holdout_dataset_fingerprint=str(raw["fresh_holdout_dataset_fingerprint"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            calibration_example_ids=tuple(str(value) for value in raw["calibration_example_ids"]),
            validation_example_ids=tuple(str(value) for value in raw["validation_example_ids"]),
            calibration_particle_family_fingerprint=str(
                raw["calibration_particle_family_fingerprint"]
            ),
            training_holdout_overlap_count=int(raw["training_holdout_overlap_count"]),
            target_coverage=float(raw["target_coverage"]),
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
            validation_root_mean_square_error=float(metrics["validation_root_mean_square_error"]),
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
            empirical_fixed_profile_leaf_gate_passed=bool(
                gates["empirical_fixed_profile_leaf"]
            ),
            fresh_root_validation=bool(gates["fresh_root_validation"]),
            counterfactual_value_gate_passed=bool(gates["counterfactual_value"]),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("native public leaf calibration fingerprint mismatch")
        return result


def _evaluate_public_leaf(
    training_dataset: NativeLeafModelDataset,
    fresh_dataset: NativeLeafModelDataset,
    provider: PublicRidgeLeafProvider,
    artifact: PublicRidgeLeafArtifact,
    *,
    target_coverage: float = PUBLIC_LEAF_TARGET_COVERAGE,
    maximum_mean_interval_width: float = PUBLIC_LEAF_MAXIMUM_MEAN_INTERVAL_WIDTH,
    maximum_rollout_standard_error: float = PUBLIC_LEAF_MAXIMUM_ROLLOUT_STANDARD_ERROR,
    numerical_margin: float = 1e-9,
) -> NativePublicLeafCalibration:
    if training_dataset.fingerprint != artifact.source_dataset_fingerprint:
        raise ValueError("public leaf model/training dataset mismatch")
    if training_dataset.fingerprint == fresh_dataset.fingerprint:
        raise ValueError("public leaf holdout reused the training dataset")
    if training_dataset.deck_fingerprint != fresh_dataset.deck_fingerprint:
        raise ValueError("public leaf holdout deck mismatch")
    if training_dataset.candidate_ids != fresh_dataset.candidate_ids:
        raise ValueError("public leaf holdout candidate support mismatch")
    if training_dataset.particle_family_fingerprint == fresh_dataset.particle_family_fingerprint:
        raise ValueError("public leaf holdout reused the training particle family")
    training_particles = {item.particle_fingerprint for item in training_dataset.examples}
    fresh_particles = {item.particle_fingerprint for item in fresh_dataset.examples}
    overlap = len(training_particles & fresh_particles)
    calibration = tuple(item for item in fresh_dataset.examples if item.variant == 2)
    validation = tuple(item for item in fresh_dataset.examples if item.variant == 3)
    if len(calibration) != len(fresh_dataset.candidate_ids) or len(validation) != len(
        fresh_dataset.candidate_ids
    ):
        raise ValueError("public leaf fresh holdout lacks one row per candidate/split")
    calibration_errors = sorted(
        abs(provider.predict(item.public_features) - item.empirical_return)
        for item in calibration
    )
    rank = min(
        len(calibration_errors),
        max(1, math.ceil((len(calibration_errors) + 1) * target_coverage)),
    )
    radius = calibration_errors[rank - 1] + numerical_margin

    def prediction(item: NativeLeafModelExample, name: str) -> float:
        return (
            provider.predict(item.public_features)
            if name == "public_ridge_leaf"
            else item.prediction(name)
        )

    def interval(item: NativeLeafModelExample) -> tuple[float, float]:
        center = provider.predict(item.public_features)
        return max(-1.0, center - radius), min(1.0, center + radius)

    def covered(item: NativeLeafModelExample) -> bool:
        lower, upper = interval(item)
        return lower - 1e-12 <= item.empirical_return <= upper + 1e-12

    widths = [interval(item)[1] - interval(item)[0] for item in validation]
    evaluation_metrics = []
    for name in PUBLIC_LEAF_EVALUATION_NAMES:
        errors = [
            abs(prediction(item, name) - item.empirical_return)
            for item in validation
        ]
        evaluation_metrics.append(
            PublicLeafEvaluationMetric(
                model_name=name,
                validation_mean_absolute_error=statistics.mean(errors),
                validation_root_mean_square_error=math.sqrt(
                    statistics.mean(value * value for value in errors)
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
    outperformed_best_reference = (
        public_metric.validation_root_mean_square_error
        < best_reference.validation_root_mean_square_error
    )
    maximum_standard_error = max(
        item.rollout_standard_error for item in calibration + validation
    )
    calibration_coverage = sum(covered(item) for item in calibration) / len(calibration)
    validation_coverage = sum(covered(item) for item in validation) / len(validation)
    gate = all(
        (
            overlap == 0,
            calibration_coverage >= target_coverage,
            validation_coverage >= target_coverage,
            statistics.mean(widths) <= maximum_mean_interval_width,
            maximum_standard_error <= maximum_rollout_standard_error,
            outperformed_best_reference,
        )
    )
    return NativePublicLeafCalibration(
        training_dataset_fingerprint=training_dataset.fingerprint,
        fresh_holdout_dataset_fingerprint=fresh_dataset.fingerprint,
        model_fingerprint=artifact.fingerprint,
        calibration_example_ids=tuple(item.example_id for item in calibration),
        validation_example_ids=tuple(item.example_id for item in validation),
        calibration_particle_family_fingerprint=fresh_dataset.particle_family_fingerprint,
        training_holdout_overlap_count=overlap,
        target_coverage=target_coverage,
        maximum_mean_interval_width_threshold=maximum_mean_interval_width,
        maximum_rollout_standard_error_threshold=maximum_rollout_standard_error,
        conformal_rank=rank,
        conformal_radius=radius,
        calibration_coverage=calibration_coverage,
        validation_coverage=validation_coverage,
        validation_mean_absolute_error=public_metric.validation_mean_absolute_error,
        validation_root_mean_square_error=public_metric.validation_root_mean_square_error,
        mean_interval_width=statistics.mean(widths),
        informative_interval_fraction=sum(value < 2.0 - 1e-12 for value in widths)
        / len(widths),
        maximum_rollout_standard_error=maximum_standard_error,
        evaluation_metrics=tuple(evaluation_metrics),
        best_reference_model=best_reference.model_name,
        best_reference_root_mean_square_error=(
            best_reference.validation_root_mean_square_error
        ),
        outperformed_best_reference=outperformed_best_reference,
        empirical_fixed_profile_leaf_gate_passed=gate,
        fresh_root_validation=True,
    )


@dataclass(frozen=True, slots=True)
class Phase6NativePublicLeafEvidence:
    benchmark_id: str
    status: str
    training_dataset_fingerprint: str
    fresh_holdout_dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    selected_regularization: float
    fresh_terminal_rollout_count: int
    fresh_native_transition_count: int
    validation_examples: int
    training_holdout_overlap_count: int
    fresh_root_validation: bool
    target_coverage: float
    maximum_mean_interval_width_threshold: float
    maximum_rollout_standard_error_threshold: float
    conformal_radius: float
    calibration_coverage: float
    validation_coverage: float
    validation_mean_absolute_error: float
    validation_root_mean_square_error: float
    mean_interval_width: float
    informative_interval_fraction: float
    maximum_rollout_standard_error: float
    best_reference_model: str
    best_reference_root_mean_square_error: float
    outperformed_best_reference: bool
    empirical_fixed_profile_leaf_gate_passed: bool
    counterfactual_values_calibrated: bool
    public_belief_calibrated: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "training_dataset_fingerprint",
            "fresh_holdout_dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6V evidence identity is required")
        if min(
            self.fresh_terminal_rollout_count,
            self.fresh_native_transition_count,
            self.validation_examples,
        ) <= 0:
            raise ValueError("Phase 6V evidence accounting is invalid")
        if not self.best_reference_model:
            raise ValueError("Phase 6V evidence requires a reference model")
        if self.training_dataset_fingerprint == self.fresh_holdout_dataset_fingerprint:
            raise ValueError("Phase 6V evidence reused its training dataset")
        for field_name in (
            "selected_regularization",
            "conformal_radius",
            "validation_coverage",
            "validation_mean_absolute_error",
            "validation_root_mean_square_error",
            "calibration_coverage",
            "mean_interval_width",
            "informative_interval_fraction",
            "maximum_rollout_standard_error",
            "best_reference_root_mean_square_error",
            "target_coverage",
            "maximum_mean_interval_width_threshold",
            "maximum_rollout_standard_error_threshold",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.outperformed_best_reference != (
            self.validation_root_mean_square_error
            < self.best_reference_root_mean_square_error
        ):
            raise ValueError("Phase 6V evidence reference comparison is inconsistent")
        expected_gate = all(
            (
                self.training_holdout_overlap_count == 0,
                self.fresh_root_validation,
                self.calibration_coverage >= self.target_coverage,
                self.validation_coverage >= self.target_coverage,
                self.mean_interval_width <= self.maximum_mean_interval_width_threshold,
                self.maximum_rollout_standard_error
                <= self.maximum_rollout_standard_error_threshold,
                self.outperformed_best_reference,
            )
        )
        if self.empirical_fixed_profile_leaf_gate_passed != expected_gate:
            raise ValueError("Phase 6V evidence gate is inconsistent with its metrics")
        if any(
            (
                self.counterfactual_values_calibrated,
                self.public_belief_calibrated,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6V empirical model cannot enable safety")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_PUBLIC_LEAF_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6V public ridge leaf evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Selected ridge regularization | {self.selected_regularization:.1f} |
| Fresh terminal continuations | {self.fresh_terminal_rollout_count} |
| Fresh native transitions | {self.fresh_native_transition_count} |
| Conformal radius | {self.conformal_radius:.6f} |
| Calibration coverage | {self.calibration_coverage:.3%} |
| Fresh validation coverage | {self.validation_coverage:.3%} |
| Fresh validation MAE | {self.validation_mean_absolute_error:.6f} |
| Fresh validation RMSE | {self.validation_root_mean_square_error:.6f} |
| Best reference model | {self.best_reference_model} |
| Best reference RMSE | {self.best_reference_root_mean_square_error:.6f} |
| Beat best reference RMSE | **{str(self.outperformed_best_reference).lower()}** |
| Mean interval width | {self.mean_interval_width:.6f} |
| Informative interval fraction | {self.informative_interval_fraction:.3%} |
| Maximum rollout standard error | {self.maximum_rollout_standard_error:.6f} |
| Empirical fixed-profile leaf gate | **{str(self.empirical_fixed_profile_leaf_gate_passed).lower()}** |
| Counterfactual values calibrated | **false** |

The model consumes public features only and is validated on a fresh native root,
but its target is still one fixed policy profile's empirical return. It cannot
produce opponent counterfactual-value bounds or enable safe resolving.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6NativePublicLeafEvidence":
        kwargs = {
            name: raw[name]
            for name in cls.__dataclass_fields__
            if name != "claims"
        }
        kwargs["claims"] = tuple(
            (str(key), str(value)) for key, value in raw["claims"].items()
        )
        result = cls(**kwargs)
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6V evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativePublicLeafArtifacts:
    fresh_dataset: NativeLeafModelDataset
    model: PublicRidgeLeafArtifact
    calibration: NativePublicLeafCalibration
    evidence: Phase6NativePublicLeafEvidence


def run_phase6_native_public_leaf(
    *,
    project_root: str | Path | None = None,
    training_dataset_path: str | Path | None = None,
    fresh_holdout_dataset_path: str | Path | None = None,
    seed: int = PHASE6_NATIVE_PUBLIC_LEAF_SEED,
) -> Phase6NativePublicLeafArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    training_path = Path(
        training_dataset_path
        or root
        / "artifacts"
        / "phase6_native_leaf_model"
        / "native_low_noise_leaf_dataset.json"
    ).resolve()
    training_dataset = NativeLeafModelDataset.from_dict(
        json.loads(training_path.read_text(encoding="utf-8"))
    )
    provider, model = train_public_ridge_leaf(training_dataset)
    if fresh_holdout_dataset_path is None:
        fresh_artifacts = run_phase6_native_leaf_model(
            project_root=root,
            seed=seed,
            config=NativeLeafModelConfig(),
        )
        fresh_dataset = fresh_artifacts.dataset
    else:
        fresh_dataset = NativeLeafModelDataset.from_dict(
            json.loads(Path(fresh_holdout_dataset_path).read_text(encoding="utf-8"))
        )
    calibration = _evaluate_public_leaf(
        training_dataset,
        fresh_dataset,
        provider,
        model,
    )
    claims = (
        ("model", "public_only_ridge_fixed_profile_leaf"),
        ("hyperparameter_selection", "leave_candidate_out_training_only"),
        ("calibration", "fresh_root_variant_2"),
        ("validation", "fresh_root_variant_3"),
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
    evidence = Phase6NativePublicLeafEvidence(
        benchmark_id="phase6v-native-public-ridge-leaf-v1",
        status="phase6v_native_public_leaf_validated_partial",
        training_dataset_fingerprint=training_dataset.fingerprint,
        fresh_holdout_dataset_fingerprint=fresh_dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        selected_regularization=model.selected_regularization,
        fresh_terminal_rollout_count=len(fresh_dataset.examples)
        * fresh_dataset.rollouts_per_leaf,
        fresh_native_transition_count=fresh_dataset.native_transition_count,
        validation_examples=len(calibration.validation_example_ids),
        training_holdout_overlap_count=calibration.training_holdout_overlap_count,
        fresh_root_validation=calibration.fresh_root_validation,
        target_coverage=calibration.target_coverage,
        maximum_mean_interval_width_threshold=(
            calibration.maximum_mean_interval_width_threshold
        ),
        maximum_rollout_standard_error_threshold=(
            calibration.maximum_rollout_standard_error_threshold
        ),
        conformal_radius=calibration.conformal_radius,
        calibration_coverage=calibration.calibration_coverage,
        validation_coverage=calibration.validation_coverage,
        validation_mean_absolute_error=calibration.validation_mean_absolute_error,
        validation_root_mean_square_error=(
            calibration.validation_root_mean_square_error
        ),
        mean_interval_width=calibration.mean_interval_width,
        informative_interval_fraction=calibration.informative_interval_fraction,
        maximum_rollout_standard_error=calibration.maximum_rollout_standard_error,
        best_reference_model=calibration.best_reference_model,
        best_reference_root_mean_square_error=(
            calibration.best_reference_root_mean_square_error
        ),
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
    return Phase6NativePublicLeafArtifacts(fresh_dataset, model, calibration, evidence)


def main() -> None:
    print(run_phase6_native_public_leaf().evidence.to_json())


if __name__ == "__main__":
    main()
