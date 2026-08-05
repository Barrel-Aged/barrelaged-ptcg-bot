"""Split-safe public-feature ridge leaf model for fixed-profile native returns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import torch

from .native_blueprint_features import NATIVE_PUBLIC_FEATURE_NAMES
from .phase6_native_leaf_model import NativeLeafModelDataset
from .trajectory_neural import NamedFeatureNormalizer


NATIVE_PUBLIC_LEAF_SCHEMA_VERSION = 1
PUBLIC_LEAF_REGULARIZATION_GRID = (0.1, 1.0, 10.0, 100.0, 1_000.0)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RidgeCrossValidationMetric:
    regularization: float
    leave_candidate_out_rmse: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("ridge regularization must be positive and finite")
        if not math.isfinite(self.leave_candidate_out_rmse) or self.leave_candidate_out_rmse < 0:
            raise ValueError("ridge cross-validation RMSE is invalid")


@dataclass(frozen=True, slots=True)
class PublicRidgeLeafArtifact:
    source_dataset_fingerprint: str
    deck_fingerprint: str
    public_normalizer: NamedFeatureNormalizer
    regularization_grid: tuple[float, ...]
    cross_validation_metrics: tuple[RidgeCrossValidationMetric, ...]
    selected_regularization: float
    weights: tuple[float, ...]
    train_example_ids: tuple[str, ...]
    train_candidate_ids: tuple[str, ...]
    train_rows_fingerprint: str
    trained_on_variants: tuple[int, ...]
    fixed_profile_empirical_target: bool = True
    public_features_only: bool = True
    selected_truth_identifier_input: bool = False
    counterfactual_value_provider: bool = False
    safe_resolver_compatible: bool = False
    official_engine_executed: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "source_dataset_fingerprint",
            "deck_fingerprint",
            "train_rows_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.regularization_grid != PUBLIC_LEAF_REGULARIZATION_GRID:
            raise ValueError("public leaf regularization grid drifted")
        if tuple(item.regularization for item in self.cross_validation_metrics) != self.regularization_grid:
            raise ValueError("public leaf CV metrics do not cover the grid")
        best = min(
            self.cross_validation_metrics,
            key=lambda item: (item.leave_candidate_out_rmse, item.regularization),
        ).regularization
        if self.selected_regularization != best:
            raise ValueError("public leaf selected regularization is not the CV winner")
        if len(self.weights) != len(self.public_normalizer.names) + 1:
            raise ValueError("public leaf weight width is invalid")
        if self.public_normalizer.names != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("public leaf normalizer schema is not the native public schema")
        if self.public_normalizer.fit_rows_fingerprint != self.train_rows_fingerprint:
            raise ValueError("public leaf normalizer was not fit on the declared rows")
        if not all(math.isfinite(value) for value in self.weights):
            raise ValueError("public leaf weights must be finite")
        if not self.train_example_ids or len(set(self.train_example_ids)) != len(self.train_example_ids):
            raise ValueError("public leaf training IDs must be unique")
        if not self.train_candidate_ids or len(set(self.train_candidate_ids)) != len(self.train_candidate_ids):
            raise ValueError("public leaf training candidates must be unique")
        if (
            not self.trained_on_variants
            or tuple(sorted(set(self.trained_on_variants))) != self.trained_on_variants
            or min(self.trained_on_variants) < 0
        ):
            raise ValueError("public leaf training variants must be sorted and unique")
        if len(self.train_example_ids) != len(self.trained_on_variants) * len(
            self.train_candidate_ids
        ):
            raise ValueError("public leaf requires one row per candidate/variant pair")
        if not self.fixed_profile_empirical_target or not self.public_features_only:
            raise ValueError("public leaf training scope is invalid")
        if any(
            (
                self.selected_truth_identifier_input,
                self.counterfactual_value_provider,
                self.safe_resolver_compatible,
                self.official_engine_executed,
            )
        ):
            raise ValueError("public ridge leaf promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": NATIVE_PUBLIC_LEAF_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "public_normalizer": self.public_normalizer.to_dict(),
            "regularization_grid": list(self.regularization_grid),
            "cross_validation_metrics": [
                asdict(item) for item in self.cross_validation_metrics
            ],
            "selected_regularization": self.selected_regularization,
            "weights": list(self.weights),
            "train_example_ids": list(self.train_example_ids),
            "train_candidate_ids": list(self.train_candidate_ids),
            "train_rows_fingerprint": self.train_rows_fingerprint,
            "trained_on_variants": list(self.trained_on_variants),
            "claims": {
                "fixed_profile_empirical_target": self.fixed_profile_empirical_target,
                "public_features_only": self.public_features_only,
                "selected_truth_identifier_input": self.selected_truth_identifier_input,
                "counterfactual_value_provider": self.counterfactual_value_provider,
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_engine_executed": self.official_engine_executed,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRidgeLeafArtifact":
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            public_normalizer=NamedFeatureNormalizer.from_dict(raw["public_normalizer"]),
            regularization_grid=tuple(float(value) for value in raw["regularization_grid"]),
            cross_validation_metrics=tuple(
                RidgeCrossValidationMetric(**dict(item))
                for item in raw["cross_validation_metrics"]
            ),
            selected_regularization=float(raw["selected_regularization"]),
            weights=tuple(float(value) for value in raw["weights"]),
            train_example_ids=tuple(str(value) for value in raw["train_example_ids"]),
            train_candidate_ids=tuple(str(value) for value in raw["train_candidate_ids"]),
            train_rows_fingerprint=str(raw["train_rows_fingerprint"]),
            trained_on_variants=tuple(int(value) for value in raw["trained_on_variants"]),
            fixed_profile_empirical_target=bool(claims["fixed_profile_empirical_target"]),
            public_features_only=bool(claims["public_features_only"]),
            selected_truth_identifier_input=bool(claims["selected_truth_identifier_input"]),
            counterfactual_value_provider=bool(claims["counterfactual_value_provider"]),
            safe_resolver_compatible=bool(claims["safe_resolver_compatible"]),
            official_engine_executed=bool(claims["official_engine_executed"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("public ridge leaf artifact fingerprint mismatch")
        return result


class PublicRidgeLeafProvider:
    """Restorable public-feature scalar for a fixed policy profile only."""

    def __init__(self, artifact: PublicRidgeLeafArtifact) -> None:
        self.artifact = artifact

    @property
    def claim_boundary(self) -> Mapping[str, bool]:
        return {
            "fixed_profile_empirical_value": True,
            "public_features_only": True,
            "counterfactual_values": False,
            "safe_resolving": False,
            "official_engine_execution": False,
        }

    def predict(self, public_features: Sequence[tuple[str, float]]) -> float:
        values = self.artifact.public_normalizer.transform(public_features)
        result = self.artifact.weights[0] + sum(
            weight * value
            for weight, value in zip(self.artifact.weights[1:], values, strict=True)
        )
        return max(-1.0, min(1.0, float(result)))


def _fit_weights(
    rows: Sequence[Sequence[tuple[str, float]]],
    targets: Sequence[float],
    *,
    normalizer: NamedFeatureNormalizer,
    regularization: float,
) -> tuple[float, ...]:
    matrix = torch.tensor(
        [normalizer.transform(row) for row in rows],
        dtype=torch.float64,
    )
    matrix = torch.cat(
        (torch.ones((len(matrix), 1), dtype=torch.float64), matrix),
        dim=1,
    )
    target = torch.tensor(targets, dtype=torch.float64)
    penalty = torch.eye(matrix.shape[1], dtype=torch.float64)
    penalty[0, 0] = 0.0
    weights = torch.linalg.solve(
        matrix.T @ matrix + regularization * penalty,
        matrix.T @ target,
    )
    return tuple(float(value) for value in weights.tolist())


def train_public_ridge_leaf(
    dataset: NativeLeafModelDataset,
    *,
    trained_variants: Sequence[int] = (0, 1),
) -> tuple[PublicRidgeLeafProvider, PublicRidgeLeafArtifact]:
    variants = tuple(int(value) for value in trained_variants)
    if not variants or tuple(sorted(set(variants))) != variants or min(variants) < 0:
        raise ValueError("public ridge leaf variants must be sorted and unique")
    train = tuple(item for item in dataset.examples if item.variant in variants)
    if len(train) != len(variants) * len(dataset.candidate_ids):
        raise ValueError("public ridge leaf requires one row per candidate/variant pair")
    expected_support = {
        (candidate_id, variant)
        for candidate_id in dataset.candidate_ids
        for variant in variants
    }
    observed_support = {
        (item.candidate_id_supervision_only, item.variant) for item in train
    }
    if observed_support != expected_support:
        raise ValueError("public ridge leaf training support is incomplete or duplicated")
    train_rows_fingerprint = _fingerprint(
        [(item.example_id, item.public_features, item.empirical_return) for item in train]
    )
    metrics = []
    for regularization in PUBLIC_LEAF_REGULARIZATION_GRID:
        squared_errors = []
        for held_candidate in dataset.candidate_ids:
            fold_train = tuple(
                item for item in train if item.candidate_id_supervision_only != held_candidate
            )
            fold_test = tuple(
                item for item in train if item.candidate_id_supervision_only == held_candidate
            )
            fold_fingerprint = _fingerprint(
                [(item.example_id, item.public_features) for item in fold_train]
            )
            normalizer = NamedFeatureNormalizer.fit(
                [item.public_features for item in fold_train],
                fit_rows_fingerprint=fold_fingerprint,
            )
            weights = _fit_weights(
                [item.public_features for item in fold_train],
                [item.empirical_return for item in fold_train],
                normalizer=normalizer,
                regularization=regularization,
            )
            for item in fold_test:
                values = normalizer.transform(item.public_features)
                prediction = max(
                    -1.0,
                    min(
                        1.0,
                        weights[0]
                        + sum(
                            weight * value
                            for weight, value in zip(weights[1:], values, strict=True)
                        ),
                    ),
                )
                squared_errors.append((prediction - item.empirical_return) ** 2)
        metrics.append(
            RidgeCrossValidationMetric(
                regularization=regularization,
                leave_candidate_out_rmse=math.sqrt(
                    sum(squared_errors) / len(squared_errors)
                ),
            )
        )
    selected = min(
        metrics,
        key=lambda item: (item.leave_candidate_out_rmse, item.regularization),
    ).regularization
    normalizer = NamedFeatureNormalizer.fit(
        [item.public_features for item in train],
        fit_rows_fingerprint=train_rows_fingerprint,
    )
    weights = _fit_weights(
        [item.public_features for item in train],
        [item.empirical_return for item in train],
        normalizer=normalizer,
        regularization=selected,
    )
    artifact = PublicRidgeLeafArtifact(
        source_dataset_fingerprint=dataset.fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        public_normalizer=normalizer,
        regularization_grid=PUBLIC_LEAF_REGULARIZATION_GRID,
        cross_validation_metrics=tuple(metrics),
        selected_regularization=selected,
        weights=weights,
        train_example_ids=tuple(item.example_id for item in train),
        train_candidate_ids=dataset.candidate_ids,
        train_rows_fingerprint=train_rows_fingerprint,
        trained_on_variants=variants,
    )
    return PublicRidgeLeafProvider(artifact), artifact


def restore_public_ridge_leaf(
    artifact: PublicRidgeLeafArtifact,
) -> PublicRidgeLeafProvider:
    return PublicRidgeLeafProvider(artifact)
