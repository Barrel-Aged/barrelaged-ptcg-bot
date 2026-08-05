"""Phase 6AK collection-heldout semiparametric conditional-value audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

import torch

from .beliefs import OPPONENT_INTENTS
from .phase6_branch_conditioned_values import (
    PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES,
    PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR,
    PHASE6_BRANCH_VALUE_COVERAGE,
    BranchConditionedDataset,
    BranchConditionedRoot,
    BranchValuePointMetric,
    BranchValueRootExample,
    BranchValueSliceMetric,
    _branch_heuristic_predictions,
    _build_branch_conditioned_root,
    _contrast_mse,
    _contrast_root_errors,
    _conformal_radius,
    _fingerprint,
    _interval_result,
    _mse,
    _point_metric,
    _root_heuristic_predictions,
    _targets,
    _validate_digest,
    build_branch_value_examples,
    build_branch_value_examples_from_roots,
)
from .phase6_expected_value_surfaces import (
    PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH,
    PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR,
    PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE,
    _capture_complete_match_late_root,
    _load_expected_value_sources,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_opponent_deviation_values import OPPONENT_DEVIATION_ACTION_IDS
from .phase6_semiparametric_branch_hybrid import (
    BranchHybridModel,
    BranchHybridValidationDataset,
    _hybrid_predictions,
)
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_EXPANDED_ROOT_MEAN_SCHEMA_VERSION = 1
PHASE6_EXPANDED_ROOT_MEAN_CALIBRATION_SEED = 6_053
PHASE6_EXPANDED_ROOT_MEAN_VALIDATION_SEED = 6_054
PHASE6_EXPANDED_ROOT_MEAN_BOOTSTRAP_SEED = 6_055
PHASE6_EXPANDED_ROOT_MEAN_FORMS = ("direct", "root_heuristic_residual")
PHASE6_EXPANDED_ROOT_MEAN_PENALTY_PER_ROOT_GRID = (
    0.25,
    0.50,
    1.0,
    2.0,
    3.0,
    5.0,
)
PHASE6_EXPANDED_ROOT_MEAN_FOLDS = (
    "action_first",
    "phase6aj_action_first",
    "canonical",
    "engine_first",
    "hand_first",
)
PHASE6_EXPANDED_ROOT_MEAN_ROLES = ("calibration", "validation")


def _load_phase6ak_design(
    project_root: str | Path | None,
) -> tuple[BranchConditionedDataset, BranchHybridValidationDataset, BranchHybridModel]:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    ai_directory = root / "artifacts" / "phase6_branch_conditioned_values"
    aj_directory = root / "artifacts" / "phase6_semiparametric_branch_hybrid"
    ai = BranchConditionedDataset.from_dict(
        json.loads(
            (ai_directory / "branch_conditioned_dataset.json").read_text(
                encoding="utf-8"
            )
        )
    )
    aj = BranchHybridValidationDataset.from_dict(
        json.loads(
            (aj_directory / "branch_hybrid_validation_dataset.json").read_text(
                encoding="utf-8"
            )
        )
    )
    source_model = BranchHybridModel.from_dict(
        json.loads(
            (aj_directory / "branch_hybrid_model.json").read_text(encoding="utf-8")
        )
    )
    if (
        aj.source_branch_dataset_fingerprint != ai.fingerprint
        or aj.source_branch_model_fingerprint != source_model.source_model.fingerprint
        or source_model.source_dataset_fingerprint != ai.fingerprint
    ):
        raise ValueError("Phase 6AK source artifacts are not mutually bound")
    return ai, aj, source_model


def _development_fold(example: BranchValueRootExample) -> str:
    if example.match_id.startswith("phase6aj-"):
        return "phase6aj_action_first"
    result = example.policy_variant_id_supervision_only
    if result not in PHASE6_EXPANDED_ROOT_MEAN_FOLDS:
        raise ValueError("Phase 6AK development fold is unsupported")
    return result


def _root_mean_targets(
    examples: Sequence[BranchValueRootExample],
) -> torch.Tensor:
    return torch.tensor(
        [
            statistics.mean(
                action.empirical_player0_value for action in example.actions
            )
            for example in examples
        ],
        dtype=torch.float64,
    )


def _root_heuristic_tensor(
    examples: Sequence[BranchValueRootExample],
) -> torch.Tensor:
    return torch.tensor(
        [example.root_heuristic_value for example in examples],
        dtype=torch.float64,
    )


def _fit_ridge(
    examples: Sequence[BranchValueRootExample],
    *,
    model_form: str,
    penalty_per_root: float,
) -> tuple[NamedFeatureNormalizer, float, tuple[float, ...]]:
    if model_form not in PHASE6_EXPANDED_ROOT_MEAN_FORMS:
        raise ValueError("Phase 6AK ridge form is unsupported")
    if penalty_per_root not in PHASE6_EXPANDED_ROOT_MEAN_PENALTY_PER_ROOT_GRID:
        raise ValueError("Phase 6AK ridge penalty is unsupported")
    materialized = tuple(examples)
    if not materialized:
        raise ValueError("Phase 6AK ridge fit is empty")
    normalizer = NamedFeatureNormalizer.fit(
        [item.root_features for item in materialized],
        fit_rows_fingerprint=_fingerprint(
            [item.root_id for item in materialized]
        ),
    )
    features = torch.tensor(
        [normalizer.transform(item.root_features) for item in materialized],
        dtype=torch.float64,
    )
    design = torch.cat(
        [torch.ones((len(materialized), 1), dtype=torch.float64), features],
        dim=1,
    )
    targets = _root_mean_targets(materialized)
    if model_form == "root_heuristic_residual":
        targets = targets - _root_heuristic_tensor(materialized)
    penalty = penalty_per_root * len(materialized)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * penalty
    regularizer[0, 0] = 0.0
    coefficients = torch.linalg.solve(
        design.T @ design + regularizer,
        design.T @ targets,
    )
    return (
        normalizer,
        float(coefficients[0].item()),
        tuple(float(value) for value in coefficients[1:].tolist()),
    )


def _ridge_predictions(
    examples: Sequence[BranchValueRootExample],
    *,
    model_form: str,
    normalizer: NamedFeatureNormalizer,
    intercept: float,
    coefficients: Sequence[float],
) -> list[list[float]]:
    materialized = tuple(examples)
    weights = tuple(float(value) for value in coefficients)
    root_means = []
    for example in materialized:
        transformed = normalizer.transform(example.root_features)
        value = intercept + sum(
            weight * feature
            for weight, feature in zip(weights, transformed, strict=True)
        )
        if model_form == "root_heuristic_residual":
            value += example.root_heuristic_value
        root_means.append(max(-1.0, min(1.0, value)))
    branch = _branch_heuristic_predictions(materialized)
    return [
        [
            max(-1.0, min(1.0, root_mean + value - statistics.mean(group)))
            for value in group
        ]
        for root_mean, group in zip(root_means, branch, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class ExpandedRootMeanFoldMetric:
    fold_id: str
    fit_root_count: int
    heldout_root_count: int
    heldout_mse: float
    heldout_zero_mse: float
    heldout_root_heuristic_mse: float
    heldout_branch_heuristic_mse: float
    normalized_mse_ratio: float

    def __post_init__(self) -> None:
        if self.fold_id not in PHASE6_EXPANDED_ROOT_MEAN_FOLDS:
            raise ValueError("Phase 6AK fold metric identity is invalid")
        if min(self.fit_root_count, self.heldout_root_count) <= 0:
            raise ValueError("Phase 6AK fold metric count is invalid")
        values = (
            self.heldout_mse,
            self.heldout_zero_mse,
            self.heldout_root_heuristic_mse,
            self.heldout_branch_heuristic_mse,
            self.normalized_mse_ratio,
        )
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("Phase 6AK fold metric is invalid")
        expected = self.heldout_mse / min(
            self.heldout_zero_mse,
            self.heldout_root_heuristic_mse,
            self.heldout_branch_heuristic_mse,
        )
        if abs(self.normalized_mse_ratio - expected) > 1e-12:
            raise ValueError("Phase 6AK normalized fold metric is inconsistent")


@dataclass(frozen=True, slots=True)
class ExpandedRootMeanSelectionMetric:
    model_form: str
    penalty_per_root: float
    fold_metrics: tuple[ExpandedRootMeanFoldMetric, ...]
    worst_normalized_mse_ratio: float
    mean_normalized_mse_ratio: float
    mean_heldout_mse: float

    def __post_init__(self) -> None:
        if self.model_form not in PHASE6_EXPANDED_ROOT_MEAN_FORMS:
            raise ValueError("Phase 6AK selection form is unsupported")
        if self.penalty_per_root not in PHASE6_EXPANDED_ROOT_MEAN_PENALTY_PER_ROOT_GRID:
            raise ValueError("Phase 6AK selection penalty is unsupported")
        if tuple(item.fold_id for item in self.fold_metrics) != (
            PHASE6_EXPANDED_ROOT_MEAN_FOLDS
        ):
            raise ValueError("Phase 6AK selection folds are incomplete")
        ratios = [item.normalized_mse_ratio for item in self.fold_metrics]
        mses = [item.heldout_mse for item in self.fold_metrics]
        if (
            abs(self.worst_normalized_mse_ratio - max(ratios)) > 1e-12
            or abs(self.mean_normalized_mse_ratio - statistics.mean(ratios))
            > 1e-12
            or abs(self.mean_heldout_mse - statistics.mean(mses)) > 1e-12
        ):
            raise ValueError("Phase 6AK selection aggregate is inconsistent")


def _selection_key(
    item: ExpandedRootMeanSelectionMetric,
) -> tuple[float, float, float, int, float]:
    return (
        item.worst_normalized_mse_ratio,
        item.mean_normalized_mse_ratio,
        item.mean_heldout_mse,
        0 if item.model_form == "root_heuristic_residual" else 1,
        item.penalty_per_root,
    )


@dataclass(frozen=True, slots=True)
class ExpandedRootMeanModel:
    source_phase6ai_dataset_fingerprint: str
    source_phase6aj_dataset_fingerprint: str
    source_phase6aj_model_fingerprint: str
    design_root_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    selected_model_form: str
    selected_penalty_per_root: float
    selection_metrics: tuple[ExpandedRootMeanSelectionMetric, ...]
    root_normalizer: NamedFeatureNormalizer
    intercept: float
    coefficients: tuple[float, ...]
    branch_advantage_scale: float = 1.0
    all_prior_targets_are_development_data: bool = True
    intent_labels_as_features: bool = False
    policy_labels_as_features: bool = False
    learned_action_advantage_used: bool = False
    official_engine_executed: bool = False
    counterfactual_value_provider: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_phase6ai_dataset_fingerprint",
            "source_phase6aj_dataset_fingerprint",
            "source_phase6aj_model_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if len(self.design_root_ids) != 216 or len(set(self.design_root_ids)) != 216:
            raise ValueError("Phase 6AK design-root ledger is incomplete")
        if len(self.candidate_ids) != 6 or len(set(self.candidate_ids)) != 6:
            raise ValueError("Phase 6AK candidate support is incomplete")
        expected_grid = {
            (form, penalty)
            for form in PHASE6_EXPANDED_ROOT_MEAN_FORMS
            for penalty in PHASE6_EXPANDED_ROOT_MEAN_PENALTY_PER_ROOT_GRID
        }
        actual_grid = {
            (item.model_form, item.penalty_per_root)
            for item in self.selection_metrics
        }
        if actual_grid != expected_grid or len(self.selection_metrics) != len(expected_grid):
            raise ValueError("Phase 6AK selection grid is incomplete")
        selected = min(self.selection_metrics, key=_selection_key)
        if (
            self.selected_model_form,
            self.selected_penalty_per_root,
        ) != (selected.model_form, selected.penalty_per_root):
            raise ValueError("Phase 6AK selected model is inconsistent")
        if len(self.root_normalizer.names) != 208:
            raise ValueError("Phase 6AK root feature width drifted")
        if len(self.coefficients) != len(self.root_normalizer.names):
            raise ValueError("Phase 6AK coefficient width drifted")
        if any(
            not math.isfinite(value)
            for value in (self.intercept, *self.coefficients)
        ):
            raise ValueError("Phase 6AK coefficients are nonfinite")
        if self.branch_advantage_scale != 1.0:
            raise ValueError("Phase 6AK branch advantage scale drifted")
        if any(
            (
                not self.all_prior_targets_are_development_data,
                self.intent_labels_as_features,
                self.policy_labels_as_features,
                self.learned_action_advantage_used,
                self.official_engine_executed,
                self.counterfactual_value_provider,
            )
        ):
            raise ValueError("Phase 6AK model promoted its abstraction boundary")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPANDED_ROOT_MEAN_SCHEMA_VERSION,
            "source_phase6ai_dataset_fingerprint": self.source_phase6ai_dataset_fingerprint,
            "source_phase6aj_dataset_fingerprint": self.source_phase6aj_dataset_fingerprint,
            "source_phase6aj_model_fingerprint": self.source_phase6aj_model_fingerprint,
            "design_root_ids": list(self.design_root_ids),
            "candidate_ids": list(self.candidate_ids),
            "selected_model_form": self.selected_model_form,
            "selected_penalty_per_root": self.selected_penalty_per_root,
            "selection_metrics": [
                {
                    "model_form": item.model_form,
                    "penalty_per_root": item.penalty_per_root,
                    "fold_metrics": [asdict(fold) for fold in item.fold_metrics],
                    "worst_normalized_mse_ratio": item.worst_normalized_mse_ratio,
                    "mean_normalized_mse_ratio": item.mean_normalized_mse_ratio,
                    "mean_heldout_mse": item.mean_heldout_mse,
                }
                for item in self.selection_metrics
            ],
            "root_normalizer": self.root_normalizer.to_dict(),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "branch_advantage_scale": self.branch_advantage_scale,
            "claims": {
                "all_prior_targets_are_development_data": self.all_prior_targets_are_development_data,
                "intent_labels_as_features": self.intent_labels_as_features,
                "policy_labels_as_features": self.policy_labels_as_features,
                "learned_action_advantage_used": self.learned_action_advantage_used,
                "official_engine_executed": self.official_engine_executed,
                "counterfactual_value_provider": self.counterfactual_value_provider,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpandedRootMeanModel":
        claims = raw["claims"]
        metrics = []
        for item in raw["selection_metrics"]:
            metrics.append(
                ExpandedRootMeanSelectionMetric(
                    model_form=str(item["model_form"]),
                    penalty_per_root=float(item["penalty_per_root"]),
                    fold_metrics=tuple(
                        ExpandedRootMeanFoldMetric(**dict(fold))
                        for fold in item["fold_metrics"]
                    ),
                    worst_normalized_mse_ratio=float(
                        item["worst_normalized_mse_ratio"]
                    ),
                    mean_normalized_mse_ratio=float(
                        item["mean_normalized_mse_ratio"]
                    ),
                    mean_heldout_mse=float(item["mean_heldout_mse"]),
                )
            )
        result = cls(
            source_phase6ai_dataset_fingerprint=str(
                raw["source_phase6ai_dataset_fingerprint"]
            ),
            source_phase6aj_dataset_fingerprint=str(
                raw["source_phase6aj_dataset_fingerprint"]
            ),
            source_phase6aj_model_fingerprint=str(
                raw["source_phase6aj_model_fingerprint"]
            ),
            design_root_ids=tuple(str(value) for value in raw["design_root_ids"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            selected_model_form=str(raw["selected_model_form"]),
            selected_penalty_per_root=float(raw["selected_penalty_per_root"]),
            selection_metrics=tuple(metrics),
            root_normalizer=NamedFeatureNormalizer.from_dict(raw["root_normalizer"]),
            intercept=float(raw["intercept"]),
            coefficients=tuple(float(value) for value in raw["coefficients"]),
            branch_advantage_scale=float(raw["branch_advantage_scale"]),
            all_prior_targets_are_development_data=bool(
                claims["all_prior_targets_are_development_data"]
            ),
            intent_labels_as_features=bool(claims["intent_labels_as_features"]),
            policy_labels_as_features=bool(claims["policy_labels_as_features"]),
            learned_action_advantage_used=bool(claims["learned_action_advantage_used"]),
            official_engine_executed=bool(claims["official_engine_executed"]),
            counterfactual_value_provider=bool(claims["counterfactual_value_provider"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AK model fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ExpandedRootMeanProvider:
    model: ExpandedRootMeanModel

    def predict_examples(
        self,
        examples: Sequence[BranchValueRootExample],
    ) -> list[list[float]]:
        return _ridge_predictions(
            examples,
            model_form=self.model.selected_model_form,
            normalizer=self.model.root_normalizer,
            intercept=self.model.intercept,
            coefficients=self.model.coefficients,
        )


def restore_expanded_root_mean_model(
    model: ExpandedRootMeanModel,
) -> ExpandedRootMeanProvider:
    return ExpandedRootMeanProvider(model=model)


def train_expanded_root_mean_model(
    *,
    project_root: str | Path | None = None,
) -> ExpandedRootMeanModel:
    ai, aj, source_model = _load_phase6ak_design(project_root)
    ai_examples = build_branch_value_examples(ai)
    aj_examples = build_branch_value_examples_from_roots(
        aj.roots,
        aj.candidate_ids,
        expected_count=36,
    )
    examples = ai_examples + aj_examples
    if len(examples) != 216 or len({item.root_id for item in examples}) != 216:
        raise RuntimeError("Phase 6AK development example ledger is incomplete")
    metrics = []
    for model_form in PHASE6_EXPANDED_ROOT_MEAN_FORMS:
        for penalty_per_root in PHASE6_EXPANDED_ROOT_MEAN_PENALTY_PER_ROOT_GRID:
            fold_metrics = []
            for fold_id in PHASE6_EXPANDED_ROOT_MEAN_FOLDS:
                fit = tuple(
                    item for item in examples if _development_fold(item) != fold_id
                )
                heldout = tuple(
                    item for item in examples if _development_fold(item) == fold_id
                )
                normalizer, intercept, coefficients = _fit_ridge(
                    fit,
                    model_form=model_form,
                    penalty_per_root=penalty_per_root,
                )
                predictions = _ridge_predictions(
                    heldout,
                    model_form=model_form,
                    normalizer=normalizer,
                    intercept=intercept,
                    coefficients=coefficients,
                )
                targets = _targets(heldout)
                zero = [[0.0] * len(OPPONENT_DEVIATION_ACTION_IDS) for _ in heldout]
                root = _root_heuristic_predictions(heldout)
                branch = _branch_heuristic_predictions(heldout)
                mse = _mse(targets, predictions)
                fold_metrics.append(
                    ExpandedRootMeanFoldMetric(
                        fold_id=fold_id,
                        fit_root_count=len(fit),
                        heldout_root_count=len(heldout),
                        heldout_mse=mse,
                        heldout_zero_mse=_mse(targets, zero),
                        heldout_root_heuristic_mse=_mse(targets, root),
                        heldout_branch_heuristic_mse=_mse(targets, branch),
                        normalized_mse_ratio=mse
                        / min(
                            _mse(targets, zero),
                            _mse(targets, root),
                            _mse(targets, branch),
                        ),
                    )
                )
            ratios = [item.normalized_mse_ratio for item in fold_metrics]
            mses = [item.heldout_mse for item in fold_metrics]
            metrics.append(
                ExpandedRootMeanSelectionMetric(
                    model_form=model_form,
                    penalty_per_root=penalty_per_root,
                    fold_metrics=tuple(fold_metrics),
                    worst_normalized_mse_ratio=max(ratios),
                    mean_normalized_mse_ratio=statistics.mean(ratios),
                    mean_heldout_mse=statistics.mean(mses),
                )
            )
    selected = min(metrics, key=_selection_key)
    normalizer, intercept, coefficients = _fit_ridge(
        examples,
        model_form=selected.model_form,
        penalty_per_root=selected.penalty_per_root,
    )
    model = ExpandedRootMeanModel(
        source_phase6ai_dataset_fingerprint=ai.fingerprint,
        source_phase6aj_dataset_fingerprint=aj.fingerprint,
        source_phase6aj_model_fingerprint=source_model.fingerprint,
        design_root_ids=tuple(sorted(item.root_id for item in examples)),
        candidate_ids=ai.value_dataset.candidate_ids,
        selected_model_form=selected.model_form,
        selected_penalty_per_root=selected.penalty_per_root,
        selection_metrics=tuple(metrics),
        root_normalizer=normalizer,
        intercept=intercept,
        coefficients=coefficients,
    )
    if ExpandedRootMeanModel.from_dict(json.loads(model.to_json())) != model:
        raise RuntimeError("Phase 6AK model roundtrip changed values")
    return model


def _role_seed(role: str) -> int:
    if role == "calibration":
        return PHASE6_EXPANDED_ROOT_MEAN_CALIBRATION_SEED
    if role == "validation":
        return PHASE6_EXPANDED_ROOT_MEAN_VALIDATION_SEED
    raise ValueError("Phase 6AK panel role is unsupported")


def _role_namespace(role: str) -> str:
    if role not in PHASE6_EXPANDED_ROOT_MEAN_ROLES:
        raise ValueError("Phase 6AK panel role is unsupported")
    return f"phase6ak-{role}"


@dataclass(frozen=True, slots=True)
class ExpandedRootMeanPanel:
    role: str
    source_phase6ai_dataset_fingerprint: str
    source_phase6aj_dataset_fingerprint: str
    source_model_fingerprint: str
    design_root_ids: tuple[str, ...]
    seed: int
    match_namespace: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    intent_ids: tuple[str, ...]
    roots: tuple[BranchConditionedRoot, ...]
    prior_roots_reused: int = 0
    public_branch_features_only: bool = True
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if self.role not in PHASE6_EXPANDED_ROOT_MEAN_ROLES:
            raise ValueError("Phase 6AK panel role is unsupported")
        for name in (
            "source_phase6ai_dataset_fingerprint",
            "source_phase6aj_dataset_fingerprint",
            "source_model_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.seed != _role_seed(self.role):
            raise ValueError("Phase 6AK panel seed drifted")
        if self.match_namespace != _role_namespace(self.role):
            raise ValueError("Phase 6AK panel namespace drifted")
        if len(self.design_root_ids) != 216 or len(set(self.design_root_ids)) != 216:
            raise ValueError("Phase 6AK panel design ledger is incomplete")
        if len(self.roots) != 36 or len({item.root_id for item in self.roots}) != 36:
            raise ValueError("Phase 6AK panel root ledger is incomplete")
        root_ids = {item.root_id for item in self.roots}
        if root_ids & set(self.design_root_ids):
            raise ValueError("Phase 6AK panel reused a development root")
        expected = {
            (candidate, intent)
            for candidate in self.candidate_ids
            for intent in self.intent_ids
        }
        actual = {
            (
                item.value_root.candidate_id_hypothesis_input,
                item.value_root.intent_id_supervision_only,
            )
            for item in self.roots
        }
        if actual != expected or len(actual) != len(self.roots):
            raise ValueError("Phase 6AK panel candidate/intent cross is incomplete")
        prefix = f"{self.match_namespace}-action_first-"
        for root in self.roots:
            value = root.value_root
            if (
                not value.match_id.startswith(prefix)
                or value.split != "validation"
                or value.policy_variant_id_supervision_only != "action_first"
                or value.replica != 0
            ):
                raise ValueError("Phase 6AK root escaped its frozen protocol")
        if any(
            (
                self.prior_roots_reused != 0,
                not self.public_branch_features_only,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6AK panel promoted its abstraction boundary")
        if _raw_private_key_count(self.to_dict(include_fingerprint=False)) != 0:
            raise ValueError("Phase 6AK panel persisted a forbidden raw private key")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def terminal_rollout_count(self) -> int:
        return sum(item.value_root.terminal_rollout_count for item in self.roots)

    @property
    def live_selection_count(self) -> int:
        return sum(item.value_root.match_selection_count for item in self.roots)

    @property
    def native_transition_count(self) -> int:
        return sum(
            surface.native_transition_count
            for item in self.roots
            for surface in item.value_root.surfaces
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPANDED_ROOT_MEAN_SCHEMA_VERSION,
            "role": self.role,
            "source_phase6ai_dataset_fingerprint": self.source_phase6ai_dataset_fingerprint,
            "source_phase6aj_dataset_fingerprint": self.source_phase6aj_dataset_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "design_root_ids": list(self.design_root_ids),
            "seed": self.seed,
            "match_namespace": self.match_namespace,
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": dict(self.candidate_deck_fingerprints),
            "intent_ids": list(self.intent_ids),
            "roots": [item.to_dict() for item in self.roots],
            "claims": {
                "prior_roots_reused": self.prior_roots_reused,
                "public_branch_features_only": self.public_branch_features_only,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpandedRootMeanPanel":
        claims = raw["claims"]
        result = cls(
            role=str(raw["role"]),
            source_phase6ai_dataset_fingerprint=str(
                raw["source_phase6ai_dataset_fingerprint"]
            ),
            source_phase6aj_dataset_fingerprint=str(
                raw["source_phase6aj_dataset_fingerprint"]
            ),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            design_root_ids=tuple(str(value) for value in raw["design_root_ids"]),
            seed=int(raw["seed"]),
            match_namespace=str(raw["match_namespace"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            candidate_deck_fingerprints=tuple(
                (candidate, str(raw["candidate_deck_fingerprints"][candidate]))
                for candidate in raw["candidate_ids"]
            ),
            intent_ids=tuple(str(value) for value in raw["intent_ids"]),
            roots=tuple(BranchConditionedRoot.from_dict(item) for item in raw["roots"]),
            prior_roots_reused=int(claims["prior_roots_reused"]),
            public_branch_features_only=bool(claims["public_branch_features_only"]),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AK panel fingerprint mismatch")
        return result


def collect_expanded_root_mean_panel(
    role: str,
    *,
    project_root: str | Path | None = None,
    model: ExpandedRootMeanModel | None = None,
    existing_roots: Sequence[BranchConditionedRoot] = (),
    root_observer: Callable[[BranchConditionedRoot, int, int], None] | None = None,
) -> ExpandedRootMeanPanel:
    if role not in PHASE6_EXPANDED_ROOT_MEAN_ROLES:
        raise ValueError("Phase 6AK panel role is unsupported")
    frozen_model = model or train_expanded_root_mean_model(project_root=project_root)
    ai, aj, source_model = _load_phase6ak_design(project_root)
    if (
        frozen_model.source_phase6ai_dataset_fingerprint != ai.fingerprint
        or frozen_model.source_phase6aj_dataset_fingerprint != aj.fingerprint
        or frozen_model.source_phase6aj_model_fingerprint != source_model.fingerprint
    ):
        raise ValueError("Phase 6AK panel/model provenance mismatch")
    sources = _load_expected_value_sources(project_root)
    runtime_text = str(sources.runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

    expected_keys = tuple(
        (candidate.archetype, intent.value)
        for candidate in sources.candidates
        for intent in OPPONENT_INTENTS
    )
    materialized = tuple(existing_roots)
    by_key = {
        (
            item.value_root.candidate_id_hypothesis_input,
            item.value_root.intent_id_supervision_only,
        ): item
        for item in materialized
    }
    if len(by_key) != len(materialized) or not set(by_key).issubset(expected_keys):
        raise ValueError("Phase 6AK resume roots are duplicate/outside the grid")
    prefix = f"{_role_namespace(role)}-action_first-"
    if any(not item.match_id.startswith(prefix) for item in materialized):
        raise ValueError("Phase 6AK resume root belongs to the wrong role")
    candidates = {item.archetype: item for item in sources.candidates}
    intents = {item.value: item for item in OPPONENT_INTENTS}
    for key in expected_keys:
        if key in by_key:
            continue
        candidate_id, intent_id = key
        match_id = f"{prefix}{candidate_id}-{intent_id}-r0"
        last_error: Exception | None = None
        for attempt in range(1, PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS + 1):
            try:
                capture = _capture_complete_match_late_root(
                    battle_start=battle_start,
                    battle_select=battle_select,
                    battle_finish=battle_finish,
                    target=sources.target,
                    candidate=candidates[candidate_id],
                    intent=intents[intent_id],
                    policy_variant="action_first",
                    replica=0,
                    capture_attempt=attempt,
                    maximum_live_selections=PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
                    match_id=match_id,
                )
                root = _build_branch_conditioned_root(
                    capture,
                    sources=sources,
                    candidate=candidates[candidate_id],
                    intent=intents[intent_id],
                    policy_variant="action_first",
                    split="validation",
                    replica=0,
                    match_id=match_id,
                    seed=_role_seed(role),
                    maximum_rollout_transitions=PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
                    maximum_tactical_transitions=PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
                    experiment_label="Phase 6AK",
                )
                by_key[key] = root
                if root_observer is not None:
                    root_observer(root, len(by_key), len(expected_keys))
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Phase 6AK exhausted attempts for {role}/{candidate_id}/{intent_id}"
            ) from last_error
    roots = tuple(by_key[key] for key in expected_keys)
    panel = ExpandedRootMeanPanel(
        role=role,
        source_phase6ai_dataset_fingerprint=ai.fingerprint,
        source_phase6aj_dataset_fingerprint=aj.fingerprint,
        source_model_fingerprint=frozen_model.fingerprint,
        design_root_ids=frozen_model.design_root_ids,
        seed=_role_seed(role),
        match_namespace=_role_namespace(role),
        candidate_ids=ai.value_dataset.candidate_ids,
        candidate_deck_fingerprints=ai.value_dataset.candidate_deck_fingerprints,
        intent_ids=ai.value_dataset.intent_ids,
        roots=roots,
    )
    if ExpandedRootMeanPanel.from_dict(json.loads(panel.to_json())) != panel:
        raise RuntimeError("Phase 6AK panel roundtrip changed values")
    return panel


@dataclass(frozen=True, slots=True)
class ExpandedRootMeanCalibration:
    source_model_fingerprint: str
    calibration_panel_fingerprint: str
    calibration_root_ids: tuple[str, ...]
    conformal_radius: float
    calibration_point_metric: BranchValuePointMetric
    validation_roots_collected_before_freeze: int = 0
    validation_targets_used_for_calibration: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.source_model_fingerprint, "source_model_fingerprint")
        _validate_digest(
            self.calibration_panel_fingerprint,
            "calibration_panel_fingerprint",
        )
        if len(self.calibration_root_ids) != 36 or len(set(self.calibration_root_ids)) != 36:
            raise ValueError("Phase 6AK calibration root ledger is incomplete")
        if not 0 <= self.conformal_radius <= 2:
            raise ValueError("Phase 6AK conformal radius is invalid")
        if self.calibration_point_metric.role != "fresh_calibration_bounds":
            raise ValueError("Phase 6AK calibration point role is invalid")
        if any(
            (
                self.validation_roots_collected_before_freeze != 0,
                self.validation_targets_used_for_calibration,
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
            )
        ):
            raise ValueError("Phase 6AK calibration promoted future/downstream state")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPANDED_ROOT_MEAN_SCHEMA_VERSION,
            "source_model_fingerprint": self.source_model_fingerprint,
            "calibration_panel_fingerprint": self.calibration_panel_fingerprint,
            "calibration_root_ids": list(self.calibration_root_ids),
            "conformal_radius": self.conformal_radius,
            "calibration_point_metric": asdict(self.calibration_point_metric),
            "claims": {
                "validation_roots_collected_before_freeze": self.validation_roots_collected_before_freeze,
                "validation_targets_used_for_calibration": self.validation_targets_used_for_calibration,
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "safe_resolver_enabled": self.safe_resolver_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpandedRootMeanCalibration":
        claims = raw["claims"]
        result = cls(
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            calibration_panel_fingerprint=str(raw["calibration_panel_fingerprint"]),
            calibration_root_ids=tuple(str(value) for value in raw["calibration_root_ids"]),
            conformal_radius=float(raw["conformal_radius"]),
            calibration_point_metric=BranchValuePointMetric(
                **dict(raw["calibration_point_metric"])
            ),
            validation_roots_collected_before_freeze=int(
                claims["validation_roots_collected_before_freeze"]
            ),
            validation_targets_used_for_calibration=bool(
                claims["validation_targets_used_for_calibration"]
            ),
            counterfactual_value_compatible=bool(
                claims["counterfactual_value_compatible"]
            ),
            safe_resolver_enabled=bool(claims["safe_resolver_enabled"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AK calibration fingerprint mismatch")
        return result


def calibrate_expanded_root_mean_model(
    panel: ExpandedRootMeanPanel,
    model: ExpandedRootMeanModel,
) -> ExpandedRootMeanCalibration:
    if panel.role != "calibration":
        raise ValueError("Phase 6AK bounds require the calibration panel")
    if (
        panel.source_model_fingerprint != model.fingerprint
        or panel.source_phase6ai_dataset_fingerprint
        != model.source_phase6ai_dataset_fingerprint
        or panel.source_phase6aj_dataset_fingerprint
        != model.source_phase6aj_dataset_fingerprint
        or panel.design_root_ids != model.design_root_ids
    ):
        raise ValueError("Phase 6AK calibration panel/model provenance mismatch")
    examples = build_branch_value_examples_from_roots(
        panel.roots,
        panel.candidate_ids,
        expected_count=36,
    )
    predictions = restore_expanded_root_mean_model(model).predict_examples(examples)
    result = ExpandedRootMeanCalibration(
        source_model_fingerprint=model.fingerprint,
        calibration_panel_fingerprint=panel.fingerprint,
        calibration_root_ids=tuple(sorted(item.root_id for item in examples)),
        conformal_radius=_conformal_radius(examples, predictions),
        calibration_point_metric=_point_metric(
            "fresh_calibration_bounds",
            examples,
            predictions,
        ),
    )
    if ExpandedRootMeanCalibration.from_dict(json.loads(result.to_json())) != result:
        raise RuntimeError("Phase 6AK calibration roundtrip changed values")
    return result


def _bootstrap_zero_contrast_lower_bound(
    targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> float:
    model_errors = _contrast_root_errors(targets, predictions)
    zero_errors = _contrast_root_errors(
        targets,
        [[0.0] * len(OPPONENT_DEVIATION_ACTION_IDS) for _ in targets],
    )
    improvements = [
        baseline - model
        for baseline, model in zip(zero_errors, model_errors, strict=True)
    ]
    rng = random.Random(PHASE6_EXPANDED_ROOT_MEAN_BOOTSTRAP_SEED)
    means = []
    for _ in range(PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(len(improvements)) for _ in improvements]
        means.append(statistics.mean(improvements[index] for index in indices))
    means.sort()
    lower_index = max(
        0,
        math.ceil(
            (1.0 - PHASE6_BRANCH_VALUE_COVERAGE)
            * PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES
        )
        - 1,
    )
    return means[lower_index]


@dataclass(frozen=True, slots=True)
class ExpandedRootMeanEvaluation:
    validation_panel_fingerprint: str
    source_model_fingerprint: str
    source_calibration_fingerprint: str
    source_phase6aj_baseline_model_fingerprint: str
    calibration_root_ids: tuple[str, ...]
    validation_root_ids: tuple[str, ...]
    validation_point_metric: BranchValuePointMetric
    phase6aj_baseline_validation_mse: float
    slice_metrics: tuple[BranchValueSliceMetric, ...]
    maximum_target_standard_error: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    zero_contrast_bootstrap_lower_bound: float
    sampling_precision_gate_passed: bool
    point_mse_gate_passed: bool
    contrast_relative_gate_passed: bool
    contrast_retention_gate_passed: bool
    contrast_bootstrap_gate_passed: bool
    interval_gate_passed: bool
    collection_holdout_gate_passed: bool
    action_sensitive_point_model_validated: bool
    conditional_expected_mean_model_admitted: bool
    counterfactual_value_compatible: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "validation_panel_fingerprint",
            "source_model_fingerprint",
            "source_calibration_fingerprint",
            "source_phase6aj_baseline_model_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if (
            len(self.calibration_root_ids) != 36
            or len(set(self.calibration_root_ids)) != 36
            or len(self.validation_root_ids) != 36
            or len(set(self.validation_root_ids)) != 36
            or set(self.calibration_root_ids) & set(self.validation_root_ids)
        ):
            raise ValueError("Phase 6AK evaluation root ledgers are invalid")
        if self.validation_point_metric.role != "fresh_validation_untouched":
            raise ValueError("Phase 6AK validation point role is invalid")
        if not self.slice_metrics:
            raise ValueError("Phase 6AK validation slices are empty")
        if self.phase6aj_baseline_validation_mse < 0 or not math.isfinite(
            self.phase6aj_baseline_validation_mse
        ):
            raise ValueError("Phase 6AK source-baseline MSE is invalid")
        for name in (
            "maximum_target_standard_error",
            "validation_coverage",
            "validation_mean_width",
            "minimum_validation_slice_coverage",
            "zero_contrast_bootstrap_lower_bound",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"Phase 6AK {name} is nonfinite")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("Phase 6AK validation coverage is invalid")
        if not 0 <= self.minimum_validation_slice_coverage <= 1:
            raise ValueError("Phase 6AK minimum slice coverage is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AK validation width is invalid")

        metric = self.validation_point_metric
        expected_sampling = (
            self.maximum_target_standard_error
            <= PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
        )
        expected_point = metric.mse < min(
            metric.zero_mse,
            metric.root_heuristic_mse,
            metric.branch_heuristic_mse,
            self.phase6aj_baseline_validation_mse,
        )
        expected_relative = metric.action_contrast_mse < (
            PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR
            * metric.zero_action_contrast_mse
        )
        expected_retention = metric.action_contrast_mse <= (
            1.01 * metric.branch_action_contrast_mse
        )
        expected_bootstrap = self.zero_contrast_bootstrap_lower_bound > 0
        expected_interval = (
            self.validation_coverage >= PHASE6_BRANCH_VALUE_COVERAGE
            and self.validation_mean_width
            <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
            and self.minimum_validation_slice_coverage
            >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
        )
        expected_gates = (
            ("sampling_precision_gate_passed", expected_sampling),
            ("point_mse_gate_passed", expected_point),
            ("contrast_relative_gate_passed", expected_relative),
            ("contrast_retention_gate_passed", expected_retention),
            ("contrast_bootstrap_gate_passed", expected_bootstrap),
            ("interval_gate_passed", expected_interval),
        )
        for name, expected in expected_gates:
            if getattr(self, name) != expected:
                raise ValueError(f"Phase 6AK {name} is inconsistent")
        point_admitted = all(
            (
                expected_sampling,
                expected_point,
                expected_relative,
                expected_retention,
                expected_bootstrap,
                self.collection_holdout_gate_passed,
            )
        )
        if self.action_sensitive_point_model_validated != point_admitted:
            raise ValueError("Phase 6AK point-model admission is inconsistent")
        if self.conditional_expected_mean_model_admitted != (
            point_admitted and expected_interval
        ):
            raise ValueError("Phase 6AK conditional-mean admission is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AK evaluation promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPANDED_ROOT_MEAN_SCHEMA_VERSION,
            "validation_panel_fingerprint": self.validation_panel_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_calibration_fingerprint": self.source_calibration_fingerprint,
            "source_phase6aj_baseline_model_fingerprint": (
                self.source_phase6aj_baseline_model_fingerprint
            ),
            "calibration_root_ids": list(self.calibration_root_ids),
            "validation_root_ids": list(self.validation_root_ids),
            "validation_point_metric": asdict(self.validation_point_metric),
            "phase6aj_baseline_validation_mse": self.phase6aj_baseline_validation_mse,
            "slice_metrics": [asdict(item) for item in self.slice_metrics],
            "maximum_target_standard_error": self.maximum_target_standard_error,
            "validation_coverage": self.validation_coverage,
            "validation_mean_width": self.validation_mean_width,
            "minimum_validation_slice_coverage": self.minimum_validation_slice_coverage,
            "bootstrap": {
                "seed": PHASE6_EXPANDED_ROOT_MEAN_BOOTSTRAP_SEED,
                "resamples": PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES,
                "zero_contrast_lower_bound": self.zero_contrast_bootstrap_lower_bound,
            },
            "gates": {
                "sampling_precision_gate_passed": self.sampling_precision_gate_passed,
                "point_mse_gate_passed": self.point_mse_gate_passed,
                "contrast_relative_gate_passed": self.contrast_relative_gate_passed,
                "contrast_retention_gate_passed": self.contrast_retention_gate_passed,
                "contrast_bootstrap_gate_passed": self.contrast_bootstrap_gate_passed,
                "interval_gate_passed": self.interval_gate_passed,
                "collection_holdout_gate_passed": self.collection_holdout_gate_passed,
                "action_sensitive_point_model_validated": self.action_sensitive_point_model_validated,
                "conditional_expected_mean_model_admitted": self.conditional_expected_mean_model_admitted,
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "equilibrium_convergence_validated": self.equilibrium_convergence_validated,
                "safe_resolver_enabled": self.safe_resolver_enabled,
                "opponent_exploitation_enabled": self.opponent_exploitation_enabled,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpandedRootMeanEvaluation":
        bootstrap = raw["bootstrap"]
        if (
            int(bootstrap["seed"]) != PHASE6_EXPANDED_ROOT_MEAN_BOOTSTRAP_SEED
            or int(bootstrap["resamples"]) != PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError("Phase 6AK bootstrap protocol drifted")
        gates = raw["gates"]
        result = cls(
            validation_panel_fingerprint=str(raw["validation_panel_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            source_calibration_fingerprint=str(raw["source_calibration_fingerprint"]),
            source_phase6aj_baseline_model_fingerprint=str(
                raw["source_phase6aj_baseline_model_fingerprint"]
            ),
            calibration_root_ids=tuple(str(value) for value in raw["calibration_root_ids"]),
            validation_root_ids=tuple(str(value) for value in raw["validation_root_ids"]),
            validation_point_metric=BranchValuePointMetric(
                **dict(raw["validation_point_metric"])
            ),
            phase6aj_baseline_validation_mse=float(
                raw["phase6aj_baseline_validation_mse"]
            ),
            slice_metrics=tuple(
                BranchValueSliceMetric(**dict(item)) for item in raw["slice_metrics"]
            ),
            maximum_target_standard_error=float(raw["maximum_target_standard_error"]),
            validation_coverage=float(raw["validation_coverage"]),
            validation_mean_width=float(raw["validation_mean_width"]),
            minimum_validation_slice_coverage=float(
                raw["minimum_validation_slice_coverage"]
            ),
            zero_contrast_bootstrap_lower_bound=float(
                bootstrap["zero_contrast_lower_bound"]
            ),
            sampling_precision_gate_passed=bool(gates["sampling_precision_gate_passed"]),
            point_mse_gate_passed=bool(gates["point_mse_gate_passed"]),
            contrast_relative_gate_passed=bool(gates["contrast_relative_gate_passed"]),
            contrast_retention_gate_passed=bool(gates["contrast_retention_gate_passed"]),
            contrast_bootstrap_gate_passed=bool(gates["contrast_bootstrap_gate_passed"]),
            interval_gate_passed=bool(gates["interval_gate_passed"]),
            collection_holdout_gate_passed=bool(
                gates["collection_holdout_gate_passed"]
            ),
            action_sensitive_point_model_validated=bool(
                gates["action_sensitive_point_model_validated"]
            ),
            conditional_expected_mean_model_admitted=bool(
                gates["conditional_expected_mean_model_admitted"]
            ),
            counterfactual_value_compatible=bool(gates["counterfactual_value_compatible"]),
            equilibrium_convergence_validated=bool(
                gates["equilibrium_convergence_validated"]
            ),
            safe_resolver_enabled=bool(gates["safe_resolver_enabled"]),
            opponent_exploitation_enabled=bool(gates["opponent_exploitation_enabled"]),
            official_strategy_control_enabled=bool(gates["official_strategy_control_enabled"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AK evaluation fingerprint mismatch")
        return result


def evaluate_expanded_root_mean_model(
    calibration_panel: ExpandedRootMeanPanel,
    validation_panel: ExpandedRootMeanPanel,
    model: ExpandedRootMeanModel,
    calibration: ExpandedRootMeanCalibration,
    phase6aj_baseline_model: BranchHybridModel,
) -> ExpandedRootMeanEvaluation:
    if calibration_panel.role != "calibration" or validation_panel.role != "validation":
        raise ValueError("Phase 6AK evaluation panel roles are invalid")
    if (
        calibration.source_model_fingerprint != model.fingerprint
        or calibration.calibration_panel_fingerprint != calibration_panel.fingerprint
        or validation_panel.source_model_fingerprint != model.fingerprint
        or model.source_phase6aj_model_fingerprint != phase6aj_baseline_model.fingerprint
    ):
        raise ValueError("Phase 6AK evaluation provenance mismatch")
    examples = build_branch_value_examples_from_roots(
        validation_panel.roots,
        validation_panel.candidate_ids,
        expected_count=36,
    )
    predictions = restore_expanded_root_mean_model(model).predict_examples(examples)
    baseline_predictions = _hybrid_predictions(
        phase6aj_baseline_model.source_model,
        examples,
        mean_blend=phase6aj_baseline_model.selected_mean_blend,
        advantage_scale=phase6aj_baseline_model.selected_advantage_scale,
    )
    point = _point_metric("fresh_validation_untouched", examples, predictions)
    baseline_mse = _mse(_targets(examples), baseline_predictions)
    covered: list[bool] = []
    widths: list[float] = []
    flat = []
    for example, group in zip(examples, predictions, strict=True):
        for action, prediction in zip(example.actions, group, strict=True):
            item_covered, width = _interval_result(
                action,
                prediction,
                calibration.conformal_radius,
            )
            covered.append(item_covered)
            widths.append(width)
            flat.append((example, action, prediction))
    slice_metrics = []
    slice_specs = (
        ("action", OPPONENT_DEVIATION_ACTION_IDS, lambda example, action: action.action_id),
        (
            "candidate",
            validation_panel.candidate_ids,
            lambda example, action: example.candidate_id_hypothesis_input,
        ),
        (
            "intent",
            validation_panel.intent_ids,
            lambda example, action: example.intent_id_supervision_only,
        ),
    )
    for dimension, identifiers, selector in slice_specs:
        for identifier in identifiers:
            indices = [
                index
                for index, (example, action, _) in enumerate(flat)
                if selector(example, action) == identifier
            ]
            if not indices:
                raise RuntimeError(f"Phase 6AK empty slice: {dimension}/{identifier}")
            slice_metrics.append(
                BranchValueSliceMetric(
                    dimension=dimension,
                    slice_id=identifier,
                    validation_row_count=len(indices),
                    validation_mse=statistics.mean(
                        (
                            flat[index][2]
                            - flat[index][1].empirical_player0_value
                        )
                        ** 2
                        for index in indices
                    ),
                    validation_coverage=(
                        sum(covered[index] for index in indices) / len(indices)
                    ),
                    validation_mean_width=statistics.mean(
                        widths[index] for index in indices
                    ),
                )
            )
    maximum_se = max(
        action.target_standard_error
        for example in examples
        for action in example.actions
    )
    coverage = sum(covered) / len(covered)
    mean_width = statistics.mean(widths)
    minimum_slice = min(item.validation_coverage for item in slice_metrics)
    bootstrap_lower = _bootstrap_zero_contrast_lower_bound(
        _targets(examples),
        predictions,
    )
    sampling_gate = maximum_se <= PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
    point_gate = point.mse < min(
        point.zero_mse,
        point.root_heuristic_mse,
        point.branch_heuristic_mse,
        baseline_mse,
    )
    relative_gate = point.action_contrast_mse < (
        PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR
        * point.zero_action_contrast_mse
    )
    retention_gate = point.action_contrast_mse <= (
        1.01 * point.branch_action_contrast_mse
    )
    bootstrap_gate = bootstrap_lower > 0
    interval_gate = (
        coverage >= PHASE6_BRANCH_VALUE_COVERAGE
        and mean_width <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
        and minimum_slice >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
    )
    calibration_ids = {item.root_id for item in calibration_panel.roots}
    validation_ids = {item.root_id for item in validation_panel.roots}
    design_ids = set(model.design_root_ids)
    collection_holdout_gate = all(
        (
            not (calibration_ids & validation_ids),
            not (calibration_ids & design_ids),
            not (validation_ids & design_ids),
            set(calibration.calibration_root_ids) == calibration_ids,
            calibration_panel.design_root_ids == model.design_root_ids,
            validation_panel.design_root_ids == model.design_root_ids,
            calibration_panel.source_phase6ai_dataset_fingerprint
            == model.source_phase6ai_dataset_fingerprint,
            validation_panel.source_phase6ai_dataset_fingerprint
            == model.source_phase6ai_dataset_fingerprint,
            calibration_panel.source_phase6aj_dataset_fingerprint
            == model.source_phase6aj_dataset_fingerprint,
            validation_panel.source_phase6aj_dataset_fingerprint
            == model.source_phase6aj_dataset_fingerprint,
            calibration.validation_roots_collected_before_freeze == 0,
            not calibration.validation_targets_used_for_calibration,
        )
    )
    point_admitted = all(
        (
            sampling_gate,
            point_gate,
            relative_gate,
            retention_gate,
            bootstrap_gate,
            collection_holdout_gate,
        )
    )
    result = ExpandedRootMeanEvaluation(
        validation_panel_fingerprint=validation_panel.fingerprint,
        source_model_fingerprint=model.fingerprint,
        source_calibration_fingerprint=calibration.fingerprint,
        source_phase6aj_baseline_model_fingerprint=phase6aj_baseline_model.fingerprint,
        calibration_root_ids=tuple(sorted(calibration_ids)),
        validation_root_ids=tuple(sorted(validation_ids)),
        validation_point_metric=point,
        phase6aj_baseline_validation_mse=baseline_mse,
        slice_metrics=tuple(slice_metrics),
        maximum_target_standard_error=maximum_se,
        validation_coverage=coverage,
        validation_mean_width=mean_width,
        minimum_validation_slice_coverage=minimum_slice,
        zero_contrast_bootstrap_lower_bound=bootstrap_lower,
        sampling_precision_gate_passed=sampling_gate,
        point_mse_gate_passed=point_gate,
        contrast_relative_gate_passed=relative_gate,
        contrast_retention_gate_passed=retention_gate,
        contrast_bootstrap_gate_passed=bootstrap_gate,
        interval_gate_passed=interval_gate,
        collection_holdout_gate_passed=collection_holdout_gate,
        action_sensitive_point_model_validated=point_admitted,
        conditional_expected_mean_model_admitted=(point_admitted and interval_gate),
    )
    if ExpandedRootMeanEvaluation.from_dict(json.loads(result.to_json())) != result:
        raise RuntimeError("Phase 6AK evaluation roundtrip changed values")
    return result


@dataclass(frozen=True, slots=True)
class Phase6ExpandedRootMeanEvidence:
    status: str
    model_fingerprint: str
    calibration_panel_fingerprint: str
    calibration_fingerprint: str
    validation_panel_fingerprint: str
    evaluation_fingerprint: str
    root_count: int
    calibration_root_count: int
    validation_root_count: int
    branch_observation_count: int
    surface_count: int
    action_target_count: int
    terminal_rollout_count: int
    live_selection_count: int
    native_transition_count: int
    conformal_radius: float
    validation_mse: float
    validation_zero_mse: float
    validation_root_heuristic_mse: float
    validation_branch_heuristic_mse: float
    phase6aj_baseline_validation_mse: float
    validation_action_contrast_mse: float
    validation_zero_action_contrast_mse: float
    validation_branch_action_contrast_mse: float
    zero_contrast_bootstrap_lower_bound: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    maximum_target_standard_error: float
    sampling_precision_gate_passed: bool
    point_mse_gate_passed: bool
    contrast_relative_gate_passed: bool
    contrast_retention_gate_passed: bool
    contrast_bootstrap_gate_passed: bool
    interval_gate_passed: bool
    collection_holdout_gate_passed: bool
    action_sensitive_point_model_validated: bool
    conditional_expected_mean_model_admitted: bool
    counterfactual_value_compatible: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "model_fingerprint",
            "calibration_panel_fingerprint",
            "calibration_fingerprint",
            "validation_panel_fingerprint",
            "evaluation_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if (
            self.root_count != 72
            or self.calibration_root_count != 36
            or self.validation_root_count != 36
            or self.branch_observation_count != 720
            or self.surface_count != 144
            or self.action_target_count != 360
            or self.terminal_rollout_count != 5_760
            or self.live_selection_count <= 0
            or self.native_transition_count <= 0
        ):
            raise ValueError("Phase 6AK evidence accounting is incomplete")
        expected_status = (
            "fresh_collection_heldout_conditional_expected_mean_admitted_downstream_fail_closed"
            if self.conditional_expected_mean_model_admitted
            else (
                "fresh_collection_heldout_action_sensitive_point_validated_intervals_rejected_downstream_fail_closed"
                if self.action_sensitive_point_model_validated
                else "fresh_expanded_root_mean_rejected_downstream_fail_closed"
            )
        )
        if self.status != expected_status:
            raise ValueError("Phase 6AK evidence status is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AK evidence promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {"schema_version": PHASE6_EXPANDED_ROOT_MEAN_SCHEMA_VERSION, **asdict(self)}
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6ExpandedRootMeanEvidence":
        values = dict(raw)
        values.pop("schema_version", None)
        fingerprint = values.pop("fingerprint", None)
        result = cls(**values)
        if fingerprint not in (None, result.fingerprint):
            raise ValueError("Phase 6AK evidence fingerprint mismatch")
        return result

    def to_markdown(self) -> str:
        return f"""# Phase 6AK expanded semiparametric root mean

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Fresh calibration / validation roots | {self.calibration_root_count} / {self.validation_root_count} |
| Immediate native branch observations | {self.branch_observation_count} |
| Supplied-allocation surfaces / root-action targets | {self.surface_count} / {self.action_target_count} |
| Genuine terminal native continuations | {self.terminal_rollout_count} |
| Genuine live selections / native transitions | {self.live_selection_count} / {self.native_transition_count} |
| Frozen conformal radius | `{self.conformal_radius:.6f}` |
| Validation ridge / Phase 6AJ / zero / root / branch MSE | `{self.validation_mse:.6f}` / `{self.phase6aj_baseline_validation_mse:.6f}` / `{self.validation_zero_mse:.6f}` / `{self.validation_root_heuristic_mse:.6f}` / `{self.validation_branch_heuristic_mse:.6f}` |
| Validation contrast ridge / zero / branch MSE | `{self.validation_action_contrast_mse:.6f}` / `{self.validation_zero_action_contrast_mse:.6f}` / `{self.validation_branch_action_contrast_mse:.6f}` |
| 90% bootstrap zero-minus-ridge contrast lower bound | `{self.zero_contrast_bootstrap_lower_bound:.6f}` |
| Validation mean-CI coverage / width / worst slice | `{self.validation_coverage:.6f}` / `{self.validation_mean_width:.6f}` / `{self.minimum_validation_slice_coverage:.6f}` |
| Maximum target SE | `{self.maximum_target_standard_error:.6f}` |
| Sampling / point / relative contrast / retention / bootstrap / interval / collection holdout | **{str(self.sampling_precision_gate_passed).lower()} / {str(self.point_mse_gate_passed).lower()} / {str(self.contrast_relative_gate_passed).lower()} / {str(self.contrast_retention_gate_passed).lower()} / {str(self.contrast_bootstrap_gate_passed).lower()} / {str(self.interval_gate_passed).lower()} / {str(self.collection_holdout_gate_passed).lower()}** |
| Action-sensitive point / conditional expected mean | **{str(self.action_sensitive_point_model_validated).lower()} / {str(self.conditional_expected_mean_model_admitted).lower()}** |
| Counterfactual / equilibrium / safe resolver / exploitation / control | **false / false / false / false / false** |

Complete matches, legal selections, retained snapshots, immediate branches,
and terminal descendants are genuine official-engine execution. The supplied
range, five-plan boundary, tactical executor, continuation policy, fixed branch
heuristic, ridge model, bootstrap, and intervals are abstractions. No Kaggle
submission or playing-strength claim is enabled.
"""


@dataclass(frozen=True, slots=True)
class Phase6ExpandedRootMeanArtifacts:
    model: ExpandedRootMeanModel
    calibration_panel: ExpandedRootMeanPanel
    calibration: ExpandedRootMeanCalibration
    validation_panel: ExpandedRootMeanPanel
    evaluation: ExpandedRootMeanEvaluation
    evidence: Phase6ExpandedRootMeanEvidence


def run_phase6_expanded_root_mean(
    *,
    project_root: str | Path | None = None,
    model: ExpandedRootMeanModel,
    calibration_panel: ExpandedRootMeanPanel,
    calibration: ExpandedRootMeanCalibration,
    validation_panel: ExpandedRootMeanPanel,
) -> Phase6ExpandedRootMeanArtifacts:
    _, _, phase6aj_model = _load_phase6ak_design(project_root)
    evaluation = evaluate_expanded_root_mean_model(
        calibration_panel,
        validation_panel,
        model,
        calibration,
        phase6aj_model,
    )
    point = evaluation.validation_point_metric
    admitted = evaluation.conditional_expected_mean_model_admitted
    point_admitted = evaluation.action_sensitive_point_model_validated
    evidence = Phase6ExpandedRootMeanEvidence(
        status=(
            "fresh_collection_heldout_conditional_expected_mean_admitted_downstream_fail_closed"
            if admitted
            else (
                "fresh_collection_heldout_action_sensitive_point_validated_intervals_rejected_downstream_fail_closed"
                if point_admitted
                else "fresh_expanded_root_mean_rejected_downstream_fail_closed"
            )
        ),
        model_fingerprint=model.fingerprint,
        calibration_panel_fingerprint=calibration_panel.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        validation_panel_fingerprint=validation_panel.fingerprint,
        evaluation_fingerprint=evaluation.fingerprint,
        root_count=len(calibration_panel.roots) + len(validation_panel.roots),
        calibration_root_count=len(calibration_panel.roots),
        validation_root_count=len(validation_panel.roots),
        branch_observation_count=sum(
            len(item.branch_observations)
            for panel in (calibration_panel, validation_panel)
            for item in panel.roots
        ),
        surface_count=sum(
            len(item.value_root.surfaces)
            for panel in (calibration_panel, validation_panel)
            for item in panel.roots
        ),
        action_target_count=(
            (len(calibration_panel.roots) + len(validation_panel.roots))
            * len(OPPONENT_DEVIATION_ACTION_IDS)
        ),
        terminal_rollout_count=(
            calibration_panel.terminal_rollout_count
            + validation_panel.terminal_rollout_count
        ),
        live_selection_count=(
            calibration_panel.live_selection_count
            + validation_panel.live_selection_count
        ),
        native_transition_count=(
            calibration_panel.native_transition_count
            + validation_panel.native_transition_count
        ),
        conformal_radius=calibration.conformal_radius,
        validation_mse=point.mse,
        validation_zero_mse=point.zero_mse,
        validation_root_heuristic_mse=point.root_heuristic_mse,
        validation_branch_heuristic_mse=point.branch_heuristic_mse,
        phase6aj_baseline_validation_mse=(
            evaluation.phase6aj_baseline_validation_mse
        ),
        validation_action_contrast_mse=point.action_contrast_mse,
        validation_zero_action_contrast_mse=point.zero_action_contrast_mse,
        validation_branch_action_contrast_mse=point.branch_action_contrast_mse,
        zero_contrast_bootstrap_lower_bound=(
            evaluation.zero_contrast_bootstrap_lower_bound
        ),
        validation_coverage=evaluation.validation_coverage,
        validation_mean_width=evaluation.validation_mean_width,
        minimum_validation_slice_coverage=(
            evaluation.minimum_validation_slice_coverage
        ),
        maximum_target_standard_error=evaluation.maximum_target_standard_error,
        sampling_precision_gate_passed=evaluation.sampling_precision_gate_passed,
        point_mse_gate_passed=evaluation.point_mse_gate_passed,
        contrast_relative_gate_passed=evaluation.contrast_relative_gate_passed,
        contrast_retention_gate_passed=evaluation.contrast_retention_gate_passed,
        contrast_bootstrap_gate_passed=evaluation.contrast_bootstrap_gate_passed,
        interval_gate_passed=evaluation.interval_gate_passed,
        collection_holdout_gate_passed=(
            evaluation.collection_holdout_gate_passed
        ),
        action_sensitive_point_model_validated=point_admitted,
        conditional_expected_mean_model_admitted=admitted,
    )
    if Phase6ExpandedRootMeanEvidence.from_dict(json.loads(evidence.to_json())) != evidence:
        raise RuntimeError("Phase 6AK evidence roundtrip changed values")
    return Phase6ExpandedRootMeanArtifacts(
        model=model,
        calibration_panel=calibration_panel,
        calibration=calibration,
        validation_panel=validation_panel,
        evaluation=evaluation,
        evidence=evidence,
    )


def _write_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_root_shards(directory: Path) -> tuple[BranchConditionedRoot, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        BranchConditionedRoot.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in sorted(directory.glob("*.json"))
    )


def main() -> None:
    import argparse

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare-model", "freeze-calibration", "evaluate"),
        required=True,
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=project_root / "artifacts" / "phase6_expanded_root_mean",
    )
    parser.add_argument(
        "--calibration-shard-directory",
        type=Path,
        default=project_root / "work" / "phase6ak_calibration_shards",
    )
    parser.add_argument(
        "--validation-shard-directory",
        type=Path,
        default=project_root / "work" / "phase6ak_validation_shards",
    )
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    model_path = target / "expanded_root_mean_model.json"
    calibration_panel_path = target / "calibration_panel.json"
    calibration_path = target / "expanded_root_mean_calibration.json"
    validation_panel_path = target / "validation_panel.json"

    if args.stage == "prepare-model":
        existing_shards = len(_load_root_shards(args.calibration_shard_directory)) + len(
            _load_root_shards(args.validation_shard_directory)
        )
        if existing_shards:
            raise RuntimeError("Phase 6AK model must be frozen before collection")
        if model_path.is_file():
            model = ExpandedRootMeanModel.from_dict(
                json.loads(model_path.read_text(encoding="utf-8"))
            )
        else:
            model = train_expanded_root_mean_model(project_root=project_root)
            _write_artifact(model_path, model.to_json() + "\n")
        selected = min(model.selection_metrics, key=_selection_key)
        print(
            json.dumps(
                {
                    "status": "phase6ak_model_frozen_before_collection",
                    "model_fingerprint": model.fingerprint,
                    "selected_model_form": model.selected_model_form,
                    "selected_penalty_per_root": model.selected_penalty_per_root,
                    "worst_normalized_mse_ratio": selected.worst_normalized_mse_ratio,
                    "mean_normalized_mse_ratio": selected.mean_normalized_mse_ratio,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not model_path.is_file():
        raise FileNotFoundError("Phase 6AK frozen model artifact is missing")
    model = ExpandedRootMeanModel.from_dict(
        json.loads(model_path.read_text(encoding="utf-8"))
    )

    if args.stage == "freeze-calibration":
        validation_shards = _load_root_shards(args.validation_shard_directory)
        if validation_shards or validation_panel_path.exists():
            raise RuntimeError("Phase 6AK validation was opened before bounds freeze")
        calibration_shards = _load_root_shards(args.calibration_shard_directory)
        if len(calibration_shards) != 36:
            raise RuntimeError("Phase 6AK requires 36 calibration shards")
        panel = collect_expanded_root_mean_panel(
            "calibration",
            project_root=project_root,
            model=model,
            existing_roots=calibration_shards,
        )
        calibration = calibrate_expanded_root_mean_model(panel, model)
        _write_artifact(calibration_panel_path, panel.to_json() + "\n")
        _write_artifact(calibration_path, calibration.to_json() + "\n")
        print(
            json.dumps(
                {
                    "status": "phase6ak_bounds_frozen_before_validation",
                    "calibration_panel_fingerprint": panel.fingerprint,
                    "calibration_fingerprint": calibration.fingerprint,
                    "conformal_radius": calibration.conformal_radius,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not calibration_panel_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("Phase 6AK frozen calibration artifacts are missing")
    calibration_panel = ExpandedRootMeanPanel.from_dict(
        json.loads(calibration_panel_path.read_text(encoding="utf-8"))
    )
    calibration = ExpandedRootMeanCalibration.from_dict(
        json.loads(calibration_path.read_text(encoding="utf-8"))
    )
    validation_shards = _load_root_shards(args.validation_shard_directory)
    if len(validation_shards) != 36:
        raise RuntimeError("Phase 6AK requires 36 validation shards")
    validation_panel = collect_expanded_root_mean_panel(
        "validation",
        project_root=project_root,
        model=model,
        existing_roots=validation_shards,
    )
    artifacts = run_phase6_expanded_root_mean(
        project_root=project_root,
        model=model,
        calibration_panel=calibration_panel,
        calibration=calibration,
        validation_panel=validation_panel,
    )
    _write_artifact(validation_panel_path, artifacts.validation_panel.to_json() + "\n")
    _write_artifact(
        target / "expanded_root_mean_evaluation.json",
        artifacts.evaluation.to_json() + "\n",
    )
    _write_artifact(
        target / "phase6ak_evidence.json",
        artifacts.evidence.to_json() + "\n",
    )
    _write_artifact(
        target / "phase6ak_evidence.md",
        artifacts.evidence.to_markdown(),
    )
    print(artifacts.evidence.to_json())


if __name__ == "__main__":
    main()
