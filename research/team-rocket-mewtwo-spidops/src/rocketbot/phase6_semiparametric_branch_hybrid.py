"""Phase 6AJ fresh confirmation of a semiparametric branch-value hybrid."""

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
    BranchValueCalibration,
    BranchValueModel,
    BranchValuePointMetric,
    BranchValueRootExample,
    BranchValueSliceMetric,
    _branch_heuristic_predictions,
    _build_branch_conditioned_root,
    _contrast_mse,
    _contrast_root_errors,
    _example_tensors,
    _fingerprint,
    _interval_result,
    _mse,
    _point_metric,
    _restore_network,
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
    PHASE6_EXPECTED_VALUE_MEAN_CI_Z,
    PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE,
    _capture_complete_match_late_root,
    _load_expected_value_sources,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_opponent_deviation_values import OPPONENT_DEVIATION_ACTION_IDS


PHASE6_BRANCH_HYBRID_SCHEMA_VERSION = 1
PHASE6_BRANCH_HYBRID_SEED = 6_051
PHASE6_BRANCH_HYBRID_BOOTSTRAP_SEED = 6_052
PHASE6_BRANCH_HYBRID_MEAN_BLEND_GRID = (0.25, 0.50, 0.75, 1.0)
PHASE6_BRANCH_HYBRID_ADVANTAGE_SCALE_GRID = (0.50, 0.75, 1.0, 1.25)


@dataclass(frozen=True, slots=True)
class BranchHybridValidationDataset:
    source_branch_dataset_fingerprint: str
    source_branch_model_fingerprint: str
    source_branch_calibration_fingerprint: str
    phase6ah_design_root_ids: tuple[str, ...]
    phase6ai_source_root_ids: tuple[str, ...]
    seed: int
    match_namespace: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    intent_ids: tuple[str, ...]
    roots: tuple[BranchConditionedRoot, ...]
    prior_roots_reused: int = 0
    public_branch_features_only: bool = True
    validation_targets_used_for_selection: bool = False
    validation_targets_used_for_calibration: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_branch_dataset_fingerprint",
            "source_branch_model_fingerprint",
            "source_branch_calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.seed != PHASE6_BRANCH_HYBRID_SEED or self.match_namespace != "phase6aj":
            raise ValueError("Phase 6AJ frozen seed/namespace drifted")
        if len(self.phase6ah_design_root_ids) != 180 or len(set(self.phase6ah_design_root_ids)) != 180:
            raise ValueError("Phase 6AJ Phase 6AH root ledger is incomplete")
        if len(self.phase6ai_source_root_ids) != 180 or len(set(self.phase6ai_source_root_ids)) != 180:
            raise ValueError("Phase 6AJ Phase 6AI root ledger is incomplete")
        if len(self.roots) != 36 or len({item.root_id for item in self.roots}) != 36:
            raise ValueError("Phase 6AJ validation panel is incomplete")
        root_ids = {item.root_id for item in self.roots}
        if root_ids & set(self.phase6ah_design_root_ids) or root_ids & set(self.phase6ai_source_root_ids):
            raise ValueError("Phase 6AJ reused a design/fitting root")
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
            raise ValueError("Phase 6AJ candidate/intent cross is incomplete")
        for root in self.roots:
            value = root.value_root
            if (
                not value.match_id.startswith("phase6aj-validation-action_first-")
                or value.split != "validation"
                or value.policy_variant_id_supervision_only != "action_first"
                or value.replica != 0
            ):
                raise ValueError("Phase 6AJ root escaped its frozen validation protocol")
        if any(
            (
                self.prior_roots_reused != 0,
                not self.public_branch_features_only,
                self.validation_targets_used_for_selection,
                self.validation_targets_used_for_calibration,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6AJ dataset promoted its abstraction boundary")
        if _raw_private_key_count(self.to_dict(include_fingerprint=False)) != 0:
            raise ValueError("Phase 6AJ dataset persisted a forbidden raw private key")

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
            "schema_version": PHASE6_BRANCH_HYBRID_SCHEMA_VERSION,
            "source_branch_dataset_fingerprint": self.source_branch_dataset_fingerprint,
            "source_branch_model_fingerprint": self.source_branch_model_fingerprint,
            "source_branch_calibration_fingerprint": self.source_branch_calibration_fingerprint,
            "phase6ah_design_root_ids": list(self.phase6ah_design_root_ids),
            "phase6ai_source_root_ids": list(self.phase6ai_source_root_ids),
            "seed": self.seed,
            "match_namespace": self.match_namespace,
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": dict(self.candidate_deck_fingerprints),
            "intent_ids": list(self.intent_ids),
            "roots": [item.to_dict() for item in self.roots],
            "claims": {
                "prior_roots_reused": self.prior_roots_reused,
                "public_branch_features_only": self.public_branch_features_only,
                "validation_targets_used_for_selection": (
                    self.validation_targets_used_for_selection
                ),
                "validation_targets_used_for_calibration": (
                    self.validation_targets_used_for_calibration
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchHybridValidationDataset":
        claims = raw["claims"]
        result = cls(
            source_branch_dataset_fingerprint=str(raw["source_branch_dataset_fingerprint"]),
            source_branch_model_fingerprint=str(raw["source_branch_model_fingerprint"]),
            source_branch_calibration_fingerprint=str(raw["source_branch_calibration_fingerprint"]),
            phase6ah_design_root_ids=tuple(str(value) for value in raw["phase6ah_design_root_ids"]),
            phase6ai_source_root_ids=tuple(str(value) for value in raw["phase6ai_source_root_ids"]),
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
            validation_targets_used_for_selection=bool(
                claims["validation_targets_used_for_selection"]
            ),
            validation_targets_used_for_calibration=bool(
                claims["validation_targets_used_for_calibration"]
            ),
            raw_private_card_ids_persisted=int(claims["raw_private_card_ids_persisted"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AJ validation dataset fingerprint mismatch")
        return result


def _load_phase6ai_artifacts(project_root: str | Path | None) -> tuple[
    BranchConditionedDataset,
    BranchValueModel,
    BranchValueCalibration,
]:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    directory = root / "artifacts" / "phase6_branch_conditioned_values"
    dataset = BranchConditionedDataset.from_dict(
        json.loads((directory / "branch_conditioned_dataset.json").read_text(encoding="utf-8"))
    )
    model = BranchValueModel.from_dict(
        json.loads((directory / "branch_value_model.json").read_text(encoding="utf-8"))
    )
    calibration = BranchValueCalibration.from_dict(
        json.loads((directory / "branch_value_calibration.json").read_text(encoding="utf-8"))
    )
    if model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("Phase 6AJ source model/dataset mismatch")
    if calibration.source_dataset_fingerprint != dataset.fingerprint or calibration.source_model_fingerprint != model.fingerprint:
        raise ValueError("Phase 6AJ source calibration mismatch")
    return dataset, model, calibration


def collect_branch_hybrid_validation_dataset(
    *,
    project_root: str | Path | None = None,
    existing_roots: Sequence[BranchConditionedRoot] = (),
    root_observer: Callable[[BranchConditionedRoot, int, int], None] | None = None,
) -> BranchHybridValidationDataset:
    source_dataset, source_model, source_calibration = _load_phase6ai_artifacts(project_root)
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
        raise ValueError("Phase 6AJ resume roots are duplicate/outside the grid")
    candidates = {item.archetype: item for item in sources.candidates}
    intents = {item.value: item for item in OPPONENT_INTENTS}
    for key in expected_keys:
        if key in by_key:
            continue
        candidate_id, intent_id = key
        candidate = candidates[candidate_id]
        intent = intents[intent_id]
        match_id = f"phase6aj-validation-action_first-{candidate_id}-{intent_id}-r0"
        last_error: Exception | None = None
        for attempt in range(1, PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS + 1):
            try:
                capture = _capture_complete_match_late_root(
                    battle_start=battle_start,
                    battle_select=battle_select,
                    battle_finish=battle_finish,
                    target=sources.target,
                    candidate=candidate,
                    intent=intent,
                    policy_variant="action_first",
                    replica=0,
                    capture_attempt=attempt,
                    maximum_live_selections=PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
                    match_id=match_id,
                )
                branch_root = _build_branch_conditioned_root(
                    capture,
                    sources=sources,
                    candidate=candidate,
                    intent=intent,
                    policy_variant="action_first",
                    split="validation",
                    replica=0,
                    match_id=match_id,
                    seed=PHASE6_BRANCH_HYBRID_SEED,
                    maximum_rollout_transitions=PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
                    maximum_tactical_transitions=PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
                    experiment_label="Phase 6AJ",
                )
                by_key[key] = branch_root
                if root_observer is not None:
                    root_observer(branch_root, len(by_key), len(expected_keys))
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Phase 6AJ exhausted attempts for {candidate_id}/{intent_id}") from last_error
    roots = tuple(by_key[key] for key in expected_keys)
    dataset = BranchHybridValidationDataset(
        source_branch_dataset_fingerprint=source_dataset.fingerprint,
        source_branch_model_fingerprint=source_model.fingerprint,
        source_branch_calibration_fingerprint=source_calibration.fingerprint,
        phase6ah_design_root_ids=source_dataset.design_root_ids,
        phase6ai_source_root_ids=tuple(sorted(item.root_id for item in source_dataset.roots)),
        seed=PHASE6_BRANCH_HYBRID_SEED,
        match_namespace="phase6aj",
        candidate_ids=source_dataset.value_dataset.candidate_ids,
        candidate_deck_fingerprints=source_dataset.value_dataset.candidate_deck_fingerprints,
        intent_ids=source_dataset.value_dataset.intent_ids,
        roots=roots,
    )
    if BranchHybridValidationDataset.from_dict(json.loads(dataset.to_json())) != dataset:
        raise RuntimeError("Phase 6AJ validation dataset roundtrip changed values")
    return dataset


def _hybrid_predictions(
    source_model: BranchValueModel,
    examples: Sequence[BranchValueRootExample],
    *,
    mean_blend: float,
    advantage_scale: float,
) -> list[list[float]]:
    network = _restore_network(
        source_model.root_normalizer,
        source_model.branch_normalizer,
        source_model.selected_hidden_dim,
        source_model.tensors,
    )
    root, branch, _, root_heuristic = _example_tensors(
        examples,
        source_model.root_normalizer,
        source_model.branch_normalizer,
    )
    with torch.no_grad():
        neural_mean, _ = network(root, branch)
        means = mean_blend * neural_mean + (1.0 - mean_blend) * root_heuristic
    branch_values = _branch_heuristic_predictions(examples)
    result = []
    for mean, action_values in zip(means.tolist(), branch_values, strict=True):
        center = statistics.mean(action_values)
        result.append(
            [
                max(-1.0, min(1.0, float(mean) + advantage_scale * (value - center)))
                for value in action_values
            ]
        )
    return result


@dataclass(frozen=True, slots=True)
class BranchHybridSelectionMetric:
    mean_blend: float
    advantage_scale: float
    selection_mse: float
    selection_zero_mse: float
    selection_root_heuristic_mse: float
    selection_branch_heuristic_mse: float
    selection_action_contrast_mse: float
    selection_zero_action_contrast_mse: float
    selection_score: float

    def __post_init__(self) -> None:
        if self.mean_blend not in PHASE6_BRANCH_HYBRID_MEAN_BLEND_GRID:
            raise ValueError("Phase 6AJ mean blend is unsupported")
        if self.advantage_scale not in PHASE6_BRANCH_HYBRID_ADVANTAGE_SCALE_GRID:
            raise ValueError("Phase 6AJ advantage scale is unsupported")
        if any(float(value) < 0 or not math.isfinite(float(value)) for value in asdict(self).values()):
            raise ValueError("Phase 6AJ selection metric is invalid")


@dataclass(frozen=True, slots=True)
class BranchHybridModel:
    source_dataset_fingerprint: str
    source_calibration_fingerprint: str
    source_model: BranchValueModel
    selected_mean_blend: float
    selected_advantage_scale: float
    selection_root_ids: tuple[str, ...]
    selection_metrics: tuple[BranchHybridSelectionMetric, ...]
    learned_action_advantage_used: bool = False
    causal_branch_heuristic_is_complete_action_advantage: bool = True
    validation_targets_used_for_selection: bool = False
    official_engine_executed: bool = False
    counterfactual_value_provider: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.source_dataset_fingerprint, "source_dataset_fingerprint")
        _validate_digest(self.source_calibration_fingerprint, "source_calibration_fingerprint")
        if self.source_model.source_dataset_fingerprint != self.source_dataset_fingerprint:
            raise ValueError("Phase 6AJ embedded source model/dataset mismatch")
        if len(self.selection_root_ids) != 36 or len(set(self.selection_root_ids)) != 36:
            raise ValueError("Phase 6AJ selection-root ledger is incomplete")
        expected_grid = {
            (mean_blend, advantage_scale)
            for mean_blend in PHASE6_BRANCH_HYBRID_MEAN_BLEND_GRID
            for advantage_scale in PHASE6_BRANCH_HYBRID_ADVANTAGE_SCALE_GRID
        }
        actual_grid = {
            (item.mean_blend, item.advantage_scale)
            for item in self.selection_metrics
        }
        if actual_grid != expected_grid or len(self.selection_metrics) != len(expected_grid):
            raise ValueError("Phase 6AJ model-selection grid is incomplete")
        selected = min(
            self.selection_metrics,
            key=lambda item: (
                item.selection_score,
                item.selection_mse,
                item.selection_action_contrast_mse,
                item.mean_blend,
                item.advantage_scale,
            ),
        )
        if (self.selected_mean_blend, self.selected_advantage_scale) != (
            selected.mean_blend,
            selected.advantage_scale,
        ):
            raise ValueError("Phase 6AJ selected configuration is inconsistent")
        if any(
            (
                self.learned_action_advantage_used,
                not self.causal_branch_heuristic_is_complete_action_advantage,
                self.validation_targets_used_for_selection,
                self.official_engine_executed,
                self.counterfactual_value_provider,
            )
        ):
            raise ValueError("Phase 6AJ model promoted its abstraction boundary")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_HYBRID_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_calibration_fingerprint": self.source_calibration_fingerprint,
            "source_model": self.source_model.to_dict(),
            "selected_mean_blend": self.selected_mean_blend,
            "selected_advantage_scale": self.selected_advantage_scale,
            "selection_root_ids": list(self.selection_root_ids),
            "selection_metrics": [asdict(item) for item in self.selection_metrics],
            "claims": {
                "learned_action_advantage_used": self.learned_action_advantage_used,
                "causal_branch_heuristic_is_complete_action_advantage": (
                    self.causal_branch_heuristic_is_complete_action_advantage
                ),
                "validation_targets_used_for_selection": self.validation_targets_used_for_selection,
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchHybridModel":
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            source_calibration_fingerprint=str(raw["source_calibration_fingerprint"]),
            source_model=BranchValueModel.from_dict(raw["source_model"]),
            selected_mean_blend=float(raw["selected_mean_blend"]),
            selected_advantage_scale=float(raw["selected_advantage_scale"]),
            selection_root_ids=tuple(str(value) for value in raw["selection_root_ids"]),
            selection_metrics=tuple(
                BranchHybridSelectionMetric(**dict(item)) for item in raw["selection_metrics"]
            ),
            learned_action_advantage_used=bool(claims["learned_action_advantage_used"]),
            causal_branch_heuristic_is_complete_action_advantage=bool(
                claims["causal_branch_heuristic_is_complete_action_advantage"]
            ),
            validation_targets_used_for_selection=bool(
                claims["validation_targets_used_for_selection"]
            ),
            official_engine_executed=bool(claims["official_engine_executed"]),
            counterfactual_value_provider=bool(claims["counterfactual_value_provider"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AJ model fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class BranchHybridCalibration:
    source_dataset_fingerprint: str
    source_model_fingerprint: str
    source_phase6ai_calibration_fingerprint: str
    calibration_root_ids: tuple[str, ...]
    conformal_radius: float
    calibration_point_metric: BranchValuePointMetric
    validation_targets_used_for_calibration: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_dataset_fingerprint",
            "source_model_fingerprint",
            "source_phase6ai_calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if len(self.calibration_root_ids) != 36 or not 0 <= self.conformal_radius <= 2:
            raise ValueError("Phase 6AJ calibration ledger/radius is invalid")
        if self.calibration_point_metric.role != "phase6ai_calibration_bounds":
            raise ValueError("Phase 6AJ calibration point metric role is invalid")
        if any(
            (
                self.validation_targets_used_for_calibration,
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
            )
        ):
            raise ValueError("Phase 6AJ calibration promoted downstream state")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_HYBRID_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_phase6ai_calibration_fingerprint": self.source_phase6ai_calibration_fingerprint,
            "calibration_root_ids": list(self.calibration_root_ids),
            "conformal_radius": self.conformal_radius,
            "calibration_point_metric": asdict(self.calibration_point_metric),
            "claims": {
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchHybridCalibration":
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            source_phase6ai_calibration_fingerprint=str(
                raw["source_phase6ai_calibration_fingerprint"]
            ),
            calibration_root_ids=tuple(str(value) for value in raw["calibration_root_ids"]),
            conformal_radius=float(raw["conformal_radius"]),
            calibration_point_metric=BranchValuePointMetric(**dict(raw["calibration_point_metric"])),
            validation_targets_used_for_calibration=bool(
                claims["validation_targets_used_for_calibration"]
            ),
            counterfactual_value_compatible=bool(claims["counterfactual_value_compatible"]),
            safe_resolver_enabled=bool(claims["safe_resolver_enabled"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AJ calibration fingerprint mismatch")
        return result


def prepare_branch_hybrid(
    source_dataset: BranchConditionedDataset,
    source_model: BranchValueModel,
    source_calibration: BranchValueCalibration,
) -> tuple[BranchHybridModel, BranchHybridCalibration]:
    if source_model.source_dataset_fingerprint != source_dataset.fingerprint:
        raise ValueError("Phase 6AJ source model/dataset mismatch")
    if (
        source_calibration.source_dataset_fingerprint != source_dataset.fingerprint
        or source_calibration.source_model_fingerprint != source_model.fingerprint
    ):
        raise ValueError("Phase 6AJ source calibration/model mismatch")
    examples = build_branch_value_examples(source_dataset)
    selection = tuple(
        item
        for item in examples
        if item.split == "calibration" and item.calibration_partition == "selection"
    )
    bounds = tuple(
        item
        for item in examples
        if item.split == "calibration" and item.calibration_partition == "bounds"
    )
    targets = _targets(selection)
    zero = [[0.0] * len(OPPONENT_DEVIATION_ACTION_IDS) for _ in selection]
    root_heuristic = [[item.root_heuristic_value] * len(OPPONENT_DEVIATION_ACTION_IDS) for item in selection]
    branch_heuristic = _branch_heuristic_predictions(selection)
    zero_mse = _mse(targets, zero)
    root_mse = _mse(targets, root_heuristic)
    branch_mse = _mse(targets, branch_heuristic)
    zero_contrast = _contrast_mse(targets, zero)
    metrics = []
    for mean_blend in PHASE6_BRANCH_HYBRID_MEAN_BLEND_GRID:
        for advantage_scale in PHASE6_BRANCH_HYBRID_ADVANTAGE_SCALE_GRID:
            predictions = _hybrid_predictions(
                source_model,
                selection,
                mean_blend=mean_blend,
                advantage_scale=advantage_scale,
            )
            mse = _mse(targets, predictions)
            contrast = _contrast_mse(targets, predictions)
            metrics.append(
                BranchHybridSelectionMetric(
                    mean_blend=mean_blend,
                    advantage_scale=advantage_scale,
                    selection_mse=mse,
                    selection_zero_mse=zero_mse,
                    selection_root_heuristic_mse=root_mse,
                    selection_branch_heuristic_mse=branch_mse,
                    selection_action_contrast_mse=contrast,
                    selection_zero_action_contrast_mse=zero_contrast,
                    selection_score=max(
                        mse / max(min(zero_mse, root_mse, branch_mse), 1e-12),
                        contrast
                        / max(
                            PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR * zero_contrast,
                            1e-12,
                        ),
                    ),
                )
            )
    selected = min(
        metrics,
        key=lambda item: (
            item.selection_score,
            item.selection_mse,
            item.selection_action_contrast_mse,
            item.mean_blend,
            item.advantage_scale,
        ),
    )
    model = BranchHybridModel(
        source_dataset_fingerprint=source_dataset.fingerprint,
        source_calibration_fingerprint=source_calibration.fingerprint,
        source_model=source_model,
        selected_mean_blend=selected.mean_blend,
        selected_advantage_scale=selected.advantage_scale,
        selection_root_ids=tuple(sorted(item.root_id for item in selection)),
        selection_metrics=tuple(metrics),
    )
    bound_predictions = _hybrid_predictions(
        source_model,
        bounds,
        mean_blend=model.selected_mean_blend,
        advantage_scale=model.selected_advantage_scale,
    )
    residuals = sorted(
        abs(prediction - action.empirical_player0_value)
        + PHASE6_EXPECTED_VALUE_MEAN_CI_Z * action.target_standard_error
        for example, group in zip(bounds, bound_predictions, strict=True)
        for action, prediction in zip(example.actions, group, strict=True)
    )
    rank = min(
        len(residuals),
        math.ceil((len(residuals) + 1) * PHASE6_BRANCH_VALUE_COVERAGE),
    )
    calibration = BranchHybridCalibration(
        source_dataset_fingerprint=source_dataset.fingerprint,
        source_model_fingerprint=model.fingerprint,
        source_phase6ai_calibration_fingerprint=source_calibration.fingerprint,
        calibration_root_ids=tuple(sorted(item.root_id for item in bounds)),
        conformal_radius=min(2.0, residuals[rank - 1]),
        calibration_point_metric=_point_metric(
            "phase6ai_calibration_bounds",
            bounds,
            bound_predictions,
        ),
    )
    if BranchHybridModel.from_dict(json.loads(model.to_json())) != model:
        raise RuntimeError("Phase 6AJ model roundtrip changed values")
    if BranchHybridCalibration.from_dict(json.loads(calibration.to_json())) != calibration:
        raise RuntimeError("Phase 6AJ calibration roundtrip changed values")
    return model, calibration


def _bootstrap_zero_contrast_lower_bound(
    targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> float:
    """Return the frozen one-sided 90% root-bootstrap lower bound."""

    model_errors = _contrast_root_errors(targets, predictions)
    zero_errors = _contrast_root_errors(
        targets,
        [[0.0] * len(OPPONENT_DEVIATION_ACTION_IDS) for _ in targets],
    )
    improvements = [
        baseline - model
        for baseline, model in zip(zero_errors, model_errors, strict=True)
    ]
    rng = random.Random(PHASE6_BRANCH_HYBRID_BOOTSTRAP_SEED)
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
class BranchHybridEvaluation:
    validation_dataset_fingerprint: str
    source_model_fingerprint: str
    source_calibration_fingerprint: str
    validation_root_ids: tuple[str, ...]
    validation_point_metric: BranchValuePointMetric
    slice_metrics: tuple[BranchValueSliceMetric, ...]
    maximum_target_standard_error: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    zero_contrast_bootstrap_lower_bound: float
    sampling_precision_gate_passed: bool
    point_mse_gate_passed: bool
    contrast_relative_gate_passed: bool
    contrast_bootstrap_gate_passed: bool
    interval_gate_passed: bool
    implementation_holdout_gate_passed: bool
    action_sensitive_point_model_validated: bool
    conditional_expected_mean_model_admitted: bool
    counterfactual_value_compatible: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "validation_dataset_fingerprint",
            "source_model_fingerprint",
            "source_calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if len(self.validation_root_ids) != 36 or len(set(self.validation_root_ids)) != 36:
            raise ValueError("Phase 6AJ validation root ledger is incomplete")
        if self.validation_point_metric.role != "fresh_validation_untouched":
            raise ValueError("Phase 6AJ validation point metric role is invalid")
        if not self.slice_metrics:
            raise ValueError("Phase 6AJ validation slices are empty")
        for name in (
            "maximum_target_standard_error",
            "validation_coverage",
            "validation_mean_width",
            "minimum_validation_slice_coverage",
            "zero_contrast_bootstrap_lower_bound",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"Phase 6AJ {name} is nonfinite")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("Phase 6AJ validation coverage is invalid")
        if not 0 <= self.minimum_validation_slice_coverage <= 1:
            raise ValueError("Phase 6AJ minimum slice coverage is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AJ validation width is invalid")

        metric = self.validation_point_metric
        expected_sampling = (
            self.maximum_target_standard_error
            <= PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
        )
        expected_point = metric.mse < min(
            metric.zero_mse,
            metric.root_heuristic_mse,
            metric.branch_heuristic_mse,
        )
        expected_contrast = metric.action_contrast_mse < (
            PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR
            * metric.zero_action_contrast_mse
        )
        expected_bootstrap = self.zero_contrast_bootstrap_lower_bound > 0
        expected_interval = (
            self.validation_coverage >= PHASE6_BRANCH_VALUE_COVERAGE
            and self.validation_mean_width
            <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
            and self.minimum_validation_slice_coverage
            >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
        )
        expected = (
            ("sampling_precision_gate_passed", expected_sampling),
            ("point_mse_gate_passed", expected_point),
            ("contrast_relative_gate_passed", expected_contrast),
            ("contrast_bootstrap_gate_passed", expected_bootstrap),
            ("interval_gate_passed", expected_interval),
        )
        for name, value in expected:
            if getattr(self, name) != value:
                raise ValueError(f"Phase 6AJ {name} is inconsistent")
        point_admitted = all(
            (
                expected_sampling,
                expected_point,
                expected_contrast,
                expected_bootstrap,
                self.implementation_holdout_gate_passed,
            )
        )
        if self.action_sensitive_point_model_validated != point_admitted:
            raise ValueError("Phase 6AJ point-model admission is inconsistent")
        if self.conditional_expected_mean_model_admitted != (
            point_admitted and expected_interval
        ):
            raise ValueError("Phase 6AJ conditional-mean admission is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AJ evaluation promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_HYBRID_SCHEMA_VERSION,
            "validation_dataset_fingerprint": self.validation_dataset_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_calibration_fingerprint": self.source_calibration_fingerprint,
            "validation_root_ids": list(self.validation_root_ids),
            "validation_point_metric": asdict(self.validation_point_metric),
            "slice_metrics": [asdict(item) for item in self.slice_metrics],
            "maximum_target_standard_error": self.maximum_target_standard_error,
            "validation_coverage": self.validation_coverage,
            "validation_mean_width": self.validation_mean_width,
            "minimum_validation_slice_coverage": self.minimum_validation_slice_coverage,
            "bootstrap": {
                "seed": PHASE6_BRANCH_HYBRID_BOOTSTRAP_SEED,
                "resamples": PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES,
                "zero_contrast_lower_bound": self.zero_contrast_bootstrap_lower_bound,
            },
            "gates": {
                "sampling_precision_gate_passed": self.sampling_precision_gate_passed,
                "point_mse_gate_passed": self.point_mse_gate_passed,
                "contrast_relative_gate_passed": self.contrast_relative_gate_passed,
                "contrast_bootstrap_gate_passed": self.contrast_bootstrap_gate_passed,
                "interval_gate_passed": self.interval_gate_passed,
                "implementation_holdout_gate_passed": self.implementation_holdout_gate_passed,
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchHybridEvaluation":
        bootstrap = raw["bootstrap"]
        if (
            int(bootstrap["seed"]) != PHASE6_BRANCH_HYBRID_BOOTSTRAP_SEED
            or int(bootstrap["resamples"]) != PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError("Phase 6AJ bootstrap protocol drifted")
        gates = raw["gates"]
        result = cls(
            validation_dataset_fingerprint=str(raw["validation_dataset_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            source_calibration_fingerprint=str(raw["source_calibration_fingerprint"]),
            validation_root_ids=tuple(str(value) for value in raw["validation_root_ids"]),
            validation_point_metric=BranchValuePointMetric(
                **dict(raw["validation_point_metric"])
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
            contrast_bootstrap_gate_passed=bool(gates["contrast_bootstrap_gate_passed"]),
            interval_gate_passed=bool(gates["interval_gate_passed"]),
            implementation_holdout_gate_passed=bool(
                gates["implementation_holdout_gate_passed"]
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
            raise ValueError("Phase 6AJ evaluation fingerprint mismatch")
        return result


def evaluate_branch_hybrid(
    dataset: BranchHybridValidationDataset,
    model: BranchHybridModel,
    calibration: BranchHybridCalibration,
) -> BranchHybridEvaluation:
    """Open the untouched panel once and apply the frozen admission gates."""

    if (
        dataset.source_branch_dataset_fingerprint != model.source_dataset_fingerprint
        or dataset.source_branch_model_fingerprint != model.source_model.fingerprint
        or dataset.source_branch_calibration_fingerprint
        != model.source_calibration_fingerprint
        or calibration.source_dataset_fingerprint != model.source_dataset_fingerprint
        or calibration.source_model_fingerprint != model.fingerprint
        or calibration.source_phase6ai_calibration_fingerprint
        != model.source_calibration_fingerprint
    ):
        raise ValueError("Phase 6AJ validation/model/calibration provenance mismatch")

    examples = build_branch_value_examples_from_roots(
        dataset.roots,
        dataset.candidate_ids,
        expected_count=36,
    )
    predictions = _hybrid_predictions(
        model.source_model,
        examples,
        mean_blend=model.selected_mean_blend,
        advantage_scale=model.selected_advantage_scale,
    )
    point = _point_metric("fresh_validation_untouched", examples, predictions)
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
            dataset.candidate_ids,
            lambda example, action: example.candidate_id_hypothesis_input,
        ),
        (
            "intent",
            dataset.intent_ids,
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
                raise RuntimeError(f"Phase 6AJ empty slice: {dimension}/{identifier}")
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
    sampling_gate = (
        maximum_se <= PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
    )
    point_gate = point.mse < min(
        point.zero_mse,
        point.root_heuristic_mse,
        point.branch_heuristic_mse,
    )
    contrast_gate = point.action_contrast_mse < (
        PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR
        * point.zero_action_contrast_mse
    )
    bootstrap_gate = bootstrap_lower > 0
    interval_gate = (
        coverage >= PHASE6_BRANCH_VALUE_COVERAGE
        and mean_width <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
        and minimum_slice >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
    )
    validation_ids = {item.root_id for item in dataset.roots}
    prior_ids = set(dataset.phase6ah_design_root_ids) | set(dataset.phase6ai_source_root_ids)
    implementation_gate = all(
        (
            not (validation_ids & prior_ids),
            not (validation_ids & set(model.selection_root_ids)),
            not (validation_ids & set(calibration.calibration_root_ids)),
            set(model.selection_root_ids).issubset(dataset.phase6ai_source_root_ids),
            set(calibration.calibration_root_ids).issubset(
                dataset.phase6ai_source_root_ids
            ),
            {item.policy_variant_id_supervision_only for item in examples}
            == {"action_first"},
            {item.split for item in examples} == {"validation"},
            not dataset.validation_targets_used_for_selection,
            not dataset.validation_targets_used_for_calibration,
        )
    )
    point_admitted = all(
        (
            sampling_gate,
            point_gate,
            contrast_gate,
            bootstrap_gate,
            implementation_gate,
        )
    )
    result = BranchHybridEvaluation(
        validation_dataset_fingerprint=dataset.fingerprint,
        source_model_fingerprint=model.fingerprint,
        source_calibration_fingerprint=calibration.fingerprint,
        validation_root_ids=tuple(sorted(validation_ids)),
        validation_point_metric=point,
        slice_metrics=tuple(slice_metrics),
        maximum_target_standard_error=maximum_se,
        validation_coverage=coverage,
        validation_mean_width=mean_width,
        minimum_validation_slice_coverage=minimum_slice,
        zero_contrast_bootstrap_lower_bound=bootstrap_lower,
        sampling_precision_gate_passed=sampling_gate,
        point_mse_gate_passed=point_gate,
        contrast_relative_gate_passed=contrast_gate,
        contrast_bootstrap_gate_passed=bootstrap_gate,
        interval_gate_passed=interval_gate,
        implementation_holdout_gate_passed=implementation_gate,
        action_sensitive_point_model_validated=point_admitted,
        conditional_expected_mean_model_admitted=(point_admitted and interval_gate),
    )
    if BranchHybridEvaluation.from_dict(json.loads(result.to_json())) != result:
        raise RuntimeError("Phase 6AJ evaluation roundtrip changed values")
    return result


@dataclass(frozen=True, slots=True)
class Phase6BranchHybridEvidence:
    status: str
    validation_dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    evaluation_fingerprint: str
    root_count: int
    branch_observation_count: int
    surface_count: int
    action_target_count: int
    terminal_rollout_count: int
    live_selection_count: int
    native_transition_count: int
    validation_mse: float
    validation_zero_mse: float
    validation_root_heuristic_mse: float
    validation_branch_heuristic_mse: float
    validation_action_contrast_mse: float
    validation_zero_action_contrast_mse: float
    zero_contrast_bootstrap_lower_bound: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    maximum_target_standard_error: float
    sampling_precision_gate_passed: bool
    point_mse_gate_passed: bool
    contrast_relative_gate_passed: bool
    contrast_bootstrap_gate_passed: bool
    interval_gate_passed: bool
    implementation_holdout_gate_passed: bool
    action_sensitive_point_model_validated: bool
    conditional_expected_mean_model_admitted: bool
    counterfactual_value_compatible: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "validation_dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
            "evaluation_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if (
            self.root_count != 36
            or self.branch_observation_count != 360
            or self.surface_count != 72
            or self.action_target_count != 180
            or self.terminal_rollout_count != 2_880
            or self.live_selection_count <= 0
            or self.native_transition_count <= 0
        ):
            raise ValueError("Phase 6AJ evidence accounting is incomplete")
        expected_status = (
            "fresh_semiparametric_conditional_expected_mean_admitted_downstream_fail_closed"
            if self.conditional_expected_mean_model_admitted
            else (
                "fresh_semiparametric_action_sensitive_point_signal_validated_intervals_rejected_downstream_fail_closed"
                if self.action_sensitive_point_model_validated
                else "fresh_semiparametric_branch_hybrid_rejected_downstream_fail_closed"
            )
        )
        if self.status != expected_status:
            raise ValueError("Phase 6AJ evidence status is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AJ evidence promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {"schema_version": PHASE6_BRANCH_HYBRID_SCHEMA_VERSION, **asdict(self)}
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6BranchHybridEvidence":
        values = dict(raw)
        values.pop("schema_version", None)
        fingerprint = values.pop("fingerprint", None)
        result = cls(**values)
        if fingerprint not in (None, result.fingerprint):
            raise ValueError("Phase 6AJ evidence fingerprint mismatch")
        return result

    def to_markdown(self) -> str:
        return f"""# Phase 6AJ semiparametric branch-value hybrid

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Fresh complete official matches / roots | {self.root_count} / {self.root_count} |
| Immediate native branch observations | {self.branch_observation_count} |
| Supplied-allocation surfaces / root-action targets | {self.surface_count} / {self.action_target_count} |
| Genuine terminal native continuations | {self.terminal_rollout_count} |
| Genuine live selections / native transitions | {self.live_selection_count} / {self.native_transition_count} |
| Validation hybrid / zero / root heuristic / branch heuristic MSE | `{self.validation_mse:.6f}` / `{self.validation_zero_mse:.6f}` / `{self.validation_root_heuristic_mse:.6f}` / `{self.validation_branch_heuristic_mse:.6f}` |
| Validation contrast hybrid / zero MSE | `{self.validation_action_contrast_mse:.6f}` / `{self.validation_zero_action_contrast_mse:.6f}` |
| 90% bootstrap zero-minus-hybrid contrast lower bound | `{self.zero_contrast_bootstrap_lower_bound:.6f}` |
| Validation mean-CI coverage / width / worst slice | `{self.validation_coverage:.6f}` / `{self.validation_mean_width:.6f}` / `{self.minimum_validation_slice_coverage:.6f}` |
| Maximum target SE | `{self.maximum_target_standard_error:.6f}` |
| Sampling / point MSE / relative contrast / bootstrap / interval / holdout | **{str(self.sampling_precision_gate_passed).lower()} / {str(self.point_mse_gate_passed).lower()} / {str(self.contrast_relative_gate_passed).lower()} / {str(self.contrast_bootstrap_gate_passed).lower()} / {str(self.interval_gate_passed).lower()} / {str(self.implementation_holdout_gate_passed).lower()}** |
| Action-sensitive point / conditional expected mean | **{str(self.action_sensitive_point_model_validated).lower()} / {str(self.conditional_expected_mean_model_admitted).lower()}** |
| Counterfactual / equilibrium / safe resolver / exploitation / control | **false / false / false / false / false** |

Complete matches, legal live selections, retained snapshot replay, immediate
branches, and terminal descendants are genuine official-engine execution. The
allocation law, five-plan boundary, tactical executor, continuation policy,
public branch heuristic, neural root mean, hybrid, bootstrap, and intervals are
abstractions. No Kaggle submission or playing-strength claim is enabled.
"""


@dataclass(frozen=True, slots=True)
class Phase6BranchHybridArtifacts:
    validation_dataset: BranchHybridValidationDataset
    model: BranchHybridModel
    calibration: BranchHybridCalibration
    evaluation: BranchHybridEvaluation
    evidence: Phase6BranchHybridEvidence


def run_phase6_semiparametric_branch_hybrid(
    *,
    project_root: str | Path | None = None,
    validation_dataset: BranchHybridValidationDataset | None = None,
    model: BranchHybridModel | None = None,
    calibration: BranchHybridCalibration | None = None,
    existing_roots: Sequence[BranchConditionedRoot] = (),
    root_observer: Callable[[BranchConditionedRoot, int, int], None] | None = None,
) -> Phase6BranchHybridArtifacts:
    source_dataset, source_model, source_calibration = _load_phase6ai_artifacts(
        project_root
    )
    if (model is None) != (calibration is None):
        raise ValueError("Phase 6AJ model and calibration must be supplied together")
    if model is None or calibration is None:
        model, calibration = prepare_branch_hybrid(
            source_dataset,
            source_model,
            source_calibration,
        )
    dataset = validation_dataset or collect_branch_hybrid_validation_dataset(
        project_root=project_root,
        existing_roots=existing_roots,
        root_observer=root_observer,
    )
    evaluation = evaluate_branch_hybrid(dataset, model, calibration)
    metric = evaluation.validation_point_metric
    admitted = evaluation.conditional_expected_mean_model_admitted
    point_admitted = evaluation.action_sensitive_point_model_validated
    evidence = Phase6BranchHybridEvidence(
        status=(
            "fresh_semiparametric_conditional_expected_mean_admitted_downstream_fail_closed"
            if admitted
            else (
                "fresh_semiparametric_action_sensitive_point_signal_validated_intervals_rejected_downstream_fail_closed"
                if point_admitted
                else "fresh_semiparametric_branch_hybrid_rejected_downstream_fail_closed"
            )
        ),
        validation_dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        evaluation_fingerprint=evaluation.fingerprint,
        root_count=len(dataset.roots),
        branch_observation_count=sum(
            len(item.branch_observations) for item in dataset.roots
        ),
        surface_count=sum(len(item.value_root.surfaces) for item in dataset.roots),
        action_target_count=len(dataset.roots) * len(OPPONENT_DEVIATION_ACTION_IDS),
        terminal_rollout_count=dataset.terminal_rollout_count,
        live_selection_count=dataset.live_selection_count,
        native_transition_count=dataset.native_transition_count,
        validation_mse=metric.mse,
        validation_zero_mse=metric.zero_mse,
        validation_root_heuristic_mse=metric.root_heuristic_mse,
        validation_branch_heuristic_mse=metric.branch_heuristic_mse,
        validation_action_contrast_mse=metric.action_contrast_mse,
        validation_zero_action_contrast_mse=metric.zero_action_contrast_mse,
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
        contrast_bootstrap_gate_passed=evaluation.contrast_bootstrap_gate_passed,
        interval_gate_passed=evaluation.interval_gate_passed,
        implementation_holdout_gate_passed=(
            evaluation.implementation_holdout_gate_passed
        ),
        action_sensitive_point_model_validated=point_admitted,
        conditional_expected_mean_model_admitted=admitted,
    )
    if Phase6BranchHybridEvidence.from_dict(json.loads(evidence.to_json())) != evidence:
        raise RuntimeError("Phase 6AJ evidence roundtrip changed values")
    return Phase6BranchHybridArtifacts(
        validation_dataset=dataset,
        model=model,
        calibration=calibration,
        evaluation=evaluation,
        evidence=evidence,
    )


def _write_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "phase6_semiparametric_branch_hybrid"
        ),
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-validation", action="store_true")
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "branch_hybrid_model.json"
    calibration_path = target / "branch_hybrid_calibration.json"
    validation_path = target / "branch_hybrid_validation_dataset.json"

    if model_path.is_file() or calibration_path.is_file():
        if not (model_path.is_file() and calibration_path.is_file()):
            raise RuntimeError("Phase 6AJ frozen model/calibration pair is incomplete")
        model = BranchHybridModel.from_dict(
            json.loads(model_path.read_text(encoding="utf-8"))
        )
        calibration = BranchHybridCalibration.from_dict(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
    else:
        source_dataset, source_model, source_calibration = _load_phase6ai_artifacts(
            None
        )
        model, calibration = prepare_branch_hybrid(
            source_dataset,
            source_model,
            source_calibration,
        )
        _write_artifact(model_path, model.to_json() + "\n")
        _write_artifact(calibration_path, calibration.to_json() + "\n")

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "phase6aj_model_and_bounds_frozen_before_validation",
                    "model_fingerprint": model.fingerprint,
                    "calibration_fingerprint": calibration.fingerprint,
                    "selected_mean_blend": model.selected_mean_blend,
                    "selected_advantage_scale": model.selected_advantage_scale,
                    "conformal_radius": calibration.conformal_radius,
                    "artifact_directory": str(target),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.reuse_validation:
        if not validation_path.is_file():
            raise FileNotFoundError(
                "--reuse-validation requires branch_hybrid_validation_dataset.json"
            )
        validation = BranchHybridValidationDataset.from_dict(
            json.loads(validation_path.read_text(encoding="utf-8"))
        )
    else:
        validation = collect_branch_hybrid_validation_dataset()
        _write_artifact(validation_path, validation.to_json() + "\n")
    artifacts = run_phase6_semiparametric_branch_hybrid(
        validation_dataset=validation,
        model=model,
        calibration=calibration,
    )
    _write_artifact(
        target / "branch_hybrid_evaluation.json",
        artifacts.evaluation.to_json() + "\n",
    )
    _write_artifact(
        target / "phase6aj_evidence.json",
        artifacts.evidence.to_json() + "\n",
    )
    _write_artifact(
        target / "phase6aj_evidence.md",
        artifacts.evidence.to_markdown(),
    )
    print(artifacts.evidence.to_json())


if __name__ == "__main__":
    main()
