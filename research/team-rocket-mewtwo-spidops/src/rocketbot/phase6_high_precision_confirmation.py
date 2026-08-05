"""Phase 6AL high-precision confirmation of the frozen Phase 6AK mean."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

from .beliefs import OPPONENT_INTENTS
from .phase6_branch_conditioned_values import (
    PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES,
    PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR,
    PHASE6_BRANCH_VALUE_COVERAGE,
    BranchConditionedRoot,
    BranchValuePointMetric,
    BranchValueSliceMetric,
    _branch_heuristic_predictions,
    _build_branch_conditioned_root,
    _contrast_root_errors,
    _fingerprint,
    _interval_result,
    _mse,
    _point_metric,
    _root_heuristic_predictions,
    _targets,
    _validate_digest,
    build_branch_value_examples_from_roots,
)
from .phase6_expected_value_surfaces import (
    PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH,
    PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MEAN_CI_Z,
    PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE,
    _capture_complete_match_late_root,
    _load_expected_value_sources,
)
from .phase6_expanded_root_mean import (
    ExpandedRootMeanModel,
    ExpandedRootMeanProvider,
    ExpandedRootMeanPanel,
    _load_phase6ak_design,
    restore_expanded_root_mean_model,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_opponent_deviation_values import OPPONENT_DEVIATION_ACTION_IDS
from .phase6_semiparametric_branch_hybrid import (
    BranchHybridModel,
    _hybrid_predictions,
)


PHASE6_HIGH_PRECISION_SCHEMA_VERSION = 1
PHASE6_HIGH_PRECISION_CALIBRATION_SEED = 6_056
PHASE6_HIGH_PRECISION_VALIDATION_SEED = 6_057
PHASE6_HIGH_PRECISION_BOOTSTRAP_SEED = 6_058
PHASE6_HIGH_PRECISION_BATCH_SEED_OFFSET = 1_000_000
PHASE6_HIGH_PRECISION_REPLAY_BATCHES = 2
PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION = 32
PHASE6_HIGH_PRECISION_CALIBRATION_COVERAGE = 0.92
PHASE6_HIGH_PRECISION_CALIBRATION_QUANTILE_RANK = 167
PHASE6_HIGH_PRECISION_MAXIMUM_TARGET_STANDARD_ERROR = 0.20
PHASE6_HIGH_PRECISION_ROOTS_PER_PANEL = 36
PHASE6_HIGH_PRECISION_SOURCE_MODEL_FINGERPRINT = (
    "5f3f73025251fde4d5e1dc155f604aa98856951030567e8baa13d2f1e999350e"
)
PHASE6_HIGH_PRECISION_ROLES = ("calibration", "validation")


def _project_root(project_root: str | Path | None) -> Path:
    return Path(project_root or Path(__file__).resolve().parents[2]).resolve()


def _role_seed(role: str) -> int:
    if role == "calibration":
        return PHASE6_HIGH_PRECISION_CALIBRATION_SEED
    if role == "validation":
        return PHASE6_HIGH_PRECISION_VALIDATION_SEED
    raise ValueError("Phase 6AL panel role is unsupported")


def _role_namespace(role: str) -> str:
    if role not in PHASE6_HIGH_PRECISION_ROLES:
        raise ValueError("Phase 6AL panel role is unsupported")
    return f"phase6al-{role}"


def _batch_seeds(role: str) -> tuple[int, ...]:
    base = _role_seed(role)
    return tuple(
        base + batch * PHASE6_HIGH_PRECISION_BATCH_SEED_OFFSET
        for batch in range(PHASE6_HIGH_PRECISION_REPLAY_BATCHES)
    )


def _load_source(
    project_root: str | Path | None,
) -> tuple[
    ExpandedRootMeanModel,
    ExpandedRootMeanPanel,
    ExpandedRootMeanPanel,
    BranchHybridModel,
]:
    root = _project_root(project_root)
    ak_directory = root / "artifacts" / "phase6_expanded_root_mean"
    model = ExpandedRootMeanModel.from_dict(
        json.loads(
            (ak_directory / "expanded_root_mean_model.json").read_text(
                encoding="utf-8"
            )
        )
    )
    calibration_panel = ExpandedRootMeanPanel.from_dict(
        json.loads(
            (ak_directory / "calibration_panel.json").read_text(encoding="utf-8")
        )
    )
    validation_panel = ExpandedRootMeanPanel.from_dict(
        json.loads(
            (ak_directory / "validation_panel.json").read_text(encoding="utf-8")
        )
    )
    _, _, phase6aj_model = _load_phase6ak_design(root)
    if model.fingerprint != PHASE6_HIGH_PRECISION_SOURCE_MODEL_FINGERPRINT:
        raise ValueError("Phase 6AL frozen Phase 6AK model fingerprint changed")
    if (
        calibration_panel.role != "calibration"
        or validation_panel.role != "validation"
        or calibration_panel.source_model_fingerprint != model.fingerprint
        or validation_panel.source_model_fingerprint != model.fingerprint
    ):
        raise ValueError("Phase 6AL source panels/model are not bound")
    return model, calibration_panel, validation_panel, phase6aj_model


@dataclass(frozen=True, slots=True)
class HighPrecisionModelReference:
    source_model_fingerprint: str
    source_phase6ak_calibration_panel_fingerprint: str
    source_phase6ak_validation_panel_fingerprint: str
    development_root_ids: tuple[str, ...]
    source_feature_count: int = 208
    source_model_retrained: bool = False
    phase6ak_targets_declared_development: bool = True
    intent_labels_as_model_input: bool = False
    official_engine_executed: bool = False
    counterfactual_value_provider: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_model_fingerprint",
            "source_phase6ak_calibration_panel_fingerprint",
            "source_phase6ak_validation_panel_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.source_model_fingerprint != PHASE6_HIGH_PRECISION_SOURCE_MODEL_FINGERPRINT:
            raise ValueError("Phase 6AL source-model reference drifted")
        if len(self.development_root_ids) != 288 or len(
            set(self.development_root_ids)
        ) != 288:
            raise ValueError("Phase 6AL development ledger is incomplete")
        if self.source_feature_count != 208:
            raise ValueError("Phase 6AL source feature width drifted")
        if any(
            (
                self.source_model_retrained,
                not self.phase6ak_targets_declared_development,
                self.intent_labels_as_model_input,
                self.official_engine_executed,
                self.counterfactual_value_provider,
            )
        ):
            raise ValueError("Phase 6AL reference promoted an unsupported claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_HIGH_PRECISION_SCHEMA_VERSION,
            "source_model_fingerprint": self.source_model_fingerprint,
            "source_phase6ak_calibration_panel_fingerprint": (
                self.source_phase6ak_calibration_panel_fingerprint
            ),
            "source_phase6ak_validation_panel_fingerprint": (
                self.source_phase6ak_validation_panel_fingerprint
            ),
            "development_root_ids": list(self.development_root_ids),
            "source_feature_count": self.source_feature_count,
            "claims": {
                "source_model_retrained": self.source_model_retrained,
                "phase6ak_targets_declared_development": (
                    self.phase6ak_targets_declared_development
                ),
                "intent_labels_as_model_input": self.intent_labels_as_model_input,
                "official_engine_executed": self.official_engine_executed,
                "counterfactual_value_provider": self.counterfactual_value_provider,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HighPrecisionModelReference":
        claims = raw["claims"]
        result = cls(
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            source_phase6ak_calibration_panel_fingerprint=str(
                raw["source_phase6ak_calibration_panel_fingerprint"]
            ),
            source_phase6ak_validation_panel_fingerprint=str(
                raw["source_phase6ak_validation_panel_fingerprint"]
            ),
            development_root_ids=tuple(str(value) for value in raw["development_root_ids"]),
            source_feature_count=int(raw["source_feature_count"]),
            source_model_retrained=bool(claims["source_model_retrained"]),
            phase6ak_targets_declared_development=bool(
                claims["phase6ak_targets_declared_development"]
            ),
            intent_labels_as_model_input=bool(claims["intent_labels_as_model_input"]),
            official_engine_executed=bool(claims["official_engine_executed"]),
            counterfactual_value_provider=bool(claims["counterfactual_value_provider"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AL model-reference fingerprint mismatch")
        return result


def prepare_high_precision_model_reference(
    project_root: str | Path | None = None,
) -> HighPrecisionModelReference:
    model, calibration_panel, validation_panel, _ = _load_source(project_root)
    development_ids = tuple(
        sorted(
            set(model.design_root_ids)
            | {item.root_id for item in calibration_panel.roots}
            | {item.root_id for item in validation_panel.roots}
        )
    )
    result = HighPrecisionModelReference(
        source_model_fingerprint=model.fingerprint,
        source_phase6ak_calibration_panel_fingerprint=calibration_panel.fingerprint,
        source_phase6ak_validation_panel_fingerprint=validation_panel.fingerprint,
        development_root_ids=development_ids,
    )
    if HighPrecisionModelReference.from_dict(json.loads(result.to_json())) != result:
        raise RuntimeError("Phase 6AL model reference roundtrip changed values")
    return result


@dataclass(frozen=True, slots=True)
class HighPrecisionRoot:
    role: str
    batch_seeds: tuple[int, ...]
    batches: tuple[BranchConditionedRoot, ...]
    one_complete_official_match_executed: bool = True
    retained_snapshot_replayed_per_batch: bool = True
    supplied_hidden_allocation_batches: bool = True
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if self.role not in PHASE6_HIGH_PRECISION_ROLES:
            raise ValueError("Phase 6AL root role is unsupported")
        if self.batch_seeds != _batch_seeds(self.role):
            raise ValueError("Phase 6AL replay-batch seeds drifted")
        if len(self.batches) != PHASE6_HIGH_PRECISION_REPLAY_BATCHES:
            raise ValueError("Phase 6AL replay-batch count drifted")
        first = self.batches[0].value_root
        same_fields = (
            "match_id",
            "candidate_id_hypothesis_input",
            "intent_id_supervision_only",
            "policy_variant_id_supervision_only",
            "public_state_fingerprint",
            "public_history_fingerprint",
            "private_observation_fingerprint",
            "search_snapshot_fingerprint",
            "public_features",
            "private_features",
            "history_features",
            "blueprint_probabilities",
            "prefix_chain_fingerprint",
            "prefix_decision_count",
            "selected_root_turn",
            "selected_root_prizes",
            "match_selection_count",
        )
        for batch in self.batches:
            value = batch.value_root
            if any(getattr(value, name) != getattr(first, name) for name in same_fields):
                raise ValueError("Phase 6AL replay batches do not share one retained root")
            if (
                value.policy_variant_id_supervision_only != "action_first"
                or value.split != "validation"
                or value.replica != 0
            ):
                raise ValueError("Phase 6AL root escaped the action-first protocol")
        prefix = f"{_role_namespace(self.role)}-action_first-"
        if not first.match_id.startswith(prefix):
            raise ValueError("Phase 6AL root belongs to the wrong namespace")
        if len({item.root_id for item in self.batches}) != len(self.batches):
            raise ValueError("Phase 6AL replay batches are not distinct")
        completions = [
            surface.completion_fingerprint
            for batch in self.batches
            for surface in batch.value_root.surfaces
        ]
        if len(completions) != 4 or len(set(completions)) != 4:
            raise ValueError("Phase 6AL supplied allocations are not distinct")
        if self.branch_observation_count != 20 or self.terminal_rollout_count != 160:
            raise ValueError("Phase 6AL root replay accounting drifted")
        if any(
            (
                not self.one_complete_official_match_executed,
                not self.retained_snapshot_replayed_per_batch,
                not self.supplied_hidden_allocation_batches,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6AL root provenance/claim boundary failed")
        if _raw_private_key_count(self.to_dict(include_fingerprint=False)) != 0:
            raise ValueError("Phase 6AL root persisted a forbidden raw private key")

    @property
    def root_id(self) -> str:
        first = self.batches[0].value_root
        return _fingerprint(
            {
                "role": self.role,
                "match_id": first.match_id,
                "public_state_fingerprint": first.public_state_fingerprint,
                "public_history_fingerprint": first.public_history_fingerprint,
                "private_observation_fingerprint": first.private_observation_fingerprint,
                "batch_root_ids": [item.root_id for item in self.batches],
            }
        )

    @property
    def match_id(self) -> str:
        return self.batches[0].value_root.match_id

    @property
    def candidate_id(self) -> str:
        return self.batches[0].value_root.candidate_id_hypothesis_input

    @property
    def intent_id(self) -> str:
        return self.batches[0].value_root.intent_id_supervision_only

    @property
    def branch_observation_count(self) -> int:
        return sum(len(item.branch_observations) for item in self.batches)

    @property
    def terminal_rollout_count(self) -> int:
        return sum(item.value_root.terminal_rollout_count for item in self.batches)

    @property
    def live_selection_count(self) -> int:
        return self.batches[0].value_root.match_selection_count

    @property
    def native_transition_count(self) -> int:
        return sum(
            surface.native_transition_count
            for batch in self.batches
            for surface in batch.value_root.surfaces
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_HIGH_PRECISION_SCHEMA_VERSION,
            "root_id": self.root_id,
            "role": self.role,
            "batch_seeds": list(self.batch_seeds),
            "batches": [item.to_dict() for item in self.batches],
            "claims": {
                "one_complete_official_match_executed": (
                    self.one_complete_official_match_executed
                ),
                "retained_snapshot_replayed_per_batch": (
                    self.retained_snapshot_replayed_per_batch
                ),
                "supplied_hidden_allocation_batches": (
                    self.supplied_hidden_allocation_batches
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = _fingerprint(result)
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HighPrecisionRoot":
        claims = raw["claims"]
        result = cls(
            role=str(raw["role"]),
            batch_seeds=tuple(int(value) for value in raw["batch_seeds"]),
            batches=tuple(BranchConditionedRoot.from_dict(item) for item in raw["batches"]),
            one_complete_official_match_executed=bool(
                claims["one_complete_official_match_executed"]
            ),
            retained_snapshot_replayed_per_batch=bool(
                claims["retained_snapshot_replayed_per_batch"]
            ),
            supplied_hidden_allocation_batches=bool(
                claims["supplied_hidden_allocation_batches"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        if raw.get("root_id") not in (None, result.root_id):
            raise ValueError("Phase 6AL root ID mismatch")
        if raw.get("fingerprint") not in (None, result.to_dict()["fingerprint"]):
            raise ValueError("Phase 6AL root fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class HighPrecisionActionTarget:
    action_id: str
    branch_features: tuple[tuple[str, float], ...]
    branch_heuristic_value: float
    terminal_returns: tuple[float, ...]
    empirical_player0_value: float
    target_standard_error: float

    def __post_init__(self) -> None:
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AL action target is unsupported")
        names = tuple(name for name, _ in self.branch_features)
        if not names or len(names) != len(set(names)):
            raise ValueError("Phase 6AL branch feature schema is invalid")
        if len(self.terminal_returns) != PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION:
            raise ValueError("Phase 6AL action target return count drifted")
        if any(value not in (-1.0, 0.0, 1.0) for value in self.terminal_returns):
            raise ValueError("Phase 6AL terminal utility is invalid")
        mean = statistics.mean(self.terminal_returns)
        standard_error = statistics.stdev(self.terminal_returns) / math.sqrt(
            len(self.terminal_returns)
        )
        if not math.isclose(self.empirical_player0_value, mean, abs_tol=1e-12):
            raise ValueError("Phase 6AL empirical action mean is inconsistent")
        if not math.isclose(self.target_standard_error, standard_error, abs_tol=1e-12):
            raise ValueError("Phase 6AL target standard error is inconsistent")
        if any(not math.isfinite(value) for _, value in self.branch_features):
            raise ValueError("Phase 6AL branch features are nonfinite")
        if (
            not -1 <= self.branch_heuristic_value <= 1
            or not -1 <= self.empirical_player0_value <= 1
            or not math.isfinite(self.target_standard_error)
            or self.target_standard_error < 0
        ):
            raise ValueError("Phase 6AL branch heuristic is invalid")


@dataclass(frozen=True, slots=True)
class HighPrecisionRootExample:
    root_id: str
    match_id: str
    split: str
    calibration_partition: str
    candidate_id_hypothesis_input: str
    intent_id_supervision_only: str
    policy_variant_id_supervision_only: str
    root_features: tuple[tuple[str, float], ...]
    root_heuristic_value: float
    actions: tuple[HighPrecisionActionTarget, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.root_id, "root_id")
        if self.split != "validation" or self.calibration_partition not in (
            "calibration",
            "validation",
        ):
            raise ValueError("Phase 6AL example split/role is invalid")
        if self.policy_variant_id_supervision_only != "action_first":
            raise ValueError("Phase 6AL example policy variant drifted")
        if len(self.root_features) != 208:
            raise ValueError("Phase 6AL root feature width drifted")
        names = tuple(name for name, _ in self.root_features)
        if len(set(names)) != len(names) or any(
            forbidden in name
            for name in names
            for forbidden in ("intent_id", "policy_variant", "supervision_only")
        ):
            raise ValueError("Phase 6AL root feature schema leaked an audit label")
        if any(not math.isfinite(value) for _, value in self.root_features):
            raise ValueError("Phase 6AL root features are nonfinite")
        if tuple(item.action_id for item in self.actions) != OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AL example action support is incomplete")


def build_high_precision_examples(
    roots: Sequence[HighPrecisionRoot],
    candidate_ids: Sequence[str],
) -> tuple[HighPrecisionRootExample, ...]:
    examples = []
    for root in roots:
        batches = build_branch_value_examples_from_roots(
            root.batches,
            candidate_ids,
            expected_count=PHASE6_HIGH_PRECISION_REPLAY_BATCHES,
        )
        first = batches[0]
        for other in batches[1:]:
            if (
                other.root_features != first.root_features
                or other.root_heuristic_value != first.root_heuristic_value
                or tuple(item.action_id for item in other.actions)
                != tuple(item.action_id for item in first.actions)
            ):
                raise ValueError("Phase 6AL batch example roots are not aligned")
        targets = []
        for action_index, action_id in enumerate(OPPONENT_DEVIATION_ACTION_IDS):
            batch_actions = tuple(item.actions[action_index] for item in batches)
            feature_names = tuple(name for name, _ in batch_actions[0].branch_features)
            if any(
                tuple(name for name, _ in item.branch_features) != feature_names
                or item.action_id != action_id
                for item in batch_actions
            ):
                raise ValueError("Phase 6AL batch action features are not aligned")
            branch_features = tuple(
                (
                    name,
                    statistics.mean(dict(item.branch_features)[name] for item in batch_actions),
                )
                for name in feature_names
            )
            returns = tuple(
                value
                for item in batch_actions
                for value in item.terminal_returns
            )
            targets.append(
                HighPrecisionActionTarget(
                    action_id=action_id,
                    branch_features=branch_features,
                    branch_heuristic_value=statistics.mean(
                        item.branch_heuristic_value for item in batch_actions
                    ),
                    terminal_returns=returns,
                    empirical_player0_value=statistics.mean(returns),
                    target_standard_error=(
                        statistics.stdev(returns) / math.sqrt(len(returns))
                    ),
                )
            )
        examples.append(
            HighPrecisionRootExample(
                root_id=root.root_id,
                match_id=root.match_id,
                split="validation",
                calibration_partition=root.role,
                candidate_id_hypothesis_input=root.candidate_id,
                intent_id_supervision_only=root.intent_id,
                policy_variant_id_supervision_only="action_first",
                root_features=first.root_features,
                root_heuristic_value=first.root_heuristic_value,
                actions=tuple(targets),
            )
        )
    return tuple(examples)


@dataclass(frozen=True, slots=True)
class HighPrecisionPanel:
    role: str
    source_reference_fingerprint: str
    source_model_fingerprint: str
    development_root_ids: tuple[str, ...]
    seed: int
    match_namespace: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    intent_ids: tuple[str, ...]
    roots: tuple[HighPrecisionRoot, ...]
    prior_roots_reused: int = 0
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if self.role not in PHASE6_HIGH_PRECISION_ROLES:
            raise ValueError("Phase 6AL panel role is unsupported")
        for name in ("source_reference_fingerprint", "source_model_fingerprint"):
            _validate_digest(getattr(self, name), name)
        if self.source_model_fingerprint != PHASE6_HIGH_PRECISION_SOURCE_MODEL_FINGERPRINT:
            raise ValueError("Phase 6AL panel source model drifted")
        if self.seed != _role_seed(self.role) or self.match_namespace != _role_namespace(
            self.role
        ):
            raise ValueError("Phase 6AL panel seed/namespace drifted")
        if len(self.development_root_ids) != 288 or len(
            set(self.development_root_ids)
        ) != 288:
            raise ValueError("Phase 6AL panel development ledger is incomplete")
        if (
            len(self.candidate_ids) != 6
            or len(set(self.candidate_ids)) != 6
            or tuple(name for name, _ in self.candidate_deck_fingerprints)
            != self.candidate_ids
            or self.intent_ids != tuple(item.value for item in OPPONENT_INTENTS)
        ):
            raise ValueError("Phase 6AL panel candidate/intent catalog drifted")
        if len(self.roots) != PHASE6_HIGH_PRECISION_ROOTS_PER_PANEL or len(
            {item.root_id for item in self.roots}
        ) != PHASE6_HIGH_PRECISION_ROOTS_PER_PANEL:
            raise ValueError("Phase 6AL panel root ledger is incomplete")
        expected = {
            (candidate, intent)
            for candidate in self.candidate_ids
            for intent in self.intent_ids
        }
        actual = {(item.candidate_id, item.intent_id) for item in self.roots}
        if actual != expected or len(actual) != len(self.roots):
            raise ValueError("Phase 6AL candidate/intent panel cross is incomplete")
        development = set(self.development_root_ids)
        batch_ids = {
            batch.root_id for item in self.roots for batch in item.batches
        }
        if development & batch_ids:
            raise ValueError("Phase 6AL panel reused a development root")
        if any(item.role != self.role for item in self.roots):
            raise ValueError("Phase 6AL panel mixed roles")
        if any((self.prior_roots_reused != 0, self.raw_private_card_ids_persisted != 0)):
            raise ValueError("Phase 6AL panel promoted an unsupported claim")
        if _raw_private_key_count(self.to_dict(include_fingerprint=False)) != 0:
            raise ValueError("Phase 6AL panel persisted a forbidden raw private key")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def terminal_rollout_count(self) -> int:
        return sum(item.terminal_rollout_count for item in self.roots)

    @property
    def branch_observation_count(self) -> int:
        return sum(item.branch_observation_count for item in self.roots)

    @property
    def live_selection_count(self) -> int:
        return sum(item.live_selection_count for item in self.roots)

    @property
    def native_transition_count(self) -> int:
        return sum(item.native_transition_count for item in self.roots)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_HIGH_PRECISION_SCHEMA_VERSION,
            "role": self.role,
            "source_reference_fingerprint": self.source_reference_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "development_root_ids": list(self.development_root_ids),
            "seed": self.seed,
            "match_namespace": self.match_namespace,
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": dict(self.candidate_deck_fingerprints),
            "intent_ids": list(self.intent_ids),
            "roots": [item.to_dict() for item in self.roots],
            "claims": {
                "prior_roots_reused": self.prior_roots_reused,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HighPrecisionPanel":
        claims = raw["claims"]
        result = cls(
            role=str(raw["role"]),
            source_reference_fingerprint=str(raw["source_reference_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            development_root_ids=tuple(str(value) for value in raw["development_root_ids"]),
            seed=int(raw["seed"]),
            match_namespace=str(raw["match_namespace"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            candidate_deck_fingerprints=tuple(
                (candidate, str(raw["candidate_deck_fingerprints"][candidate]))
                for candidate in raw["candidate_ids"]
            ),
            intent_ids=tuple(str(value) for value in raw["intent_ids"]),
            roots=tuple(HighPrecisionRoot.from_dict(item) for item in raw["roots"]),
            prior_roots_reused=int(claims["prior_roots_reused"]),
            raw_private_card_ids_persisted=int(claims["raw_private_card_ids_persisted"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AL panel fingerprint mismatch")
        return result


def collect_high_precision_panel(
    role: str,
    *,
    project_root: str | Path | None = None,
    reference: HighPrecisionModelReference | None = None,
    existing_roots: Sequence[HighPrecisionRoot] = (),
    root_observer: Callable[[HighPrecisionRoot, int, int], None] | None = None,
) -> HighPrecisionPanel:
    if role not in PHASE6_HIGH_PRECISION_ROLES:
        raise ValueError("Phase 6AL panel role is unsupported")
    frozen_reference = reference or prepare_high_precision_model_reference(project_root)
    model, ak_calibration, ak_validation, _ = _load_source(project_root)
    expected_reference = prepare_high_precision_model_reference(project_root)
    if frozen_reference != expected_reference:
        raise ValueError("Phase 6AL panel/reference provenance mismatch")
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
    by_key = {(item.candidate_id, item.intent_id): item for item in materialized}
    if len(by_key) != len(materialized) or not set(by_key).issubset(expected_keys):
        raise ValueError("Phase 6AL resume roots are duplicate/outside the grid")
    if any(item.role != role for item in materialized):
        raise ValueError("Phase 6AL resume root belongs to the wrong role")
    candidates = {item.archetype: item for item in sources.candidates}
    intents = {item.value: item for item in OPPONENT_INTENTS}
    prefix = f"{_role_namespace(role)}-action_first-"
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
                batches = tuple(
                    _build_branch_conditioned_root(
                        deepcopy(capture),
                        sources=sources,
                        candidate=candidates[candidate_id],
                        intent=intents[intent_id],
                        policy_variant="action_first",
                        split="validation",
                        replica=0,
                        match_id=match_id,
                        seed=batch_seed,
                        maximum_rollout_transitions=(
                            PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS
                        ),
                        maximum_tactical_transitions=(
                            PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS
                        ),
                        experiment_label="Phase 6AL",
                    )
                    for batch_seed in _batch_seeds(role)
                )
                root = HighPrecisionRoot(
                    role=role,
                    batch_seeds=_batch_seeds(role),
                    batches=batches,
                )
                by_key[key] = root
                if root_observer is not None:
                    root_observer(root, len(by_key), len(expected_keys))
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Phase 6AL exhausted attempts for {role}/{candidate_id}/{intent_id}"
            ) from last_error
    roots = tuple(by_key[key] for key in expected_keys)
    panel = HighPrecisionPanel(
        role=role,
        source_reference_fingerprint=frozen_reference.fingerprint,
        source_model_fingerprint=model.fingerprint,
        development_root_ids=frozen_reference.development_root_ids,
        seed=_role_seed(role),
        match_namespace=_role_namespace(role),
        candidate_ids=ak_calibration.candidate_ids,
        candidate_deck_fingerprints=ak_calibration.candidate_deck_fingerprints,
        intent_ids=ak_validation.intent_ids,
        roots=roots,
    )
    if HighPrecisionPanel.from_dict(json.loads(panel.to_json())) != panel:
        raise RuntimeError("Phase 6AL panel roundtrip changed values")
    return panel


def _conformal_radius(
    examples: Sequence[HighPrecisionRootExample],
    predictions: Sequence[Sequence[float]],
) -> float:
    residuals = sorted(
        abs(prediction - action.empirical_player0_value)
        + PHASE6_EXPECTED_VALUE_MEAN_CI_Z * action.target_standard_error
        for example, group in zip(examples, predictions, strict=True)
        for action, prediction in zip(example.actions, group, strict=True)
    )
    if len(residuals) != 180:
        raise ValueError("Phase 6AL calibration score count drifted")
    rank = min(
        len(residuals),
        math.ceil(
            (len(residuals) + 1) * PHASE6_HIGH_PRECISION_CALIBRATION_COVERAGE
        ),
    )
    if rank != PHASE6_HIGH_PRECISION_CALIBRATION_QUANTILE_RANK:
        raise RuntimeError("Phase 6AL finite-sample calibration rank drifted")
    return min(2.0, residuals[rank - 1])


@dataclass(frozen=True, slots=True)
class HighPrecisionCalibration:
    source_reference_fingerprint: str
    source_model_fingerprint: str
    calibration_panel_fingerprint: str
    calibration_root_ids: tuple[str, ...]
    conformal_radius: float
    calibration_point_metric: BranchValuePointMetric
    calibration_coverage_target: float = PHASE6_HIGH_PRECISION_CALIBRATION_COVERAGE
    calibration_quantile_rank: int = PHASE6_HIGH_PRECISION_CALIBRATION_QUANTILE_RANK
    validation_roots_collected_before_freeze: int = 0
    validation_targets_used_for_calibration: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_reference_fingerprint",
            "source_model_fingerprint",
            "calibration_panel_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if len(self.calibration_root_ids) != 36 or len(
            set(self.calibration_root_ids)
        ) != 36:
            raise ValueError("Phase 6AL calibration root ledger is incomplete")
        if not 0 <= self.conformal_radius <= 2:
            raise ValueError("Phase 6AL conformal radius is invalid")
        if (
            self.calibration_coverage_target
            != PHASE6_HIGH_PRECISION_CALIBRATION_COVERAGE
            or self.calibration_quantile_rank
            != PHASE6_HIGH_PRECISION_CALIBRATION_QUANTILE_RANK
        ):
            raise ValueError("Phase 6AL calibration protocol drifted")
        if self.calibration_point_metric.role != "phase6al_high_precision_calibration":
            raise ValueError("Phase 6AL calibration point role drifted")
        if any(
            (
                self.validation_roots_collected_before_freeze != 0,
                self.validation_targets_used_for_calibration,
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
            )
        ):
            raise ValueError("Phase 6AL calibration promoted future/downstream state")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_HIGH_PRECISION_SCHEMA_VERSION,
            "source_reference_fingerprint": self.source_reference_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "calibration_panel_fingerprint": self.calibration_panel_fingerprint,
            "calibration_root_ids": list(self.calibration_root_ids),
            "conformal_radius": self.conformal_radius,
            "calibration_point_metric": asdict(self.calibration_point_metric),
            "calibration_protocol": {
                "coverage_target": self.calibration_coverage_target,
                "quantile_rank": self.calibration_quantile_rank,
            },
            "claims": {
                "validation_roots_collected_before_freeze": (
                    self.validation_roots_collected_before_freeze
                ),
                "validation_targets_used_for_calibration": (
                    self.validation_targets_used_for_calibration
                ),
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "safe_resolver_enabled": self.safe_resolver_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HighPrecisionCalibration":
        protocol = raw["calibration_protocol"]
        claims = raw["claims"]
        result = cls(
            source_reference_fingerprint=str(raw["source_reference_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            calibration_panel_fingerprint=str(raw["calibration_panel_fingerprint"]),
            calibration_root_ids=tuple(str(value) for value in raw["calibration_root_ids"]),
            conformal_radius=float(raw["conformal_radius"]),
            calibration_point_metric=BranchValuePointMetric(
                **dict(raw["calibration_point_metric"])
            ),
            calibration_coverage_target=float(protocol["coverage_target"]),
            calibration_quantile_rank=int(protocol["quantile_rank"]),
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
            raise ValueError("Phase 6AL calibration fingerprint mismatch")
        return result


def calibrate_high_precision_confirmation(
    panel: HighPrecisionPanel,
    reference: HighPrecisionModelReference,
    model: ExpandedRootMeanModel,
) -> HighPrecisionCalibration:
    if panel.role != "calibration":
        raise ValueError("Phase 6AL bounds require the calibration panel")
    if (
        panel.source_reference_fingerprint != reference.fingerprint
        or panel.source_model_fingerprint != model.fingerprint
        or reference.source_model_fingerprint != model.fingerprint
        or panel.development_root_ids != reference.development_root_ids
    ):
        raise ValueError("Phase 6AL calibration provenance mismatch")
    examples = build_high_precision_examples(panel.roots, panel.candidate_ids)
    predictions = restore_expanded_root_mean_model(model).predict_examples(examples)
    result = HighPrecisionCalibration(
        source_reference_fingerprint=reference.fingerprint,
        source_model_fingerprint=model.fingerprint,
        calibration_panel_fingerprint=panel.fingerprint,
        calibration_root_ids=tuple(sorted(item.root_id for item in examples)),
        conformal_radius=_conformal_radius(examples, predictions),
        calibration_point_metric=_point_metric(
            "phase6al_high_precision_calibration",
            examples,
            predictions,
        ),
    )
    if HighPrecisionCalibration.from_dict(json.loads(result.to_json())) != result:
        raise RuntimeError("Phase 6AL calibration roundtrip changed values")
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
    rng = random.Random(PHASE6_HIGH_PRECISION_BOOTSTRAP_SEED)
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
class HighPrecisionEvaluation:
    validation_panel_fingerprint: str
    source_reference_fingerprint: str
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
            "source_reference_fingerprint",
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
            raise ValueError("Phase 6AL evaluation root ledgers are invalid")
        if self.validation_point_metric.role != "phase6al_high_precision_validation":
            raise ValueError("Phase 6AL validation point role drifted")
        if not self.slice_metrics:
            raise ValueError("Phase 6AL validation slices are empty")
        for name in (
            "phase6aj_baseline_validation_mse",
            "maximum_target_standard_error",
            "validation_coverage",
            "validation_mean_width",
            "minimum_validation_slice_coverage",
            "zero_contrast_bootstrap_lower_bound",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"Phase 6AL {name} is nonfinite")
        if self.phase6aj_baseline_validation_mse < 0:
            raise ValueError("Phase 6AL baseline MSE is negative")
        if not 0 <= self.maximum_target_standard_error <= 1:
            raise ValueError("Phase 6AL maximum target SE is invalid")
        if not 0 <= self.validation_coverage <= 1 or not 0 <= (
            self.minimum_validation_slice_coverage
        ) <= 1:
            raise ValueError("Phase 6AL coverage is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AL interval width is invalid")
        metric = self.validation_point_metric
        expected_sampling = (
            self.maximum_target_standard_error
            <= PHASE6_HIGH_PRECISION_MAXIMUM_TARGET_STANDARD_ERROR
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
                raise ValueError(f"Phase 6AL {name} is inconsistent")
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
            raise ValueError("Phase 6AL point admission is inconsistent")
        if self.conditional_expected_mean_model_admitted != (
            point_admitted and expected_interval
        ):
            raise ValueError("Phase 6AL conditional-mean admission is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AL evaluation promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_HIGH_PRECISION_SCHEMA_VERSION,
            "validation_panel_fingerprint": self.validation_panel_fingerprint,
            "source_reference_fingerprint": self.source_reference_fingerprint,
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
                "seed": PHASE6_HIGH_PRECISION_BOOTSTRAP_SEED,
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
                "action_sensitive_point_model_validated": (
                    self.action_sensitive_point_model_validated
                ),
                "conditional_expected_mean_model_admitted": (
                    self.conditional_expected_mean_model_admitted
                ),
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "equilibrium_convergence_validated": (
                    self.equilibrium_convergence_validated
                ),
                "safe_resolver_enabled": self.safe_resolver_enabled,
                "opponent_exploitation_enabled": self.opponent_exploitation_enabled,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HighPrecisionEvaluation":
        bootstrap = raw["bootstrap"]
        if (
            int(bootstrap["seed"]) != PHASE6_HIGH_PRECISION_BOOTSTRAP_SEED
            or int(bootstrap["resamples"]) != PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES
        ):
            raise ValueError("Phase 6AL bootstrap protocol drifted")
        gates = raw["gates"]
        result = cls(
            validation_panel_fingerprint=str(raw["validation_panel_fingerprint"]),
            source_reference_fingerprint=str(raw["source_reference_fingerprint"]),
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
            collection_holdout_gate_passed=bool(gates["collection_holdout_gate_passed"]),
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
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AL evaluation fingerprint mismatch")
        return result


def evaluate_high_precision_confirmation(
    calibration_panel: HighPrecisionPanel,
    validation_panel: HighPrecisionPanel,
    reference: HighPrecisionModelReference,
    model: ExpandedRootMeanModel,
    calibration: HighPrecisionCalibration,
    phase6aj_baseline_model: BranchHybridModel,
) -> HighPrecisionEvaluation:
    if calibration_panel.role != "calibration" or validation_panel.role != "validation":
        raise ValueError("Phase 6AL evaluation panel roles are invalid")
    if (
        calibration.source_reference_fingerprint != reference.fingerprint
        or calibration.source_model_fingerprint != model.fingerprint
        or calibration.calibration_panel_fingerprint != calibration_panel.fingerprint
        or validation_panel.source_reference_fingerprint != reference.fingerprint
        or validation_panel.source_model_fingerprint != model.fingerprint
        or model.source_phase6aj_model_fingerprint
        != phase6aj_baseline_model.fingerprint
    ):
        raise ValueError("Phase 6AL evaluation provenance mismatch")
    examples = build_high_precision_examples(
        validation_panel.roots,
        validation_panel.candidate_ids,
    )
    predictions = restore_expanded_root_mean_model(model).predict_examples(examples)
    baseline_predictions = _hybrid_predictions(
        phase6aj_baseline_model.source_model,
        examples,
        mean_blend=phase6aj_baseline_model.selected_mean_blend,
        advantage_scale=phase6aj_baseline_model.selected_advantage_scale,
    )
    point = _point_metric("phase6al_high_precision_validation", examples, predictions)
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
                raise RuntimeError(f"Phase 6AL empty slice: {dimension}/{identifier}")
            slice_metrics.append(
                BranchValueSliceMetric(
                    dimension=dimension,
                    slice_id=identifier,
                    validation_row_count=len(indices),
                    validation_mse=statistics.mean(
                        (flat[index][2] - flat[index][1].empirical_player0_value) ** 2
                        for index in indices
                    ),
                    validation_coverage=(
                        sum(covered[index] for index in indices) / len(indices)
                    ),
                    validation_mean_width=statistics.mean(widths[index] for index in indices),
                )
            )
    maximum_se = max(
        action.target_standard_error for example in examples for action in example.actions
    )
    coverage = sum(covered) / len(covered)
    mean_width = statistics.mean(widths)
    minimum_slice = min(item.validation_coverage for item in slice_metrics)
    bootstrap_lower = _bootstrap_zero_contrast_lower_bound(
        _targets(examples),
        predictions,
    )
    sampling_gate = maximum_se <= PHASE6_HIGH_PRECISION_MAXIMUM_TARGET_STANDARD_ERROR
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
    calibration_batch_ids = {
        batch.root_id for item in calibration_panel.roots for batch in item.batches
    }
    validation_batch_ids = {
        batch.root_id for item in validation_panel.roots for batch in item.batches
    }
    development_ids = set(reference.development_root_ids)
    collection_gate = all(
        (
            not (calibration_ids & validation_ids),
            not (calibration_batch_ids & validation_batch_ids),
            not (calibration_batch_ids & development_ids),
            not (validation_batch_ids & development_ids),
            set(calibration.calibration_root_ids) == calibration_ids,
            calibration_panel.development_root_ids == reference.development_root_ids,
            validation_panel.development_root_ids == reference.development_root_ids,
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
            collection_gate,
        )
    )
    result = HighPrecisionEvaluation(
        validation_panel_fingerprint=validation_panel.fingerprint,
        source_reference_fingerprint=reference.fingerprint,
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
        collection_holdout_gate_passed=collection_gate,
        action_sensitive_point_model_validated=point_admitted,
        conditional_expected_mean_model_admitted=(point_admitted and interval_gate),
    )
    if HighPrecisionEvaluation.from_dict(json.loads(result.to_json())) != result:
        raise RuntimeError("Phase 6AL evaluation roundtrip changed values")
    return result


@dataclass(frozen=True, slots=True)
class Phase6HighPrecisionEvidence:
    status: str
    reference_fingerprint: str
    model_fingerprint: str
    calibration_panel_fingerprint: str
    calibration_fingerprint: str
    validation_panel_fingerprint: str
    evaluation_fingerprint: str
    official_match_count: int
    replay_batch_count: int
    branch_observation_count: int
    action_target_count: int
    terminal_rollout_count: int
    live_selection_count: int
    native_transition_count: int
    returns_per_action: int
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
            "reference_fingerprint",
            "model_fingerprint",
            "calibration_panel_fingerprint",
            "calibration_fingerprint",
            "validation_panel_fingerprint",
            "evaluation_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if (
            self.official_match_count != 72
            or self.replay_batch_count != 144
            or self.branch_observation_count != 1_440
            or self.action_target_count != 360
            or self.terminal_rollout_count != 11_520
            or self.live_selection_count <= 0
            or self.native_transition_count <= 0
            or self.returns_per_action != PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION
        ):
            raise ValueError("Phase 6AL evidence accounting is incomplete")
        expected_status = (
            "high_precision_conditional_expected_mean_admitted_downstream_fail_closed"
            if self.conditional_expected_mean_model_admitted
            else (
                "high_precision_action_sensitive_point_validated_intervals_rejected_downstream_fail_closed"
                if self.action_sensitive_point_model_validated
                else "high_precision_confirmation_rejected_downstream_fail_closed"
            )
        )
        if self.status != expected_status:
            raise ValueError("Phase 6AL evidence status is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AL evidence promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {"schema_version": PHASE6_HIGH_PRECISION_SCHEMA_VERSION, **asdict(self)}
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6HighPrecisionEvidence":
        values = dict(raw)
        values.pop("schema_version", None)
        fingerprint = values.pop("fingerprint", None)
        result = cls(**values)
        if fingerprint not in (None, result.fingerprint):
            raise ValueError("Phase 6AL evidence fingerprint mismatch")
        return result

    def to_markdown(self) -> str:
        return f"""# Phase 6AL high-precision conditional-value confirmation

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Fresh official matches / replay batches | {self.official_match_count} / {self.replay_batch_count} |
| Immediate native branch observations | {self.branch_observation_count} |
| Root-action targets / returns per action | {self.action_target_count} / {self.returns_per_action} |
| Genuine terminal native continuations | {self.terminal_rollout_count} |
| Genuine live selections / native transitions | {self.live_selection_count} / {self.native_transition_count} |
| Frozen 92% conformal radius | `{self.conformal_radius:.6f}` |
| Validation model / Phase 6AJ / zero / root / branch MSE | `{self.validation_mse:.6f}` / `{self.phase6aj_baseline_validation_mse:.6f}` / `{self.validation_zero_mse:.6f}` / `{self.validation_root_heuristic_mse:.6f}` / `{self.validation_branch_heuristic_mse:.6f}` |
| Validation contrast model / zero / branch MSE | `{self.validation_action_contrast_mse:.6f}` / `{self.validation_zero_action_contrast_mse:.6f}` / `{self.validation_branch_action_contrast_mse:.6f}` |
| 90% bootstrap zero-minus-model contrast lower bound | `{self.zero_contrast_bootstrap_lower_bound:.6f}` |
| Validation mean-CI coverage / width / worst slice | `{self.validation_coverage:.6f}` / `{self.validation_mean_width:.6f}` / `{self.minimum_validation_slice_coverage:.6f}` |
| Maximum 32-return target SE | `{self.maximum_target_standard_error:.6f}` |
| Sampling / point / relative contrast / retention / bootstrap / interval / collection holdout | **{str(self.sampling_precision_gate_passed).lower()} / {str(self.point_mse_gate_passed).lower()} / {str(self.contrast_relative_gate_passed).lower()} / {str(self.contrast_retention_gate_passed).lower()} / {str(self.contrast_bootstrap_gate_passed).lower()} / {str(self.interval_gate_passed).lower()} / {str(self.collection_holdout_gate_passed).lower()}** |
| Action-sensitive point / conditional expected mean | **{str(self.action_sensitive_point_model_validated).lower()} / {str(self.conditional_expected_mean_model_admitted).lower()}** |
| Counterfactual / equilibrium / safe resolver / exploitation / control | **false / false / false / false / false** |

Complete matches, legal selections, retained snapshots, immediate branches,
and terminal descendants are genuine official-engine execution. The supplied
range, five-plan boundary, tactical executor, continuation policy, fixed branch
heuristic, frozen mean, bootstrap, and intervals are abstractions. No Kaggle
submission or playing-strength claim is enabled.
"""


@dataclass(frozen=True, slots=True)
class Phase6HighPrecisionArtifacts:
    reference: HighPrecisionModelReference
    calibration_panel: HighPrecisionPanel
    calibration: HighPrecisionCalibration
    validation_panel: HighPrecisionPanel
    evaluation: HighPrecisionEvaluation
    evidence: Phase6HighPrecisionEvidence


def run_phase6_high_precision_confirmation(
    *,
    project_root: str | Path | None = None,
    reference: HighPrecisionModelReference,
    calibration_panel: HighPrecisionPanel,
    calibration: HighPrecisionCalibration,
    validation_panel: HighPrecisionPanel,
) -> Phase6HighPrecisionArtifacts:
    model, _, _, phase6aj_model = _load_source(project_root)
    evaluation = evaluate_high_precision_confirmation(
        calibration_panel,
        validation_panel,
        reference,
        model,
        calibration,
        phase6aj_model,
    )
    point = evaluation.validation_point_metric
    admitted = evaluation.conditional_expected_mean_model_admitted
    point_admitted = evaluation.action_sensitive_point_model_validated
    roots = calibration_panel.roots + validation_panel.roots
    evidence = Phase6HighPrecisionEvidence(
        status=(
            "high_precision_conditional_expected_mean_admitted_downstream_fail_closed"
            if admitted
            else (
                "high_precision_action_sensitive_point_validated_intervals_rejected_downstream_fail_closed"
                if point_admitted
                else "high_precision_confirmation_rejected_downstream_fail_closed"
            )
        ),
        reference_fingerprint=reference.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_panel_fingerprint=calibration_panel.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        validation_panel_fingerprint=validation_panel.fingerprint,
        evaluation_fingerprint=evaluation.fingerprint,
        official_match_count=len(roots),
        replay_batch_count=sum(len(item.batches) for item in roots),
        branch_observation_count=sum(item.branch_observation_count for item in roots),
        action_target_count=len(roots) * len(OPPONENT_DEVIATION_ACTION_IDS),
        terminal_rollout_count=sum(item.terminal_rollout_count for item in roots),
        live_selection_count=sum(item.live_selection_count for item in roots),
        native_transition_count=sum(item.native_transition_count for item in roots),
        returns_per_action=PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION,
        conformal_radius=calibration.conformal_radius,
        validation_mse=point.mse,
        validation_zero_mse=point.zero_mse,
        validation_root_heuristic_mse=point.root_heuristic_mse,
        validation_branch_heuristic_mse=point.branch_heuristic_mse,
        phase6aj_baseline_validation_mse=evaluation.phase6aj_baseline_validation_mse,
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
        collection_holdout_gate_passed=evaluation.collection_holdout_gate_passed,
        action_sensitive_point_model_validated=point_admitted,
        conditional_expected_mean_model_admitted=admitted,
    )
    if Phase6HighPrecisionEvidence.from_dict(json.loads(evidence.to_json())) != evidence:
        raise RuntimeError("Phase 6AL evidence roundtrip changed values")
    return Phase6HighPrecisionArtifacts(
        reference=reference,
        calibration_panel=calibration_panel,
        calibration=calibration,
        validation_panel=validation_panel,
        evaluation=evaluation,
        evidence=evidence,
    )


def _write_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _load_root_shards(directory: Path) -> tuple[HighPrecisionRoot, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        HighPrecisionRoot.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    )


def main() -> None:
    import argparse

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare-reference", "freeze-calibration", "evaluate"),
        required=True,
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=project_root / "artifacts" / "phase6_high_precision_confirmation",
    )
    parser.add_argument(
        "--calibration-shard-directory",
        type=Path,
        default=project_root / "work" / "phase6al_calibration_shards",
    )
    parser.add_argument(
        "--validation-shard-directory",
        type=Path,
        default=project_root / "work" / "phase6al_validation_shards",
    )
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    reference_path = target / "high_precision_model_reference.json"
    calibration_panel_path = target / "calibration_panel.json"
    calibration_path = target / "high_precision_calibration.json"
    validation_panel_path = target / "validation_panel.json"

    if args.stage == "prepare-reference":
        existing_shards = len(_load_root_shards(args.calibration_shard_directory)) + len(
            _load_root_shards(args.validation_shard_directory)
        )
        if existing_shards:
            raise RuntimeError("Phase 6AL reference must be frozen before collection")
        if reference_path.is_file():
            reference = HighPrecisionModelReference.from_dict(
                json.loads(reference_path.read_text(encoding="utf-8"))
            )
        else:
            reference = prepare_high_precision_model_reference(project_root)
            _write_artifact(reference_path, reference.to_json() + "\n")
        print(
            json.dumps(
                {
                    "status": "phase6al_reference_frozen_before_collection",
                    "reference_fingerprint": reference.fingerprint,
                    "source_model_fingerprint": reference.source_model_fingerprint,
                    "development_root_count": len(reference.development_root_ids),
                    "returns_per_action": PHASE6_HIGH_PRECISION_RETURNS_PER_ACTION,
                    "calibration_coverage_target": (
                        PHASE6_HIGH_PRECISION_CALIBRATION_COVERAGE
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not reference_path.is_file():
        raise FileNotFoundError("Phase 6AL frozen model reference is missing")
    reference = HighPrecisionModelReference.from_dict(
        json.loads(reference_path.read_text(encoding="utf-8"))
    )
    model, _, _, _ = _load_source(project_root)

    if args.stage == "freeze-calibration":
        validation_shards = _load_root_shards(args.validation_shard_directory)
        if validation_shards or validation_panel_path.exists():
            raise RuntimeError("Phase 6AL validation opened before bounds freeze")
        calibration_shards = _load_root_shards(args.calibration_shard_directory)
        if len(calibration_shards) != PHASE6_HIGH_PRECISION_ROOTS_PER_PANEL:
            raise RuntimeError("Phase 6AL requires 36 calibration root bundles")
        panel = collect_high_precision_panel(
            "calibration",
            project_root=project_root,
            reference=reference,
            existing_roots=calibration_shards,
        )
        calibration = calibrate_high_precision_confirmation(
            panel,
            reference,
            model,
        )
        _write_artifact(calibration_panel_path, panel.to_json() + "\n")
        _write_artifact(calibration_path, calibration.to_json() + "\n")
        print(
            json.dumps(
                {
                    "status": "phase6al_bounds_frozen_before_validation",
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
        raise FileNotFoundError("Phase 6AL frozen calibration artifacts are missing")
    calibration_panel = HighPrecisionPanel.from_dict(
        json.loads(calibration_panel_path.read_text(encoding="utf-8"))
    )
    calibration = HighPrecisionCalibration.from_dict(
        json.loads(calibration_path.read_text(encoding="utf-8"))
    )
    validation_shards = _load_root_shards(args.validation_shard_directory)
    if len(validation_shards) != PHASE6_HIGH_PRECISION_ROOTS_PER_PANEL:
        raise RuntimeError("Phase 6AL requires 36 validation root bundles")
    validation_panel = collect_high_precision_panel(
        "validation",
        project_root=project_root,
        reference=reference,
        existing_roots=validation_shards,
    )
    artifacts = run_phase6_high_precision_confirmation(
        project_root=project_root,
        reference=reference,
        calibration_panel=calibration_panel,
        calibration=calibration,
        validation_panel=validation_panel,
    )
    _write_artifact(validation_panel_path, validation_panel.to_json() + "\n")
    _write_artifact(
        target / "high_precision_evaluation.json",
        artifacts.evaluation.to_json() + "\n",
    )
    _write_artifact(target / "phase6al_evidence.json", artifacts.evidence.to_json() + "\n")
    _write_artifact(target / "phase6al_evidence.md", artifacts.evidence.to_markdown())
    print(artifacts.evidence.to_json())


if __name__ == "__main__":
    main()
