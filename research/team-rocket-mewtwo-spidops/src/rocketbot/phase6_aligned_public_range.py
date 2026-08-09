"""Phase 6AA split-safe public-only candidate-range diagnostic.

The source is the frozen Phase 6Z collection: one later public root for each of
six candidate decks in each of train, calibration, and validation.  This module
fits one normalized public-feature prototype per candidate on train roots,
selects temperature and uniform shrinkage on calibration roots, and evaluates
the untouched validation roots.

The provider emits the project's :class:`OpponentRange` interface, but only at
the deck-archetype level.  It is not a hidden-hand/resource posterior, uses too
few independent roots for official calibration, and has no production-validated
observer-orientation adapter.  Counterfactual and safety gates are therefore
hard false regardless of diagnostic accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .game_theory import OpponentRange, RangeHypothesis
from .native_blueprint_features import NATIVE_PUBLIC_FEATURE_NAMES
from .phase6_aligned_counterfactual_values import (
    ALIGNED_VALUE_SPLITS,
    PublicRootAlignedValueDataset,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_ALIGNED_PUBLIC_RANGE_SCHEMA_VERSION = 1
PHASE6_ALIGNED_PUBLIC_RANGE_SEED = 6_033
PUBLIC_RANGE_TEMPERATURE_GRID = (
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1_000.0,
)
PUBLIC_RANGE_UNIFORM_MIX_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
PUBLIC_RANGE_MINIMUM_CALIBRATION_ROOTS = 60
PUBLIC_RANGE_MINIMUM_VALIDATION_ROOTS = 60


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
class AlignedPublicRangeExample:
    example_id: str
    source_root_id: str
    split: str
    target_candidate_id_supervision_only: str
    public_state_fingerprint: str
    public_history_fingerprint: str
    source_public_belief_fingerprint: str
    public_features: tuple[tuple[str, float], ...]
    target_candidate_identifier_in_features: bool = False
    private_features_present: bool = False

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "source_root_id",
            "public_state_fingerprint",
            "public_history_fingerprint",
            "source_public_belief_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("aligned public-range example split is invalid")
        if not self.target_candidate_id_supervision_only:
            raise ValueError("aligned public-range example lacks a supervision label")
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("aligned public-range feature schema mismatch")
        if self.target_candidate_identifier_in_features or self.private_features_present:
            raise ValueError("aligned public-range example contains forbidden truth/private input")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_root_id": self.source_root_id,
            "split": self.split,
            "target_candidate_id_supervision_only": (
                self.target_candidate_id_supervision_only
            ),
            "public_state_fingerprint": self.public_state_fingerprint,
            "public_history_fingerprint": self.public_history_fingerprint,
            "source_public_belief_fingerprint": (
                self.source_public_belief_fingerprint
            ),
            "public_features": [[name, value] for name, value in self.public_features],
            "claims": {
                "target_candidate_identifier_in_features": (
                    self.target_candidate_identifier_in_features
                ),
                "private_features_present": self.private_features_present,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignedPublicRangeExample":
        claims = raw["claims"]
        return cls(
            example_id=str(raw["example_id"]),
            source_root_id=str(raw["source_root_id"]),
            split=str(raw["split"]),
            target_candidate_id_supervision_only=str(
                raw["target_candidate_id_supervision_only"]
            ),
            public_state_fingerprint=str(raw["public_state_fingerprint"]),
            public_history_fingerprint=str(raw["public_history_fingerprint"]),
            source_public_belief_fingerprint=str(
                raw["source_public_belief_fingerprint"]
            ),
            public_features=tuple(
                (str(name), float(value)) for name, value in raw["public_features"]
            ),
            target_candidate_identifier_in_features=bool(
                claims["target_candidate_identifier_in_features"]
            ),
            private_features_present=bool(claims["private_features_present"]),
        )


@dataclass(frozen=True, slots=True)
class AlignedPublicRangeDataset:
    source_phase6z_dataset_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    examples: tuple[AlignedPublicRangeExample, ...]
    root_split_disjoint: bool
    source_live_roots_official_engine_executed: bool = True
    transformation_official_engine_executed: bool = False
    target_candidate_identifier_in_features: bool = False
    private_features_present: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_phase6z_dataset_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("aligned public-range candidate support is invalid")
        expected = {
            (split, candidate)
            for split in ALIGNED_VALUE_SPLITS
            for candidate in self.candidate_ids
        }
        observed = {
            (item.split, item.target_candidate_id_supervision_only)
            for item in self.examples
        }
        if observed != expected or len(self.examples) != len(expected):
            raise ValueError("aligned public-range example support is incomplete")
        if len({item.example_id for item in self.examples}) != len(self.examples):
            raise ValueError("aligned public-range example IDs are duplicated")
        if len({item.source_root_id for item in self.examples}) != len(self.examples):
            raise ValueError("aligned public-range roots are duplicated")
        split_roots = {
            split: {
                item.source_root_id for item in self.examples if item.split == split
            }
            for split in ALIGNED_VALUE_SPLITS
        }
        expected_disjoint = all(
            split_roots[left].isdisjoint(split_roots[right])
            for index, left in enumerate(ALIGNED_VALUE_SPLITS)
            for right in ALIGNED_VALUE_SPLITS[index + 1 :]
        )
        if self.root_split_disjoint != expected_disjoint or not self.root_split_disjoint:
            raise ValueError("aligned public-range root split is not disjoint")
        if not self.source_live_roots_official_engine_executed:
            raise ValueError("aligned public-range source provenance changed")
        if any(
            (
                self.transformation_official_engine_executed,
                self.target_candidate_identifier_in_features,
                self.private_features_present,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("aligned public-range dataset promoted an invalid claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_PUBLIC_RANGE_SCHEMA_VERSION,
            "source_phase6z_dataset_fingerprint": (
                self.source_phase6z_dataset_fingerprint
            ),
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "accounting": {
                "example_count": len(self.examples),
                "roots_per_split": {
                    split: sum(item.split == split for item in self.examples)
                    for split in ALIGNED_VALUE_SPLITS
                },
            },
            "examples": [item.to_dict() for item in self.examples],
            "gates": {"root_split_disjoint": self.root_split_disjoint},
            "claims": {
                "source_live_roots_official_engine_executed": (
                    self.source_live_roots_official_engine_executed
                ),
                "transformation_official_engine_executed": (
                    self.transformation_official_engine_executed
                ),
                "public_features_only": True,
                "target_candidate_identifier_in_features": (
                    self.target_candidate_identifier_in_features
                ),
                "private_features_present": self.private_features_present,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignedPublicRangeDataset":
        accounting = raw["accounting"]
        gates = raw["gates"]
        claims = raw["claims"]
        if not bool(claims["public_features_only"]):
            raise ValueError("aligned public-range public-only claim changed")
        result = cls(
            source_phase6z_dataset_fingerprint=str(
                raw["source_phase6z_dataset_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            examples=tuple(
                AlignedPublicRangeExample.from_dict(value)
                for value in raw["examples"]
            ),
            root_split_disjoint=bool(gates["root_split_disjoint"]),
            source_live_roots_official_engine_executed=bool(
                claims["source_live_roots_official_engine_executed"]
            ),
            transformation_official_engine_executed=bool(
                claims["transformation_official_engine_executed"]
            ),
            target_candidate_identifier_in_features=bool(
                claims["target_candidate_identifier_in_features"]
            ),
            private_features_present=bool(claims["private_features_present"]),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected_accounting = {
            "example_count": len(result.examples),
            "roots_per_split": {
                split: sum(item.split == split for item in result.examples)
                for split in ALIGNED_VALUE_SPLITS
            },
        }
        if accounting != expected_accounting:
            raise ValueError("aligned public-range dataset accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("aligned public-range dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class PublicRangeSelectionMetric:
    temperature: float
    uniform_mix: float
    calibration_nll: float
    calibration_brier: float
    calibration_top1_accuracy: float

    def __post_init__(self) -> None:
        if self.temperature not in PUBLIC_RANGE_TEMPERATURE_GRID:
            raise ValueError("public-range selection temperature is unsupported")
        if self.uniform_mix not in PUBLIC_RANGE_UNIFORM_MIX_GRID:
            raise ValueError("public-range selection mixture is unsupported")
        for name in ("calibration_nll", "calibration_brier"):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} is invalid")
        if not 0 <= self.calibration_top1_accuracy <= 1:
            raise ValueError("public-range calibration accuracy is invalid")


@dataclass(frozen=True, slots=True)
class PrototypePublicRangeModel:
    source_dataset_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    normalizer: NamedFeatureNormalizer
    prototypes: tuple[tuple[str, tuple[float, ...]], ...]
    temperature_grid: tuple[float, ...]
    uniform_mix_grid: tuple[float, ...]
    selection_metrics: tuple[PublicRangeSelectionMetric, ...]
    selected_temperature: float
    selected_uniform_mix: float
    train_root_ids: tuple[str, ...]
    calibration_root_ids: tuple[str, ...]
    train_rows_fingerprint: str
    public_features_only: bool = True
    selected_truth_identifier_input: bool = False
    archetype_range_only: bool = True
    hidden_hand_or_resource_posterior: bool = False
    official_public_range_calibrated: bool = False
    official_engine_executed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_dataset_fingerprint",
            "deck_fingerprint",
            "train_rows_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("prototype public-range candidate support is invalid")
        if self.normalizer.names != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("prototype public-range feature schema mismatch")
        if self.normalizer.fit_rows_fingerprint != self.train_rows_fingerprint:
            raise ValueError("prototype public-range normalizer provenance mismatch")
        if tuple(candidate for candidate, _ in self.prototypes) != self.candidate_ids:
            raise ValueError("prototype public-range prototype order/support mismatch")
        if any(len(values) != len(NATIVE_PUBLIC_FEATURE_NAMES) for _, values in self.prototypes):
            raise ValueError("prototype public-range vector width is invalid")
        if any(not math.isfinite(value) for _, values in self.prototypes for value in values):
            raise ValueError("prototype public-range vectors must be finite")
        if self.temperature_grid != PUBLIC_RANGE_TEMPERATURE_GRID:
            raise ValueError("prototype public-range temperature grid drifted")
        if self.uniform_mix_grid != PUBLIC_RANGE_UNIFORM_MIX_GRID:
            raise ValueError("prototype public-range mixture grid drifted")
        expected_pairs = tuple(
            (temperature, uniform_mix)
            for temperature in self.temperature_grid
            for uniform_mix in self.uniform_mix_grid
        )
        if tuple(
            (item.temperature, item.uniform_mix) for item in self.selection_metrics
        ) != expected_pairs:
            raise ValueError("prototype public-range selection grid is incomplete")
        selected = min(
            self.selection_metrics,
            key=lambda item: (item.calibration_nll, item.temperature, item.uniform_mix),
        )
        if (
            self.selected_temperature != selected.temperature
            or self.selected_uniform_mix != selected.uniform_mix
        ):
            raise ValueError("prototype public-range selection is not the NLL winner")
        if not self.train_root_ids or not self.calibration_root_ids:
            raise ValueError("prototype public-range split roots are missing")
        if set(self.train_root_ids).intersection(self.calibration_root_ids):
            raise ValueError("prototype public-range train/calibration roots overlap")
        if not all((self.public_features_only, self.archetype_range_only)):
            raise ValueError("prototype public-range scope claims changed")
        if any(
            (
                self.selected_truth_identifier_input,
                self.hidden_hand_or_resource_posterior,
                self.official_public_range_calibrated,
                self.official_engine_executed,
            )
        ):
            raise ValueError("prototype public-range model promoted an invalid claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def probabilities(
        self,
        public_features: Sequence[tuple[str, float]],
    ) -> tuple[tuple[str, float], ...]:
        values = self.normalizer.transform(public_features)
        logits = []
        for candidate, prototype in self.prototypes:
            squared_distance = sum(
                (value - reference) ** 2
                for value, reference in zip(values, prototype, strict=True)
            )
            logits.append((candidate, -squared_distance / self.selected_temperature))
        maximum = max(value for _, value in logits)
        exponentials = [(candidate, math.exp(value - maximum)) for candidate, value in logits]
        total = sum(value for _, value in exponentials)
        uniform = 1.0 / len(exponentials)
        return tuple(
            (
                candidate,
                (1.0 - self.selected_uniform_mix) * value / total
                + self.selected_uniform_mix * uniform,
            )
            for candidate, value in exponentials
        )

    def predict(
        self,
        public_features: Sequence[tuple[str, float]],
    ) -> OpponentRange:
        return OpponentRange.normalized(
            tuple(
                RangeHypothesis(
                    hypothesis_id=candidate,
                    weight=probability,
                    hidden_resource_counts=(),
                    intent_id="unknown",
                    style_id="unknown",
                    source=(
                        "Phase 6AA public-only deck-archetype prototype diagnostic; "
                        "not an official hidden-state posterior"
                    ),
                )
                for candidate, probability in self.probabilities(public_features)
            )
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_PUBLIC_RANGE_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "normalizer": self.normalizer.to_dict(),
            "prototypes": [
                {"candidate_id": candidate, "values": list(values)}
                for candidate, values in self.prototypes
            ],
            "temperature_grid": list(self.temperature_grid),
            "uniform_mix_grid": list(self.uniform_mix_grid),
            "selection_metrics": [asdict(item) for item in self.selection_metrics],
            "selected_temperature": self.selected_temperature,
            "selected_uniform_mix": self.selected_uniform_mix,
            "train_root_ids": list(self.train_root_ids),
            "calibration_root_ids": list(self.calibration_root_ids),
            "train_rows_fingerprint": self.train_rows_fingerprint,
            "claims": {
                "public_features_only": self.public_features_only,
                "selected_truth_identifier_input": self.selected_truth_identifier_input,
                "archetype_range_only": self.archetype_range_only,
                "hidden_hand_or_resource_posterior": (
                    self.hidden_hand_or_resource_posterior
                ),
                "official_public_range_calibrated": (
                    self.official_public_range_calibrated
                ),
                "official_engine_executed": self.official_engine_executed,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PrototypePublicRangeModel":
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            normalizer=NamedFeatureNormalizer.from_dict(raw["normalizer"]),
            prototypes=tuple(
                (
                    str(value["candidate_id"]),
                    tuple(float(item) for item in value["values"]),
                )
                for value in raw["prototypes"]
            ),
            temperature_grid=tuple(float(value) for value in raw["temperature_grid"]),
            uniform_mix_grid=tuple(float(value) for value in raw["uniform_mix_grid"]),
            selection_metrics=tuple(
                PublicRangeSelectionMetric(**dict(value))
                for value in raw["selection_metrics"]
            ),
            selected_temperature=float(raw["selected_temperature"]),
            selected_uniform_mix=float(raw["selected_uniform_mix"]),
            train_root_ids=tuple(str(value) for value in raw["train_root_ids"]),
            calibration_root_ids=tuple(
                str(value) for value in raw["calibration_root_ids"]
            ),
            train_rows_fingerprint=str(raw["train_rows_fingerprint"]),
            public_features_only=bool(claims["public_features_only"]),
            selected_truth_identifier_input=bool(
                claims["selected_truth_identifier_input"]
            ),
            archetype_range_only=bool(claims["archetype_range_only"]),
            hidden_hand_or_resource_posterior=bool(
                claims["hidden_hand_or_resource_posterior"]
            ),
            official_public_range_calibrated=bool(
                claims["official_public_range_calibrated"]
            ),
            official_engine_executed=bool(claims["official_engine_executed"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("prototype public-range model fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class PublicRangeEvaluationRow:
    example_id: str
    source_root_id: str
    split: str
    target_candidate_id_supervision_only: str
    probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _validate_digest(self.example_id, "example_id")
        _validate_digest(self.source_root_id, "source_root_id")
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("public-range evaluation split is invalid")
        identifiers = tuple(candidate for candidate, _ in self.probabilities)
        if self.target_candidate_id_supervision_only not in identifiers:
            raise ValueError("public-range evaluation target is outside support")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("public-range evaluation support is duplicated")
        if any(probability < 0 or not math.isfinite(probability) for _, probability in self.probabilities):
            raise ValueError("public-range evaluation probability is invalid")
        if not math.isclose(
            sum(probability for _, probability in self.probabilities),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("public-range evaluation probabilities are not normalized")

    @property
    def target_probability(self) -> float:
        return dict(self.probabilities)[self.target_candidate_id_supervision_only]

    @property
    def nll(self) -> float:
        return -math.log(max(self.target_probability, 1e-15))

    @property
    def brier(self) -> float:
        return sum(
            (
                probability
                - float(candidate == self.target_candidate_id_supervision_only)
            )
            ** 2
            for candidate, probability in self.probabilities
        )

    @property
    def top1_correct(self) -> bool:
        selected = min(
            self.probabilities,
            key=lambda item: (-item[1], item[0]),
        )[0]
        return selected == self.target_candidate_id_supervision_only

    @property
    def confidence(self) -> float:
        return max(probability for _, probability in self.probabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_root_id": self.source_root_id,
            "split": self.split,
            "target_candidate_id_supervision_only": (
                self.target_candidate_id_supervision_only
            ),
            "probabilities": [list(item) for item in self.probabilities],
            "metrics": {
                "nll": self.nll,
                "brier": self.brier,
                "top1_correct": self.top1_correct,
                "confidence": self.confidence,
            },
            "claims": {"target_used_only_for_evaluation": True},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRangeEvaluationRow":
        if not bool(raw["claims"]["target_used_only_for_evaluation"]):
            raise ValueError("public-range evaluation truth boundary changed")
        result = cls(
            example_id=str(raw["example_id"]),
            source_root_id=str(raw["source_root_id"]),
            split=str(raw["split"]),
            target_candidate_id_supervision_only=str(
                raw["target_candidate_id_supervision_only"]
            ),
            probabilities=tuple(
                (str(candidate), float(probability))
                for candidate, probability in raw["probabilities"]
            ),
        )
        expected = {
            "nll": result.nll,
            "brier": result.brier,
            "top1_correct": result.top1_correct,
            "confidence": result.confidence,
        }
        metrics = raw["metrics"]
        if any(
            (
                not math.isclose(float(metrics["nll"]), expected["nll"], abs_tol=1e-12),
                not math.isclose(float(metrics["brier"]), expected["brier"], abs_tol=1e-12),
                bool(metrics["top1_correct"]) != expected["top1_correct"],
                not math.isclose(
                    float(metrics["confidence"]), expected["confidence"], abs_tol=1e-12
                ),
            )
        ):
            raise ValueError("public-range evaluation derived metrics mismatch")
        return result


@dataclass(frozen=True, slots=True)
class PublicRangeSplitMetrics:
    split: str
    roots: int
    candidate_strata: int
    nll: float
    brier: float
    top1_accuracy: float
    expected_calibration_error: float

    def __post_init__(self) -> None:
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("public-range split metrics split is invalid")
        if min(self.roots, self.candidate_strata) <= 0:
            raise ValueError("public-range split metrics counts are invalid")
        for name in ("nll", "brier"):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} is invalid")
        for name in ("top1_accuracy", "expected_calibration_error"):
            value = getattr(self, name)
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{name} is not a finite fraction")


@dataclass(frozen=True, slots=True)
class AlignedPublicRangeCalibration:
    dataset_fingerprint: str
    model_fingerprint: str
    evaluation_rows: tuple[PublicRangeEvaluationRow, ...]
    train: PublicRangeSplitMetrics
    calibration: PublicRangeSplitMetrics
    validation: PublicRangeSplitMetrics
    uniform_nll: float
    uniform_brier: float
    minimum_calibration_roots: int
    minimum_validation_roots: int
    root_split_gate_passed: bool
    heldout_archetype_signal_gate_passed: bool
    sufficient_calibration_roots_gate_passed: bool
    sufficient_validation_roots_gate_passed: bool
    official_public_range_calibrated: bool = False
    hidden_hand_or_resource_posterior: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        _validate_digest(self.model_fingerprint, "model_fingerprint")
        if tuple(item.split for item in (self.train, self.calibration, self.validation)) != (
            ALIGNED_VALUE_SPLITS
        ):
            raise ValueError("public-range calibration split metric order is invalid")
        if len(self.evaluation_rows) != sum(
            item.roots for item in (self.train, self.calibration, self.validation)
        ):
            raise ValueError("public-range calibration row accounting is inconsistent")
        if min(self.minimum_calibration_roots, self.minimum_validation_roots) <= 0:
            raise ValueError("public-range minimum root gates are invalid")
        expected_uniform_nll = math.log(self.train.candidate_strata)
        expected_uniform_brier = 1.0 - 1.0 / self.train.candidate_strata
        if not math.isclose(self.uniform_nll, expected_uniform_nll, abs_tol=1e-12):
            raise ValueError("public-range uniform NLL is inconsistent")
        if not math.isclose(self.uniform_brier, expected_uniform_brier, abs_tol=1e-12):
            raise ValueError("public-range uniform Brier is inconsistent")
        expected_signal = all(
            (
                self.root_split_gate_passed,
                self.calibration.nll < self.uniform_nll,
                self.validation.nll < self.uniform_nll,
            )
        )
        if self.heldout_archetype_signal_gate_passed != expected_signal:
            raise ValueError("public-range signal gate is inconsistent")
        if self.sufficient_calibration_roots_gate_passed != (
            self.calibration.roots >= self.minimum_calibration_roots
        ):
            raise ValueError("public-range calibration-root gate is inconsistent")
        if self.sufficient_validation_roots_gate_passed != (
            self.validation.roots >= self.minimum_validation_roots
        ):
            raise ValueError("public-range validation-root gate is inconsistent")
        if any(
            (
                self.official_public_range_calibrated,
                self.hidden_hand_or_resource_posterior,
                self.counterfactual_value_compatible,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("public-range calibration promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_PUBLIC_RANGE_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "evaluation_rows": [item.to_dict() for item in self.evaluation_rows],
            "split_metrics": {
                "train": asdict(self.train),
                "calibration": asdict(self.calibration),
                "validation": asdict(self.validation),
                "uniform_nll": self.uniform_nll,
                "uniform_brier": self.uniform_brier,
            },
            "thresholds": {
                "minimum_calibration_roots": self.minimum_calibration_roots,
                "minimum_validation_roots": self.minimum_validation_roots,
            },
            "gates": {
                "root_split": self.root_split_gate_passed,
                "heldout_archetype_signal": self.heldout_archetype_signal_gate_passed,
                "sufficient_calibration_roots": (
                    self.sufficient_calibration_roots_gate_passed
                ),
                "sufficient_validation_roots": (
                    self.sufficient_validation_roots_gate_passed
                ),
                "official_public_range_calibrated": (
                    self.official_public_range_calibrated
                ),
                "counterfactual_value_compatible": (
                    self.counterfactual_value_compatible
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
            "claims": {
                "deck_archetype_range_only": True,
                "hidden_hand_or_resource_posterior": (
                    self.hidden_hand_or_resource_posterior
                ),
                "observer_orientation_production_validated": False,
            },
            "blockers": [
                "six calibration and six validation roots are below the predeclared 60/60 minimum",
                "the output covers deck archetypes, not hidden hands, resources, or intents",
                "the Phase 6Z feature orientation lacks a production-validated observer adapter",
                "the source hidden-completion prior and upstream official reach remain unresolved",
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignedPublicRangeCalibration":
        split_metrics = raw["split_metrics"]
        thresholds = raw["thresholds"]
        gates = raw["gates"]
        claims = raw["claims"]
        if not bool(claims["deck_archetype_range_only"]):
            raise ValueError("public-range deck-archetype scope changed")
        if bool(claims["observer_orientation_production_validated"]):
            raise ValueError("public-range observer orientation cannot be promoted")
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            evaluation_rows=tuple(
                PublicRangeEvaluationRow.from_dict(value)
                for value in raw["evaluation_rows"]
            ),
            train=PublicRangeSplitMetrics(**dict(split_metrics["train"])),
            calibration=PublicRangeSplitMetrics(**dict(split_metrics["calibration"])),
            validation=PublicRangeSplitMetrics(**dict(split_metrics["validation"])),
            uniform_nll=float(split_metrics["uniform_nll"]),
            uniform_brier=float(split_metrics["uniform_brier"]),
            minimum_calibration_roots=int(
                thresholds["minimum_calibration_roots"]
            ),
            minimum_validation_roots=int(thresholds["minimum_validation_roots"]),
            root_split_gate_passed=bool(gates["root_split"]),
            heldout_archetype_signal_gate_passed=bool(
                gates["heldout_archetype_signal"]
            ),
            sufficient_calibration_roots_gate_passed=bool(
                gates["sufficient_calibration_roots"]
            ),
            sufficient_validation_roots_gate_passed=bool(
                gates["sufficient_validation_roots"]
            ),
            official_public_range_calibrated=bool(
                gates["official_public_range_calibrated"]
            ),
            hidden_hand_or_resource_posterior=bool(
                claims["hidden_hand_or_resource_posterior"]
            ),
            counterfactual_value_compatible=bool(
                gates["counterfactual_value_compatible"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if len(raw.get("blockers", ())) != 4:
            raise ValueError("public-range calibration blockers changed")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("public-range calibration fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6AlignedPublicRangeEvidence:
    benchmark_id: str
    status: str
    source_phase6z_dataset_fingerprint: str
    dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    candidate_strata: int
    train_roots: int
    calibration_roots: int
    validation_roots: int
    selected_temperature: float
    selected_uniform_mix: float
    calibration_nll: float
    validation_nll: float
    uniform_nll: float
    validation_brier: float
    uniform_brier: float
    validation_top1_accuracy: float
    validation_expected_calibration_error: float
    heldout_archetype_signal_gate_passed: bool
    official_public_range_calibrated: bool
    hidden_hand_or_resource_posterior: bool
    counterfactual_value_compatible: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AA evidence identity is missing")
        for name in (
            "source_phase6z_dataset_fingerprint",
            "dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if min(
            self.root_count,
            self.candidate_strata,
            self.train_roots,
            self.calibration_roots,
            self.validation_roots,
        ) <= 0:
            raise ValueError("Phase 6AA evidence accounting is invalid")
        if self.root_count != self.train_roots + self.calibration_roots + self.validation_roots:
            raise ValueError("Phase 6AA split accounting is inconsistent")
        if any(
            (
                self.official_public_range_calibrated,
                self.hidden_hand_or_resource_posterior,
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AA evidence cannot enable downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_PUBLIC_RANGE_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AA aligned public-range evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Public roots | {self.root_count} |
| Candidate strata | {self.candidate_strata} |
| Train / calibration / validation roots | {self.train_roots} / {self.calibration_roots} / {self.validation_roots} |
| Selected temperature / uniform mix | `{self.selected_temperature}` / `{self.selected_uniform_mix}` |
| Calibration NLL | `{self.calibration_nll:.6f}` |
| Validation NLL / uniform | `{self.validation_nll:.6f}` / `{self.uniform_nll:.6f}` |
| Validation Brier / uniform | `{self.validation_brier:.6f}` / `{self.uniform_brier:.6f}` |
| Validation top-1 accuracy | `{self.validation_top1_accuracy:.6f}` |
| Validation ECE | `{self.validation_expected_calibration_error:.6f}` |
| Held-out archetype-signal gate | **{str(self.heldout_archetype_signal_gate_passed).lower()}** |
| Official public range calibrated | **false** |

The source roots came from genuine official battles, but this prototype fit and
evaluation are offline abstractions. It predicts only six deck archetypes from
public features; it does not model hidden hands/resources/intents and has only
six calibration plus six validation roots. No counterfactual, resolver, or
official-control capability is enabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6AlignedPublicRangeEvidence":
        fields = {name: raw[name] for name in cls.__dataclass_fields__ if name != "claims"}
        result = cls(
            **fields,
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AA evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6AlignedPublicRangeArtifacts:
    dataset: AlignedPublicRangeDataset
    model: PrototypePublicRangeModel
    calibration: AlignedPublicRangeCalibration
    evidence: Phase6AlignedPublicRangeEvidence


def build_aligned_public_range_dataset(
    source: PublicRootAlignedValueDataset,
) -> AlignedPublicRangeDataset:
    examples = tuple(
        AlignedPublicRangeExample(
            example_id=_fingerprint(
                {
                    "source_root": root.root_id,
                    "split": root.split,
                    "candidate_label_only": root.candidate_id,
                    "public_features": root.public_features,
                }
            ),
            source_root_id=root.root_id,
            split=root.split,
            target_candidate_id_supervision_only=root.candidate_id,
            public_state_fingerprint=root.public_state_fingerprint,
            public_history_fingerprint=root.public_history_fingerprint,
            source_public_belief_fingerprint=root.public_belief_fingerprint,
            public_features=root.public_features,
        )
        for root in source.roots
    )
    dataset = AlignedPublicRangeDataset(
        source_phase6z_dataset_fingerprint=source.fingerprint,
        runtime_fingerprint=source.runtime_fingerprint,
        deck_fingerprint=source.deck_fingerprint,
        candidate_ids=source.candidate_ids,
        examples=examples,
        root_split_disjoint=source.collection_disjoint_roots_verified,
    )
    if _raw_private_key_count(dataset.to_dict()) != 0:
        raise RuntimeError("Phase 6AA serialized a forbidden raw private key")
    return dataset


def _probabilities_from_components(
    public_features: Sequence[tuple[str, float]],
    *,
    normalizer: NamedFeatureNormalizer,
    prototypes: Sequence[tuple[str, tuple[float, ...]]],
    temperature: float,
    uniform_mix: float,
) -> tuple[tuple[str, float], ...]:
    values = normalizer.transform(public_features)
    logits = []
    for candidate, prototype in prototypes:
        squared_distance = sum(
            (value - reference) ** 2
            for value, reference in zip(values, prototype, strict=True)
        )
        logits.append((candidate, -squared_distance / temperature))
    maximum = max(value for _, value in logits)
    exponentials = [(candidate, math.exp(value - maximum)) for candidate, value in logits]
    total = sum(value for _, value in exponentials)
    uniform = 1.0 / len(exponentials)
    return tuple(
        (
            candidate,
            (1.0 - uniform_mix) * value / total + uniform_mix * uniform,
        )
        for candidate, value in exponentials
    )


def _selection_summary(
    examples: Sequence[AlignedPublicRangeExample],
    *,
    normalizer: NamedFeatureNormalizer,
    prototypes: Sequence[tuple[str, tuple[float, ...]]],
    temperature: float,
    uniform_mix: float,
) -> tuple[float, float, float]:
    nll = []
    brier = []
    correct = []
    for example in examples:
        probabilities = _probabilities_from_components(
            example.public_features,
            normalizer=normalizer,
            prototypes=prototypes,
            temperature=temperature,
            uniform_mix=uniform_mix,
        )
        distribution = dict(probabilities)
        target = example.target_candidate_id_supervision_only
        nll.append(-math.log(max(distribution[target], 1e-15)))
        brier.append(
            sum(
                (probability - float(candidate == target)) ** 2
                for candidate, probability in probabilities
            )
        )
        predicted = min(probabilities, key=lambda item: (-item[1], item[0]))[0]
        correct.append(predicted == target)
    return (
        sum(nll) / len(nll),
        sum(brier) / len(brier),
        sum(correct) / len(correct),
    )


def train_prototype_public_range(
    dataset: AlignedPublicRangeDataset,
) -> PrototypePublicRangeModel:
    train = tuple(item for item in dataset.examples if item.split == "train")
    calibration = tuple(
        item for item in dataset.examples if item.split == "calibration"
    )
    train_rows_fingerprint = _fingerprint(
        [(item.example_id, item.public_features) for item in train]
    )
    normalizer = NamedFeatureNormalizer.fit(
        [item.public_features for item in train],
        fit_rows_fingerprint=train_rows_fingerprint,
    )
    prototypes = []
    for candidate in dataset.candidate_ids:
        rows = tuple(
            item
            for item in train
            if item.target_candidate_id_supervision_only == candidate
        )
        if len(rows) != 1:
            raise ValueError("Phase 6AA requires one train root per candidate")
        prototypes.append((candidate, normalizer.transform(rows[0].public_features)))
    metrics = []
    for temperature in PUBLIC_RANGE_TEMPERATURE_GRID:
        for uniform_mix in PUBLIC_RANGE_UNIFORM_MIX_GRID:
            nll, brier, accuracy = _selection_summary(
                calibration,
                normalizer=normalizer,
                prototypes=prototypes,
                temperature=temperature,
                uniform_mix=uniform_mix,
            )
            metrics.append(
                PublicRangeSelectionMetric(
                    temperature=temperature,
                    uniform_mix=uniform_mix,
                    calibration_nll=nll,
                    calibration_brier=brier,
                    calibration_top1_accuracy=accuracy,
                )
            )
    selected = min(
        metrics,
        key=lambda item: (item.calibration_nll, item.temperature, item.uniform_mix),
    )
    return PrototypePublicRangeModel(
        source_dataset_fingerprint=dataset.fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        candidate_ids=dataset.candidate_ids,
        normalizer=normalizer,
        prototypes=tuple(prototypes),
        temperature_grid=PUBLIC_RANGE_TEMPERATURE_GRID,
        uniform_mix_grid=PUBLIC_RANGE_UNIFORM_MIX_GRID,
        selection_metrics=tuple(metrics),
        selected_temperature=selected.temperature,
        selected_uniform_mix=selected.uniform_mix,
        train_root_ids=tuple(item.source_root_id for item in train),
        calibration_root_ids=tuple(item.source_root_id for item in calibration),
        train_rows_fingerprint=train_rows_fingerprint,
    )


def _split_metrics(
    rows: Sequence[PublicRangeEvaluationRow],
    split: str,
) -> PublicRangeSplitMetrics:
    selected = tuple(item for item in rows if item.split == split)
    confidence_correct = tuple(
        (item.confidence, item.top1_correct) for item in selected
    )
    ece = 0.0
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        bucket = tuple(
            (confidence, correct)
            for confidence, correct in confidence_correct
            if lower <= confidence < upper
            or (upper >= 1.0 and confidence <= 1.0)
        )
        if bucket:
            accuracy = sum(correct for _, correct in bucket) / len(bucket)
            confidence = sum(value for value, _ in bucket) / len(bucket)
            ece += len(bucket) / len(selected) * abs(accuracy - confidence)
    return PublicRangeSplitMetrics(
        split=split,
        roots=len(selected),
        candidate_strata=len(
            {item.target_candidate_id_supervision_only for item in selected}
        ),
        nll=sum(item.nll for item in selected) / len(selected),
        brier=sum(item.brier for item in selected) / len(selected),
        top1_accuracy=sum(item.top1_correct for item in selected) / len(selected),
        expected_calibration_error=ece,
    )


def evaluate_aligned_public_range(
    dataset: AlignedPublicRangeDataset,
    model: PrototypePublicRangeModel,
) -> AlignedPublicRangeCalibration:
    if model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("Phase 6AA model/dataset fingerprint mismatch")
    rows = tuple(
        PublicRangeEvaluationRow(
            example_id=item.example_id,
            source_root_id=item.source_root_id,
            split=item.split,
            target_candidate_id_supervision_only=(
                item.target_candidate_id_supervision_only
            ),
            probabilities=model.probabilities(item.public_features),
        )
        for item in dataset.examples
    )
    train = _split_metrics(rows, "train")
    calibration = _split_metrics(rows, "calibration")
    validation = _split_metrics(rows, "validation")
    uniform_nll = math.log(len(dataset.candidate_ids))
    uniform_brier = 1.0 - 1.0 / len(dataset.candidate_ids)
    signal_gate = all(
        (
            dataset.root_split_disjoint,
            calibration.nll < uniform_nll,
            validation.nll < uniform_nll,
        )
    )
    return AlignedPublicRangeCalibration(
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        evaluation_rows=rows,
        train=train,
        calibration=calibration,
        validation=validation,
        uniform_nll=uniform_nll,
        uniform_brier=uniform_brier,
        minimum_calibration_roots=PUBLIC_RANGE_MINIMUM_CALIBRATION_ROOTS,
        minimum_validation_roots=PUBLIC_RANGE_MINIMUM_VALIDATION_ROOTS,
        root_split_gate_passed=dataset.root_split_disjoint,
        heldout_archetype_signal_gate_passed=signal_gate,
        sufficient_calibration_roots_gate_passed=(
            calibration.roots >= PUBLIC_RANGE_MINIMUM_CALIBRATION_ROOTS
        ),
        sufficient_validation_roots_gate_passed=(
            validation.roots >= PUBLIC_RANGE_MINIMUM_VALIDATION_ROOTS
        ),
    )


def run_phase6_aligned_public_range(
    *,
    project_root: str | Path | None = None,
    source_phase6z_dataset_path: str | Path | None = None,
) -> Phase6AlignedPublicRangeArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    source_path = Path(
        source_phase6z_dataset_path
        or root
        / "artifacts"
        / "phase6_aligned_counterfactual_values"
        / "public_root_aligned_value_dataset.json"
    ).resolve()
    source = PublicRootAlignedValueDataset.from_dict(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    dataset = build_aligned_public_range_dataset(source)
    restored_dataset = AlignedPublicRangeDataset.from_dict(
        json.loads(json.dumps(dataset.to_dict()))
    )
    if restored_dataset.fingerprint != dataset.fingerprint:
        raise RuntimeError("Phase 6AA dataset roundtrip drifted")
    model = train_prototype_public_range(dataset)
    restored_model = PrototypePublicRangeModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    if restored_model.fingerprint != model.fingerprint:
        raise RuntimeError("Phase 6AA model roundtrip drifted")
    calibration = evaluate_aligned_public_range(dataset, model)
    claims = (
        ("source_roots", "genuine_official_battle_execution_in_phase6z"),
        ("prototype_fit", "offline_research_abstraction"),
        ("model_input", "public_features_only"),
        ("range_scope", "six_deck_archetypes_only"),
        ("hidden_hands_resources_intents", "not_modeled"),
        ("observer_orientation_adapter", "not_production_validated"),
        ("official_public_range", "not_calibrated"),
        ("counterfactual_values", "not_compatible"),
        ("safe_resolver", "not_enabled"),
        ("official_strategy_control", "disabled"),
        ("pokemon_strength", "not_claimed"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6AlignedPublicRangeEvidence(
        benchmark_id="phase6aa-aligned-public-archetype-range-v1",
        status=(
            "phase6aa_archetype_signal_validated_range_uncalibrated_partial"
            if calibration.heldout_archetype_signal_gate_passed
            else "phase6aa_public_range_diagnostic_rejected_partial"
        ),
        source_phase6z_dataset_fingerprint=source.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=len(dataset.examples),
        candidate_strata=len(dataset.candidate_ids),
        train_roots=calibration.train.roots,
        calibration_roots=calibration.calibration.roots,
        validation_roots=calibration.validation.roots,
        selected_temperature=model.selected_temperature,
        selected_uniform_mix=model.selected_uniform_mix,
        calibration_nll=calibration.calibration.nll,
        validation_nll=calibration.validation.nll,
        uniform_nll=calibration.uniform_nll,
        validation_brier=calibration.validation.brier,
        uniform_brier=calibration.uniform_brier,
        validation_top1_accuracy=calibration.validation.top1_accuracy,
        validation_expected_calibration_error=(
            calibration.validation.expected_calibration_error
        ),
        heldout_archetype_signal_gate_passed=(
            calibration.heldout_archetype_signal_gate_passed
        ),
        official_public_range_calibrated=False,
        hidden_hand_or_resource_posterior=False,
        counterfactual_value_compatible=False,
        safe_resolver_enabled=False,
        official_strategy_control_enabled=False,
        claims=claims,
    )
    return Phase6AlignedPublicRangeArtifacts(dataset, model, calibration, evidence)


def main() -> None:
    print(run_phase6_aligned_public_range().evidence.to_json())


if __name__ == "__main__":
    main()
