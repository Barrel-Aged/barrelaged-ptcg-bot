"""Phase 6T held-out allocation and history-conditioned range diagnostic.

This milestone addresses two Phase 6S omissions without promoting them into a
safety claim: hidden cards are resampled *between* zones, and the Phase 6I
public-only range head is evaluated along native descendant histories.  The
selected completion is used only as an evaluation label.  Native transitions
are genuine licensed-engine search execution; allocation particles, belief
features, predictions, and metrics are research abstractions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .game_theory import AbstractAction, InformationSetActor
from .native_public_range import LoggedTrajectoryNativePublicRangeProvider
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    PriorityBaseline,
    rocket_compatibility_profile,
)
from .official_search_game import (
    HiddenZoneCompletion,
    OfficialNativeSearchGame,
    sample_hidden_zone_completion,
)
from .official_trajectory import OfficialTrajectoryDataset
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _runtime_basics
from .trajectory_neural import (
    TrajectoryBootstrapArtifact,
    restore_logged_trajectory_bootstrap,
)


PHASE6_NATIVE_BELIEF_CALIBRATION_SCHEMA_VERSION = 1
PHASE6_NATIVE_BELIEF_CALIBRATION_SEED = 6_026


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


@dataclass(frozen=True, slots=True)
class NativeBeliefDiagnosticConfig:
    minimum_independent_roots: int = 6
    minimum_validation_rows: int = 60
    minimum_candidate_strata: int = 6
    minimum_nll_gain_over_uniform: float = 0.05
    minimum_nll_gain_over_causal: float = 0.0
    maximum_expected_calibration_error: float = 0.20

    def __post_init__(self) -> None:
        if min(
            self.minimum_independent_roots,
            self.minimum_validation_rows,
            self.minimum_candidate_strata,
        ) <= 0:
            raise ValueError("native belief diagnostic minimums must be positive")
        if min(
            self.minimum_nll_gain_over_uniform,
            self.minimum_nll_gain_over_causal,
            self.maximum_expected_calibration_error,
        ) < 0:
            raise ValueError("native belief diagnostic thresholds cannot be negative")
        if self.maximum_expected_calibration_error > 1:
            raise ValueError("expected calibration error must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class NativeRangeEvaluationRow:
    row_id: str
    root_id: str
    particle_id: str
    particle_fingerprint: str
    candidate_id_supervision_only: str
    particle_split: str
    particle_variant: int
    observation_index: int
    native_depth: int
    turn: int
    public_state_fingerprint: str
    public_history_fingerprint: str
    model_input_fingerprint: str
    revealed_opponent_card_count: int
    neural_distribution: tuple[tuple[str, float], ...]
    causal_distribution: tuple[tuple[str, float], ...]
    model_input_public_features: tuple[tuple[str, float], ...]
    source_descendant_official_engine_executed: bool = True
    hidden_completion_truth_input: bool = False
    private_feature_input: bool = False
    future_reveal_input: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "row_id",
            "root_id",
            "particle_fingerprint",
            "public_state_fingerprint",
            "public_history_fingerprint",
            "model_input_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.particle_id or not self.candidate_id_supervision_only:
            raise ValueError("native range row particle/candidate IDs are required")
        if self.particle_split not in {"calibration", "validation"}:
            raise ValueError("native range row split is invalid")
        if min(
            self.particle_variant,
            self.observation_index,
            self.native_depth,
            self.turn,
            self.revealed_opponent_card_count,
        ) < 0:
            raise ValueError("native range row counters cannot be negative")
        neural = dict(self.neural_distribution)
        causal = dict(self.causal_distribution)
        if set(neural) != set(causal) or self.candidate_id_supervision_only not in neural:
            raise ValueError("native range row distributions/truth support disagree")
        for name, distribution in (("neural", neural), ("causal", causal)):
            if any(not math.isfinite(value) or value < 0 for value in distribution.values()):
                raise ValueError(f"{name} range probabilities are invalid")
            if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-8):
                raise ValueError(f"{name} range probabilities must sum to one")
        if not self.source_descendant_official_engine_executed:
            raise ValueError("native range row must come from a genuine descendant state")
        if any(
            (
                self.hidden_completion_truth_input,
                self.private_feature_input,
                self.future_reveal_input,
            )
        ):
            raise ValueError("native range model cannot consume evaluation-only truth/private data")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "neural_distribution": dict(self.neural_distribution),
            "causal_distribution": dict(self.causal_distribution),
            "model_input_public_features": dict(self.model_input_public_features),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeRangeEvaluationRow":
        def pairs(name: str) -> tuple[tuple[str, float], ...]:
            values = raw[name]
            materialized = values.items() if isinstance(values, Mapping) else values
            return tuple((str(key), float(value)) for key, value in materialized)

        return cls(
            row_id=str(raw["row_id"]),
            root_id=str(raw["root_id"]),
            particle_id=str(raw["particle_id"]),
            particle_fingerprint=str(raw["particle_fingerprint"]),
            candidate_id_supervision_only=str(raw["candidate_id_supervision_only"]),
            particle_split=str(raw["particle_split"]),
            particle_variant=int(raw["particle_variant"]),
            observation_index=int(raw["observation_index"]),
            native_depth=int(raw["native_depth"]),
            turn=int(raw["turn"]),
            public_state_fingerprint=str(raw["public_state_fingerprint"]),
            public_history_fingerprint=str(raw["public_history_fingerprint"]),
            model_input_fingerprint=str(raw["model_input_fingerprint"]),
            revealed_opponent_card_count=int(raw["revealed_opponent_card_count"]),
            neural_distribution=pairs("neural_distribution"),
            causal_distribution=pairs("causal_distribution"),
            model_input_public_features=pairs("model_input_public_features"),
            source_descendant_official_engine_executed=bool(
                raw["source_descendant_official_engine_executed"]
            ),
            hidden_completion_truth_input=bool(raw["hidden_completion_truth_input"]),
            private_feature_input=bool(raw["private_feature_input"]),
            future_reveal_input=bool(raw["future_reveal_input"]),
        )


@dataclass(frozen=True, slots=True)
class NativeAllocationRangeDataset:
    runtime_fingerprint: str
    deck_fingerprint: str
    source_replay_dataset_fingerprint: str
    source_range_model_fingerprint: str
    range_provider_fingerprint: str
    candidate_ids: tuple[str, ...]
    allocation_family_fingerprint: str
    independent_root_count: int
    particle_variants: int
    observations_per_particle: int
    native_transition_count: int
    rows: tuple[NativeRangeEvaluationRow, ...]
    allocation_heldout: bool = True
    selected_truth_in_model_input: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_replay_dataset_fingerprint",
            "source_range_model_fingerprint",
            "range_provider_fingerprint",
            "allocation_family_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("native allocation range candidates must be unique")
        if min(
            self.independent_root_count,
            self.particle_variants,
            self.observations_per_particle,
        ) <= 0 or self.native_transition_count < 0:
            raise ValueError("native allocation range counts are invalid")
        if not self.rows:
            raise ValueError("native allocation range dataset cannot be empty")
        if any(
            set(name for name, _ in row.neural_distribution) != set(self.candidate_ids)
            for row in self.rows
        ):
            raise ValueError("native range row support does not match dataset candidates")
        if not self.allocation_heldout or self.selected_truth_in_model_input:
            raise ValueError("native allocation dataset must remain held-out and truth-free")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("native allocation dataset cannot persist raw private IDs")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_BELIEF_CALIBRATION_SCHEMA_VERSION,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_replay_dataset_fingerprint": self.source_replay_dataset_fingerprint,
            "source_range_model_fingerprint": self.source_range_model_fingerprint,
            "range_provider_fingerprint": self.range_provider_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "allocation_family_fingerprint": self.allocation_family_fingerprint,
            "independent_root_count": self.independent_root_count,
            "particle_variants": self.particle_variants,
            "observations_per_particle": self.observations_per_particle,
            "native_transition_count": self.native_transition_count,
            "rows": [row.to_dict() for row in self.rows],
            "claims": {
                "allocation_heldout": self.allocation_heldout,
                "selected_truth_in_model_input": self.selected_truth_in_model_input,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeAllocationRangeDataset":
        claims = raw["claims"]
        result = cls(
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_replay_dataset_fingerprint=str(raw["source_replay_dataset_fingerprint"]),
            source_range_model_fingerprint=str(raw["source_range_model_fingerprint"]),
            range_provider_fingerprint=str(raw["range_provider_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            allocation_family_fingerprint=str(raw["allocation_family_fingerprint"]),
            independent_root_count=int(raw["independent_root_count"]),
            particle_variants=int(raw["particle_variants"]),
            observations_per_particle=int(raw["observations_per_particle"]),
            native_transition_count=int(raw["native_transition_count"]),
            rows=tuple(NativeRangeEvaluationRow.from_dict(item) for item in raw["rows"]),
            allocation_heldout=bool(claims["allocation_heldout"]),
            selected_truth_in_model_input=bool(claims["selected_truth_in_model_input"]),
            raw_private_card_ids_persisted=int(claims["raw_private_card_ids_persisted"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("native allocation range dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class RangeDistributionMetrics:
    rows: int
    nll: float
    brier: float
    top1_accuracy: float
    expected_calibration_error: float

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("range metrics require rows")
        for field_name in (
            "nll",
            "brier",
            "top1_accuracy",
            "expected_calibration_error",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"range metric {field_name} is invalid")
        if self.top1_accuracy > 1 or self.expected_calibration_error > 1:
            raise ValueError("range accuracy/calibration metrics must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class NativeRangeSplitEvaluation:
    split: str
    candidate_strata: int
    particles: int
    neural: RangeDistributionMetrics
    causal: RangeDistributionMetrics
    uniform: RangeDistributionMetrics
    neural_nll_gain_over_uniform: float
    neural_nll_gain_over_causal: float

    def __post_init__(self) -> None:
        if self.split not in {"calibration", "validation"}:
            raise ValueError("native range evaluation split is invalid")
        if self.candidate_strata <= 0 or self.particles <= 0:
            raise ValueError("native range evaluation strata/particles are invalid")
        if len({self.neural.rows, self.causal.rows, self.uniform.rows}) != 1:
            raise ValueError("native range metric row counts disagree")
        for field_name in (
            "neural_nll_gain_over_uniform",
            "neural_nll_gain_over_causal",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError("native range NLL gains must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "candidate_strata": self.candidate_strata,
            "particles": self.particles,
            "neural": asdict(self.neural),
            "causal": asdict(self.causal),
            "uniform": asdict(self.uniform),
            "neural_nll_gain_over_uniform": self.neural_nll_gain_over_uniform,
            "neural_nll_gain_over_causal": self.neural_nll_gain_over_causal,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeRangeSplitEvaluation":
        return cls(
            split=str(raw["split"]),
            candidate_strata=int(raw["candidate_strata"]),
            particles=int(raw["particles"]),
            neural=RangeDistributionMetrics(**dict(raw["neural"])),
            causal=RangeDistributionMetrics(**dict(raw["causal"])),
            uniform=RangeDistributionMetrics(**dict(raw["uniform"])),
            neural_nll_gain_over_uniform=float(raw["neural_nll_gain_over_uniform"]),
            neural_nll_gain_over_causal=float(raw["neural_nll_gain_over_causal"]),
        )


@dataclass(frozen=True, slots=True)
class NativeBeliefCalibrationArtifact:
    dataset_fingerprint: str
    config: NativeBeliefDiagnosticConfig
    calibration: NativeRangeSplitEvaluation
    validation: NativeRangeSplitEvaluation
    allocation_generalization_gate_passed: bool
    public_belief_calibrated: bool = False
    official_hidden_state_posterior: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        validation = self.validation
        expected = all(
            (
                validation.neural.rows >= self.config.minimum_validation_rows,
                validation.candidate_strata >= self.config.minimum_candidate_strata,
                validation.neural_nll_gain_over_uniform
                >= self.config.minimum_nll_gain_over_uniform,
                validation.neural_nll_gain_over_causal
                >= self.config.minimum_nll_gain_over_causal,
                validation.neural.expected_calibration_error
                <= self.config.maximum_expected_calibration_error,
            )
        )
        if self.allocation_generalization_gate_passed != expected:
            raise ValueError("native allocation range gate disagrees with metrics")
        if self.public_belief_calibrated or self.official_hidden_state_posterior:
            raise ValueError("native allocation diagnostic cannot claim official calibration")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_BELIEF_CALIBRATION_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config": asdict(self.config),
            "calibration": self.calibration.to_dict(),
            "validation": self.validation.to_dict(),
            "gates": {
                "allocation_generalization": self.allocation_generalization_gate_passed,
                "public_belief_calibrated": self.public_belief_calibrated,
                "official_hidden_state_posterior": self.official_hidden_state_posterior,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeBeliefCalibrationArtifact":
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            config=NativeBeliefDiagnosticConfig(**dict(raw["config"])),
            calibration=NativeRangeSplitEvaluation.from_dict(raw["calibration"]),
            validation=NativeRangeSplitEvaluation.from_dict(raw["validation"]),
            allocation_generalization_gate_passed=bool(gates["allocation_generalization"]),
            public_belief_calibrated=bool(gates["public_belief_calibrated"]),
            official_hidden_state_posterior=bool(gates["official_hidden_state_posterior"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("native belief calibration fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativeBeliefCalibrationEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    particle_count: int
    row_count: int
    validation_rows: int
    candidate_strata: int
    native_transition_count: int
    validation_neural_nll: float
    validation_causal_nll: float
    validation_uniform_nll: float
    validation_neural_top1: float
    validation_neural_ece: float
    allocation_generalization_gate_passed: bool
    public_belief_calibrated: bool
    counterfactual_values_calibrated: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    raw_private_card_ids_persisted: int
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in ("dataset_fingerprint", "calibration_fingerprint"):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6T evidence identity is required")
        if min(
            self.root_count,
            self.particle_count,
            self.row_count,
            self.validation_rows,
            self.candidate_strata,
        ) <= 0 or self.native_transition_count < 0:
            raise ValueError("Phase 6T evidence counts are invalid")
        if any(
            (
                self.public_belief_calibrated,
                self.counterfactual_values_calibrated,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6T diagnostic cannot enable safety/deployment")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("Phase 6T cannot persist raw private card IDs")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_BELIEF_CALIBRATION_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6T native allocation/range evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Independent native roots | {self.root_count} |
| Allocation particles | {self.particle_count} |
| Public decision rows | {self.row_count} |
| Validation rows | {self.validation_rows} |
| Genuine native transitions | {self.native_transition_count} |
| Neural validation NLL | {self.validation_neural_nll:.6f} |
| Causal reveal validation NLL | {self.validation_causal_nll:.6f} |
| Uniform validation NLL | {self.validation_uniform_nll:.6f} |
| Neural validation top-1 | {self.validation_neural_top1:.3%} |
| Neural validation ECE | {self.validation_neural_ece:.6f} |
| Allocation diagnostic gate | **{str(self.allocation_generalization_gate_passed).lower()}** |
| Public belief calibrated | **false** |

The source observations follow genuine licensed native-search transitions. The
hidden allocations and range labels are supplied research particles, and the
Phase 6I range head remains an offline abstraction. No counterfactual bound,
resolver, official strategy controller, performance claim, or Kaggle
submission is enabled by this diagnostic.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6NativeBeliefCalibrationEvidence":
        result = cls(
            benchmark_id=str(raw["benchmark_id"]),
            status=str(raw["status"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            calibration_fingerprint=str(raw["calibration_fingerprint"]),
            root_count=int(raw["root_count"]),
            particle_count=int(raw["particle_count"]),
            row_count=int(raw["row_count"]),
            validation_rows=int(raw["validation_rows"]),
            candidate_strata=int(raw["candidate_strata"]),
            native_transition_count=int(raw["native_transition_count"]),
            validation_neural_nll=float(raw["validation_neural_nll"]),
            validation_causal_nll=float(raw["validation_causal_nll"]),
            validation_uniform_nll=float(raw["validation_uniform_nll"]),
            validation_neural_top1=float(raw["validation_neural_top1"]),
            validation_neural_ece=float(raw["validation_neural_ece"]),
            allocation_generalization_gate_passed=bool(
                raw["allocation_generalization_gate_passed"]
            ),
            public_belief_calibrated=bool(raw["public_belief_calibrated"]),
            counterfactual_values_calibrated=bool(raw["counterfactual_values_calibrated"]),
            safe_resolver_enabled=bool(raw["safe_resolver_enabled"]),
            official_strategy_control_enabled=bool(raw["official_strategy_control_enabled"]),
            raw_private_card_ids_persisted=int(raw["raw_private_card_ids_persisted"]),
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6T evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativeBeliefCalibrationArtifacts:
    dataset: NativeAllocationRangeDataset
    calibration: NativeBeliefCalibrationArtifact
    evidence: Phase6NativeBeliefCalibrationEvidence


def _distribution_metrics(
    rows: Sequence[NativeRangeEvaluationRow],
    *,
    source: str,
) -> RangeDistributionMetrics:
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("range metrics require rows")
    candidate_ids = tuple(name for name, _ in materialized[0].neural_distribution)
    uniform_probability = 1.0 / len(candidate_ids)
    losses = []
    briers = []
    confidence_correct: list[tuple[float, float]] = []
    for row in materialized:
        if source == "neural":
            distribution = dict(row.neural_distribution)
        elif source == "causal":
            distribution = dict(row.causal_distribution)
        elif source == "uniform":
            distribution = {name: uniform_probability for name in candidate_ids}
        else:
            raise ValueError("unknown range metric source")
        truth = row.candidate_id_supervision_only
        losses.append(-math.log(max(distribution[truth], 1e-12)))
        briers.append(
            sum(
                (probability - float(name == truth)) ** 2
                for name, probability in distribution.items()
            )
        )
        best = max(sorted(distribution), key=distribution.__getitem__)
        confidence_correct.append((distribution[best], float(best == truth)))
    ece = 0.0
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        bucket = [
            (confidence, correct)
            for confidence, correct in confidence_correct
            if lower <= confidence < upper or (upper >= 1.0 and confidence <= 1.0)
        ]
        if bucket:
            accuracy = sum(correct for _, correct in bucket) / len(bucket)
            confidence = sum(value for value, _ in bucket) / len(bucket)
            ece += len(bucket) / len(materialized) * abs(accuracy - confidence)
    return RangeDistributionMetrics(
        rows=len(materialized),
        nll=sum(losses) / len(losses),
        brier=sum(briers) / len(briers),
        top1_accuracy=sum(correct for _, correct in confidence_correct) / len(materialized),
        expected_calibration_error=ece,
    )


def _evaluate_split(
    rows: Sequence[NativeRangeEvaluationRow],
    split: str,
) -> NativeRangeSplitEvaluation:
    selected = tuple(row for row in rows if row.particle_split == split)
    neural = _distribution_metrics(selected, source="neural")
    causal = _distribution_metrics(selected, source="causal")
    uniform = _distribution_metrics(selected, source="uniform")
    return NativeRangeSplitEvaluation(
        split=split,
        candidate_strata=len({row.candidate_id_supervision_only for row in selected}),
        particles=len({(row.root_id, row.particle_id) for row in selected}),
        neural=neural,
        causal=causal,
        uniform=uniform,
        neural_nll_gain_over_uniform=uniform.nll - neural.nll,
        neural_nll_gain_over_causal=causal.nll - neural.nll,
    )


def evaluate_native_allocation_ranges(
    dataset: NativeAllocationRangeDataset,
    config: NativeBeliefDiagnosticConfig | None = None,
) -> NativeBeliefCalibrationArtifact:
    config = config or NativeBeliefDiagnosticConfig()
    calibration = _evaluate_split(dataset.rows, "calibration")
    validation = _evaluate_split(dataset.rows, "validation")
    gate = all(
        (
            dataset.independent_root_count >= config.minimum_independent_roots,
            validation.neural.rows >= config.minimum_validation_rows,
            validation.candidate_strata >= config.minimum_candidate_strata,
            validation.neural_nll_gain_over_uniform
            >= config.minimum_nll_gain_over_uniform,
            validation.neural_nll_gain_over_causal
            >= config.minimum_nll_gain_over_causal,
            validation.neural.expected_calibration_error
            <= config.maximum_expected_calibration_error,
        )
    )
    # The artifact's narrow metric gate deliberately excludes root-count so a
    # consumer cannot mistake a good correlated slice for public calibration.
    metric_gate = all(
        (
            validation.neural.rows >= config.minimum_validation_rows,
            validation.candidate_strata >= config.minimum_candidate_strata,
            validation.neural_nll_gain_over_uniform
            >= config.minimum_nll_gain_over_uniform,
            validation.neural_nll_gain_over_causal
            >= config.minimum_nll_gain_over_causal,
            validation.neural.expected_calibration_error
            <= config.maximum_expected_calibration_error,
        )
    )
    if gate and not metric_gate:
        raise AssertionError("native allocation gate implication failed")
    return NativeBeliefCalibrationArtifact(
        dataset_fingerprint=dataset.fingerprint,
        config=config,
        calibration=calibration,
        validation=validation,
        allocation_generalization_gate_passed=metric_gate,
    )


def _sample_chance(game: OfficialNativeSearchGame, state, rng: random.Random):
    outcomes = game.chance_outcomes(state)
    threshold = rng.random()
    cumulative = 0.0
    for outcome in outcomes:
        cumulative += outcome.probability
        if threshold <= cumulative + 1e-15:
            return outcome
    return outcomes[-1]


def _matching_action(
    actions: Sequence[AbstractAction],
    selected: Sequence[int],
) -> AbstractAction:
    indices = tuple(int(value) for value in selected)
    matches = [action for action in actions if action.official_option_indices == indices]
    if len(matches) != 1:
        raise RuntimeError("native belief baseline action did not lift exactly once")
    return matches[0]


def _allocation_particles(
    observation: Mapping[str, Any],
    *,
    root_index: int,
    target: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    basic_card_ids: Sequence[int],
    variants: int,
    seed: int,
) -> tuple[tuple[HiddenZoneCompletion, ...], dict[str, tuple[str, int, str]]]:
    if variants < 3:
        raise ValueError("Phase 6T requires calibration and validation allocations")
    probability = 1.0 / (len(candidates) * variants)
    particles = []
    metadata: dict[str, tuple[str, int, str]] = {}
    for candidate in candidates:
        for variant in range(variants):
            split = "validation" if variant == variants - 1 else "calibration"
            identifier = (
                f"root-{root_index}:{candidate.archetype}:allocation-{variant}:{split}"
            )
            particle = sample_hidden_zone_completion(
                observation,
                hypothesis_id=identifier,
                probability=probability,
                allocation_seed=_seed(seed, root_index, candidate.fingerprint, variant),
                your_candidate_id=target.archetype,
                opponent_candidate_id=candidate.archetype,
                your_archetype_id=target.archetype,
                opponent_archetype_id=candidate.archetype,
                your_style_id="unknown",
                opponent_style_id="unknown",
                your_card_ids=target.card_ids,
                opponent_card_ids=candidate.card_ids,
                your_deck_fingerprint=target.fingerprint,
                opponent_deck_fingerprint=candidate.fingerprint,
                basic_card_ids=basic_card_ids,
                manual_coin=True,
                source=(
                    "Phase 6T seeded public-consistent hidden-zone allocation "
                    "particle; evaluation abstraction, not official posterior"
                ),
            )
            particles.append(particle)
            metadata[identifier] = (candidate.archetype, variant, split)
    fingerprints = [particle.fingerprint for particle in particles]
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("Phase 6T allocation particles are not distinct")
    return tuple(particles), metadata


def run_phase6_native_belief_calibration(
    *,
    project_root: str | Path | None = None,
    source_replay_dataset_path: str | Path | None = None,
    source_range_model_path: str | Path | None = None,
    independent_roots: int = 2,
    particle_variants: int = 3,
    observations_per_particle: int = 6,
    maximum_transitions_per_particle: int = 256,
    config: NativeBeliefDiagnosticConfig | None = None,
    seed: int = PHASE6_NATIVE_BELIEF_CALIBRATION_SEED,
) -> Phase6NativeBeliefCalibrationArtifacts:
    if min(
        independent_roots,
        particle_variants,
        observations_per_particle,
        maximum_transitions_per_particle,
    ) <= 0:
        raise ValueError("Phase 6T limits must be positive")
    if particle_variants < 3:
        raise ValueError("Phase 6T requires at least three allocation variants")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    replay_path = Path(
        source_replay_dataset_path
        or root
        / "artifacts"
        / "phase6_official_trajectory"
        / "official_trajectory_dataset.json"
    ).resolve()
    model_path = Path(
        source_range_model_path
        or root
        / "artifacts"
        / "phase6_trajectory_neural"
        / "logged_trajectory_bootstrap.json"
    ).resolve()
    replay_dataset = OfficialTrajectoryDataset.from_dict(
        json.loads(replay_path.read_text(encoding="utf-8"))
    )
    model_artifact = TrajectoryBootstrapArtifact.from_dict(
        json.loads(model_path.read_text(encoding="utf-8"))
    )
    if model_artifact.dataset_fingerprint != replay_dataset.fingerprint:
        raise ValueError("Phase 6T range model/replay dataset mismatch")
    bootstrap = restore_logged_trajectory_bootstrap(model_artifact)
    range_provider = LoggedTrajectoryNativePublicRangeProvider(
        bootstrap,
        replay_dataset.candidates,
    )
    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    if set(model_artifact.candidate_ids) != set(profile_by_id):
        raise ValueError("Phase 6T candidate catalog/native profiles mismatch")
    candidates = tuple(profile_by_id[name] for name in model_artifact.candidate_ids)
    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_start  # type: ignore[import-not-found]

    rows: list[NativeRangeEvaluationRow] = []
    particle_fingerprints: list[str] = []
    total_transitions = 0
    runtime_fingerprint = ""
    basics = _runtime_basics(runtime)
    for root_index in range(independent_roots):
        observation, start = battle_start(
            list(target.card_ids),
            list(candidates[0].card_ids),
        )
        if observation is None:
            raise RuntimeError(
                f"official engine rejected Phase 6T root {root_index}: "
                f"player={start.errorPlayer}, error={start.errorType}"
            )
        try:
            particles, metadata = _allocation_particles(
                observation,
                root_index=root_index,
                target=target,
                candidates=candidates,
                basic_card_ids=basics,
                variants=particle_variants,
                seed=seed,
            )
            particle_fingerprints.extend(item.fingerprint for item in particles)
            with OfficialNativeSearchGame(
                observation,
                particles,
                runtime_directory=runtime,
            ) as game:
                runtime_fingerprint = game.provenance.ruleset_fingerprint
                for particle_index, outcome in enumerate(
                    game.chance_outcomes(game.root_state())
                ):
                    state = game.transition(
                        game.root_state(),
                        None,
                        chance_outcome=outcome,
                    )
                    completion = game.completion(state.hypothesis_id or "")
                    candidate_id, variant, split = metadata[completion.hypothesis_id]
                    agents = (
                        PriorityBaseline(target.card_ids, style="tempo"),
                        PriorityBaseline(profile_by_id[candidate_id].card_ids, style="tempo"),
                    )
                    observed = 0
                    rng = random.Random(
                        _seed(seed, "trajectory", root_index, particle_index)
                    )
                    for _ in range(maximum_transitions_per_particle):
                        if state.terminal or observed >= observations_per_particle:
                            break
                        if state.actor is InformationSetActor.CHANCE:
                            state = game.transition(
                                state,
                                None,
                                chance_outcome=_sample_chance(game, state, rng),
                            )
                            continue
                        player = game.player_to_act(state)
                        if player not in (0, 1):
                            raise RuntimeError("Phase 6T native state lacks a player")
                        information_set = game.information_set(state, player=player)
                        if player == 0:
                            base_belief = game.public_belief(state, player=0)
                            prediction = range_provider.predict(
                                state.observation,
                                base_belief,
                            )
                            neural = tuple(
                                (item.hypothesis_id, item.weight)
                                for item in prediction.opponent_range.hypotheses
                            )
                            causal = tuple(
                                (item.hypothesis_id, item.weight)
                                for item in prediction.causal_reveal_range.hypotheses
                            )
                            input_fingerprint = _fingerprint(
                                {
                                    "public_features": prediction.model_input_public_features,
                                    "provider": prediction.provider_fingerprint,
                                    "truth_input": False,
                                }
                            )
                            row_id = _fingerprint(
                                {
                                    "root": root_index,
                                    "particle": completion.fingerprint,
                                    "state": state.state_id,
                                    "observation_index": observed,
                                    "model_input": input_fingerprint,
                                }
                            )
                            rows.append(
                                NativeRangeEvaluationRow(
                                    row_id=row_id,
                                    root_id=_fingerprint(
                                        {
                                            "root_index": root_index,
                                            "public_root": state.path[0],
                                            "range_model": model_artifact.fingerprint,
                                        }
                                    ),
                                    particle_id=completion.hypothesis_id,
                                    particle_fingerprint=completion.fingerprint,
                                    candidate_id_supervision_only=candidate_id,
                                    particle_split=split,
                                    particle_variant=variant,
                                    observation_index=observed,
                                    native_depth=state.depth,
                                    turn=max(0, int(base_belief.turn)),
                                    public_state_fingerprint=(
                                        base_belief.public_state_fingerprint
                                    ),
                                    public_history_fingerprint=(
                                        base_belief.public_history_fingerprint
                                    ),
                                    model_input_fingerprint=input_fingerprint,
                                    revealed_opponent_card_count=(
                                        prediction.revealed_opponent_card_count
                                    ),
                                    neural_distribution=neural,
                                    causal_distribution=causal,
                                    model_input_public_features=(
                                        prediction.model_input_public_features
                                    ),
                                )
                            )
                            observed += 1
                        selection = agents[player].choose(state.observation)
                        state = game.transition(
                            state,
                            _matching_action(
                                information_set.legal_actions,
                                selection,
                            ),
                        )
                    if observed != observations_per_particle:
                        raise RuntimeError(
                            "Phase 6T particle did not yield the requested public rows"
                        )
                total_transitions += game.transition_count
        finally:
            battle_finish()

    allocation_family_fingerprint = _fingerprint(
        {
            "particle_fingerprints": particle_fingerprints,
            "root_count": independent_roots,
            "variants": particle_variants,
            "allocation_heldout": True,
            "order_only": False,
        }
    )
    provisional = NativeAllocationRangeDataset(
        runtime_fingerprint=runtime_fingerprint,
        deck_fingerprint=target.fingerprint,
        source_replay_dataset_fingerprint=replay_dataset.fingerprint,
        source_range_model_fingerprint=model_artifact.fingerprint,
        range_provider_fingerprint=range_provider.fingerprint,
        candidate_ids=model_artifact.candidate_ids,
        allocation_family_fingerprint=allocation_family_fingerprint,
        independent_root_count=independent_roots,
        particle_variants=particle_variants,
        observations_per_particle=observations_per_particle,
        native_transition_count=total_transitions,
        rows=tuple(rows),
    )
    serialized = provisional.to_dict()
    raw_private = _raw_private_key_count(serialized)
    if raw_private != 0:
        raise RuntimeError("Phase 6T serialized a forbidden raw private key")
    dataset = provisional
    restored = NativeAllocationRangeDataset.from_dict(json.loads(json.dumps(serialized)))
    if restored.fingerprint != dataset.fingerprint:
        raise RuntimeError("Phase 6T dataset roundtrip drifted")
    calibration = evaluate_native_allocation_ranges(dataset, config)
    validation = calibration.validation
    claims = (
        ("source_descendant_observations", "genuine_official_search_execution"),
        ("hidden_allocations", "supplied_research_particles_not_official_belief"),
        ("range_model", "phase6i_public_only_offline_abstraction"),
        ("selected_completion_truth_input", "false_evaluation_label_only"),
        ("allocation_generalization_gate", str(calibration.allocation_generalization_gate_passed).lower()),
        ("public_belief_calibrated", "false"),
        ("counterfactual_value_bounds", "not_built"),
        ("safe_resolver", "not_enabled"),
        ("official_strategy_control", "disabled"),
        ("pokemon_strength", "not_claimed"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6NativeBeliefCalibrationEvidence(
        benchmark_id="phase6t-native-allocation-range-v1",
        status="phase6t_native_allocation_range_diagnostic_validated_partial",
        dataset_fingerprint=dataset.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=independent_roots,
        particle_count=independent_roots * len(candidates) * particle_variants,
        row_count=len(dataset.rows),
        validation_rows=validation.neural.rows,
        candidate_strata=validation.candidate_strata,
        native_transition_count=dataset.native_transition_count,
        validation_neural_nll=validation.neural.nll,
        validation_causal_nll=validation.causal.nll,
        validation_uniform_nll=validation.uniform.nll,
        validation_neural_top1=validation.neural.top1_accuracy,
        validation_neural_ece=validation.neural.expected_calibration_error,
        allocation_generalization_gate_passed=(
            calibration.allocation_generalization_gate_passed
        ),
        public_belief_calibrated=False,
        counterfactual_values_calibrated=False,
        safe_resolver_enabled=False,
        official_strategy_control_enabled=False,
        raw_private_card_ids_persisted=raw_private,
        claims=claims,
    )
    return Phase6NativeBeliefCalibrationArtifacts(dataset, calibration, evidence)


def main() -> None:
    print(run_phase6_native_belief_calibration().evidence.to_json())


if __name__ == "__main__":
    main()
