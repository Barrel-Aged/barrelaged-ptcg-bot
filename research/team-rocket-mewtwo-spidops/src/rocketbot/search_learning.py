"""Validated search-teacher collection and self-play orchestration for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import torch

from .beliefs import OPPONENT_INTENTS, RESOURCE_CATEGORIES, OpponentBeliefState
from .features import StateTensorizer
from .model import (
    SEARCH_UNCERTAINTY_NAMES,
    BeliefAwareEvaluation,
    Phase2BeliefAwareAgent,
)
from .observation import ActionHistoryTensorizer, AgentObservation
from .plans import StrategicPlanCandidate
from .policy import ActionPrefix, ActionToken, TacticalCandidate
from .search import PlanExecutionAdapter, PlanSearchResult
from .search_gate import (
    DecisionMode,
    RoutineDecision,
    SearchComputeBank,
    SearchGateDataset,
    SearchGateFeatures,
    SearchGateSupervision,
    SelectiveDecision,
    SelectiveSearchPolicy,
)
from .strategy import STRATEGIC_AXES


def stable_fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible metadata."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def model_fingerprint(module: torch.nn.Module) -> str:
    """Hash parameter names, shapes, dtypes, and exact tensor bytes."""

    digest = sha256()
    for name, tensor in sorted(module.state_dict().items()):
        materialized = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(materialized.dtype).encode("ascii"))
        digest.update(json.dumps(list(materialized.shape)).encode("ascii"))
        digest.update(materialized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _validate_fingerprint(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 fingerprint")


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Reproducibility identity attached to every admitted teacher root."""

    deck_fingerprint: str
    environment_fingerprint: str
    teacher_model_fingerprint: str
    search_config_fingerprint: str
    random_seed: int
    episode_id: str
    code_version: str = "0.5.0"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value, label in (
            (self.deck_fingerprint, "deck_fingerprint"),
            (self.environment_fingerprint, "environment_fingerprint"),
            (self.teacher_model_fingerprint, "teacher_model_fingerprint"),
            (self.search_config_fingerprint, "search_config_fingerprint"),
        ):
            _validate_fingerprint(value, label)
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        if not self.episode_id.strip() or not self.code_version.strip():
            raise ValueError("episode_id and code_version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "deck_fingerprint": self.deck_fingerprint,
            "environment_fingerprint": self.environment_fingerprint,
            "teacher_model_fingerprint": self.teacher_model_fingerprint,
            "search_config_fingerprint": self.search_config_fingerprint,
            "random_seed": self.random_seed,
            "episode_id": self.episode_id,
            "code_version": self.code_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArtifactProvenance":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            environment_fingerprint=str(raw["environment_fingerprint"]),
            teacher_model_fingerprint=str(raw["teacher_model_fingerprint"]),
            search_config_fingerprint=str(raw["search_config_fingerprint"]),
            random_seed=int(raw["random_seed"]),
            episode_id=str(raw["episode_id"]),
            code_version=str(raw.get("code_version", "0.5.0")),
        )


@dataclass(frozen=True, slots=True)
class OutcomeValidation:
    """Independent paired evidence that the search choice beat the routine choice."""

    source: str
    routine_return: float
    search_return: float
    confidence: float
    paired_samples: int
    terminal_verified: bool
    environment_fingerprint: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("outcome-validation source must be non-empty")
        if not all(math.isfinite(value) for value in (self.routine_return, self.search_return)):
            raise ValueError("paired outcome returns must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("outcome-validation confidence must lie in [0, 1]")
        if self.paired_samples <= 0:
            raise ValueError("paired outcome evidence requires at least one sample")
        _validate_fingerprint(self.environment_fingerprint, "environment_fingerprint")

    @property
    def validated_gain(self) -> float:
        return self.search_return - self.routine_return

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "routine_return": self.routine_return,
            "search_return": self.search_return,
            "confidence": self.confidence,
            "paired_samples": self.paired_samples,
            "terminal_verified": self.terminal_verified,
            "environment_fingerprint": self.environment_fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OutcomeValidation":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            source=str(raw["source"]),
            routine_return=float(raw["routine_return"]),
            search_return=float(raw["search_return"]),
            confidence=float(raw["confidence"]),
            paired_samples=int(raw["paired_samples"]),
            terminal_verified=bool(raw["terminal_verified"]),
            environment_fingerprint=str(raw["environment_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.detail.strip():
            raise ValueError("audit check name/detail must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AuditCheck":
        return cls(str(raw["name"]), bool(raw["passed"]), str(raw["detail"]))


@dataclass(frozen=True, slots=True)
class TeacherCandidate:
    token: ActionToken
    features: tuple[float, ...]
    legal: bool = True

    def __post_init__(self) -> None:
        if not self.features or not all(math.isfinite(value) for value in self.features):
            raise ValueError("teacher candidate features must be finite and non-empty")

    @classmethod
    def capture(cls, candidate: TacticalCandidate) -> "TeacherCandidate":
        return cls(candidate.token, candidate.features, candidate.legal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token.to_dict(),
            "features": list(self.features),
            "legal": self.legal,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TeacherCandidate":
        return cls(
            token=ActionToken.from_dict(raw["token"]),
            features=tuple(float(value) for value in raw["features"]),
            legal=bool(raw.get("legal", True)),
        )


@dataclass(frozen=True, slots=True)
class TacticalTeacherStep:
    plan_step_id: str
    sequence_index: int
    candidates: tuple[TeacherCandidate, ...]
    selected_index: int

    def __post_init__(self) -> None:
        if not self.plan_step_id.strip() or self.sequence_index < 0:
            raise ValueError("tactical teacher step identity is invalid")
        if not self.candidates or not 0 <= self.selected_index < len(self.candidates):
            raise ValueError("tactical teacher selection index is invalid")
        dimensions = {len(candidate.features) for candidate in self.candidates}
        if len(dimensions) != 1:
            raise ValueError("teacher candidates must share a feature dimension")
        if not self.candidates[self.selected_index].legal:
            raise ValueError("teacher cannot select an illegal candidate")
        if not any(candidate.legal for candidate in self.candidates):
            raise ValueError("teacher candidate set requires a legal action")

    @property
    def selected(self) -> TeacherCandidate:
        return self.candidates[self.selected_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_step_id": self.plan_step_id,
            "sequence_index": self.sequence_index,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_index": self.selected_index,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TacticalTeacherStep":
        return cls(
            plan_step_id=str(raw["plan_step_id"]),
            sequence_index=int(raw["sequence_index"]),
            candidates=tuple(
                TeacherCandidate.from_dict(candidate) for candidate in raw["candidates"]
            ),
            selected_index=int(raw["selected_index"]),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualTarget:
    plan_id: str
    objective: str
    rank: int
    expected_strategic_values: tuple[float, ...]
    expected_value: float
    search_score: float
    value_stddev: float
    score_gap: float
    reliability_weight: float
    checks: tuple[AuditCheck, ...]

    def __post_init__(self) -> None:
        if not self.plan_id.strip() or not self.objective.strip() or self.rank <= 0:
            raise ValueError("counterfactual identity/rank are invalid")
        if len(self.expected_strategic_values) != len(STRATEGIC_AXES):
            raise ValueError("counterfactual strategic target must contain six values")
        numeric = self.expected_strategic_values + (
            self.expected_value,
            self.search_score,
            self.value_stddev,
            self.score_gap,
            self.reliability_weight,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("counterfactual targets must be finite")
        if self.value_stddev < 0.0 or self.score_gap < 0.0:
            raise ValueError("counterfactual uncertainty/gap cannot be negative")
        if not 0.0 <= self.reliability_weight <= 1.0 or not self.checks:
            raise ValueError("counterfactual reliability/checks are invalid")

    @property
    def accepted(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "objective": self.objective,
            "rank": self.rank,
            "expected_strategic_values": list(self.expected_strategic_values),
            "expected_value": self.expected_value,
            "search_score": self.search_score,
            "value_stddev": self.value_stddev,
            "score_gap": self.score_gap,
            "reliability_weight": self.reliability_weight,
            "checks": [check.to_dict() for check in self.checks],
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CounterfactualTarget":
        target = cls(
            plan_id=str(raw["plan_id"]),
            objective=str(raw["objective"]),
            rank=int(raw["rank"]),
            expected_strategic_values=tuple(
                float(value) for value in raw["expected_strategic_values"]
            ),
            expected_value=float(raw["expected_value"]),
            search_score=float(raw["search_score"]),
            value_stddev=float(raw["value_stddev"]),
            score_gap=float(raw["score_gap"]),
            reliability_weight=float(raw["reliability_weight"]),
            checks=tuple(AuditCheck.from_dict(check) for check in raw["checks"]),
        )
        if "accepted" in raw and bool(raw["accepted"]) is not target.accepted:
            raise ValueError("serialized counterfactual disposition is inconsistent")
        return target


class DistillationDisposition(str, Enum):
    DISTILL_TO_ROUTINE = "distill_to_routine"
    RETAIN_EXPERT_SEARCH = "retain_expert_search"


def encode_strategic_plan(
    plan: StrategicPlanCandidate,
    plan_dim: int,
) -> tuple[float, ...]:
    """Encode a searched symbolic plan into the persistent latent target space."""

    if plan_dim <= 0:
        raise ValueError("plan_dim must be positive")
    values = [math.tanh(value) for value in plan.axis_alignment]
    values.extend(float(axis is plan.objective) for axis in STRATEGIC_AXES)
    digest = sha256(
        json.dumps(plan.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).digest()
    index = 0
    while len(values) < plan_dim:
        values.append(digest[index % len(digest)] / 127.5 - 1.0)
        index += 1
    return tuple(values[:plan_dim])


@dataclass(frozen=True, slots=True)
class SearchTeacherRecord:
    """One admitted matched-root teacher example with all Phase 5 targets."""

    example_id: str
    observation: AgentObservation
    routine: RoutineDecision
    routine_plan_id: str | None
    search_result: PlanSearchResult
    teacher_action: ActionPrefix
    teacher_action_consensus: float
    tactical_steps: tuple[TacticalTeacherStep, ...]
    plan_target: tuple[float, ...]
    belief_plan_target: tuple[float, ...]
    belief_resource_target: tuple[float, ...]
    strategic_value_target: tuple[float, ...]
    meta_value_target: float
    uncertainty_target: tuple[float, ...]
    gate_features: SearchGateFeatures
    compute_cost: float
    search_value_cost: float
    predicted_search_gain: float
    action_changed: bool
    plan_changed: bool
    disposition: DistillationDisposition
    counterfactuals: tuple[CounterfactualTarget, ...]
    outcome_validation: OutcomeValidation
    provenance: ArtifactProvenance
    admission_checks: tuple[AuditCheck, ...]
    sample_weight: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_fingerprint(self.example_id, "example_id")
        expected_id = stable_fingerprint(
            {
                "observation_digest": self.observation.digest,
                "episode_id": self.provenance.episode_id,
                "teacher_model": self.provenance.teacher_model_fingerprint,
                "search_config": self.provenance.search_config_fingerprint,
            }
        )
        if self.example_id != expected_id:
            raise ValueError("teacher example ID disagrees with its root/provenance")
        if self.observation.public_state.match_id != self.search_result.match_id:
            raise ValueError("teacher observation and search result must share a match")
        if self.routine_plan_id is not None and not self.routine_plan_id.strip():
            raise ValueError("routine_plan_id must be non-empty when provided")
        expected_plan_changed = (
            self.routine_plan_id is not None
            and self.routine_plan_id != self.search_result.chosen_plan_id
        )
        if self.plan_changed is not expected_plan_changed:
            raise ValueError("plan_changed disagrees with routine/teacher plan IDs")
        if self.action_changed is not (self.teacher_action != self.routine.action_prefix):
            raise ValueError("action_changed disagrees with routine/teacher prefixes")
        if not self.teacher_action.is_complete or not self.tactical_steps:
            raise ValueError("teacher action/steps must be complete and non-empty")
        if tuple(step.sequence_index for step in self.tactical_steps) != tuple(
            range(len(self.tactical_steps))
        ):
            raise ValueError("tactical teacher sequence indices must be contiguous")
        first_step_id = self.tactical_steps[0].plan_step_id
        captured_first_action = ActionPrefix(
            tuple(
                step.selected.token
                for step in self.tactical_steps
                if step.plan_step_id == first_step_id
            )
        )
        if captured_first_action != self.teacher_action:
            raise ValueError("teacher action disagrees with captured tactical targets")
        if not 0.0 <= self.teacher_action_consensus <= 1.0:
            raise ValueError("teacher action consensus must lie in [0, 1]")
        action_masses: dict[str, float] = {}
        for branch in self.search_result.chosen.branches:
            key = branch.rollout.executed_steps[0].prefix.to_json()
            action_masses[key] = action_masses.get(key, 0.0) + branch.scenario_probability
        modal_key = max(action_masses, key=action_masses.__getitem__)
        if (
            self.teacher_action.to_json() != modal_key
            or not math.isclose(
                self.teacher_action_consensus,
                action_masses[modal_key],
                abs_tol=1e-8,
            )
        ):
            raise ValueError("teacher action/consensus disagree with search branches")
        if not self.plan_target or not all(math.isfinite(value) for value in self.plan_target):
            raise ValueError("plan target must be finite and non-empty")
        if self.plan_target != encode_strategic_plan(
            self.search_result.chosen.plan,
            len(self.plan_target),
        ):
            raise ValueError("plan latent target disagrees with the chosen symbolic plan")
        if len(self.belief_plan_target) != len(OPPONENT_INTENTS):
            raise ValueError("belief plan target has the wrong dimension")
        if not math.isclose(sum(self.belief_plan_target), 1.0, abs_tol=1e-6):
            raise ValueError("belief plan target must sum to one")
        if len(self.belief_resource_target) != len(RESOURCE_CATEGORIES):
            raise ValueError("belief resource target has the wrong dimension")
        if any(not 0.0 <= value <= 1.0 for value in self.belief_resource_target):
            raise ValueError("belief resource targets must lie in [0, 1]")
        expected_plan_belief = [0.0] * len(OPPONENT_INTENTS)
        expected_resource_belief = [0.0] * len(RESOURCE_CATEGORIES)
        for evidence in self.search_result.scenario_evidence:
            probability = evidence.posterior_probability
            expected_plan_belief[OPPONENT_INTENTS.index(evidence.scenario.intent)] += probability
            for index, count in enumerate(evidence.scenario.resources.presence):
                expected_resource_belief[index] += probability * float(count > 0)
        if not all(
            math.isclose(actual, expected, abs_tol=1e-8)
            for actual, expected in zip(self.belief_plan_target, expected_plan_belief)
        ) or not all(
            math.isclose(actual, expected, abs_tol=1e-8)
            for actual, expected in zip(
                self.belief_resource_target,
                expected_resource_belief,
            )
        ):
            raise ValueError("belief targets disagree with search scenario posterior")
        if len(self.strategic_value_target) != len(STRATEGIC_AXES):
            raise ValueError("strategic target must contain six values")
        if len(self.uncertainty_target) != len(SEARCH_UNCERTAINTY_NAMES):
            raise ValueError("uncertainty target has the wrong dimension")
        if not all(
            math.isfinite(value)
            for value in self.strategic_value_target
            + self.uncertainty_target
            + (
                self.meta_value_target,
                self.compute_cost,
                self.search_value_cost,
                self.predicted_search_gain,
            )
        ):
            raise ValueError("teacher numeric targets must be finite")
        if (
            any(value < 0.0 for value in self.uncertainty_target)
            or self.compute_cost < 0.0
            or self.search_value_cost < 0.0
        ):
            raise ValueError("teacher uncertainty/cost cannot be negative")
        expected_gain = self.search_result.chosen.expected_value - self.routine.value_estimate
        if not math.isclose(self.predicted_search_gain, expected_gain, abs_tol=1e-8):
            raise ValueError("teacher predicted gain disagrees with matched root")
        expected_uncertainty = (
            self.search_result.uncertainty.scenario_entropy,
            self.search_result.uncertainty.proposal_entropy,
            self.search_result.uncertainty.chosen_value_stddev,
            self.search_result.uncertainty.plan_margin,
        )
        if self.uncertainty_target != expected_uncertainty:
            raise ValueError("uncertainty target disagrees with search evidence")
        if self.strategic_value_target != self.search_result.chosen.expected_strategic_values:
            raise ValueError("strategic target disagrees with the chosen plan evaluation")
        if not math.isclose(
            self.meta_value_target,
            self.search_result.chosen.expected_value,
            abs_tol=1e-8,
        ):
            raise ValueError("meta-value target disagrees with the chosen plan evaluation")
        if not (self.action_changed or self.plan_changed):
            raise ValueError("teacher record must change an action or strategic plan")
        if not self.counterfactuals or not any(target.accepted for target in self.counterfactuals):
            raise ValueError("teacher record requires an accepted audited counterfactual")
        plan_ids = [target.plan_id for target in self.counterfactuals]
        if len(plan_ids) != len(set(plan_ids)) or self.search_result.chosen_plan_id in plan_ids:
            raise ValueError("counterfactual plan IDs are invalid")
        plan_lookup = {
            evaluation.plan.plan_id: evaluation
            for evaluation in self.search_result.plan_evaluations
        }
        for target in self.counterfactuals:
            if target.plan_id not in plan_lookup:
                raise ValueError("counterfactual target is absent from search result")
            evaluation = plan_lookup[target.plan_id]
            expected_gap = max(
                self.search_result.chosen.search_score - evaluation.search_score,
                0.0,
            )
            if (
                target.expected_strategic_values
                != evaluation.expected_strategic_values
                or not math.isclose(target.expected_value, evaluation.expected_value, abs_tol=1e-8)
                or not math.isclose(target.search_score, evaluation.search_score, abs_tol=1e-8)
                or not math.isclose(target.value_stddev, evaluation.value_stddev, abs_tol=1e-8)
                or not math.isclose(target.score_gap, expected_gap, abs_tol=1e-8)
            ):
                raise ValueError("counterfactual target disagrees with search evaluation")
        if not self.admission_checks or not all(check.passed for check in self.admission_checks):
            raise ValueError("admitted teacher records require all checks to pass")
        if not 0.0 < self.sample_weight <= 1.0:
            raise ValueError("teacher sample weight must lie in (0, 1]")
        if (
            self.outcome_validation.environment_fingerprint
            != self.provenance.environment_fingerprint
        ):
            raise ValueError("outcome evidence and provenance environment disagree")
        if (
            not self.outcome_validation.terminal_verified
            or self.outcome_validation.validated_gain <= self.search_value_cost
        ):
            raise ValueError("admitted teacher lacks positive verified net utility")

    @property
    def accepted_counterfactuals(self) -> tuple[CounterfactualTarget, ...]:
        return tuple(target for target in self.counterfactuals if target.accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "observation": self.observation.to_dict(),
            "routine": self.routine.to_dict(),
            "routine_plan_id": self.routine_plan_id,
            "search_result": self.search_result.to_dict(),
            "teacher_action": [token.to_dict() for token in self.teacher_action.tokens],
            "teacher_action_consensus": self.teacher_action_consensus,
            "tactical_steps": [step.to_dict() for step in self.tactical_steps],
            "plan_target": list(self.plan_target),
            "belief_plan_target": list(self.belief_plan_target),
            "belief_resource_target": list(self.belief_resource_target),
            "strategic_value_target": list(self.strategic_value_target),
            "meta_value_target": self.meta_value_target,
            "uncertainty_target": {
                name: self.uncertainty_target[index]
                for index, name in enumerate(SEARCH_UNCERTAINTY_NAMES)
            },
            "gate_features": self.gate_features.to_dict(),
            "compute_cost": self.compute_cost,
            "search_value_cost": self.search_value_cost,
            "predicted_search_gain": self.predicted_search_gain,
            "action_changed": self.action_changed,
            "plan_changed": self.plan_changed,
            "disposition": self.disposition.value,
            "counterfactuals": [target.to_dict() for target in self.counterfactuals],
            "outcome_validation": self.outcome_validation.to_dict(),
            "provenance": self.provenance.to_dict(),
            "admission_checks": [check.to_dict() for check in self.admission_checks],
            "sample_weight": self.sample_weight,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchTeacherRecord":
        uncertainty = raw["uncertainty_target"]
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            example_id=str(raw["example_id"]),
            observation=AgentObservation.from_dict(raw["observation"]),
            routine=RoutineDecision.from_dict(raw["routine"]),
            routine_plan_id=(
                None if raw.get("routine_plan_id") is None else str(raw["routine_plan_id"])
            ),
            search_result=PlanSearchResult.from_dict(raw["search_result"]),
            teacher_action=ActionPrefix(
                tuple(ActionToken.from_dict(token) for token in raw["teacher_action"])
            ),
            teacher_action_consensus=float(raw["teacher_action_consensus"]),
            tactical_steps=tuple(
                TacticalTeacherStep.from_dict(step) for step in raw["tactical_steps"]
            ),
            plan_target=tuple(float(value) for value in raw["plan_target"]),
            belief_plan_target=tuple(float(value) for value in raw["belief_plan_target"]),
            belief_resource_target=tuple(
                float(value) for value in raw["belief_resource_target"]
            ),
            strategic_value_target=tuple(
                float(value) for value in raw["strategic_value_target"]
            ),
            meta_value_target=float(raw["meta_value_target"]),
            uncertainty_target=tuple(
                float(uncertainty[name]) for name in SEARCH_UNCERTAINTY_NAMES
            ),
            gate_features=SearchGateFeatures.from_dict(raw["gate_features"]),
            compute_cost=float(raw["compute_cost"]),
            search_value_cost=float(raw["search_value_cost"]),
            predicted_search_gain=float(raw["predicted_search_gain"]),
            action_changed=bool(raw["action_changed"]),
            plan_changed=bool(raw["plan_changed"]),
            disposition=DistillationDisposition(str(raw["disposition"])),
            counterfactuals=tuple(
                CounterfactualTarget.from_dict(target) for target in raw["counterfactuals"]
            ),
            outcome_validation=OutcomeValidation.from_dict(raw["outcome_validation"]),
            provenance=ArtifactProvenance.from_dict(raw["provenance"]),
            admission_checks=tuple(
                AuditCheck.from_dict(check) for check in raw["admission_checks"]
            ),
            sample_weight=float(raw["sample_weight"]),
        )


@dataclass(frozen=True, slots=True)
class TeacherAdmission:
    candidate_id: str
    checks: tuple[AuditCheck, ...]
    record: SearchTeacherRecord | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_fingerprint(self.candidate_id, "candidate_id")
        if not self.checks:
            raise ValueError("teacher admission requires audit checks")
        if (self.record is not None) is not all(check.passed for check in self.checks):
            raise ValueError("teacher admission record/check disposition is inconsistent")
        if self.record is not None and self.record.example_id != self.candidate_id:
            raise ValueError("teacher admission and record IDs disagree")
        if self.record is not None and self.record.admission_checks != self.checks:
            raise ValueError("teacher admission and record checks disagree")

    @property
    def accepted(self) -> bool:
        return self.record is not None

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "accepted": self.accepted,
            "checks": [check.to_dict() for check in self.checks],
            "rejection_reasons": list(self.rejection_reasons),
            "record": None if self.record is None else self.record.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TeacherAdmission":
        record = raw.get("record")
        admission = cls(
            schema_version=int(raw.get("schema_version", 1)),
            candidate_id=str(raw["candidate_id"]),
            checks=tuple(AuditCheck.from_dict(check) for check in raw["checks"]),
            record=None if record is None else SearchTeacherRecord.from_dict(record),
        )
        if "accepted" in raw and bool(raw["accepted"]) is not admission.accepted:
            raise ValueError("serialized teacher admission disposition is inconsistent")
        return admission


@dataclass(frozen=True, slots=True)
class SearchTeacherDataset:
    records: tuple[SearchTeacherRecord, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("search-teacher dataset cannot be empty")
        example_ids = [record.example_id for record in self.records]
        root_ids = [record.observation.digest for record in self.records]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("search-teacher dataset contains duplicate example IDs")
        if len(root_ids) != len(set(root_ids)):
            raise ValueError("search-teacher dataset contains duplicate public roots")

    @property
    def fingerprint(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def disposition_counts(self) -> dict[DistillationDisposition, int]:
        return {
            disposition: sum(record.disposition is disposition for record in self.records)
            for disposition in DistillationDisposition
        }

    def for_episodes(self, episode_ids: Sequence[str]) -> "SearchTeacherDataset":
        selected = tuple(
            record for record in self.records if record.provenance.episode_id in episode_ids
        )
        return SearchTeacherDataset(selected)

    def gate_dataset(self) -> SearchGateDataset:
        return SearchGateDataset(
            tuple(
                SearchGateSupervision(
                    features=record.gate_features,
                    realized_search_gain=record.outcome_validation.validated_gain,
                    search_cost=record.search_value_cost,
                    source="validated_phase5_self_play",
                )
                for record in self.records
            )
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "records": [record.to_dict() for record in self.records],
            },
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, encoded: str) -> "SearchTeacherDataset":
        raw = json.loads(encoded)
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            records=tuple(
                SearchTeacherRecord.from_dict(record) for record in raw["records"]
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchRetentionTarget:
    """Revalidate value of search after the student has absorbed a teacher root."""

    teacher_example_id: str
    disposition: DistillationDisposition
    refreshed_features: SearchGateFeatures
    distilled_routine_return: float
    expert_return: float
    search_value_cost: float
    confidence: float
    source: str = "post_distillation_paired_replay"
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_fingerprint(self.teacher_example_id, "teacher_example_id")
        if not all(
            math.isfinite(value)
            for value in (
                self.distilled_routine_return,
                self.expert_return,
                self.search_value_cost,
                self.confidence,
            )
        ):
            raise ValueError("search-retention targets must be finite")
        if self.search_value_cost < 0.0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("search-retention cost/confidence are invalid")
        if not self.source.strip():
            raise ValueError("search-retention source must be non-empty")
        if (
            self.disposition is DistillationDisposition.RETAIN_EXPERT_SEARCH
        ) is not self.should_search:
            raise ValueError("retention disposition disagrees with remaining search utility")

    @property
    def remaining_search_gain(self) -> float:
        return self.expert_return - self.distilled_routine_return

    @property
    def should_search(self) -> bool:
        return self.remaining_search_gain > self.search_value_cost

    def to_supervision(self) -> SearchGateSupervision:
        return SearchGateSupervision(
            features=self.refreshed_features,
            realized_search_gain=self.remaining_search_gain,
            search_cost=self.search_value_cost,
            source="validated_post_distillation_retention",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "teacher_example_id": self.teacher_example_id,
            "disposition": self.disposition.value,
            "refreshed_features": self.refreshed_features.to_dict(),
            "distilled_routine_return": self.distilled_routine_return,
            "expert_return": self.expert_return,
            "search_value_cost": self.search_value_cost,
            "confidence": self.confidence,
            "source": self.source,
            "remaining_search_gain": self.remaining_search_gain,
            "should_search": self.should_search,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SearchRetentionTarget":
        target = cls(
            schema_version=int(raw.get("schema_version", 1)),
            teacher_example_id=str(raw["teacher_example_id"]),
            disposition=DistillationDisposition(str(raw["disposition"])),
            refreshed_features=SearchGateFeatures.from_dict(raw["refreshed_features"]),
            distilled_routine_return=float(raw["distilled_routine_return"]),
            expert_return=float(raw["expert_return"]),
            search_value_cost=float(raw["search_value_cost"]),
            confidence=float(raw["confidence"]),
            source=str(raw.get("source", "post_distillation_paired_replay")),
        )
        if "should_search" in raw and bool(raw["should_search"]) is not target.should_search:
            raise ValueError("serialized retention decision is inconsistent")
        return target


@dataclass(frozen=True, slots=True)
class SearchTargetValidationConfig:
    min_predicted_gain: float = 0.05
    min_validated_gain: float = 0.05
    min_validation_confidence: float = 0.80
    min_paired_samples: int = 2
    min_action_consensus: float = 0.55
    max_teacher_value_stddev: float = 0.25
    max_counterfactuals: int = 2
    max_counterfactual_score_gap: float = 0.35
    max_counterfactual_stddev: float = 0.30
    retain_scenario_entropy: float = 0.40
    retain_value_stddev: float = 0.12

    def __post_init__(self) -> None:
        finite = (
            self.min_predicted_gain,
            self.min_validated_gain,
            self.min_validation_confidence,
            self.min_action_consensus,
            self.max_teacher_value_stddev,
            self.max_counterfactual_score_gap,
            self.max_counterfactual_stddev,
            self.retain_scenario_entropy,
            self.retain_value_stddev,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in finite):
            raise ValueError("search-target validation thresholds must be finite/non-negative")
        if not 0.0 <= self.min_validation_confidence <= 1.0:
            raise ValueError("minimum validation confidence must lie in [0, 1]")
        if not 0.0 <= self.min_action_consensus <= 1.0:
            raise ValueError("minimum action consensus must lie in [0, 1]")
        if self.min_paired_samples <= 0 or self.max_counterfactuals <= 0:
            raise ValueError("validation sample/counterfactual counts must be positive")


class CounterfactualAuditor:
    """Admit only alternatives evaluated on the same complete belief support."""

    def __init__(self, config: SearchTargetValidationConfig | None = None) -> None:
        self.config = config or SearchTargetValidationConfig()

    def audit(self, result: PlanSearchResult) -> tuple[CounterfactualTarget, ...]:
        chosen = result.chosen
        chosen_scenarios = {branch.scenario_id for branch in chosen.branches}
        alternatives = sorted(
            (
                evaluation
                for evaluation in result.plan_evaluations
                if evaluation.plan.plan_id != result.chosen_plan_id
            ),
            key=lambda evaluation: evaluation.search_score,
            reverse=True,
        )[: self.config.max_counterfactuals]
        audited: list[CounterfactualTarget] = []
        for rank, alternative in enumerate(alternatives, start=1):
            scenario_ids = {branch.scenario_id for branch in alternative.branches}
            probability_mass = sum(
                branch.scenario_probability for branch in alternative.branches
            )
            complete = all(
                executed.prefix.is_complete
                for branch in alternative.branches
                for executed in branch.rollout.executed_steps
            )
            score_gap = max(chosen.search_score - alternative.search_score, 0.0)
            finite_targets = all(
                math.isfinite(value)
                for value in alternative.expected_strategic_values
                + (
                    alternative.expected_value,
                    alternative.search_score,
                    alternative.value_stddev,
                )
            )
            checks = (
                AuditCheck(
                    "same_root_scenario_support",
                    scenario_ids == chosen_scenarios,
                    "Alternative and chosen plan must use identical scenario IDs.",
                ),
                AuditCheck(
                    "normalized_posterior",
                    math.isclose(probability_mass, 1.0, abs_tol=1e-6),
                    "Alternative branch posterior must sum to one.",
                ),
                AuditCheck(
                    "complete_autoregressive_rollouts",
                    complete,
                    "Every alternative branch must terminate every plan step.",
                ),
                AuditCheck(
                    "finite_counterfactual_targets",
                    finite_targets,
                    "All alternative values and uncertainties must be finite.",
                ),
                AuditCheck(
                    "bounded_counterfactual_uncertainty",
                    alternative.value_stddev <= self.config.max_counterfactual_stddev,
                    "Alternative posterior value deviation must stay within the audit bound.",
                ),
                AuditCheck(
                    "shortlist_score_gap",
                    score_gap <= self.config.max_counterfactual_score_gap,
                    "Only strategically competitive alternatives become dense labels.",
                ),
            )
            reliability = min(
                1.0,
                max(
                    0.0,
                    math.exp(-alternative.value_stddev)
                    * math.exp(-score_gap)
                    * min(max(probability_mass, 0.0), 1.0),
                ),
            )
            audited.append(
                CounterfactualTarget(
                    plan_id=alternative.plan.plan_id,
                    objective=alternative.plan.objective.value,
                    rank=rank,
                    expected_strategic_values=alternative.expected_strategic_values,
                    expected_value=alternative.expected_value,
                    search_score=alternative.search_score,
                    value_stddev=alternative.value_stddev,
                    score_gap=score_gap,
                    reliability_weight=reliability,
                    checks=checks,
                )
            )
        return tuple(audited)


class SearchTeacherCollector:
    """Validate a selected search result and derive every durable teacher target."""

    def __init__(
        self,
        adapter: PlanExecutionAdapter,
        config: SearchTargetValidationConfig | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config or SearchTargetValidationConfig()
        self.counterfactual_auditor = CounterfactualAuditor(self.config)

    @staticmethod
    def _modal_action(result: PlanSearchResult) -> tuple[ActionPrefix, float, str]:
        weighted: dict[str, float] = {}
        by_key: dict[str, ActionPrefix] = {}
        branch_keys: dict[str, str] = {}
        for branch in result.chosen.branches:
            prefix = branch.rollout.executed_steps[0].prefix
            key = prefix.to_json()
            weighted[key] = weighted.get(key, 0.0) + branch.scenario_probability
            by_key[key] = prefix
            branch_keys[branch.scenario_id] = key
        modal_key = max(weighted, key=weighted.__getitem__)
        modal_branches = [
            branch
            for branch in result.chosen.branches
            if branch_keys[branch.scenario_id] == modal_key
        ]
        representative = max(
            modal_branches,
            key=lambda branch: branch.scenario_probability,
        )
        return by_key[modal_key], weighted[modal_key], representative.scenario_id

    def _capture_tactical_steps(
        self,
        observation: AgentObservation,
        result: PlanSearchResult,
        representative_scenario_id: str,
    ) -> tuple[TacticalTeacherStep, ...]:
        plan = result.chosen.plan
        branch = next(
            branch
            for branch in result.chosen.branches
            if branch.scenario_id == representative_scenario_id
        )
        evidence = next(
            evidence
            for evidence in result.scenario_evidence
            if evidence.scenario.scenario_id == representative_scenario_id
        )
        scenario = evidence.scenario.with_prior(evidence.posterior_probability)
        rollout_state = self.adapter.start(observation, plan, scenario)
        captured: list[TacticalTeacherStep] = []
        sequence_index = 0
        for executed in branch.rollout.executed_steps:
            prefix = ActionPrefix()
            for selection in executed.selections:
                candidates = tuple(
                    self.adapter.next_candidates(
                        rollout_state,
                        plan,
                        executed.step,
                        prefix,
                        scenario,
                    )
                )
                matches = [
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.token == selection.token
                ]
                if len(matches) != 1:
                    raise ValueError("teacher rollout token cannot be uniquely reconstructed")
                captured.append(
                    TacticalTeacherStep(
                        plan_step_id=executed.step.step_id,
                        sequence_index=sequence_index,
                        candidates=tuple(
                            TeacherCandidate.capture(candidate) for candidate in candidates
                        ),
                        selected_index=matches[0],
                    )
                )
                sequence_index += 1
                prefix = prefix.append(selection.token)
            if prefix != executed.prefix or not prefix.is_complete:
                raise ValueError("reconstructed teacher prefix disagrees with search rollout")
            rollout_state = self.adapter.commit_action(
                rollout_state,
                plan,
                executed.step,
                prefix,
                scenario,
            )
        return tuple(captured)

    @staticmethod
    def _belief_targets(result: PlanSearchResult) -> tuple[tuple[float, ...], tuple[float, ...]]:
        plan = [0.0] * len(OPPONENT_INTENTS)
        resources = [0.0] * len(RESOURCE_CATEGORIES)
        for evidence in result.scenario_evidence:
            probability = evidence.posterior_probability
            plan[OPPONENT_INTENTS.index(evidence.scenario.intent)] += probability
            for index, count in enumerate(evidence.scenario.resources.presence):
                resources[index] += probability * float(count > 0)
        total = sum(plan)
        if total <= 0.0:
            raise ValueError("search posterior has no plan probability mass")
        return (
            tuple(value / total for value in plan),
            tuple(min(1.0, max(0.0, value)) for value in resources),
        )

    def collect(
        self,
        observation: AgentObservation,
        routine: RoutineDecision,
        routine_plan_id: str | None,
        decision: SelectiveDecision,
        outcome_validation: OutcomeValidation,
        provenance: ArtifactProvenance,
        *,
        plan_dim: int,
        search_value_cost: float,
    ) -> TeacherAdmission:
        if not math.isfinite(search_value_cost) or search_value_cost < 0.0:
            raise ValueError("search_value_cost must be finite and non-negative")
        candidate_id = stable_fingerprint(
            {
                "observation_digest": observation.digest,
                "episode_id": provenance.episode_id,
                "teacher_model": provenance.teacher_model_fingerprint,
                "search_config": provenance.search_config_fingerprint,
            }
        )
        checks: list[AuditCheck] = []
        result = decision.search_result
        checks.append(
            AuditCheck(
                "expert_search_present",
                decision.mode is DecisionMode.EXPERT_SEARCH and result is not None,
                "Only roots actually improved by selected expert search can teach.",
            )
        )
        if result is None:
            return TeacherAdmission(candidate_id, tuple(checks))

        predicted_gain = result.chosen.expected_value - routine.value_estimate
        checks.extend(
            (
                AuditCheck(
                    "matched_public_root",
                    observation.public_state.match_id == result.match_id,
                    "Observation and search result must refer to the same match root.",
                ),
                AuditCheck(
                    "search_artifact_roundtrip",
                    PlanSearchResult.from_json(result.to_json()) == result,
                    "The complete search teacher artifact must be canonical JSON.",
                ),
                AuditCheck(
                    "predicted_gain_margin",
                    predicted_gain >= self.config.min_predicted_gain,
                    "Search expected value must beat the routine estimate by the configured margin.",
                ),
                AuditCheck(
                    "validated_gain_margin",
                    outcome_validation.validated_gain >= self.config.min_validated_gain,
                    "Independent paired outcomes must confirm a meaningful gain.",
                ),
                AuditCheck(
                    "positive_net_search_utility",
                    outcome_validation.validated_gain > search_value_cost,
                    "Validated improvement must repay its value-equivalent search cost.",
                ),
                AuditCheck(
                    "validation_confidence",
                    outcome_validation.confidence >= self.config.min_validation_confidence,
                    "Paired outcome confidence must clear the admission threshold.",
                ),
                AuditCheck(
                    "validation_sample_count",
                    outcome_validation.paired_samples >= self.config.min_paired_samples,
                    "Paired evidence must contain enough samples.",
                ),
                AuditCheck(
                    "terminal_outcome_verified",
                    outcome_validation.terminal_verified,
                    "Outcome validator must attest terminal or complete-horizon replay.",
                ),
                AuditCheck(
                    "environment_fingerprint_match",
                    outcome_validation.environment_fingerprint
                    == provenance.environment_fingerprint,
                    "Outcome evidence must come from the recorded environment version.",
                ),
                AuditCheck(
                    "bounded_teacher_uncertainty",
                    result.chosen.value_stddev <= self.config.max_teacher_value_stddev,
                    "Teacher value variance must stay below the reliability bound.",
                ),
            )
        )

        try:
            teacher_action, consensus, representative_id = self._modal_action(result)
            tactical_steps = self._capture_tactical_steps(
                observation,
                result,
                representative_id,
            )
            reconstruction_ok = True
            reconstruction_detail = "Modal teacher sequence was reconstructed from legal candidates."
        except (RuntimeError, ValueError, StopIteration) as error:
            teacher_action = ActionPrefix()
            consensus = 0.0
            tactical_steps = ()
            reconstruction_ok = False
            reconstruction_detail = f"Teacher reconstruction failed: {error}"
        action_changed = teacher_action != routine.action_prefix
        plan_changed = (
            routine_plan_id is not None and routine_plan_id != result.chosen_plan_id
        )
        checks.extend(
            (
                AuditCheck(
                    "legal_teacher_reconstruction",
                    reconstruction_ok,
                    reconstruction_detail,
                ),
                AuditCheck(
                    "posterior_action_consensus",
                    consensus >= self.config.min_action_consensus,
                    "Modal first action must carry enough posterior branch mass.",
                ),
                AuditCheck(
                    "meaningful_action_or_plan_change",
                    action_changed or plan_changed,
                    "Search must change the routine action or strategic plan.",
                ),
            )
        )

        counterfactuals = self.counterfactual_auditor.audit(result)
        checks.append(
            AuditCheck(
                "audited_counterfactual_shortlist",
                bool(counterfactuals) and any(target.accepted for target in counterfactuals),
                "At least one same-root alternative must pass every counterfactual check.",
            )
        )
        if not all(check.passed for check in checks):
            return TeacherAdmission(candidate_id, tuple(checks))

        belief_plan, belief_resources = self._belief_targets(result)
        uncertainty = (
            result.uncertainty.scenario_entropy,
            result.uncertainty.proposal_entropy,
            result.uncertainty.chosen_value_stddev,
            result.uncertainty.plan_margin,
        )
        disposition = (
            DistillationDisposition.RETAIN_EXPERT_SEARCH
            if (
                result.uncertainty.scenario_entropy
                >= self.config.retain_scenario_entropy
                or result.chosen.value_stddev >= self.config.retain_value_stddev
            )
            else DistillationDisposition.DISTILL_TO_ROUTINE
        )
        sample_weight = min(
            1.0,
            max(
                0.05,
                outcome_validation.confidence
                * consensus
                * min(outcome_validation.validated_gain / 0.20, 1.0),
            ),
        )
        record = SearchTeacherRecord(
            example_id=candidate_id,
            observation=observation,
            routine=routine,
            routine_plan_id=routine_plan_id,
            search_result=result,
            teacher_action=teacher_action,
            teacher_action_consensus=consensus,
            tactical_steps=tactical_steps,
            plan_target=encode_strategic_plan(result.chosen.plan, plan_dim),
            belief_plan_target=belief_plan,
            belief_resource_target=belief_resources,
            strategic_value_target=result.chosen.expected_strategic_values,
            meta_value_target=result.chosen.expected_value,
            uncertainty_target=uncertainty,
            gate_features=decision.gate.features,
            compute_cost=decision.gate.search_cost,
            search_value_cost=search_value_cost,
            predicted_search_gain=predicted_gain,
            action_changed=action_changed,
            plan_changed=plan_changed,
            disposition=disposition,
            counterfactuals=counterfactuals,
            outcome_validation=outcome_validation,
            provenance=provenance,
            admission_checks=tuple(checks),
            sample_weight=sample_weight,
        )
        return TeacherAdmission(candidate_id, tuple(checks), record)


@dataclass(frozen=True, slots=True)
class SelfPlayRoot:
    observation: AgentObservation
    belief: OpponentBeliefState
    routine: RoutineDecision
    routine_plan_id: str | None
    compute_cost: float = 1.0
    search_value_cost: float = 0.08
    strategic_value_override: tuple[float, ...] | None = None
    meta_weight_override: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.observation.public_state.match_id != self.belief.match_id:
            raise ValueError("self-play root observation and belief match IDs disagree")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (self.compute_cost, self.search_value_cost)
        ):
            raise ValueError("self-play compute/value costs must be finite/non-negative")
        if self.strategic_value_override is not None and len(
            self.strategic_value_override
        ) != len(STRATEGIC_AXES):
            raise ValueError("strategic override must contain six values")
        if self.meta_weight_override is not None:
            if len(self.meta_weight_override) != len(STRATEGIC_AXES):
                raise ValueError("meta-weight override must contain six values")
            if not math.isclose(sum(self.meta_weight_override), 1.0, abs_tol=1e-6):
                raise ValueError("meta-weight override must sum to one")


class SearchGuidedSelfPlayEnvironment(Protocol):
    episode_id: str
    environment_fingerprint: str

    def reset(self, seed: int) -> None: ...

    @property
    def terminal(self) -> bool: ...

    def current_root(self) -> SelfPlayRoot: ...

    def validate(
        self,
        root: SelfPlayRoot,
        search_result: PlanSearchResult,
    ) -> OutcomeValidation: ...

    def advance(self, decision: SelectiveDecision) -> None: ...


@dataclass(frozen=True, slots=True)
class SelfPlayCollectionStatistics:
    turns: int
    routine_turns: int
    searched_turns: int
    admitted_teachers: int
    rejected_teachers: int

    def __post_init__(self) -> None:
        values = (
            self.turns,
            self.routine_turns,
            self.searched_turns,
            self.admitted_teachers,
            self.rejected_teachers,
        )
        if any(value < 0 for value in values):
            raise ValueError("self-play collection counts cannot be negative")
        if self.routine_turns + self.searched_turns != self.turns:
            raise ValueError("self-play routine/search counts must cover all turns")
        if self.admitted_teachers + self.rejected_teachers != self.searched_turns:
            raise ValueError("every searched turn must receive an admission disposition")

    def to_dict(self) -> dict[str, int]:
        return {
            "turns": self.turns,
            "routine_turns": self.routine_turns,
            "searched_turns": self.searched_turns,
            "admitted_teachers": self.admitted_teachers,
            "rejected_teachers": self.rejected_teachers,
        }


@dataclass(frozen=True, slots=True)
class SelfPlayCollectionResult:
    episode_id: str
    decisions: tuple[SelectiveDecision, ...]
    admissions: tuple[TeacherAdmission, ...]
    statistics: SelfPlayCollectionStatistics
    final_bank: SearchComputeBank

    @property
    def records(self) -> tuple[SearchTeacherRecord, ...]:
        return tuple(
            admission.record
            for admission in self.admissions
            if admission.record is not None
        )


class SearchGuidedSelfPlayRunner:
    """Run selected expert search inside an environment-owned self-play episode."""

    def __init__(
        self,
        agent: Phase2BeliefAwareAgent,
        selective_policy: SelectiveSearchPolicy,
        collector: SearchTeacherCollector,
        *,
        max_turns: int = 100,
        max_history_events: int = 32,
    ) -> None:
        if max_turns <= 0 or max_history_events <= 0:
            raise ValueError("self-play runner limits must be positive")
        self.agent = agent
        self.selective_policy = selective_policy
        self.collector = collector
        self.max_turns = max_turns
        self.max_history_events = max_history_events
        self.state_tensorizer = StateTensorizer()
        self.history_tensorizer = ActionHistoryTensorizer()

    def _evaluate(self, root: SelfPlayRoot) -> BeliefAwareEvaluation:
        state = self.state_tensorizer(
            [root.observation.public_state],
            perspective=root.observation.observer,
        )
        history, mask = self.history_tensorizer(
            [root.observation],
            max_events=self.max_history_events,
        )
        self.agent.eval()
        with torch.no_grad():
            evaluation = self.agent(state, history, mask)
            if root.strategic_value_override is not None:
                evaluation.head_values = {
                    axis: torch.tensor(
                        [[root.strategic_value_override[index]]],
                        dtype=evaluation.plan.dtype,
                        device=evaluation.plan.device,
                    )
                    for index, axis in enumerate(STRATEGIC_AXES)
                }
            if root.meta_weight_override is not None:
                evaluation.meta_weights = torch.tensor(
                    [root.meta_weight_override],
                    dtype=evaluation.plan.dtype,
                    device=evaluation.plan.device,
                )
            if root.strategic_value_override is not None:
                stacked = torch.cat(
                    [evaluation.head_values[axis] for axis in STRATEGIC_AXES],
                    dim=-1,
                )
                evaluation.meta_value = (
                    evaluation.meta_weights * stacked
                ).sum(dim=-1, keepdim=True)
        return evaluation

    def run_episode(
        self,
        environment: SearchGuidedSelfPlayEnvironment,
        bank: SearchComputeBank,
        *,
        deck_fingerprint: str,
        search_config_fingerprint: str,
        seed: int,
    ) -> SelfPlayCollectionResult:
        _validate_fingerprint(deck_fingerprint, "deck_fingerprint")
        _validate_fingerprint(search_config_fingerprint, "search_config_fingerprint")
        _validate_fingerprint(
            environment.environment_fingerprint,
            "environment_fingerprint",
        )
        environment.reset(seed)
        provenance = ArtifactProvenance(
            deck_fingerprint=deck_fingerprint,
            environment_fingerprint=environment.environment_fingerprint,
            teacher_model_fingerprint=model_fingerprint(self.agent),
            search_config_fingerprint=search_config_fingerprint,
            random_seed=seed,
            episode_id=environment.episode_id,
        )
        decisions: list[SelectiveDecision] = []
        admissions: list[TeacherAdmission] = []
        turns = routine_turns = searched_turns = admitted = rejected = 0
        current_bank = bank
        while not environment.terminal and turns < self.max_turns:
            root = environment.current_root()
            evaluation = self._evaluate(root)
            decision = self.selective_policy.decide(
                root.observation,
                evaluation,
                root.belief,
                root.routine,
                current_bank,
                search_cost=root.compute_cost,
            )
            decisions.append(decision)
            current_bank = decision.gate.bank_after
            turns += 1
            if decision.mode is DecisionMode.EXPERT_SEARCH:
                searched_turns += 1
                assert decision.search_result is not None
                evidence = environment.validate(root, decision.search_result)
                admission = self.collector.collect(
                    root.observation,
                    root.routine,
                    root.routine_plan_id,
                    decision,
                    evidence,
                    provenance,
                    plan_dim=self.agent.config.plan_dim,
                    search_value_cost=root.search_value_cost,
                )
                admissions.append(admission)
                if admission.accepted:
                    admitted += 1
                else:
                    rejected += 1
            else:
                routine_turns += 1
            environment.advance(decision)
            current_bank = current_bank.refill(1)
        if not environment.terminal:
            raise RuntimeError("self-play episode exceeded its turn limit")
        return SelfPlayCollectionResult(
            episode_id=environment.episode_id,
            decisions=tuple(decisions),
            admissions=tuple(admissions),
            statistics=SelfPlayCollectionStatistics(
                turns=turns,
                routine_turns=routine_turns,
                searched_turns=searched_turns,
                admitted_teachers=admitted,
                rejected_teachers=rejected,
            ),
            final_bank=current_bank,
        )
