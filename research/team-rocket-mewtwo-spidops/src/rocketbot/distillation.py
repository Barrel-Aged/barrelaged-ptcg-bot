"""Multi-task distillation of validated Phase 5 search-teacher records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .features import StateTensorizer
from .model import Phase2BeliefAwareAgent
from .observation import ActionHistoryTensorizer
from .plans import StrategicPlanCandidate
from .policy import CandidateBatch
from .search_learning import SearchTeacherRecord, stable_fingerprint
from .strategy import STRATEGIC_AXES, stack_strategic_values


PLAN_DESCRIPTOR_NAMES: tuple[str, ...] = (
    *(f"axis_{axis.value}" for axis in STRATEGIC_AXES),
    *(f"objective_{axis.value}" for axis in STRATEGIC_AXES),
    "matchup_setup",
    "matchup_prize_race",
    "matchup_lock_disruption",
    "matchup_board_denial",
    "matchup_resource_conservation",
    "matchup_combo_assembly",
    "proposal_prior",
    "proposal_score",
    "horizon",
    "required_resource_fraction",
)


def plan_descriptor(plan: StrategicPlanCandidate) -> tuple[float, ...]:
    values = tuple(math.tanh(value) for value in plan.axis_alignment)
    objective = tuple(float(axis is plan.objective) for axis in STRATEGIC_AXES)
    matchups = tuple(math.tanh(value) for value in plan.opponent_matchups)
    context = (
        plan.prior,
        math.tanh(plan.proposal_score),
        min(plan.horizon_turns / 6.0, 1.0),
        min(len(plan.required_resources) / 6.0, 1.0),
    )
    descriptor = values + objective + matchups + context
    if len(descriptor) != len(PLAN_DESCRIPTOR_NAMES):
        raise RuntimeError("plan descriptor implementation disagrees with public schema")
    return descriptor


@dataclass(slots=True)
class PlanPolicyOutput:
    expected_value: torch.Tensor
    search_score: torch.Tensor
    uncertainty: torch.Tensor


class PlanValuePolicyHead(nn.Module):
    """Cheap student scorer for searched and audited alternative plans."""

    def __init__(
        self,
        state_embedding_dim: int,
        plan_dim: int,
        *,
        descriptor_dim: int = len(PLAN_DESCRIPTOR_NAMES),
        hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        if min(state_embedding_dim, plan_dim, descriptor_dim, hidden_dim) <= 0:
            raise ValueError("plan-policy dimensions must be positive")
        self.state_embedding_dim = state_embedding_dim
        self.plan_dim = plan_dim
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_embedding_dim + plan_dim + descriptor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state_embedding: torch.Tensor,
        plan_latent: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> PlanPolicyOutput:
        if state_embedding.ndim != 2 or state_embedding.shape[-1] != self.state_embedding_dim:
            raise ValueError(
                f"state_embedding must have shape [B, {self.state_embedding_dim}]"
            )
        if plan_latent.shape != (state_embedding.shape[0], self.plan_dim):
            raise ValueError(f"plan_latent must have shape [B, {self.plan_dim}]")
        if descriptors.ndim != 3 or descriptors.shape[0] != state_embedding.shape[0]:
            raise ValueError("plan descriptors must have shape [B, N, D]")
        if descriptors.shape[-1] != self.descriptor_dim:
            raise ValueError(f"plan descriptor dimension must be {self.descriptor_dim}")
        candidate_count = descriptors.shape[1]
        context = torch.cat((state_embedding, plan_latent), dim=-1).unsqueeze(1)
        context = context.expand(-1, candidate_count, -1)
        hidden = self.trunk(torch.cat((context, descriptors), dim=-1))
        return PlanPolicyOutput(
            expected_value=torch.tanh(self.value_head(hidden).squeeze(-1)),
            search_score=self.score_head(hidden).squeeze(-1),
            uncertainty=F.softplus(self.uncertainty_head(hidden).squeeze(-1)),
        )


class SearchGuidedStudent(nn.Module):
    """The deployable agent plus a cheap plan scorer distilled from expert search."""

    def __init__(
        self,
        agent: Phase2BeliefAwareAgent,
        *,
        plan_policy_hidden_dim: int = 48,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.plan_policy = PlanValuePolicyHead(
            agent.config.state_embedding_dim,
            agent.config.plan_dim,
            hidden_dim=plan_policy_hidden_dim,
        )

    def score_plans(
        self,
        state_embedding: torch.Tensor,
        plan_latent: torch.Tensor,
        plans: Sequence[StrategicPlanCandidate],
    ) -> PlanPolicyOutput:
        if not plans:
            raise ValueError("at least one plan is required")
        descriptors = torch.tensor(
            [[plan_descriptor(plan) for plan in plans]],
            dtype=state_embedding.dtype,
            device=state_embedding.device,
        )
        if state_embedding.shape[0] != 1:
            descriptors = descriptors.expand(state_embedding.shape[0], -1, -1)
        return self.plan_policy(state_embedding, plan_latent, descriptors)


@dataclass(frozen=True, slots=True)
class DistillationLossWeights:
    tactical_policy: float = 1.0
    plan_memory: float = 0.6
    belief_plan: float = 0.5
    belief_resources: float = 0.4
    strategic_values: float = 0.7
    meta_value: float = 0.5
    uncertainty: float = 0.5
    plan_value: float = 0.5
    plan_score: float = 0.7
    plan_uncertainty: float = 0.3
    counterfactual_ranking: float = 0.6

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values()) if hasattr(self, "__dict__") else (
            self.tactical_policy,
            self.plan_memory,
            self.belief_plan,
            self.belief_resources,
            self.strategic_values,
            self.meta_value,
            self.uncertainty,
            self.plan_value,
            self.plan_score,
            self.plan_uncertainty,
            self.counterfactual_ranking,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("distillation loss weights must be finite/non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("at least one distillation loss weight must be positive")


DISTILLATION_COMPONENTS: tuple[str, ...] = (
    "tactical_policy",
    "plan_memory",
    "belief_plan",
    "belief_resources",
    "strategic_values",
    "meta_value",
    "uncertainty",
    "plan_value",
    "plan_score",
    "plan_uncertainty",
    "counterfactual_ranking",
)


@dataclass(slots=True)
class SearchGuidedDistillationLoss:
    total: torch.Tensor
    tactical_policy: torch.Tensor
    plan_memory: torch.Tensor
    belief_plan: torch.Tensor
    belief_resources: torch.Tensor
    strategic_values: torch.Tensor
    meta_value: torch.Tensor
    uncertainty: torch.Tensor
    plan_value: torch.Tensor
    plan_score: torch.Tensor
    plan_uncertainty: torch.Tensor
    counterfactual_ranking: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": self.total.detach().item(),
            **{
                name: getattr(self, name).detach().item()
                for name in DISTILLATION_COMPONENTS
            },
        }


@dataclass(frozen=True, slots=True)
class SearchGuidedDistillationMetrics:
    total_loss: float
    tactical_token_accuracy: float
    plan_target_mse: float
    belief_plan_cross_entropy: float
    belief_resource_mae: float
    strategic_value_mae: float
    meta_value_mae: float
    uncertainty_mae: float
    plan_value_mae: float
    plan_top1_accuracy: float
    counterfactual_ranking_accuracy: float

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class DistillationCheckpointManifest:
    """Identity and evidence needed to pair model weights with their teachers."""

    dataset_fingerprint: str
    deck_fingerprint: str
    environment_fingerprints: tuple[str, ...]
    teacher_model_fingerprints: tuple[str, ...]
    student_model_fingerprint: str
    search_config_fingerprint: str
    training_seed: int
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    training_episode_ids: tuple[str, ...]
    heldout_episode_ids: tuple[str, ...]
    code_version: str = "0.5.0"
    schema_version: int = 1

    def __post_init__(self) -> None:
        fingerprints = (
            self.dataset_fingerprint,
            self.deck_fingerprint,
            self.student_model_fingerprint,
            self.search_config_fingerprint,
            *self.environment_fingerprints,
            *self.teacher_model_fingerprints,
        )
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in fingerprints
        ):
            raise ValueError("checkpoint manifest fingerprints must be lowercase SHA-256")
        if not self.environment_fingerprints or not self.teacher_model_fingerprints:
            raise ValueError("checkpoint manifest requires environment and teacher identities")
        if self.training_seed < 0 or self.optimizer_steps <= 0:
            raise ValueError("checkpoint seed/optimizer steps are invalid")
        if not all(math.isfinite(value) and value >= 0.0 for value in (
            self.initial_loss,
            self.final_loss,
        )):
            raise ValueError("checkpoint losses must be finite/non-negative")
        if not self.training_episode_ids or not self.heldout_episode_ids:
            raise ValueError("checkpoint manifest requires train and held-out episode IDs")
        if set(self.training_episode_ids) & set(self.heldout_episode_ids):
            raise ValueError("training and held-out episodes must be disjoint")
        if not self.code_version.strip():
            raise ValueError("checkpoint code version must be non-empty")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "environment_fingerprints": list(self.environment_fingerprints),
            "teacher_model_fingerprints": list(self.teacher_model_fingerprints),
            "student_model_fingerprint": self.student_model_fingerprint,
            "search_config_fingerprint": self.search_config_fingerprint,
            "training_seed": self.training_seed,
            "optimizer_steps": self.optimizer_steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "training_episode_ids": list(self.training_episode_ids),
            "heldout_episode_ids": list(self.heldout_episode_ids),
            "code_version": self.code_version,
        }

    @property
    def checkpoint_id(self) -> str:
        return stable_fingerprint(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint_id": self.checkpoint_id, **self._payload()}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DistillationCheckpointManifest":
        manifest = cls(
            schema_version=int(raw.get("schema_version", 1)),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            environment_fingerprints=tuple(
                str(value) for value in raw["environment_fingerprints"]
            ),
            teacher_model_fingerprints=tuple(
                str(value) for value in raw["teacher_model_fingerprints"]
            ),
            student_model_fingerprint=str(raw["student_model_fingerprint"]),
            search_config_fingerprint=str(raw["search_config_fingerprint"]),
            training_seed=int(raw["training_seed"]),
            optimizer_steps=int(raw["optimizer_steps"]),
            initial_loss=float(raw["initial_loss"]),
            final_loss=float(raw["final_loss"]),
            training_episode_ids=tuple(str(value) for value in raw["training_episode_ids"]),
            heldout_episode_ids=tuple(str(value) for value in raw["heldout_episode_ids"]),
            code_version=str(raw.get("code_version", "0.5.0")),
        )
        if "checkpoint_id" in raw and str(raw["checkpoint_id"]) != manifest.checkpoint_id:
            raise ValueError("checkpoint manifest fingerprint is inconsistent")
        return manifest

    @classmethod
    def from_json(cls, encoded: str) -> "DistillationCheckpointManifest":
        return cls.from_dict(json.loads(encoded))


class SearchGuidedDistiller:
    """Train every roadmap-named subsystem from admitted search records."""

    def __init__(
        self,
        student: SearchGuidedStudent,
        *,
        weights: DistillationLossWeights | None = None,
        max_history_events: int = 32,
    ) -> None:
        if max_history_events <= 0:
            raise ValueError("max_history_events must be positive")
        self.student = student
        self.weights = weights or DistillationLossWeights()
        self.max_history_events = max_history_events
        self.state_tensorizer = StateTensorizer()
        self.history_tensorizer = ActionHistoryTensorizer()

    def _evaluation(self, record: SearchTeacherRecord):
        device = next(self.student.parameters()).device
        state = self.state_tensorizer(
            [record.observation.public_state],
            perspective=record.observation.observer,
            device=device,
        )
        history, mask = self.history_tensorizer(
            [record.observation],
            max_events=self.max_history_events,
            device=device,
        )
        return self.student.agent(state, history, mask)

    def _record_losses(
        self,
        record: SearchTeacherRecord,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        evaluation = self._evaluation(record)
        device = evaluation.plan.device
        dtype = evaluation.plan.dtype

        hidden = self.student.agent.initial_tactical_hidden(evaluation)
        tactical_losses: list[torch.Tensor] = []
        tactical_correct = 0
        for teacher_step in record.tactical_steps:
            candidate_batch = CandidateBatch(
                features=torch.tensor(
                    [[candidate.features for candidate in teacher_step.candidates]],
                    dtype=dtype,
                    device=device,
                ),
                legal_mask=torch.tensor(
                    [[candidate.legal for candidate in teacher_step.candidates]],
                    dtype=torch.bool,
                    device=device,
                ),
            )
            logits = self.student.agent.tactical_policy.score_next(hidden, candidate_batch)
            target = torch.tensor([teacher_step.selected_index], device=device)
            tactical_losses.append(F.cross_entropy(logits, target))
            tactical_correct += int(logits.argmax(dim=-1).item() == teacher_step.selected_index)
            hidden = self.student.agent.tactical_policy.advance(
                hidden,
                candidate_batch.features[:, teacher_step.selected_index],
            )
        tactical_policy = torch.stack(tactical_losses).mean()

        plan_target = torch.tensor(
            [record.plan_target],
            dtype=dtype,
            device=device,
        )
        plan_memory = F.mse_loss(evaluation.plan, plan_target)

        belief_plan_target = torch.tensor(
            [record.belief_plan_target],
            dtype=dtype,
            device=device,
        )
        belief_plan = -(
            belief_plan_target * F.log_softmax(evaluation.belief.raw_plan_logits, dim=-1)
        ).sum(dim=-1).mean()
        belief_resource_target = torch.tensor(
            [record.belief_resource_target],
            dtype=dtype,
            device=device,
        )
        belief_resources = F.binary_cross_entropy_with_logits(
            evaluation.belief.raw_resource_logits,
            belief_resource_target,
        )

        predicted_strategic = stack_strategic_values(evaluation.head_values)
        strategic_target = torch.tensor(
            [record.strategic_value_target],
            dtype=dtype,
            device=device,
        )
        strategic_values = F.mse_loss(predicted_strategic, strategic_target)
        meta_target = torch.tensor(
            [[record.meta_value_target]],
            dtype=dtype,
            device=device,
        )
        meta_value = F.mse_loss(evaluation.meta_value, meta_target)
        uncertainty_target = torch.tensor(
            [record.uncertainty_target],
            dtype=dtype,
            device=device,
        )
        uncertainty = F.smooth_l1_loss(
            evaluation.search_uncertainty,
            uncertainty_target,
        )

        plan_lookup = {
            plan_evaluation.plan.plan_id: plan_evaluation
            for plan_evaluation in record.search_result.plan_evaluations
        }
        accepted = record.accepted_counterfactuals
        candidates = (record.search_result.chosen.plan,) + tuple(
            plan_lookup[target.plan_id].plan for target in accepted
        )
        descriptors = torch.tensor(
            [[plan_descriptor(plan) for plan in candidates]],
            dtype=dtype,
            device=device,
        )
        plan_output = self.student.plan_policy(
            evaluation.state_embedding,
            evaluation.plan,
            descriptors,
        )
        value_targets = torch.tensor(
            [[record.search_result.chosen.expected_value]
             + [target.expected_value for target in accepted]],
            dtype=dtype,
            device=device,
        )
        score_targets = torch.tensor(
            [[record.search_result.chosen.search_score]
             + [target.search_score for target in accepted]],
            dtype=dtype,
            device=device,
        )
        uncertainty_targets = torch.tensor(
            [[record.search_result.chosen.value_stddev]
             + [target.value_stddev for target in accepted]],
            dtype=dtype,
            device=device,
        )
        reliability = torch.tensor(
            [[1.0] + [target.reliability_weight for target in accepted]],
            dtype=dtype,
            device=device,
        )
        normalizer = reliability.sum().clamp_min(1e-8)
        plan_value = (
            reliability * (plan_output.expected_value - value_targets).square()
        ).sum() / normalizer
        plan_score = (
            reliability * (plan_output.search_score - score_targets).square()
        ).sum() / normalizer
        plan_uncertainty = (
            reliability * F.smooth_l1_loss(
                plan_output.uncertainty,
                uncertainty_targets,
                reduction="none",
            )
        ).sum() / normalizer
        ranking_losses = []
        for index, target in enumerate(accepted, start=1):
            margin = min(max(target.score_gap, 0.02), 0.25)
            ranking_losses.append(
                F.relu(
                    plan_output.search_score.new_tensor(margin)
                    - (plan_output.search_score[0, 0] - plan_output.search_score[0, index])
                ) * target.reliability_weight
            )
        counterfactual_ranking = torch.stack(ranking_losses).mean()

        losses = {
            "tactical_policy": tactical_policy,
            "plan_memory": plan_memory,
            "belief_plan": belief_plan,
            "belief_resources": belief_resources,
            "strategic_values": strategic_values,
            "meta_value": meta_value,
            "uncertainty": uncertainty,
            "plan_value": plan_value,
            "plan_score": plan_score,
            "plan_uncertainty": plan_uncertainty,
            "counterfactual_ranking": counterfactual_ranking,
        }
        diagnostics = {
            "tactical_correct": float(tactical_correct),
            "tactical_count": float(len(record.tactical_steps)),
            "plan_target_squared_error": float(
                (evaluation.plan.detach() - plan_target).square().mean().item()
            ),
            "belief_plan_cross_entropy": float(belief_plan.detach().item()),
            "belief_resource_absolute_error": float(
                (
                    evaluation.belief.resource_probabilities.detach()
                    - belief_resource_target
                ).abs().mean().item()
            ),
            "strategic_absolute_error": float(
                (predicted_strategic.detach() - strategic_target).abs().mean().item()
            ),
            "meta_absolute_error": float(
                (evaluation.meta_value.detach() - meta_target).abs().mean().item()
            ),
            "uncertainty_absolute_error": float(
                (
                    evaluation.search_uncertainty.detach() - uncertainty_target
                ).abs().mean().item()
            ),
            "plan_value_absolute_error": float(
                (plan_output.expected_value.detach() - value_targets).abs().mean().item()
            ),
            "plan_top1_correct": float(
                plan_output.search_score.detach().argmax(dim=-1).item() == 0
            ),
            "ranking_correct": float(
                all(
                    plan_output.search_score.detach()[0, 0]
                    > plan_output.search_score.detach()[0, index]
                    for index in range(1, plan_output.search_score.shape[1])
                )
            ),
        }
        return losses, diagnostics

    def loss(
        self,
        records: Sequence[SearchTeacherRecord],
    ) -> SearchGuidedDistillationLoss:
        if not records:
            raise ValueError("at least one search-teacher record is required")
        weighted: dict[str, torch.Tensor] = {}
        total_weight = sum(record.sample_weight for record in records)
        for record in records:
            losses, _ = self._record_losses(record)
            for name, value in losses.items():
                weighted[name] = weighted.get(name, value.new_tensor(0.0)) + (
                    record.sample_weight * value
                )
        averaged = {name: value / total_weight for name, value in weighted.items()}
        total = sum(
            getattr(self.weights, name) * averaged[name]
            for name in DISTILLATION_COMPONENTS
        )
        return SearchGuidedDistillationLoss(total=total, **averaged)

    def evaluate(
        self,
        records: Sequence[SearchTeacherRecord],
    ) -> SearchGuidedDistillationMetrics:
        if not records:
            raise ValueError("at least one search-teacher record is required")
        self.student.eval()
        diagnostics: list[dict[str, float]] = []
        with torch.no_grad():
            loss = self.loss(records)
            for record in records:
                _, record_diagnostics = self._record_losses(record)
                diagnostics.append(record_diagnostics)
        tactical_correct = sum(item["tactical_correct"] for item in diagnostics)
        tactical_count = sum(item["tactical_count"] for item in diagnostics)

        def mean(name: str) -> float:
            return sum(item[name] for item in diagnostics) / len(diagnostics)

        return SearchGuidedDistillationMetrics(
            total_loss=loss.total.detach().item(),
            tactical_token_accuracy=tactical_correct / max(tactical_count, 1.0),
            plan_target_mse=mean("plan_target_squared_error"),
            belief_plan_cross_entropy=mean("belief_plan_cross_entropy"),
            belief_resource_mae=mean("belief_resource_absolute_error"),
            strategic_value_mae=mean("strategic_absolute_error"),
            meta_value_mae=mean("meta_absolute_error"),
            uncertainty_mae=mean("uncertainty_absolute_error"),
            plan_value_mae=mean("plan_value_absolute_error"),
            plan_top1_accuracy=mean("plan_top1_correct"),
            counterfactual_ranking_accuracy=mean("ranking_correct"),
        )

    def fit(
        self,
        records: Sequence[SearchTeacherRecord],
        *,
        steps: int = 200,
        learning_rate: float = 0.01,
        gradient_clip: float = 5.0,
    ) -> tuple[float, ...]:
        if not records or steps <= 0:
            raise ValueError("distillation fit requires records and positive steps")
        if learning_rate <= 0.0 or gradient_clip <= 0.0:
            raise ValueError("distillation optimizer settings must be positive")
        optimizer = torch.optim.Adam(self.student.parameters(), lr=learning_rate)
        trajectory: list[float] = []
        self.student.train()
        for _ in range(steps):
            loss = self.loss(records)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), gradient_clip)
            optimizer.step()
            trajectory.append(loss.total.detach().item())
        return tuple(trajectory)
