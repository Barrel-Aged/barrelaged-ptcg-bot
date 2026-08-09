"""Neural opponent-intent/resource model, calibration, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .beliefs import (
    OPPONENT_INTENTS,
    RESOURCE_CATEGORIES,
    OpponentBeliefState,
    OpponentPlanPosterior,
)


class MaskedActionHistoryEncoder(nn.Module):
    """A GRUCell that leaves hidden state unchanged on padded timesteps."""

    def __init__(self, action_feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if action_feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("history encoder dimensions must be positive")
        self.action_feature_dim = action_feature_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(action_feature_dim, hidden_dim)

    def forward(
        self,
        action_history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        if action_history.ndim != 3 or action_history.shape[-1] != self.action_feature_dim:
            raise ValueError(
                f"action_history must have shape [B, T, {self.action_feature_dim}]"
            )
        if history_mask.shape != action_history.shape[:2]:
            raise ValueError("history_mask must have shape [B, T]")
        if history_mask.dtype != torch.bool:
            raise ValueError("history_mask must have bool dtype")

        hidden = action_history.new_zeros(action_history.shape[0], self.hidden_dim)
        for time_index in range(action_history.shape[1]):
            candidate = self.cell(action_history[:, time_index], hidden)
            hidden = torch.where(
                history_mask[:, time_index].unsqueeze(-1),
                candidate,
                hidden,
            )
        return hidden


@dataclass(frozen=True, slots=True)
class BeliefCalibrator:
    """Held-out scalar temperature calibration for plan and resource logits."""

    plan_temperature: float = 1.0
    resource_temperature: float = 1.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.plan_temperature) or self.plan_temperature <= 0.0:
            raise ValueError("plan_temperature must be finite and positive")
        if (
            not math.isfinite(self.resource_temperature)
            or self.resource_temperature <= 0.0
        ):
            raise ValueError("resource_temperature must be finite and positive")

    @classmethod
    def fit(
        cls,
        plan_logits: torch.Tensor,
        plan_labels: torch.Tensor,
        resource_logits: torch.Tensor,
        resource_targets: torch.Tensor,
        *,
        resource_mask: torch.Tensor | None = None,
        candidate_temperatures: Sequence[float] | None = None,
    ) -> "BeliefCalibrator":
        if plan_logits.ndim != 2 or plan_logits.shape[-1] != len(OPPONENT_INTENTS):
            raise ValueError("plan_logits have the wrong shape")
        if plan_labels.shape != (plan_logits.shape[0],):
            raise ValueError("plan_labels must have shape [B]")
        if resource_logits.shape != resource_targets.shape:
            raise ValueError("resource logits and targets must share a shape")
        if resource_logits.ndim != 2 or resource_logits.shape[-1] != len(RESOURCE_CATEGORIES):
            raise ValueError("resource logits have the wrong shape")
        if resource_mask is not None and resource_mask.shape != resource_targets.shape:
            raise ValueError("resource_mask must match resource_targets")

        candidates = tuple(candidate_temperatures or (
            0.25,
            0.35,
            0.5,
            0.7,
            0.85,
            1.0,
            1.2,
            1.5,
            2.0,
            2.75,
            4.0,
        ))
        if 1.0 not in candidates:
            candidates = candidates + (1.0,)
        if any(not math.isfinite(value) or value <= 0.0 for value in candidates):
            raise ValueError("candidate temperatures must be finite and positive")

        with torch.no_grad():
            plan_losses = [
                float(F.cross_entropy(plan_logits / temperature, plan_labels).item())
                for temperature in candidates
            ]
            plan_temperature = candidates[min(range(len(candidates)), key=plan_losses.__getitem__)]

            resource_losses: list[float] = []
            for temperature in candidates:
                elementwise = F.binary_cross_entropy_with_logits(
                    resource_logits / temperature,
                    resource_targets,
                    reduction="none",
                )
                if resource_mask is None:
                    loss = elementwise.mean()
                else:
                    denominator = resource_mask.sum().clamp_min(1)
                    loss = (elementwise * resource_mask).sum() / denominator
                resource_losses.append(float(loss.item()))
            resource_temperature = candidates[
                min(range(len(candidates)), key=resource_losses.__getitem__)
            ]
        return cls(
            plan_temperature=float(plan_temperature),
            resource_temperature=float(resource_temperature),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "schema_version": self.schema_version,
            "plan_temperature": self.plan_temperature,
            "resource_temperature": self.resource_temperature,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BeliefCalibrator":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            plan_temperature=float(raw["plan_temperature"]),
            resource_temperature=float(raw["resource_temperature"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, encoded: str) -> "BeliefCalibrator":
        return cls.from_dict(json.loads(encoded))


@dataclass(slots=True)
class BeliefModelOutput:
    raw_plan_logits: torch.Tensor
    raw_resource_logits: torch.Tensor
    plan_logits: torch.Tensor
    resource_logits: torch.Tensor
    plan_probabilities: torch.Tensor
    resource_probabilities: torch.Tensor
    belief_embedding: torch.Tensor
    history_embedding: torch.Tensor

    @property
    def context_vector(self) -> torch.Tensor:
        return torch.cat((self.plan_probabilities, self.resource_probabilities), dim=-1)

    def capture(
        self,
        *,
        match_id: str,
        observed_through_turn: int,
        batch_index: int = 0,
    ) -> OpponentBeliefState:
        plan = tuple(
            float(value)
            for value in self.plan_probabilities[batch_index].detach().cpu().tolist()
        )
        resources = tuple(
            float(value)
            for value in self.resource_probabilities[batch_index].detach().cpu().tolist()
        )
        return OpponentBeliefState(
            match_id=match_id,
            observed_through_turn=observed_through_turn,
            plan=OpponentPlanPosterior(plan),
            resource_probabilities=resources,
        )


class OpponentBeliefModel(nn.Module):
    """Infer strategic intent and hidden resource categories from state/history."""

    def __init__(
        self,
        state_embedding_dim: int,
        action_feature_dim: int,
        history_hidden_dim: int,
        belief_embedding_dim: int,
    ) -> None:
        super().__init__()
        if min(
            state_embedding_dim,
            action_feature_dim,
            history_hidden_dim,
            belief_embedding_dim,
        ) <= 0:
            raise ValueError("belief-model dimensions must be positive")
        self.state_embedding_dim = state_embedding_dim
        self.action_feature_dim = action_feature_dim
        self.history_hidden_dim = history_hidden_dim
        self.belief_embedding_dim = belief_embedding_dim
        self.context_dim = len(OPPONENT_INTENTS) + len(RESOURCE_CATEGORIES)
        self.history_encoder = MaskedActionHistoryEncoder(
            action_feature_dim,
            history_hidden_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(state_embedding_dim + history_hidden_dim, belief_embedding_dim),
            nn.LayerNorm(belief_embedding_dim),
            nn.SiLU(),
            nn.Linear(belief_embedding_dim, belief_embedding_dim),
            nn.SiLU(),
        )
        self.plan_head = nn.Linear(belief_embedding_dim, len(OPPONENT_INTENTS))
        self.resource_head = nn.Linear(belief_embedding_dim, len(RESOURCE_CATEGORIES))

    def forward(
        self,
        state_embedding: torch.Tensor,
        action_history: torch.Tensor,
        history_mask: torch.Tensor,
        *,
        calibrator: BeliefCalibrator | None = None,
    ) -> BeliefModelOutput:
        if state_embedding.ndim != 2 or state_embedding.shape[-1] != self.state_embedding_dim:
            raise ValueError(
                f"state_embedding must have shape [B, {self.state_embedding_dim}]"
            )
        if action_history.shape[0] != state_embedding.shape[0]:
            raise ValueError("state and history batch sizes must match")
        history_embedding = self.history_encoder(action_history, history_mask)
        belief_embedding = self.fusion(
            torch.cat((state_embedding, history_embedding), dim=-1)
        )
        raw_plan_logits = self.plan_head(belief_embedding)
        raw_resource_logits = self.resource_head(belief_embedding)
        fitted = calibrator or BeliefCalibrator()
        plan_logits = raw_plan_logits / fitted.plan_temperature
        resource_logits = raw_resource_logits / fitted.resource_temperature
        return BeliefModelOutput(
            raw_plan_logits=raw_plan_logits,
            raw_resource_logits=raw_resource_logits,
            plan_logits=plan_logits,
            resource_logits=resource_logits,
            plan_probabilities=torch.softmax(plan_logits, dim=-1),
            resource_probabilities=torch.sigmoid(resource_logits),
            belief_embedding=belief_embedding,
            history_embedding=history_embedding,
        )


@dataclass(frozen=True, slots=True)
class BeliefMetrics:
    plan_nll: float
    plan_brier: float
    plan_accuracy: float
    plan_ece: float
    resource_bce: float
    resource_brier: float
    resource_accuracy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "plan_nll": self.plan_nll,
            "plan_brier": self.plan_brier,
            "plan_accuracy": self.plan_accuracy,
            "plan_ece": self.plan_ece,
            "resource_bce": self.resource_bce,
            "resource_brier": self.resource_brier,
            "resource_accuracy": self.resource_accuracy,
        }


def evaluate_beliefs(
    output: BeliefModelOutput,
    plan_labels: torch.Tensor,
    resource_targets: torch.Tensor,
    *,
    resource_mask: torch.Tensor | None = None,
    calibration_bins: int = 10,
) -> BeliefMetrics:
    """Evaluate plan calibration and revealed-resource prediction quality."""

    plan_probabilities = output.plan_probabilities
    resource_probabilities = output.resource_probabilities
    if plan_labels.shape != (plan_probabilities.shape[0],):
        raise ValueError("plan_labels must have shape [B]")
    if resource_targets.shape != resource_probabilities.shape:
        raise ValueError("resource_targets have the wrong shape")
    if resource_mask is None:
        resource_mask = torch.ones_like(resource_targets)
    if resource_mask.shape != resource_targets.shape:
        raise ValueError("resource_mask has the wrong shape")
    if calibration_bins <= 0:
        raise ValueError("calibration_bins must be positive")

    with torch.no_grad():
        selected = plan_probabilities.gather(1, plan_labels.unsqueeze(1)).squeeze(1)
        plan_nll = -torch.log(selected.clamp_min(1e-8)).mean()
        one_hot = F.one_hot(plan_labels, num_classes=len(OPPONENT_INTENTS)).float()
        plan_brier = ((plan_probabilities - one_hot) ** 2).sum(dim=-1).mean()
        confidence, prediction = plan_probabilities.max(dim=-1)
        correct = prediction.eq(plan_labels).float()
        plan_accuracy = correct.mean()
        ece = plan_probabilities.new_tensor(0.0)
        boundaries = torch.linspace(
            0.0,
            1.0,
            calibration_bins + 1,
            device=plan_probabilities.device,
        )
        for index in range(calibration_bins):
            lower = boundaries[index]
            upper = boundaries[index + 1]
            in_bin = (confidence >= lower) & (
                confidence <= upper if index == calibration_bins - 1 else confidence < upper
            )
            if in_bin.any():
                proportion = in_bin.float().mean()
                ece += proportion * (confidence[in_bin].mean() - correct[in_bin].mean()).abs()

        denominator = resource_mask.sum().clamp_min(1.0)
        resource_bce = (
            F.binary_cross_entropy(
                resource_probabilities,
                resource_targets,
                reduction="none",
            )
            * resource_mask
        ).sum() / denominator
        resource_brier = (
            ((resource_probabilities - resource_targets) ** 2) * resource_mask
        ).sum() / denominator
        resource_accuracy = (
            (
                resource_probabilities.ge(0.5).eq(resource_targets.ge(0.5)).float()
                * resource_mask
            ).sum()
            / denominator
        )
    return BeliefMetrics(
        plan_nll=float(plan_nll.item()),
        plan_brier=float(plan_brier.item()),
        plan_accuracy=float(plan_accuracy.item()),
        plan_ece=float(ece.item()),
        resource_bce=float(resource_bce.item()),
        resource_brier=float(resource_brier.item()),
        resource_accuracy=float(resource_accuracy.item()),
    )
