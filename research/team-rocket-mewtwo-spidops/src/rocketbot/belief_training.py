"""Trajectory supervision, batching, and losses for Phase 2 beliefs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from .belief_model import BeliefModelOutput
from .beliefs import (
    OPPONENT_INTENTS,
    RESOURCE_CATEGORIES,
    OpponentIntent,
    ResourceCategory,
)
from .features import StateTensorizer
from .observation import ActionHistoryTensorizer, AgentObservation


@dataclass(frozen=True, slots=True)
class BeliefSupervision:
    """One trajectory prefix plus labels learned from later revealed outcomes."""

    observation: AgentObservation
    revealed_intent: OpponentIntent
    resource_targets: tuple[float, ...]
    resource_known_mask: tuple[float, ...]
    source: str = "trajectory_reveal"
    schema_version: int = 1

    def __post_init__(self) -> None:
        expected = len(RESOURCE_CATEGORIES)
        if len(self.resource_targets) != expected or len(self.resource_known_mask) != expected:
            raise ValueError(f"resource targets and masks must each contain {expected} values")
        if any(value not in (0.0, 1.0) for value in self.resource_targets):
            raise ValueError("resource targets must be binary")
        if any(value not in (0.0, 1.0) for value in self.resource_known_mask):
            raise ValueError("resource known-mask values must be binary")
        if not self.source.strip():
            raise ValueError("supervision source must be non-empty")

    @classmethod
    def from_reveals(
        cls,
        observation: AgentObservation,
        *,
        revealed_intent: OpponentIntent,
        present: Iterable[ResourceCategory] = (),
        absent: Iterable[ResourceCategory] = (),
        source: str = "trajectory_reveal",
    ) -> "BeliefSupervision":
        present_set = set(present)
        absent_set = set(absent)
        overlap = present_set & absent_set
        if overlap:
            raise ValueError(f"resource categories cannot be both present and absent: {overlap}")
        targets = tuple(float(category in present_set) for category in RESOURCE_CATEGORIES)
        mask = tuple(
            float(category in present_set or category in absent_set)
            for category in RESOURCE_CATEGORIES
        )
        return cls(
            observation=observation,
            revealed_intent=revealed_intent,
            resource_targets=targets,
            resource_known_mask=mask,
            source=source,
        )

    @classmethod
    def fully_labeled(
        cls,
        observation: AgentObservation,
        *,
        revealed_intent: OpponentIntent,
        present: Iterable[ResourceCategory] = (),
        source: str = "fully_revealed_trajectory",
    ) -> "BeliefSupervision":
        present_set = set(present)
        return cls(
            observation=observation,
            revealed_intent=revealed_intent,
            resource_targets=tuple(
                float(category in present_set) for category in RESOURCE_CATEGORIES
            ),
            resource_known_mask=(1.0,) * len(RESOURCE_CATEGORIES),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observation": self.observation.to_dict(),
            "revealed_intent": self.revealed_intent.value,
            "resource_targets": {
                category.value: self.resource_targets[index]
                for index, category in enumerate(RESOURCE_CATEGORIES)
            },
            "resource_known_mask": {
                category.value: self.resource_known_mask[index]
                for index, category in enumerate(RESOURCE_CATEGORIES)
            },
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BeliefSupervision":
        targets = raw["resource_targets"]
        mask = raw["resource_known_mask"]
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            observation=AgentObservation.from_dict(raw["observation"]),
            revealed_intent=OpponentIntent(str(raw["revealed_intent"])),
            resource_targets=tuple(
                float(targets[category.value]) for category in RESOURCE_CATEGORIES
            ),
            resource_known_mask=tuple(
                float(mask[category.value]) for category in RESOURCE_CATEGORIES
            ),
            source=str(raw.get("source", "trajectory_reveal")),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryBeliefDataset:
    examples: tuple[BeliefSupervision, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("belief dataset cannot be empty")

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "examples": [example.to_dict() for example in self.examples],
            },
            ensure_ascii=False,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, encoded: str) -> "TrajectoryBeliefDataset":
        raw = json.loads(encoded)
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            examples=tuple(
                BeliefSupervision.from_dict(item) for item in raw["examples"]
            ),
        )


@dataclass(slots=True)
class BeliefTrainingBatch:
    state_features: torch.Tensor
    action_history: torch.Tensor
    history_mask: torch.Tensor
    plan_labels: torch.Tensor
    resource_targets: torch.Tensor
    resource_known_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "BeliefTrainingBatch":
        return BeliefTrainingBatch(
            state_features=self.state_features.to(device),
            action_history=self.action_history.to(device),
            history_mask=self.history_mask.to(device),
            plan_labels=self.plan_labels.to(device),
            resource_targets=self.resource_targets.to(device),
            resource_known_mask=self.resource_known_mask.to(device),
        )


class BeliefBatchBuilder:
    """Collate public trajectory prefixes and later reveal labels."""

    def __init__(self, *, max_history_events: int = 32) -> None:
        if max_history_events <= 0:
            raise ValueError("max_history_events must be positive")
        self.max_history_events = max_history_events
        self.state_tensorizer = StateTensorizer()
        self.history_tensorizer = ActionHistoryTensorizer()

    def __call__(
        self,
        examples: Sequence[BeliefSupervision],
        *,
        device: torch.device | str | None = None,
    ) -> BeliefTrainingBatch:
        if not examples:
            raise ValueError("at least one supervision example is required")
        state_rows = [
            self.state_tensorizer(
                [example.observation.public_state],
                perspective=example.observation.observer,
                device=device,
            )
            for example in examples
        ]
        state_features = torch.cat(state_rows, dim=0)
        action_history, history_mask = self.history_tensorizer(
            [example.observation for example in examples],
            max_events=self.max_history_events,
            device=device,
        )
        plan_labels = torch.tensor(
            [OPPONENT_INTENTS.index(example.revealed_intent) for example in examples],
            dtype=torch.long,
            device=device,
        )
        resource_targets = torch.tensor(
            [example.resource_targets for example in examples],
            dtype=torch.float32,
            device=device,
        )
        resource_known_mask = torch.tensor(
            [example.resource_known_mask for example in examples],
            dtype=torch.float32,
            device=device,
        )
        return BeliefTrainingBatch(
            state_features=state_features,
            action_history=action_history,
            history_mask=history_mask,
            plan_labels=plan_labels,
            resource_targets=resource_targets,
            resource_known_mask=resource_known_mask,
        )


@dataclass(slots=True)
class BeliefLoss:
    total: torch.Tensor
    plan: torch.Tensor
    resources: torch.Tensor


def belief_supervision_loss(
    output: BeliefModelOutput,
    plan_labels: torch.Tensor,
    resource_targets: torch.Tensor,
    resource_known_mask: torch.Tensor,
    *,
    plan_weight: float = 1.0,
    resource_weight: float = 1.0,
) -> BeliefLoss:
    """Train on intent outcomes and only those private resources later revealed."""

    if plan_labels.shape != (output.raw_plan_logits.shape[0],):
        raise ValueError("plan_labels must have shape [B]")
    if resource_targets.shape != output.raw_resource_logits.shape:
        raise ValueError("resource_targets have the wrong shape")
    if resource_known_mask.shape != resource_targets.shape:
        raise ValueError("resource_known_mask has the wrong shape")
    if plan_weight < 0.0 or resource_weight < 0.0:
        raise ValueError("loss weights cannot be negative")

    plan_loss = F.cross_entropy(output.raw_plan_logits, plan_labels)
    elementwise = F.binary_cross_entropy_with_logits(
        output.raw_resource_logits,
        resource_targets,
        reduction="none",
    )
    known = resource_known_mask.sum()
    if known.item() == 0:
        resource_loss = elementwise.sum() * 0.0
    else:
        resource_loss = (elementwise * resource_known_mask).sum() / known
    total = plan_weight * plan_loss + resource_weight * resource_loss
    return BeliefLoss(total=total, plan=plan_loss, resources=resource_loss)
