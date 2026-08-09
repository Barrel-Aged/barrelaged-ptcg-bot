"""Persistent latent game-plan representation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class PlanLatent:
    """Detached, serializable plan memory for one match at one turn."""

    match_id: str
    created_turn: int
    last_updated_turn: int
    update_count: int
    vector: tuple[float, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.match_id.strip():
            raise ValueError("match_id must be non-empty")
        if self.created_turn < 0 or self.last_updated_turn < self.created_turn:
            raise ValueError("plan turn metadata is inconsistent")
        if self.update_count < 0:
            raise ValueError("update_count cannot be negative")
        if not self.vector:
            raise ValueError("plan vector cannot be empty")
        if not all(math.isfinite(value) for value in self.vector):
            raise ValueError("plan vector must contain only finite values")

    @property
    def plan_dim(self) -> int:
        return len(self.vector)

    def to_tensor(
        self,
        *,
        batch_dimension: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        tensor = torch.tensor(self.vector, dtype=dtype, device=device)
        return tensor.unsqueeze(0) if batch_dimension else tensor

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "created_turn": self.created_turn,
            "last_updated_turn": self.last_updated_turn,
            "update_count": self.update_count,
            "vector": list(self.vector),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PlanLatent":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            match_id=str(raw["match_id"]),
            created_turn=int(raw["created_turn"]),
            last_updated_turn=int(raw["last_updated_turn"]),
            update_count=int(raw["update_count"]),
            vector=tuple(float(value) for value in raw["vector"]),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            separators=None if indent is not None else (",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, encoded: str) -> "PlanLatent":
        return cls.from_dict(json.loads(encoded))

    @classmethod
    def capture(
        cls,
        tensor: torch.Tensor,
        *,
        match_id: str,
        created_turn: int,
        last_updated_turn: int,
        update_count: int,
        batch_index: int = 0,
    ) -> "PlanLatent":
        if tensor.ndim == 2:
            tensor = tensor[batch_index]
        if tensor.ndim != 1:
            raise ValueError("a plan snapshot must be captured from [P] or [B, P]")
        vector = tuple(float(value) for value in tensor.detach().cpu().tolist())
        return cls(
            match_id=match_id,
            created_turn=created_turn,
            last_updated_turn=last_updated_turn,
            update_count=update_count,
            vector=vector,
        )


class PlanMemory(nn.Module):
    """Recurrently updates a branch-local plan latent from a state embedding."""

    def __init__(self, state_embedding_dim: int, plan_dim: int) -> None:
        super().__init__()
        if state_embedding_dim <= 0 or plan_dim <= 0:
            raise ValueError("plan-memory dimensions must be positive")
        self.state_embedding_dim = state_embedding_dim
        self.plan_dim = plan_dim
        self.initial_projection = nn.Linear(state_embedding_dim, plan_dim)
        self.update_cell = nn.GRUCell(state_embedding_dim, plan_dim)

    def forward(
        self,
        state_embedding: torch.Tensor,
        previous_plan: torch.Tensor | None = None,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_embedding.ndim != 2 or state_embedding.shape[-1] != self.state_embedding_dim:
            raise ValueError(
                f"state_embedding must have shape [B, {self.state_embedding_dim}]"
            )

        initialized = torch.tanh(self.initial_projection(state_embedding))
        if previous_plan is None:
            return initialized
        if previous_plan.shape != (state_embedding.shape[0], self.plan_dim):
            raise ValueError(
                f"previous_plan must have shape [B, {self.plan_dim}]"
            )

        updated = self.update_cell(state_embedding, previous_plan)
        if reset_mask is None:
            return updated
        if reset_mask.shape != (state_embedding.shape[0],):
            raise ValueError("reset_mask must have shape [B]")
        return torch.where(reset_mask.to(dtype=torch.bool).unsqueeze(-1), initialized, updated)
