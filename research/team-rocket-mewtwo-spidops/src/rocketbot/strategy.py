"""Inspectable strategic values and their state-conditioned aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch
from torch import nn


class StrategicAxis(str, Enum):
    SETUP = "setup"
    TEMPO = "tempo"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    PRIZE_CONTROL = "prize_control"
    BOARD_CONTROL = "board_control"
    OPTIONALITY = "optionality"


STRATEGIC_AXES: tuple[StrategicAxis, ...] = tuple(StrategicAxis)


class StrategicValueHeads(nn.Module):
    """Six independent scalar heads with a fixed, public ordering."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("strategic-head dimensions must be positive")
        self.input_dim = input_dim
        self.heads = nn.ModuleDict(
            {
                axis.value: nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Tanh(),
                )
                for axis in STRATEGIC_AXES
            }
        )

    def forward(self, strategic_embedding: torch.Tensor) -> dict[StrategicAxis, torch.Tensor]:
        if strategic_embedding.ndim != 2 or strategic_embedding.shape[-1] != self.input_dim:
            raise ValueError(f"strategic_embedding must have shape [B, {self.input_dim}]")
        return {
            axis: self.heads[axis.value](strategic_embedding)
            for axis in STRATEGIC_AXES
        }


def stack_strategic_values(
    values: Mapping[StrategicAxis, torch.Tensor],
) -> torch.Tensor:
    """Stack a public value mapping into the fixed `[B, 6]` axis order."""

    missing = [axis for axis in STRATEGIC_AXES if axis not in values]
    if missing:
        raise ValueError(f"missing strategic values: {missing}")
    return torch.cat([values[axis] for axis in STRATEGIC_AXES], dim=-1)


@dataclass(slots=True)
class MetaValueOutput:
    value: torch.Tensor
    weights: torch.Tensor


class StateConditionedMetaValue(nn.Module):
    """Learns a convex, state-and-plan-conditioned combination of six values."""

    def __init__(self, state_embedding_dim: int, plan_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if min(state_embedding_dim, plan_dim, hidden_dim) <= 0:
            raise ValueError("meta-value dimensions must be positive")
        self.state_embedding_dim = state_embedding_dim
        self.plan_dim = plan_dim
        self.gate = nn.Sequential(
            nn.Linear(state_embedding_dim + plan_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(STRATEGIC_AXES)),
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        plan: torch.Tensor,
        values: Mapping[StrategicAxis, torch.Tensor] | torch.Tensor,
    ) -> MetaValueOutput:
        if state_embedding.ndim != 2 or state_embedding.shape[-1] != self.state_embedding_dim:
            raise ValueError(
                f"state_embedding must have shape [B, {self.state_embedding_dim}]"
            )
        if plan.shape != (state_embedding.shape[0], self.plan_dim):
            raise ValueError(f"plan must have shape [B, {self.plan_dim}]")
        stacked = stack_strategic_values(values) if isinstance(values, Mapping) else values
        expected = (state_embedding.shape[0], len(STRATEGIC_AXES))
        if stacked.shape != expected:
            raise ValueError(f"strategic values must have shape {expected}")

        weights = torch.softmax(self.gate(torch.cat((state_embedding, plan), dim=-1)), dim=-1)
        aggregate = (weights * stacked).sum(dim=-1, keepdim=True)
        return MetaValueOutput(value=aggregate, weights=weights)
