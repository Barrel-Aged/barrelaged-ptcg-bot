"""Autoregressive tactical action policy contracts and reference network."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn

from .state import GameState


class ActionComponent(str, Enum):
    ACTION_KIND = "action_kind"
    SOURCE = "source"
    TARGET = "target"
    MODE = "mode"
    AMOUNT = "amount"
    END = "end"


@dataclass(frozen=True, slots=True)
class ActionToken:
    """One semantic selection in a variable-length action sequence."""

    component: ActionComponent
    choice_id: str
    label: str

    def __post_init__(self) -> None:
        if not self.choice_id.strip() or not self.label.strip():
            raise ValueError("action token choice_id and label must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "component": self.component.value,
            "choice_id": self.choice_id,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActionToken":
        return cls(
            component=ActionComponent(str(raw["component"])),
            choice_id=str(raw["choice_id"]),
            label=str(raw["label"]),
        )


@dataclass(frozen=True, slots=True)
class ActionPrefix:
    """Serializable selections made so far by the tactical decoder."""

    tokens: tuple[ActionToken, ...] = ()

    def __post_init__(self) -> None:
        end_indices = [
            index
            for index, token in enumerate(self.tokens)
            if token.component is ActionComponent.END
        ]
        if end_indices and end_indices != [len(self.tokens) - 1]:
            raise ValueError("END may appear only once as the final token")

    @property
    def is_complete(self) -> bool:
        return bool(self.tokens and self.tokens[-1].component is ActionComponent.END)

    def append(self, token: ActionToken) -> "ActionPrefix":
        if self.is_complete:
            raise ValueError("cannot append to a completed action prefix")
        return ActionPrefix(self.tokens + (token,))

    def to_json(self) -> str:
        return json.dumps(
            [token.to_dict() for token in self.tokens],
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, encoded: str) -> "ActionPrefix":
        return cls(tuple(ActionToken.from_dict(raw) for raw in json.loads(encoded)))


@dataclass(frozen=True, slots=True)
class TacticalCandidate:
    """Adapter-produced candidate with fixed-width numeric features."""

    token: ActionToken
    features: tuple[float, ...]
    legal: bool = True

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("candidate features cannot be empty")
        if not all(math.isfinite(value) for value in self.features):
            raise ValueError("candidate features must be finite")


class LegalActionProvider(Protocol):
    """Future simulator adapter seam; Phase 1 does not implement card rules."""

    def next_candidates(
        self,
        state: GameState,
        prefix: ActionPrefix,
    ) -> Sequence[TacticalCandidate]: ...


@dataclass(slots=True)
class CandidateBatch:
    """Padded candidate features and legality for one autoregressive step."""

    features: torch.Tensor
    legal_mask: torch.Tensor

    def validate(self, *, feature_dim: int | None = None) -> None:
        if self.features.ndim != 3:
            raise ValueError("candidate features must have shape [B, N, C]")
        if self.legal_mask.shape != self.features.shape[:2]:
            raise ValueError("legal_mask must have shape [B, N]")
        if self.legal_mask.dtype is not torch.bool:
            raise ValueError("legal_mask must have bool dtype")
        if feature_dim is not None and self.features.shape[-1] != feature_dim:
            raise ValueError(f"candidate feature dimension must be {feature_dim}")
        if not torch.all(self.legal_mask.any(dim=-1)).item():
            raise ValueError("each batch row must contain at least one legal candidate")

    @classmethod
    def from_single_step(
        cls,
        candidates: Sequence[TacticalCandidate],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "CandidateBatch":
        if not candidates:
            raise ValueError("candidate list cannot be empty")
        feature_dim = len(candidates[0].features)
        if any(len(candidate.features) != feature_dim for candidate in candidates):
            raise ValueError("all candidates must share a feature dimension")
        batch = cls(
            features=torch.tensor(
                [[candidate.features for candidate in candidates]],
                dtype=dtype,
                device=device,
            ),
            legal_mask=torch.tensor(
                [[candidate.legal for candidate in candidates]],
                dtype=torch.bool,
                device=device,
            ),
        )
        batch.validate(feature_dim=feature_dim)
        return batch


class AutoregressiveTacticalPolicy(nn.Module, ABC):
    """Simulator-neutral interface for variable-length tactical actions."""

    @abstractmethod
    def initial_hidden(
        self,
        state_embedding: torch.Tensor,
        plan: torch.Tensor,
        belief_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create decoder state before the first action component.

        Phase 2 implementations may require a belief context. Passing `None`
        selects an explicit zero-belief ablation rather than an implicit guess.
        """

    @abstractmethod
    def score_next(
        self,
        hidden: torch.Tensor,
        candidates: CandidateBatch,
    ) -> torch.Tensor:
        """Return masked logits `[B, N]` for the next legal candidate set."""

    @abstractmethod
    def advance(
        self,
        hidden: torch.Tensor,
        chosen_candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        """Condition the decoder on a selected candidate before the next step."""


class NeuralAutoregressiveTacticalPolicy(AutoregressiveTacticalPolicy):
    """Reference recurrent decoder with dynamic candidate scoring."""

    def __init__(
        self,
        state_embedding_dim: int,
        plan_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        belief_context_dim: int = 0,
    ) -> None:
        super().__init__()
        if min(state_embedding_dim, plan_dim, candidate_feature_dim, hidden_dim) <= 0:
            raise ValueError("policy dimensions must be positive")
        if belief_context_dim < 0:
            raise ValueError("belief_context_dim cannot be negative")
        self.state_embedding_dim = state_embedding_dim
        self.plan_dim = plan_dim
        self.candidate_feature_dim = candidate_feature_dim
        self.hidden_dim = hidden_dim
        self.belief_context_dim = belief_context_dim
        self.context_projection = nn.Sequential(
            nn.Linear(state_embedding_dim + plan_dim + belief_context_dim, hidden_dim),
            nn.Tanh(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def initial_hidden(
        self,
        state_embedding: torch.Tensor,
        plan: torch.Tensor,
        belief_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state_embedding.ndim != 2 or state_embedding.shape[-1] != self.state_embedding_dim:
            raise ValueError(
                f"state_embedding must have shape [B, {self.state_embedding_dim}]"
            )
        if plan.shape != (state_embedding.shape[0], self.plan_dim):
            raise ValueError(f"plan must have shape [B, {self.plan_dim}]")
        if belief_context is None:
            belief_context = state_embedding.new_zeros(
                state_embedding.shape[0],
                self.belief_context_dim,
            )
        expected_belief_shape = (state_embedding.shape[0], self.belief_context_dim)
        if belief_context.shape != expected_belief_shape:
            raise ValueError(
                f"belief_context must have shape {expected_belief_shape}"
            )
        return self.context_projection(
            torch.cat((state_embedding, plan, belief_context), dim=-1)
        )

    def score_next(
        self,
        hidden: torch.Tensor,
        candidates: CandidateBatch,
    ) -> torch.Tensor:
        candidates.validate(feature_dim=self.candidate_feature_dim)
        if hidden.shape != (candidates.features.shape[0], self.hidden_dim):
            raise ValueError(f"hidden must have shape [B, {self.hidden_dim}]")
        encoded = self.candidate_encoder(candidates.features)
        query = self.query_projection(hidden).unsqueeze(1)
        logits = (encoded * query).sum(dim=-1) / math.sqrt(self.hidden_dim)
        return logits.masked_fill(~candidates.legal_mask, float("-inf"))

    def advance(
        self,
        hidden: torch.Tensor,
        chosen_candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"hidden must have shape [B, {self.hidden_dim}]")
        expected = (hidden.shape[0], self.candidate_feature_dim)
        if chosen_candidate_features.shape != expected:
            raise ValueError(f"chosen candidate features must have shape {expected}")
        encoded = self.candidate_encoder(chosen_candidate_features)
        return self.decoder_cell(encoded, hidden)
