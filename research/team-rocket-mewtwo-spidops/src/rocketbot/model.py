"""Integrated strategic agents through Phase 5 search-guided learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .belief_model import BeliefCalibrator, BeliefModelOutput, OpponentBeliefModel
from .beliefs import OPPONENT_INTENTS, RESOURCE_CATEGORIES, OpponentBeliefState
from .features import STATE_FEATURE_NAMES
from .observation import ACTION_HISTORY_FEATURE_NAMES
from .plan import PlanLatent, PlanMemory
from .policy import NeuralAutoregressiveTacticalPolicy
from .strategy import (
    MetaValueOutput,
    StateConditionedMetaValue,
    StrategicAxis,
    StrategicValueHeads,
)


SEARCH_UNCERTAINTY_NAMES: tuple[str, ...] = (
    "scenario_entropy",
    "proposal_entropy",
    "chosen_value_stddev",
    "plan_margin",
)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    state_feature_dim: int = len(STATE_FEATURE_NAMES)
    state_embedding_dim: int = 64
    plan_dim: int = 32
    strategic_hidden_dim: int = 64
    meta_hidden_dim: int = 64
    candidate_feature_dim: int = 16
    policy_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if min(
            self.state_feature_dim,
            self.state_embedding_dim,
            self.plan_dim,
            self.strategic_hidden_dim,
            self.meta_hidden_dim,
            self.candidate_feature_dim,
            self.policy_hidden_dim,
        ) <= 0:
            raise ValueError("all agent dimensions must be positive")


class StateEncoder(nn.Module):
    """Compact Phase 1 baseline behind a replaceable embedding boundary."""

    def __init__(self, feature_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"features must have shape [B, {self.feature_dim}]")
        return self.network(features)


@dataclass(slots=True)
class StrategicEvaluation:
    state_embedding: torch.Tensor
    plan: torch.Tensor
    head_values: dict[StrategicAxis, torch.Tensor]
    meta_value: torch.Tensor
    meta_weights: torch.Tensor

    def capture_plan(
        self,
        *,
        match_id: str,
        created_turn: int,
        last_updated_turn: int,
        update_count: int,
        batch_index: int = 0,
    ) -> PlanLatent:
        return PlanLatent.capture(
            self.plan,
            match_id=match_id,
            created_turn=created_turn,
            last_updated_turn=last_updated_turn,
            update_count=update_count,
            batch_index=batch_index,
        )


class Phase1StrategicAgent(nn.Module):
    """State encoder, persistent plan, six critics, aggregator, and policy seam."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()
        cfg = self.config
        self.state_encoder = StateEncoder(cfg.state_feature_dim, cfg.state_embedding_dim)
        self.plan_memory = PlanMemory(cfg.state_embedding_dim, cfg.plan_dim)
        strategic_input_dim = cfg.state_embedding_dim + cfg.plan_dim
        self.value_heads = StrategicValueHeads(
            strategic_input_dim,
            cfg.strategic_hidden_dim,
        )
        self.meta_value = StateConditionedMetaValue(
            cfg.state_embedding_dim,
            cfg.plan_dim,
            cfg.meta_hidden_dim,
        )
        self.tactical_policy = NeuralAutoregressiveTacticalPolicy(
            cfg.state_embedding_dim,
            cfg.plan_dim,
            cfg.candidate_feature_dim,
            cfg.policy_hidden_dim,
        )

    def forward(
        self,
        state_features: torch.Tensor,
        previous_plan: torch.Tensor | None = None,
        *,
        reset_mask: torch.Tensor | None = None,
    ) -> StrategicEvaluation:
        state_embedding = self.state_encoder(state_features)
        plan = self.plan_memory(
            state_embedding,
            previous_plan,
            reset_mask=reset_mask,
        )
        strategic_embedding = torch.cat((state_embedding, plan), dim=-1)
        values = self.value_heads(strategic_embedding)
        aggregate: MetaValueOutput = self.meta_value(state_embedding, plan, values)
        return StrategicEvaluation(
            state_embedding=state_embedding,
            plan=plan,
            head_values=values,
            meta_value=aggregate.value,
            meta_weights=aggregate.weights,
        )


@dataclass(frozen=True, slots=True)
class Phase2AgentConfig(AgentConfig):
    """Phase 1 dimensions plus opponent-history and belief capacity."""

    action_history_feature_dim: int = len(ACTION_HISTORY_FEATURE_NAMES)
    history_hidden_dim: int = 32
    belief_embedding_dim: int = 32

    def __post_init__(self) -> None:
        AgentConfig.__post_init__(self)
        if min(
            self.action_history_feature_dim,
            self.history_hidden_dim,
            self.belief_embedding_dim,
        ) <= 0:
            raise ValueError("Phase 2 belief dimensions must be positive")

    @property
    def belief_context_dim(self) -> int:
        return len(OPPONENT_INTENTS) + len(RESOURCE_CATEGORIES)


@dataclass(slots=True)
class BeliefAwareEvaluation:
    raw_state_embedding: torch.Tensor
    state_embedding: torch.Tensor
    belief: BeliefModelOutput
    plan: torch.Tensor
    head_values: dict[StrategicAxis, torch.Tensor]
    meta_value: torch.Tensor
    meta_weights: torch.Tensor
    search_uncertainty: torch.Tensor

    def capture_plan(
        self,
        *,
        match_id: str,
        created_turn: int,
        last_updated_turn: int,
        update_count: int,
        batch_index: int = 0,
    ) -> PlanLatent:
        return PlanLatent.capture(
            self.plan,
            match_id=match_id,
            created_turn=created_turn,
            last_updated_turn=last_updated_turn,
            update_count=update_count,
            batch_index=batch_index,
        )

    def capture_belief(
        self,
        *,
        match_id: str,
        observed_through_turn: int,
        batch_index: int = 0,
    ) -> OpponentBeliefState:
        return self.belief.capture(
            match_id=match_id,
            observed_through_turn=observed_through_turn,
            batch_index=batch_index,
        )


class Phase2BeliefAwareAgent(nn.Module):
    """Condition plan memory, values, meta-value, and tactics on beliefs."""

    def __init__(self, config: Phase2AgentConfig | None = None) -> None:
        super().__init__()
        self.config = config or Phase2AgentConfig()
        cfg = self.config
        self.state_encoder = StateEncoder(cfg.state_feature_dim, cfg.state_embedding_dim)
        self.opponent_belief_model = OpponentBeliefModel(
            cfg.state_embedding_dim,
            cfg.action_history_feature_dim,
            cfg.history_hidden_dim,
            cfg.belief_embedding_dim,
        )
        self.belief_conditioner = nn.Sequential(
            nn.Linear(
                cfg.state_embedding_dim + cfg.belief_context_dim,
                cfg.state_embedding_dim,
            ),
            nn.LayerNorm(cfg.state_embedding_dim),
            nn.SiLU(),
        )
        self.plan_memory = PlanMemory(cfg.state_embedding_dim, cfg.plan_dim)
        strategic_input_dim = cfg.state_embedding_dim + cfg.plan_dim
        self.value_heads = StrategicValueHeads(
            strategic_input_dim,
            cfg.strategic_hidden_dim,
        )
        self.meta_value = StateConditionedMetaValue(
            cfg.state_embedding_dim,
            cfg.plan_dim,
            cfg.meta_hidden_dim,
        )
        self.search_uncertainty_head = nn.Sequential(
            nn.Linear(
                cfg.state_embedding_dim + cfg.plan_dim + cfg.belief_context_dim,
                cfg.strategic_hidden_dim,
            ),
            nn.SiLU(),
            nn.Linear(cfg.strategic_hidden_dim, len(SEARCH_UNCERTAINTY_NAMES)),
            nn.Softplus(),
        )
        self.tactical_policy = NeuralAutoregressiveTacticalPolicy(
            cfg.state_embedding_dim,
            cfg.plan_dim,
            cfg.candidate_feature_dim,
            cfg.policy_hidden_dim,
            belief_context_dim=cfg.belief_context_dim,
        )

    def forward(
        self,
        state_features: torch.Tensor,
        action_history: torch.Tensor,
        history_mask: torch.Tensor,
        previous_plan: torch.Tensor | None = None,
        *,
        reset_mask: torch.Tensor | None = None,
        calibrator: BeliefCalibrator | None = None,
    ) -> BeliefAwareEvaluation:
        raw_state_embedding = self.state_encoder(state_features)
        belief = self.opponent_belief_model(
            raw_state_embedding,
            action_history,
            history_mask,
            calibrator=calibrator,
        )
        belief_delta = self.belief_conditioner(
            torch.cat((raw_state_embedding, belief.context_vector), dim=-1)
        )
        state_embedding = raw_state_embedding + belief_delta
        plan = self.plan_memory(
            state_embedding,
            previous_plan,
            reset_mask=reset_mask,
        )
        strategic_embedding = torch.cat((state_embedding, plan), dim=-1)
        values = self.value_heads(strategic_embedding)
        aggregate: MetaValueOutput = self.meta_value(state_embedding, plan, values)
        search_uncertainty = self.search_uncertainty_head(
            torch.cat((state_embedding, plan, belief.context_vector), dim=-1)
        )
        return BeliefAwareEvaluation(
            raw_state_embedding=raw_state_embedding,
            state_embedding=state_embedding,
            belief=belief,
            plan=plan,
            head_values=values,
            meta_value=aggregate.value,
            meta_weights=aggregate.weights,
            search_uncertainty=search_uncertainty,
        )

    def initial_tactical_hidden(
        self,
        evaluation: BeliefAwareEvaluation,
        *,
        belief_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Start tactical decoding with calibrated beliefs or an explicit ablation."""

        context = evaluation.belief.context_vector if belief_context is None else belief_context
        return self.tactical_policy.initial_hidden(
            evaluation.state_embedding,
            evaluation.plan,
            context,
        )
