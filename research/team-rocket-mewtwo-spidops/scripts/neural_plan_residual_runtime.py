"""Recurrent neural plan residual for the frozen Team Rocket blueprint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from soldier import features as F
from soldier.belief import Belief
from soldier.engine import random_legal_choice
from soldier.net import attack_safe_logits, feat_to_tensors


SCHEMA = "rocket-neural-plan-residual/v1"
STRATEGIC_AXES = (
    "setup",
    "tempo",
    "resource_efficiency",
    "prize_control",
    "board_control",
    "optionality",
)
FIXED_INITIALIZATION_SEED = 780_001


class NeuralPlanResidual(nn.Module):
    """Small persistent planner over frozen relational state/option tokens."""

    def __init__(
        self,
        global_width: int = 384,
        option_width: int = 64,
        memory: int = 128,
        max_abs_bias: float = 2.5,
    ) -> None:
        super().__init__()
        self.global_width = int(global_width)
        self.option_width = int(option_width)
        self.memory_width = int(memory)
        self.max_abs_bias = float(max_abs_bias)
        if self.global_width <= 0 or self.option_width <= 0 or self.memory_width <= 0:
            raise ValueError("neural plan dimensions must be positive")
        if not (0.0 < self.max_abs_bias <= 5.0):
            raise ValueError("max_abs_bias must be in (0, 5]")

        # A fixed initialization aligns independent seeds so parameter soups are
        # meaningful.  Query output is exactly zero, making the initial acting
        # policy action-identical to the frozen blueprint.
        previous_state = torch.random.get_rng_state()
        torch.manual_seed(FIXED_INITIALIZATION_SEED)
        try:
            random_projection = torch.randn(self.global_width, self.option_width)
            projection, _ = torch.linalg.qr(random_projection, mode="reduced")
            self.projection = nn.Parameter(projection.contiguous())
            self.gru = nn.GRUCell(self.global_width, self.memory_width)
            self.plan = nn.Sequential(
                nn.Linear(self.memory_width, self.memory_width),
                nn.Tanh(),
            )
            self.query = nn.Linear(self.memory_width, self.option_width)
            self.strategic_heads = nn.Linear(
                self.memory_width, len(STRATEGIC_AXES)
            )
            self.meta_weights = nn.Linear(
                self.memory_width, len(STRATEGIC_AXES)
            )
        finally:
            torch.random.set_rng_state(previous_state)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.strategic_heads.weight)
        nn.init.zeros_(self.strategic_heads.bias)
        nn.init.zeros_(self.meta_weights.weight)
        nn.init.zeros_(self.meta_weights.bias)

    def initial_memory(self, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(1, self.memory_width, device=device)

    def plan_latent(self, memory: torch.Tensor) -> torch.Tensor:
        return self.plan(memory)

    def option_residual(
        self,
        option_tokens: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Return one bounded residual per option token."""
        latent = self.plan_latent(memory)
        query = self.query(latent).squeeze(0)
        projected = option_tokens @ self.projection
        raw = projected @ query / math.sqrt(self.option_width)
        return torch.clamp(raw, -self.max_abs_bias, self.max_abs_bias)

    def values(
        self, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.plan_latent(memory)
        axes = torch.tanh(self.strategic_heads(latent)).squeeze(0)
        weights = torch.softmax(self.meta_weights(latent), dim=-1).squeeze(0)
        overall = torch.sum(axes * weights)
        return overall, axes, weights

    def advance(self, global_token: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        token = global_token.reshape(1, self.global_width)
        return self.gru(token, memory)

    def config(self) -> dict[str, Any]:
        return {
            "global_width": self.global_width,
            "option_width": self.option_width,
            "memory": self.memory_width,
            "max_abs_bias": self.max_abs_bias,
            "strategic_axes": list(STRATEGIC_AXES),
            "fixed_initialization_seed": FIXED_INITIALIZATION_SEED,
        }


def _unsqueeze(feature: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    return feature[key].unsqueeze(0)


@torch.no_grad()
def encode_public_tokens(
    net: nn.Module,
    numpy_feature: Mapping[str, np.ndarray],
    *,
    forbid_end_when_attack: bool,
    options: Sequence[Mapping[str, Any]],
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one ordinary public feature dict through the frozen blueprint."""
    feature = feat_to_tensors(numpy_feature, device)
    poke_count = int(feature["poke_card"].shape[0])
    option_count = int(feature["opt_card"].shape[0])
    poke_mask = torch.ones(1, poke_count, device=device)
    poke_mask[0, -1] = 0.0
    option_mask = torch.ones(1, option_count, device=device)
    global_token, option_tokens = net._tokens(
        _unsqueeze(feature, "global"),
        _unsqueeze(feature, "belief_global"),
        _unsqueeze(feature, "hand_card"),
        _unsqueeze(feature, "my_discard_card"),
        _unsqueeze(feature, "opp_discard_card"),
        _unsqueeze(feature, "stadium_card"),
        _unsqueeze(feature, "my_unseen_card"),
        _unsqueeze(feature, "opp_unseen_card"),
        _unsqueeze(feature, "poke_feats"),
        _unsqueeze(feature, "poke_card"),
        _unsqueeze(feature, "opt_feats"),
        _unsqueeze(feature, "opt_unseen"),
        _unsqueeze(feature, "opt_card"),
        _unsqueeze(feature, "opt_attack"),
        _unsqueeze(feature, "opt_target"),
        poke_mask,
        option_mask,
        kpc=_unsqueeze(feature, "known_prized_card") if net.v4 else None,
        efc=_unsqueeze(feature, "effect_card") if net.v4 else None,
        bpz=_unsqueeze(feature, "belief_prized") if net.v4 else None,
        ctc=_unsqueeze(feature, "context_card") if net.v5 else None,
        pwk=_unsqueeze(feature, "poke_weak") if net.v5 else None,
    )
    logits = net.logit_head(option_tokens).squeeze(0).squeeze(-1)
    if "opt_avail" in feature:
        logits = logits.masked_fill(feature["opt_avail"] < 0.5, float("-inf"))
    if forbid_end_when_attack:
        logits = attack_safe_logits(logits, options)
    return global_token.squeeze(0), option_tokens.squeeze(0), logits


def residual_payload(
    residual: NeuralPlanResidual,
    *,
    base_model_sha256: str,
    deck_sha256: str,
    training_seed: int,
    iteration: int,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "config": residual.config(),
        "state_dict": {
            name: value.detach().cpu() for name, value in residual.state_dict().items()
        },
        "base_model_sha256": str(base_model_sha256),
        "deck_sha256": str(deck_sha256),
        "training_seed": int(training_seed),
        "iteration": int(iteration),
        "provenance": dict(provenance or {}),
    }


def load_neural_plan_residual(
    path: str | Path,
    *,
    expected_base_model_sha256: str | None = None,
    expected_deck_sha256: str | None = None,
) -> tuple[NeuralPlanResidual, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported neural plan residual payload")
    if (
        expected_base_model_sha256 is not None
        and payload.get("base_model_sha256") != expected_base_model_sha256
    ):
        raise ValueError("neural residual base-model hash mismatch")
    if (
        expected_deck_sha256 is not None
        and payload.get("deck_sha256") != expected_deck_sha256
    ):
        raise ValueError("neural residual deck hash mismatch")
    config = dict(payload.get("config") or {})
    if tuple(config.pop("strategic_axes", ())) != STRATEGIC_AXES:
        raise ValueError("neural residual strategic axes mismatch")
    if int(config.pop("fixed_initialization_seed", -1)) != FIXED_INITIALIZATION_SEED:
        raise ValueError("neural residual initialization version mismatch")
    residual = NeuralPlanResidual(**config)
    residual.load_state_dict(payload["state_dict"])
    return residual.eval(), payload


class NeuralPlanResidualAgent:
    """Arena-safe greedy actor with a persistent learned latent plan."""

    def __init__(
        self,
        net: nn.Module,
        residual_path: str | Path,
        *,
        deck_ids: Sequence[int],
        opponent_deck_priors: Mapping[str, Sequence[int]] | None = None,
        forbid_end_when_attack: bool = True,
        expected_base_model_sha256: str | None = None,
        expected_deck_sha256: str | None = None,
    ) -> None:
        self.net = net.cpu().eval()
        self.residual, self.payload = load_neural_plan_residual(
            residual_path,
            expected_base_model_sha256=expected_base_model_sha256,
            expected_deck_sha256=expected_deck_sha256,
        )
        self._deck_ids = list(map(int, deck_ids))
        self._opponent_deck_priors = opponent_deck_priors
        self.slot_link = bool(getattr(net, "slot_link", False))
        self.forbid_end_when_attack = bool(forbid_end_when_attack)
        self.reset_game()

    def reset_game(self) -> None:
        """Reset all per-game recurrent, belief, and telemetry state."""
        self.memory = self.residual.initial_memory()
        self.belief = Belief(
            self._deck_ids,
            opponent_deck_priors=self._opponent_deck_priors,
        )
        self.decision_count = 0
        self.intervention_count = 0
        self.exception_fallback_count = 0
        self.last_plan_values: dict[str, float] = {}
        self.last_meta_weights: dict[str, float] = {}
        self.last_decision: dict[str, Any] | None = None

    @torch.no_grad()
    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        selection = observation.get("select") or {}
        options = selection.get("option") or ()
        high = min(int(selection.get("maxCount") or 0), len(options))
        if high == 0:
            return []
        try:
            base_feature = F.extract(
                observation,
                belief=self.belief,
                slot_link=self.slot_link,
            )
            memory_before = self.memory
            picked: list[int] = []
            local_changes = 0
            chosen_biases: list[float] = []
            memory_token: torch.Tensor | None = None
            for pick in range(high):
                feature = (
                    base_feature
                    if high == 1
                    else F.mark_selection(base_feature, picked, high)
                )
                global_token, option_tokens, base_logits = encode_public_tokens(
                    self.net,
                    feature,
                    forbid_end_when_attack=self.forbid_end_when_attack,
                    options=options,
                )
                if memory_token is None:
                    memory_token = global_token
                residual_logits = self.residual.option_residual(
                    option_tokens, memory_before
                )
                combined = base_logits + residual_logits
                if self.forbid_end_when_attack:
                    combined = attack_safe_logits(combined, options)
                baseline = int(torch.argmax(base_logits))
                chosen = int(torch.argmax(combined))
                local_changes += int(chosen != baseline)
                chosen_biases.append(float(residual_logits[chosen]))
                picked.append(chosen)
            if memory_token is None:
                raise RuntimeError("neural plan decision had no memory token")
            overall, axes, weights = self.residual.values(memory_before)
            self.memory = self.residual.advance(memory_token, memory_before).detach()
            self.decision_count += 1
            self.intervention_count += int(local_changes > 0)
            self.last_plan_values = {
                name: float(axes[index]) for index, name in enumerate(STRATEGIC_AXES)
            }
            self.last_meta_weights = {
                name: float(weights[index]) for index, name in enumerate(STRATEGIC_AXES)
            }
            self.last_decision = {
                "schema": SCHEMA,
                "chosen": list(picked),
                "changed_picks": int(local_changes),
                "chosen_biases": chosen_biases,
                "overall_value": float(overall),
                "plan_values": self.last_plan_values,
                "meta_weights": self.last_meta_weights,
                "intervention_games": self.intervention_count,
            }
            return picked
        except Exception:
            self.exception_fallback_count += 1
            return random_legal_choice(selection)

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "decisions": self.decision_count,
            "intervention_decisions": self.intervention_count,
            "exception_fallbacks": self.exception_fallback_count,
            "plan_values": dict(self.last_plan_values),
            "meta_weights": dict(self.last_meta_weights),
        }
