"""Fail-closed ensemble over aligned recurrent neural plan residuals."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from soldier import features as F
from soldier.belief import Belief
from soldier.engine import random_legal_choice
from soldier.net import attack_safe_logits

from neural_plan_residual_runtime import (
    NeuralPlanResidual,
    STRATEGIC_AXES,
    encode_public_tokens,
)


SCHEMA = "rocket-neural-plan-ensemble/v1"
MODES = ("blueprint", "mean_logits", "consensus")


def _load_source(payload: Mapping[str, Any]) -> NeuralPlanResidual:
    config = dict(payload.get("config") or {})
    config.pop("strategic_axes", None)
    config.pop("fixed_initialization_seed", None)
    model = NeuralPlanResidual(**config)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def load_ensemble(
    path: str | Path,
    *,
    expected_base_model_sha256: str | None = None,
    expected_deck_sha256: str | None = None,
) -> tuple[list[NeuralPlanResidual], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported neural-plan ensemble payload")
    if payload.get("mode") not in MODES:
        raise ValueError("unsupported neural-plan ensemble mode")
    if expected_base_model_sha256 and payload.get("base_model_sha256") != expected_base_model_sha256:
        raise ValueError("ensemble base-model hash mismatch")
    if expected_deck_sha256 and payload.get("deck_sha256") != expected_deck_sha256:
        raise ValueError("ensemble deck hash mismatch")
    sources = list(payload.get("sources") or ())
    if len(sources) != 4:
        raise ValueError("N2 ensemble requires exactly four sources")
    for source in sources:
        if source.get("schema") != "rocket-neural-plan-residual/v1":
            raise ValueError("ensemble source schema mismatch")
        if source.get("base_model_sha256") != payload.get("base_model_sha256"):
            raise ValueError("ensemble source blueprint mismatch")
        if source.get("deck_sha256") != payload.get("deck_sha256"):
            raise ValueError("ensemble source deck mismatch")
    models = [_load_source(source) for source in sources]
    return models, payload


class NeuralPlanEnsembleAgent:
    """Greedy official actor with mean or consensus residual arbitration."""

    def __init__(
        self,
        net: torch.nn.Module,
        ensemble_path: str | Path,
        *,
        deck_ids: Sequence[int],
        opponent_deck_priors: Mapping[str, Sequence[int]] | None = None,
        forbid_end_when_attack: bool = True,
        expected_base_model_sha256: str | None = None,
        expected_deck_sha256: str | None = None,
    ) -> None:
        self.net = net.cpu().eval()
        self.models, self.payload = load_ensemble(
            ensemble_path,
            expected_base_model_sha256=expected_base_model_sha256,
            expected_deck_sha256=expected_deck_sha256,
        )
        self.mode = str(self.payload["mode"])
        self.scale = float(self.payload.get("scale", 1.0))
        self.min_votes = int(self.payload.get("min_votes", 4))
        if not (0.0 <= self.scale <= 1.0):
            raise ValueError("ensemble scale must be in [0, 1]")
        if not (1 <= self.min_votes <= 4):
            raise ValueError("ensemble min_votes must be in [1, 4]")
        self._deck_ids = list(map(int, deck_ids))
        self._opponent_deck_priors = opponent_deck_priors
        self.slot_link = bool(getattr(net, "slot_link", False))
        self.forbid_end_when_attack = bool(forbid_end_when_attack)
        self.reset_game()

    def reset_game(self) -> None:
        self.memories = [model.initial_memory() for model in self.models]
        self.belief = Belief(
            self._deck_ids,
            opponent_deck_priors=self._opponent_deck_priors,
        )
        self.decision_count = 0
        self.intervention_count = 0
        self.exception_fallback_count = 0
        self.consensus_rejection_count = 0
        self.last_plan_values: dict[str, float] = {}
        self.last_meta_weights: dict[str, float] = {}
        self.last_decision: dict[str, Any] | None = None
        self.last_global_token: torch.Tensor | None = None
        self.global_token_history: list[torch.Tensor] = []

    def _pick(
        self,
        base_logits: torch.Tensor,
        residuals: list[torch.Tensor],
        options: Sequence[Mapping[str, Any]],
    ) -> tuple[int, int, list[int], float]:
        baseline = int(torch.argmax(base_logits))
        source_choices = []
        for residual in residuals:
            logits = base_logits + residual
            if self.forbid_end_when_attack:
                logits = attack_safe_logits(logits, options)
            source_choices.append(int(torch.argmax(logits)))
        if self.mode == "blueprint":
            return baseline, baseline, source_choices, 0.0
        mean_residual = torch.stack(residuals, dim=0).mean(dim=0)
        if self.mode == "mean_logits":
            combined = base_logits + self.scale * mean_residual
            if self.forbid_end_when_attack:
                combined = attack_safe_logits(combined, options)
            chosen = int(torch.argmax(combined))
            return chosen, baseline, source_choices, float(self.scale * mean_residual[chosen])
        vote, count = Counter(source_choices).most_common(1)[0]
        if vote != baseline and count >= self.min_votes:
            chosen = int(vote)
        else:
            chosen = baseline
            if any(choice != baseline for choice in source_choices):
                self.consensus_rejection_count += 1
        return chosen, baseline, source_choices, float(mean_residual[chosen])

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
            picked: list[int] = []
            changes = 0
            vote_rows: list[list[int]] = []
            chosen_biases: list[float] = []
            memory_token: torch.Tensor | None = None
            for _ in range(high):
                feature = base_feature if high == 1 else F.mark_selection(base_feature, picked, high)
                global_token, option_tokens, base_logits = encode_public_tokens(
                    self.net,
                    feature,
                    forbid_end_when_attack=self.forbid_end_when_attack,
                    options=options,
                )
                if memory_token is None:
                    memory_token = global_token
                residuals = [
                    model.option_residual(option_tokens, memory)
                    for model, memory in zip(self.models, self.memories)
                ]
                chosen, baseline, source_choices, chosen_bias = self._pick(
                    base_logits, residuals, options
                )
                picked.append(chosen)
                changes += int(chosen != baseline)
                vote_rows.append(source_choices)
                chosen_biases.append(chosen_bias)
            if memory_token is None:
                raise RuntimeError("ensemble decision had no memory token")
            self.last_global_token = memory_token.detach().cpu()
            self.global_token_history.append(self.last_global_token)
            values = [model.values(memory) for model, memory in zip(self.models, self.memories)]
            axes = torch.stack([value[1] for value in values]).mean(dim=0)
            weights = torch.stack([value[2] for value in values]).mean(dim=0)
            weights = weights / weights.sum().clamp_min(1.0e-8)
            overall = torch.sum(axes * weights)
            self.memories = [
                model.advance(memory_token, memory).detach()
                for model, memory in zip(self.models, self.memories)
            ]
            self.decision_count += 1
            self.intervention_count += int(changes > 0)
            self.last_plan_values = {
                name: float(axes[index]) for index, name in enumerate(STRATEGIC_AXES)
            }
            self.last_meta_weights = {
                name: float(weights[index]) for index, name in enumerate(STRATEGIC_AXES)
            }
            self.last_decision = {
                "schema": SCHEMA,
                "mode": self.mode,
                "chosen": picked,
                "changed_picks": changes,
                "source_choices": vote_rows,
                "chosen_biases": chosen_biases,
                "overall_value": float(overall),
            }
            return picked
        except Exception:
            self.exception_fallback_count += 1
            return random_legal_choice(selection)

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "decisions": self.decision_count,
            "intervention_decisions": self.intervention_count,
            "consensus_rejections": self.consensus_rejection_count,
            "exception_fallbacks": self.exception_fallback_count,
            "plan_values": dict(self.last_plan_values),
            "meta_weights": dict(self.last_meta_weights),
        }
