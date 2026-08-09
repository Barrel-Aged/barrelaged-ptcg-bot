"""Confidence-gated composition of the admitted neural and deterministic policies."""
from __future__ import annotations

from typing import Any

from bot.agent import Agent as BaselineAgent
from lotus_px026_bc_policy_v17 import LucarioPromotionClonePolicyV17
from soldier.net import NNAgent


def _legal(action: Any, selection: dict[str, Any]) -> bool:
    if not isinstance(action, list) or any(not isinstance(index, int) for index in action):
        return False
    option_count = len(selection.get("option") or [])
    if len(action) != len(set(action)):
        return False
    if not all(0 <= index < option_count for index in action):
        return False
    return int(selection.get("minCount") or 0) <= len(action) <= min(
        int(selection.get("maxCount") or 0), option_count
    )


class ConfidenceSafeHybridAgent:
    """Run both policies and defer low-confidence neural rows to V121.

    Calling both policies on every observation is intentional: neither policy's
    belief or turn memory depends on whether its proposed action was selected.
    ``last_trace`` is diagnostic metadata only and is never returned to the game.
    """

    def __init__(
        self,
        net,
        deck_ids: list[int],
        threshold: float,
        opponent_deck_priors=None,
    ) -> None:
        self.threshold = float(threshold)
        self.neural = NNAgent(
            net,
            greedy=True,
            deck_ids=deck_ids,
            forbid_end_when_attack=True,
            opponent_deck_priors=opponent_deck_priors,
        )
        self.baseline = BaselineAgent(
            policy=LucarioPromotionClonePolicyV17(our_deck=deck_ids, margin=0.0)
        )
        self.last_trace: dict[str, Any] = {}

    def __call__(self, observation: dict[str, Any]) -> list[int]:
        selection = observation.get("select")
        if selection is None:
            return self.baseline.act(observation)

        baseline_action = self.baseline.act(observation)
        neural_action = self.neural(observation)
        confidence = getattr(self.neural, "last_confidence", None)
        margin = getattr(self.neural, "last_margin", None)
        neural_legal = _legal(neural_action, selection)
        baseline_legal = _legal(baseline_action, selection)

        route_to_baseline = (
            not neural_legal
            or confidence is None
            or float(confidence) < self.threshold
        )
        if route_to_baseline and baseline_legal:
            action = baseline_action
            source = "baseline"
        elif neural_legal:
            action = neural_action
            source = "neural"
        elif baseline_legal:
            action = baseline_action
            source = "baseline_invalid_neural"
        else:
            maximum = min(
                int(selection.get("maxCount") or 0),
                len(selection.get("option") or []),
            )
            action = list(range(maximum))
            source = "legal_fallback"

        self.last_trace = {
            "source": source,
            "threshold": self.threshold,
            "confidence": confidence,
            "margin": margin,
            "neural_action": neural_action,
            "baseline_action": baseline_action,
            "disagreement": neural_action != baseline_action,
        }
        return action
