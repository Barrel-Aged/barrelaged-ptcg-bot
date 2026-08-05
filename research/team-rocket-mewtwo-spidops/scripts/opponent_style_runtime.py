"""Public-history opponent-style classifier and activation state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


SCHEMA = "rocket-opponent-style-classifier/v1"


class OpponentStyleClassifier(nn.Module):
    def __init__(self, input_width: int = 384, hidden_width: int = 64) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.norm = nn.LayerNorm(self.input_width)
        self.gru = nn.GRUCell(self.input_width, self.hidden_width)
        self.head = nn.Linear(self.hidden_width, 1)

    def initial_hidden(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(1, self.hidden_width, device=device)

    def step(
        self, token: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(token.reshape(1, self.input_width))
        next_hidden = self.gru(normalized, hidden)
        logit = self.head(next_hidden).squeeze()
        return logit, next_hidden

    def sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_hidden(tokens.device)
        logits = []
        for token in tokens:
            logit, hidden = self.step(token, hidden)
            logits.append(logit)
        if not logits:
            raise ValueError("style sequence is empty")
        return torch.stack(logits)


def load_style_classifier(
    path: str | Path,
    *,
    expected_base_model_sha256: str | None = None,
    expected_deck_sha256: str | None = None,
) -> tuple[OpponentStyleClassifier, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported opponent-style classifier")
    if expected_base_model_sha256 and payload.get("base_model_sha256") != expected_base_model_sha256:
        raise ValueError("style classifier blueprint mismatch")
    if expected_deck_sha256 and payload.get("deck_sha256") != expected_deck_sha256:
        raise ValueError("style classifier deck mismatch")
    model = OpponentStyleClassifier(**dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


class StyleActivationState:
    """Fail-closed consecutive-evidence activation state."""

    def __init__(self, classifier: OpponentStyleClassifier, payload: dict[str, Any]) -> None:
        self.classifier = classifier
        self.temperature = float(payload["temperature"])
        self.threshold = float(payload["activation_threshold"])
        self.minimum_decisions = int(payload["minimum_decisions"])
        self.consecutive_required = int(payload["consecutive_required"])
        if self.temperature <= 0 or not (0.0 <= self.threshold <= 1.0):
            raise ValueError("invalid style calibration")
        self.reset()

    def reset(self) -> None:
        self.hidden = self.classifier.initial_hidden()
        self.decisions = 0
        self.consecutive = 0
        self.activated = False
        self.activation_decision: int | None = None
        self.last_probability = 0.0

    @torch.no_grad()
    def observe(self, token: torch.Tensor) -> float:
        logit, self.hidden = self.classifier.step(token, self.hidden)
        self.decisions += 1
        probability = float(torch.sigmoid(logit / self.temperature))
        self.last_probability = probability
        if not self.activated:
            if probability >= self.threshold:
                self.consecutive += 1
            else:
                self.consecutive = 0
            if (
                self.decisions >= self.minimum_decisions
                and self.consecutive >= self.consecutive_required
            ):
                self.activated = True
                self.activation_decision = self.decisions
        return probability
