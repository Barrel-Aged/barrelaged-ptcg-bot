"""Regularized explicit-public-event opponent style classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SCHEMA = "rocket-public-event-style-classifier/v1"
FEATURE_WIDTH = 644


class PublicEventStyleClassifier(nn.Module):
    def __init__(self, input_width: int = FEATURE_WIDTH) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.norm = nn.LayerNorm(self.input_width)
        self.head = nn.Linear(self.input_width, 1)

    def initial_hidden(self, device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.zeros(1, 1, device=device)

    def step(
        self, token: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logit = self.head(self.norm(token.reshape(1, self.input_width))).squeeze()
        return logit, hidden

    def sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError("public-event style sequence is empty or malformed")
        return self.head(self.norm(tokens)).squeeze(-1)


def load_public_event_style_classifier(
    path: str | Path,
    *,
    expected_base_model_sha256: str | None = None,
    expected_deck_sha256: str | None = None,
) -> tuple[PublicEventStyleClassifier, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported public-event style classifier")
    if expected_base_model_sha256 and payload.get("base_model_sha256") != expected_base_model_sha256:
        raise ValueError("public-event classifier blueprint mismatch")
    if expected_deck_sha256 and payload.get("deck_sha256") != expected_deck_sha256:
        raise ValueError("public-event classifier deck mismatch")
    model = PublicEventStyleClassifier(**dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload
