"""Leakage-audited explicit public opponent-action features."""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Mapping

import numpy as np
import torch

from neural_plan_ensemble_runtime import NeuralPlanEnsembleAgent


EVENT_BINS = 512
BOARD_BINS = 128
SCALAR_WIDTH = 4
FEATURE_WIDTH = EVENT_BINS + BOARD_BINS + SCALAR_WIDTH
ALLOWED_EVENT_FIELDS = (
    "cardId",
    "cardIdTarget",
    "cardIdBefore",
    "cardIdAfter",
    "cardIdActive",
    "cardIdBench",
    "attackId",
    "fromArea",
    "toArea",
    "value",
    "putDamageCounter",
    "isRecover",
    "result",
    "reason",
)


def _bucket(token: str, bins: int) -> int:
    digest = sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % bins


def _scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return str(value)


def _walk_public_cards(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _walk_public_cards(item)
    elif isinstance(value, Mapping):
        card_id = value.get("id")
        if isinstance(card_id, (int, float)):
            yield int(card_id)
        for key in ("preEvolution", "tools", "energyCards"):
            yield from _walk_public_cards(value.get(key) or ())


class PublicEventFeatureTracker:
    """Cumulative opponent event hashes plus current public-board hashes."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.event_counts = np.zeros(EVENT_BINS, dtype=np.float32)

    def _add(self, token: str) -> None:
        self.event_counts[_bucket(token, EVENT_BINS)] += 1.0

    def observe(self, observation: Mapping[str, Any]) -> torch.Tensor:
        current = observation.get("current") or {}
        your_index = int(current.get("yourIndex") or 0)
        opponent = 1 - your_index
        for raw in observation.get("logs") or ():
            if not isinstance(raw, Mapping) or int(raw.get("playerIndex", -1)) != opponent:
                continue
            event_type = int(raw.get("type", -1))
            self._add(f"event:{event_type}")
            present = []
            for field in ALLOWED_EVENT_FIELDS:
                value = _scalar(raw.get(field))
                if value is None:
                    continue
                present.append((field, value))
                self._add(f"event:{event_type}:{field}:{value}")
            if present:
                joined = "|".join(f"{field}={value}" for field, value in sorted(present))
                self._add(f"event:{event_type}:joint:{joined}")
        event_features = np.log1p(self.event_counts)
        board_features = np.zeros(BOARD_BINS, dtype=np.float32)
        players = current.get("players") or ()
        opponent_view = players[opponent] if len(players) == 2 else {}
        opponent_view = opponent_view if isinstance(opponent_view, Mapping) else {}
        for zone in ("active", "bench", "discard"):
            for card_id in _walk_public_cards(opponent_view.get(zone) or ()):
                board_features[_bucket(f"{zone}:{card_id}", BOARD_BINS)] += 1.0
        for card in current.get("stadium") or ():
            if isinstance(card, Mapping) and int(card.get("playerIndex", -1)) == opponent:
                card_id = card.get("id")
                if isinstance(card_id, (int, float)):
                    board_features[_bucket(f"stadium:{int(card_id)}", BOARD_BINS)] += 1.0
        board_features = np.log1p(board_features)
        scalars = np.asarray(
            [
                min(1.0, float(current.get("turn") or 0) / 30.0),
                min(1.0, float(opponent_view.get("handCount") or 0) / 20.0),
                min(1.0, float(opponent_view.get("deckCount") or 0) / 60.0),
                min(1.0, float(len(opponent_view.get("prize") or ())) / 6.0),
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(np.concatenate((event_features, board_features, scalars)))


class EventTrackingEnsembleAgent(NeuralPlanEnsembleAgent):
    """N2 agent augmented only with public event features for data collection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.event_tracker = PublicEventFeatureTracker()
        super().__init__(*args, **kwargs)

    def reset_game(self) -> None:
        super().reset_game()
        self.event_tracker.reset()
        self.public_event_feature_history: list[torch.Tensor] = []

    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        self.public_event_feature_history.append(self.event_tracker.observe(observation))
        return super().__call__(observation)
