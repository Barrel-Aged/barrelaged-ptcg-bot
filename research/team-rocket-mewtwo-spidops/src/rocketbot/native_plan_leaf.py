"""Uncalibrated bounded leaf adapter for native turn-plan experiments."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any

from .depth_limited_game import LeafValueEstimate
from .game_theory import AbstractGameState, AbstractInformationSetGame
from .native_plan_game import NativePlanGameState, OfficialNativePlanGame


NATIVE_PLAN_LEAF_SCHEMA_VERSION = 1


def _fingerprint(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class UncalibratedNativePlanLeafValueProvider:
    """Clamp the Phase 6K scalar value for variance experiments only.

    The returned interval is the complete native utility range. It is
    intentionally marked uncalibrated and cannot support safety admission.
    """

    def __init__(self, game: OfficialNativePlanGame) -> None:
        self.game = game
        self._fingerprint = _fingerprint(
            {
                "schema": NATIVE_PLAN_LEAF_SCHEMA_VERSION,
                "provider": "uncalibrated_phase6k_clamped_leaf",
                "game": game.fingerprint,
                "tactical_model": game.tactical_provider.artifact.fingerprint,
                "range": [-1.0, 1.0],
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def calibrated(self) -> bool:
        return False

    def estimate(
        self,
        game: AbstractInformationSetGame,
        state: AbstractGameState,
        *,
        player: int,
    ) -> LeafValueEstimate:
        if game is not self.game or not isinstance(state, NativePlanGameState):
            raise ValueError("native leaf provider received another game/state")
        acting_player = self.game.player_to_act(state)
        if acting_player not in (0, 1) or player not in (0, 1):
            raise ValueError("native leaf provider requires a player decision")
        view = self.game.bridge.view(self.game.native_game, state.native_state)
        prediction = self.game.tactical_provider.predict(
            view.information_set, view.public_belief
        )
        acting_value = max(-1.0, min(1.0, float(prediction.state_value)))
        value = acting_value if player == acting_player else -acting_value
        return LeafValueEstimate(
            player=player,
            value=value,
            lower=-1.0,
            upper=1.0,
            confidence=0.0,
            source=(
                "uncalibrated Phase 6K scalar clamped to native utility range; "
                "variance experiment only"
            ),
            calibrated=False,
        )
