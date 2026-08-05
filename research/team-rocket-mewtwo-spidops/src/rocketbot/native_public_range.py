"""Public-only neural opponent ranges for the native search abstraction.

The Phase 6I range head was trained on common/public replay features.  This
module exposes that head without the behavior-policy/private-feature path and
adapts native observations to the same feature schema.  It never receives a
selected hidden completion or a truth label.  The resulting distribution is a
research belief model, not official-engine state and not a safety certificate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

from .game_theory import (
    InformationSetActor,
    OpponentRange,
    PublicBeliefState,
    RangeHypothesis,
)
from .native_blueprint_features import native_public_feature_items
from .official_bridge import sanitize_official_public_state
from .official_trajectory import CausalPublicRangeUpdater, PublicRangeCandidate
from .trajectory_neural import LoggedTrajectoryBootstrapProvider


NATIVE_PUBLIC_RANGE_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _public_card_ids(player: Mapping[str, Any]) -> tuple[int, ...]:
    values: list[int] = []
    for card_id in player.get("discard", ()):
        if isinstance(card_id, int) and card_id > 0:
            values.append(card_id)
    for pokemon in tuple(player.get("active", ())) + tuple(player.get("bench", ())):
        if not isinstance(pokemon, Mapping):
            continue
        for card_id in (
            pokemon.get("id"),
            *tuple(pokemon.get("energy_cards", ())),
            *tuple(pokemon.get("tools", ())),
            *tuple(pokemon.get("pre_evolution", ())),
        ):
            if isinstance(card_id, int) and card_id > 0:
                values.append(card_id)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class NativePublicRangePrediction:
    """Auditable public-only range result at one native information set."""

    opponent_range: OpponentRange
    causal_reveal_range: OpponentRange
    model_input_public_features: tuple[tuple[str, float], ...]
    revealed_opponent_card_count: int
    provider_fingerprint: str
    hidden_completion_truth_input: bool = False
    private_feature_input: bool = False
    future_reveal_input: bool = False
    calibrated_for_native_search: bool = False

    def __post_init__(self) -> None:
        support = {item.hypothesis_id for item in self.opponent_range.hypotheses}
        causal_support = {
            item.hypothesis_id for item in self.causal_reveal_range.hypotheses
        }
        if support != causal_support or not support:
            raise ValueError("native neural and causal ranges must share support")
        if self.revealed_opponent_card_count < 0:
            raise ValueError("revealed opponent-card count cannot be negative")
        if len(self.provider_fingerprint) != 64:
            raise ValueError("native public-range provider fingerprint is invalid")
        if any(
            (
                self.hidden_completion_truth_input,
                self.private_feature_input,
                self.future_reveal_input,
                self.calibrated_for_native_search,
            )
        ):
            raise ValueError("diagnostic native range cannot promote calibration/input claims")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NATIVE_PUBLIC_RANGE_SCHEMA_VERSION,
            "opponent_range": self.opponent_range.to_dict(),
            "causal_reveal_range": self.causal_reveal_range.to_dict(),
            "model_input_public_features": dict(self.model_input_public_features),
            "revealed_opponent_card_count": self.revealed_opponent_card_count,
            "provider_fingerprint": self.provider_fingerprint,
            "input_claims": {
                "hidden_completion_truth_input": self.hidden_completion_truth_input,
                "private_feature_input": self.private_feature_input,
                "future_reveal_input": self.future_reveal_input,
            },
            "calibrated_for_native_search": self.calibrated_for_native_search,
        }


class LoggedTrajectoryNativePublicRangeProvider:
    """Adapt the Phase 6I public range head to native-search observations."""

    def __init__(
        self,
        bootstrap: LoggedTrajectoryBootstrapProvider,
        candidates: Sequence[PublicRangeCandidate],
    ) -> None:
        materialized = tuple(candidates)
        if not materialized:
            raise ValueError("native range provider requires a candidate catalog")
        candidate_ids = tuple(item.candidate_id for item in materialized)
        if candidate_ids != bootstrap.artifact.candidate_ids:
            raise ValueError("native range catalog/model candidate support mismatch")
        self.bootstrap = bootstrap
        self.candidates = materialized
        self.causal_updater = CausalPublicRangeUpdater(materialized)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema": NATIVE_PUBLIC_RANGE_SCHEMA_VERSION,
                "algorithm": "phase6i_public_head_with_causal_reveal_features",
                "bootstrap": self.bootstrap.artifact.fingerprint,
                "causal_updater": self.causal_updater.fingerprint,
                "candidate_fingerprints": [
                    item.fingerprint for item in self.candidates
                ],
                "hidden_completion_truth_input": False,
                "private_feature_input": False,
                "future_reveal_input": False,
                "native_calibration_claim": False,
            }
        )

    def predict(
        self,
        observation: Mapping[str, Any],
        base_belief: PublicBeliefState,
    ) -> NativePublicRangePrediction:
        if base_belief.acting_player is not InformationSetActor.SELF:
            raise ValueError("native public range requires an acting-player belief")
        public = sanitize_official_public_state(observation)
        actor = int(public.get("acting_player", -1))
        players = public.get("players", ())
        if actor not in (0, 1) or not isinstance(players, Sequence) or len(players) != 2:
            raise ValueError("native public range requires a two-player decision")
        opponent = players[1 - actor]
        if not isinstance(opponent, Mapping):
            raise ValueError("native public opponent view must be a mapping")
        revealed = Counter(_public_card_ids(opponent))
        causal_range = self.causal_updater.range_from_revealed(revealed)
        causal_belief = PublicBeliefState(
            public_state_fingerprint=base_belief.public_state_fingerprint,
            public_history_fingerprint=base_belief.public_history_fingerprint,
            turn=base_belief.turn,
            acting_player=base_belief.acting_player,
            opponent_range=causal_range,
            public_features=base_belief.public_features,
            updater_fingerprint=self.causal_updater.fingerprint,
        )
        model_features = native_public_feature_items(observation, causal_belief)
        probabilities = dict(self.bootstrap.predict_public_range(model_features))
        candidate_by_id = {item.candidate_id: item for item in self.candidates}
        if set(probabilities) != set(candidate_by_id):
            raise RuntimeError("native range model changed candidate support")
        opponent_range = OpponentRange.normalized(
            tuple(
                RangeHypothesis(
                    hypothesis_id=candidate.candidate_id,
                    weight=probabilities[candidate.candidate_id],
                    hidden_resource_counts=(),
                    intent_id=candidate.archetype_id,
                    style_id=candidate.style_id,
                    source=(
                        "Phase 6I history-conditioned public range head; "
                        "native public observation only; no selected completion truth"
                    ),
                )
                for candidate in self.candidates
            )
        )
        return NativePublicRangePrediction(
            opponent_range=opponent_range,
            causal_reveal_range=causal_range,
            model_input_public_features=model_features,
            revealed_opponent_card_count=sum(revealed.values()),
            provider_fingerprint=self.fingerprint,
        )

    def belief(
        self,
        observation: Mapping[str, Any],
        base_belief: PublicBeliefState,
    ) -> PublicBeliefState:
        """Return a policy-consumable belief while retaining public-only inputs."""

        prediction = self.predict(observation, base_belief)
        features = tuple(
            (
                name,
                (
                    prediction.opponent_range.entropy
                    if name == "opponent_range_entropy"
                    else prediction.opponent_range.effective_sample_size
                    if name == "opponent_range_effective_sample_size"
                    else value
                ),
            )
            for name, value in prediction.model_input_public_features
        )
        if not all(math.isfinite(value) for _, value in features):
            raise RuntimeError("native neural range produced non-finite public features")
        return PublicBeliefState(
            public_state_fingerprint=base_belief.public_state_fingerprint,
            public_history_fingerprint=base_belief.public_history_fingerprint,
            turn=base_belief.turn,
            acting_player=base_belief.acting_player,
            opponent_range=prediction.opponent_range,
            public_features=features,
            updater_fingerprint=self.fingerprint,
        )
