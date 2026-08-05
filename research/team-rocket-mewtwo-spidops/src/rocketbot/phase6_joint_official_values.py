"""Phase 6AC same-root official-prefix and truth-sample action values.

Phase 6AB recorded genuine player-action prefixes but collected them separately
from Phase 6Z values.  This module binds three objects at one exact live root:

* the complete redacted deterministic behavior prefix;
* one realized hidden state read transiently from the official visualizer; and
* repeated terminal values for every one of the five abstract turn plans.

The visualizer truth is used only to initialize the licensed native branch API
and to form opaque digests.  Raw hidden identities never enter policy features
or serialized artifacts.  Root and one-step live/search parity are mandatory.

The native wrapper still does not disclose absolute shuffle, Prize, draw, or
coin reach.  Target and sampling deliberately reference the same opaque
official chance law, so its pointwise likelihood ratio cancels symbolically.
That makes an on-policy sample contribution ratio available without inventing
an absolute probability.  One truth sample per independently reached root is
not a calibrated counterfactual value, bound, resolver, equilibrium result, or
official strategy-control capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

from .competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    OptionType,
    SelectionContract,
    _safe_int,
    _state,
    resolve_official_deck,
)
from .counterfactual_reach import CommonChanceLawReachRatio
from .game_theory import ActionDistribution, InformationSetActor
from .native_blueprint_features import NativeBlueprintFeatureBridge
from .native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanBlueprintProvider,
    restore_native_plan_blueprint,
)
from .native_plan_game import OfficialNativePlanGame
from .native_regret_blueprint import (
    NativeRegretBlueprintArtifact,
    restore_native_regret_blueprint,
)
from .official_bridge import (
    official_private_observation_payload,
    sanitize_official_public_state,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    rocket_compatibility_profile,
)
from .official_search_game import HiddenZoneCompletion, OfficialNativeSearchGame
from .phase6_aligned_counterfactual_values import (
    ALIGNED_VALUE_SPLITS,
    PublicRootAlignedValueDataset,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _matching_action
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence
from .phase6_official_prefix_reach import (
    OFFICIAL_PREFIX_CAPTURE_CONFIG,
    OfficialPrefixReachTrace,
    _capture_official_prefix,
)
from .phase6_opponent_deviation_values import (
    OPPONENT_DEVIATION_ACTION_IDS,
    OpponentPlanActionValue,
    _rollout_average_blueprint,
)


PHASE6_JOINT_OFFICIAL_VALUE_SCHEMA_VERSION = 1
PHASE6_JOINT_OFFICIAL_VALUE_SEED = 6_035
JOINT_OFFICIAL_ROLLOUTS_PER_ACTION = 3


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8],
        "big",
    )


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _truth_card_ids(items: Any, name: str) -> tuple[int, ...]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise ValueError(f"official truth {name} is not a card sequence")
    result = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError(f"official truth {name} contains a hidden/non-card entry")
        card_id = _safe_int(item.get("id"), 0)
        if card_id <= 0:
            raise ValueError(f"official truth {name} contains an invalid card ID")
        result.append(card_id)
    return tuple(result)


def _official_truth_completion(
    observation: Mapping[str, Any],
    truth_frame: Mapping[str, Any],
    *,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    style: str,
) -> HiddenZoneCompletion:
    truth_current = truth_frame.get("current")
    if not isinstance(truth_current, Mapping):
        raise ValueError("official truth frame has no current state")
    truth_players = truth_current.get("players")
    if not isinstance(truth_players, Sequence) or len(truth_players) < 2:
        raise ValueError("official truth frame has no two-player state")
    live_players = _state(observation).get("players")
    if not isinstance(live_players, Sequence) or len(live_players) < 2:
        raise ValueError("official root has no two-player state")
    root_player = _safe_int(_state(observation).get("yourIndex"), -1)
    if root_player != 1:
        raise ValueError("Phase 6AC requires a player-1 official root")
    your_truth = truth_players[root_player]
    opponent_truth = truth_players[1 - root_player]
    if not isinstance(your_truth, Mapping) or not isinstance(opponent_truth, Mapping):
        raise ValueError("official truth player state is malformed")
    opponent_active: tuple[int, ...] = ()
    live_opponent = live_players[1 - root_player]
    if isinstance(live_opponent, Mapping):
        live_active = live_opponent.get("active", ())
        if (
            isinstance(live_active, Sequence)
            and live_active
            and live_active[0] is None
        ):
            opponent_active = _truth_card_ids(
                opponent_truth.get("active", ()),
                "opponent active",
            )
    completion = HiddenZoneCompletion(
        hypothesis_id="realized-official-truth",
        probability=1.0,
        root_player_index=root_player,
        your_candidate_id=candidate.archetype,
        opponent_candidate_id=target.archetype,
        your_archetype_id=candidate.archetype,
        opponent_archetype_id=target.archetype,
        your_style_id=style,
        opponent_style_id=style,
        your_deck_fingerprint=candidate.fingerprint,
        opponent_deck_fingerprint=target.fingerprint,
        your_deck=_truth_card_ids(your_truth.get("deck", ()), "acting deck"),
        your_prize=_truth_card_ids(your_truth.get("prize", ()), "acting prize"),
        opponent_deck=_truth_card_ids(
            opponent_truth.get("deck", ()),
            "other deck",
        ),
        opponent_prize=_truth_card_ids(
            opponent_truth.get("prize", ()),
            "other prize",
        ),
        opponent_hand=_truth_card_ids(
            opponent_truth.get("hand", ()),
            "other hand",
        ),
        opponent_active=opponent_active,
        manual_coin=True,
        public_evidence_adjustments=0,
        source=(
            "transient realized state from the genuine official visualizer; "
            "raw truth is supervision-only and is never serialized or a policy input"
        ),
    )
    expected_counts = {
        "acting_deck_count": _safe_int(live_players[root_player].get("deckCount"), -1),
        "acting_prize_count": len(live_players[root_player].get("prize", ())),
        "other_deck_count": _safe_int(live_players[1 - root_player].get("deckCount"), -1),
        "other_prize_count": len(live_players[1 - root_player].get("prize", ())),
        "other_hand_count": _safe_int(live_players[1 - root_player].get("handCount"), -1),
        "other_hidden_active_count": len(opponent_active),
    }
    observed_counts = dict(_completion_zone_counts(completion))
    if expected_counts != observed_counts:
        raise ValueError("official truth completion zone counts do not match live root")
    return completion


def _completion_zone_counts(
    completion: HiddenZoneCompletion,
) -> tuple[tuple[str, int], ...]:
    return (
        ("acting_deck_count", len(completion.your_deck)),
        ("acting_prize_count", len(completion.your_prize)),
        ("other_deck_count", len(completion.opponent_deck)),
        ("other_prize_count", len(completion.opponent_prize)),
        ("other_hand_count", len(completion.opponent_hand)),
        ("other_hidden_active_count", len(completion.opponent_active)),
    )


def _resolve_live_coin_results(
    game: OfficialNativeSearchGame,
    state: Any,
    live_observation: Mapping[str, Any],
) -> tuple[Any, int]:
    coin_results = [
        bool(item.get("head"))
        for item in live_observation.get("logs", ())
        if isinstance(item, Mapping) and _safe_int(item.get("type"), -1) == 22
    ]
    used = 0
    while state.actor is InformationSetActor.CHANCE:
        if used >= len(coin_results):
            raise RuntimeError("native replay requires an unobserved live coin result")
        head = coin_results[used]
        used += 1
        outcome = next(
            (
                item
                for item in game.chance_outcomes(state)
                if (":head:" in item.outcome_id) == head
            ),
            None,
        )
        if outcome is None:
            raise RuntimeError("native replay could not match the live coin result")
        state = game.transition(state, None, chance_outcome=outcome)
    if used != len(coin_results):
        raise RuntimeError("live replay exposed unmatched coin results")
    return state, used


@dataclass(frozen=True, slots=True)
class JointOfficialTruthValueSurface:
    split: str
    candidate_id: str
    prefix_trace: OfficialPrefixReachTrace
    common_chance_ratio: CommonChanceLawReachRatio
    truth_frame_fingerprint: str
    truth_completion_fingerprint: str
    truth_zone_counts: tuple[tuple[str, int], ...]
    source_game_fingerprint: str
    source_state_id: str
    information_set_id: str
    public_belief_fingerprint: str
    public_history_fingerprint: str
    private_observation_fingerprint: str
    action_values: tuple[OpponentPlanActionValue, ...]
    blueprint_opponent_value: float
    best_response_action_id: str
    best_response_opponent_value: float
    empirical_deviation_gain: float
    empirical_action_span: float
    live_replay_coin_count: int
    visualizer_root_public_parity: bool = True
    visualizer_root_acting_private_zone_parity: bool = True
    native_root_public_parity: bool = True
    native_root_private_parity: bool = True
    native_live_next_public_parity: bool = True
    native_live_next_private_parity: bool = True
    joint_prefix_truth_value_root: bool = True
    realized_hidden_state_official_sample: bool = True
    descendant_transitions_official_engine_executed: bool = True
    hidden_truth_selected_for_policy_input: bool = False
    raw_private_card_ids_persisted: int = 0
    absolute_chance_reach_available: bool = False
    calibrated_counterfactual_value: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        if self.split not in ALIGNED_VALUE_SPLITS or not self.candidate_id:
            raise ValueError("joint official surface split/candidate is invalid")
        for name in (
            "truth_frame_fingerprint",
            "truth_completion_fingerprint",
            "source_game_fingerprint",
            "source_state_id",
            "public_belief_fingerprint",
            "public_history_fingerprint",
            "private_observation_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.information_set_id:
            raise ValueError("joint official surface information set is missing")
        if self.prefix_trace.split != self.split:
            raise ValueError("joint official surface prefix split mismatch")
        if self.prefix_trace.candidate_id != self.candidate_id:
            raise ValueError("joint official surface prefix candidate mismatch")
        if (
            self.prefix_trace.root_public_history_fingerprint
            != self.public_history_fingerprint
        ):
            raise ValueError("joint official surface prefix/value root mismatch")
        expected_zone_names = (
            "acting_deck_count",
            "acting_prize_count",
            "other_deck_count",
            "other_prize_count",
            "other_hand_count",
            "other_hidden_active_count",
        )
        if tuple(name for name, _ in self.truth_zone_counts) != expected_zone_names:
            raise ValueError("joint official truth zone schema mismatch")
        if any(value < 0 for _, value in self.truth_zone_counts):
            raise ValueError("joint official truth zone count is negative")
        if self.common_chance_ratio.target_player != 1:
            raise ValueError("joint official chance ratio must target player 1")
        if not math.isclose(
            self.common_chance_ratio.counterfactual_likelihood_ratio,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("joint official deterministic on-policy ratio drifted")
        if tuple(item.action_id for item in self.action_values) != (
            OPPONENT_DEVIATION_ACTION_IDS
        ):
            raise ValueError("joint official plan-action support/order is invalid")
        blueprint = sum(
            item.blueprint_probability * item.empirical_opponent_value
            for item in self.action_values
        )
        if not math.isclose(self.blueprint_opponent_value, blueprint, abs_tol=1e-12):
            raise ValueError("joint official blueprint value is inconsistent")
        best = max(
            self.action_values,
            key=lambda item: (
                item.empirical_opponent_value,
                -OPPONENT_DEVIATION_ACTION_IDS.index(item.action_id),
            ),
        )
        if (
            self.best_response_action_id != best.action_id
            or not math.isclose(
                self.best_response_opponent_value,
                best.empirical_opponent_value,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("joint official best-response summary is inconsistent")
        if not math.isclose(
            self.empirical_deviation_gain,
            self.best_response_opponent_value - self.blueprint_opponent_value,
            abs_tol=1e-12,
        ):
            raise ValueError("joint official deviation gain is inconsistent")
        span = max(item.empirical_opponent_value for item in self.action_values) - min(
            item.empirical_opponent_value for item in self.action_values
        )
        if not math.isclose(self.empirical_action_span, span, abs_tol=1e-12):
            raise ValueError("joint official action span is inconsistent")
        if self.live_replay_coin_count < 0:
            raise ValueError("joint official replay coin count is negative")
        if not all(
            (
                self.visualizer_root_public_parity,
                self.visualizer_root_acting_private_zone_parity,
                self.native_root_public_parity,
                self.native_root_private_parity,
                self.native_live_next_public_parity,
                self.native_live_next_private_parity,
                self.joint_prefix_truth_value_root,
                self.realized_hidden_state_official_sample,
                self.descendant_transitions_official_engine_executed,
            )
        ):
            raise ValueError("joint official surface fidelity/provenance gate failed")
        if any(
            (
                self.hidden_truth_selected_for_policy_input,
                self.raw_private_card_ids_persisted != 0,
                self.absolute_chance_reach_available,
                self.calibrated_counterfactual_value,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("joint official surface promoted an unsupported capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_JOINT_OFFICIAL_VALUE_SCHEMA_VERSION,
            "split": self.split,
            "candidate_id": self.candidate_id,
            "prefix_trace": self.prefix_trace.to_dict(),
            "common_chance_ratio": self.common_chance_ratio.to_dict(),
            "truth": {
                "frame_fingerprint": self.truth_frame_fingerprint,
                "completion_fingerprint": self.truth_completion_fingerprint,
                "zone_counts": dict(self.truth_zone_counts),
            },
            "root": {
                "source_game_fingerprint": self.source_game_fingerprint,
                "source_state_id": self.source_state_id,
                "information_set_id": self.information_set_id,
                "public_belief_fingerprint": self.public_belief_fingerprint,
                "public_history_fingerprint": self.public_history_fingerprint,
                "private_observation_fingerprint": (
                    self.private_observation_fingerprint
                ),
            },
            "action_values": [item.to_dict() for item in self.action_values],
            "summary": {
                "blueprint_opponent_value": self.blueprint_opponent_value,
                "best_response_action_id": self.best_response_action_id,
                "best_response_opponent_value": self.best_response_opponent_value,
                "empirical_deviation_gain": self.empirical_deviation_gain,
                "empirical_action_span": self.empirical_action_span,
                "live_replay_coin_count": self.live_replay_coin_count,
            },
            "gates": {
                "visualizer_root_public_parity": self.visualizer_root_public_parity,
                "visualizer_root_acting_private_zone_parity": (
                    self.visualizer_root_acting_private_zone_parity
                ),
                "native_root_public_parity": self.native_root_public_parity,
                "native_root_private_parity": self.native_root_private_parity,
                "native_live_next_public_parity": (
                    self.native_live_next_public_parity
                ),
                "native_live_next_private_parity": (
                    self.native_live_next_private_parity
                ),
                "joint_prefix_truth_value_root": self.joint_prefix_truth_value_root,
                "absolute_chance_reach_available": (
                    self.absolute_chance_reach_available
                ),
                "calibrated_counterfactual_value": (
                    self.calibrated_counterfactual_value
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
            "claims": {
                "realized_hidden_state_official_sample": (
                    self.realized_hidden_state_official_sample
                ),
                "descendant_transitions_official_engine_executed": (
                    self.descendant_transitions_official_engine_executed
                ),
                "hidden_truth_selected_for_policy_input": (
                    self.hidden_truth_selected_for_policy_input
                ),
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
                "abstract_five_plan_and_continuation_policy": True,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "JointOfficialTruthValueSurface":
        truth = raw["truth"]
        root = raw["root"]
        summary = raw["summary"]
        gates = raw["gates"]
        claims = raw["claims"]
        if not bool(claims["abstract_five_plan_and_continuation_policy"]):
            raise ValueError("joint official abstraction claim changed")
        result = cls(
            split=str(raw["split"]),
            candidate_id=str(raw["candidate_id"]),
            prefix_trace=OfficialPrefixReachTrace.from_dict(raw["prefix_trace"]),
            common_chance_ratio=CommonChanceLawReachRatio.from_dict(
                raw["common_chance_ratio"]
            ),
            truth_frame_fingerprint=str(truth["frame_fingerprint"]),
            truth_completion_fingerprint=str(truth["completion_fingerprint"]),
            truth_zone_counts=tuple(
                (str(name), int(truth["zone_counts"][name]))
                for name in (
                    "acting_deck_count",
                    "acting_prize_count",
                    "other_deck_count",
                    "other_prize_count",
                    "other_hand_count",
                    "other_hidden_active_count",
                )
            ),
            source_game_fingerprint=str(root["source_game_fingerprint"]),
            source_state_id=str(root["source_state_id"]),
            information_set_id=str(root["information_set_id"]),
            public_belief_fingerprint=str(root["public_belief_fingerprint"]),
            public_history_fingerprint=str(root["public_history_fingerprint"]),
            private_observation_fingerprint=str(
                root["private_observation_fingerprint"]
            ),
            action_values=tuple(
                OpponentPlanActionValue.from_dict(item)
                for item in raw["action_values"]
            ),
            blueprint_opponent_value=float(summary["blueprint_opponent_value"]),
            best_response_action_id=str(summary["best_response_action_id"]),
            best_response_opponent_value=float(
                summary["best_response_opponent_value"]
            ),
            empirical_deviation_gain=float(summary["empirical_deviation_gain"]),
            empirical_action_span=float(summary["empirical_action_span"]),
            live_replay_coin_count=int(summary["live_replay_coin_count"]),
            visualizer_root_public_parity=bool(
                gates["visualizer_root_public_parity"]
            ),
            visualizer_root_acting_private_zone_parity=bool(
                gates["visualizer_root_acting_private_zone_parity"]
            ),
            native_root_public_parity=bool(gates["native_root_public_parity"]),
            native_root_private_parity=bool(gates["native_root_private_parity"]),
            native_live_next_public_parity=bool(
                gates["native_live_next_public_parity"]
            ),
            native_live_next_private_parity=bool(
                gates["native_live_next_private_parity"]
            ),
            joint_prefix_truth_value_root=bool(
                gates["joint_prefix_truth_value_root"]
            ),
            realized_hidden_state_official_sample=bool(
                claims["realized_hidden_state_official_sample"]
            ),
            descendant_transitions_official_engine_executed=bool(
                claims["descendant_transitions_official_engine_executed"]
            ),
            hidden_truth_selected_for_policy_input=bool(
                claims["hidden_truth_selected_for_policy_input"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
            absolute_chance_reach_available=bool(
                gates["absolute_chance_reach_available"]
            ),
            calibrated_counterfactual_value=bool(
                gates["calibrated_counterfactual_value"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("joint official surface fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class JointOfficialValueDataset:
    source_phase6z_dataset_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    rollouts_per_action: int
    surfaces: tuple[JointOfficialTruthValueSurface, ...]
    native_transition_count: int
    macro_transition_count: int
    complete_split_candidate_support: bool = True
    same_root_prefix_truth_value_gate_passed: bool = True
    official_truth_sample_gate_passed: bool = True
    live_native_replay_fidelity_gate_passed: bool = True
    common_chance_likelihood_ratio_available: bool = True
    absolute_chance_reach_available: bool = False
    repeated_truth_samples_per_information_set: bool = False
    calibrated_counterfactual_values: bool = False
    safe_resolver_compatible: bool = False
    opponent_exploitation_compatible: bool = False
    official_strategy_control_enabled: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_phase6z_dataset_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("joint official candidate support is invalid")
        expected = {
            (split, candidate)
            for split in ALIGNED_VALUE_SPLITS
            for candidate in self.candidate_ids
        }
        observed = {(item.split, item.candidate_id) for item in self.surfaces}
        if observed != expected or len(self.surfaces) != len(expected):
            raise ValueError("joint official split/candidate support is incomplete")
        if len({item.fingerprint for item in self.surfaces}) != len(self.surfaces):
            raise ValueError("joint official surfaces are duplicated")
        if self.rollouts_per_action < 2:
            raise ValueError("joint official values require repeated rollouts")
        if any(
            len(value.terminal_returns) != self.rollouts_per_action
            for item in self.surfaces
            for value in item.action_values
        ):
            raise ValueError("joint official rollout shape is inconsistent")
        if min(self.native_transition_count, self.macro_transition_count) <= 0:
            raise ValueError("joint official transition accounting is invalid")
        if not all(
            (
                self.complete_split_candidate_support,
                self.same_root_prefix_truth_value_gate_passed,
                self.official_truth_sample_gate_passed,
                self.live_native_replay_fidelity_gate_passed,
                self.common_chance_likelihood_ratio_available,
            )
        ):
            raise ValueError("joint official substrate gate failed")
        if any(
            (
                self.absolute_chance_reach_available,
                self.repeated_truth_samples_per_information_set,
                self.calibrated_counterfactual_values,
                self.safe_resolver_compatible,
                self.opponent_exploitation_compatible,
                self.official_strategy_control_enabled,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("joint official dataset promoted an unsupported capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def live_selection_count(self) -> int:
        return sum(len(item.prefix_trace.steps) + 1 for item in self.surfaces)

    @property
    def action_target_count(self) -> int:
        return sum(len(item.action_values) for item in self.surfaces)

    @property
    def terminal_rollout_count(self) -> int:
        return sum(
            len(value.terminal_returns)
            for item in self.surfaces
            for value in item.action_values
        )

    @property
    def maximum_rollout_standard_error(self) -> float:
        return max(
            value.rollout_standard_error
            for item in self.surfaces
            for value in item.action_values
        )

    @property
    def nontrivial_action_surface_fraction(self) -> float:
        return sum(item.empirical_action_span > 1e-12 for item in self.surfaces) / len(
            self.surfaces
        )

    @property
    def maximum_samples_per_information_set(self) -> int:
        counts: dict[str, int] = {}
        for item in self.surfaces:
            counts[item.information_set_id] = counts.get(item.information_set_id, 0) + 1
        return max(counts.values())

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_JOINT_OFFICIAL_VALUE_SCHEMA_VERSION,
            "source_phase6z_dataset_fingerprint": (
                self.source_phase6z_dataset_fingerprint
            ),
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "source_plan_blueprint_fingerprint": (
                self.source_plan_blueprint_fingerprint
            ),
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "rollouts_per_action": self.rollouts_per_action,
            "surfaces": [item.to_dict() for item in self.surfaces],
            "accounting": {
                "root_count": len(self.surfaces),
                "live_selection_count": self.live_selection_count,
                "action_target_count": self.action_target_count,
                "terminal_rollout_count": self.terminal_rollout_count,
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "maximum_rollout_standard_error": (
                    self.maximum_rollout_standard_error
                ),
                "nontrivial_action_surface_fraction": (
                    self.nontrivial_action_surface_fraction
                ),
                "maximum_samples_per_information_set": (
                    self.maximum_samples_per_information_set
                ),
            },
            "gates": {
                "complete_split_candidate_support": (
                    self.complete_split_candidate_support
                ),
                "same_root_prefix_truth_value": (
                    self.same_root_prefix_truth_value_gate_passed
                ),
                "official_truth_sample": self.official_truth_sample_gate_passed,
                "live_native_replay_fidelity": (
                    self.live_native_replay_fidelity_gate_passed
                ),
                "common_chance_likelihood_ratio_available": (
                    self.common_chance_likelihood_ratio_available
                ),
                "absolute_chance_reach_available": (
                    self.absolute_chance_reach_available
                ),
                "repeated_truth_samples_per_information_set": (
                    self.repeated_truth_samples_per_information_set
                ),
                "calibrated_counterfactual_values": (
                    self.calibrated_counterfactual_values
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "opponent_exploitation_compatible": (
                    self.opponent_exploitation_compatible
                ),
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
            "claims": {
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
                "absolute_official_chance_probabilities_inferred": False,
                "official_hidden_state_posterior": False,
                "abstract_five_plan_boundary": True,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "JointOfficialValueDataset":
        accounting = raw["accounting"]
        gates = raw["gates"]
        claims = raw["claims"]
        if any(
            (
                bool(claims["absolute_official_chance_probabilities_inferred"]),
                bool(claims["official_hidden_state_posterior"]),
                not bool(claims["abstract_five_plan_boundary"]),
            )
        ):
            raise ValueError("joint official dataset claim schema changed")
        result = cls(
            source_phase6z_dataset_fingerprint=str(
                raw["source_phase6z_dataset_fingerprint"]
            ),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(
                raw["source_plan_blueprint_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            rollouts_per_action=int(raw["rollouts_per_action"]),
            surfaces=tuple(
                JointOfficialTruthValueSurface.from_dict(item)
                for item in raw["surfaces"]
            ),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            complete_split_candidate_support=bool(
                gates["complete_split_candidate_support"]
            ),
            same_root_prefix_truth_value_gate_passed=bool(
                gates["same_root_prefix_truth_value"]
            ),
            official_truth_sample_gate_passed=bool(gates["official_truth_sample"]),
            live_native_replay_fidelity_gate_passed=bool(
                gates["live_native_replay_fidelity"]
            ),
            common_chance_likelihood_ratio_available=bool(
                gates["common_chance_likelihood_ratio_available"]
            ),
            absolute_chance_reach_available=bool(
                gates["absolute_chance_reach_available"]
            ),
            repeated_truth_samples_per_information_set=bool(
                gates["repeated_truth_samples_per_information_set"]
            ),
            calibrated_counterfactual_values=bool(
                gates["calibrated_counterfactual_values"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            opponent_exploitation_compatible=bool(
                gates["opponent_exploitation_compatible"]
            ),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected_accounting = {
            "root_count": len(result.surfaces),
            "live_selection_count": result.live_selection_count,
            "action_target_count": result.action_target_count,
            "terminal_rollout_count": result.terminal_rollout_count,
            "native_transition_count": result.native_transition_count,
            "macro_transition_count": result.macro_transition_count,
            "maximum_rollout_standard_error": result.maximum_rollout_standard_error,
            "nontrivial_action_surface_fraction": (
                result.nontrivial_action_surface_fraction
            ),
            "maximum_samples_per_information_set": (
                result.maximum_samples_per_information_set
            ),
        }
        if dict(accounting) != expected_accounting:
            raise ValueError("joint official dataset accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("joint official dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6JointOfficialValueEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    root_count: int
    candidate_strata: int
    live_selection_count: int
    action_target_count: int
    terminal_rollout_count: int
    native_transition_count: int
    macro_transition_count: int
    maximum_rollout_standard_error: float
    nontrivial_action_surface_fraction: float
    maximum_samples_per_information_set: int
    same_root_prefix_truth_value_gate_passed: bool
    official_truth_sample_gate_passed: bool
    live_native_replay_fidelity_gate_passed: bool
    common_chance_likelihood_ratio_available: bool
    absolute_chance_reach_available: bool
    calibrated_counterfactual_values: bool
    safe_resolver_enabled: bool
    opponent_exploitation_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AC evidence identity is missing")
        for name in ("dataset_fingerprint", "runtime_fingerprint", "deck_fingerprint"):
            _validate_digest(getattr(self, name), name)
        if min(
            self.root_count,
            self.candidate_strata,
            self.live_selection_count,
            self.action_target_count,
            self.terminal_rollout_count,
            self.native_transition_count,
            self.macro_transition_count,
            self.maximum_samples_per_information_set,
        ) <= 0:
            raise ValueError("Phase 6AC evidence accounting is invalid")
        if self.maximum_rollout_standard_error < 0:
            raise ValueError("Phase 6AC rollout precision is invalid")
        if not 0 <= self.nontrivial_action_surface_fraction <= 1:
            raise ValueError("Phase 6AC action-surface fraction is invalid")
        if not all(
            (
                self.same_root_prefix_truth_value_gate_passed,
                self.official_truth_sample_gate_passed,
                self.live_native_replay_fidelity_gate_passed,
                self.common_chance_likelihood_ratio_available,
            )
        ):
            raise ValueError("Phase 6AC substrate gate failed")
        if any(
            (
                self.absolute_chance_reach_available,
                self.calibrated_counterfactual_values,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AC evidence enabled downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_JOINT_OFFICIAL_VALUE_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AC joint official-value evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Joint live roots | {self.root_count} |
| Candidate strata | {self.candidate_strata} |
| Genuine live selections | {self.live_selection_count} |
| Five-plan action targets | {self.action_target_count} |
| Genuine terminal continuations | {self.terminal_rollout_count} |
| Genuine native transitions | {self.native_transition_count} |
| Maximum rollout SE | `{self.maximum_rollout_standard_error:.6f}` |
| Nontrivial action surfaces | `{self.nontrivial_action_surface_fraction:.6f}` |
| Max truth samples / information set | {self.maximum_samples_per_information_set} |
| Same-root prefix / truth / values | **true** |
| Official truth-sample and replay fidelity | **true** / **true** |
| Common-chance likelihood ratio available | **true** |
| Absolute chance reach available | **false** |
| Calibrated counterfactual values | **false** |
| Safe resolver / exploitation / official control | **false** / **false** / **false** |

Every prefix and realized hidden sample comes from a genuine official battle;
all descendant values execute through the licensed native branch API. Raw truth
is transient and never a policy input or artifact field. The five-plan boundary,
frozen continuation policy, and rollout estimator are abstractions. Identical
target/sampling chance laws give a symbolic ratio of one without disclosing
absolute chance reach. One sample per information set is insufficient for value
calibration or safety, so no resolver, exploit overlay, or official control is
enabled.
"""

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
    ) -> "Phase6JointOfficialValueEvidence":
        fields = {name: raw[name] for name in cls.__dataclass_fields__ if name != "claims"}
        result = cls(
            **fields,
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AC evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6JointOfficialValueArtifacts:
    dataset: JointOfficialValueDataset
    evidence: Phase6JointOfficialValueEvidence


def _build_surface_at_live_root(
    observation: Mapping[str, Any],
    prefix_trace: OfficialPrefixReachTrace,
    *,
    visualize_data: Any,
    battle_select: Any,
    runtime: Path,
    runtime_fingerprint: str,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    style: str,
    tactical_provider: Any,
    plan_provider: NativePlanBlueprintProvider,
    bridge: NativeBlueprintFeatureBridge,
    rollouts_per_action: int,
    maximum_rollout_transitions: int,
    maximum_tactical_transitions: int,
    seed: int,
) -> tuple[JointOfficialTruthValueSurface, int, int]:
    frames = json.loads(visualize_data())
    if not isinstance(frames, Sequence) or not frames or not isinstance(frames[-1], Mapping):
        raise RuntimeError("Phase 6AC visualizer returned no truth frame")
    truth_frame = frames[-1]
    visualizer_public_parity = (
        sanitize_official_public_state(truth_frame)
        == sanitize_official_public_state(observation)
    )
    truth_private = official_private_observation_payload(truth_frame)
    live_private = official_private_observation_payload(observation)
    visualizer_private_zone_parity = all(
        truth_private[name] == live_private[name]
        for name in ("own_hand", "own_looking")
    )
    if not visualizer_public_parity or not visualizer_private_zone_parity:
        raise RuntimeError(
            "Phase 6AC visualizer/live root parity failed: "
            f"public={visualizer_public_parity}, "
            f"private_zones={visualizer_private_zone_parity}"
        )
    completion = _official_truth_completion(
        observation,
        truth_frame,
        target=target,
        candidate=candidate,
        style=style,
    )
    chance_law_fingerprint = _fingerprint(
        {
            "runtime_ruleset": runtime_fingerprint,
            "target_deck": target.fingerprint,
            "candidate": candidate.fingerprint,
            "law": "same opaque official live BattleStart/Select chance law",
        }
    )
    chance_ratio = CommonChanceLawReachRatio(
        target_player=1,
        chance_law_fingerprint=chance_law_fingerprint,
        player_0_target_reach=prefix_trace.declared_player_0_behavior_reach,
        player_1_target_reach=prefix_trace.declared_player_1_behavior_reach,
        player_0_sampling_reach=prefix_trace.declared_player_0_behavior_reach,
        player_1_sampling_reach=prefix_trace.declared_player_1_behavior_reach,
    )
    macro_transitions = 0
    with OfficialNativeSearchGame(
        observation,
        (completion,),
        runtime_directory=runtime,
    ) as native_game:
        chance_root = native_game.root_state()
        scenario = native_game.transition(
            chance_root,
            None,
            chance_outcome=native_game.chance_outcomes(chance_root)[0],
        )
        native_public_parity = (
            sanitize_official_public_state(scenario.observation)
            == sanitize_official_public_state(observation)
        )
        native_private_parity = (
            official_private_observation_payload(scenario.observation)
            == official_private_observation_payload(observation)
        )
        if not native_public_parity or not native_private_parity:
            raise RuntimeError("Phase 6AC truth completion did not reproduce root")

        contract = SelectionContract.from_observation(observation)
        select = observation.get("select")
        options = select.get("option", ()) if isinstance(select, Mapping) else ()
        end_indices = tuple(
            index
            for index, option in enumerate(options)
            if isinstance(option, Mapping)
            and _safe_int(option.get("type"), -1) == int(OptionType.END)
        )
        if len(end_indices) != 1:
            raise RuntimeError("Phase 6AC root has no unique deterministic END action")
        # END advances through the next turn's mandatory draw without invoking a
        # card-effect shuffle.  This validates hidden deck order and future
        # state reconstruction, not merely root-state equality.
        live_selection = contract.validate(end_indices)
        native_action = _matching_action(
            native_game.legal_actions(scenario),
            live_selection,
        )
        native_next = native_game.transition(scenario, native_action)
        live_next = battle_select(list(live_selection))
        if live_next is None:
            raise RuntimeError("Phase 6AC live replay ended without an observation")
        native_next, coin_count = _resolve_live_coin_results(
            native_game,
            native_next,
            live_next,
        )
        next_public_parity = (
            sanitize_official_public_state(native_next.observation)
            == sanitize_official_public_state(live_next)
        )
        next_private_parity = (
            official_private_observation_payload(native_next.observation)
            == official_private_observation_payload(live_next)
        )
        if not next_public_parity or not next_private_parity:
            raise RuntimeError("Phase 6AC native/live next-step replay diverged")

        macro = OfficialNativePlanGame(
            native_game,
            scenario,
            bridge=bridge,
            tactical_provider=tactical_provider,
            maximum_tactical_transitions=maximum_tactical_transitions,
        )
        if macro.player_to_act(macro.root_state()) != 1:
            raise RuntimeError("Phase 6AC root is not a player-1 plan decision")
        state = macro.root_state()
        information_set = macro.information_set(state, player=1)
        belief = macro.public_belief(state, player=1)
        distribution: ActionDistribution = plan_provider.strategy(
            information_set,
            belief,
        )
        distribution.validate_support(information_set)
        if tuple(action.action_id for action in information_set.legal_actions) != (
            OPPONENT_DEVIATION_ACTION_IDS
        ):
            raise RuntimeError("Phase 6AC plan action support drifted")
        private_fingerprint = information_set.private_observation_fingerprint
        if private_fingerprint is None:
            raise RuntimeError("Phase 6AC information set lacks a private digest")
        rollout_seeds = tuple(
            _seed(
                seed,
                "joint-official-truth-rollout",
                prefix_trace.split,
                candidate.archetype,
                rollout,
            )
            for rollout in range(rollouts_per_action)
        )
        action_values = []
        for action in information_set.legal_actions:
            child = macro.transition(state, action)
            macro_transitions += 1
            forced_tactical = child.tactical_transitions - state.tactical_transitions
            returns = []
            depths = []
            rollout_macro_transitions = 0
            for rollout_seed in rollout_seeds:
                value, transitions, terminal_depth = _rollout_average_blueprint(
                    macro,
                    child,
                    plan_provider,
                    value_player=1,
                    rng=random.Random(rollout_seed),
                    maximum_macro_transitions=maximum_rollout_transitions,
                )
                returns.append(value)
                depths.append(terminal_depth)
                rollout_macro_transitions += transitions
                macro_transitions += transitions
            action_values.append(
                OpponentPlanActionValue(
                    action_id=action.action_id,
                    blueprint_probability=distribution.probability(action.action_id),
                    terminal_returns=tuple(returns),
                    empirical_opponent_value=statistics.mean(returns),
                    rollout_standard_error=(
                        statistics.stdev(returns) / math.sqrt(len(returns))
                    ),
                    forced_action_tactical_transitions=forced_tactical,
                    rollout_macro_transitions=rollout_macro_transitions,
                    terminal_native_depth_minimum=min(depths),
                    terminal_native_depth_maximum=max(depths),
                )
            )
        values = tuple(action_values)
        blueprint_value = sum(
            item.blueprint_probability * item.empirical_opponent_value
            for item in values
        )
        best = max(
            values,
            key=lambda item: (
                item.empirical_opponent_value,
                -OPPONENT_DEVIATION_ACTION_IDS.index(item.action_id),
            ),
        )
        span = max(item.empirical_opponent_value for item in values) - min(
            item.empirical_opponent_value for item in values
        )
        surface = JointOfficialTruthValueSurface(
            split=prefix_trace.split,
            candidate_id=candidate.archetype,
            prefix_trace=prefix_trace,
            common_chance_ratio=chance_ratio,
            truth_frame_fingerprint=_fingerprint(truth_frame),
            truth_completion_fingerprint=completion.fingerprint,
            truth_zone_counts=_completion_zone_counts(completion),
            source_game_fingerprint=native_game.fingerprint,
            source_state_id=scenario.state_id,
            information_set_id=information_set.info_set_id,
            public_belief_fingerprint=belief.fingerprint,
            public_history_fingerprint=scenario.public_history_fingerprint,
            private_observation_fingerprint=private_fingerprint,
            action_values=values,
            blueprint_opponent_value=blueprint_value,
            best_response_action_id=best.action_id,
            best_response_opponent_value=best.empirical_opponent_value,
            empirical_deviation_gain=(
                best.empirical_opponent_value - blueprint_value
            ),
            empirical_action_span=span,
            live_replay_coin_count=coin_count,
            visualizer_root_public_parity=visualizer_public_parity,
            visualizer_root_acting_private_zone_parity=(
                visualizer_private_zone_parity
            ),
            native_root_public_parity=native_public_parity,
            native_root_private_parity=native_private_parity,
            native_live_next_public_parity=next_public_parity,
            native_live_next_private_parity=next_private_parity,
        )
        return surface, native_game.transition_count, macro_transitions


def run_phase6_joint_official_values(
    *,
    project_root: str | Path | None = None,
    source_phase6z_dataset_path: str | Path | None = None,
    source_tactical_blueprint_path: str | Path | None = None,
    source_plan_blueprint_path: str | Path | None = None,
    source_phase6p_evidence_path: str | Path | None = None,
    rollouts_per_action: int = JOINT_OFFICIAL_ROLLOUTS_PER_ACTION,
    maximum_live_selections: int = 900,
    maximum_capture_attempts: int = 8,
    maximum_rollout_transitions: int = 512,
    maximum_tactical_transitions: int = 512,
    seed: int = PHASE6_JOINT_OFFICIAL_VALUE_SEED,
) -> Phase6JointOfficialValueArtifacts:
    if rollouts_per_action < 2:
        raise ValueError("Phase 6AC requires repeated terminal rollouts")
    if min(
        maximum_live_selections,
        maximum_capture_attempts,
        maximum_rollout_transitions,
        maximum_tactical_transitions,
    ) <= 0:
        raise ValueError("Phase 6AC limits must be positive")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    source_path = Path(
        source_phase6z_dataset_path
        or root
        / "artifacts"
        / "phase6_aligned_counterfactual_values"
        / "public_root_aligned_value_dataset.json"
    ).resolve()
    tactical_path = Path(
        source_tactical_blueprint_path
        or root / "artifacts" / "phase6_native_blueprint" / "native_regret_blueprint.json"
    ).resolve()
    plan_path = Path(
        source_plan_blueprint_path
        or root
        / "artifacts"
        / "phase6_native_plan_iterations"
        / "iterative_native_plan_blueprint.json"
    ).resolve()
    phase6p_path = Path(
        source_phase6p_evidence_path
        or root
        / "artifacts"
        / "phase6_native_plan_iterations"
        / "phase6p_evidence.json"
    ).resolve()
    source = PublicRootAlignedValueDataset.from_dict(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    tactical_artifact = NativeRegretBlueprintArtifact.from_dict(
        json.loads(tactical_path.read_text(encoding="utf-8"))
    )
    tactical_provider = restore_native_regret_blueprint(tactical_artifact)
    plan_artifact = NativePlanBlueprintArtifact.from_dict(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    plan_provider: NativePlanBlueprintProvider = restore_native_plan_blueprint(
        plan_artifact
    )
    phase6p = Phase6NativePlanIterationsEvidence.from_dict(
        json.loads(phase6p_path.read_text(encoding="utf-8"))
    )
    if phase6p.final_model_fingerprint != plan_artifact.fingerprint:
        raise ValueError("Phase 6AC plan model/evidence mismatch")
    if plan_artifact.source_tactical_blueprint_fingerprint != tactical_artifact.fingerprint:
        raise ValueError("Phase 6AC plan/tactical provenance mismatch")
    if source.runtime_fingerprint != tactical_artifact.runtime_fingerprint:
        raise ValueError("Phase 6AC source/runtime fingerprint mismatch")
    if source.deck_fingerprint != tactical_artifact.deck_fingerprint:
        raise ValueError("Phase 6AC source/deck fingerprint mismatch")

    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    if target.fingerprint != source.deck_fingerprint:
        raise ValueError("Phase 6AC target deck/source mismatch")
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    if set(profile_by_id) != set(source.candidate_ids):
        raise ValueError("Phase 6AC candidate catalog/source mismatch")
    candidates = tuple(profile_by_id[name] for name in source.candidate_ids)
    bridge = NativeBlueprintFeatureBridge()
    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
        visualize_data,
    )

    surfaces: list[JointOfficialTruthValueSurface] = []
    native_transitions = 0
    macro_transitions = 0
    for split in ALIGNED_VALUE_SPLITS:
        style, ordinal, base_reverse = OFFICIAL_PREFIX_CAPTURE_CONFIG[split]
        for candidate_index, candidate in enumerate(candidates):
            capture_error: RuntimeError | None = None
            for capture_attempt in range(1, maximum_capture_attempts + 1):
                reverse = (
                    base_reverse
                    ^ bool(candidate_index % 2)
                    ^ bool((capture_attempt - 1) % 2)
                )
                holder: list[tuple[JointOfficialTruthValueSurface, int, int]] = []

                def at_root(
                    observation: Mapping[str, Any],
                    trace: OfficialPrefixReachTrace,
                ) -> None:
                    holder.append(
                        _build_surface_at_live_root(
                            observation,
                            trace,
                            visualize_data=visualize_data,
                            battle_select=battle_select,
                            runtime=runtime,
                            runtime_fingerprint=source.runtime_fingerprint,
                            target=target,
                            candidate=candidate,
                            style=style,
                            tactical_provider=tactical_provider,
                            plan_provider=plan_provider,
                            bridge=bridge,
                            rollouts_per_action=rollouts_per_action,
                            maximum_rollout_transitions=maximum_rollout_transitions,
                            maximum_tactical_transitions=maximum_tactical_transitions,
                            seed=seed,
                        )
                    )

                try:
                    _capture_official_prefix(
                        battle_start,
                        battle_select,
                        battle_finish,
                        target=target,
                        candidate=candidate,
                        split=split,
                        style=style,
                        ordinal=ordinal,
                        reverse=reverse,
                        capture_attempt=capture_attempt,
                        maximum_live_selections=maximum_live_selections,
                        root_callback=at_root,
                    )
                    if len(holder) != 1:
                        raise RuntimeError("Phase 6AC root callback did not emit one surface")
                    surface, transitions, macro_count = holder[0]
                    surfaces.append(surface)
                    native_transitions += transitions
                    macro_transitions += macro_count
                    break
                except RuntimeError as exc:
                    capture_error = exc
            else:
                raise RuntimeError(
                    "Phase 6AC exhausted same-root collection attempts for "
                    f"{split}/{candidate.archetype}"
                ) from capture_error

    dataset = JointOfficialValueDataset(
        source_phase6z_dataset_fingerprint=source.fingerprint,
        source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
        source_plan_blueprint_fingerprint=plan_artifact.fingerprint,
        runtime_fingerprint=source.runtime_fingerprint,
        deck_fingerprint=target.fingerprint,
        candidate_ids=source.candidate_ids,
        rollouts_per_action=rollouts_per_action,
        surfaces=tuple(surfaces),
        native_transition_count=native_transitions,
        macro_transition_count=macro_transitions,
    )
    if _raw_private_key_count(dataset.to_dict()) != 0:
        raise RuntimeError("Phase 6AC serialized a forbidden raw private key")
    restored = JointOfficialValueDataset.from_dict(dataset.to_dict())
    if restored != dataset:
        raise RuntimeError("Phase 6AC dataset roundtrip changed values")
    evidence = Phase6JointOfficialValueEvidence(
        benchmark_id="phase6ac-joint-official-prefix-truth-values",
        status="joint_substrate_complete_fail_closed",
        dataset_fingerprint=dataset.fingerprint,
        runtime_fingerprint=dataset.runtime_fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        root_count=len(dataset.surfaces),
        candidate_strata=len(dataset.candidate_ids),
        live_selection_count=dataset.live_selection_count,
        action_target_count=dataset.action_target_count,
        terminal_rollout_count=dataset.terminal_rollout_count,
        native_transition_count=dataset.native_transition_count,
        macro_transition_count=dataset.macro_transition_count,
        maximum_rollout_standard_error=dataset.maximum_rollout_standard_error,
        nontrivial_action_surface_fraction=(
            dataset.nontrivial_action_surface_fraction
        ),
        maximum_samples_per_information_set=(
            dataset.maximum_samples_per_information_set
        ),
        same_root_prefix_truth_value_gate_passed=True,
        official_truth_sample_gate_passed=True,
        live_native_replay_fidelity_gate_passed=True,
        common_chance_likelihood_ratio_available=True,
        absolute_chance_reach_available=False,
        calibrated_counterfactual_values=False,
        safe_resolver_enabled=False,
        opponent_exploitation_enabled=False,
        official_strategy_control_enabled=False,
        claims=(
            (
                "genuine",
                "live prefixes, realized official hidden-state samples, root/next-step "
                "replay parity, and licensed native descendant transitions",
            ),
            (
                "symbolic",
                "target/sampling common-chance cancellation and deterministic "
                "on-policy counterfactual likelihood ratio",
            ),
            (
                "abstract",
                "five turn plans, frozen tactical/continuation policies, and rollout values",
            ),
            (
                "unavailable",
                "absolute chance reach, repeated truth samples per information set, "
                "calibrated counterfactual values, safety, exploitation, and control",
            ),
        ),
    )
    restored_evidence = Phase6JointOfficialValueEvidence.from_dict(evidence.to_dict())
    if restored_evidence != evidence:
        raise RuntimeError("Phase 6AC evidence roundtrip changed values")
    return Phase6JointOfficialValueArtifacts(dataset=dataset, evidence=evidence)


def main() -> None:
    print(run_phase6_joint_official_values().evidence.to_json())


if __name__ == "__main__":
    main()
