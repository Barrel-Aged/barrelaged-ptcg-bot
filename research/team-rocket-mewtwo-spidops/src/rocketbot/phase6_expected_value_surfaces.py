"""Phase 6AH repeated root/action/hypothesis expected-value surfaces.

Phase 6AG found transferable point signal in complete official games, but a
single terminal result per state could not support useful expected-value
intervals.  Phase 6AH therefore collects fresh complete matches, retains a
pre-registered late player-0 MAIN snapshot, and executes repeated licensed
native continuations for every abstract plan action behind two independently
sampled, public-consistent hidden allocations.

The complete matches, legal selections, retained native snapshots, and native
descendant transitions are genuine official-engine execution.  The hidden
allocation law, five plan actions, frozen continuation blueprint, neural model,
and empirical intervals are research abstractions.  Even a passing conditional
mean gate is not an equilibrium result, an opponent counterfactual value, a
safe-resolving certificate, official strategy control, or a strength claim.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .beliefs import OPPONENT_INTENTS, OpponentIntent
from .competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    OptionType,
    SelectContext,
    SelectionContract,
    _safe_int,
    _state,
    resolve_official_deck,
)
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
from .neural_blueprint import TensorArtifact
from .official_bridge import (
    official_private_observation_payload,
    sanitize_official_public_state,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    PriorityBaseline,
    rocket_compatibility_profile,
)
from .official_search_game import (
    HiddenZoneCompletion,
    OfficialNativeSearchGame,
    _root_public_history,
    sample_hidden_zone_completion,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _runtime_basics
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence
from .phase6_official_intent_calibration import (
    OFFICIAL_INTENT_POLICY_VARIANT_IDS,
    StrategicIntentPriorityBaseline,
    official_intent_policy_variant_fingerprint,
)
from .phase6_official_policy_generalization import (
    Phase6OfficialPolicyGeneralizationEvidence,
)
from .phase6_official_value_leaf import Phase6OfficialValueLeafEvidence
from .phase6_opponent_deviation_values import (
    OPPONENT_DEVIATION_ACTION_IDS,
    _rollout_average_blueprint,
)
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_EXPECTED_VALUE_SCHEMA_VERSION = 1
PHASE6_EXPECTED_VALUE_SEED = 6_048
PHASE6_EXPECTED_VALUE_TARGET_PLAYER = 0
PHASE6_EXPECTED_VALUE_MINIMUM_OPPONENT_MAIN_TURNS = 2
PHASE6_EXPECTED_VALUE_ROOT_LOOKBACK = 3
PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT = 2
PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION = 8
PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS = 2_000
PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS = 6
PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS = 512
PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS = 512
PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL = (
    ("canonical", "train", 1),
    ("engine_first", "train", 1),
    ("hand_first", "calibration", 2),
    ("action_first", "validation", 1),
)
PHASE6_EXPECTED_VALUE_HIDDEN_GRID = (32, 64)
PHASE6_EXPECTED_VALUE_WEIGHT_DECAY_GRID = (0.0, 1e-4)
PHASE6_EXPECTED_VALUE_BLEND_GRID = (0.50, 0.75, 1.0)
PHASE6_EXPECTED_VALUE_EPOCHS = 50
PHASE6_EXPECTED_VALUE_LEARNING_RATE = 0.01
PHASE6_EXPECTED_VALUE_COVERAGE = 0.90
PHASE6_EXPECTED_VALUE_MEAN_CI_Z = 1.645
PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR = 0.30
PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH = 1.75
PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE = 0.75
PHASE6_EXPECTED_VALUE_ACTION_CATEGORIES = (
    "ability",
    "attach",
    "attack",
    "end",
    "evolve",
    "play",
    "retreat",
    "other",
)
PHASE6_EXPECTED_VALUE_HISTORY_FEATURE_NAMES = tuple(
    f"{side}_main_action_{category}_count"
    for side in ("own", "opponent")
    for category in PHASE6_EXPECTED_VALUE_ACTION_CATEGORIES
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*items: Any) -> int:
    return int.from_bytes(sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big")


def _target_reverse_for_capture(match_id: str, capture_attempt: int) -> bool:
    """Keep attempt one stable and alternate the starting direction on retries."""

    if not match_id or capture_attempt <= 0:
        raise ValueError("capture direction requires a match id and positive attempt")
    base_reverse = bool(_seed(match_id, 1, "target-direction") % 2)
    return base_reverse ^ bool((capture_attempt - 1) % 2)


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _protocol_by_variant() -> dict[str, tuple[str, int]]:
    result = {
        variant: (split, replicas)
        for variant, split, replicas in PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL
    }
    if tuple(result) != OFFICIAL_INTENT_POLICY_VARIANT_IDS:
        raise RuntimeError("Phase 6AH policy-implementation order drifted")
    return result


def _calibration_partition(variant: str, replica: int) -> str:
    split, replicas = _protocol_by_variant()[variant]
    if replica not in range(replicas):
        raise ValueError("Phase 6AH replica is outside its policy protocol")
    if split == "train":
        return "train"
    if split == "calibration":
        return "selection" if replica == 0 else "bounds"
    return "untouched"


def _coarse_action_category(option_type: int) -> str:
    mapping = {
        int(OptionType.ABILITY): "ability",
        int(OptionType.ATTACH): "attach",
        int(OptionType.ATTACK): "attack",
        int(OptionType.END): "end",
        int(OptionType.EVOLVE): "evolve",
        int(OptionType.PLAY): "play",
        int(OptionType.RETREAT): "retreat",
    }
    return mapping.get(int(option_type), "other")


@dataclass(frozen=True, slots=True)
class ExpectedValueActionEstimate:
    action_id: str
    blueprint_probability: float
    terminal_returns: tuple[float, ...]
    empirical_player0_value: float
    rollout_standard_error: float
    forced_action_tactical_transitions: int
    rollout_macro_transitions: int
    terminal_native_depth_minimum: int
    terminal_native_depth_maximum: int

    def __post_init__(self) -> None:
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AH action is outside the five-plan support")
        if not 0 <= _finite(self.blueprint_probability, "blueprint_probability") <= 1:
            raise ValueError("Phase 6AH blueprint probability is invalid")
        if len(self.terminal_returns) != PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION:
            raise ValueError("Phase 6AH per-particle rollout count drifted")
        if any(value not in (-1.0, 0.0, 1.0) for value in self.terminal_returns):
            raise ValueError("Phase 6AH terminal utility is invalid")
        expected_mean = statistics.mean(self.terminal_returns)
        expected_se = statistics.stdev(self.terminal_returns) / math.sqrt(
            len(self.terminal_returns)
        )
        if not math.isclose(self.empirical_player0_value, expected_mean, abs_tol=1e-12):
            raise ValueError("Phase 6AH action mean is inconsistent")
        if not math.isclose(self.rollout_standard_error, expected_se, abs_tol=1e-12):
            raise ValueError("Phase 6AH action standard error is inconsistent")
        if min(self.forced_action_tactical_transitions, self.rollout_macro_transitions) < 0:
            raise ValueError("Phase 6AH transition accounting is invalid")
        if (
            min(self.terminal_native_depth_minimum, self.terminal_native_depth_maximum) <= 0
            or self.terminal_native_depth_minimum > self.terminal_native_depth_maximum
        ):
            raise ValueError("Phase 6AH terminal depth accounting is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "terminal_returns": list(self.terminal_returns),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpectedValueActionEstimate":
        return cls(
            action_id=str(raw["action_id"]),
            blueprint_probability=float(raw["blueprint_probability"]),
            terminal_returns=tuple(float(value) for value in raw["terminal_returns"]),
            empirical_player0_value=float(raw["empirical_player0_value"]),
            rollout_standard_error=float(raw["rollout_standard_error"]),
            forced_action_tactical_transitions=int(
                raw["forced_action_tactical_transitions"]
            ),
            rollout_macro_transitions=int(raw["rollout_macro_transitions"]),
            terminal_native_depth_minimum=int(raw["terminal_native_depth_minimum"]),
            terminal_native_depth_maximum=int(raw["terminal_native_depth_maximum"]),
        )


@dataclass(frozen=True, slots=True)
class ExpectedValueParticleSurface:
    allocation_index: int
    completion_fingerprint: str
    public_evidence_adjustments: int
    source_game_fingerprint: str
    source_state_id: str
    action_values: tuple[ExpectedValueActionEstimate, ...]
    native_transition_count: int
    macro_transition_count: int
    tactical_transition_count: int
    root_replay_public_parity: bool = True
    root_replay_acting_private_parity: bool = True
    hidden_allocation_official: bool = False

    def __post_init__(self) -> None:
        if self.allocation_index not in range(PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT):
            raise ValueError("Phase 6AH allocation index is invalid")
        for name in ("completion_fingerprint", "source_game_fingerprint", "source_state_id"):
            _validate_digest(getattr(self, name), name)
        if self.public_evidence_adjustments != 0:
            raise ValueError("Phase 6AH realized-candidate allocation required an adjustment")
        if tuple(item.action_id for item in self.action_values) != OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AH plan-action support/order drifted")
        if min(
            self.native_transition_count,
            self.macro_transition_count,
            self.tactical_transition_count,
        ) < 0:
            raise ValueError("Phase 6AH surface transition accounting is invalid")
        if not self.root_replay_public_parity or not self.root_replay_acting_private_parity:
            raise ValueError("Phase 6AH retained root did not replay exactly")
        if self.hidden_allocation_official:
            raise ValueError("Phase 6AH promoted its supplied allocation law")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "allocation_index": self.allocation_index,
            "completion_fingerprint": self.completion_fingerprint,
            "public_evidence_adjustments": self.public_evidence_adjustments,
            "source_game_fingerprint": self.source_game_fingerprint,
            "source_state_id": self.source_state_id,
            "action_values": [item.to_dict() for item in self.action_values],
            "accounting": {
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "tactical_transition_count": self.tactical_transition_count,
            },
            "gates": {
                "root_replay_public_parity": self.root_replay_public_parity,
                "root_replay_acting_private_parity": (
                    self.root_replay_acting_private_parity
                ),
                "hidden_allocation_official": self.hidden_allocation_official,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpectedValueParticleSurface":
        accounting = raw["accounting"]
        gates = raw["gates"]
        result = cls(
            allocation_index=int(raw["allocation_index"]),
            completion_fingerprint=str(raw["completion_fingerprint"]),
            public_evidence_adjustments=int(raw["public_evidence_adjustments"]),
            source_game_fingerprint=str(raw["source_game_fingerprint"]),
            source_state_id=str(raw["source_state_id"]),
            action_values=tuple(
                ExpectedValueActionEstimate.from_dict(item)
                for item in raw["action_values"]
            ),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            tactical_transition_count=int(accounting["tactical_transition_count"]),
            root_replay_public_parity=bool(gates["root_replay_public_parity"]),
            root_replay_acting_private_parity=bool(
                gates["root_replay_acting_private_parity"]
            ),
            hidden_allocation_official=bool(gates["hidden_allocation_official"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AH surface fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ExpectedValueRoot:
    match_id: str
    split: str
    calibration_partition: str
    policy_variant_id_supervision_only: str
    policy_variant_fingerprint: str
    candidate_id_hypothesis_input: str
    intent_id_supervision_only: str
    replica: int
    capture_attempt: int
    target_reverse: bool
    opponent_reverse: bool
    match_winner: int
    match_turns: int
    match_selection_count: int
    eligible_root_count: int
    selected_root_lookback: int
    selected_root_turn: int
    selected_root_prizes: tuple[int, int]
    prefix_decision_count: int
    prefix_chain_fingerprint: str
    public_state_fingerprint: str
    public_history_fingerprint: str
    private_observation_fingerprint: str
    search_snapshot_fingerprint: str
    public_features: tuple[tuple[str, float], ...]
    private_features: tuple[tuple[str, float], ...]
    history_features: tuple[tuple[str, float], ...]
    blueprint_probabilities: tuple[tuple[str, float], ...]
    surfaces: tuple[ExpectedValueParticleSurface, ...]
    complete_official_match_executed: bool = True
    retained_post_match_snapshot_replayed: bool = True
    descendant_transitions_official_engine_executed: bool = True
    policy_variant_identifier_selected_for_model_input: bool = False
    intent_identifier_selected_for_model_input: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if not self.match_id or self.policy_variant_id_supervision_only not in _protocol_by_variant():
            raise ValueError("Phase 6AH root identity is invalid")
        expected_split, replicas = _protocol_by_variant()[
            self.policy_variant_id_supervision_only
        ]
        if self.split != expected_split or self.replica not in range(replicas):
            raise ValueError("Phase 6AH root escaped its policy split")
        if self.calibration_partition != _calibration_partition(
            self.policy_variant_id_supervision_only,
            self.replica,
        ):
            raise ValueError("Phase 6AH calibration role drifted")
        if self.policy_variant_fingerprint != official_intent_policy_variant_fingerprint(
            self.policy_variant_id_supervision_only
        ):
            raise ValueError("Phase 6AH policy fingerprint mismatch")
        if self.intent_id_supervision_only not in {item.value for item in OPPONENT_INTENTS}:
            raise ValueError("Phase 6AH intent label is invalid")
        if not self.candidate_id_hypothesis_input:
            raise ValueError("Phase 6AH candidate hypothesis is empty")
        if self.capture_attempt <= 0 or self.target_reverse == self.opponent_reverse:
            raise ValueError("Phase 6AH collection direction/attempt is invalid")
        if self.match_winner not in (0, 1, 2) or min(
            self.match_turns,
            self.match_selection_count,
            self.eligible_root_count,
            self.prefix_decision_count,
        ) <= 0:
            raise ValueError("Phase 6AH complete-match accounting is invalid")
        if not 1 <= self.selected_root_lookback <= PHASE6_EXPECTED_VALUE_ROOT_LOOKBACK:
            raise ValueError("Phase 6AH selected-root lookback is invalid")
        if self.selected_root_turn < 0 or any(
            not 0 <= value <= 6 for value in self.selected_root_prizes
        ):
            raise ValueError("Phase 6AH root progress is invalid")
        for name in (
            "prefix_chain_fingerprint",
            "public_state_fingerprint",
            "public_history_fingerprint",
            "private_observation_fingerprint",
            "search_snapshot_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        for feature_group, group_name in (
            (self.public_features, "public"),
            (self.private_features, "private"),
            (self.history_features, "history"),
        ):
            names = [name for name, _ in feature_group]
            if not names or len(names) != len(set(names)):
                raise ValueError(f"Phase 6AH {group_name} feature schema is invalid")
            for name, value in feature_group:
                if not name:
                    raise ValueError(f"Phase 6AH {group_name} feature name is empty")
                _finite(value, name)
        if tuple(name for name, _ in self.history_features) != (
            PHASE6_EXPECTED_VALUE_HISTORY_FEATURE_NAMES
        ):
            raise ValueError("Phase 6AH history feature schema drifted")
        if tuple(name for name, _ in self.blueprint_probabilities) != (
            OPPONENT_DEVIATION_ACTION_IDS
        ) or not math.isclose(
            sum(value for _, value in self.blueprint_probabilities),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Phase 6AH blueprint distribution is invalid")
        if len(self.surfaces) != PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT or tuple(
            item.allocation_index for item in self.surfaces
        ) != tuple(range(PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT)):
            raise ValueError("Phase 6AH root particle support is incomplete")
        if len({item.completion_fingerprint for item in self.surfaces}) != len(
            self.surfaces
        ):
            raise ValueError("Phase 6AH root particles are not distinct")
        if not all(
            (
                self.complete_official_match_executed,
                self.retained_post_match_snapshot_replayed,
                self.descendant_transitions_official_engine_executed,
            )
        ) or any(
            (
                self.policy_variant_identifier_selected_for_model_input,
                self.intent_identifier_selected_for_model_input,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6AH root promoted an unsupported capability")

    @property
    def root_id(self) -> str:
        return _fingerprint(
            {
                "match_id": self.match_id,
                "public_state": self.public_state_fingerprint,
                "public_history": self.public_history_fingerprint,
                "private_observation": self.private_observation_fingerprint,
                "candidate_hypothesis": self.candidate_id_hypothesis_input,
                "surfaces": [item.fingerprint for item in self.surfaces],
            }
        )

    @property
    def terminal_rollout_count(self) -> int:
        return sum(
            len(action.terminal_returns)
            for surface in self.surfaces
            for action in surface.action_values
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPECTED_VALUE_SCHEMA_VERSION,
            "root_id": self.root_id,
            "match": {
                "match_id": self.match_id,
                "split": self.split,
                "calibration_partition": self.calibration_partition,
                "policy_variant_id_supervision_only": (
                    self.policy_variant_id_supervision_only
                ),
                "policy_variant_fingerprint": self.policy_variant_fingerprint,
                "candidate_id_hypothesis_input": self.candidate_id_hypothesis_input,
                "intent_id_supervision_only": self.intent_id_supervision_only,
                "replica": self.replica,
                "capture_attempt": self.capture_attempt,
                "target_reverse": self.target_reverse,
                "opponent_reverse": self.opponent_reverse,
                "winner": self.match_winner,
                "turns": self.match_turns,
                "selection_count": self.match_selection_count,
            },
            "root": {
                "eligible_root_count": self.eligible_root_count,
                "selected_root_lookback": self.selected_root_lookback,
                "selected_root_turn": self.selected_root_turn,
                "selected_root_prizes": list(self.selected_root_prizes),
                "prefix_decision_count": self.prefix_decision_count,
                "prefix_chain_fingerprint": self.prefix_chain_fingerprint,
                "public_state_fingerprint": self.public_state_fingerprint,
                "public_history_fingerprint": self.public_history_fingerprint,
                "private_observation_fingerprint": (
                    self.private_observation_fingerprint
                ),
                "search_snapshot_fingerprint": self.search_snapshot_fingerprint,
            },
            "features": {
                "public": [list(item) for item in self.public_features],
                "private": [list(item) for item in self.private_features],
                "history": [list(item) for item in self.history_features],
                "blueprint_probabilities": [
                    list(item) for item in self.blueprint_probabilities
                ],
            },
            "surfaces": [item.to_dict() for item in self.surfaces],
            "claims": {
                "complete_official_match_executed": (
                    self.complete_official_match_executed
                ),
                "retained_post_match_snapshot_replayed": (
                    self.retained_post_match_snapshot_replayed
                ),
                "descendant_transitions_official_engine_executed": (
                    self.descendant_transitions_official_engine_executed
                ),
                "policy_variant_identifier_selected_for_model_input": (
                    self.policy_variant_identifier_selected_for_model_input
                ),
                "intent_identifier_selected_for_model_input": (
                    self.intent_identifier_selected_for_model_input
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = _fingerprint(result)
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpectedValueRoot":
        match = raw["match"]
        root = raw["root"]
        features = raw["features"]
        claims = raw["claims"]
        result = cls(
            match_id=str(match["match_id"]),
            split=str(match["split"]),
            calibration_partition=str(match["calibration_partition"]),
            policy_variant_id_supervision_only=str(
                match["policy_variant_id_supervision_only"]
            ),
            policy_variant_fingerprint=str(match["policy_variant_fingerprint"]),
            candidate_id_hypothesis_input=str(match["candidate_id_hypothesis_input"]),
            intent_id_supervision_only=str(match["intent_id_supervision_only"]),
            replica=int(match["replica"]),
            capture_attempt=int(match["capture_attempt"]),
            target_reverse=bool(match["target_reverse"]),
            opponent_reverse=bool(match["opponent_reverse"]),
            match_winner=int(match["winner"]),
            match_turns=int(match["turns"]),
            match_selection_count=int(match["selection_count"]),
            eligible_root_count=int(root["eligible_root_count"]),
            selected_root_lookback=int(root["selected_root_lookback"]),
            selected_root_turn=int(root["selected_root_turn"]),
            selected_root_prizes=tuple(int(value) for value in root["selected_root_prizes"]),
            prefix_decision_count=int(root["prefix_decision_count"]),
            prefix_chain_fingerprint=str(root["prefix_chain_fingerprint"]),
            public_state_fingerprint=str(root["public_state_fingerprint"]),
            public_history_fingerprint=str(root["public_history_fingerprint"]),
            private_observation_fingerprint=str(
                root["private_observation_fingerprint"]
            ),
            search_snapshot_fingerprint=str(root["search_snapshot_fingerprint"]),
            public_features=tuple(
                (str(name), float(value)) for name, value in features["public"]
            ),
            private_features=tuple(
                (str(name), float(value)) for name, value in features["private"]
            ),
            history_features=tuple(
                (str(name), float(value)) for name, value in features["history"]
            ),
            blueprint_probabilities=tuple(
                (str(name), float(value))
                for name, value in features["blueprint_probabilities"]
            ),
            surfaces=tuple(
                ExpectedValueParticleSurface.from_dict(item)
                for item in raw["surfaces"]
            ),
            complete_official_match_executed=bool(
                claims["complete_official_match_executed"]
            ),
            retained_post_match_snapshot_replayed=bool(
                claims["retained_post_match_snapshot_replayed"]
            ),
            descendant_transitions_official_engine_executed=bool(
                claims["descendant_transitions_official_engine_executed"]
            ),
            policy_variant_identifier_selected_for_model_input=bool(
                claims["policy_variant_identifier_selected_for_model_input"]
            ),
            intent_identifier_selected_for_model_input=bool(
                claims["intent_identifier_selected_for_model_input"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        if raw.get("root_id") not in (None, result.root_id):
            raise ValueError("Phase 6AH root ID mismatch")
        expected = result.to_dict()
        if raw.get("fingerprint") not in (None, expected["fingerprint"]):
            raise ValueError("Phase 6AH root fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ExpectedValueDataset:
    source_phase6af_evidence_fingerprint: str
    source_phase6ag_evidence_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    intent_ids: tuple[str, ...]
    policy_protocol: tuple[tuple[str, str, int], ...]
    roots: tuple[ExpectedValueRoot, ...]
    complete_official_matches_executed: bool = True
    retained_snapshots_replayed_after_match: bool = True
    common_random_numbers_across_actions_and_allocations: bool = True
    hidden_allocation_prior_official: bool = False
    continuation_blueprint_equilibrium_validated: bool = False
    counterfactual_values: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_phase6af_evidence_fingerprint",
            "source_phase6ag_evidence_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.policy_protocol != PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL:
            raise ValueError("Phase 6AH policy protocol drifted")
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Phase 6AH candidate catalog is invalid")
        if tuple(name for name, _ in self.candidate_deck_fingerprints) != self.candidate_ids:
            raise ValueError("Phase 6AH candidate fingerprint catalog drifted")
        for _, digest in self.candidate_deck_fingerprints:
            _validate_digest(digest, "candidate_deck_fingerprint")
        if self.intent_ids != tuple(item.value for item in OPPONENT_INTENTS):
            raise ValueError("Phase 6AH intent catalog drifted")
        expected = {
            (variant, candidate, intent, replica)
            for variant, _, replicas in self.policy_protocol
            for candidate in self.candidate_ids
            for intent in self.intent_ids
            for replica in range(replicas)
        }
        observed = {
            (
                item.policy_variant_id_supervision_only,
                item.candidate_id_hypothesis_input,
                item.intent_id_supervision_only,
                item.replica,
            )
            for item in self.roots
        }
        if observed != expected or len(self.roots) != len(expected):
            raise ValueError("Phase 6AH collection grid is incomplete or duplicated")
        if len({item.match_id for item in self.roots}) != len(self.roots):
            raise ValueError("Phase 6AH match IDs are not unique")
        public_schemas = {
            tuple(name for name, _ in item.public_features) for item in self.roots
        }
        private_schemas = {
            tuple(name for name, _ in item.private_features) for item in self.roots
        }
        if len(public_schemas) != 1 or len(private_schemas) != 1:
            raise ValueError("Phase 6AH root feature schemas differ")
        if any(
            (
                not self.complete_official_matches_executed,
                not self.retained_snapshots_replayed_after_match,
                not self.common_random_numbers_across_actions_and_allocations,
                self.hidden_allocation_prior_official,
                self.continuation_blueprint_equilibrium_validated,
                self.counterfactual_values,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6AH dataset promoted an unsupported capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def split_root_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (split, sum(item.split == split for item in self.roots))
            for split in ("train", "calibration", "validation")
        )

    @property
    def terminal_rollout_count(self) -> int:
        return sum(item.terminal_rollout_count for item in self.roots)

    @property
    def live_selection_count(self) -> int:
        return sum(item.match_selection_count for item in self.roots)

    @property
    def native_transition_count(self) -> int:
        return sum(
            surface.native_transition_count
            for root in self.roots
            for surface in root.surfaces
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPECTED_VALUE_SCHEMA_VERSION,
            "sources": {
                "phase6af_evidence_fingerprint": (
                    self.source_phase6af_evidence_fingerprint
                ),
                "phase6ag_evidence_fingerprint": (
                    self.source_phase6ag_evidence_fingerprint
                ),
                "tactical_blueprint_fingerprint": (
                    self.source_tactical_blueprint_fingerprint
                ),
                "plan_blueprint_fingerprint": self.source_plan_blueprint_fingerprint,
                "runtime_fingerprint": self.runtime_fingerprint,
                "deck_fingerprint": self.deck_fingerprint,
            },
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": [
                list(item) for item in self.candidate_deck_fingerprints
            ],
            "intent_ids": list(self.intent_ids),
            "policy_protocol": [list(item) for item in self.policy_protocol],
            "collection_protocol": {
                "target_player": PHASE6_EXPECTED_VALUE_TARGET_PLAYER,
                "minimum_opponent_main_turns": (
                    PHASE6_EXPECTED_VALUE_MINIMUM_OPPONENT_MAIN_TURNS
                ),
                "root_lookback": PHASE6_EXPECTED_VALUE_ROOT_LOOKBACK,
                "particles_per_root": PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT,
                "rollouts_per_particle_action": (
                    PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION
                ),
                "plan_action_ids": list(OPPONENT_DEVIATION_ACTION_IDS),
            },
            "accounting": {
                "root_count": len(self.roots),
                "split_root_counts": dict(self.split_root_counts),
                "surface_count": sum(len(item.surfaces) for item in self.roots),
                "action_target_count": len(self.roots)
                * len(OPPONENT_DEVIATION_ACTION_IDS),
                "terminal_rollout_count": self.terminal_rollout_count,
                "live_selection_count": self.live_selection_count,
                "native_transition_count": self.native_transition_count,
            },
            "roots": [item.to_dict() for item in self.roots],
            "claims": {
                "complete_official_matches_executed": (
                    self.complete_official_matches_executed
                ),
                "retained_snapshots_replayed_after_match": (
                    self.retained_snapshots_replayed_after_match
                ),
                "common_random_numbers_across_actions_and_allocations": (
                    self.common_random_numbers_across_actions_and_allocations
                ),
                "hidden_allocation_prior_official": (
                    self.hidden_allocation_prior_official
                ),
                "continuation_blueprint_equilibrium_validated": (
                    self.continuation_blueprint_equilibrium_validated
                ),
                "counterfactual_values": self.counterfactual_values,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpectedValueDataset":
        sources = raw["sources"]
        claims = raw["claims"]
        protocol = raw["collection_protocol"]
        if (
            int(protocol["target_player"]) != PHASE6_EXPECTED_VALUE_TARGET_PLAYER
            or int(protocol["minimum_opponent_main_turns"])
            != PHASE6_EXPECTED_VALUE_MINIMUM_OPPONENT_MAIN_TURNS
            or int(protocol["root_lookback"]) != PHASE6_EXPECTED_VALUE_ROOT_LOOKBACK
            or int(protocol["particles_per_root"])
            != PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT
            or int(protocol["rollouts_per_particle_action"])
            != PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION
            or tuple(str(value) for value in protocol["plan_action_ids"])
            != OPPONENT_DEVIATION_ACTION_IDS
        ):
            raise ValueError("Phase 6AH serialized collection protocol drifted")
        result = cls(
            source_phase6af_evidence_fingerprint=str(
                sources["phase6af_evidence_fingerprint"]
            ),
            source_phase6ag_evidence_fingerprint=str(
                sources["phase6ag_evidence_fingerprint"]
            ),
            source_tactical_blueprint_fingerprint=str(
                sources["tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(
                sources["plan_blueprint_fingerprint"]
            ),
            runtime_fingerprint=str(sources["runtime_fingerprint"]),
            deck_fingerprint=str(sources["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            candidate_deck_fingerprints=tuple(
                (str(item[0]), str(item[1]))
                for item in raw["candidate_deck_fingerprints"]
            ),
            intent_ids=tuple(str(value) for value in raw["intent_ids"]),
            policy_protocol=tuple(
                (str(item[0]), str(item[1]), int(item[2]))
                for item in raw["policy_protocol"]
            ),
            roots=tuple(ExpectedValueRoot.from_dict(item) for item in raw["roots"]),
            complete_official_matches_executed=bool(
                claims["complete_official_matches_executed"]
            ),
            retained_snapshots_replayed_after_match=bool(
                claims["retained_snapshots_replayed_after_match"]
            ),
            common_random_numbers_across_actions_and_allocations=bool(
                claims["common_random_numbers_across_actions_and_allocations"]
            ),
            hidden_allocation_prior_official=bool(
                claims["hidden_allocation_prior_official"]
            ),
            continuation_blueprint_equilibrium_validated=bool(
                claims["continuation_blueprint_equilibrium_validated"]
            ),
            counterfactual_values=bool(claims["counterfactual_values"]),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AH dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class _CapturedRootSnapshot:
    """Transient official observation retained until native replay is opened."""

    observation: Mapping[str, Any]
    selected_root_turn: int
    selected_root_prizes: tuple[int, int]
    prefix_decision_count: int
    prefix_chain_fingerprint: str
    history_features: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class _CompleteMatchCapture:
    snapshot: _CapturedRootSnapshot
    winner: int
    turns: int
    selection_count: int
    eligible_root_count: int
    selected_root_lookback: int
    capture_attempt: int
    target_reverse: bool
    opponent_reverse: bool


@dataclass(frozen=True, slots=True)
class _ExpectedValueSources:
    root: Path
    runtime: Path
    target: KaggleDeckProfile
    candidates: tuple[KaggleDeckProfile, ...]
    tactical_artifact: NativeRegretBlueprintArtifact
    plan_artifact: NativePlanBlueprintArtifact
    tactical_provider: Any
    plan_provider: NativePlanBlueprintProvider
    phase6af: Phase6OfficialPolicyGeneralizationEvidence
    phase6ag: Phase6OfficialValueLeafEvidence


def _public_prize_counts(observation: Mapping[str, Any]) -> tuple[int, int]:
    public = sanitize_official_public_state(observation)
    players = public.get("players", ())
    if not isinstance(players, Sequence) or len(players) != 2:
        raise RuntimeError("Phase 6AH official root lacks two public players")
    counts = tuple(
        _safe_int(player.get("prize_count"), -1)
        if isinstance(player, Mapping)
        else -1
        for player in players
    )
    if len(counts) != 2 or any(value not in range(7) for value in counts):
        raise RuntimeError("Phase 6AH public prize counts are invalid")
    return counts  # type: ignore[return-value]


def _selected_main_action_types(
    observation: Mapping[str, Any],
    selection: Sequence[int],
) -> tuple[int, ...]:
    select = observation.get("select")
    if not isinstance(select, Mapping) or _safe_int(select.get("context"), -1) != int(
        SelectContext.MAIN
    ):
        return ()
    options = select.get("option", ())
    if not isinstance(options, Sequence):
        raise RuntimeError("Phase 6AH MAIN options are malformed")
    result = []
    for index in selection:
        option = options[index]
        if not isinstance(option, Mapping):
            raise RuntimeError("Phase 6AH selected MAIN option is malformed")
        result.append(_safe_int(option.get("type"), -1))
    return tuple(result)


def _capture_complete_match_late_root(
    *,
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    intent: OpponentIntent,
    policy_variant: str,
    replica: int,
    capture_attempt: int,
    maximum_live_selections: int,
    match_id: str,
) -> _CompleteMatchCapture:
    target_reverse = _target_reverse_for_capture(match_id, capture_attempt)
    opponent_reverse = not target_reverse
    observation, start = battle_start(list(target.card_ids), list(candidate.card_ids))
    if observation is None:
        raise RuntimeError(
            "official engine rejected Phase 6AH decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    policies = (
        PriorityBaseline(
            target.card_ids,
            style="tempo",
            reverse=target_reverse,
        ),
        StrategicIntentPriorityBaseline(
            candidate.card_ids,
            intent=intent,
            reverse=opponent_reverse,
            variant_id=policy_variant,
        ),
    )
    opponent_main_turns: set[int] = set()
    retained_target_turns: set[int] = set()
    eligible: list[_CapturedRootSnapshot] = []
    history = Counter({name: 0 for name in PHASE6_EXPECTED_VALUE_HISTORY_FEATURE_NAMES})
    chain = _fingerprint(
        {
            "protocol": "phase6ah-prefix-chain-v1",
            "match": match_id,
            "attempt": capture_attempt,
        }
    )
    selections = 0
    try:
        while (
            observation is not None
            and _safe_int(_state(observation).get("result"), -1) == -1
            and selections < maximum_live_selections
        ):
            current = _state(observation)
            actor = _safe_int(current.get("yourIndex"), -1)
            turn = _safe_int(current.get("turn"), -1)
            select = observation.get("select")
            if actor not in (0, 1) or turn < 0 or not isinstance(select, Mapping):
                raise RuntimeError("Phase 6AH official decision state is malformed")
            context = _safe_int(select.get("context"), -1)
            if actor == 1 and context == int(SelectContext.MAIN):
                opponent_main_turns.add(turn)
            if (
                actor == PHASE6_EXPECTED_VALUE_TARGET_PLAYER
                and context == int(SelectContext.MAIN)
                and len(opponent_main_turns)
                >= PHASE6_EXPECTED_VALUE_MINIMUM_OPPONENT_MAIN_TURNS
                and turn not in retained_target_turns
                and observation.get("search_begin_input") is not None
            ):
                retained_target_turns.add(turn)
                eligible.append(
                    _CapturedRootSnapshot(
                        observation=deepcopy(dict(observation)),
                        selected_root_turn=turn,
                        selected_root_prizes=_public_prize_counts(observation),
                        prefix_decision_count=selections,
                        prefix_chain_fingerprint=chain,
                        history_features=tuple(
                            (name, float(history[name]))
                            for name in PHASE6_EXPECTED_VALUE_HISTORY_FEATURE_NAMES
                        ),
                    )
                )
            contract = SelectionContract.from_observation(observation)
            selection = contract.validate(policies[actor].choose(observation))
            public_before = sanitize_official_public_state(observation)
            public_history_before = _root_public_history(observation)
            selected_types = _selected_main_action_types(observation, selection)
            next_observation = battle_select(list(selection))
            selections += 1
            if next_observation is None:
                raise RuntimeError("Phase 6AH battle ended without an observation")
            chain = _fingerprint(
                {
                    "previous": chain,
                    "decision": selections,
                    "actor": actor,
                    "turn": turn,
                    "context": context,
                    "contract": asdict(contract),
                    "selected_count": len(selection),
                    "selected_public_main_action_types": selected_types,
                    "public_before": public_before,
                    "public_history_before": public_history_before,
                    "public_after": sanitize_official_public_state(next_observation),
                }
            )
            side = "own" if actor == PHASE6_EXPECTED_VALUE_TARGET_PLAYER else "opponent"
            for option_type in selected_types:
                history[
                    f"{side}_main_action_{_coarse_action_category(option_type)}_count"
                ] += 1
            observation = next_observation
        if observation is None:
            raise RuntimeError("Phase 6AH battle lost its terminal observation")
        winner = _safe_int(_state(observation).get("result"), -1)
        turns = _safe_int(_state(observation).get("turn"), -1)
        if winner not in (0, 1, 2):
            raise RuntimeError(
                "Phase 6AH match failed to terminate within "
                f"{maximum_live_selections} selections"
            )
        if not eligible:
            raise RuntimeError("Phase 6AH match produced no eligible player-0 MAIN root")
        lookback = min(PHASE6_EXPECTED_VALUE_ROOT_LOOKBACK, len(eligible))
        selected = eligible[-lookback]
        return _CompleteMatchCapture(
            snapshot=selected,
            winner=winner,
            turns=turns,
            selection_count=selections,
            eligible_root_count=len(eligible),
            selected_root_lookback=lookback,
            capture_attempt=capture_attempt,
            target_reverse=target_reverse,
            opponent_reverse=opponent_reverse,
        )
    finally:
        battle_finish()


def _load_expected_value_sources(project_root: str | Path | None) -> _ExpectedValueSources:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    tactical_path = (
        root / "artifacts" / "phase6_native_blueprint" / "native_regret_blueprint.json"
    )
    plan_path = (
        root
        / "artifacts"
        / "phase6_native_plan_iterations"
        / "iterative_native_plan_blueprint.json"
    )
    phase6p_path = (
        root / "artifacts" / "phase6_native_plan_iterations" / "phase6p_evidence.json"
    )
    phase6af_path = (
        root
        / "artifacts"
        / "phase6_official_policy_generalization"
        / "phase6af_evidence.json"
    )
    phase6ag_path = (
        root / "artifacts" / "phase6_official_value_leaf" / "phase6ag_evidence.json"
    )
    tactical = NativeRegretBlueprintArtifact.from_dict(
        json.loads(tactical_path.read_text(encoding="utf-8"))
    )
    plan = NativePlanBlueprintArtifact.from_dict(
        json.loads(plan_path.read_text(encoding="utf-8"))
    )
    phase6p = Phase6NativePlanIterationsEvidence.from_dict(
        json.loads(phase6p_path.read_text(encoding="utf-8"))
    )
    phase6af = Phase6OfficialPolicyGeneralizationEvidence.from_dict(
        json.loads(phase6af_path.read_text(encoding="utf-8"))
    )
    phase6ag = Phase6OfficialValueLeafEvidence.from_dict(
        json.loads(phase6ag_path.read_text(encoding="utf-8"))
    )
    if phase6p.final_model_fingerprint != plan.fingerprint:
        raise ValueError("Phase 6AH plan evidence/model mismatch")
    if plan.source_tactical_blueprint_fingerprint != tactical.fingerprint:
        raise ValueError("Phase 6AH plan/tactical provenance mismatch")
    if phase6p.runtime_fingerprint != tactical.runtime_fingerprint:
        raise ValueError("Phase 6AH runtime provenance mismatch")
    if not phase6af.implementation_heldout_joint_range_calibrated or any(
        (
            phase6af.human_latent_intent_calibrated,
            phase6af.counterfactual_value_compatible,
        )
    ):
        raise ValueError("Phase 6AH requires fail-closed Phase 6AF range evidence")
    if not phase6ag.implementation_heldout_point_leaf_validated or any(
        (
            phase6ag.counterfactual_value_compatible,
            phase6ag.safe_resolver_enabled,
        )
    ):
        raise ValueError("Phase 6AH requires fail-closed Phase 6AG point evidence")
    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profiles = {profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)}
    if set(profiles) != set(tactical.candidate_ids):
        raise ValueError("Phase 6AH candidate catalog/blueprint mismatch")
    if target.fingerprint != tactical.deck_fingerprint or target.fingerprint != plan.deck_fingerprint:
        raise ValueError("Phase 6AH target deck/blueprint mismatch")
    candidates = tuple(profiles[name] for name in tactical.candidate_ids)
    return _ExpectedValueSources(
        root=root,
        runtime=runtime,
        target=target,
        candidates=candidates,
        tactical_artifact=tactical,
        plan_artifact=plan,
        tactical_provider=restore_native_regret_blueprint(tactical),
        plan_provider=restore_native_plan_blueprint(plan),
        phase6af=phase6af,
        phase6ag=phase6ag,
    )


def _build_expected_value_root(
    capture: _CompleteMatchCapture,
    *,
    sources: _ExpectedValueSources,
    candidate: KaggleDeckProfile,
    intent: OpponentIntent,
    policy_variant: str,
    split: str,
    replica: int,
    match_id: str,
    seed: int,
    maximum_rollout_transitions: int,
    maximum_tactical_transitions: int,
    experiment_label: str = "Phase 6AH",
    action_branch_observer: Callable[
        [int, Any, Any, Any, OfficialNativePlanGame, Any], None
    ]
    | None = None,
) -> ExpectedValueRoot:
    if not experiment_label:
        raise ValueError("expected-value experiment label cannot be empty")
    observation = capture.snapshot.observation
    if _safe_int(_state(observation).get("yourIndex"), -1) != 0:
        raise RuntimeError("Phase 6AH retained root is not player 0")
    select = observation.get("select")
    if not isinstance(select, Mapping) or _safe_int(select.get("context"), -1) != int(
        SelectContext.MAIN
    ):
        raise RuntimeError("Phase 6AH retained root is not a MAIN decision")
    basics = _runtime_basics(sources.runtime)
    completions = tuple(
        sample_hidden_zone_completion(
            observation,
            hypothesis_id=f"{match_id}:allocation:{allocation_index}",
            probability=1.0 / PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT,
            allocation_seed=_seed(seed, match_id, "allocation", allocation_index),
            your_candidate_id=sources.target.archetype,
            opponent_candidate_id=candidate.archetype,
            your_archetype_id=sources.target.archetype,
            opponent_archetype_id=candidate.archetype,
            your_style_id="tempo_priority",
            opponent_style_id=f"declared_{intent.value}",
            your_card_ids=sources.target.card_ids,
            opponent_card_ids=candidate.card_ids,
            your_deck_fingerprint=sources.target.fingerprint,
            opponent_deck_fingerprint=candidate.fingerprint,
            basic_card_ids=basics,
            source=(
                f"{experiment_label} independently sampled public-consistent realized-"
                "candidate allocation; research prior, not official belief"
            ),
        )
        for allocation_index in range(PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT)
    )
    if any(item.public_evidence_adjustments != 0 for item in completions):
        raise RuntimeError("Phase 6AH realized candidate required public-card adjustment")
    rollout_seeds = tuple(
        _seed(seed, match_id, "paired-terminal-rollout", rollout)
        for rollout in range(PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION)
    )
    bridge = NativeBlueprintFeatureBridge()
    surfaces = []
    root_public_features: tuple[tuple[str, float], ...] | None = None
    root_private_features: tuple[tuple[str, float], ...] | None = None
    root_blueprint: tuple[tuple[str, float], ...] | None = None
    root_information_set_id: str | None = None
    root_belief_fingerprint: str | None = None
    with OfficialNativeSearchGame(
        observation,
        completions,
        runtime_directory=sources.runtime,
    ) as native_game:
        chance_root = native_game.root_state()
        outcomes = native_game.chance_outcomes(chance_root)
        if len(outcomes) != PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT:
            raise RuntimeError("Phase 6AH hidden-allocation support drifted")
        outcome_by_id = {item.outcome_id: item for item in outcomes}
        for allocation_index, completion in enumerate(completions):
            native_before = native_game.transition_count
            scenario = native_game.transition(
                chance_root,
                None,
                chance_outcome=outcome_by_id[
                    f"completion:{completion.fingerprint}"
                ],
            )
            public_parity = sanitize_official_public_state(scenario.observation) == (
                sanitize_official_public_state(observation)
            )
            private_parity = official_private_observation_payload(scenario.observation) == (
                official_private_observation_payload(observation)
            )
            if not public_parity or not private_parity:
                raise RuntimeError("Phase 6AH retained native root replay diverged")
            macro = OfficialNativePlanGame(
                native_game,
                scenario,
                bridge=bridge,
                tactical_provider=sources.tactical_provider,
                maximum_tactical_transitions=maximum_tactical_transitions,
            )
            state = macro.root_state()
            if macro.player_to_act(state) != PHASE6_EXPECTED_VALUE_TARGET_PLAYER:
                raise RuntimeError("Phase 6AH macro root is not a player-0 decision")
            information_set = macro.information_set(state, player=0)
            belief = macro.public_belief(state, player=0)
            distribution: ActionDistribution = sources.plan_provider.strategy(
                information_set,
                belief,
            )
            distribution.validate_support(information_set)
            if tuple(action.action_id for action in information_set.legal_actions) != (
                OPPONENT_DEVIATION_ACTION_IDS
            ):
                raise RuntimeError("Phase 6AH five-plan action support drifted")
            public_features = tuple(
                (str(name), float(value)) for name, value in belief.public_features
            )
            private_features = tuple(
                (str(name), float(value)) for name, value in information_set.private_features
            )
            blueprint = tuple(
                (action.action_id, distribution.probability(action.action_id))
                for action in information_set.legal_actions
            )
            if root_public_features is None:
                root_public_features = public_features
                root_private_features = private_features
                root_blueprint = blueprint
                root_information_set_id = information_set.info_set_id
                root_belief_fingerprint = belief.fingerprint
            elif (
                public_features != root_public_features
                or private_features != root_private_features
                or blueprint != root_blueprint
                or information_set.info_set_id != root_information_set_id
                or belief.fingerprint != root_belief_fingerprint
            ):
                raise RuntimeError("Phase 6AH policy-visible root changed across allocations")
            action_values = []
            for action in information_set.legal_actions:
                child = macro.transition(state, action)
                forced_tactical = child.tactical_transitions - state.tactical_transitions
                if action_branch_observer is not None:
                    action_branch_observer(
                        allocation_index,
                        action,
                        state,
                        child,
                        macro,
                        belief,
                    )
                returns = []
                depths = []
                rollout_macro_transitions = 0
                for rollout_seed in rollout_seeds:
                    value, transitions, terminal_depth = _rollout_average_blueprint(
                        macro,
                        child,
                        sources.plan_provider,
                        value_player=PHASE6_EXPECTED_VALUE_TARGET_PLAYER,
                        rng=random.Random(rollout_seed),
                        maximum_macro_transitions=maximum_rollout_transitions,
                    )
                    returns.append(float(value))
                    depths.append(terminal_depth)
                    rollout_macro_transitions += transitions
                action_values.append(
                    ExpectedValueActionEstimate(
                        action_id=action.action_id,
                        blueprint_probability=distribution.probability(action.action_id),
                        terminal_returns=tuple(returns),
                        empirical_player0_value=statistics.mean(returns),
                        rollout_standard_error=(
                            statistics.stdev(returns) / math.sqrt(len(returns))
                        ),
                        forced_action_tactical_transitions=forced_tactical,
                        rollout_macro_transitions=rollout_macro_transitions,
                        terminal_native_depth_minimum=min(depths),
                        terminal_native_depth_maximum=max(depths),
                    )
                )
            surfaces.append(
                ExpectedValueParticleSurface(
                    allocation_index=allocation_index,
                    completion_fingerprint=completion.fingerprint,
                    public_evidence_adjustments=completion.public_evidence_adjustments,
                    source_game_fingerprint=native_game.fingerprint,
                    source_state_id=scenario.state_id,
                    action_values=tuple(action_values),
                    native_transition_count=native_game.transition_count - native_before,
                    macro_transition_count=macro.macro_transition_count,
                    tactical_transition_count=macro.tactical_transition_count,
                    root_replay_public_parity=public_parity,
                    root_replay_acting_private_parity=private_parity,
                )
            )
    if root_public_features is None or root_private_features is None or root_blueprint is None:
        raise RuntimeError("Phase 6AH root produced no value surfaces")
    root = ExpectedValueRoot(
        match_id=match_id,
        split=split,
        calibration_partition=_calibration_partition(policy_variant, replica),
        policy_variant_id_supervision_only=policy_variant,
        policy_variant_fingerprint=official_intent_policy_variant_fingerprint(
            policy_variant
        ),
        candidate_id_hypothesis_input=candidate.archetype,
        intent_id_supervision_only=intent.value,
        replica=replica,
        capture_attempt=capture.capture_attempt,
        target_reverse=capture.target_reverse,
        opponent_reverse=capture.opponent_reverse,
        match_winner=capture.winner,
        match_turns=capture.turns,
        match_selection_count=capture.selection_count,
        eligible_root_count=capture.eligible_root_count,
        selected_root_lookback=capture.selected_root_lookback,
        selected_root_turn=capture.snapshot.selected_root_turn,
        selected_root_prizes=capture.snapshot.selected_root_prizes,
        prefix_decision_count=capture.snapshot.prefix_decision_count,
        prefix_chain_fingerprint=capture.snapshot.prefix_chain_fingerprint,
        public_state_fingerprint=_fingerprint(
            sanitize_official_public_state(observation)
        ),
        public_history_fingerprint=_root_public_history(observation),
        private_observation_fingerprint=_fingerprint(
            official_private_observation_payload(observation)
        ),
        search_snapshot_fingerprint=_fingerprint(
            str(observation["search_begin_input"])
        ),
        public_features=root_public_features,
        private_features=root_private_features,
        history_features=capture.snapshot.history_features,
        blueprint_probabilities=root_blueprint,
        surfaces=tuple(surfaces),
    )
    if _raw_private_key_count(root.to_dict()) != 0:
        raise RuntimeError("Phase 6AH root serialized a forbidden raw private key")
    if ExpectedValueRoot.from_dict(json.loads(root.to_json())) != root:
        raise RuntimeError("Phase 6AH root roundtrip changed values")
    return root


def collect_expected_value_dataset(
    *,
    project_root: str | Path | None = None,
    existing_roots: Sequence[ExpectedValueRoot] = (),
    root_observer: Callable[[ExpectedValueRoot, int, int], None] | None = None,
    maximum_live_selections: int = PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    maximum_capture_attempts: int = PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    maximum_rollout_transitions: int = PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    maximum_tactical_transitions: int = PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    seed: int = PHASE6_EXPECTED_VALUE_SEED,
) -> ExpectedValueDataset:
    if min(
        maximum_live_selections,
        maximum_capture_attempts,
        maximum_rollout_transitions,
        maximum_tactical_transitions,
        seed,
    ) <= 0:
        raise ValueError("Phase 6AH collection limits and seed must be positive")
    sources = _load_expected_value_sources(project_root)
    runtime_text = str(sources.runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
    )

    expected_keys = tuple(
        (variant, candidate.archetype, intent.value, replica)
        for variant, _, replicas in PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL
        for candidate in sources.candidates
        for intent in OPPONENT_INTENTS
        for replica in range(replicas)
    )
    materialized = tuple(existing_roots)
    by_key = {
        (
            item.policy_variant_id_supervision_only,
            item.candidate_id_hypothesis_input,
            item.intent_id_supervision_only,
            item.replica,
        ): item
        for item in materialized
    }
    if len(by_key) != len(materialized) or not set(by_key).issubset(expected_keys):
        raise ValueError("Phase 6AH resume roots are duplicate or outside the frozen grid")
    candidate_by_id = {item.archetype: item for item in sources.candidates}
    intent_by_id = {item.value: item for item in OPPONENT_INTENTS}
    total = len(expected_keys)
    for ordinal, key in enumerate(expected_keys, 1):
        if key in by_key:
            continue
        variant, candidate_id, intent_id, replica = key
        split, _ = _protocol_by_variant()[variant]
        candidate = candidate_by_id[candidate_id]
        intent = intent_by_id[intent_id]
        match_id = (
            f"phase6ah-{split}-{variant}-{candidate_id}-{intent_id}-r{replica}"
        )
        last_error: Exception | None = None
        for attempt in range(1, maximum_capture_attempts + 1):
            try:
                capture = _capture_complete_match_late_root(
                    battle_start=battle_start,
                    battle_select=battle_select,
                    battle_finish=battle_finish,
                    target=sources.target,
                    candidate=candidate,
                    intent=intent,
                    policy_variant=variant,
                    replica=replica,
                    capture_attempt=attempt,
                    maximum_live_selections=maximum_live_selections,
                    match_id=match_id,
                )
                root = _build_expected_value_root(
                    capture,
                    sources=sources,
                    candidate=candidate,
                    intent=intent,
                    policy_variant=variant,
                    split=split,
                    replica=replica,
                    match_id=match_id,
                    seed=seed,
                    maximum_rollout_transitions=maximum_rollout_transitions,
                    maximum_tactical_transitions=maximum_tactical_transitions,
                )
                by_key[key] = root
                if root_observer is not None:
                    root_observer(root, len(by_key), total)
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                "Phase 6AH exhausted capture attempts for "
                f"{variant}/{candidate_id}/{intent_id}/r{replica}"
            ) from last_error
    roots = tuple(by_key[key] for key in expected_keys)
    dataset = ExpectedValueDataset(
        source_phase6af_evidence_fingerprint=sources.phase6af.fingerprint,
        source_phase6ag_evidence_fingerprint=sources.phase6ag.fingerprint,
        source_tactical_blueprint_fingerprint=sources.tactical_artifact.fingerprint,
        source_plan_blueprint_fingerprint=sources.plan_artifact.fingerprint,
        runtime_fingerprint=sources.tactical_artifact.runtime_fingerprint,
        deck_fingerprint=sources.target.fingerprint,
        candidate_ids=tuple(item.archetype for item in sources.candidates),
        candidate_deck_fingerprints=tuple(
            (item.archetype, item.fingerprint) for item in sources.candidates
        ),
        intent_ids=tuple(item.value for item in OPPONENT_INTENTS),
        policy_protocol=PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL,
        roots=roots,
    )
    serialized = dataset.to_dict()
    if _raw_private_key_count(serialized) != 0:
        raise RuntimeError("Phase 6AH dataset serialized a forbidden raw private key")
    restored = ExpectedValueDataset.from_dict(json.loads(json.dumps(serialized)))
    if restored != dataset:
        raise RuntimeError("Phase 6AH dataset roundtrip changed values")
    return dataset


@dataclass(frozen=True, slots=True)
class ExpectedValueRow:
    row_id: str
    root_id: str
    match_id: str
    split: str
    calibration_partition: str
    candidate_id_hypothesis_input: str
    intent_id_supervision_only: str
    policy_variant_id_supervision_only: str
    action_id: str
    features: tuple[tuple[str, float], ...]
    heuristic_value: float
    terminal_returns: tuple[float, ...]
    empirical_player0_value: float
    target_standard_error: float

    def __post_init__(self) -> None:
        for name in ("row_id", "root_id"):
            _validate_digest(getattr(self, name), name)
        if not self.match_id or self.split not in {"train", "calibration", "validation"}:
            raise ValueError("Phase 6AH row identity/split is invalid")
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AH row action is invalid")
        if self.intent_id_supervision_only not in {item.value for item in OPPONENT_INTENTS}:
            raise ValueError("Phase 6AH row intent audit label is invalid")
        if self.policy_variant_id_supervision_only not in _protocol_by_variant():
            raise ValueError("Phase 6AH row policy audit label is invalid")
        names = tuple(name for name, _ in self.features)
        if not names or len(names) != len(set(names)):
            raise ValueError("Phase 6AH row feature schema is invalid")
        if any(
            forbidden in name
            for name in names
            for forbidden in ("intent_id", "policy_variant", "supervision_only")
        ):
            raise ValueError("Phase 6AH row leaked a supervision-only label")
        if not any(name.startswith("candidate_hypothesis::") for name in names):
            raise ValueError("Phase 6AH row omitted its supplied candidate hypothesis")
        if not any(name.startswith("plan_action::") for name in names):
            raise ValueError("Phase 6AH row omitted its plan action")
        for name, value in self.features:
            _finite(value, name)
        if not -1 <= _finite(self.heuristic_value, "heuristic_value") <= 1:
            raise ValueError("Phase 6AH heuristic is outside [-1, 1]")
        expected_count = (
            PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT
            * PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION
        )
        if len(self.terminal_returns) != expected_count or any(
            value not in (-1.0, 0.0, 1.0) for value in self.terminal_returns
        ):
            raise ValueError("Phase 6AH aggregated terminal returns are invalid")
        mean = statistics.mean(self.terminal_returns)
        standard_error = statistics.stdev(self.terminal_returns) / math.sqrt(
            len(self.terminal_returns)
        )
        if not math.isclose(self.empirical_player0_value, mean, abs_tol=1e-12):
            raise ValueError("Phase 6AH row target mean is inconsistent")
        if not math.isclose(self.target_standard_error, standard_error, abs_tol=1e-12):
            raise ValueError("Phase 6AH row target standard error is inconsistent")


def _root_heuristic_value(root: ExpectedValueRoot) -> float:
    public = dict(root.public_features)
    prize_advantage = (
        public["opponent_prize_count"] - public["own_prize_count"]
    ) / 6.0
    hp_advantage = (
        public.get("own_board_hp", 0.0) - public.get("opponent_board_hp", 0.0)
    ) / 600.0
    board_advantage = (
        public["own_board_count"] - public["opponent_board_count"]
    ) / 6.0
    return math.tanh(
        2.0 * prize_advantage + 0.35 * hp_advantage + 0.20 * board_advantage
    )


def expected_value_feature_items(
    root: ExpectedValueRoot,
    action_id: str,
    candidate_ids: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    if action_id not in OPPONENT_DEVIATION_ACTION_IDS:
        raise ValueError("Phase 6AH feature action is unsupported")
    if root.candidate_id_hypothesis_input not in candidate_ids:
        raise ValueError("Phase 6AH root candidate is outside the feature catalog")
    return (
        tuple((f"public::{name}", value) for name, value in root.public_features)
        + tuple((f"private::{name}", value) for name, value in root.private_features)
        + tuple((f"history::{name}", value) for name, value in root.history_features)
        + tuple(
            (
                f"candidate_hypothesis::{candidate_id}",
                1.0 if candidate_id == root.candidate_id_hypothesis_input else 0.0,
            )
            for candidate_id in candidate_ids
        )
        + tuple(
            (
                f"plan_action::{candidate_action}",
                1.0 if candidate_action == action_id else 0.0,
            )
            for candidate_action in OPPONENT_DEVIATION_ACTION_IDS
        )
    )


def build_expected_value_rows(dataset: ExpectedValueDataset) -> tuple[ExpectedValueRow, ...]:
    rows = []
    for root in dataset.roots:
        by_allocation = {
            surface.allocation_index: {
                item.action_id: item for item in surface.action_values
            }
            for surface in root.surfaces
        }
        for action_id in OPPONENT_DEVIATION_ACTION_IDS:
            terminal_returns = tuple(
                value
                for allocation_index in range(PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT)
                for value in by_allocation[allocation_index][action_id].terminal_returns
            )
            features = expected_value_feature_items(
                root,
                action_id,
                dataset.candidate_ids,
            )
            row_id = _fingerprint(
                {
                    "root": root.root_id,
                    "action": action_id,
                    "features": features,
                    "returns": terminal_returns,
                }
            )
            rows.append(
                ExpectedValueRow(
                    row_id=row_id,
                    root_id=root.root_id,
                    match_id=root.match_id,
                    split=root.split,
                    calibration_partition=root.calibration_partition,
                    candidate_id_hypothesis_input=(
                        root.candidate_id_hypothesis_input
                    ),
                    intent_id_supervision_only=root.intent_id_supervision_only,
                    policy_variant_id_supervision_only=(
                        root.policy_variant_id_supervision_only
                    ),
                    action_id=action_id,
                    features=features,
                    heuristic_value=_root_heuristic_value(root),
                    terminal_returns=terminal_returns,
                    empirical_player0_value=statistics.mean(terminal_returns),
                    target_standard_error=(
                        statistics.stdev(terminal_returns)
                        / math.sqrt(len(terminal_returns))
                    ),
                )
            )
    if len(rows) != len(dataset.roots) * len(OPPONENT_DEVIATION_ACTION_IDS):
        raise RuntimeError("Phase 6AH row accounting drifted")
    if len({item.row_id for item in rows}) != len(rows):
        raise ValueError("Phase 6AH row identities collided")
    if len({tuple(name for name, _ in item.features) for item in rows}) != 1:
        raise ValueError("Phase 6AH row feature schemas differ")
    return tuple(rows)


class _ExpectedValueNetwork(nn.Module):
    def __init__(self, input_width: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_width, hidden_dim, dtype=torch.float64)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64)
        self.output_layer = nn.Linear(hidden_dim, 1, dtype=torch.float64)

    def forward(self, features: Tensor) -> Tensor:
        hidden = torch.tanh(self.input_layer(features))
        hidden = torch.tanh(self.hidden_layer(hidden))
        return torch.tanh(self.output_layer(hidden).squeeze(-1))


def _expected_value_tensors(network: _ExpectedValueNetwork) -> tuple[TensorArtifact, ...]:
    return tuple(
        TensorArtifact(
            name=name,
            shape=tuple(tensor.shape),
            values=tuple(float(value) for value in tensor.detach().reshape(-1).tolist()),
        )
        for name, tensor in network.state_dict().items()
    )


def _restore_expected_value_network(
    *,
    normalizer: NamedFeatureNormalizer,
    hidden_dim: int,
    tensors: Sequence[TensorArtifact],
) -> _ExpectedValueNetwork:
    network = _ExpectedValueNetwork(len(normalizer.names), hidden_dim)
    expected = network.state_dict()
    supplied = {item.name: item for item in tensors}
    if set(expected) != set(supplied):
        raise ValueError("Phase 6AH tensor names do not match the architecture")
    restored = {}
    for name, template in expected.items():
        item = supplied[name]
        if item.shape != tuple(template.shape):
            raise ValueError(f"Phase 6AH tensor shape mismatch: {name}")
        restored[name] = torch.tensor(item.values, dtype=torch.float64).reshape(
            item.shape
        )
    network.load_state_dict(restored, strict=True)
    network.eval()
    return network


def _mse(targets: Sequence[float], predictions: Sequence[float]) -> float:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Phase 6AH MSE vectors are empty or misaligned")
    return sum(
        (prediction - target) ** 2
        for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)


def _action_contrast_mse(
    rows: Sequence[ExpectedValueRow],
    predictions: Sequence[float],
) -> float:
    grouped: dict[str, list[tuple[ExpectedValueRow, float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        grouped[row.root_id].append((row, float(prediction)))
    errors = []
    for root_rows in grouped.values():
        if len(root_rows) != len(OPPONENT_DEVIATION_ACTION_IDS):
            raise ValueError("Phase 6AH contrast root lacks five actions")
        target_mean = statistics.mean(
            item.empirical_player0_value for item, _ in root_rows
        )
        prediction_mean = statistics.mean(value for _, value in root_rows)
        errors.extend(
            (
                (prediction - prediction_mean)
                - (row.empirical_player0_value - target_mean)
            )
            ** 2
            for row, prediction in root_rows
        )
    return sum(errors) / len(errors)


def _zero_contrast_mse(rows: Sequence[ExpectedValueRow]) -> float:
    return _action_contrast_mse(rows, [0.0] * len(rows))


def _expected_value_training_tensors(
    rows: Sequence[ExpectedValueRow],
    normalizer: NamedFeatureNormalizer,
) -> tuple[Tensor, Tensor]:
    features = torch.tensor(
        [normalizer.transform(row.features) for row in rows],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [row.empirical_player0_value for row in rows],
        dtype=torch.float64,
    )
    return features, targets


def _train_expected_value_network(
    rows: Sequence[ExpectedValueRow],
    normalizer: NamedFeatureNormalizer,
    *,
    hidden_dim: int,
    weight_decay: float,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> tuple[_ExpectedValueNetwork, float, float]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = _ExpectedValueNetwork(len(normalizer.names), hidden_dim)
    features, targets = _expected_value_training_tensors(rows, normalizer)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    def loss_value() -> Tensor:
        return torch.mean((network(features) - targets) ** 2)

    with torch.no_grad():
        initial = float(loss_value())
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for _ in range(epochs):
            loss = loss_value()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
            optimizer.step()
    finally:
        torch.set_num_threads(previous_threads)
    with torch.no_grad():
        final = float(loss_value())
    if not final < initial:
        raise RuntimeError("Phase 6AH neural expected-value loss did not decrease")
    network.eval()
    return network, initial, final


@dataclass(frozen=True, slots=True)
class ExpectedValueSelectionMetric:
    hidden_dim: int
    weight_decay: float
    blend_weight: float
    initial_training_loss: float
    final_training_loss: float
    selection_mse: float
    selection_zero_mse: float
    selection_heuristic_mse: float
    selection_action_contrast_mse: float
    selection_zero_action_contrast_mse: float
    selection_score: float

    def __post_init__(self) -> None:
        if self.hidden_dim not in PHASE6_EXPECTED_VALUE_HIDDEN_GRID:
            raise ValueError("Phase 6AH selected an unsupported hidden width")
        if self.weight_decay not in PHASE6_EXPECTED_VALUE_WEIGHT_DECAY_GRID:
            raise ValueError("Phase 6AH selected unsupported weight decay")
        if self.blend_weight not in PHASE6_EXPECTED_VALUE_BLEND_GRID:
            raise ValueError("Phase 6AH selected an unsupported blend")
        for name, value in asdict(self).items():
            if name == "hidden_dim":
                continue
            if float(value) < 0 or not math.isfinite(float(value)):
                raise ValueError(f"Phase 6AH selection metric {name} is invalid")
        if not self.final_training_loss < self.initial_training_loss:
            raise ValueError("Phase 6AH training loss did not decrease")


@dataclass(frozen=True, slots=True)
class ExpectedValueModel:
    source_dataset_fingerprint: str
    normalizer: NamedFeatureNormalizer
    seed: int
    epochs: int
    learning_rate: float
    selected_hidden_dim: int
    selected_weight_decay: float
    selected_blend_weight: float
    selection_metrics: tuple[ExpectedValueSelectionMetric, ...]
    train_root_ids: tuple[str, ...]
    calibration_selection_root_ids: tuple[str, ...]
    tensors: tuple[TensorArtifact, ...]
    policy_visible_features_plus_candidate_and_action_only: bool = True
    intent_identifier_as_feature: bool = False
    policy_variant_identifier_as_feature: bool = False
    official_engine_executed: bool = False
    equilibrium_trained: bool = False
    counterfactual_value_provider: bool = False
    safe_resolver_compatible: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.source_dataset_fingerprint, "source_dataset_fingerprint")
        if min(self.seed, self.epochs, self.selected_hidden_dim) <= 0:
            raise ValueError("Phase 6AH model configuration is invalid")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("Phase 6AH model learning rate is invalid")
        if self.selected_hidden_dim not in PHASE6_EXPECTED_VALUE_HIDDEN_GRID:
            raise ValueError("Phase 6AH model hidden width is unsupported")
        if self.selected_weight_decay not in PHASE6_EXPECTED_VALUE_WEIGHT_DECAY_GRID:
            raise ValueError("Phase 6AH model weight decay is unsupported")
        if self.selected_blend_weight not in PHASE6_EXPECTED_VALUE_BLEND_GRID:
            raise ValueError("Phase 6AH model blend is unsupported")
        if not self.selection_metrics or not self.tensors:
            raise ValueError("Phase 6AH model diagnostics/tensors are empty")
        selected = min(
            self.selection_metrics,
            key=lambda item: (
                item.selection_score,
                item.selection_mse,
                item.selection_action_contrast_mse,
                item.hidden_dim,
                item.weight_decay,
                -item.blend_weight,
            ),
        )
        if (
            self.selected_hidden_dim,
            self.selected_weight_decay,
            self.selected_blend_weight,
        ) != (selected.hidden_dim, selected.weight_decay, selected.blend_weight):
            raise ValueError("Phase 6AH selected model record is inconsistent")
        if not self.train_root_ids or not self.calibration_selection_root_ids:
            raise ValueError("Phase 6AH model split roots are empty")
        if set(self.train_root_ids) & set(self.calibration_selection_root_ids):
            raise ValueError("Phase 6AH train and selection roots overlap")
        if not self.policy_visible_features_plus_candidate_and_action_only or any(
            (
                self.intent_identifier_as_feature,
                self.policy_variant_identifier_as_feature,
                self.official_engine_executed,
                self.equilibrium_trained,
                self.counterfactual_value_provider,
                self.safe_resolver_compatible,
            )
        ):
            raise ValueError("Phase 6AH model promoted its abstraction boundary")
        _restore_expected_value_network(
            normalizer=self.normalizer,
            hidden_dim=self.selected_hidden_dim,
            tensors=self.tensors,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPECTED_VALUE_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "normalizer": self.normalizer.to_dict(),
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "selected_hidden_dim": self.selected_hidden_dim,
            "selected_weight_decay": self.selected_weight_decay,
            "selected_blend_weight": self.selected_blend_weight,
            "selection_metrics": [asdict(item) for item in self.selection_metrics],
            "train_root_ids": list(self.train_root_ids),
            "calibration_selection_root_ids": list(
                self.calibration_selection_root_ids
            ),
            "tensors": [
                {
                    "name": item.name,
                    "shape": list(item.shape),
                    "values": list(item.values),
                }
                for item in self.tensors
            ],
            "claims": {
                "policy_visible_features_plus_candidate_and_action_only": (
                    self.policy_visible_features_plus_candidate_and_action_only
                ),
                "intent_identifier_as_feature": self.intent_identifier_as_feature,
                "policy_variant_identifier_as_feature": (
                    self.policy_variant_identifier_as_feature
                ),
                "official_engine_executed": self.official_engine_executed,
                "equilibrium_trained": self.equilibrium_trained,
                "counterfactual_value_provider": self.counterfactual_value_provider,
                "safe_resolver_compatible": self.safe_resolver_compatible,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpectedValueModel":
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            normalizer=NamedFeatureNormalizer.from_dict(raw["normalizer"]),
            seed=int(raw["seed"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            selected_hidden_dim=int(raw["selected_hidden_dim"]),
            selected_weight_decay=float(raw["selected_weight_decay"]),
            selected_blend_weight=float(raw["selected_blend_weight"]),
            selection_metrics=tuple(
                ExpectedValueSelectionMetric(**dict(item))
                for item in raw["selection_metrics"]
            ),
            train_root_ids=tuple(str(value) for value in raw["train_root_ids"]),
            calibration_selection_root_ids=tuple(
                str(value) for value in raw["calibration_selection_root_ids"]
            ),
            tensors=tuple(
                TensorArtifact(
                    name=str(item["name"]),
                    shape=tuple(int(value) for value in item["shape"]),
                    values=tuple(float(value) for value in item["values"]),
                )
                for item in raw["tensors"]
            ),
            policy_visible_features_plus_candidate_and_action_only=bool(
                claims["policy_visible_features_plus_candidate_and_action_only"]
            ),
            intent_identifier_as_feature=bool(claims["intent_identifier_as_feature"]),
            policy_variant_identifier_as_feature=bool(
                claims["policy_variant_identifier_as_feature"]
            ),
            official_engine_executed=bool(claims["official_engine_executed"]),
            equilibrium_trained=bool(claims["equilibrium_trained"]),
            counterfactual_value_provider=bool(
                claims["counterfactual_value_provider"]
            ),
            safe_resolver_compatible=bool(claims["safe_resolver_compatible"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AH model fingerprint mismatch")
        return result


class ExpectedValueProvider:
    """Conditional root/action mean provider; explicitly not counterfactual."""

    def __init__(self, model: ExpectedValueModel) -> None:
        self.model = model
        self.network = _restore_expected_value_network(
            normalizer=model.normalizer,
            hidden_dim=model.selected_hidden_dim,
            tensors=model.tensors,
        )

    def predict_features(
        self,
        features: Sequence[tuple[str, float]],
        *,
        heuristic_value: float,
    ) -> float:
        normalized = self.model.normalizer.transform(features)
        with torch.no_grad():
            neural = float(
                self.network(torch.tensor([normalized], dtype=torch.float64))[0]
            )
        blend = self.model.selected_blend_weight
        return max(
            -1.0,
            min(1.0, blend * neural + (1.0 - blend) * heuristic_value),
        )

    def predict_row(self, row: ExpectedValueRow) -> float:
        return self.predict_features(
            row.features,
            heuristic_value=row.heuristic_value,
        )


def restore_expected_value_model(model: ExpectedValueModel) -> ExpectedValueProvider:
    return ExpectedValueProvider(model)


def train_expected_value_model(
    dataset: ExpectedValueDataset,
    *,
    seed: int = PHASE6_EXPECTED_VALUE_SEED,
    epochs: int = PHASE6_EXPECTED_VALUE_EPOCHS,
    learning_rate: float = PHASE6_EXPECTED_VALUE_LEARNING_RATE,
) -> ExpectedValueModel:
    rows = build_expected_value_rows(dataset)
    train_rows = tuple(row for row in rows if row.split == "train")
    selection_rows = tuple(
        row
        for row in rows
        if row.split == "calibration" and row.calibration_partition == "selection"
    )
    if not train_rows or not selection_rows:
        raise ValueError("Phase 6AH train/selection rows are empty")
    normalizer = NamedFeatureNormalizer.fit(
        [row.features for row in train_rows],
        fit_rows_fingerprint=_fingerprint([row.row_id for row in train_rows]),
    )
    targets = [row.empirical_player0_value for row in selection_rows]
    zero_mse = _mse(targets, [0.0] * len(targets))
    heuristic_mse = _mse(
        targets,
        [row.heuristic_value for row in selection_rows],
    )
    zero_contrast_mse = _zero_contrast_mse(selection_rows)
    candidates: list[tuple[ExpectedValueSelectionMetric, _ExpectedValueNetwork]] = []
    for hidden_dim in PHASE6_EXPECTED_VALUE_HIDDEN_GRID:
        for weight_decay in PHASE6_EXPECTED_VALUE_WEIGHT_DECAY_GRID:
            network, initial, final = _train_expected_value_network(
                train_rows,
                normalizer,
                hidden_dim=hidden_dim,
                weight_decay=weight_decay,
                seed=seed + hidden_dim + int(weight_decay * 1e8),
                epochs=epochs,
                learning_rate=learning_rate,
            )
            with torch.no_grad():
                neural = network(
                    torch.tensor(
                        [normalizer.transform(row.features) for row in selection_rows],
                        dtype=torch.float64,
                    )
                ).tolist()
            for blend in PHASE6_EXPECTED_VALUE_BLEND_GRID:
                predictions = [
                    max(
                        -1.0,
                        min(
                            1.0,
                            blend * float(value)
                            + (1.0 - blend) * row.heuristic_value,
                        ),
                    )
                    for row, value in zip(selection_rows, neural, strict=True)
                ]
                mse = _mse(targets, predictions)
                contrast_mse = _action_contrast_mse(selection_rows, predictions)
                reference = max(min(zero_mse, heuristic_mse), 1e-12)
                contrast_reference = max(zero_contrast_mse, 1e-12)
                metric = ExpectedValueSelectionMetric(
                    hidden_dim=hidden_dim,
                    weight_decay=weight_decay,
                    blend_weight=blend,
                    initial_training_loss=initial,
                    final_training_loss=final,
                    selection_mse=mse,
                    selection_zero_mse=zero_mse,
                    selection_heuristic_mse=heuristic_mse,
                    selection_action_contrast_mse=contrast_mse,
                    selection_zero_action_contrast_mse=zero_contrast_mse,
                    selection_score=max(
                        mse / reference,
                        contrast_mse / contrast_reference,
                    ),
                )
                candidates.append((metric, network))
    selected_metric, selected_network = min(
        candidates,
        key=lambda item: (
            item[0].selection_score,
            item[0].selection_mse,
            item[0].selection_action_contrast_mse,
            item[0].hidden_dim,
            item[0].weight_decay,
            -item[0].blend_weight,
        ),
    )
    model = ExpectedValueModel(
        source_dataset_fingerprint=dataset.fingerprint,
        normalizer=normalizer,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        selected_hidden_dim=selected_metric.hidden_dim,
        selected_weight_decay=selected_metric.weight_decay,
        selected_blend_weight=selected_metric.blend_weight,
        selection_metrics=tuple(metric for metric, _ in candidates),
        train_root_ids=tuple(sorted({row.root_id for row in train_rows})),
        calibration_selection_root_ids=tuple(
            sorted({row.root_id for row in selection_rows})
        ),
        tensors=_expected_value_tensors(selected_network),
    )
    restored = ExpectedValueModel.from_dict(json.loads(model.to_json()))
    if restored.fingerprint != model.fingerprint:
        raise RuntimeError("Phase 6AH model roundtrip drifted")
    return model


@dataclass(frozen=True, slots=True)
class ExpectedValuePointMetric:
    role: str
    row_count: int
    root_count: int
    mse: float
    zero_mse: float
    heuristic_mse: float
    action_contrast_mse: float
    zero_action_contrast_mse: float

    def __post_init__(self) -> None:
        if not self.role or min(self.row_count, self.root_count) <= 0:
            raise ValueError("Phase 6AH point metric identity is invalid")
        for name in (
            "mse",
            "zero_mse",
            "heuristic_mse",
            "action_contrast_mse",
            "zero_action_contrast_mse",
        ):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"Phase 6AH point metric {name} is invalid")


@dataclass(frozen=True, slots=True)
class ExpectedValueSliceMetric:
    dimension: str
    slice_id: str
    validation_row_count: int
    validation_mse: float
    validation_coverage: float
    validation_mean_width: float

    def __post_init__(self) -> None:
        if self.dimension not in {"action", "candidate", "intent"} or not self.slice_id:
            raise ValueError("Phase 6AH interval slice identity is invalid")
        if self.validation_row_count <= 0:
            raise ValueError("Phase 6AH interval slice is empty")
        if self.validation_mse < 0 or not math.isfinite(self.validation_mse):
            raise ValueError("Phase 6AH slice MSE is invalid")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("Phase 6AH slice coverage is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AH slice width is invalid")


def _expected_value_point_metric(
    role: str,
    rows: Sequence[ExpectedValueRow],
    predictions: Sequence[float],
) -> ExpectedValuePointMetric:
    targets = [row.empirical_player0_value for row in rows]
    return ExpectedValuePointMetric(
        role=role,
        row_count=len(rows),
        root_count=len({row.root_id for row in rows}),
        mse=_mse(targets, predictions),
        zero_mse=_mse(targets, [0.0] * len(rows)),
        heuristic_mse=_mse(
            targets,
            [row.heuristic_value for row in rows],
        ),
        action_contrast_mse=_action_contrast_mse(rows, predictions),
        zero_action_contrast_mse=_zero_contrast_mse(rows),
    )


def _expected_value_conformal_radius(
    rows: Sequence[ExpectedValueRow],
    predictions: Sequence[float],
) -> float:
    residuals = sorted(
        abs(prediction - row.empirical_player0_value)
        + PHASE6_EXPECTED_VALUE_MEAN_CI_Z * row.target_standard_error
        for row, prediction in zip(rows, predictions, strict=True)
    )
    if not residuals:
        raise ValueError("Phase 6AH conformal residual set is empty")
    rank = min(
        len(residuals),
        math.ceil((len(residuals) + 1) * PHASE6_EXPECTED_VALUE_COVERAGE),
    )
    return min(2.0, residuals[rank - 1])


def _interval_result(
    row: ExpectedValueRow,
    prediction: float,
    radius: float,
) -> tuple[bool, float]:
    predicted_lower = max(-1.0, prediction - radius)
    predicted_upper = min(1.0, prediction + radius)
    target_lower = max(
        -1.0,
        row.empirical_player0_value
        - PHASE6_EXPECTED_VALUE_MEAN_CI_Z * row.target_standard_error,
    )
    target_upper = min(
        1.0,
        row.empirical_player0_value
        + PHASE6_EXPECTED_VALUE_MEAN_CI_Z * row.target_standard_error,
    )
    covered = (
        predicted_lower <= target_lower + 1e-12
        and target_upper <= predicted_upper + 1e-12
    )
    return covered, predicted_upper - predicted_lower


@dataclass(frozen=True, slots=True)
class ExpectedValueCalibration:
    source_dataset_fingerprint: str
    source_model_fingerprint: str
    requested_coverage: float
    mean_confidence_z: float
    calibration_bound_root_ids: tuple[str, ...]
    validation_root_ids: tuple[str, ...]
    conformal_radius: float
    point_metrics: tuple[ExpectedValuePointMetric, ...]
    slice_metrics: tuple[ExpectedValueSliceMetric, ...]
    maximum_target_standard_error: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    sampling_precision_gate_passed: bool
    point_model_gate_passed: bool
    interval_gate_passed: bool
    implementation_holdout_gate_passed: bool
    conditional_expected_mean_model_admitted: bool
    counterfactual_value_compatible: bool = False
    safe_resolver_compatible: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("source_dataset_fingerprint", "source_model_fingerprint"):
            _validate_digest(getattr(self, name), name)
        if self.requested_coverage != PHASE6_EXPECTED_VALUE_COVERAGE:
            raise ValueError("Phase 6AH requested coverage drifted")
        if self.mean_confidence_z != PHASE6_EXPECTED_VALUE_MEAN_CI_Z:
            raise ValueError("Phase 6AH empirical mean confidence level drifted")
        if not self.calibration_bound_root_ids or not self.validation_root_ids:
            raise ValueError("Phase 6AH bound/validation roots are empty")
        if set(self.calibration_bound_root_ids) & set(self.validation_root_ids):
            raise ValueError("Phase 6AH bound and validation roots overlap")
        if not 0 <= self.conformal_radius <= 2:
            raise ValueError("Phase 6AH conformal radius is invalid")
        if tuple(item.role for item in self.point_metrics) != (
            "calibration_selection",
            "calibration_bounds",
            "validation_untouched",
        ):
            raise ValueError("Phase 6AH point metrics are incomplete")
        if not self.slice_metrics:
            raise ValueError("Phase 6AH slice metrics are empty")
        if not 0 <= self.maximum_target_standard_error <= 1:
            raise ValueError("Phase 6AH target standard error is invalid")
        for name in (
            "validation_coverage",
            "minimum_validation_slice_coverage",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Phase 6AH {name} is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AH validation interval width is invalid")
        expected_sampling = self.maximum_target_standard_error <= (
            PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
        )
        if self.sampling_precision_gate_passed != expected_sampling:
            raise ValueError("Phase 6AH sampling precision gate is inconsistent")
        validation = self.point_metrics[-1]
        expected_point = (
            validation.mse < min(validation.zero_mse, validation.heuristic_mse)
            and validation.action_contrast_mse
            < validation.zero_action_contrast_mse
        )
        if self.point_model_gate_passed != expected_point:
            raise ValueError("Phase 6AH point-model gate is inconsistent")
        expected_interval = (
            self.validation_coverage >= PHASE6_EXPECTED_VALUE_COVERAGE
            and self.validation_mean_width
            <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
            and self.minimum_validation_slice_coverage
            >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
        )
        if self.interval_gate_passed != expected_interval:
            raise ValueError("Phase 6AH interval gate is inconsistent")
        expected_admission = all(
            (
                self.sampling_precision_gate_passed,
                self.point_model_gate_passed,
                self.interval_gate_passed,
                self.implementation_holdout_gate_passed,
            )
        )
        if self.conditional_expected_mean_model_admitted != expected_admission:
            raise ValueError("Phase 6AH expected-mean admission gate is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.safe_resolver_compatible,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AH calibration promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPECTED_VALUE_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
            "requested_coverage": self.requested_coverage,
            "mean_confidence_z": self.mean_confidence_z,
            "calibration_bound_root_ids": list(self.calibration_bound_root_ids),
            "validation_root_ids": list(self.validation_root_ids),
            "conformal_radius": self.conformal_radius,
            "point_metrics": [asdict(item) for item in self.point_metrics],
            "slice_metrics": [asdict(item) for item in self.slice_metrics],
            "maximum_target_standard_error": self.maximum_target_standard_error,
            "validation_coverage": self.validation_coverage,
            "validation_mean_width": self.validation_mean_width,
            "minimum_validation_slice_coverage": (
                self.minimum_validation_slice_coverage
            ),
            "gates": {
                "sampling_precision_gate_passed": self.sampling_precision_gate_passed,
                "point_model_gate_passed": self.point_model_gate_passed,
                "interval_gate_passed": self.interval_gate_passed,
                "implementation_holdout_gate_passed": (
                    self.implementation_holdout_gate_passed
                ),
                "conditional_expected_mean_model_admitted": (
                    self.conditional_expected_mean_model_admitted
                ),
                "counterfactual_value_compatible": (
                    self.counterfactual_value_compatible
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "opponent_exploitation_enabled": self.opponent_exploitation_enabled,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExpectedValueCalibration":
        gates = raw["gates"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            requested_coverage=float(raw["requested_coverage"]),
            mean_confidence_z=float(raw["mean_confidence_z"]),
            calibration_bound_root_ids=tuple(
                str(value) for value in raw["calibration_bound_root_ids"]
            ),
            validation_root_ids=tuple(
                str(value) for value in raw["validation_root_ids"]
            ),
            conformal_radius=float(raw["conformal_radius"]),
            point_metrics=tuple(
                ExpectedValuePointMetric(**dict(item))
                for item in raw["point_metrics"]
            ),
            slice_metrics=tuple(
                ExpectedValueSliceMetric(**dict(item))
                for item in raw["slice_metrics"]
            ),
            maximum_target_standard_error=float(
                raw["maximum_target_standard_error"]
            ),
            validation_coverage=float(raw["validation_coverage"]),
            validation_mean_width=float(raw["validation_mean_width"]),
            minimum_validation_slice_coverage=float(
                raw["minimum_validation_slice_coverage"]
            ),
            sampling_precision_gate_passed=bool(
                gates["sampling_precision_gate_passed"]
            ),
            point_model_gate_passed=bool(gates["point_model_gate_passed"]),
            interval_gate_passed=bool(gates["interval_gate_passed"]),
            implementation_holdout_gate_passed=bool(
                gates["implementation_holdout_gate_passed"]
            ),
            conditional_expected_mean_model_admitted=bool(
                gates["conditional_expected_mean_model_admitted"]
            ),
            counterfactual_value_compatible=bool(
                gates["counterfactual_value_compatible"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            opponent_exploitation_enabled=bool(
                gates["opponent_exploitation_enabled"]
            ),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AH calibration fingerprint mismatch")
        return result


def calibrate_expected_value_model(
    dataset: ExpectedValueDataset,
    model: ExpectedValueModel,
) -> ExpectedValueCalibration:
    if model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("Phase 6AH model/dataset mismatch")
    rows = build_expected_value_rows(dataset)
    selection_rows = tuple(
        row
        for row in rows
        if row.split == "calibration" and row.calibration_partition == "selection"
    )
    bound_rows = tuple(
        row
        for row in rows
        if row.split == "calibration" and row.calibration_partition == "bounds"
    )
    validation_rows = tuple(row for row in rows if row.split == "validation")
    if not selection_rows or not bound_rows or not validation_rows:
        raise ValueError("Phase 6AH calibration partitions are empty")
    provider = restore_expected_value_model(model)
    selection_predictions = [provider.predict_row(row) for row in selection_rows]
    bound_predictions = [provider.predict_row(row) for row in bound_rows]
    validation_predictions = [provider.predict_row(row) for row in validation_rows]
    radius = _expected_value_conformal_radius(bound_rows, bound_predictions)
    interval_results = [
        _interval_result(row, prediction, radius)
        for row, prediction in zip(
            validation_rows,
            validation_predictions,
            strict=True,
        )
    ]
    covered = [item[0] for item in interval_results]
    widths = [item[1] for item in interval_results]

    slice_metrics = []
    slice_specs = (
        (
            "action",
            OPPONENT_DEVIATION_ACTION_IDS,
            lambda row: row.action_id,
        ),
        (
            "candidate",
            dataset.candidate_ids,
            lambda row: row.candidate_id_hypothesis_input,
        ),
        (
            "intent",
            dataset.intent_ids,
            lambda row: row.intent_id_supervision_only,
        ),
    )
    for dimension, identifiers, selector in slice_specs:
        for identifier in identifiers:
            indices = [
                index
                for index, row in enumerate(validation_rows)
                if selector(row) == identifier
            ]
            if not indices:
                raise RuntimeError(f"Phase 6AH validation slice is empty: {dimension}/{identifier}")
            slice_metrics.append(
                ExpectedValueSliceMetric(
                    dimension=dimension,
                    slice_id=identifier,
                    validation_row_count=len(indices),
                    validation_mse=_mse(
                        [
                            validation_rows[index].empirical_player0_value
                            for index in indices
                        ],
                        [validation_predictions[index] for index in indices],
                    ),
                    validation_coverage=(
                        sum(covered[index] for index in indices) / len(indices)
                    ),
                    validation_mean_width=(
                        sum(widths[index] for index in indices) / len(indices)
                    ),
                )
            )
    point_metrics = (
        _expected_value_point_metric(
            "calibration_selection",
            selection_rows,
            selection_predictions,
        ),
        _expected_value_point_metric(
            "calibration_bounds",
            bound_rows,
            bound_predictions,
        ),
        _expected_value_point_metric(
            "validation_untouched",
            validation_rows,
            validation_predictions,
        ),
    )
    validation = point_metrics[-1]
    maximum_se = max(row.target_standard_error for row in rows)
    overall_coverage = sum(covered) / len(covered)
    overall_width = sum(widths) / len(widths)
    minimum_slice_coverage = min(item.validation_coverage for item in slice_metrics)
    sampling_gate = (
        maximum_se <= PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
    )
    point_gate = (
        validation.mse < min(validation.zero_mse, validation.heuristic_mse)
        and validation.action_contrast_mse < validation.zero_action_contrast_mse
    )
    interval_gate = (
        overall_coverage >= PHASE6_EXPECTED_VALUE_COVERAGE
        and overall_width <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
        and minimum_slice_coverage >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
    )
    train_roots = {
        row.root_id for row in rows if row.split == "train"
    }
    selection_roots = {row.root_id for row in selection_rows}
    bound_roots = {row.root_id for row in bound_rows}
    validation_roots = {row.root_id for row in validation_rows}
    implementation_gate = (
        not (
            (train_roots & selection_roots)
            or (train_roots & bound_roots)
            or (train_roots & validation_roots)
            or (selection_roots & bound_roots)
            or (selection_roots & validation_roots)
            or (bound_roots & validation_roots)
        )
        and {row.policy_variant_id_supervision_only for row in validation_rows}
        == {"action_first"}
        and {row.policy_variant_id_supervision_only for row in bound_rows}
        == {"hand_first"}
        and set(model.train_root_ids) == train_roots
        and set(model.calibration_selection_root_ids) == selection_roots
    )
    calibration = ExpectedValueCalibration(
        source_dataset_fingerprint=dataset.fingerprint,
        source_model_fingerprint=model.fingerprint,
        requested_coverage=PHASE6_EXPECTED_VALUE_COVERAGE,
        mean_confidence_z=PHASE6_EXPECTED_VALUE_MEAN_CI_Z,
        calibration_bound_root_ids=tuple(sorted(bound_roots)),
        validation_root_ids=tuple(sorted(validation_roots)),
        conformal_radius=radius,
        point_metrics=point_metrics,
        slice_metrics=tuple(slice_metrics),
        maximum_target_standard_error=maximum_se,
        validation_coverage=overall_coverage,
        validation_mean_width=overall_width,
        minimum_validation_slice_coverage=minimum_slice_coverage,
        sampling_precision_gate_passed=sampling_gate,
        point_model_gate_passed=point_gate,
        interval_gate_passed=interval_gate,
        implementation_holdout_gate_passed=implementation_gate,
        conditional_expected_mean_model_admitted=(
            sampling_gate and point_gate and interval_gate and implementation_gate
        ),
    )
    restored = ExpectedValueCalibration.from_dict(json.loads(calibration.to_json()))
    if restored != calibration:
        raise RuntimeError("Phase 6AH calibration roundtrip changed values")
    return calibration


@dataclass(frozen=True, slots=True)
class Phase6ExpectedValueEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    surface_count: int
    action_target_count: int
    terminal_rollout_count: int
    live_selection_count: int
    native_transition_count: int
    train_root_count: int
    calibration_root_count: int
    validation_root_count: int
    maximum_target_standard_error: float
    validation_mse: float
    validation_zero_mse: float
    validation_heuristic_mse: float
    validation_action_contrast_mse: float
    validation_zero_action_contrast_mse: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    sampling_precision_gate_passed: bool
    implementation_heldout_point_model_validated: bool
    implementation_heldout_conditional_expected_mean_admitted: bool
    counterfactual_value_compatible: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False
    claims: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AH evidence identity is invalid")
        for name in (
            "dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        for name in (
            "root_count",
            "surface_count",
            "action_target_count",
            "terminal_rollout_count",
            "live_selection_count",
            "native_transition_count",
            "train_root_count",
            "calibration_root_count",
            "validation_root_count",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"Phase 6AH evidence {name} is invalid")
        for name in (
            "maximum_target_standard_error",
            "validation_mse",
            "validation_zero_mse",
            "validation_heuristic_mse",
            "validation_action_contrast_mse",
            "validation_zero_action_contrast_mse",
            "validation_mean_width",
        ):
            if getattr(self, name) < 0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"Phase 6AH evidence {name} is invalid")
        for name in ("validation_coverage", "minimum_validation_slice_coverage"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"Phase 6AH evidence {name} is invalid")
        expected_sampling = self.maximum_target_standard_error <= (
            PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
        )
        if self.sampling_precision_gate_passed != expected_sampling:
            raise ValueError("Phase 6AH evidence sampling gate is inconsistent")
        expected_point = (
            self.validation_mse
            < min(self.validation_zero_mse, self.validation_heuristic_mse)
            and self.validation_action_contrast_mse
            < self.validation_zero_action_contrast_mse
        )
        if self.implementation_heldout_point_model_validated != expected_point:
            raise ValueError("Phase 6AH evidence point gate is inconsistent")
        if (
            self.implementation_heldout_conditional_expected_mean_admitted
            and not self.implementation_heldout_point_model_validated
        ):
            raise ValueError("Phase 6AH expected-mean admission bypassed its point gate")
        expected_status = (
            "implementation_heldout_conditional_expected_mean_admitted_downstream_fail_closed"
            if self.implementation_heldout_conditional_expected_mean_admitted
            else (
                "implementation_heldout_conditional_expected_point_signal_intervals_rejected_downstream_fail_closed"
                if self.implementation_heldout_point_model_validated
                else "implementation_heldout_conditional_expected_value_rejected_downstream_fail_closed"
            )
        )
        if self.status != expected_status:
            raise ValueError("Phase 6AH evidence status is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AH evidence promoted downstream control")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6AH evidence claims contain duplicates")
        if self.claims != tuple(sorted(self.claims)):
            raise ValueError("Phase 6AH evidence claims must use canonical key order")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_EXPECTED_VALUE_SCHEMA_VERSION,
            **asdict(self),
        }
        result["claims"] = dict(self.claims)
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AH repeated official expected-value surfaces

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Complete official matches / retained roots | {self.root_count} / {self.root_count} |
| Hidden-allocation surfaces | {self.surface_count} |
| Aggregated root/action targets | {self.action_target_count} |
| Genuine native terminal continuations | {self.terminal_rollout_count} |
| Genuine live selections / native transitions | {self.live_selection_count} / {self.native_transition_count} |
| Train / calibration / validation roots | {self.train_root_count} / {self.calibration_root_count} / {self.validation_root_count} |
| Maximum target SE | `{self.maximum_target_standard_error:.6f}` |
| Validation neural / zero / heuristic MSE | `{self.validation_mse:.6f}` / `{self.validation_zero_mse:.6f}` / `{self.validation_heuristic_mse:.6f}` |
| Validation action-contrast / zero-contrast MSE | `{self.validation_action_contrast_mse:.6f}` / `{self.validation_zero_action_contrast_mse:.6f}` |
| Validation mean-CI coverage / mean width | `{self.validation_coverage:.6f}` / `{self.validation_mean_width:.6f}` |
| Minimum action/candidate/intent slice coverage | `{self.minimum_validation_slice_coverage:.6f}` |
| Sampling precision / heldout point gate | **{str(self.sampling_precision_gate_passed).lower()}** / **{str(self.implementation_heldout_point_model_validated).lower()}** |
| Conditional expected-mean model admitted | **{str(self.implementation_heldout_conditional_expected_mean_admitted).lower()}** |
| Counterfactual / equilibrium / safe resolver | **false / false / false** |
| Exploitation / official strategy control | **false / false** |

Complete matches, legal live selections, retained engine snapshots, exact root
replay, and native descendant transitions are genuine official-engine execution.
The hidden-allocation law, five turn plans, tactical executor, continuation
blueprint, neural regressor, and empirical intervals are abstractions. Values
are conditional on those supplied abstractions and are not opponent
counterfactual values or a safety certificate. No Kaggle submission or
performance claim is enabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6ExpectedValueEvidence":
        values = dict(raw)
        values.pop("schema_version", None)
        fingerprint = values.pop("fingerprint", None)
        values["claims"] = tuple(
            (str(name), str(value)) for name, value in values["claims"].items()
        )
        result = cls(**values)
        if fingerprint not in (None, result.fingerprint):
            raise ValueError("Phase 6AH evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6ExpectedValueArtifacts:
    dataset: ExpectedValueDataset
    model: ExpectedValueModel
    calibration: ExpectedValueCalibration
    evidence: Phase6ExpectedValueEvidence


def run_phase6_expected_value_surfaces(
    *,
    project_root: str | Path | None = None,
    source_dataset: ExpectedValueDataset | None = None,
    existing_roots: Sequence[ExpectedValueRoot] = (),
    root_observer: Callable[[ExpectedValueRoot, int, int], None] | None = None,
    seed: int = PHASE6_EXPECTED_VALUE_SEED,
) -> Phase6ExpectedValueArtifacts:
    dataset = source_dataset or collect_expected_value_dataset(
        project_root=project_root,
        existing_roots=existing_roots,
        root_observer=root_observer,
        seed=seed,
    )
    model = train_expected_value_model(dataset, seed=seed)
    calibration = calibrate_expected_value_model(dataset, model)
    validation = calibration.point_metrics[-1]
    split_counts = dict(dataset.split_root_counts)
    admitted = calibration.conditional_expected_mean_model_admitted
    point = calibration.point_model_gate_passed
    evidence = Phase6ExpectedValueEvidence(
        benchmark_id="phase6ah-repeated-official-root-action-hypothesis-values",
        status=(
            "implementation_heldout_conditional_expected_mean_admitted_downstream_fail_closed"
            if admitted
            else (
                "implementation_heldout_conditional_expected_point_signal_intervals_rejected_downstream_fail_closed"
                if point
                else "implementation_heldout_conditional_expected_value_rejected_downstream_fail_closed"
            )
        ),
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=len(dataset.roots),
        surface_count=sum(len(item.surfaces) for item in dataset.roots),
        action_target_count=len(dataset.roots) * len(OPPONENT_DEVIATION_ACTION_IDS),
        terminal_rollout_count=dataset.terminal_rollout_count,
        live_selection_count=dataset.live_selection_count,
        native_transition_count=dataset.native_transition_count,
        train_root_count=split_counts["train"],
        calibration_root_count=split_counts["calibration"],
        validation_root_count=split_counts["validation"],
        maximum_target_standard_error=(
            calibration.maximum_target_standard_error
        ),
        validation_mse=validation.mse,
        validation_zero_mse=validation.zero_mse,
        validation_heuristic_mse=validation.heuristic_mse,
        validation_action_contrast_mse=validation.action_contrast_mse,
        validation_zero_action_contrast_mse=(
            validation.zero_action_contrast_mse
        ),
        validation_coverage=calibration.validation_coverage,
        validation_mean_width=calibration.validation_mean_width,
        minimum_validation_slice_coverage=(
            calibration.minimum_validation_slice_coverage
        ),
        sampling_precision_gate_passed=(
            calibration.sampling_precision_gate_passed
        ),
        implementation_heldout_point_model_validated=point,
        implementation_heldout_conditional_expected_mean_admitted=admitted,
        claims=(
            (
                "genuine_official_execution",
                "complete matches, legal live selections, retained snapshot replay, "
                "and native descendant transitions",
            ),
            (
                "research_abstractions",
                "hidden-allocation law, five plans, tactical executor, frozen "
                "continuation blueprint, neural model, and intervals",
            ),
            (
                "unavailable",
                "opponent counterfactual values, equilibrium convergence, safety, "
                "exploitation, official strategy control, and strength claims",
            ),
            (
                "value_target",
                "conditional expected terminal utility under the supplied allocation "
                "law and frozen abstract continuation blueprint",
            ),
        ),
    )
    restored_evidence = Phase6ExpectedValueEvidence.from_dict(
        json.loads(evidence.to_json())
    )
    if restored_evidence != evidence:
        differences = tuple(
            name
            for name in evidence.__dataclass_fields__
            if getattr(restored_evidence, name) != getattr(evidence, name)
        )
        raise RuntimeError(
            f"Phase 6AH evidence roundtrip changed values: {differences}"
        )
    return Phase6ExpectedValueArtifacts(
        dataset=dataset,
        model=model,
        calibration=calibration,
        evidence=evidence,
    )


def main() -> None:
    """Run the package CLI; the script wrapper adds resumable shard progress."""

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "phase6_expected_value_surfaces"
        ),
    )
    parser.add_argument("--reuse-source", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    target = args.artifact_directory.resolve()
    dataset_path = target / "expected_value_surface_dataset.json"
    if args.reuse_source:
        dataset = ExpectedValueDataset.from_dict(
            json.loads(dataset_path.read_text(encoding="utf-8"))
        )
    else:
        dataset = collect_expected_value_dataset(project_root=project_root)
        target.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(dataset.to_json() + "\n", encoding="utf-8")
    artifacts = run_phase6_expected_value_surfaces(
        project_root=project_root,
        source_dataset=dataset,
    )
    target.mkdir(parents=True, exist_ok=True)
    for name, text_value in (
        ("expected_value_model.json", artifacts.model.to_json() + "\n"),
        (
            "expected_value_calibration.json",
            artifacts.calibration.to_json() + "\n",
        ),
        ("phase6ah_evidence.json", artifacts.evidence.to_json() + "\n"),
        ("phase6ah_evidence.md", artifacts.evidence.to_markdown()),
    ):
        (target / name).write_text(text_value, encoding="utf-8")
    print(artifacts.evidence.to_json())


if __name__ == "__main__":
    main()
