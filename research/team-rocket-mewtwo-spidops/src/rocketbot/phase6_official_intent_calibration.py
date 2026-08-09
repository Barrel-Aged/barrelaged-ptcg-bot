"""Phase 6AE seat-decoupled official opponent-intent calibration.

Phase 6AD calibrated a narrow six-list hidden-card range, but its collection
used the same declared behavior style for both seats.  That is not admissible
intent evidence: an observer could learn its own policy instead of the
opponent's behavior.  This module fixes player 0 to one reference policy and
varies only player 1 across six declared strategic-intent proxy policies.

Every stored action semantic is public (coarse MAIN action type only).  Raw
selection indices, private card identities, and visualizer truth never enter
the intent model.  Transient truth is used only to re-audit the Phase 6AD
candidate/resource component under this independently collected policy panel.
The result can validate a declared-policy intent proxy in scope; it cannot claim
to observe a human opponent's latent intent or enable values, resolving,
exploitation, or official strategy control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

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
from .game_theory import OpponentRange, RangeHypothesis
from .native_blueprint_features import (
    NATIVE_PUBLIC_FEATURE_NAMES,
    native_observer_public_feature_items,
)
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
from .official_search_game import _root_public_history
from .phase6_aligned_counterfactual_values import ALIGNED_VALUE_SPLITS
from .phase6_aligned_public_range import (
    PUBLIC_RANGE_TEMPERATURE_GRID,
    PUBLIC_RANGE_UNIFORM_MIX_GRID,
)
from .phase6_joint_official_values import _official_truth_completion
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_official_range_calibration import (
    OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT,
    OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE,
    OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
    OFFICIAL_RANGE_RESOURCE_NAMES,
    OFFICIAL_RANGE_ZONES,
    CandidateParticleSummary,
    OfficialPublicRangeModel,
    OfficialRangeCalibration,
    OfficialRangeTruthDataset,
    Phase6OfficialRangeEvidence,
    _candidate_particle_summaries,
    _ece,
    _mixed_resource_predictions,
    _resource_counts,
    _uniform_candidate_range,
    _zone_sums,
)
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_OFFICIAL_INTENT_SCHEMA_VERSION = 1
PHASE6_OFFICIAL_INTENT_SEED = 6_045
OFFICIAL_INTENT_OBSERVER_PLAYER = 0
OFFICIAL_INTENT_OPPONENT_PLAYER = 1
OFFICIAL_INTENT_TARGET_POLICY = "fixed_player0_tempo"
OFFICIAL_INTENT_ROOTS_PER_CANDIDATE_INTENT_SPLIT = 3
OFFICIAL_INTENT_PARTICLES_PER_CANDIDATE = OFFICIAL_RANGE_PARTICLES_PER_CANDIDATE
OFFICIAL_INTENT_MINIMUM_CALIBRATION_ROOTS = 100
OFFICIAL_INTENT_MINIMUM_VALIDATION_ROOTS = 100
OFFICIAL_INTENT_ECE_LIMIT = 0.15
OFFICIAL_JOINT_ECE_LIMIT = 0.15
OFFICIAL_INTENT_MINIMUM_TOP1 = 0.75
OFFICIAL_JOINT_MINIMUM_TOP1 = 0.70
OFFICIAL_INTENT_L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
OFFICIAL_INTENT_TRAINING_STEPS = 500
OFFICIAL_INTENT_LEARNING_RATE = 0.05
OFFICIAL_INTENT_SEQUENCE_SLOTS = 10
OFFICIAL_INTENT_ACTION_TOKENS = (
    "play",
    "attach",
    "evolve",
    "ability",
    "retreat",
    "attack",
    "end",
    "other",
)
OFFICIAL_INTENT_ACTION_TYPE_TO_TOKEN = {
    int(OptionType.PLAY): "play",
    int(OptionType.ATTACH): "attach",
    int(OptionType.EVOLVE): "evolve",
    int(OptionType.ABILITY): "ability",
    int(OptionType.RETREAT): "retreat",
    int(OptionType.ATTACK): "attack",
    int(OptionType.END): "end",
}
OFFICIAL_INTENT_ROOT_FEATURE_NAMES = (
    "turn",
    "turn_action_count",
    "opponent_board_count",
    "opponent_board_hp",
    "opponent_deck_count",
    "opponent_hand_count",
    "opponent_prize_count",
    "opponent_public_revealed_copies",
)


def _intent_feature_names() -> tuple[str, ...]:
    names = [f"root_{name}" for name in OFFICIAL_INTENT_ROOT_FEATURE_NAMES]
    names.extend(
        (
            "opponent_main_action_count",
            "opponent_main_transition_count",
        )
    )
    for token in OFFICIAL_INTENT_ACTION_TOKENS:
        names.extend(
            (
                f"opponent_main_count_{token}",
                f"opponent_main_first_position_{token}",
                f"opponent_main_mean_position_{token}",
                f"opponent_main_last_position_{token}",
            )
        )
    for left in OFFICIAL_INTENT_ACTION_TOKENS:
        for right in OFFICIAL_INTENT_ACTION_TOKENS:
            names.append(f"opponent_main_transition_{left}_to_{right}")
    for slot in range(OFFICIAL_INTENT_SEQUENCE_SLOTS):
        for token in OFFICIAL_INTENT_ACTION_TOKENS:
            names.append(f"opponent_main_slot_{slot:02d}_{token}")
    return tuple(sorted(names))


OFFICIAL_INTENT_FEATURE_NAMES = _intent_feature_names()


OFFICIAL_INTENT_PRIORITIES: Mapping[OpponentIntent, Mapping[int, float]] = {
    OpponentIntent.SETUP: {
        int(OptionType.EVOLVE): 240.0,
        int(OptionType.ATTACH): 225.0,
        int(OptionType.ABILITY): 210.0,
        int(OptionType.PLAY): 195.0,
        int(OptionType.RETREAT): 130.0,
        int(OptionType.END): 110.0,
        int(OptionType.ATTACK): 100.0,
    },
    OpponentIntent.PRIZE_RACE: {
        int(OptionType.ATTACK): 260.0,
        int(OptionType.ATTACH): 225.0,
        # If an attack is unavailable after one attachment, passing is a
        # clearer prize-race declaration than spending the turn on setup.
        int(OptionType.END): 200.0,
        int(OptionType.EVOLVE): 185.0,
        int(OptionType.ABILITY): 175.0,
        int(OptionType.PLAY): 150.0,
        int(OptionType.RETREAT): 80.0,
    },
    OpponentIntent.LOCK_DISRUPTION: {
        int(OptionType.PLAY): 260.0,
        int(OptionType.ABILITY): 235.0,
        int(OptionType.EVOLVE): 190.0,
        int(OptionType.ATTACH): 175.0,
        int(OptionType.RETREAT): 145.0,
        int(OptionType.END): 110.0,
        int(OptionType.ATTACK): 105.0,
    },
    OpponentIntent.BOARD_DENIAL: {
        int(OptionType.ABILITY): 260.0,
        int(OptionType.ATTACK): 235.0,
        int(OptionType.PLAY): 205.0,
        int(OptionType.RETREAT): 185.0,
        int(OptionType.EVOLVE): 165.0,
        int(OptionType.ATTACH): 145.0,
        int(OptionType.END): 0.0,
    },
    OpponentIntent.RESOURCE_CONSERVATION: {
        int(OptionType.ATTACK): 260.0,
        int(OptionType.END): 235.0,
        int(OptionType.RETREAT): 180.0,
        int(OptionType.ABILITY): 120.0,
        int(OptionType.EVOLVE): 100.0,
        int(OptionType.ATTACH): 80.0,
        int(OptionType.PLAY): 60.0,
    },
    OpponentIntent.COMBO_ASSEMBLY: {
        int(OptionType.ABILITY): 260.0,
        int(OptionType.PLAY): 245.0,
        int(OptionType.EVOLVE): 225.0,
        int(OptionType.ATTACH): 205.0,
        # Combo lines cash out after setup; unlike lock/disruption, they are
        # permitted to attack before voluntarily ending the turn.
        int(OptionType.ATTACK): 160.0,
        int(OptionType.END): 130.0,
        int(OptionType.RETREAT): 90.0,
    },
}


# A declared intent is more than an action-type tie break.  The bounded MAIN
# action budget gives each controlled proxy a stable stopping rule on every
# turn: conservation passes immediately, denial takes one high-impact action,
# setup/prize lines take two, disruption takes three, and combo assembly keeps
# investing.  These are deliberately simple research controls, not learned or
# asserted human intent semantics.
OFFICIAL_INTENT_MAIN_ACTION_BUDGET: Mapping[OpponentIntent, int] = {
    OpponentIntent.SETUP: 2,
    OpponentIntent.PRIZE_RACE: 2,
    OpponentIntent.LOCK_DISRUPTION: 3,
    OpponentIntent.BOARD_DENIAL: 1,
    OpponentIntent.RESOURCE_CONSERVATION: 0,
    OpponentIntent.COMBO_ASSEMBLY: 6,
}


# Phase 6AF uses three split-disjoint implementations of the same declared
# intent families.  The canonical budgets stay fixed so the variants perturb
# *how* an intent pursues its line rather than silently changing the label.
OFFICIAL_INTENT_POLICY_VARIANT_IDS = (
    "canonical",
    "engine_first",
    "hand_first",
    "action_first",
)
OFFICIAL_INTENT_POLICY_VARIANT_BIASES: Mapping[str, Mapping[int, float]] = {
    "canonical": {},
    "engine_first": {
        int(OptionType.ABILITY): 45.0,
        int(OptionType.EVOLVE): 30.0,
        int(OptionType.PLAY): -10.0,
        int(OptionType.ATTACH): -5.0,
    },
    "hand_first": {
        int(OptionType.PLAY): 35.0,
        int(OptionType.ATTACH): 25.0,
        int(OptionType.ABILITY): -10.0,
        int(OptionType.EVOLVE): -5.0,
    },
    "action_first": {
        int(OptionType.ATTACK): 25.0,
        int(OptionType.RETREAT): 20.0,
        int(OptionType.END): 10.0,
        int(OptionType.PLAY): -10.0,
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def official_intent_policy_variant_fingerprint(variant_id: str) -> str:
    """Fingerprint one controlled policy implementation without private state."""

    if variant_id not in OFFICIAL_INTENT_POLICY_VARIANT_IDS:
        raise ValueError("unsupported official intent policy variant")
    return _fingerprint(
        {
            "variant_id": variant_id,
            "priority_biases": sorted(
                OFFICIAL_INTENT_POLICY_VARIANT_BIASES[variant_id].items()
            ),
            "base_priorities": {
                intent.value: sorted(priorities.items())
                for intent, priorities in OFFICIAL_INTENT_PRIORITIES.items()
            },
            "main_action_budgets": {
                intent.value: budget
                for intent, budget in OFFICIAL_INTENT_MAIN_ACTION_BUDGET.items()
            },
        }
    )


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


class StrategicIntentPriorityBaseline:
    """Legal deterministic policy whose MAIN ordering declares one intent proxy."""

    def __init__(
        self,
        deck: Sequence[int],
        *,
        intent: OpponentIntent,
        reverse: bool,
        variant_id: str = "canonical",
    ) -> None:
        if len(deck) != 60:
            raise ValueError("intent baseline deck must contain 60 IDs")
        if intent not in OPPONENT_INTENTS:
            raise ValueError("intent baseline requires a supported intent")
        if variant_id not in OFFICIAL_INTENT_POLICY_VARIANT_IDS:
            raise ValueError("intent baseline policy variant is unsupported")
        self.deck = tuple(int(value) for value in deck)
        self.intent = intent
        self.reverse = bool(reverse)
        self.variant_id = variant_id
        self._fallback = PriorityBaseline(
            self.deck,
            style="tempo",
            reverse=self.reverse,
        )
        self._main_turn: int | None = None
        self._main_actions_this_turn = 0

    def reset(self) -> None:
        self._fallback.reset()
        self._main_turn = None
        self._main_actions_this_turn = 0

    def choose(self, observation: Mapping[str, Any]) -> list[int]:
        select = observation.get("select")
        if select is None:
            return list(self.deck)
        if not isinstance(select, Mapping):
            raise ValueError("intent baseline selection payload is malformed")
        if _safe_int(select.get("context"), -1) != int(SelectContext.MAIN):
            return self._fallback.choose(observation)
        contract = SelectionContract.from_observation(observation)
        options = select.get("option", ())
        turn = _safe_int(_state(observation).get("turn"), 0)
        if turn != self._main_turn:
            self._main_turn = turn
            self._main_actions_this_turn = 0
        priorities = dict(OFFICIAL_INTENT_PRIORITIES[self.intent])
        for option_type, bias in OFFICIAL_INTENT_POLICY_VARIANT_BIASES[
            self.variant_id
        ].items():
            priorities[option_type] = priorities.get(option_type, 50.0) + bias
        if self._main_actions_this_turn >= OFFICIAL_INTENT_MAIN_ACTION_BUDGET[
            self.intent
        ]:
            priorities[int(OptionType.END)] = 10_000.0
        ranked = sorted(
            range(len(options)),
            key=lambda index: (
                -priorities.get(_safe_int(options[index].get("type"), -1), 50.0),
                -index if self.reverse else index,
            ),
        )
        selected = ranked[: contract.max_count]
        self._main_actions_this_turn += len(selected)
        return selected


@dataclass(frozen=True, slots=True)
class OfficialIntentDecisionStep:
    step_index: int
    actor: int
    turn: int
    selection_type: int
    selection_context: int
    legal_option_count: int
    minimum_selection_count: int
    maximum_selection_count: int
    selected_count: int
    selected_public_main_action_types: tuple[int, ...]
    public_state_before_fingerprint: str
    public_history_before_fingerprint: str
    selection_contract_fingerprint: str
    selected_selection_fingerprint: str
    public_state_after_fingerprint: str
    public_history_after_fingerprint: str
    official_battle_selection_executed: bool = True
    official_chance_probability_available: bool = False
    raw_selection_indices_persisted: int = 0
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.actor not in (0, 1) or self.turn < 0:
            raise ValueError("official intent decision identity is invalid")
        if not (
            0
            <= self.minimum_selection_count
            <= self.selected_count
            <= self.maximum_selection_count
            <= self.legal_option_count
        ):
            raise ValueError("official intent selection counts are invalid")
        for name in (
            "public_state_before_fingerprint",
            "public_history_before_fingerprint",
            "selection_contract_fingerprint",
            "selected_selection_fingerprint",
            "public_state_after_fingerprint",
            "public_history_after_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        expected_semantics = (
            self.actor == OFFICIAL_INTENT_OPPONENT_PLAYER
            and self.selection_context == int(SelectContext.MAIN)
        )
        if bool(self.selected_public_main_action_types) != expected_semantics:
            raise ValueError("official intent action semantics escaped the opponent MAIN boundary")
        valid_option_types = {int(value) for value in OptionType}
        if any(value not in valid_option_types for value in self.selected_public_main_action_types):
            raise ValueError("official intent action semantic is unsupported")
        if any(
            (
                not self.official_battle_selection_executed,
                self.official_chance_probability_available,
                self.raw_selection_indices_persisted != 0,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official intent step promoted an unsupported claim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "actor": self.actor,
            "turn": self.turn,
            "selection_type": self.selection_type,
            "selection_context": self.selection_context,
            "legal_option_count": self.legal_option_count,
            "minimum_selection_count": self.minimum_selection_count,
            "maximum_selection_count": self.maximum_selection_count,
            "selected_count": self.selected_count,
            "selected_public_main_action_types": list(
                self.selected_public_main_action_types
            ),
            "public_state_before_fingerprint": self.public_state_before_fingerprint,
            "public_history_before_fingerprint": self.public_history_before_fingerprint,
            "selection_contract_fingerprint": self.selection_contract_fingerprint,
            "selected_selection_fingerprint": self.selected_selection_fingerprint,
            "public_state_after_fingerprint": self.public_state_after_fingerprint,
            "public_history_after_fingerprint": self.public_history_after_fingerprint,
            "claims": {
                "official_battle_selection_executed": (
                    self.official_battle_selection_executed
                ),
                "official_chance_probability_available": (
                    self.official_chance_probability_available
                ),
                "raw_selection_indices_persisted": self.raw_selection_indices_persisted,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
                "selected_action_semantics_are_public_and_coarse": True,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialIntentDecisionStep":
        claims = raw["claims"]
        if not bool(claims["selected_action_semantics_are_public_and_coarse"]):
            raise ValueError("official intent public-action claim changed")
        return cls(
            step_index=int(raw["step_index"]),
            actor=int(raw["actor"]),
            turn=int(raw["turn"]),
            selection_type=int(raw["selection_type"]),
            selection_context=int(raw["selection_context"]),
            legal_option_count=int(raw["legal_option_count"]),
            minimum_selection_count=int(raw["minimum_selection_count"]),
            maximum_selection_count=int(raw["maximum_selection_count"]),
            selected_count=int(raw["selected_count"]),
            selected_public_main_action_types=tuple(
                int(value) for value in raw["selected_public_main_action_types"]
            ),
            public_state_before_fingerprint=str(raw["public_state_before_fingerprint"]),
            public_history_before_fingerprint=str(raw["public_history_before_fingerprint"]),
            selection_contract_fingerprint=str(raw["selection_contract_fingerprint"]),
            selected_selection_fingerprint=str(raw["selected_selection_fingerprint"]),
            public_state_after_fingerprint=str(raw["public_state_after_fingerprint"]),
            public_history_after_fingerprint=str(raw["public_history_after_fingerprint"]),
            official_battle_selection_executed=bool(
                claims["official_battle_selection_executed"]
            ),
            official_chance_probability_available=bool(
                claims["official_chance_probability_available"]
            ),
            raw_selection_indices_persisted=int(
                claims["raw_selection_indices_persisted"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )


@dataclass(frozen=True, slots=True)
class OfficialIntentPrefixTrace:
    split: str
    candidate_id: str
    intent_id_supervision_only: str
    collection_ordinal: int
    capture_attempt: int
    captured_turn: int
    target_policy_id: str
    target_reverse: bool
    opponent_reverse: bool
    root_public_state_fingerprint: str
    root_public_history_fingerprint: str
    trace_chain_fingerprint: str
    steps: tuple[OfficialIntentDecisionStep, ...]
    requested_opponent_main_ordinal: int = 2
    observer_player: int = OFFICIAL_INTENT_OBSERVER_PLAYER
    opponent_player: int = OFFICIAL_INTENT_OPPONENT_PLAYER
    official_battle_root_executed: bool = True
    target_policy_fixed_across_intents: bool = True
    opponent_policy_is_declared_intent_proxy: bool = True

    def __post_init__(self) -> None:
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("official intent trace split is invalid")
        if not self.candidate_id or self.intent_id_supervision_only not in {
            item.value for item in OPPONENT_INTENTS
        }:
            raise ValueError("official intent trace labels are invalid")
        if self.collection_ordinal < 0 or self.capture_attempt <= 0 or self.captured_turn < 0:
            raise ValueError("official intent trace collection identity is invalid")
        if self.target_policy_id != OFFICIAL_INTENT_TARGET_POLICY:
            raise ValueError("official intent target policy changed")
        if self.opponent_reverse == self.target_reverse:
            raise ValueError("official intent seat tie-break directions must differ")
        if (self.observer_player, self.opponent_player) != (0, 1):
            raise ValueError("official intent seat orientation changed")
        if self.requested_opponent_main_ordinal != 2:
            raise ValueError("official intent collection boundary changed")
        for name in (
            "root_public_state_fingerprint",
            "root_public_history_fingerprint",
            "trace_chain_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.steps or tuple(item.step_index for item in self.steps) != tuple(
            range(len(self.steps))
        ):
            raise ValueError("official intent trace steps are incomplete")
        for left, right in zip(self.steps, self.steps[1:]):
            if (
                left.public_state_after_fingerprint
                != right.public_state_before_fingerprint
                or left.public_history_after_fingerprint
                != right.public_history_before_fingerprint
            ):
                raise ValueError("official intent public trace chain is discontinuous")
        if (
            self.steps[-1].public_state_after_fingerprint
            != self.root_public_state_fingerprint
            or self.steps[-1].public_history_after_fingerprint
            != self.root_public_history_fingerprint
        ):
            raise ValueError("official intent trace does not end at its root")
        if _fingerprint([item.to_dict() for item in self.steps]) != self.trace_chain_fingerprint:
            raise ValueError("official intent trace-chain fingerprint mismatch")
        if not all(
            (
                self.official_battle_root_executed,
                self.target_policy_fixed_across_intents,
                self.opponent_policy_is_declared_intent_proxy,
            )
        ):
            raise ValueError("official intent trace provenance failed")
        if not self.opponent_main_action_types:
            raise ValueError("official intent trace contains no opponent MAIN behavior")

    @property
    def opponent_main_action_types(self) -> tuple[int, ...]:
        return tuple(
            action_type
            for step in self.steps
            for action_type in step.selected_public_main_action_types
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "candidate_id": self.candidate_id,
            "intent_id_supervision_only": self.intent_id_supervision_only,
            "collection_ordinal": self.collection_ordinal,
            "capture_attempt": self.capture_attempt,
            "captured_turn": self.captured_turn,
            "policy": {
                "target_policy_id": self.target_policy_id,
                "target_reverse": self.target_reverse,
                "opponent_reverse": self.opponent_reverse,
            },
            "root": {
                "public_state_fingerprint": self.root_public_state_fingerprint,
                "public_history_fingerprint": self.root_public_history_fingerprint,
            },
            "trace_chain_fingerprint": self.trace_chain_fingerprint,
            "steps": [item.to_dict() for item in self.steps],
            "boundary": {
                "requested_opponent_main_ordinal": self.requested_opponent_main_ordinal,
                "observer_player": self.observer_player,
                "opponent_player": self.opponent_player,
            },
            "claims": {
                "official_battle_root_executed": self.official_battle_root_executed,
                "target_policy_fixed_across_intents": (
                    self.target_policy_fixed_across_intents
                ),
                "opponent_policy_is_declared_intent_proxy": (
                    self.opponent_policy_is_declared_intent_proxy
                ),
                "opponent_main_action_types_are_model_input": True,
                "target_action_types_are_model_input": False,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialIntentPrefixTrace":
        policy = raw["policy"]
        root = raw["root"]
        boundary = raw["boundary"]
        claims = raw["claims"]
        if not bool(claims["opponent_main_action_types_are_model_input"]):
            raise ValueError("official intent opponent-action input claim changed")
        if bool(claims["target_action_types_are_model_input"]):
            raise ValueError("official intent model consumed target behavior")
        return cls(
            split=str(raw["split"]),
            candidate_id=str(raw["candidate_id"]),
            intent_id_supervision_only=str(raw["intent_id_supervision_only"]),
            collection_ordinal=int(raw["collection_ordinal"]),
            capture_attempt=int(raw["capture_attempt"]),
            captured_turn=int(raw["captured_turn"]),
            target_policy_id=str(policy["target_policy_id"]),
            target_reverse=bool(policy["target_reverse"]),
            opponent_reverse=bool(policy["opponent_reverse"]),
            root_public_state_fingerprint=str(root["public_state_fingerprint"]),
            root_public_history_fingerprint=str(root["public_history_fingerprint"]),
            trace_chain_fingerprint=str(raw["trace_chain_fingerprint"]),
            steps=tuple(OfficialIntentDecisionStep.from_dict(item) for item in raw["steps"]),
            requested_opponent_main_ordinal=int(
                boundary["requested_opponent_main_ordinal"]
            ),
            observer_player=int(boundary["observer_player"]),
            opponent_player=int(boundary["opponent_player"]),
            official_battle_root_executed=bool(
                claims["official_battle_root_executed"]
            ),
            target_policy_fixed_across_intents=bool(
                claims["target_policy_fixed_across_intents"]
            ),
            opponent_policy_is_declared_intent_proxy=bool(
                claims["opponent_policy_is_declared_intent_proxy"]
            ),
        )


def official_intent_feature_items(
    public_features: Sequence[tuple[str, float]],
    trace: OfficialIntentPrefixTrace,
) -> tuple[tuple[str, float], ...]:
    """Build a fixed-width opponent-only public behavior representation."""

    public = dict(public_features)
    if tuple(name for name, _ in public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
        raise ValueError("official intent root public feature schema mismatch")
    sequence = tuple(
        OFFICIAL_INTENT_ACTION_TYPE_TO_TOKEN.get(value, "other")
        for value in trace.opponent_main_action_types
    )
    length = len(sequence)
    position_scale = float(max(length, 1))
    transition_scale = float(max(length - 1, 1))
    features: list[tuple[str, float]] = [
        (f"root_{name}", float(public[name]))
        for name in OFFICIAL_INTENT_ROOT_FEATURE_NAMES
    ]
    features.extend(
        (
            ("opponent_main_action_count", float(length)),
            ("opponent_main_transition_count", float(max(length - 1, 0))),
        )
    )
    for token in OFFICIAL_INTENT_ACTION_TOKENS:
        positions = tuple(
            index + 1 for index, value in enumerate(sequence) if value == token
        )
        features.extend(
            (
                (f"opponent_main_count_{token}", float(len(positions))),
                (
                    f"opponent_main_first_position_{token}",
                    float(positions[0] / position_scale) if positions else 0.0,
                ),
                (
                    f"opponent_main_mean_position_{token}",
                    float(sum(positions) / len(positions) / position_scale)
                    if positions
                    else 0.0,
                ),
                (
                    f"opponent_main_last_position_{token}",
                    float(positions[-1] / position_scale) if positions else 0.0,
                ),
            )
        )
    transitions = tuple(zip(sequence, sequence[1:]))
    for left in OFFICIAL_INTENT_ACTION_TOKENS:
        for right in OFFICIAL_INTENT_ACTION_TOKENS:
            features.append(
                (
                    f"opponent_main_transition_{left}_to_{right}",
                    sum(pair == (left, right) for pair in transitions)
                    / transition_scale,
                )
            )
    for slot in range(OFFICIAL_INTENT_SEQUENCE_SLOTS):
        observed = sequence[slot] if slot < length else None
        for token in OFFICIAL_INTENT_ACTION_TOKENS:
            features.append(
                (f"opponent_main_slot_{slot:02d}_{token}", float(observed == token))
            )
    result = tuple(sorted(features))
    if tuple(name for name, _ in result) != OFFICIAL_INTENT_FEATURE_NAMES:
        raise RuntimeError("official intent feature schema construction changed")
    return result


@dataclass(frozen=True, slots=True)
class OfficialIntentTruthExample:
    example_id: str
    source_root_id: str
    split: str
    candidate_id_supervision_only: str
    intent_id_supervision_only: str
    collection_ordinal: int
    prefix_trace: OfficialIntentPrefixTrace
    public_state_fingerprint: str
    public_history_fingerprint: str
    candidate_public_features: tuple[tuple[str, float], ...]
    intent_public_features: tuple[tuple[str, float], ...]
    candidate_public_abstraction_fingerprint: str
    intent_public_abstraction_fingerprint: str
    truth_sample_fingerprint: str
    truth_resource_counts: tuple[int, ...]
    truth_zone_counts: tuple[int, int, int]
    candidate_particle_summaries: tuple[CandidateParticleSummary, ...]
    visualizer_public_parity: bool = True
    visualizer_acting_private_parity: bool = True
    candidate_identifier_selected_for_model_input: bool = False
    intent_identifier_selected_for_model_input: bool = False
    truth_resource_counts_selected_for_model_input: bool = False
    target_seat_action_semantics_selected_for_model_input: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "example_id",
            "source_root_id",
            "public_state_fingerprint",
            "public_history_fingerprint",
            "candidate_public_abstraction_fingerprint",
            "intent_public_abstraction_fingerprint",
            "truth_sample_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("official intent example split is invalid")
        if not self.candidate_id_supervision_only or self.intent_id_supervision_only not in {
            item.value for item in OPPONENT_INTENTS
        }:
            raise ValueError("official intent example supervision is invalid")
        if self.collection_ordinal < 0:
            raise ValueError("official intent collection ordinal is invalid")
        if (
            self.prefix_trace.split != self.split
            or self.prefix_trace.candidate_id != self.candidate_id_supervision_only
            or self.prefix_trace.intent_id_supervision_only
            != self.intent_id_supervision_only
            or self.prefix_trace.collection_ordinal != self.collection_ordinal
        ):
            raise ValueError("official intent trace/example identity mismatch")
        if (
            self.prefix_trace.root_public_state_fingerprint
            != self.public_state_fingerprint
            or self.prefix_trace.root_public_history_fingerprint
            != self.public_history_fingerprint
        ):
            raise ValueError("official intent trace/example public-root mismatch")
        if tuple(name for name, _ in self.candidate_public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("official intent candidate feature schema mismatch")
        if tuple(name for name, _ in self.intent_public_features) != OFFICIAL_INTENT_FEATURE_NAMES:
            raise ValueError("official intent behavior feature schema mismatch")
        if _fingerprint(self.candidate_public_features) != self.candidate_public_abstraction_fingerprint:
            raise ValueError("official intent candidate abstraction fingerprint mismatch")
        if _fingerprint(self.intent_public_features) != self.intent_public_abstraction_fingerprint:
            raise ValueError("official intent behavior abstraction fingerprint mismatch")
        if len(self.truth_resource_counts) != len(OFFICIAL_RANGE_RESOURCE_NAMES):
            raise ValueError("official intent truth-resource width is invalid")
        if any(value < 0 for value in self.truth_resource_counts):
            raise ValueError("official intent truth-resource count is invalid")
        if tuple(int(value) for value in _zone_sums(self.truth_resource_counts)) != self.truth_zone_counts:
            raise ValueError("official intent truth zone/resource accounting mismatch")
        identifiers = tuple(item.candidate_id for item in self.candidate_particle_summaries)
        if len(identifiers) != len(set(identifiers)) or self.candidate_id_supervision_only not in identifiers:
            raise ValueError("official intent candidate-particle support is invalid")
        expected_zone_sums = tuple(float(value) for value in self.truth_zone_counts)
        for summary in self.candidate_particle_summaries:
            if any(
                not math.isclose(observed, expected, abs_tol=1e-8)
                for observed, expected in zip(
                    _zone_sums(summary.expected_resource_counts),
                    expected_zone_sums,
                    strict=True,
                )
            ):
                raise ValueError("official intent candidate particles changed public zone sizes")
        true_summary = next(
            item
            for item in self.candidate_particle_summaries
            if item.candidate_id == self.candidate_id_supervision_only
        )
        if true_summary.maximum_public_evidence_adjustments != 0:
            raise ValueError("official intent realized candidate required deck adjustment")
        if not self.visualizer_public_parity or not self.visualizer_acting_private_parity:
            raise ValueError("official intent visualizer/live parity failed")
        if any(
            (
                self.candidate_identifier_selected_for_model_input,
                self.intent_identifier_selected_for_model_input,
                self.truth_resource_counts_selected_for_model_input,
                self.target_seat_action_semantics_selected_for_model_input,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official intent example promoted hidden/confounded input")

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_root_id": self.source_root_id,
            "split": self.split,
            "candidate_id_supervision_only": self.candidate_id_supervision_only,
            "intent_id_supervision_only": self.intent_id_supervision_only,
            "collection_ordinal": self.collection_ordinal,
            "prefix_trace": self.prefix_trace.to_dict(),
            "public": {
                "state_fingerprint": self.public_state_fingerprint,
                "history_fingerprint": self.public_history_fingerprint,
                "candidate_abstraction_fingerprint": (
                    self.candidate_public_abstraction_fingerprint
                ),
                "intent_abstraction_fingerprint": (
                    self.intent_public_abstraction_fingerprint
                ),
                "candidate_features": [list(item) for item in self.candidate_public_features],
                "intent_features": [list(item) for item in self.intent_public_features],
                "observer_player": OFFICIAL_INTENT_OBSERVER_PLAYER,
                "opponent_player": OFFICIAL_INTENT_OPPONENT_PLAYER,
            },
            "truth_supervision": {
                "sample_fingerprint": self.truth_sample_fingerprint,
                "resource_counts": [
                    [name, count]
                    for name, count in zip(
                        OFFICIAL_RANGE_RESOURCE_NAMES,
                        self.truth_resource_counts,
                        strict=True,
                    )
                ],
                "zone_counts": dict(
                    zip(OFFICIAL_RANGE_ZONES, self.truth_zone_counts, strict=True)
                ),
            },
            "candidate_particle_summaries": [
                item.to_dict() for item in self.candidate_particle_summaries
            ],
            "gates": {
                "visualizer_public_parity": self.visualizer_public_parity,
                "visualizer_acting_private_parity": self.visualizer_acting_private_parity,
            },
            "claims": {
                "public_features_only": True,
                "candidate_identifier_selected_for_model_input": (
                    self.candidate_identifier_selected_for_model_input
                ),
                "intent_identifier_selected_for_model_input": (
                    self.intent_identifier_selected_for_model_input
                ),
                "truth_resource_counts_selected_for_model_input": (
                    self.truth_resource_counts_selected_for_model_input
                ),
                "target_seat_action_semantics_selected_for_model_input": (
                    self.target_seat_action_semantics_selected_for_model_input
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
                "intent_label_is_declared_policy_proxy": True,
                "intent_label_is_official_latent_truth": False,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialIntentTruthExample":
        public = raw["public"]
        truth = raw["truth_supervision"]
        claims = raw["claims"]
        gates = raw["gates"]
        if not bool(claims["public_features_only"]):
            raise ValueError("official intent public-only claim changed")
        if not bool(claims["intent_label_is_declared_policy_proxy"]):
            raise ValueError("official intent proxy-label claim changed")
        if bool(claims["intent_label_is_official_latent_truth"]):
            raise ValueError("official intent proxy was promoted to latent truth")
        if (int(public["observer_player"]), int(public["opponent_player"])) != (0, 1):
            raise ValueError("official intent serialized seat orientation changed")
        resources = tuple(truth["resource_counts"])
        if tuple(str(item[0]) for item in resources) != OFFICIAL_RANGE_RESOURCE_NAMES:
            raise ValueError("official intent resource names changed")
        return cls(
            example_id=str(raw["example_id"]),
            source_root_id=str(raw["source_root_id"]),
            split=str(raw["split"]),
            candidate_id_supervision_only=str(raw["candidate_id_supervision_only"]),
            intent_id_supervision_only=str(raw["intent_id_supervision_only"]),
            collection_ordinal=int(raw["collection_ordinal"]),
            prefix_trace=OfficialIntentPrefixTrace.from_dict(raw["prefix_trace"]),
            public_state_fingerprint=str(public["state_fingerprint"]),
            public_history_fingerprint=str(public["history_fingerprint"]),
            candidate_public_features=tuple(
                (str(name), float(value)) for name, value in public["candidate_features"]
            ),
            intent_public_features=tuple(
                (str(name), float(value)) for name, value in public["intent_features"]
            ),
            candidate_public_abstraction_fingerprint=str(
                public["candidate_abstraction_fingerprint"]
            ),
            intent_public_abstraction_fingerprint=str(
                public["intent_abstraction_fingerprint"]
            ),
            truth_sample_fingerprint=str(truth["sample_fingerprint"]),
            truth_resource_counts=tuple(int(item[1]) for item in resources),
            truth_zone_counts=tuple(
                int(truth["zone_counts"][zone]) for zone in OFFICIAL_RANGE_ZONES
            ),
            candidate_particle_summaries=tuple(
                CandidateParticleSummary.from_dict(item)
                for item in raw["candidate_particle_summaries"]
            ),
            visualizer_public_parity=bool(gates["visualizer_public_parity"]),
            visualizer_acting_private_parity=bool(
                gates["visualizer_acting_private_parity"]
            ),
            candidate_identifier_selected_for_model_input=bool(
                claims["candidate_identifier_selected_for_model_input"]
            ),
            intent_identifier_selected_for_model_input=bool(
                claims["intent_identifier_selected_for_model_input"]
            ),
            truth_resource_counts_selected_for_model_input=bool(
                claims["truth_resource_counts_selected_for_model_input"]
            ),
            target_seat_action_semantics_selected_for_model_input=bool(
                claims["target_seat_action_semantics_selected_for_model_input"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )


@dataclass(frozen=True, slots=True)
class OfficialIntentTruthDataset:
    source_phase6ad_dataset_fingerprint: str
    source_phase6ad_model_fingerprint: str
    source_phase6ad_calibration_fingerprint: str
    source_phase6ad_evidence_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    candidate_deck_fingerprints: tuple[tuple[str, str], ...]
    intent_ids: tuple[str, ...]
    roots_per_candidate_intent_split: int
    particles_per_candidate: int
    examples: tuple[OfficialIntentTruthExample, ...]
    collection_split_disjoint: bool
    target_policy_fixed_across_intents: bool
    opponent_only_action_semantics: bool
    observer_orientation_validated: bool
    official_battle_roots_executed: bool = True
    intent_labels_are_declared_policy_proxies: bool = True
    intent_labels_are_official_latent_truth: bool = False
    model_transformation_official_engine_executed: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_phase6ad_dataset_fingerprint",
            "source_phase6ad_model_fingerprint",
            "source_phase6ad_calibration_fingerprint",
            "source_phase6ad_evidence_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("official intent candidate catalog is invalid")
        if tuple(candidate for candidate, _ in self.candidate_deck_fingerprints) != self.candidate_ids:
            raise ValueError("official intent candidate fingerprints are misaligned")
        for _, fingerprint in self.candidate_deck_fingerprints:
            _validate_digest(fingerprint, "candidate_deck_fingerprint")
        if self.intent_ids != tuple(item.value for item in OPPONENT_INTENTS):
            raise ValueError("official intent label catalog changed")
        if self.roots_per_candidate_intent_split <= 0 or self.particles_per_candidate <= 1:
            raise ValueError("official intent collection sizes are invalid")
        expected_count = (
            len(ALIGNED_VALUE_SPLITS)
            * len(self.candidate_ids)
            * len(self.intent_ids)
            * self.roots_per_candidate_intent_split
        )
        if len(self.examples) != expected_count:
            raise ValueError("official intent example count is incomplete")
        expected_support = {
            (split, candidate, intent, ordinal)
            for split in ALIGNED_VALUE_SPLITS
            for candidate in self.candidate_ids
            for intent in self.intent_ids
            for ordinal in range(self.roots_per_candidate_intent_split)
        }
        observed_support = {
            (
                item.split,
                item.candidate_id_supervision_only,
                item.intent_id_supervision_only,
                item.collection_ordinal,
            )
            for item in self.examples
        }
        if observed_support != expected_support:
            raise ValueError("official intent split/candidate/intent support is incomplete")
        if len({item.example_id for item in self.examples}) != len(self.examples):
            raise ValueError("official intent example IDs are duplicated")
        if len({item.source_root_id for item in self.examples}) != len(self.examples):
            raise ValueError("official intent source roots are duplicated")
        split_roots = {
            split: {
                item.source_root_id for item in self.examples if item.split == split
            }
            for split in ALIGNED_VALUE_SPLITS
        }
        observed_disjoint = all(
            split_roots[left].isdisjoint(split_roots[right])
            for index, left in enumerate(ALIGNED_VALUE_SPLITS)
            for right in ALIGNED_VALUE_SPLITS[index + 1 :]
        )
        if self.collection_split_disjoint != observed_disjoint or not observed_disjoint:
            raise ValueError("official intent collection splits overlap")
        if not all(
            (
                self.target_policy_fixed_across_intents,
                self.opponent_only_action_semantics,
                self.observer_orientation_validated,
                self.official_battle_roots_executed,
                self.intent_labels_are_declared_policy_proxies,
            )
        ):
            raise ValueError("official intent collection provenance failed")
        if any(
            (
                self.intent_labels_are_official_latent_truth,
                self.model_transformation_official_engine_executed,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official intent dataset promoted an invalid claim")
        if any(
            tuple(summary.candidate_id for summary in item.candidate_particle_summaries)
            != self.candidate_ids
            or any(
                summary.particle_count != self.particles_per_candidate
                for summary in item.candidate_particle_summaries
            )
            for item in self.examples
        ):
            raise ValueError("official intent candidate-particle grid is incomplete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def live_selection_count(self) -> int:
        return sum(len(item.prefix_trace.steps) for item in self.examples)

    @property
    def opponent_main_action_count(self) -> int:
        return sum(
            len(item.prefix_trace.opponent_main_action_types) for item in self.examples
        )

    @property
    def roots_per_split(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (split, sum(item.split == split for item in self.examples))
            for split in ALIGNED_VALUE_SPLITS
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_INTENT_SCHEMA_VERSION,
            "sources": {
                "phase6ad_dataset_fingerprint": self.source_phase6ad_dataset_fingerprint,
                "phase6ad_model_fingerprint": self.source_phase6ad_model_fingerprint,
                "phase6ad_calibration_fingerprint": (
                    self.source_phase6ad_calibration_fingerprint
                ),
                "phase6ad_evidence_fingerprint": self.source_phase6ad_evidence_fingerprint,
            },
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "candidate_deck_fingerprints": [
                {"candidate_id": candidate, "fingerprint": fingerprint}
                for candidate, fingerprint in self.candidate_deck_fingerprints
            ],
            "intent_ids": list(self.intent_ids),
            "collection": {
                "roots_per_candidate_intent_split": (
                    self.roots_per_candidate_intent_split
                ),
                "particles_per_candidate": self.particles_per_candidate,
                "root_count": len(self.examples),
                "roots_per_split": dict(self.roots_per_split),
                "live_selection_count": self.live_selection_count,
                "opponent_main_action_count": self.opponent_main_action_count,
            },
            "examples": [item.to_dict() for item in self.examples],
            "gates": {
                "collection_split_disjoint": self.collection_split_disjoint,
                "target_policy_fixed_across_intents": (
                    self.target_policy_fixed_across_intents
                ),
                "opponent_only_action_semantics": self.opponent_only_action_semantics,
                "observer_orientation_validated": self.observer_orientation_validated,
            },
            "claims": {
                "official_battle_roots_executed": self.official_battle_roots_executed,
                "intent_labels_are_declared_policy_proxies": (
                    self.intent_labels_are_declared_policy_proxies
                ),
                "intent_labels_are_official_latent_truth": (
                    self.intent_labels_are_official_latent_truth
                ),
                "model_transformation_official_engine_executed": (
                    self.model_transformation_official_engine_executed
                ),
                "public_model_inputs_only": True,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialIntentTruthDataset":
        sources = raw["sources"]
        collection = raw["collection"]
        gates = raw["gates"]
        claims = raw["claims"]
        if not bool(claims["public_model_inputs_only"]):
            raise ValueError("official intent public-only model claim changed")
        result = cls(
            source_phase6ad_dataset_fingerprint=str(
                sources["phase6ad_dataset_fingerprint"]
            ),
            source_phase6ad_model_fingerprint=str(
                sources["phase6ad_model_fingerprint"]
            ),
            source_phase6ad_calibration_fingerprint=str(
                sources["phase6ad_calibration_fingerprint"]
            ),
            source_phase6ad_evidence_fingerprint=str(
                sources["phase6ad_evidence_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            candidate_deck_fingerprints=tuple(
                (str(item["candidate_id"]), str(item["fingerprint"]))
                for item in raw["candidate_deck_fingerprints"]
            ),
            intent_ids=tuple(str(value) for value in raw["intent_ids"]),
            roots_per_candidate_intent_split=int(
                collection["roots_per_candidate_intent_split"]
            ),
            particles_per_candidate=int(collection["particles_per_candidate"]),
            examples=tuple(
                OfficialIntentTruthExample.from_dict(item) for item in raw["examples"]
            ),
            collection_split_disjoint=bool(gates["collection_split_disjoint"]),
            target_policy_fixed_across_intents=bool(
                gates["target_policy_fixed_across_intents"]
            ),
            opponent_only_action_semantics=bool(
                gates["opponent_only_action_semantics"]
            ),
            observer_orientation_validated=bool(
                gates["observer_orientation_validated"]
            ),
            official_battle_roots_executed=bool(
                claims["official_battle_roots_executed"]
            ),
            intent_labels_are_declared_policy_proxies=bool(
                claims["intent_labels_are_declared_policy_proxies"]
            ),
            intent_labels_are_official_latent_truth=bool(
                claims["intent_labels_are_official_latent_truth"]
            ),
            model_transformation_official_engine_executed=bool(
                claims["model_transformation_official_engine_executed"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected_collection = {
            "roots_per_candidate_intent_split": result.roots_per_candidate_intent_split,
            "particles_per_candidate": result.particles_per_candidate,
            "root_count": len(result.examples),
            "roots_per_split": dict(result.roots_per_split),
            "live_selection_count": result.live_selection_count,
            "opponent_main_action_count": result.opponent_main_action_count,
        }
        if dict(collection) != expected_collection:
            raise ValueError("official intent collection accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official intent dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OfficialIntentSelectionMetric:
    l2_penalty: float
    temperature: float
    uniform_mix: float
    calibration_nll: float
    calibration_brier: float
    calibration_top1_accuracy: float
    calibration_ece: float

    def __post_init__(self) -> None:
        if self.l2_penalty not in OFFICIAL_INTENT_L2_GRID:
            raise ValueError("official intent L2 penalty is unsupported")
        if self.temperature not in PUBLIC_RANGE_TEMPERATURE_GRID:
            raise ValueError("official intent temperature is unsupported")
        if self.uniform_mix not in PUBLIC_RANGE_UNIFORM_MIX_GRID:
            raise ValueError("official intent uniform mixture is unsupported")
        if any(
            value < 0 or not math.isfinite(value)
            for value in (self.calibration_nll, self.calibration_brier)
        ):
            raise ValueError("official intent selection loss is invalid")
        if not 0 <= self.calibration_top1_accuracy <= 1:
            raise ValueError("official intent selection accuracy is invalid")
        if not 0 <= self.calibration_ece <= 1:
            raise ValueError("official intent selection ECE is invalid")


def _softmax_probabilities(
    logits: Sequence[float],
    *,
    temperature: float,
    uniform_mix: float,
) -> tuple[float, ...]:
    maximum = max(logits)
    exponentials = tuple(math.exp((value - maximum) / temperature) for value in logits)
    total = sum(exponentials)
    uniform = 1.0 / len(exponentials)
    probabilities = tuple(
        (1.0 - uniform_mix) * value / total + uniform_mix * uniform
        for value in exponentials
    )
    normalization = sum(probabilities)
    return tuple(value / normalization for value in probabilities)


@dataclass(frozen=True, slots=True)
class OfficialIntentRangeModel:
    source_dataset_fingerprint: str
    source_phase6ad_model_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    intent_ids: tuple[str, ...]
    normalizer: NamedFeatureNormalizer
    weights: tuple[tuple[float, ...], ...]
    biases: tuple[float, ...]
    selection_metrics: tuple[OfficialIntentSelectionMetric, ...]
    selected_l2_penalty: float
    selected_temperature: float
    selected_uniform_mix: float
    selected_training_loss: float
    train_root_ids: tuple[str, ...]
    calibration_root_ids: tuple[str, ...]
    train_rows_fingerprint: str
    observer_player: int = OFFICIAL_INTENT_OBSERVER_PLAYER
    opponent_player: int = OFFICIAL_INTENT_OPPONENT_PLAYER
    target_policy_id: str = OFFICIAL_INTENT_TARGET_POLICY
    public_features_only: bool = True
    opponent_action_semantics_only: bool = True
    selected_candidate_identifier_input: bool = False
    selected_intent_identifier_input: bool = False
    target_policy_action_input: bool = False
    intent_labels_are_official_latent_truth: bool = False
    official_engine_executed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_dataset_fingerprint",
            "source_phase6ad_model_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "train_rows_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or self.intent_ids != tuple(
            item.value for item in OPPONENT_INTENTS
        ):
            raise ValueError("official intent model label catalogs are invalid")
        if self.normalizer.names != OFFICIAL_INTENT_FEATURE_NAMES:
            raise ValueError("official intent model feature schema mismatch")
        if self.normalizer.fit_rows_fingerprint != self.train_rows_fingerprint:
            raise ValueError("official intent model normalizer provenance mismatch")
        if len(self.weights) != len(self.intent_ids) or len(self.biases) != len(
            self.intent_ids
        ):
            raise ValueError("official intent model output width is invalid")
        if any(len(row) != len(OFFICIAL_INTENT_FEATURE_NAMES) for row in self.weights):
            raise ValueError("official intent model feature width is invalid")
        if any(
            not math.isfinite(value)
            for row in self.weights
            for value in row
        ) or any(not math.isfinite(value) for value in self.biases):
            raise ValueError("official intent model parameters are non-finite")
        expected_grid = tuple(
            (l2_penalty, temperature, uniform_mix)
            for l2_penalty in OFFICIAL_INTENT_L2_GRID
            for temperature in PUBLIC_RANGE_TEMPERATURE_GRID
            for uniform_mix in PUBLIC_RANGE_UNIFORM_MIX_GRID
        )
        if tuple(
            (item.l2_penalty, item.temperature, item.uniform_mix)
            for item in self.selection_metrics
        ) != expected_grid:
            raise ValueError("official intent model selection grid is incomplete")
        selected = min(
            self.selection_metrics,
            key=lambda item: (
                -item.calibration_top1_accuracy,
                item.calibration_ece,
                item.calibration_nll,
                item.calibration_brier,
                item.l2_penalty,
                item.temperature,
                item.uniform_mix,
            ),
        )
        if (
            self.selected_l2_penalty,
            self.selected_temperature,
            self.selected_uniform_mix,
        ) != (
            selected.l2_penalty,
            selected.temperature,
            selected.uniform_mix,
        ):
            raise ValueError("official intent selected configuration is not optimal")
        if self.selected_training_loss < 0 or not math.isfinite(
            self.selected_training_loss
        ):
            raise ValueError("official intent selected training loss is invalid")
        if not self.train_root_ids or not self.calibration_root_ids:
            raise ValueError("official intent split roots are missing")
        if set(self.train_root_ids).intersection(self.calibration_root_ids):
            raise ValueError("official intent train/calibration roots overlap")
        if (self.observer_player, self.opponent_player) != (0, 1):
            raise ValueError("official intent model seat orientation changed")
        if self.target_policy_id != OFFICIAL_INTENT_TARGET_POLICY:
            raise ValueError("official intent model target policy changed")
        if not self.public_features_only or not self.opponent_action_semantics_only:
            raise ValueError("official intent model public-input claims changed")
        if any(
            (
                self.selected_candidate_identifier_input,
                self.selected_intent_identifier_input,
                self.target_policy_action_input,
                self.intent_labels_are_official_latent_truth,
                self.official_engine_executed,
            )
        ):
            raise ValueError("official intent model promoted an invalid claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def logits(
        self,
        intent_public_features: Sequence[tuple[str, float]],
    ) -> tuple[float, ...]:
        values = self.normalizer.transform(intent_public_features)
        return tuple(
            bias + sum(weight * value for weight, value in zip(row, values, strict=True))
            for row, bias in zip(self.weights, self.biases, strict=True)
        )

    def intent_probabilities(
        self,
        intent_public_features: Sequence[tuple[str, float]],
    ) -> tuple[tuple[str, float], ...]:
        probabilities = _softmax_probabilities(
            self.logits(intent_public_features),
            temperature=self.selected_temperature,
            uniform_mix=self.selected_uniform_mix,
        )
        return tuple(zip(self.intent_ids, probabilities, strict=True))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_INTENT_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_phase6ad_model_fingerprint": (
                self.source_phase6ad_model_fingerprint
            ),
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "intent_ids": list(self.intent_ids),
            "normalizer": self.normalizer.to_dict(),
            "parameters": {
                "weights": [list(row) for row in self.weights],
                "biases": list(self.biases),
            },
            "selection_grid": {
                "l2_penalties": list(OFFICIAL_INTENT_L2_GRID),
                "temperatures": list(PUBLIC_RANGE_TEMPERATURE_GRID),
                "uniform_mixes": list(PUBLIC_RANGE_UNIFORM_MIX_GRID),
                "training_steps": OFFICIAL_INTENT_TRAINING_STEPS,
                "learning_rate": OFFICIAL_INTENT_LEARNING_RATE,
                "selection_objective": [
                    "maximum_calibration_top1_accuracy",
                    "minimum_calibration_ece",
                    "minimum_calibration_nll",
                    "minimum_calibration_brier",
                ],
                "metrics": [asdict(item) for item in self.selection_metrics],
                "selected": {
                    "l2_penalty": self.selected_l2_penalty,
                    "temperature": self.selected_temperature,
                    "uniform_mix": self.selected_uniform_mix,
                    "training_loss": self.selected_training_loss,
                },
            },
            "split_provenance": {
                "train_root_ids": list(self.train_root_ids),
                "calibration_root_ids": list(self.calibration_root_ids),
                "train_rows_fingerprint": self.train_rows_fingerprint,
            },
            "orientation": {
                "observer_player": self.observer_player,
                "opponent_player": self.opponent_player,
                "target_policy_id": self.target_policy_id,
            },
            "claims": {
                "public_features_only": self.public_features_only,
                "opponent_action_semantics_only": self.opponent_action_semantics_only,
                "selected_candidate_identifier_input": (
                    self.selected_candidate_identifier_input
                ),
                "selected_intent_identifier_input": (
                    self.selected_intent_identifier_input
                ),
                "target_policy_action_input": self.target_policy_action_input,
                "intent_labels_are_official_latent_truth": (
                    self.intent_labels_are_official_latent_truth
                ),
                "official_engine_executed": self.official_engine_executed,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialIntentRangeModel":
        parameters = raw["parameters"]
        grid = raw["selection_grid"]
        selected = grid["selected"]
        split = raw["split_provenance"]
        orientation = raw["orientation"]
        claims = raw["claims"]
        if tuple(float(value) for value in grid["l2_penalties"]) != OFFICIAL_INTENT_L2_GRID:
            raise ValueError("official intent L2 grid changed")
        if tuple(float(value) for value in grid["temperatures"]) != PUBLIC_RANGE_TEMPERATURE_GRID:
            raise ValueError("official intent temperature grid changed")
        if tuple(float(value) for value in grid["uniform_mixes"]) != PUBLIC_RANGE_UNIFORM_MIX_GRID:
            raise ValueError("official intent mixture grid changed")
        if int(grid["training_steps"]) != OFFICIAL_INTENT_TRAINING_STEPS or not math.isclose(
            float(grid["learning_rate"]), OFFICIAL_INTENT_LEARNING_RATE, abs_tol=1e-15
        ):
            raise ValueError("official intent training protocol changed")
        if tuple(str(value) for value in grid["selection_objective"]) != (
            "maximum_calibration_top1_accuracy",
            "minimum_calibration_ece",
            "minimum_calibration_nll",
            "minimum_calibration_brier",
        ):
            raise ValueError("official intent selection objective changed")
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            source_phase6ad_model_fingerprint=str(
                raw["source_phase6ad_model_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            intent_ids=tuple(str(value) for value in raw["intent_ids"]),
            normalizer=NamedFeatureNormalizer.from_dict(raw["normalizer"]),
            weights=tuple(
                tuple(float(value) for value in row) for row in parameters["weights"]
            ),
            biases=tuple(float(value) for value in parameters["biases"]),
            selection_metrics=tuple(
                OfficialIntentSelectionMetric(**dict(item)) for item in grid["metrics"]
            ),
            selected_l2_penalty=float(selected["l2_penalty"]),
            selected_temperature=float(selected["temperature"]),
            selected_uniform_mix=float(selected["uniform_mix"]),
            selected_training_loss=float(selected["training_loss"]),
            train_root_ids=tuple(str(value) for value in split["train_root_ids"]),
            calibration_root_ids=tuple(
                str(value) for value in split["calibration_root_ids"]
            ),
            train_rows_fingerprint=str(split["train_rows_fingerprint"]),
            observer_player=int(orientation["observer_player"]),
            opponent_player=int(orientation["opponent_player"]),
            target_policy_id=str(orientation["target_policy_id"]),
            public_features_only=bool(claims["public_features_only"]),
            opponent_action_semantics_only=bool(
                claims["opponent_action_semantics_only"]
            ),
            selected_candidate_identifier_input=bool(
                claims["selected_candidate_identifier_input"]
            ),
            selected_intent_identifier_input=bool(
                claims["selected_intent_identifier_input"]
            ),
            target_policy_action_input=bool(claims["target_policy_action_input"]),
            intent_labels_are_official_latent_truth=bool(
                claims["intent_labels_are_official_latent_truth"]
            ),
            official_engine_executed=bool(claims["official_engine_executed"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official intent model fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OfficialJointBeliefProvider:
    """Compose frozen candidate/resource and intent components into 36 hypotheses."""

    candidate_model: OfficialPublicRangeModel
    intent_model: OfficialIntentRangeModel

    def __post_init__(self) -> None:
        if (
            self.intent_model.source_phase6ad_model_fingerprint
            != self.candidate_model.fingerprint
        ):
            raise ValueError("official joint provider component fingerprints mismatch")
        if self.intent_model.candidate_ids != self.candidate_model.candidate_ids:
            raise ValueError("official joint provider candidate catalogs mismatch")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "candidate_model": self.candidate_model.fingerprint,
                "intent_model": self.intent_model.fingerprint,
                "composition": "independent calibrated candidate_x_intent product",
            }
        )

    def opponent_range(
        self,
        candidate_public_features: Sequence[tuple[str, float]],
        intent_public_features: Sequence[tuple[str, float]],
        candidate_summaries: Sequence[CandidateParticleSummary],
    ) -> OpponentRange:
        candidate_probabilities = self.candidate_model.candidate_probabilities(
            candidate_public_features,
            candidate_summaries,
        )
        intent_probabilities = self.intent_model.intent_probabilities(
            intent_public_features
        )
        summaries = {item.candidate_id: item for item in candidate_summaries}
        hypotheses = []
        for candidate, candidate_probability in candidate_probabilities:
            summary = summaries[candidate]
            resource_counts = tuple(
                (name, max(0, int(round(value))))
                for name, value in zip(
                    OFFICIAL_RANGE_RESOURCE_NAMES,
                    summary.expected_resource_counts,
                    strict=True,
                )
            )
            for intent, intent_probability in intent_probabilities:
                hypotheses.append(
                    RangeHypothesis(
                        hypothesis_id=f"official-joint:{candidate}:{intent}",
                        weight=candidate_probability * intent_probability,
                        hidden_resource_counts=resource_counts,
                        intent_id=intent,
                        style_id="declared_intent_priority_proxy",
                        source=(
                            "Phase 6AD public candidate/resource range times "
                            "Phase 6AE opponent-only public behavior intent proxy"
                        ),
                    )
                )
        return OpponentRange.normalized(hypotheses)


def _linear_logits(
    values: Sequence[float],
    weights: Sequence[Sequence[float]],
    biases: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        bias + sum(weight * value for weight, value in zip(row, values, strict=True))
        for row, bias in zip(weights, biases, strict=True)
    )


def _intent_metrics_from_parameters(
    examples: Sequence[OfficialIntentTruthExample],
    *,
    normalizer: NamedFeatureNormalizer,
    weights: Sequence[Sequence[float]],
    biases: Sequence[float],
    temperature: float,
    uniform_mix: float,
) -> tuple[float, float, float, float]:
    nll = []
    brier = []
    correct = []
    confidence_correct = []
    intent_ids = tuple(item.value for item in OPPONENT_INTENTS)
    for example in examples:
        values = normalizer.transform(example.intent_public_features)
        probabilities = _softmax_probabilities(
            _linear_logits(values, weights, biases),
            temperature=temperature,
            uniform_mix=uniform_mix,
        )
        target = intent_ids.index(example.intent_id_supervision_only)
        nll.append(-math.log(max(probabilities[target], 1e-15)))
        brier.append(
            sum(
                (probability - float(index == target)) ** 2
                for index, probability in enumerate(probabilities)
            )
        )
        predicted = min(
            range(len(probabilities)),
            key=lambda index: (-probabilities[index], intent_ids[index]),
        )
        correct.append(predicted == target)
        confidence_correct.append((max(probabilities), predicted == target))
    return (
        sum(nll) / len(nll),
        sum(brier) / len(brier),
        sum(correct) / len(correct),
        min(1.0, max(0.0, _ece(confidence_correct))),
    )


def _fit_linear_intent_model(
    train_values: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    l2_penalty: float,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], float]:
    if train_values.dtype != torch.float64 or train_labels.dtype != torch.long:
        raise ValueError("official intent fit tensors use the wrong dtypes")
    weights = torch.zeros(
        (len(OPPONENT_INTENTS), train_values.shape[1]),
        dtype=torch.float64,
        requires_grad=True,
    )
    biases = torch.zeros(len(OPPONENT_INTENTS), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(
        (weights, biases),
        lr=OFFICIAL_INTENT_LEARNING_RATE,
    )
    loss = torch.tensor(float("nan"), dtype=torch.float64)
    for _ in range(OFFICIAL_INTENT_TRAINING_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = train_values @ weights.T + biases
        loss = F.cross_entropy(logits, train_labels) + l2_penalty * weights.square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = train_values @ weights.T + biases
        final_loss = F.cross_entropy(logits, train_labels) + l2_penalty * weights.square().mean()
    return (
        tuple(tuple(float(value) for value in row) for row in weights.detach().tolist()),
        tuple(float(value) for value in biases.detach().tolist()),
        float(final_loss.item()),
    )


def train_official_intent_range(
    dataset: OfficialIntentTruthDataset,
) -> OfficialIntentRangeModel:
    train = tuple(item for item in dataset.examples if item.split == "train")
    calibration = tuple(
        item for item in dataset.examples if item.split == "calibration"
    )
    train_rows_fingerprint = _fingerprint(
        [(item.example_id, item.intent_public_features) for item in train]
    )
    normalizer = NamedFeatureNormalizer.fit(
        [item.intent_public_features for item in train],
        fit_rows_fingerprint=train_rows_fingerprint,
    )
    train_values = torch.tensor(
        [normalizer.transform(item.intent_public_features) for item in train],
        dtype=torch.float64,
    )
    intent_ids = tuple(item.value for item in OPPONENT_INTENTS)
    train_labels = torch.tensor(
        [intent_ids.index(item.intent_id_supervision_only) for item in train],
        dtype=torch.long,
    )
    fits: dict[float, tuple[tuple[tuple[float, ...], ...], tuple[float, ...], float]] = {}
    metrics = []
    for l2_penalty in OFFICIAL_INTENT_L2_GRID:
        fit = _fit_linear_intent_model(
            train_values,
            train_labels,
            l2_penalty=l2_penalty,
        )
        fits[l2_penalty] = fit
        weights, biases, _ = fit
        for temperature in PUBLIC_RANGE_TEMPERATURE_GRID:
            for uniform_mix in PUBLIC_RANGE_UNIFORM_MIX_GRID:
                nll, brier, accuracy, calibration_ece = (
                    _intent_metrics_from_parameters(
                        calibration,
                        normalizer=normalizer,
                        weights=weights,
                        biases=biases,
                        temperature=temperature,
                        uniform_mix=uniform_mix,
                    )
                )
                metrics.append(
                    OfficialIntentSelectionMetric(
                        l2_penalty=l2_penalty,
                        temperature=temperature,
                        uniform_mix=uniform_mix,
                        calibration_nll=nll,
                        calibration_brier=brier,
                        calibration_top1_accuracy=accuracy,
                        calibration_ece=calibration_ece,
                    )
                )
    selected = min(
        metrics,
        key=lambda item: (
            -item.calibration_top1_accuracy,
            item.calibration_ece,
            item.calibration_nll,
            item.calibration_brier,
            item.l2_penalty,
            item.temperature,
            item.uniform_mix,
        ),
    )
    weights, biases, training_loss = fits[selected.l2_penalty]
    return OfficialIntentRangeModel(
        source_dataset_fingerprint=dataset.fingerprint,
        source_phase6ad_model_fingerprint=dataset.source_phase6ad_model_fingerprint,
        runtime_fingerprint=dataset.runtime_fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        candidate_ids=dataset.candidate_ids,
        intent_ids=dataset.intent_ids,
        normalizer=normalizer,
        weights=weights,
        biases=biases,
        selection_metrics=tuple(metrics),
        selected_l2_penalty=selected.l2_penalty,
        selected_temperature=selected.temperature,
        selected_uniform_mix=selected.uniform_mix,
        selected_training_loss=training_loss,
        train_root_ids=tuple(item.source_root_id for item in train),
        calibration_root_ids=tuple(item.source_root_id for item in calibration),
        train_rows_fingerprint=train_rows_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class OfficialJointBeliefEvaluationRow:
    example_id: str
    source_root_id: str
    split: str
    target_candidate_id_supervision_only: str
    target_intent_id_supervision_only: str
    candidate_probabilities: tuple[tuple[str, float], ...]
    intent_probabilities: tuple[tuple[str, float], ...]
    joint_probabilities: tuple[tuple[str, float], ...]
    predicted_resource_counts: tuple[float, ...]
    resource_presence_probabilities: tuple[float, ...]
    uniform_candidate_resource_presence_probabilities: tuple[float, ...]
    oracle_candidate_resource_presence_probabilities: tuple[float, ...]
    truth_resource_counts: tuple[int, ...]
    opponent_reverse: bool

    def __post_init__(self) -> None:
        _validate_digest(self.example_id, "example_id")
        _validate_digest(self.source_root_id, "source_root_id")
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("official joint evaluation split is invalid")
        candidate_ids = tuple(name for name, _ in self.candidate_probabilities)
        intent_ids = tuple(name for name, _ in self.intent_probabilities)
        joint_ids = tuple(name for name, _ in self.joint_probabilities)
        if self.target_candidate_id_supervision_only not in candidate_ids:
            raise ValueError("official joint evaluation candidate target is unsupported")
        if self.target_intent_id_supervision_only not in intent_ids:
            raise ValueError("official joint evaluation intent target is unsupported")
        if len(set(candidate_ids)) != len(candidate_ids) or len(set(intent_ids)) != len(intent_ids):
            raise ValueError("official joint evaluation marginal support is duplicated")
        if len(set(joint_ids)) != len(joint_ids) or len(joint_ids) != len(candidate_ids) * len(intent_ids):
            raise ValueError("official joint evaluation joint support is invalid")
        for distribution in (
            self.candidate_probabilities,
            self.intent_probabilities,
            self.joint_probabilities,
        ):
            if any(value < 0 or not math.isfinite(value) for _, value in distribution):
                raise ValueError("official joint evaluation probability is invalid")
            if not math.isclose(sum(value for _, value in distribution), 1.0, abs_tol=1e-8):
                raise ValueError("official joint evaluation distribution is not normalized")
        target_joint = (
            f"{self.target_candidate_id_supervision_only}|"
            f"{self.target_intent_id_supervision_only}"
        )
        if target_joint not in joint_ids:
            raise ValueError("official joint evaluation pair target is unsupported")
        width = len(OFFICIAL_RANGE_RESOURCE_NAMES)
        if any(
            len(values) != width
            for values in (
                self.predicted_resource_counts,
                self.resource_presence_probabilities,
                self.uniform_candidate_resource_presence_probabilities,
                self.oracle_candidate_resource_presence_probabilities,
                self.truth_resource_counts,
            )
        ):
            raise ValueError("official joint evaluation resource width is invalid")
        if any(value < 0 or not math.isfinite(value) for value in self.predicted_resource_counts):
            raise ValueError("official joint predicted resource count is invalid")
        for values in (
            self.resource_presence_probabilities,
            self.uniform_candidate_resource_presence_probabilities,
            self.oracle_candidate_resource_presence_probabilities,
        ):
            if any(value < 0 or value > 1 or not math.isfinite(value) for value in values):
                raise ValueError("official joint resource probability is invalid")

    @staticmethod
    def _categorical_metrics(
        distribution: Sequence[tuple[str, float]],
        target: str,
    ) -> tuple[float, float, float, bool]:
        probabilities = dict(distribution)
        nll = -math.log(max(probabilities[target], 1e-15))
        brier = sum(
            (probability - float(name == target)) ** 2
            for name, probability in distribution
        )
        confidence = max(probability for _, probability in distribution)
        predicted = min(distribution, key=lambda item: (-item[1], item[0]))[0]
        return nll, brier, confidence, predicted == target

    @property
    def candidate_metrics(self) -> tuple[float, float, float, bool]:
        return self._categorical_metrics(
            self.candidate_probabilities,
            self.target_candidate_id_supervision_only,
        )

    @property
    def intent_metrics(self) -> tuple[float, float, float, bool]:
        return self._categorical_metrics(
            self.intent_probabilities,
            self.target_intent_id_supervision_only,
        )

    @property
    def joint_target(self) -> str:
        return (
            f"{self.target_candidate_id_supervision_only}|"
            f"{self.target_intent_id_supervision_only}"
        )

    @property
    def joint_metrics(self) -> tuple[float, float, float, bool]:
        return self._categorical_metrics(self.joint_probabilities, self.joint_target)

    @property
    def truth_presence(self) -> tuple[float, ...]:
        return tuple(float(value > 0) for value in self.truth_resource_counts)

    @staticmethod
    def _mean_brier(predicted: Sequence[float], truth: Sequence[float]) -> float:
        return sum(
            (left - right) ** 2 for left, right in zip(predicted, truth, strict=True)
        ) / len(truth)

    @property
    def resource_presence_brier(self) -> float:
        return self._mean_brier(
            self.resource_presence_probabilities,
            self.truth_presence,
        )

    @property
    def uniform_candidate_resource_brier(self) -> float:
        return self._mean_brier(
            self.uniform_candidate_resource_presence_probabilities,
            self.truth_presence,
        )

    @property
    def oracle_candidate_resource_brier(self) -> float:
        return self._mean_brier(
            self.oracle_candidate_resource_presence_probabilities,
            self.truth_presence,
        )

    @property
    def resource_count_mae(self) -> float:
        return sum(
            abs(predicted - truth)
            for predicted, truth in zip(
                self.predicted_resource_counts,
                self.truth_resource_counts,
                strict=True,
            )
        ) / len(self.truth_resource_counts)

    def to_dict(self) -> dict[str, Any]:
        candidate_nll, candidate_brier, candidate_confidence, candidate_correct = (
            self.candidate_metrics
        )
        intent_nll, intent_brier, intent_confidence, intent_correct = self.intent_metrics
        joint_nll, joint_brier, joint_confidence, joint_correct = self.joint_metrics
        return {
            "example_id": self.example_id,
            "source_root_id": self.source_root_id,
            "split": self.split,
            "target_candidate_id_supervision_only": (
                self.target_candidate_id_supervision_only
            ),
            "target_intent_id_supervision_only": self.target_intent_id_supervision_only,
            "candidate_probabilities": [list(item) for item in self.candidate_probabilities],
            "intent_probabilities": [list(item) for item in self.intent_probabilities],
            "joint_probabilities": [list(item) for item in self.joint_probabilities],
            "predicted_resource_counts": list(self.predicted_resource_counts),
            "resource_presence_probabilities": list(
                self.resource_presence_probabilities
            ),
            "uniform_candidate_resource_presence_probabilities": list(
                self.uniform_candidate_resource_presence_probabilities
            ),
            "oracle_candidate_resource_presence_probabilities": list(
                self.oracle_candidate_resource_presence_probabilities
            ),
            "truth_resource_counts": list(self.truth_resource_counts),
            "opponent_reverse": self.opponent_reverse,
            "metrics": {
                "candidate_nll": candidate_nll,
                "candidate_brier": candidate_brier,
                "candidate_confidence": candidate_confidence,
                "candidate_top1_correct": candidate_correct,
                "intent_nll": intent_nll,
                "intent_brier": intent_brier,
                "intent_confidence": intent_confidence,
                "intent_top1_correct": intent_correct,
                "joint_nll": joint_nll,
                "joint_brier": joint_brier,
                "joint_confidence": joint_confidence,
                "joint_top1_correct": joint_correct,
                "resource_presence_brier": self.resource_presence_brier,
                "uniform_candidate_resource_brier": (
                    self.uniform_candidate_resource_brier
                ),
                "oracle_candidate_resource_brier": self.oracle_candidate_resource_brier,
                "resource_count_mae": self.resource_count_mae,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialJointBeliefEvaluationRow":
        result = cls(
            example_id=str(raw["example_id"]),
            source_root_id=str(raw["source_root_id"]),
            split=str(raw["split"]),
            target_candidate_id_supervision_only=str(
                raw["target_candidate_id_supervision_only"]
            ),
            target_intent_id_supervision_only=str(
                raw["target_intent_id_supervision_only"]
            ),
            candidate_probabilities=tuple(
                (str(name), float(value)) for name, value in raw["candidate_probabilities"]
            ),
            intent_probabilities=tuple(
                (str(name), float(value)) for name, value in raw["intent_probabilities"]
            ),
            joint_probabilities=tuple(
                (str(name), float(value)) for name, value in raw["joint_probabilities"]
            ),
            predicted_resource_counts=tuple(
                float(value) for value in raw["predicted_resource_counts"]
            ),
            resource_presence_probabilities=tuple(
                float(value) for value in raw["resource_presence_probabilities"]
            ),
            uniform_candidate_resource_presence_probabilities=tuple(
                float(value)
                for value in raw["uniform_candidate_resource_presence_probabilities"]
            ),
            oracle_candidate_resource_presence_probabilities=tuple(
                float(value)
                for value in raw["oracle_candidate_resource_presence_probabilities"]
            ),
            truth_resource_counts=tuple(int(value) for value in raw["truth_resource_counts"]),
            opponent_reverse=bool(raw["opponent_reverse"]),
        )
        expected = result.to_dict()["metrics"]
        observed = raw["metrics"]
        for name, value in expected.items():
            if isinstance(value, bool):
                if bool(observed[name]) != value:
                    raise ValueError("official joint boolean metric mismatch")
            elif not math.isclose(float(observed[name]), float(value), abs_tol=1e-12):
                raise ValueError("official joint numeric metric mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OfficialJointBeliefSplitMetrics:
    split: str
    roots: int
    candidate_strata: int
    intent_strata: int
    candidate_nll: float
    candidate_brier: float
    candidate_top1_accuracy: float
    candidate_ece: float
    intent_nll: float
    intent_brier: float
    intent_top1_accuracy: float
    intent_ece: float
    joint_nll: float
    joint_brier: float
    joint_top1_accuracy: float
    joint_ece: float
    resource_presence_brier: float
    uniform_candidate_resource_brier: float
    oracle_candidate_resource_brier: float
    resource_ece: float
    resource_count_mae: float

    def __post_init__(self) -> None:
        if self.split not in ALIGNED_VALUE_SPLITS or min(
            self.roots,
            self.candidate_strata,
            self.intent_strata,
        ) <= 0:
            raise ValueError("official joint split metric identity is invalid")
        for name in (
            "candidate_nll",
            "candidate_brier",
            "candidate_ece",
            "intent_nll",
            "intent_brier",
            "intent_ece",
            "joint_nll",
            "joint_brier",
            "joint_ece",
            "resource_presence_brier",
            "uniform_candidate_resource_brier",
            "oracle_candidate_resource_brier",
            "resource_ece",
            "resource_count_mae",
        ):
            if getattr(self, name) < 0 or not math.isfinite(getattr(self, name)):
                raise ValueError(f"official joint {name} is invalid")
        for name in (
            "candidate_top1_accuracy",
            "intent_top1_accuracy",
            "joint_top1_accuracy",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"official joint {name} is invalid")


@dataclass(frozen=True, slots=True)
class OfficialIntentSliceMetric:
    slice_kind: str
    slice_id: str
    roots: int
    intent_nll: float
    intent_top1_accuracy: float

    def __post_init__(self) -> None:
        if self.slice_kind not in ("candidate", "opponent_reverse"):
            raise ValueError("official intent slice kind is invalid")
        if not self.slice_id or self.roots <= 0:
            raise ValueError("official intent slice identity is invalid")
        if self.intent_nll < 0 or not math.isfinite(self.intent_nll):
            raise ValueError("official intent slice NLL is invalid")
        if not 0 <= self.intent_top1_accuracy <= 1:
            raise ValueError("official intent slice accuracy is invalid")


@dataclass(frozen=True, slots=True)
class OfficialJointBeliefCalibration:
    dataset_fingerprint: str
    candidate_model_fingerprint: str
    intent_model_fingerprint: str
    joint_provider_fingerprint: str
    evaluation_rows: tuple[OfficialJointBeliefEvaluationRow, ...]
    train: OfficialJointBeliefSplitMetrics
    calibration: OfficialJointBeliefSplitMetrics
    validation: OfficialJointBeliefSplitMetrics
    validation_intent_slices: tuple[OfficialIntentSliceMetric, ...]
    uniform_candidate_nll: float
    uniform_candidate_brier: float
    uniform_intent_nll: float
    uniform_intent_brier: float
    uniform_joint_nll: float
    uniform_joint_brier: float
    minimum_calibration_roots: int
    minimum_validation_roots: int
    candidate_ece_limit: float
    intent_ece_limit: float
    joint_ece_limit: float
    resource_ece_limit: float
    minimum_intent_top1: float
    minimum_joint_top1: float
    source_phase6ad_gate_passed: bool
    root_split_gate_passed: bool
    sufficient_calibration_roots_gate_passed: bool
    sufficient_validation_roots_gate_passed: bool
    observer_and_seat_decoupling_gate_passed: bool
    heldout_candidate_transfer_gate_passed: bool
    heldout_declared_intent_proxy_gate_passed: bool
    heldout_joint_range_gate_passed: bool
    heldout_hidden_resource_gate_passed: bool
    in_scope_joint_candidate_intent_resource_range_calibrated: bool
    declared_policy_intent_proxy_calibrated: bool
    opponent_latent_intent_calibrated: bool = False
    full_official_public_belief_calibrated: bool = False
    counterfactual_value_compatible: bool = False
    safe_resolver_compatible: bool = False
    opponent_exploitation_compatible: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "dataset_fingerprint",
            "candidate_model_fingerprint",
            "intent_model_fingerprint",
            "joint_provider_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if tuple(item.split for item in (self.train, self.calibration, self.validation)) != ALIGNED_VALUE_SPLITS:
            raise ValueError("official joint split metric order is invalid")
        if len(self.evaluation_rows) != sum(
            item.roots for item in (self.train, self.calibration, self.validation)
        ):
            raise ValueError("official joint evaluation row accounting is inconsistent")
        if not self.validation_intent_slices:
            raise ValueError("official joint validation slices are missing")
        expected_in_scope = all(
            (
                self.source_phase6ad_gate_passed,
                self.root_split_gate_passed,
                self.sufficient_calibration_roots_gate_passed,
                self.sufficient_validation_roots_gate_passed,
                self.observer_and_seat_decoupling_gate_passed,
                self.heldout_candidate_transfer_gate_passed,
                self.heldout_declared_intent_proxy_gate_passed,
                self.heldout_joint_range_gate_passed,
                self.heldout_hidden_resource_gate_passed,
            )
        )
        if self.in_scope_joint_candidate_intent_resource_range_calibrated != expected_in_scope:
            raise ValueError("official joint in-scope gate is inconsistent")
        if self.declared_policy_intent_proxy_calibrated != (
            self.heldout_declared_intent_proxy_gate_passed
            and self.observer_and_seat_decoupling_gate_passed
            and self.sufficient_calibration_roots_gate_passed
            and self.sufficient_validation_roots_gate_passed
        ):
            raise ValueError("official declared-intent proxy gate is inconsistent")
        if any(
            (
                self.opponent_latent_intent_calibrated,
                self.full_official_public_belief_calibrated,
                self.counterfactual_value_compatible,
                self.safe_resolver_compatible,
                self.opponent_exploitation_compatible,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("official joint calibration promoted a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_INTENT_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "candidate_model_fingerprint": self.candidate_model_fingerprint,
            "intent_model_fingerprint": self.intent_model_fingerprint,
            "joint_provider_fingerprint": self.joint_provider_fingerprint,
            "evaluation_rows": [item.to_dict() for item in self.evaluation_rows],
            "metrics": {
                "train": asdict(self.train),
                "calibration": asdict(self.calibration),
                "validation": asdict(self.validation),
                "validation_intent_slices": [
                    asdict(item) for item in self.validation_intent_slices
                ],
                "uniform_candidate_nll": self.uniform_candidate_nll,
                "uniform_candidate_brier": self.uniform_candidate_brier,
                "uniform_intent_nll": self.uniform_intent_nll,
                "uniform_intent_brier": self.uniform_intent_brier,
                "uniform_joint_nll": self.uniform_joint_nll,
                "uniform_joint_brier": self.uniform_joint_brier,
            },
            "thresholds": {
                "minimum_calibration_roots": self.minimum_calibration_roots,
                "minimum_validation_roots": self.minimum_validation_roots,
                "candidate_ece_limit": self.candidate_ece_limit,
                "intent_ece_limit": self.intent_ece_limit,
                "joint_ece_limit": self.joint_ece_limit,
                "resource_ece_limit": self.resource_ece_limit,
                "minimum_intent_top1": self.minimum_intent_top1,
                "minimum_joint_top1": self.minimum_joint_top1,
            },
            "gates": {
                "source_phase6ad": self.source_phase6ad_gate_passed,
                "root_split": self.root_split_gate_passed,
                "sufficient_calibration_roots": (
                    self.sufficient_calibration_roots_gate_passed
                ),
                "sufficient_validation_roots": (
                    self.sufficient_validation_roots_gate_passed
                ),
                "observer_and_seat_decoupling": (
                    self.observer_and_seat_decoupling_gate_passed
                ),
                "heldout_candidate_transfer": (
                    self.heldout_candidate_transfer_gate_passed
                ),
                "heldout_declared_intent_proxy": (
                    self.heldout_declared_intent_proxy_gate_passed
                ),
                "heldout_joint_range": self.heldout_joint_range_gate_passed,
                "heldout_hidden_resource": self.heldout_hidden_resource_gate_passed,
                "in_scope_joint_candidate_intent_resource_range_calibrated": (
                    self.in_scope_joint_candidate_intent_resource_range_calibrated
                ),
                "declared_policy_intent_proxy_calibrated": (
                    self.declared_policy_intent_proxy_calibrated
                ),
                "opponent_latent_intent_calibrated": (
                    self.opponent_latent_intent_calibrated
                ),
                "full_official_public_belief_calibrated": (
                    self.full_official_public_belief_calibrated
                ),
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "opponent_exploitation_compatible": (
                    self.opponent_exploitation_compatible
                ),
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
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialJointBeliefCalibration":
        metrics = raw["metrics"]
        thresholds = raw["thresholds"]
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            candidate_model_fingerprint=str(raw["candidate_model_fingerprint"]),
            intent_model_fingerprint=str(raw["intent_model_fingerprint"]),
            joint_provider_fingerprint=str(raw["joint_provider_fingerprint"]),
            evaluation_rows=tuple(
                OfficialJointBeliefEvaluationRow.from_dict(item)
                for item in raw["evaluation_rows"]
            ),
            train=OfficialJointBeliefSplitMetrics(**dict(metrics["train"])),
            calibration=OfficialJointBeliefSplitMetrics(
                **dict(metrics["calibration"])
            ),
            validation=OfficialJointBeliefSplitMetrics(**dict(metrics["validation"])),
            validation_intent_slices=tuple(
                OfficialIntentSliceMetric(**dict(item))
                for item in metrics["validation_intent_slices"]
            ),
            uniform_candidate_nll=float(metrics["uniform_candidate_nll"]),
            uniform_candidate_brier=float(metrics["uniform_candidate_brier"]),
            uniform_intent_nll=float(metrics["uniform_intent_nll"]),
            uniform_intent_brier=float(metrics["uniform_intent_brier"]),
            uniform_joint_nll=float(metrics["uniform_joint_nll"]),
            uniform_joint_brier=float(metrics["uniform_joint_brier"]),
            minimum_calibration_roots=int(thresholds["minimum_calibration_roots"]),
            minimum_validation_roots=int(thresholds["minimum_validation_roots"]),
            candidate_ece_limit=float(thresholds["candidate_ece_limit"]),
            intent_ece_limit=float(thresholds["intent_ece_limit"]),
            joint_ece_limit=float(thresholds["joint_ece_limit"]),
            resource_ece_limit=float(thresholds["resource_ece_limit"]),
            minimum_intent_top1=float(thresholds["minimum_intent_top1"]),
            minimum_joint_top1=float(thresholds["minimum_joint_top1"]),
            source_phase6ad_gate_passed=bool(gates["source_phase6ad"]),
            root_split_gate_passed=bool(gates["root_split"]),
            sufficient_calibration_roots_gate_passed=bool(
                gates["sufficient_calibration_roots"]
            ),
            sufficient_validation_roots_gate_passed=bool(
                gates["sufficient_validation_roots"]
            ),
            observer_and_seat_decoupling_gate_passed=bool(
                gates["observer_and_seat_decoupling"]
            ),
            heldout_candidate_transfer_gate_passed=bool(
                gates["heldout_candidate_transfer"]
            ),
            heldout_declared_intent_proxy_gate_passed=bool(
                gates["heldout_declared_intent_proxy"]
            ),
            heldout_joint_range_gate_passed=bool(gates["heldout_joint_range"]),
            heldout_hidden_resource_gate_passed=bool(
                gates["heldout_hidden_resource"]
            ),
            in_scope_joint_candidate_intent_resource_range_calibrated=bool(
                gates["in_scope_joint_candidate_intent_resource_range_calibrated"]
            ),
            declared_policy_intent_proxy_calibrated=bool(
                gates["declared_policy_intent_proxy_calibrated"]
            ),
            opponent_latent_intent_calibrated=bool(
                gates["opponent_latent_intent_calibrated"]
            ),
            full_official_public_belief_calibrated=bool(
                gates["full_official_public_belief_calibrated"]
            ),
            counterfactual_value_compatible=bool(
                gates["counterfactual_value_compatible"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            opponent_exploitation_compatible=bool(
                gates["opponent_exploitation_compatible"]
            ),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official joint calibration fingerprint mismatch")
        return result


def _joint_evaluation_row(
    example: OfficialIntentTruthExample,
    *,
    candidate_model: OfficialPublicRangeModel,
    intent_model: OfficialIntentRangeModel,
) -> OfficialJointBeliefEvaluationRow:
    candidate_probabilities = candidate_model.candidate_probabilities(
        example.candidate_public_features,
        example.candidate_particle_summaries,
    )
    intent_probabilities = intent_model.intent_probabilities(
        example.intent_public_features
    )
    joint_raw = tuple(
        (f"{candidate}|{intent}", candidate_probability * intent_probability)
        for candidate, candidate_probability in candidate_probabilities
        for intent, intent_probability in intent_probabilities
    )
    joint_total = sum(value for _, value in joint_raw)
    joint_probabilities = tuple(
        (name, value / joint_total) for name, value in joint_raw
    )
    expected, presence = _mixed_resource_predictions(
        candidate_probabilities,
        example.candidate_particle_summaries,
    )
    uniform_candidate_probabilities = tuple(
        (candidate, 1.0 / len(candidate_model.candidate_ids))
        for candidate in candidate_model.candidate_ids
    )
    _, uniform_presence = _mixed_resource_predictions(
        uniform_candidate_probabilities,
        example.candidate_particle_summaries,
    )
    oracle = next(
        item
        for item in example.candidate_particle_summaries
        if item.candidate_id == example.candidate_id_supervision_only
    )
    return OfficialJointBeliefEvaluationRow(
        example_id=example.example_id,
        source_root_id=example.source_root_id,
        split=example.split,
        target_candidate_id_supervision_only=example.candidate_id_supervision_only,
        target_intent_id_supervision_only=example.intent_id_supervision_only,
        candidate_probabilities=candidate_probabilities,
        intent_probabilities=intent_probabilities,
        joint_probabilities=joint_probabilities,
        predicted_resource_counts=expected,
        resource_presence_probabilities=presence,
        uniform_candidate_resource_presence_probabilities=uniform_presence,
        oracle_candidate_resource_presence_probabilities=(
            oracle.resource_presence_probabilities
        ),
        truth_resource_counts=example.truth_resource_counts,
        opponent_reverse=example.prefix_trace.opponent_reverse,
    )


def _joint_split_metrics(
    rows: Sequence[OfficialJointBeliefEvaluationRow],
    split: str,
) -> OfficialJointBeliefSplitMetrics:
    selected = tuple(item for item in rows if item.split == split)
    if not selected:
        raise ValueError("official joint split has no rows")
    candidate_metrics = tuple(item.candidate_metrics for item in selected)
    intent_metrics = tuple(item.intent_metrics for item in selected)
    joint_metrics = tuple(item.joint_metrics for item in selected)
    resource_pairs = tuple(
        (probability, truth > 0)
        for item in selected
        for probability, truth in zip(
            item.resource_presence_probabilities,
            item.truth_resource_counts,
            strict=True,
        )
    )
    return OfficialJointBeliefSplitMetrics(
        split=split,
        roots=len(selected),
        candidate_strata=len(
            {item.target_candidate_id_supervision_only for item in selected}
        ),
        intent_strata=len(
            {item.target_intent_id_supervision_only for item in selected}
        ),
        candidate_nll=sum(item[0] for item in candidate_metrics) / len(selected),
        candidate_brier=sum(item[1] for item in candidate_metrics) / len(selected),
        candidate_top1_accuracy=sum(item[3] for item in candidate_metrics)
        / len(selected),
        candidate_ece=_ece(tuple((item[2], item[3]) for item in candidate_metrics)),
        intent_nll=sum(item[0] for item in intent_metrics) / len(selected),
        intent_brier=sum(item[1] for item in intent_metrics) / len(selected),
        intent_top1_accuracy=sum(item[3] for item in intent_metrics) / len(selected),
        intent_ece=_ece(tuple((item[2], item[3]) for item in intent_metrics)),
        joint_nll=sum(item[0] for item in joint_metrics) / len(selected),
        joint_brier=sum(item[1] for item in joint_metrics) / len(selected),
        joint_top1_accuracy=sum(item[3] for item in joint_metrics) / len(selected),
        joint_ece=_ece(tuple((item[2], item[3]) for item in joint_metrics)),
        resource_presence_brier=sum(
            item.resource_presence_brier for item in selected
        )
        / len(selected),
        uniform_candidate_resource_brier=sum(
            item.uniform_candidate_resource_brier for item in selected
        )
        / len(selected),
        oracle_candidate_resource_brier=sum(
            item.oracle_candidate_resource_brier for item in selected
        )
        / len(selected),
        resource_ece=_ece(resource_pairs),
        resource_count_mae=sum(item.resource_count_mae for item in selected)
        / len(selected),
    )


def _intent_slice_metric(
    rows: Sequence[OfficialJointBeliefEvaluationRow],
    *,
    slice_kind: str,
    slice_id: str,
) -> OfficialIntentSliceMetric:
    if slice_kind == "candidate":
        selected = tuple(
            item
            for item in rows
            if item.target_candidate_id_supervision_only == slice_id
        )
    elif slice_kind == "opponent_reverse":
        expected = slice_id == "true"
        selected = tuple(item for item in rows if item.opponent_reverse == expected)
    else:
        raise ValueError("official intent slice kind is unsupported")
    metrics = tuple(item.intent_metrics for item in selected)
    return OfficialIntentSliceMetric(
        slice_kind=slice_kind,
        slice_id=slice_id,
        roots=len(selected),
        intent_nll=sum(item[0] for item in metrics) / len(metrics),
        intent_top1_accuracy=sum(item[3] for item in metrics) / len(metrics),
    )


def evaluate_official_joint_belief(
    dataset: OfficialIntentTruthDataset,
    candidate_model: OfficialPublicRangeModel,
    source_candidate_calibration: OfficialRangeCalibration,
    intent_model: OfficialIntentRangeModel,
) -> OfficialJointBeliefCalibration:
    if candidate_model.fingerprint != dataset.source_phase6ad_model_fingerprint:
        raise ValueError("official joint candidate model/source mismatch")
    if source_candidate_calibration.fingerprint != dataset.source_phase6ad_calibration_fingerprint:
        raise ValueError("official joint candidate calibration/source mismatch")
    if intent_model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("official joint intent model/dataset mismatch")
    provider = OfficialJointBeliefProvider(candidate_model, intent_model)
    rows = tuple(
        _joint_evaluation_row(
            item,
            candidate_model=candidate_model,
            intent_model=intent_model,
        )
        for item in dataset.examples
    )
    train = _joint_split_metrics(rows, "train")
    calibration = _joint_split_metrics(rows, "calibration")
    validation = _joint_split_metrics(rows, "validation")
    validation_rows = tuple(item for item in rows if item.split == "validation")
    slices = tuple(
        _intent_slice_metric(
            validation_rows,
            slice_kind="candidate",
            slice_id=candidate,
        )
        for candidate in dataset.candidate_ids
    ) + tuple(
        _intent_slice_metric(
            validation_rows,
            slice_kind="opponent_reverse",
            slice_id=value,
        )
        for value in ("false", "true")
    )
    uniform_candidate_nll = math.log(len(dataset.candidate_ids))
    uniform_candidate_brier = 1.0 - 1.0 / len(dataset.candidate_ids)
    uniform_intent_nll = math.log(len(dataset.intent_ids))
    uniform_intent_brier = 1.0 - 1.0 / len(dataset.intent_ids)
    uniform_joint_nll = math.log(len(dataset.candidate_ids) * len(dataset.intent_ids))
    uniform_joint_brier = 1.0 - 1.0 / (
        len(dataset.candidate_ids) * len(dataset.intent_ids)
    )
    source_gate = source_candidate_calibration.in_scope_official_hidden_card_range_calibrated
    enough_calibration = calibration.roots >= OFFICIAL_INTENT_MINIMUM_CALIBRATION_ROOTS
    enough_validation = validation.roots >= OFFICIAL_INTENT_MINIMUM_VALIDATION_ROOTS
    candidate_gate = all(
        (
            calibration.candidate_nll < uniform_candidate_nll,
            validation.candidate_nll < uniform_candidate_nll,
            calibration.candidate_brier < uniform_candidate_brier,
            validation.candidate_brier < uniform_candidate_brier,
            calibration.candidate_ece <= OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT,
            validation.candidate_ece <= OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT,
        )
    )
    intent_gate = all(
        (
            calibration.intent_nll < uniform_intent_nll,
            validation.intent_nll < uniform_intent_nll,
            calibration.intent_brier < uniform_intent_brier,
            validation.intent_brier < uniform_intent_brier,
            calibration.intent_ece <= OFFICIAL_INTENT_ECE_LIMIT,
            validation.intent_ece <= OFFICIAL_INTENT_ECE_LIMIT,
            calibration.intent_top1_accuracy >= OFFICIAL_INTENT_MINIMUM_TOP1,
            validation.intent_top1_accuracy >= OFFICIAL_INTENT_MINIMUM_TOP1,
            all(
                item.intent_nll < uniform_intent_nll
                and item.intent_top1_accuracy >= 2.0 / 3.0
                for item in slices
                if item.slice_kind == "candidate"
            ),
            all(
                item.intent_top1_accuracy >= 0.70
                for item in slices
                if item.slice_kind == "opponent_reverse"
            ),
        )
    )
    joint_gate = all(
        (
            calibration.joint_nll < uniform_joint_nll,
            validation.joint_nll < uniform_joint_nll,
            calibration.joint_brier < uniform_joint_brier,
            validation.joint_brier < uniform_joint_brier,
            calibration.joint_ece <= OFFICIAL_JOINT_ECE_LIMIT,
            validation.joint_ece <= OFFICIAL_JOINT_ECE_LIMIT,
            calibration.joint_top1_accuracy >= OFFICIAL_JOINT_MINIMUM_TOP1,
            validation.joint_top1_accuracy >= OFFICIAL_JOINT_MINIMUM_TOP1,
        )
    )
    resource_gate = all(
        (
            calibration.resource_presence_brier
            <= calibration.uniform_candidate_resource_brier,
            validation.resource_presence_brier
            <= validation.uniform_candidate_resource_brier,
            calibration.resource_ece <= OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
            validation.resource_ece <= OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
        )
    )
    observer_gate = all(
        (
            dataset.target_policy_fixed_across_intents,
            dataset.opponent_only_action_semantics,
            dataset.observer_orientation_validated,
        )
    )
    in_scope = all(
        (
            source_gate,
            dataset.collection_split_disjoint,
            enough_calibration,
            enough_validation,
            observer_gate,
            candidate_gate,
            intent_gate,
            joint_gate,
            resource_gate,
        )
    )
    return OfficialJointBeliefCalibration(
        dataset_fingerprint=dataset.fingerprint,
        candidate_model_fingerprint=candidate_model.fingerprint,
        intent_model_fingerprint=intent_model.fingerprint,
        joint_provider_fingerprint=provider.fingerprint,
        evaluation_rows=rows,
        train=train,
        calibration=calibration,
        validation=validation,
        validation_intent_slices=slices,
        uniform_candidate_nll=uniform_candidate_nll,
        uniform_candidate_brier=uniform_candidate_brier,
        uniform_intent_nll=uniform_intent_nll,
        uniform_intent_brier=uniform_intent_brier,
        uniform_joint_nll=uniform_joint_nll,
        uniform_joint_brier=uniform_joint_brier,
        minimum_calibration_roots=OFFICIAL_INTENT_MINIMUM_CALIBRATION_ROOTS,
        minimum_validation_roots=OFFICIAL_INTENT_MINIMUM_VALIDATION_ROOTS,
        candidate_ece_limit=OFFICIAL_RANGE_CANDIDATE_ECE_LIMIT,
        intent_ece_limit=OFFICIAL_INTENT_ECE_LIMIT,
        joint_ece_limit=OFFICIAL_JOINT_ECE_LIMIT,
        resource_ece_limit=OFFICIAL_RANGE_RESOURCE_ECE_LIMIT,
        minimum_intent_top1=OFFICIAL_INTENT_MINIMUM_TOP1,
        minimum_joint_top1=OFFICIAL_JOINT_MINIMUM_TOP1,
        source_phase6ad_gate_passed=source_gate,
        root_split_gate_passed=dataset.collection_split_disjoint,
        sufficient_calibration_roots_gate_passed=enough_calibration,
        sufficient_validation_roots_gate_passed=enough_validation,
        observer_and_seat_decoupling_gate_passed=observer_gate,
        heldout_candidate_transfer_gate_passed=candidate_gate,
        heldout_declared_intent_proxy_gate_passed=intent_gate,
        heldout_joint_range_gate_passed=joint_gate,
        heldout_hidden_resource_gate_passed=resource_gate,
        in_scope_joint_candidate_intent_resource_range_calibrated=in_scope,
        declared_policy_intent_proxy_calibrated=(
            intent_gate and observer_gate and enough_calibration and enough_validation
        ),
    )


@dataclass(frozen=True, slots=True)
class Phase6OfficialIntentEvidence:
    benchmark_id: str
    status: str
    source_phase6ad_dataset_fingerprint: str
    source_phase6ad_model_fingerprint: str
    dataset_fingerprint: str
    intent_model_fingerprint: str
    joint_provider_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    roots_per_split: int
    candidate_strata: int
    intent_strata: int
    live_selection_count: int
    opponent_main_action_count: int
    particles_per_candidate: int
    total_public_consistent_particles: int
    selected_l2_penalty: float
    selected_temperature: float
    selected_uniform_mix: float
    validation_candidate_nll: float
    validation_candidate_ece: float
    validation_intent_nll: float
    validation_intent_top1_accuracy: float
    validation_intent_ece: float
    validation_joint_nll: float
    validation_joint_top1_accuracy: float
    validation_joint_ece: float
    validation_resource_presence_brier: float
    validation_resource_ece: float
    in_scope_joint_candidate_intent_resource_range_calibrated: bool
    declared_policy_intent_proxy_calibrated: bool
    opponent_latent_intent_calibrated: bool
    full_official_public_belief_calibrated: bool
    counterfactual_value_compatible: bool
    safe_resolver_enabled: bool
    opponent_exploitation_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AE evidence identity is missing")
        for name in (
            "source_phase6ad_dataset_fingerprint",
            "source_phase6ad_model_fingerprint",
            "dataset_fingerprint",
            "intent_model_fingerprint",
            "joint_provider_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if min(
            self.root_count,
            self.roots_per_split,
            self.candidate_strata,
            self.intent_strata,
            self.live_selection_count,
            self.opponent_main_action_count,
            self.particles_per_candidate,
            self.total_public_consistent_particles,
        ) <= 0:
            raise ValueError("Phase 6AE evidence accounting is invalid")
        if (
            self.in_scope_joint_candidate_intent_resource_range_calibrated
            and not self.declared_policy_intent_proxy_calibrated
        ):
            raise ValueError("Phase 6AE intent proxy accounting is invalid")
        if any(
            (
                self.opponent_latent_intent_calibrated,
                self.full_official_public_belief_calibrated,
                self.counterfactual_value_compatible,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AE evidence enabled a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_INTENT_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AE seat-decoupled official intent evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Fresh official roots | {self.root_count} |
| Train / calibration / validation roots | {self.roots_per_split} / {self.roots_per_split} / {self.roots_per_split} |
| Candidate / declared-intent strata | {self.candidate_strata} / {self.intent_strata} |
| Genuine live selections | {self.live_selection_count} |
| Opponent public MAIN actions | {self.opponent_main_action_count} |
| Public-consistent allocation particles | {self.total_public_consistent_particles} |
| Validation candidate NLL / ECE | `{self.validation_candidate_nll:.6f}` / `{self.validation_candidate_ece:.6f}` |
| Validation declared-intent NLL / top-1 / ECE | `{self.validation_intent_nll:.6f}` / `{self.validation_intent_top1_accuracy:.6f}` / `{self.validation_intent_ece:.6f}` |
| Validation joint NLL / top-1 / ECE | `{self.validation_joint_nll:.6f}` / `{self.validation_joint_top1_accuracy:.6f}` / `{self.validation_joint_ece:.6f}` |
| Validation resource Brier / ECE | `{self.validation_resource_presence_brier:.6f}` / `{self.validation_resource_ece:.6f}` |
| In-scope candidate x declared-intent x resource range | **{str(self.in_scope_joint_candidate_intent_resource_range_calibrated).lower()}** |
| Declared-policy intent proxy calibrated | **{str(self.declared_policy_intent_proxy_calibrated).lower()}** |
| Opponent latent intent / full official belief calibrated | **false / false** |
| Counterfactual values / safe resolver / official control | **false / false / false** |

Official battles, selections, public action semantics, and transient resource
labels are genuine execution. The six intent policies are controlled research
proxies, and the model/composition remains abstract. A proxy pass is not proof
that latent human intent is identifiable. No resolver, exploitation overlay,
official strategy control, Kaggle submission, or strength claim is enabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6OfficialIntentEvidence":
        fields = {
            name: raw[name]
            for name in cls.__dataclass_fields__
            if name != "claims"
        }
        result = cls(
            **fields,
            claims=tuple(
                (str(key), str(value)) for key, value in raw["claims"].items()
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AE evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialIntentArtifacts:
    dataset: OfficialIntentTruthDataset
    intent_model: OfficialIntentRangeModel
    calibration: OfficialJointBeliefCalibration
    evidence: Phase6OfficialIntentEvidence


def _build_intent_truth_example(
    observation: Mapping[str, Any],
    trace: OfficialIntentPrefixTrace,
    *,
    visualize_data: Any,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    intent: OpponentIntent,
    particles_per_candidate: int,
    seed: int,
) -> OfficialIntentTruthExample:
    frames = json.loads(visualize_data())
    if not isinstance(frames, Sequence) or not frames or not isinstance(frames[-1], Mapping):
        raise RuntimeError("Phase 6AE visualizer returned no truth frame")
    truth_frame = frames[-1]
    visualizer_public_parity = (
        sanitize_official_public_state(truth_frame)
        == sanitize_official_public_state(observation)
    )
    truth_private = official_private_observation_payload(truth_frame)
    live_private = official_private_observation_payload(observation)
    visualizer_private_parity = all(
        truth_private[name] == live_private[name]
        for name in ("own_hand", "own_looking")
    )
    if not visualizer_public_parity or not visualizer_private_parity:
        raise RuntimeError("Phase 6AE visualizer/live root parity failed")
    completion = _official_truth_completion(
        observation,
        truth_frame,
        target=target,
        candidate=candidate,
        style=intent.value,
    )
    acting_hand = tuple(int(value) for value in live_private["own_hand"])
    truth_counts = _resource_counts(
        hand=acting_hand,
        prize=completion.your_prize,
        deck=completion.your_deck,
    )
    truth_zone_counts = (
        len(acting_hand),
        len(completion.your_prize),
        len(completion.your_deck),
    )
    candidate_public_features = native_observer_public_feature_items(
        observation,
        _uniform_candidate_range(tuple(item.archetype for item in candidates)),
        observer_player=OFFICIAL_INTENT_OBSERVER_PLAYER,
    )
    intent_public_features = official_intent_feature_items(
        candidate_public_features,
        trace,
    )
    public_state_fingerprint = _fingerprint(
        sanitize_official_public_state(observation)
    )
    public_history_fingerprint = _root_public_history(observation)
    truth_sample_fingerprint = _fingerprint(truth_frame)
    source_root_id = _fingerprint(
        {
            "split": trace.split,
            "candidate": candidate.archetype,
            "intent": intent.value,
            "collection_ordinal": trace.collection_ordinal,
            "public_state": public_state_fingerprint,
            "public_history": public_history_fingerprint,
            "truth_sample": truth_sample_fingerprint,
        }
    )
    summaries = _candidate_particle_summaries(
        observation,
        target=target,
        candidates=candidates,
        particles_per_candidate=particles_per_candidate,
        root_token=source_root_id,
        style=intent.value,
        seed=_seed(seed, "phase6ae-particles"),
    )
    example_id = _fingerprint(
        {
            "source_root": source_root_id,
            "candidate_features": candidate_public_features,
            "intent_features": intent_public_features,
            "candidate_label_supervision_only": candidate.archetype,
            "intent_label_supervision_only": intent.value,
            "truth_resources_supervision_only": truth_counts,
        }
    )
    return OfficialIntentTruthExample(
        example_id=example_id,
        source_root_id=source_root_id,
        split=trace.split,
        candidate_id_supervision_only=candidate.archetype,
        intent_id_supervision_only=intent.value,
        collection_ordinal=trace.collection_ordinal,
        prefix_trace=trace,
        public_state_fingerprint=public_state_fingerprint,
        public_history_fingerprint=public_history_fingerprint,
        candidate_public_features=candidate_public_features,
        intent_public_features=intent_public_features,
        candidate_public_abstraction_fingerprint=_fingerprint(
            candidate_public_features
        ),
        intent_public_abstraction_fingerprint=_fingerprint(intent_public_features),
        truth_sample_fingerprint=truth_sample_fingerprint,
        truth_resource_counts=truth_counts,
        truth_zone_counts=truth_zone_counts,
        candidate_particle_summaries=summaries,
        visualizer_public_parity=visualizer_public_parity,
        visualizer_acting_private_parity=visualizer_private_parity,
    )


def _capture_official_intent_root(
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    visualize_data: Any,
    *,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    intent: OpponentIntent,
    split: str,
    collection_ordinal: int,
    capture_attempt: int,
    target_reverse: bool,
    particles_per_candidate: int,
    maximum_live_selections: int,
    seed: int,
    opponent_policy_variant_id: str = "canonical",
) -> OfficialIntentTruthExample:
    observation, start = battle_start(list(target.card_ids), list(candidate.card_ids))
    if observation is None:
        raise RuntimeError(
            "official engine rejected Phase 6AE decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        policies = (
            PriorityBaseline(
                target.card_ids,
                style="tempo",
                reverse=target_reverse,
            ),
            StrategicIntentPriorityBaseline(
                candidate.card_ids,
                intent=intent,
                reverse=not target_reverse,
                variant_id=opponent_policy_variant_id,
            ),
        )
        observed_opponent_main_turns: list[int] = []
        steps: list[OfficialIntentDecisionStep] = []
        while (
            observation is not None
            and _safe_int(_state(observation).get("result"), -1) == -1
            and len(steps) < maximum_live_selections
        ):
            state = _state(observation)
            actor = _safe_int(state.get("yourIndex"), -1)
            turn = _safe_int(state.get("turn"), 0)
            select = observation.get("select")
            context = (
                _safe_int(select.get("context"), -1)
                if isinstance(select, Mapping)
                else -1
            )
            if actor == OFFICIAL_INTENT_OPPONENT_PLAYER and context == int(
                SelectContext.MAIN
            ):
                if turn not in observed_opponent_main_turns:
                    observed_opponent_main_turns.append(turn)
                if len(observed_opponent_main_turns) >= 2:
                    if observation.get("search_begin_input") is None:
                        raise RuntimeError("Phase 6AE live root lacks native search input")
                    public_state = _fingerprint(
                        sanitize_official_public_state(observation)
                    )
                    public_history = _root_public_history(observation)
                    trace = OfficialIntentPrefixTrace(
                        split=split,
                        candidate_id=candidate.archetype,
                        intent_id_supervision_only=intent.value,
                        collection_ordinal=collection_ordinal,
                        capture_attempt=capture_attempt,
                        captured_turn=turn,
                        target_policy_id=OFFICIAL_INTENT_TARGET_POLICY,
                        target_reverse=target_reverse,
                        opponent_reverse=not target_reverse,
                        root_public_state_fingerprint=public_state,
                        root_public_history_fingerprint=public_history,
                        trace_chain_fingerprint=_fingerprint(
                            [item.to_dict() for item in steps]
                        ),
                        steps=tuple(steps),
                    )
                    return _build_intent_truth_example(
                        observation,
                        trace,
                        visualize_data=visualize_data,
                        target=target,
                        candidate=candidate,
                        candidates=candidates,
                        intent=intent,
                        particles_per_candidate=particles_per_candidate,
                        seed=seed,
                    )
            if actor not in (0, 1):
                raise RuntimeError("Phase 6AE official actor is invalid")
            contract = SelectionContract.from_observation(observation)
            selection = contract.validate(policies[actor].choose(observation))
            selection_type = (
                _safe_int(select.get("type"), -1)
                if isinstance(select, Mapping)
                else -1
            )
            options = select.get("option", ()) if isinstance(select, Mapping) else ()
            selected_public_main_action_types = (
                tuple(
                    _safe_int(options[index].get("type"), -1) for index in selection
                )
                if actor == OFFICIAL_INTENT_OPPONENT_PLAYER
                and context == int(SelectContext.MAIN)
                else ()
            )
            public_before = _fingerprint(
                sanitize_official_public_state(observation)
            )
            history_before = _root_public_history(observation)
            contract_fingerprint = _fingerprint(
                {
                    "selection_type": selection_type,
                    "selection_context": context,
                    "minimum_count": contract.min_count,
                    "maximum_count": contract.max_count,
                    "option_count": contract.option_count,
                }
            )
            selected_fingerprint = _fingerprint(
                {
                    "contract_fingerprint": contract_fingerprint,
                    "selected_indices": selection,
                }
            )
            next_observation = battle_select(list(selection))
            if next_observation is None:
                raise RuntimeError("Phase 6AE battle ended before the requested root")
            public_after = _fingerprint(
                sanitize_official_public_state(next_observation)
            )
            history_after = _root_public_history(next_observation)
            steps.append(
                OfficialIntentDecisionStep(
                    step_index=len(steps),
                    actor=actor,
                    turn=turn,
                    selection_type=selection_type,
                    selection_context=context,
                    legal_option_count=contract.option_count,
                    minimum_selection_count=contract.min_count,
                    maximum_selection_count=contract.max_count,
                    selected_count=len(selection),
                    selected_public_main_action_types=(
                        selected_public_main_action_types
                    ),
                    public_state_before_fingerprint=public_before,
                    public_history_before_fingerprint=history_before,
                    selection_contract_fingerprint=contract_fingerprint,
                    selected_selection_fingerprint=selected_fingerprint,
                    public_state_after_fingerprint=public_after,
                    public_history_after_fingerprint=history_after,
                )
            )
            observation = next_observation
        raise RuntimeError(
            "Phase 6AE could not capture the opponent's second MAIN root within "
            f"{maximum_live_selections} selections"
        )
    finally:
        battle_finish()


def collect_official_intent_truth_dataset(
    *,
    source_dataset: OfficialRangeTruthDataset,
    source_model: OfficialPublicRangeModel,
    source_calibration: OfficialRangeCalibration,
    source_evidence: Phase6OfficialRangeEvidence,
    target: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    visualize_data: Any,
    roots_per_candidate_intent_split: int,
    particles_per_candidate: int,
    maximum_live_selections: int,
    maximum_capture_attempts: int,
    seed: int,
) -> OfficialIntentTruthDataset:
    if roots_per_candidate_intent_split <= 0 or particles_per_candidate <= 1:
        raise ValueError("Phase 6AE collection sizes are invalid")
    examples = []
    for split_index, split in enumerate(ALIGNED_VALUE_SPLITS):
        for candidate_index, candidate in enumerate(candidates):
            for intent_index, intent in enumerate(OPPONENT_INTENTS):
                for collection_ordinal in range(roots_per_candidate_intent_split):
                    capture_error: RuntimeError | None = None
                    for attempt in range(1, maximum_capture_attempts + 1):
                        target_reverse = bool(
                            (
                                split_index
                                + candidate_index
                                + intent_index
                                + collection_ordinal
                                + attempt
                                - 1
                            )
                            % 2
                        )
                        try:
                            examples.append(
                                _capture_official_intent_root(
                                    battle_start,
                                    battle_select,
                                    battle_finish,
                                    visualize_data,
                                    target=target,
                                    candidate=candidate,
                                    candidates=candidates,
                                    intent=intent,
                                    split=split,
                                    collection_ordinal=collection_ordinal,
                                    capture_attempt=attempt,
                                    target_reverse=target_reverse,
                                    particles_per_candidate=particles_per_candidate,
                                    maximum_live_selections=maximum_live_selections,
                                    seed=seed,
                                )
                            )
                            break
                        except RuntimeError as exc:
                            capture_error = exc
                    else:
                        raise RuntimeError(
                            "Phase 6AE exhausted capture attempts for "
                            f"{split}/{candidate.archetype}/{intent.value}/"
                            f"{collection_ordinal}"
                        ) from capture_error
    dataset = OfficialIntentTruthDataset(
        source_phase6ad_dataset_fingerprint=source_dataset.fingerprint,
        source_phase6ad_model_fingerprint=source_model.fingerprint,
        source_phase6ad_calibration_fingerprint=source_calibration.fingerprint,
        source_phase6ad_evidence_fingerprint=source_evidence.fingerprint,
        runtime_fingerprint=source_dataset.runtime_fingerprint,
        deck_fingerprint=source_dataset.deck_fingerprint,
        candidate_ids=source_dataset.candidate_ids,
        candidate_deck_fingerprints=source_dataset.candidate_deck_fingerprints,
        intent_ids=tuple(item.value for item in OPPONENT_INTENTS),
        roots_per_candidate_intent_split=roots_per_candidate_intent_split,
        particles_per_candidate=particles_per_candidate,
        examples=tuple(examples),
        collection_split_disjoint=True,
        target_policy_fixed_across_intents=True,
        opponent_only_action_semantics=True,
        observer_orientation_validated=True,
    )
    if _raw_private_key_count(dataset.to_dict()) != 0:
        raise RuntimeError("Phase 6AE serialized a forbidden raw private key")
    restored = OfficialIntentTruthDataset.from_dict(dataset.to_dict())
    if restored != dataset:
        raise RuntimeError("Phase 6AE dataset roundtrip changed values")
    return dataset


def run_phase6_official_intent_calibration(
    *,
    project_root: str | Path | None = None,
    source_artifact_directory: str | Path | None = None,
    roots_per_candidate_intent_split: int = (
        OFFICIAL_INTENT_ROOTS_PER_CANDIDATE_INTENT_SPLIT
    ),
    particles_per_candidate: int = OFFICIAL_INTENT_PARTICLES_PER_CANDIDATE,
    maximum_live_selections: int = 900,
    maximum_capture_attempts: int = 8,
    seed: int = PHASE6_OFFICIAL_INTENT_SEED,
) -> Phase6OfficialIntentArtifacts:
    if min(
        roots_per_candidate_intent_split,
        particles_per_candidate,
        maximum_live_selections,
        maximum_capture_attempts,
    ) <= 0:
        raise ValueError("Phase 6AE limits must be positive")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    source_directory = Path(
        source_artifact_directory
        or root / "artifacts" / "phase6_official_range_calibration"
    ).resolve()
    source_dataset = OfficialRangeTruthDataset.from_dict(
        json.loads(
            (source_directory / "official_range_truth_dataset.json").read_text(
                encoding="utf-8"
            )
        )
    )
    source_model = OfficialPublicRangeModel.from_dict(
        json.loads(
            (source_directory / "official_public_range_model.json").read_text(
                encoding="utf-8"
            )
        )
    )
    source_calibration = OfficialRangeCalibration.from_dict(
        json.loads(
            (source_directory / "official_range_calibration.json").read_text(
                encoding="utf-8"
            )
        )
    )
    source_evidence = Phase6OfficialRangeEvidence.from_dict(
        json.loads(
            (source_directory / "phase6ad_evidence.json").read_text(
                encoding="utf-8"
            )
        )
    )
    if not all(
        (
            source_calibration.in_scope_official_hidden_card_range_calibrated,
            source_evidence.in_scope_official_hidden_card_range_calibrated,
            source_model.source_dataset_fingerprint == source_dataset.fingerprint,
            source_calibration.dataset_fingerprint == source_dataset.fingerprint,
            source_calibration.model_fingerprint == source_model.fingerprint,
        )
    ):
        raise ValueError("Phase 6AE requires the admitted Phase 6AD source")
    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    if set(profile_by_id) != set(source_dataset.candidate_ids):
        raise ValueError("Phase 6AE candidate catalog/source mismatch")
    candidates = tuple(profile_by_id[name] for name in source_dataset.candidate_ids)
    if target.fingerprint != source_dataset.deck_fingerprint:
        raise ValueError("Phase 6AE target deck/source mismatch")
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    runtime_text = str(runtime)
    import sys

    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
        visualize_data,
    )

    dataset = collect_official_intent_truth_dataset(
        source_dataset=source_dataset,
        source_model=source_model,
        source_calibration=source_calibration,
        source_evidence=source_evidence,
        target=target,
        candidates=candidates,
        battle_start=battle_start,
        battle_select=battle_select,
        battle_finish=battle_finish,
        visualize_data=visualize_data,
        roots_per_candidate_intent_split=roots_per_candidate_intent_split,
        particles_per_candidate=particles_per_candidate,
        maximum_live_selections=maximum_live_selections,
        maximum_capture_attempts=maximum_capture_attempts,
        seed=seed,
    )
    intent_model = train_official_intent_range(dataset)
    restored_model = OfficialIntentRangeModel.from_dict(intent_model.to_dict())
    if restored_model != intent_model:
        raise RuntimeError("Phase 6AE intent model roundtrip changed values")
    calibration = evaluate_official_joint_belief(
        dataset,
        source_model,
        source_calibration,
        intent_model,
    )
    restored_calibration = OfficialJointBeliefCalibration.from_dict(
        calibration.to_dict()
    )
    if restored_calibration != calibration:
        raise RuntimeError("Phase 6AE calibration roundtrip changed values")
    evidence = Phase6OfficialIntentEvidence(
        benchmark_id="phase6ae-seat-decoupled-official-intent-v1",
        status=(
            "in_scope_joint_declared_intent_range_calibrated_latent_intent_fail_closed"
            if calibration.in_scope_joint_candidate_intent_resource_range_calibrated
            else "official_declared_intent_range_diagnostic_rejected_fail_closed"
        ),
        source_phase6ad_dataset_fingerprint=source_dataset.fingerprint,
        source_phase6ad_model_fingerprint=source_model.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        intent_model_fingerprint=intent_model.fingerprint,
        joint_provider_fingerprint=calibration.joint_provider_fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=len(dataset.examples),
        roots_per_split=dict(dataset.roots_per_split)["train"],
        candidate_strata=len(dataset.candidate_ids),
        intent_strata=len(dataset.intent_ids),
        live_selection_count=dataset.live_selection_count,
        opponent_main_action_count=dataset.opponent_main_action_count,
        particles_per_candidate=dataset.particles_per_candidate,
        total_public_consistent_particles=(
            len(dataset.examples)
            * len(dataset.candidate_ids)
            * dataset.particles_per_candidate
        ),
        selected_l2_penalty=intent_model.selected_l2_penalty,
        selected_temperature=intent_model.selected_temperature,
        selected_uniform_mix=intent_model.selected_uniform_mix,
        validation_candidate_nll=calibration.validation.candidate_nll,
        validation_candidate_ece=calibration.validation.candidate_ece,
        validation_intent_nll=calibration.validation.intent_nll,
        validation_intent_top1_accuracy=(
            calibration.validation.intent_top1_accuracy
        ),
        validation_intent_ece=calibration.validation.intent_ece,
        validation_joint_nll=calibration.validation.joint_nll,
        validation_joint_top1_accuracy=calibration.validation.joint_top1_accuracy,
        validation_joint_ece=calibration.validation.joint_ece,
        validation_resource_presence_brier=(
            calibration.validation.resource_presence_brier
        ),
        validation_resource_ece=calibration.validation.resource_ece,
        in_scope_joint_candidate_intent_resource_range_calibrated=(
            calibration.in_scope_joint_candidate_intent_resource_range_calibrated
        ),
        declared_policy_intent_proxy_calibrated=(
            calibration.declared_policy_intent_proxy_calibrated
        ),
        opponent_latent_intent_calibrated=False,
        full_official_public_belief_calibrated=False,
        counterfactual_value_compatible=False,
        safe_resolver_enabled=False,
        opponent_exploitation_enabled=False,
        official_strategy_control_enabled=False,
        claims=(
            ("genuine", "fresh official roots, selections, public actions, and transient resource labels"),
            ("seat_control", "player 0 fixed; only player 1 declared intent policy varies"),
            ("model_input", "player-0 public root plus player-1 coarse public MAIN actions only"),
            ("proxy_scope", "six controlled intent policies crossed with six exact candidate decks"),
            ("abstract", "intent policies, linear model, candidate-intent independence, particles, and calibration"),
            ("latent_intent", "not_established"),
            ("downstream", "values, safety, exploitation, and control disabled"),
            ("kaggle_submission", "not_attempted"),
            ("pokemon_strength", "not_claimed"),
            ("phase6_complete", "false"),
        ),
    )
    restored_evidence = Phase6OfficialIntentEvidence.from_dict(evidence.to_dict())
    if restored_evidence != evidence:
        raise RuntimeError("Phase 6AE evidence roundtrip changed values")
    return Phase6OfficialIntentArtifacts(dataset, intent_model, calibration, evidence)


def main() -> None:
    print(run_phase6_official_intent_calibration().evidence.to_json())


if __name__ == "__main__":
    main()
