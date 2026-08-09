"""Phase 6AB redacted official-prefix behavior-reach ledger.

The live competition engine accepts player selections and advances all internal
chance events, but its public adapter does not expose the probability of the
shuffle, prize, draw, or coin outcomes that occurred between selections.  This
module records every declared deterministic player choice from ``battle_start``
to a later player-1 MAIN root.  Selection identities, public states, and public
histories are persisted only as digests.

Consequently the player-policy factors in each trace are genuinely known and
the selections were genuinely executed by the official engine.  Chance reach,
full sampling reach, counterfactual importance weights, and alignment with the
separately collected Phase 6Z value roots remain unavailable by construction.
This is an auditable partial ledger, not counterfactual-value evidence and not
authorization for safe resolving or official strategy control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    SelectContext,
    SelectionContract,
    _safe_int,
    _state,
    resolve_official_deck,
)
from .official_bridge import sanitize_official_public_state
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    PriorityBaseline,
    rocket_compatibility_profile,
)
from .official_search_game import _root_public_history
from .phase6_aligned_counterfactual_values import (
    ALIGNED_VALUE_SPLITS,
    PublicRootAlignedValueDataset,
)
from .phase6_bridge import official_runtime_fingerprint
from .phase6_native_external_sampling import _raw_private_key_count


PHASE6_OFFICIAL_PREFIX_REACH_SCHEMA_VERSION = 1
OFFICIAL_PREFIX_TARGET_PLAYER = 1
OFFICIAL_PREFIX_CAPTURE_CONFIG: Mapping[str, tuple[str, int, bool]] = {
    "train": ("tempo", 2, False),
    "calibration": ("attack_rush", 2, True),
    "validation": ("control", 2, False),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _unit_probability(value: float, name: str) -> None:
    if not math.isclose(float(value), 1.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be one for the declared deterministic policy")


@dataclass(frozen=True, slots=True)
class OfficialPrefixDecisionStep:
    """One redacted player selection genuinely executed by ``battle_select``."""

    step_index: int
    actor: int
    turn: int
    selection_type: int
    selection_context: int
    public_state_before_fingerprint: str
    public_history_before_fingerprint: str
    selection_contract_fingerprint: str
    selected_selection_fingerprint: str
    legal_option_count: int
    minimum_selection_count: int
    maximum_selection_count: int
    selected_count: int
    public_state_after_fingerprint: str
    public_history_after_fingerprint: str
    declared_behavior_probability: float = 1.0
    sampling_probability: float = 1.0
    official_battle_selection_executed: bool = True
    official_chance_probability_available: bool = False
    raw_selection_indices_persisted: int = 0
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.actor not in (0, 1) or self.turn < 0:
            raise ValueError("official prefix decision identity is invalid")
        for name in (
            "public_state_before_fingerprint",
            "public_history_before_fingerprint",
            "selection_contract_fingerprint",
            "selected_selection_fingerprint",
            "public_state_after_fingerprint",
            "public_history_after_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not (
            0
            <= self.minimum_selection_count
            <= self.selected_count
            <= self.maximum_selection_count
            <= self.legal_option_count
        ):
            raise ValueError("official prefix selection counts are invalid")
        _unit_probability(
            self.declared_behavior_probability,
            "declared_behavior_probability",
        )
        _unit_probability(self.sampling_probability, "sampling_probability")
        if any(
            (
                not self.official_battle_selection_executed,
                self.official_chance_probability_available,
                self.raw_selection_indices_persisted != 0,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official prefix step promoted an unsupported claim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "actor": self.actor,
            "turn": self.turn,
            "selection_type": self.selection_type,
            "selection_context": self.selection_context,
            "public_state_before_fingerprint": (
                self.public_state_before_fingerprint
            ),
            "public_history_before_fingerprint": (
                self.public_history_before_fingerprint
            ),
            "selection_contract_fingerprint": self.selection_contract_fingerprint,
            "selected_selection_fingerprint": self.selected_selection_fingerprint,
            "legal_option_count": self.legal_option_count,
            "minimum_selection_count": self.minimum_selection_count,
            "maximum_selection_count": self.maximum_selection_count,
            "selected_count": self.selected_count,
            "public_state_after_fingerprint": self.public_state_after_fingerprint,
            "public_history_after_fingerprint": self.public_history_after_fingerprint,
            "reach": {
                "declared_behavior_probability": self.declared_behavior_probability,
                "sampling_probability": self.sampling_probability,
                "official_chance_probability": None,
            },
            "claims": {
                "official_battle_selection_executed": (
                    self.official_battle_selection_executed
                ),
                "official_chance_probability_available": (
                    self.official_chance_probability_available
                ),
                "raw_selection_indices_persisted": (
                    self.raw_selection_indices_persisted
                ),
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPrefixDecisionStep":
        reach = raw["reach"]
        claims = raw["claims"]
        if reach.get("official_chance_probability") is not None:
            raise ValueError("official prefix step invented a chance probability")
        return cls(
            step_index=int(raw["step_index"]),
            actor=int(raw["actor"]),
            turn=int(raw["turn"]),
            selection_type=int(raw["selection_type"]),
            selection_context=int(raw["selection_context"]),
            public_state_before_fingerprint=str(
                raw["public_state_before_fingerprint"]
            ),
            public_history_before_fingerprint=str(
                raw["public_history_before_fingerprint"]
            ),
            selection_contract_fingerprint=str(
                raw["selection_contract_fingerprint"]
            ),
            selected_selection_fingerprint=str(
                raw["selected_selection_fingerprint"]
            ),
            legal_option_count=int(raw["legal_option_count"]),
            minimum_selection_count=int(raw["minimum_selection_count"]),
            maximum_selection_count=int(raw["maximum_selection_count"]),
            selected_count=int(raw["selected_count"]),
            public_state_after_fingerprint=str(raw["public_state_after_fingerprint"]),
            public_history_after_fingerprint=str(
                raw["public_history_after_fingerprint"]
            ),
            declared_behavior_probability=float(
                reach["declared_behavior_probability"]
            ),
            sampling_probability=float(reach["sampling_probability"]),
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
class OfficialPrefixReachTrace:
    """A complete player-action prefix with intentionally incomplete chance reach."""

    split: str
    candidate_id: str
    baseline_style: str
    baseline_reverse_for_player_0: bool
    requested_player_1_main_ordinal: int
    capture_attempt: int
    captured_turn: int
    target_player: int
    root_public_state_fingerprint: str
    root_public_history_fingerprint: str
    trace_chain_fingerprint: str
    steps: tuple[OfficialPrefixDecisionStep, ...]
    declared_player_0_behavior_reach: float = 1.0
    declared_player_1_behavior_reach: float = 1.0
    declared_player_behavior_sampling_reach: float = 1.0
    official_chance_reach: float | None = None
    full_sampling_reach: float | None = None
    counterfactual_importance_weight: float | None = None
    official_battle_root_executed: bool = True
    official_action_prefix_complete: bool = True
    deterministic_behavior_policy_declared: bool = True
    behavior_and_sampling_policies_match: bool = True
    official_chance_probability_available: bool = False
    upstream_official_reach_complete: bool = False
    aligned_with_phase6z_value_root: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if self.split not in ALIGNED_VALUE_SPLITS or not self.candidate_id:
            raise ValueError("official prefix trace split/candidate is invalid")
        expected_style, expected_ordinal, _ = OFFICIAL_PREFIX_CAPTURE_CONFIG[self.split]
        if (
            self.baseline_style != expected_style
            or self.requested_player_1_main_ordinal != expected_ordinal
        ):
            raise ValueError("official prefix trace capture configuration drifted")
        if min(self.capture_attempt, self.captured_turn) <= 0:
            raise ValueError("official prefix trace capture accounting is invalid")
        if self.target_player != OFFICIAL_PREFIX_TARGET_PLAYER:
            raise ValueError("official prefix trace target player drifted")
        for name in (
            "root_public_state_fingerprint",
            "root_public_history_fingerprint",
            "trace_chain_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.steps:
            raise ValueError("official prefix trace cannot be empty")
        if tuple(item.step_index for item in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("official prefix step indices must be contiguous")
        for left, right in zip(self.steps, self.steps[1:]):
            if (
                left.public_state_after_fingerprint
                != right.public_state_before_fingerprint
                or left.public_history_after_fingerprint
                != right.public_history_before_fingerprint
            ):
                raise ValueError("official prefix public-state/history chain is broken")
        final = self.steps[-1]
        if (
            final.public_state_after_fingerprint
            != self.root_public_state_fingerprint
            or final.public_history_after_fingerprint
            != self.root_public_history_fingerprint
        ):
            raise ValueError("official prefix does not terminate at its declared root")
        expected_chain = _fingerprint([item.to_dict() for item in self.steps])
        if self.trace_chain_fingerprint != expected_chain:
            raise ValueError("official prefix trace-chain fingerprint mismatch")
        for value, name in (
            (self.declared_player_0_behavior_reach, "player_0_behavior_reach"),
            (self.declared_player_1_behavior_reach, "player_1_behavior_reach"),
            (
                self.declared_player_behavior_sampling_reach,
                "player_behavior_sampling_reach",
            ),
        ):
            _unit_probability(value, name)
        if any(
            value is not None
            for value in (
                self.official_chance_reach,
                self.full_sampling_reach,
                self.counterfactual_importance_weight,
            )
        ):
            raise ValueError("official prefix trace invented unavailable reach terms")
        if not all(
            (
                self.official_battle_root_executed,
                self.official_action_prefix_complete,
                self.deterministic_behavior_policy_declared,
                self.behavior_and_sampling_policies_match,
            )
        ):
            raise ValueError("official prefix trace lost its genuine behavior provenance")
        if any(
            (
                self.official_chance_probability_available,
                self.upstream_official_reach_complete,
                self.aligned_with_phase6z_value_root,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official prefix trace promoted an unsupported capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def player_0_decisions(self) -> int:
        return sum(item.actor == 0 for item in self.steps)

    @property
    def player_1_decisions(self) -> int:
        return sum(item.actor == 1 for item in self.steps)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_PREFIX_REACH_SCHEMA_VERSION,
            "split": self.split,
            "candidate_id": self.candidate_id,
            "capture": {
                "baseline_style": self.baseline_style,
                "baseline_reverse_for_player_0": (
                    self.baseline_reverse_for_player_0
                ),
                "requested_player_1_main_ordinal": (
                    self.requested_player_1_main_ordinal
                ),
                "capture_attempt": self.capture_attempt,
                "captured_turn": self.captured_turn,
                "target_player": self.target_player,
            },
            "root_public_state_fingerprint": self.root_public_state_fingerprint,
            "root_public_history_fingerprint": self.root_public_history_fingerprint,
            "trace_chain_fingerprint": self.trace_chain_fingerprint,
            "steps": [item.to_dict() for item in self.steps],
            "accounting": {
                "player_0_decisions": self.player_0_decisions,
                "player_1_decisions": self.player_1_decisions,
            },
            "reach": {
                "declared_player_0_behavior_reach": (
                    self.declared_player_0_behavior_reach
                ),
                "declared_player_1_behavior_reach": (
                    self.declared_player_1_behavior_reach
                ),
                "declared_player_behavior_sampling_reach": (
                    self.declared_player_behavior_sampling_reach
                ),
                "official_chance_reach": self.official_chance_reach,
                "full_sampling_reach": self.full_sampling_reach,
                "counterfactual_importance_weight": (
                    self.counterfactual_importance_weight
                ),
            },
            "claims": {
                "official_battle_root_executed": self.official_battle_root_executed,
                "official_action_prefix_complete": self.official_action_prefix_complete,
                "deterministic_behavior_policy_declared": (
                    self.deterministic_behavior_policy_declared
                ),
                "behavior_and_sampling_policies_match": (
                    self.behavior_and_sampling_policies_match
                ),
                "official_chance_probability_available": (
                    self.official_chance_probability_available
                ),
                "upstream_official_reach_complete": (
                    self.upstream_official_reach_complete
                ),
                "aligned_with_phase6z_value_root": (
                    self.aligned_with_phase6z_value_root
                ),
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPrefixReachTrace":
        capture = raw["capture"]
        reach = raw["reach"]
        claims = raw["claims"]
        result = cls(
            split=str(raw["split"]),
            candidate_id=str(raw["candidate_id"]),
            baseline_style=str(capture["baseline_style"]),
            baseline_reverse_for_player_0=bool(
                capture["baseline_reverse_for_player_0"]
            ),
            requested_player_1_main_ordinal=int(
                capture["requested_player_1_main_ordinal"]
            ),
            capture_attempt=int(capture["capture_attempt"]),
            captured_turn=int(capture["captured_turn"]),
            target_player=int(capture["target_player"]),
            root_public_state_fingerprint=str(
                raw["root_public_state_fingerprint"]
            ),
            root_public_history_fingerprint=str(
                raw["root_public_history_fingerprint"]
            ),
            trace_chain_fingerprint=str(raw["trace_chain_fingerprint"]),
            steps=tuple(
                OfficialPrefixDecisionStep.from_dict(item) for item in raw["steps"]
            ),
            declared_player_0_behavior_reach=float(
                reach["declared_player_0_behavior_reach"]
            ),
            declared_player_1_behavior_reach=float(
                reach["declared_player_1_behavior_reach"]
            ),
            declared_player_behavior_sampling_reach=float(
                reach["declared_player_behavior_sampling_reach"]
            ),
            official_chance_reach=(
                None
                if reach["official_chance_reach"] is None
                else float(reach["official_chance_reach"])
            ),
            full_sampling_reach=(
                None
                if reach["full_sampling_reach"] is None
                else float(reach["full_sampling_reach"])
            ),
            counterfactual_importance_weight=(
                None
                if reach["counterfactual_importance_weight"] is None
                else float(reach["counterfactual_importance_weight"])
            ),
            official_battle_root_executed=bool(
                claims["official_battle_root_executed"]
            ),
            official_action_prefix_complete=bool(
                claims["official_action_prefix_complete"]
            ),
            deterministic_behavior_policy_declared=bool(
                claims["deterministic_behavior_policy_declared"]
            ),
            behavior_and_sampling_policies_match=bool(
                claims["behavior_and_sampling_policies_match"]
            ),
            official_chance_probability_available=bool(
                claims["official_chance_probability_available"]
            ),
            upstream_official_reach_complete=bool(
                claims["upstream_official_reach_complete"]
            ),
            aligned_with_phase6z_value_root=bool(
                claims["aligned_with_phase6z_value_root"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        accounting = raw["accounting"]
        if (
            int(accounting["player_0_decisions"]) != result.player_0_decisions
            or int(accounting["player_1_decisions"]) != result.player_1_decisions
        ):
            raise ValueError("official prefix trace accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official prefix trace fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OfficialPrefixReachDataset:
    source_phase6z_dataset_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    traces: tuple[OfficialPrefixReachTrace, ...]
    behavior_action_ledger_complete: bool = True
    declared_deterministic_behavior_reach_available: bool = True
    official_chance_probability_coverage: float = 0.0
    official_chance_reach_available: bool = False
    full_upstream_reach_available: bool = False
    official_counterfactual_importance_weights_available: bool = False
    phase6z_value_root_alignment_available: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_phase6z_dataset_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("official prefix candidate support is invalid")
        expected = {
            (split, candidate)
            for split in ALIGNED_VALUE_SPLITS
            for candidate in self.candidate_ids
        }
        observed = {(item.split, item.candidate_id) for item in self.traces}
        if observed != expected or len(self.traces) != len(expected):
            raise ValueError("official prefix trace support is incomplete")
        if len({item.fingerprint for item in self.traces}) != len(self.traces):
            raise ValueError("official prefix traces are duplicated")
        if any(item.target_player != OFFICIAL_PREFIX_TARGET_PLAYER for item in self.traces):
            raise ValueError("official prefix dataset target player drifted")
        if not self.behavior_action_ledger_complete:
            raise ValueError("official prefix behavior ledger is incomplete")
        if not self.declared_deterministic_behavior_reach_available:
            raise ValueError("official prefix deterministic reach was lost")
        if not math.isclose(self.official_chance_probability_coverage, 0.0, abs_tol=1e-12):
            raise ValueError("official prefix dataset invented chance coverage")
        if any(
            (
                self.official_chance_reach_available,
                self.full_upstream_reach_available,
                self.official_counterfactual_importance_weights_available,
                self.phase6z_value_root_alignment_available,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("official prefix dataset promoted an unsupported capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def official_selection_execution_count(self) -> int:
        return sum(len(item.steps) for item in self.traces)

    @property
    def player_0_decisions(self) -> int:
        return sum(item.player_0_decisions for item in self.traces)

    @property
    def player_1_decisions(self) -> int:
        return sum(item.player_1_decisions for item in self.traces)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_PREFIX_REACH_SCHEMA_VERSION,
            "source_phase6z_dataset_fingerprint": (
                self.source_phase6z_dataset_fingerprint
            ),
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "traces": [item.to_dict() for item in self.traces],
            "accounting": {
                "trace_count": len(self.traces),
                "official_selection_execution_count": (
                    self.official_selection_execution_count
                ),
                "player_0_decisions": self.player_0_decisions,
                "player_1_decisions": self.player_1_decisions,
            },
            "gates": {
                "behavior_action_ledger_complete": (
                    self.behavior_action_ledger_complete
                ),
                "declared_deterministic_behavior_reach_available": (
                    self.declared_deterministic_behavior_reach_available
                ),
                "official_chance_probability_coverage": (
                    self.official_chance_probability_coverage
                ),
                "official_chance_reach_available": (
                    self.official_chance_reach_available
                ),
                "full_upstream_reach_available": self.full_upstream_reach_available,
                "official_counterfactual_importance_weights_available": (
                    self.official_counterfactual_importance_weights_available
                ),
                "phase6z_value_root_alignment_available": (
                    self.phase6z_value_root_alignment_available
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
            },
            "claims": {
                "official_engine_selection_execution": True,
                "official_engine_internal_chance_execution": True,
                "declared_behavior_policy_source": "PriorityBaseline",
                "official_behavior_strategy": False,
                "official_chance_catalog_or_likelihood_exposed": False,
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OfficialPrefixReachDataset":
        gates = raw["gates"]
        claims = raw["claims"]
        if claims != {
            "official_engine_selection_execution": True,
            "official_engine_internal_chance_execution": True,
            "declared_behavior_policy_source": "PriorityBaseline",
            "official_behavior_strategy": False,
            "official_chance_catalog_or_likelihood_exposed": False,
            "raw_private_card_ids_persisted": claims["raw_private_card_ids_persisted"],
        }:
            raise ValueError("official prefix dataset claim schema changed")
        result = cls(
            source_phase6z_dataset_fingerprint=str(
                raw["source_phase6z_dataset_fingerprint"]
            ),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            traces=tuple(
                OfficialPrefixReachTrace.from_dict(item) for item in raw["traces"]
            ),
            behavior_action_ledger_complete=bool(
                gates["behavior_action_ledger_complete"]
            ),
            declared_deterministic_behavior_reach_available=bool(
                gates["declared_deterministic_behavior_reach_available"]
            ),
            official_chance_probability_coverage=float(
                gates["official_chance_probability_coverage"]
            ),
            official_chance_reach_available=bool(
                gates["official_chance_reach_available"]
            ),
            full_upstream_reach_available=bool(
                gates["full_upstream_reach_available"]
            ),
            official_counterfactual_importance_weights_available=bool(
                gates["official_counterfactual_importance_weights_available"]
            ),
            phase6z_value_root_alignment_available=bool(
                gates["phase6z_value_root_alignment_available"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        accounting = raw["accounting"]
        expected_accounting = {
            "trace_count": len(result.traces),
            "official_selection_execution_count": (
                result.official_selection_execution_count
            ),
            "player_0_decisions": result.player_0_decisions,
            "player_1_decisions": result.player_1_decisions,
        }
        if dict(accounting) != expected_accounting:
            raise ValueError("official prefix dataset accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("official prefix dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialPrefixReachEvidence:
    benchmark_id: str
    status: str
    source_phase6z_dataset_fingerprint: str
    dataset_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    trace_count: int
    candidate_strata: int
    official_selection_execution_count: int
    player_0_decisions: int
    player_1_decisions: int
    minimum_steps_per_trace: int
    maximum_steps_per_trace: int
    behavior_action_ledger_gate_passed: bool
    deterministic_behavior_reach_gate_passed: bool
    official_chance_probability_coverage: float
    official_chance_reach_available: bool
    full_upstream_reach_available: bool
    official_counterfactual_importance_weights_available: bool
    phase6z_value_root_alignment_available: bool
    safe_resolver_enabled: bool
    opponent_exploitation_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6AB evidence identity is missing")
        for name in (
            "source_phase6z_dataset_fingerprint",
            "dataset_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if min(
            self.trace_count,
            self.candidate_strata,
            self.official_selection_execution_count,
            self.player_0_decisions,
            self.player_1_decisions,
            self.minimum_steps_per_trace,
            self.maximum_steps_per_trace,
        ) <= 0:
            raise ValueError("Phase 6AB evidence accounting is invalid")
        if self.minimum_steps_per_trace > self.maximum_steps_per_trace:
            raise ValueError("Phase 6AB trace range is invalid")
        if not all(
            (
                self.behavior_action_ledger_gate_passed,
                self.deterministic_behavior_reach_gate_passed,
            )
        ):
            raise ValueError("Phase 6AB genuine behavior gates unexpectedly failed")
        if not math.isclose(self.official_chance_probability_coverage, 0.0, abs_tol=1e-12):
            raise ValueError("Phase 6AB evidence invented chance coverage")
        if any(
            (
                self.official_chance_reach_available,
                self.full_upstream_reach_available,
                self.official_counterfactual_importance_weights_available,
                self.phase6z_value_root_alignment_available,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AB evidence enabled downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OFFICIAL_PREFIX_REACH_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AB official-prefix reach evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Later official roots | {self.trace_count} |
| Candidate strata | {self.candidate_strata} |
| Genuine `battle_select` executions | {self.official_selection_execution_count} |
| Player 0 / player 1 decisions | {self.player_0_decisions} / {self.player_1_decisions} |
| Steps per trace, min / max | {self.minimum_steps_per_trace} / {self.maximum_steps_per_trace} |
| Complete behavior-action ledger | **true** |
| Declared deterministic player reach | **true** |
| Official chance-probability coverage | `{self.official_chance_probability_coverage:.1f}` |
| Official chance reach available | **false** |
| Full upstream / counterfactual reach available | **false** / **false** |
| Aligned to Phase 6Z value roots | **false** |
| Safe resolver / exploitation / official control | **false** / **false** / **false** |

Every stored player choice was executed by the genuine official engine under a
declared deterministic `PriorityBaseline`, so its behavior and sampling factor
is exactly one. The engine also executed its internal automatic chance events,
but the live adapter exposes neither their outcome catalog nor likelihoods.
Chance reach, total sampling reach, and counterfactual importance weights stay
`null`; this independently recollected prefix cannot be joined to Phase 6Z
value roots. No resolver, exploitation overlay, or official control is enabled.
"""

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
    ) -> "Phase6OfficialPrefixReachEvidence":
        fields = {name: raw[name] for name in cls.__dataclass_fields__ if name != "claims"}
        result = cls(
            **fields,
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AB evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OfficialPrefixReachArtifacts:
    dataset: OfficialPrefixReachDataset
    evidence: Phase6OfficialPrefixReachEvidence


def _capture_official_prefix(
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    *,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    split: str,
    style: str,
    ordinal: int,
    reverse: bool,
    capture_attempt: int,
    maximum_live_selections: int,
    root_callback: Any | None = None,
) -> OfficialPrefixReachTrace:
    observation, start = battle_start(list(target.card_ids), list(candidate.card_ids))
    if observation is None:
        raise RuntimeError(
            "official engine rejected Phase 6AB decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        policies = (
            PriorityBaseline(target.card_ids, style=style, reverse=reverse),
            PriorityBaseline(candidate.card_ids, style=style, reverse=not reverse),
        )
        observed_player_1_main_turns: list[int] = []
        steps: list[OfficialPrefixDecisionStep] = []
        while (
            observation is not None
            and _safe_int(_state(observation).get("result"), -1) == -1
            and len(steps) < maximum_live_selections
        ):
            state = _state(observation)
            actor = _safe_int(state.get("yourIndex"), 0)
            turn = _safe_int(state.get("turn"), 0)
            select = observation.get("select")
            context = (
                _safe_int(select.get("context"), -1)
                if isinstance(select, Mapping)
                else -1
            )
            if actor == OFFICIAL_PREFIX_TARGET_PLAYER and context == int(SelectContext.MAIN):
                if turn not in observed_player_1_main_turns:
                    observed_player_1_main_turns.append(turn)
                if len(observed_player_1_main_turns) >= ordinal:
                    if observation.get("search_begin_input") is None:
                        raise RuntimeError(
                            "Phase 6AB live root lacks exact native search input"
                        )
                    public_state = _fingerprint(
                        sanitize_official_public_state(observation)
                    )
                    public_history = _root_public_history(observation)
                    trace = OfficialPrefixReachTrace(
                        split=split,
                        candidate_id=candidate.archetype,
                        baseline_style=style,
                        baseline_reverse_for_player_0=reverse,
                        requested_player_1_main_ordinal=ordinal,
                        capture_attempt=capture_attempt,
                        captured_turn=turn,
                        target_player=OFFICIAL_PREFIX_TARGET_PLAYER,
                        root_public_state_fingerprint=public_state,
                        root_public_history_fingerprint=public_history,
                        trace_chain_fingerprint=_fingerprint(
                            [item.to_dict() for item in steps]
                        ),
                        steps=tuple(steps),
                    )
                    if root_callback is not None:
                        root_callback(observation, trace)
                    return trace
            contract = SelectionContract.from_observation(observation)
            selection = contract.validate(policies[actor].choose(observation))
            selection_type = (
                _safe_int(select.get("type"), -1)
                if isinstance(select, Mapping)
                else -1
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
                raise RuntimeError("Phase 6AB battle ended before the requested root")
            public_after = _fingerprint(
                sanitize_official_public_state(next_observation)
            )
            history_after = _root_public_history(next_observation)
            steps.append(
                OfficialPrefixDecisionStep(
                    step_index=len(steps),
                    actor=actor,
                    turn=turn,
                    selection_type=selection_type,
                    selection_context=context,
                    public_state_before_fingerprint=public_before,
                    public_history_before_fingerprint=history_before,
                    selection_contract_fingerprint=contract_fingerprint,
                    selected_selection_fingerprint=selected_fingerprint,
                    legal_option_count=contract.option_count,
                    minimum_selection_count=contract.min_count,
                    maximum_selection_count=contract.max_count,
                    selected_count=len(selection),
                    public_state_after_fingerprint=public_after,
                    public_history_after_fingerprint=history_after,
                )
            )
            observation = next_observation
        raise RuntimeError(
            f"Phase 6AB could not capture player-1 MAIN root ordinal {ordinal} "
            f"within {maximum_live_selections} selections"
        )
    finally:
        battle_finish()


def run_phase6_official_prefix_reach(
    *,
    project_root: str | Path | None = None,
    source_phase6z_dataset_path: str | Path | None = None,
    maximum_live_selections: int = 900,
    maximum_capture_attempts: int = 8,
) -> Phase6OfficialPrefixReachArtifacts:
    if min(maximum_live_selections, maximum_capture_attempts) <= 0:
        raise ValueError("Phase 6AB capture limits must be positive")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    source_path = Path(
        source_phase6z_dataset_path
        or root
        / "artifacts"
        / "phase6_aligned_counterfactual_values"
        / "public_root_aligned_value_dataset.json"
    ).resolve()
    source = PublicRootAlignedValueDataset.from_dict(
        json.loads(source_path.read_text(encoding="utf-8"))
    )
    runtime_fingerprint = official_runtime_fingerprint(runtime)
    if runtime_fingerprint != source.runtime_fingerprint:
        raise ValueError("Phase 6AB official runtime/Phase 6Z source mismatch")
    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    if target.fingerprint != source.deck_fingerprint:
        raise ValueError("Phase 6AB target deck/Phase 6Z source mismatch")
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    if set(profile_by_id) != set(source.candidate_ids):
        raise ValueError("Phase 6AB candidate catalog/Phase 6Z source mismatch")
    candidates = tuple(profile_by_id[name] for name in source.candidate_ids)

    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
    )

    traces: list[OfficialPrefixReachTrace] = []
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
                try:
                    trace = _capture_official_prefix(
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
                    )
                    traces.append(trace)
                    break
                except RuntimeError as exc:
                    capture_error = exc
            else:
                raise RuntimeError(
                    "Phase 6AB exhausted official-prefix capture attempts for "
                    f"{split}/{candidate.archetype}"
                ) from capture_error

    dataset = OfficialPrefixReachDataset(
        source_phase6z_dataset_fingerprint=source.fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        deck_fingerprint=target.fingerprint,
        candidate_ids=source.candidate_ids,
        traces=tuple(traces),
    )
    if _raw_private_key_count(dataset.to_dict()) != 0:
        raise RuntimeError("Phase 6AB serialized a forbidden raw private key")
    restored = OfficialPrefixReachDataset.from_dict(dataset.to_dict())
    if restored != dataset:
        raise RuntimeError("Phase 6AB dataset roundtrip changed values")
    step_counts = tuple(len(item.steps) for item in traces)
    evidence = Phase6OfficialPrefixReachEvidence(
        benchmark_id="phase6ab-official-prefix-reach-ledger",
        status="diagnostic_complete_fail_closed",
        source_phase6z_dataset_fingerprint=source.fingerprint,
        dataset_fingerprint=dataset.fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        deck_fingerprint=target.fingerprint,
        trace_count=len(traces),
        candidate_strata=len(source.candidate_ids),
        official_selection_execution_count=(
            dataset.official_selection_execution_count
        ),
        player_0_decisions=dataset.player_0_decisions,
        player_1_decisions=dataset.player_1_decisions,
        minimum_steps_per_trace=min(step_counts),
        maximum_steps_per_trace=max(step_counts),
        behavior_action_ledger_gate_passed=True,
        deterministic_behavior_reach_gate_passed=True,
        official_chance_probability_coverage=0.0,
        official_chance_reach_available=False,
        full_upstream_reach_available=False,
        official_counterfactual_importance_weights_available=False,
        phase6z_value_root_alignment_available=False,
        safe_resolver_enabled=False,
        opponent_exploitation_enabled=False,
        official_strategy_control_enabled=False,
        claims=(
            (
                "genuine",
                "battle_start/battle_select executions, legal selection contracts, "
                "redacted public state/history chains, deterministic player-policy reach",
            ),
            (
                "unavailable",
                "official shuffle/draw/prize/coin outcome likelihoods, full sampling "
                "reach, counterfactual weights, and Phase 6Z value-root joins",
            ),
            (
                "disabled",
                "safe resolving, opponent exploitation, and official strategy control",
            ),
        ),
    )
    restored_evidence = Phase6OfficialPrefixReachEvidence.from_dict(evidence.to_dict())
    if restored_evidence != evidence:
        raise RuntimeError("Phase 6AB evidence roundtrip changed values")
    return Phase6OfficialPrefixReachArtifacts(dataset=dataset, evidence=evidence)


def main() -> None:
    artifacts = run_phase6_official_prefix_reach()
    print(artifacts.evidence.to_json())
