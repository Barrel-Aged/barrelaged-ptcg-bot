"""Phase 6Z public-root-aligned native conditional action values.

This experiment fixes the principal data-shape defect exposed by Phase 6Y:
several hidden allocations are now evaluated behind the *same* later public
root, and complete public roots are kept disjoint across train, calibration,
and validation collections.  Each terminal continuation below executes through
the licensed official native search engine.

The resulting labels are nevertheless conditional rollout values, not safe
counterfactual values.  The hidden-completion prior is supplied by this research
adapter, the acting player's observed hand is fixed across allocation variants,
upstream official reach is absent, and the five turn plans remain an abstract
macro action space.  Those limitations are machine-checked and serialized.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping, Sequence

import torch

from .competition import (
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    SelectContext,
    SelectionContract,
    _safe_int,
    _state,
    resolve_official_deck,
)
from .game_theory import ActionDistribution
from .native_blueprint_features import (
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeBlueprintFeatureBridge,
)
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
from .official_bridge import sanitize_official_public_state
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    PriorityBaseline,
    rocket_compatibility_profile,
)
from .official_search_game import (
    HiddenZoneCompletion,
    OfficialNativeSearchGame,
    sample_hidden_zone_completion,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _runtime_basics
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence
from .phase6_opponent_deviation_values import (
    OPPONENT_DEVIATION_ACTION_IDS,
    OpponentPlanActionValue,
    _rollout_average_blueprint,
)
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_ALIGNED_VALUE_SCHEMA_VERSION = 1
PHASE6_ALIGNED_VALUE_SEED = 6_032
ALIGNED_VALUE_SPLITS = ("train", "calibration", "validation")
ALIGNED_VALUE_PARTICLE_VARIANTS = 3
ALIGNED_VALUE_ROLLOUTS_PER_ACTION = 3
ALIGNED_VALUE_REGULARIZATION_GRID = (0.1, 1.0, 10.0, 100.0, 1_000.0)
ALIGNED_VALUE_TARGET_COVERAGE = 0.90

_ROOT_CAPTURE_CONFIG = {
    "train": ("tempo", 2, False),
    "calibration": ("attack_rush", 2, True),
    "validation": ("control", 2, False),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big"
    )


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _allocation_fingerprint(completion: HiddenZoneCompletion) -> str:
    """Bind raw hidden zones into a digest without persisting the identities."""

    return _fingerprint(
        (
            completion.your_deck,
            completion.your_prize,
            completion.opponent_deck,
            completion.opponent_prize,
            completion.opponent_hand,
            completion.opponent_active,
        )
    )


@dataclass(frozen=True, slots=True)
class AlignedPublicRoot:
    root_id: str
    split: str
    candidate_id: str
    baseline_style: str
    requested_player_1_main_ordinal: int
    capture_attempt: int
    captured_turn: int
    live_selection_count: int
    public_state_fingerprint: str
    public_history_fingerprint: str
    information_set_id: str
    public_belief_fingerprint: str
    public_features: tuple[tuple[str, float], ...]
    allocation_surface_ids: tuple[str, ...]
    live_official_battle_executed: bool = True
    acting_player_observed_hand_fixed_across_allocations: bool = True

    def __post_init__(self) -> None:
        for name in (
            "root_id",
            "public_state_fingerprint",
            "public_history_fingerprint",
            "public_belief_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("aligned root split is invalid")
        if not self.candidate_id or not self.information_set_id:
            raise ValueError("aligned root identity is incomplete")
        expected_style, expected_ordinal, _ = _ROOT_CAPTURE_CONFIG[self.split]
        if self.baseline_style != expected_style:
            raise ValueError("aligned root baseline style drifted")
        if self.requested_player_1_main_ordinal != expected_ordinal:
            raise ValueError("aligned root capture ordinal drifted")
        if min(self.capture_attempt, self.captured_turn, self.live_selection_count) <= 0:
            raise ValueError("aligned root live capture accounting is invalid")
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("aligned root public feature schema mismatch")
        if len(self.allocation_surface_ids) != ALIGNED_VALUE_PARTICLE_VARIANTS:
            raise ValueError("aligned root does not contain the required allocations")
        if len(set(self.allocation_surface_ids)) != len(self.allocation_surface_ids):
            raise ValueError("aligned root surface IDs must be unique")
        for value in self.allocation_surface_ids:
            _validate_digest(value, "allocation_surface_id")
        if not all(
            (
                self.live_official_battle_executed,
                self.acting_player_observed_hand_fixed_across_allocations,
            )
        ):
            raise ValueError("aligned root provenance boundary changed")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "public_features": [[name, value] for name, value in self.public_features],
            "allocation_surface_ids": list(self.allocation_surface_ids),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignedPublicRoot":
        return cls(
            root_id=str(raw["root_id"]),
            split=str(raw["split"]),
            candidate_id=str(raw["candidate_id"]),
            baseline_style=str(raw["baseline_style"]),
            requested_player_1_main_ordinal=int(
                raw["requested_player_1_main_ordinal"]
            ),
            capture_attempt=int(raw["capture_attempt"]),
            captured_turn=int(raw["captured_turn"]),
            live_selection_count=int(raw["live_selection_count"]),
            public_state_fingerprint=str(raw["public_state_fingerprint"]),
            public_history_fingerprint=str(raw["public_history_fingerprint"]),
            information_set_id=str(raw["information_set_id"]),
            public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
            public_features=tuple(
                (str(name), float(value)) for name, value in raw["public_features"]
            ),
            allocation_surface_ids=tuple(
                str(value) for value in raw["allocation_surface_ids"]
            ),
            live_official_battle_executed=bool(
                raw["live_official_battle_executed"]
            ),
            acting_player_observed_hand_fixed_across_allocations=bool(
                raw["acting_player_observed_hand_fixed_across_allocations"]
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignedAllocationSurface:
    surface_id: str
    root_id: str
    split: str
    candidate_id_hypothesis_input: str
    allocation_variant: int
    completion_fingerprint: str
    hidden_allocation_fingerprint: str
    source_game_fingerprint: str
    source_state_id: str
    information_set_id: str
    public_belief_fingerprint: str
    public_history_fingerprint: str
    private_observation_fingerprint: str
    public_state_fingerprint: str
    public_features: tuple[tuple[str, float], ...]
    action_values: tuple[OpponentPlanActionValue, ...]
    public_evidence_adjustments: int = 0
    source_descendants_official_engine_executed: bool = True
    hidden_completion_prior_official: bool = False
    acting_player_observed_hand_resampled: bool = False
    upstream_official_reach_complete: bool = False
    exact_counterfactual_values: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "surface_id",
            "root_id",
            "completion_fingerprint",
            "hidden_allocation_fingerprint",
            "source_game_fingerprint",
            "source_state_id",
            "public_belief_fingerprint",
            "public_history_fingerprint",
            "private_observation_fingerprint",
            "public_state_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if self.split not in ALIGNED_VALUE_SPLITS:
            raise ValueError("aligned surface split is invalid")
        if not self.candidate_id_hypothesis_input or not self.information_set_id:
            raise ValueError("aligned surface identity is incomplete")
        if self.allocation_variant not in range(ALIGNED_VALUE_PARTICLE_VARIANTS):
            raise ValueError("aligned surface allocation variant is invalid")
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("aligned surface public feature schema mismatch")
        if tuple(value.action_id for value in self.action_values) != (
            OPPONENT_DEVIATION_ACTION_IDS
        ):
            raise ValueError("aligned surface action support/order is invalid")
        if not math.isclose(
            sum(value.blueprint_probability for value in self.action_values),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("aligned surface blueprint distribution is not normalized")
        if any(
            (
                self.public_evidence_adjustments != 0,
                not self.source_descendants_official_engine_executed,
                self.hidden_completion_prior_official,
                self.acting_player_observed_hand_resampled,
                self.upstream_official_reach_complete,
                self.exact_counterfactual_values,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("aligned surface promoted an unavailable capability")

    @property
    def maximum_rollout_standard_error(self) -> float:
        return max(value.rollout_standard_error for value in self.action_values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "root_id": self.root_id,
            "split": self.split,
            "candidate_id_hypothesis_input": self.candidate_id_hypothesis_input,
            "allocation_variant": self.allocation_variant,
            "completion_fingerprint": self.completion_fingerprint,
            "hidden_allocation_fingerprint": self.hidden_allocation_fingerprint,
            "source_game_fingerprint": self.source_game_fingerprint,
            "source_state_id": self.source_state_id,
            "information_set_id": self.information_set_id,
            "public_belief_fingerprint": self.public_belief_fingerprint,
            "public_history_fingerprint": self.public_history_fingerprint,
            "private_observation_fingerprint": self.private_observation_fingerprint,
            "public_state_fingerprint": self.public_state_fingerprint,
            "public_features": [[name, value] for name, value in self.public_features],
            "action_values": [value.to_dict() for value in self.action_values],
            "claims": {
                "public_evidence_adjustments": self.public_evidence_adjustments,
                "source_descendants_official_engine_executed": (
                    self.source_descendants_official_engine_executed
                ),
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "acting_player_observed_hand_resampled": (
                    self.acting_player_observed_hand_resampled
                ),
                "upstream_official_reach_complete": self.upstream_official_reach_complete,
                "exact_counterfactual_values": self.exact_counterfactual_values,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignedAllocationSurface":
        claims = raw["claims"]
        return cls(
            surface_id=str(raw["surface_id"]),
            root_id=str(raw["root_id"]),
            split=str(raw["split"]),
            candidate_id_hypothesis_input=str(
                raw["candidate_id_hypothesis_input"]
            ),
            allocation_variant=int(raw["allocation_variant"]),
            completion_fingerprint=str(raw["completion_fingerprint"]),
            hidden_allocation_fingerprint=str(
                raw["hidden_allocation_fingerprint"]
            ),
            source_game_fingerprint=str(raw["source_game_fingerprint"]),
            source_state_id=str(raw["source_state_id"]),
            information_set_id=str(raw["information_set_id"]),
            public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
            public_history_fingerprint=str(raw["public_history_fingerprint"]),
            private_observation_fingerprint=str(
                raw["private_observation_fingerprint"]
            ),
            public_state_fingerprint=str(raw["public_state_fingerprint"]),
            public_features=tuple(
                (str(name), float(value)) for name, value in raw["public_features"]
            ),
            action_values=tuple(
                OpponentPlanActionValue.from_dict(value)
                for value in raw["action_values"]
            ),
            public_evidence_adjustments=int(claims["public_evidence_adjustments"]),
            source_descendants_official_engine_executed=bool(
                claims["source_descendants_official_engine_executed"]
            ),
            hidden_completion_prior_official=bool(
                claims["hidden_completion_prior_official"]
            ),
            acting_player_observed_hand_resampled=bool(
                claims["acting_player_observed_hand_resampled"]
            ),
            upstream_official_reach_complete=bool(
                claims["upstream_official_reach_complete"]
            ),
            exact_counterfactual_values=bool(claims["exact_counterfactual_values"]),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )


@dataclass(frozen=True, slots=True)
class PublicRootAlignedValueDataset:
    runtime_fingerprint: str
    deck_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    candidate_ids: tuple[str, ...]
    particle_variants_per_root: int
    rollouts_per_action: int
    native_transition_count: int
    macro_transition_count: int
    tactical_transition_count: int
    roots: tuple[AlignedPublicRoot, ...]
    surfaces: tuple[AlignedAllocationSurface, ...]
    public_root_alignment_verified: bool
    collection_disjoint_roots_verified: bool
    supplied_hidden_completion_prior: bool = True
    acting_player_hand_range_complete: bool = False
    upstream_official_reach_complete: bool = False
    native_counterfactual_values_calibrated: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for name in (
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("aligned dataset candidates must be unique")
        if self.particle_variants_per_root != ALIGNED_VALUE_PARTICLE_VARIANTS:
            raise ValueError("aligned dataset particle count drifted")
        if self.rollouts_per_action < 2:
            raise ValueError("aligned dataset requires repeated rollouts")
        if min(
            self.native_transition_count,
            self.macro_transition_count,
            self.tactical_transition_count,
        ) <= 0:
            raise ValueError("aligned dataset transition accounting is invalid")
        expected_roots = {
            (split, candidate)
            for split in ALIGNED_VALUE_SPLITS
            for candidate in self.candidate_ids
        }
        observed_roots = {(root.split, root.candidate_id) for root in self.roots}
        if observed_roots != expected_roots or len(self.roots) != len(expected_roots):
            raise ValueError("aligned dataset root support is incomplete")
        if len({root.root_id for root in self.roots}) != len(self.roots):
            raise ValueError("aligned dataset public roots are not disjoint")
        roots_by_id = {root.root_id: root for root in self.roots}
        if len(roots_by_id) != len(self.roots):
            raise ValueError("aligned dataset root identifiers are duplicated")
        expected_surfaces = {
            (root.root_id, variant)
            for root in self.roots
            for variant in range(self.particle_variants_per_root)
        }
        observed_surfaces = {
            (surface.root_id, surface.allocation_variant)
            for surface in self.surfaces
        }
        if observed_surfaces != expected_surfaces or len(self.surfaces) != len(
            expected_surfaces
        ):
            raise ValueError("aligned dataset allocation support is incomplete")
        if len({surface.surface_id for surface in self.surfaces}) != len(self.surfaces):
            raise ValueError("aligned dataset surface IDs are duplicated")
        for root in self.roots:
            group = tuple(surface for surface in self.surfaces if surface.root_id == root.root_id)
            if set(root.allocation_surface_ids) != {surface.surface_id for surface in group}:
                raise ValueError("aligned root/surface manifest mismatch")
            if any(
                any(
                    (
                    surface.split != root.split,
                    surface.candidate_id_hypothesis_input != root.candidate_id,
                    surface.public_state_fingerprint != root.public_state_fingerprint,
                    surface.public_history_fingerprint != root.public_history_fingerprint,
                    surface.information_set_id != root.information_set_id,
                    surface.public_belief_fingerprint != root.public_belief_fingerprint,
                    surface.public_features != root.public_features,
                    )
                )
                for surface in group
            ):
                raise ValueError("aligned root contains a public-information mismatch")
            if len({surface.hidden_allocation_fingerprint for surface in group}) != len(group):
                raise ValueError("aligned root hidden allocations are not distinct")
        expected_alignment = all(
            len(
                {
                    (
                        surface.public_state_fingerprint,
                        surface.public_history_fingerprint,
                        surface.information_set_id,
                        surface.public_belief_fingerprint,
                        surface.public_features,
                    )
                    for surface in self.surfaces
                    if surface.root_id == root.root_id
                }
            )
            == 1
            for root in self.roots
        )
        if self.public_root_alignment_verified != expected_alignment:
            raise ValueError("aligned dataset public-root gate is inconsistent")
        split_roots = {
            split: {root.root_id for root in self.roots if root.split == split}
            for split in ALIGNED_VALUE_SPLITS
        }
        expected_disjoint = all(
            split_roots[left].isdisjoint(split_roots[right])
            for index, left in enumerate(ALIGNED_VALUE_SPLITS)
            for right in ALIGNED_VALUE_SPLITS[index + 1 :]
        )
        if self.collection_disjoint_roots_verified != expected_disjoint:
            raise ValueError("aligned dataset split-disjoint gate is inconsistent")
        if not all(
            (
                self.public_root_alignment_verified,
                self.collection_disjoint_roots_verified,
                self.supplied_hidden_completion_prior,
            )
        ):
            raise ValueError("aligned dataset substrate gates must hold")
        if any(
            (
                self.acting_player_hand_range_complete,
                self.upstream_official_reach_complete,
                self.native_counterfactual_values_calibrated,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("aligned dataset promoted an unavailable capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def action_target_count(self) -> int:
        return len(self.surfaces) * len(OPPONENT_DEVIATION_ACTION_IDS)

    @property
    def terminal_rollout_count(self) -> int:
        return self.action_target_count * self.rollouts_per_action

    @property
    def maximum_rollout_standard_error(self) -> float:
        return max(surface.maximum_rollout_standard_error for surface in self.surfaces)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_VALUE_SCHEMA_VERSION,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "source_plan_blueprint_fingerprint": self.source_plan_blueprint_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "particle_variants_per_root": self.particle_variants_per_root,
            "rollouts_per_action": self.rollouts_per_action,
            "accounting": {
                "root_count": len(self.roots),
                "surface_count": len(self.surfaces),
                "action_target_count": self.action_target_count,
                "terminal_rollout_count": self.terminal_rollout_count,
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "tactical_transition_count": self.tactical_transition_count,
                "maximum_rollout_standard_error": self.maximum_rollout_standard_error,
            },
            "roots": [root.to_dict() for root in self.roots],
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "gates": {
                "public_root_alignment_verified": self.public_root_alignment_verified,
                "collection_disjoint_roots_verified": (
                    self.collection_disjoint_roots_verified
                ),
                "native_counterfactual_values_calibrated": (
                    self.native_counterfactual_values_calibrated
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
            "claims": {
                "live_roots_official_battle_executed": True,
                "source_descendants_official_search_executed": True,
                "five_plan_action_boundary_abstract": True,
                "supplied_hidden_completion_prior": self.supplied_hidden_completion_prior,
                "acting_player_hand_range_complete": (
                    self.acting_player_hand_range_complete
                ),
                "upstream_official_reach_complete": self.upstream_official_reach_complete,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRootAlignedValueDataset":
        accounting = raw["accounting"]
        gates = raw["gates"]
        claims = raw["claims"]
        if not all(
            bool(claims[name])
            for name in (
                "live_roots_official_battle_executed",
                "source_descendants_official_search_executed",
                "five_plan_action_boundary_abstract",
            )
        ):
            raise ValueError("aligned dataset execution provenance changed")
        result = cls(
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(
                raw["source_plan_blueprint_fingerprint"]
            ),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            particle_variants_per_root=int(raw["particle_variants_per_root"]),
            rollouts_per_action=int(raw["rollouts_per_action"]),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            tactical_transition_count=int(accounting["tactical_transition_count"]),
            roots=tuple(AlignedPublicRoot.from_dict(value) for value in raw["roots"]),
            surfaces=tuple(
                AlignedAllocationSurface.from_dict(value) for value in raw["surfaces"]
            ),
            public_root_alignment_verified=bool(
                gates["public_root_alignment_verified"]
            ),
            collection_disjoint_roots_verified=bool(
                gates["collection_disjoint_roots_verified"]
            ),
            supplied_hidden_completion_prior=bool(
                claims["supplied_hidden_completion_prior"]
            ),
            acting_player_hand_range_complete=bool(
                claims["acting_player_hand_range_complete"]
            ),
            upstream_official_reach_complete=bool(
                claims["upstream_official_reach_complete"]
            ),
            native_counterfactual_values_calibrated=bool(
                gates["native_counterfactual_values_calibrated"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected_accounting = {
            "root_count": len(result.roots),
            "surface_count": len(result.surfaces),
            "action_target_count": result.action_target_count,
            "terminal_rollout_count": result.terminal_rollout_count,
            "native_transition_count": result.native_transition_count,
            "macro_transition_count": result.macro_transition_count,
            "tactical_transition_count": result.tactical_transition_count,
            "maximum_rollout_standard_error": result.maximum_rollout_standard_error,
        }
        if any(
            not math.isclose(float(accounting[name]), float(value), abs_tol=1e-12)
            for name, value in expected_accounting.items()
        ):
            raise ValueError("aligned dataset accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("aligned dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class RootCrossValidationMetric:
    regularization: float
    leave_public_root_out_rmse: float

    def __post_init__(self) -> None:
        if self.regularization <= 0 or not math.isfinite(self.regularization):
            raise ValueError("aligned ridge regularization is invalid")
        if self.leave_public_root_out_rmse < 0 or not math.isfinite(
            self.leave_public_root_out_rmse
        ):
            raise ValueError("aligned root CV RMSE is invalid")


@dataclass(frozen=True, slots=True)
class HypothesisConditionedValueModel:
    source_dataset_fingerprint: str
    deck_fingerprint: str
    candidate_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    normalizer: NamedFeatureNormalizer
    regularization_grid: tuple[float, ...]
    cross_validation_metrics: tuple[RootCrossValidationMetric, ...]
    selected_regularization: float
    weights: tuple[float, ...]
    train_root_ids: tuple[str, ...]
    train_row_ids: tuple[str, ...]
    train_rows_fingerprint: str
    hypothesis_identifier_input: bool = True
    conditional_rollout_target: bool = True
    counterfactual_value_provider: bool = False
    safe_resolver_compatible: bool = False
    official_engine_executed: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_dataset_fingerprint",
            "deck_fingerprint",
            "train_rows_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("aligned model candidate support is invalid")
        if self.action_ids != OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("aligned model action support drifted")
        if self.regularization_grid != ALIGNED_VALUE_REGULARIZATION_GRID:
            raise ValueError("aligned model regularization grid drifted")
        if tuple(item.regularization for item in self.cross_validation_metrics) != (
            self.regularization_grid
        ):
            raise ValueError("aligned model CV grid is incomplete")
        selected = min(
            self.cross_validation_metrics,
            key=lambda item: (item.leave_public_root_out_rmse, item.regularization),
        ).regularization
        if self.selected_regularization != selected:
            raise ValueError("aligned model regularization is not the CV winner")
        expected_names = (
            NATIVE_PUBLIC_FEATURE_NAMES
            + tuple(f"hypothesis::{item}" for item in self.candidate_ids)
            + tuple(f"action::{item}" for item in self.action_ids)
        )
        if self.normalizer.names != expected_names:
            raise ValueError("aligned model feature schema mismatch")
        if self.normalizer.fit_rows_fingerprint != self.train_rows_fingerprint:
            raise ValueError("aligned model normalizer provenance mismatch")
        if len(self.weights) != len(expected_names) + 1 or not all(
            math.isfinite(value) for value in self.weights
        ):
            raise ValueError("aligned model weights are invalid")
        if not self.train_root_ids or len(set(self.train_root_ids)) != len(
            self.train_root_ids
        ):
            raise ValueError("aligned model train roots are invalid")
        if not self.train_row_ids or len(set(self.train_row_ids)) != len(
            self.train_row_ids
        ):
            raise ValueError("aligned model train rows are invalid")
        if not self.hypothesis_identifier_input or not self.conditional_rollout_target:
            raise ValueError("aligned model scope claims changed")
        if any(
            (
                self.counterfactual_value_provider,
                self.safe_resolver_compatible,
                self.official_engine_executed,
            )
        ):
            raise ValueError("aligned model promoted an unavailable capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def features(
        self,
        public_features: Sequence[tuple[str, float]],
        candidate_id: str,
        action_id: str,
    ) -> tuple[tuple[str, float], ...]:
        return _model_features(
            public_features,
            candidate_id,
            action_id,
            candidate_ids=self.candidate_ids,
            action_ids=self.action_ids,
        )

    def predict(
        self,
        public_features: Sequence[tuple[str, float]],
        candidate_id: str,
        action_id: str,
    ) -> float:
        values = self.normalizer.transform(
            self.features(public_features, candidate_id, action_id)
        )
        prediction = self.weights[0] + sum(
            weight * value
            for weight, value in zip(self.weights[1:], values, strict=True)
        )
        return max(-1.0, min(1.0, float(prediction)))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_VALUE_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "action_ids": list(self.action_ids),
            "normalizer": self.normalizer.to_dict(),
            "regularization_grid": list(self.regularization_grid),
            "cross_validation_metrics": [
                asdict(item) for item in self.cross_validation_metrics
            ],
            "selected_regularization": self.selected_regularization,
            "weights": list(self.weights),
            "train_root_ids": list(self.train_root_ids),
            "train_row_ids": list(self.train_row_ids),
            "train_rows_fingerprint": self.train_rows_fingerprint,
            "claims": {
                "hypothesis_identifier_input": self.hypothesis_identifier_input,
                "conditional_rollout_target": self.conditional_rollout_target,
                "counterfactual_value_provider": self.counterfactual_value_provider,
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_engine_executed": self.official_engine_executed,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HypothesisConditionedValueModel":
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            action_ids=tuple(str(value) for value in raw["action_ids"]),
            normalizer=NamedFeatureNormalizer.from_dict(raw["normalizer"]),
            regularization_grid=tuple(
                float(value) for value in raw["regularization_grid"]
            ),
            cross_validation_metrics=tuple(
                RootCrossValidationMetric(**dict(value))
                for value in raw["cross_validation_metrics"]
            ),
            selected_regularization=float(raw["selected_regularization"]),
            weights=tuple(float(value) for value in raw["weights"]),
            train_root_ids=tuple(str(value) for value in raw["train_root_ids"]),
            train_row_ids=tuple(str(value) for value in raw["train_row_ids"]),
            train_rows_fingerprint=str(raw["train_rows_fingerprint"]),
            hypothesis_identifier_input=bool(claims["hypothesis_identifier_input"]),
            conditional_rollout_target=bool(claims["conditional_rollout_target"]),
            counterfactual_value_provider=bool(claims["counterfactual_value_provider"]),
            safe_resolver_compatible=bool(claims["safe_resolver_compatible"]),
            official_engine_executed=bool(claims["official_engine_executed"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("aligned model fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class AlignedActionIntervalMetric:
    action_id: str
    calibration_examples: int
    validation_examples: int
    conformal_rank: int
    conformal_radius: float
    calibration_coverage: float
    validation_coverage: float
    validation_mean_interval_width: float
    validation_rmse: float
    validation_zero_baseline_rmse: float

    def __post_init__(self) -> None:
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("aligned interval action is unsupported")
        if min(self.calibration_examples, self.validation_examples) <= 0:
            raise ValueError("aligned interval split is empty")
        if not 1 <= self.conformal_rank <= self.calibration_examples:
            raise ValueError("aligned interval conformal rank is invalid")
        if self.conformal_radius < 0 or not math.isfinite(self.conformal_radius):
            raise ValueError("aligned interval radius is invalid")
        for name in ("calibration_coverage", "validation_coverage"):
            value = getattr(self, name)
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite fraction")
        for name in (
            "validation_mean_interval_width",
            "validation_rmse",
            "validation_zero_baseline_rmse",
        ):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class AlignedValueCalibration:
    dataset_fingerprint: str
    model_fingerprint: str
    target_coverage: float
    action_metrics: tuple[AlignedActionIntervalMetric, ...]
    validation_coverage: float
    validation_rmse: float
    validation_zero_baseline_rmse: float
    maximum_rollout_standard_error: float
    public_root_alignment_gate_passed: bool
    collection_disjoint_gate_passed: bool
    empirical_conditional_model_gate_passed: bool
    native_counterfactual_value_gate_passed: bool = False
    safe_resolver_compatible: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        _validate_digest(self.model_fingerprint, "model_fingerprint")
        if not 0 < self.target_coverage < 1:
            raise ValueError("aligned target coverage is invalid")
        if tuple(item.action_id for item in self.action_metrics) != (
            OPPONENT_DEVIATION_ACTION_IDS
        ):
            raise ValueError("aligned calibration action metrics are incomplete")
        for name in ("validation_coverage",):
            value = getattr(self, name)
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite fraction")
        for name in (
            "validation_rmse",
            "validation_zero_baseline_rmse",
            "maximum_rollout_standard_error",
        ):
            value = getattr(self, name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        expected_empirical_gate = all(
            (
                self.public_root_alignment_gate_passed,
                self.collection_disjoint_gate_passed,
                self.validation_coverage >= self.target_coverage,
                self.validation_rmse <= self.validation_zero_baseline_rmse,
                self.maximum_rollout_standard_error <= 0.70,
            )
        )
        if self.empirical_conditional_model_gate_passed != expected_empirical_gate:
            raise ValueError("aligned empirical model gate is inconsistent")
        if any(
            (
                self.native_counterfactual_value_gate_passed,
                self.safe_resolver_compatible,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("aligned calibration promoted unavailable control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_VALUE_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "target_coverage": self.target_coverage,
            "action_metrics": [asdict(item) for item in self.action_metrics],
            "metrics": {
                "validation_coverage": self.validation_coverage,
                "validation_rmse": self.validation_rmse,
                "validation_zero_baseline_rmse": self.validation_zero_baseline_rmse,
                "maximum_rollout_standard_error": self.maximum_rollout_standard_error,
            },
            "gates": {
                "public_root_alignment": self.public_root_alignment_gate_passed,
                "collection_disjoint": self.collection_disjoint_gate_passed,
                "empirical_conditional_model": (
                    self.empirical_conditional_model_gate_passed
                ),
                "native_counterfactual_value": (
                    self.native_counterfactual_value_gate_passed
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
            "blockers": [
                "hidden-completion prior is supplied rather than an official posterior",
                "the acting player's observed hand is fixed across allocation variants",
                "root-relative collection omits upstream official reach probabilities",
                "five plan actions and the fitted ridge model are research abstractions",
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignedValueCalibration":
        metrics = raw["metrics"]
        gates = raw["gates"]
        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            model_fingerprint=str(raw["model_fingerprint"]),
            target_coverage=float(raw["target_coverage"]),
            action_metrics=tuple(
                AlignedActionIntervalMetric(**dict(value))
                for value in raw["action_metrics"]
            ),
            validation_coverage=float(metrics["validation_coverage"]),
            validation_rmse=float(metrics["validation_rmse"]),
            validation_zero_baseline_rmse=float(
                metrics["validation_zero_baseline_rmse"]
            ),
            maximum_rollout_standard_error=float(
                metrics["maximum_rollout_standard_error"]
            ),
            public_root_alignment_gate_passed=bool(
                gates["public_root_alignment"]
            ),
            collection_disjoint_gate_passed=bool(gates["collection_disjoint"]),
            empirical_conditional_model_gate_passed=bool(
                gates["empirical_conditional_model"]
            ),
            native_counterfactual_value_gate_passed=bool(
                gates["native_counterfactual_value"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if len(raw.get("blockers", ())) != 4:
            raise ValueError("aligned calibration blockers changed")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("aligned calibration fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6AlignedValueEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    surface_count: int
    action_target_count: int
    terminal_rollout_count: int
    native_transition_count: int
    train_root_count: int
    calibration_root_count: int
    validation_root_count: int
    validation_coverage: float
    validation_rmse: float
    validation_zero_baseline_rmse: float
    maximum_rollout_standard_error: float
    public_root_alignment_gate_passed: bool
    collection_disjoint_gate_passed: bool
    empirical_conditional_model_gate_passed: bool
    native_counterfactual_values_calibrated: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6Z evidence identity is missing")
        for name in (
            "dataset_fingerprint",
            "model_fingerprint",
            "calibration_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if min(
            self.root_count,
            self.surface_count,
            self.action_target_count,
            self.terminal_rollout_count,
            self.native_transition_count,
            self.train_root_count,
            self.calibration_root_count,
            self.validation_root_count,
        ) <= 0:
            raise ValueError("Phase 6Z evidence accounting is invalid")
        if self.root_count != (
            self.train_root_count
            + self.calibration_root_count
            + self.validation_root_count
        ):
            raise ValueError("Phase 6Z root split accounting is inconsistent")
        if not all(
            (
                self.public_root_alignment_gate_passed,
                self.collection_disjoint_gate_passed,
            )
        ):
            raise ValueError("Phase 6Z requires aligned, disjoint root collections")
        if any(
            (
                self.native_counterfactual_values_calibrated,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6Z evidence cannot enable downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_ALIGNED_VALUE_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6Z public-root-aligned value evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Disjoint public roots | {self.root_count} |
| Hidden-allocation surfaces | {self.surface_count} |
| Forced plan-action targets | {self.action_target_count} |
| Genuine terminal continuations | {self.terminal_rollout_count} |
| Genuine native transitions | {self.native_transition_count} |
| Validation interval coverage | `{self.validation_coverage:.6f}` |
| Validation RMSE | `{self.validation_rmse:.6f}` |
| Zero-baseline validation RMSE | `{self.validation_zero_baseline_rmse:.6f}` |
| Maximum rollout standard error | `{self.maximum_rollout_standard_error:.6f}` |
| Public-root alignment gate | **{str(self.public_root_alignment_gate_passed).lower()}** |
| Collection-disjoint gate | **{str(self.collection_disjoint_gate_passed).lower()}** |
| Conditional-model diagnostic gate | **{str(self.empirical_conditional_model_gate_passed).lower()}** |
| Native counterfactual values calibrated | **false** |

The live anchors and descendant transitions genuinely execute through the
official engine. The hidden prior, five-plan boundary, fitted value model, and
interval gates are research abstractions. Acting-player hands are fixed within
each aligned root, upstream reach is absent, and no resolver or official
strategy control is enabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6AlignedValueEvidence":
        fields = {name: raw[name] for name in cls.__dataclass_fields__ if name != "claims"}
        result = cls(
            **fields,
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6Z evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6AlignedValueArtifacts:
    dataset: PublicRootAlignedValueDataset
    model: HypothesisConditionedValueModel
    calibration: AlignedValueCalibration
    evidence: Phase6AlignedValueEvidence


def _capture_later_player_1_root(
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    *,
    target: KaggleDeckProfile,
    candidate: KaggleDeckProfile,
    style: str,
    ordinal: int,
    reverse: bool,
    maximum_live_selections: int,
) -> tuple[dict[str, Any], int, int]:
    observation, start = battle_start(list(target.card_ids), list(candidate.card_ids))
    if observation is None:
        raise RuntimeError(
            "official engine rejected Phase 6Z decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        policies = (
            PriorityBaseline(target.card_ids, style=style, reverse=reverse),
            PriorityBaseline(candidate.card_ids, style=style, reverse=not reverse),
        )
        observed_player_1_main_turns: list[int] = []
        selections = 0
        while (
            observation is not None
            and _safe_int(_state(observation).get("result"), -1) == -1
            and selections < maximum_live_selections
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
            if actor == 1 and context == int(SelectContext.MAIN):
                if turn not in observed_player_1_main_turns:
                    observed_player_1_main_turns.append(turn)
                if len(observed_player_1_main_turns) >= ordinal:
                    if observation.get("search_begin_input") is None:
                        raise RuntimeError("Phase 6Z live root lacks exact search input")
                    return deepcopy(dict(observation)), turn, selections
            contract = SelectionContract.from_observation(observation)
            selection = contract.validate(policies[actor].choose(observation))
            observation = battle_select(list(selection))
            selections += 1
        raise RuntimeError(
            f"Phase 6Z could not capture player-1 MAIN root ordinal {ordinal} "
            f"within {maximum_live_selections} selections"
        )
    except Exception:
        battle_finish()
        raise


def _aligned_completions(
    observation: Mapping[str, Any],
    *,
    split: str,
    candidate: KaggleDeckProfile,
    target: KaggleDeckProfile,
    basic_card_ids: Sequence[int],
    variants: int,
    seed: int,
) -> tuple[tuple[HiddenZoneCompletion, ...], dict[str, tuple[int, str]]]:
    completions: list[HiddenZoneCompletion] = []
    metadata: dict[str, tuple[int, str]] = {}
    allocation_fingerprints: set[str] = set()
    probability = 1.0 / variants
    for variant in range(variants):
        completion: HiddenZoneCompletion | None = None
        allocation_fingerprint = ""
        for attempt in range(128):
            identifier = f"{split}:{candidate.archetype}:allocation-{variant}"
            candidate_completion = sample_hidden_zone_completion(
                observation,
                hypothesis_id=identifier,
                probability=probability,
                allocation_seed=_seed(
                    seed,
                    split,
                    candidate.fingerprint,
                    variant,
                    attempt,
                ),
                your_candidate_id=candidate.archetype,
                opponent_candidate_id=target.archetype,
                your_archetype_id=candidate.archetype,
                opponent_archetype_id=target.archetype,
                your_style_id=_ROOT_CAPTURE_CONFIG[split][0],
                opponent_style_id=_ROOT_CAPTURE_CONFIG[split][0],
                your_card_ids=candidate.card_ids,
                opponent_card_ids=target.card_ids,
                your_deck_fingerprint=candidate.fingerprint,
                opponent_deck_fingerprint=target.fingerprint,
                basic_card_ids=basic_card_ids,
                source=(
                    "Phase 6Z same-candidate public-root-aligned hidden allocation; "
                    "acting player's observed hand remains fixed"
                ),
            )
            if candidate_completion.public_evidence_adjustments != 0:
                raise RuntimeError(
                    "Phase 6Z candidate root required public-evidence substitution"
                )
            candidate_fingerprint = _allocation_fingerprint(candidate_completion)
            if candidate_fingerprint not in allocation_fingerprints:
                completion = candidate_completion
                allocation_fingerprint = candidate_fingerprint
                break
        if completion is None:
            raise RuntimeError("Phase 6Z could not sample distinct hidden allocations")
        allocation_fingerprints.add(allocation_fingerprint)
        completions.append(completion)
        metadata[completion.hypothesis_id] = (variant, allocation_fingerprint)
    return tuple(completions), metadata


def _model_features(
    public_features: Sequence[tuple[str, float]],
    candidate_id: str,
    action_id: str,
    *,
    candidate_ids: Sequence[str],
    action_ids: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    materialized = tuple((str(name), float(value)) for name, value in public_features)
    if tuple(name for name, _ in materialized) != NATIVE_PUBLIC_FEATURE_NAMES:
        raise ValueError("aligned model public feature schema mismatch")
    if candidate_id not in candidate_ids or action_id not in action_ids:
        raise ValueError("aligned model categorical input is unsupported")
    return (
        materialized
        + tuple(
            (f"hypothesis::{item}", float(item == candidate_id))
            for item in candidate_ids
        )
        + tuple((f"action::{item}", float(item == action_id)) for item in action_ids)
    )


def _training_rows(
    dataset: PublicRootAlignedValueDataset,
    split: str,
) -> tuple[tuple[str, str, tuple[tuple[str, float], ...], float], ...]:
    rows = []
    for surface in dataset.surfaces:
        if surface.split != split:
            continue
        for value in surface.action_values:
            row_id = _fingerprint((surface.surface_id, value.action_id))
            features = _model_features(
                surface.public_features,
                surface.candidate_id_hypothesis_input,
                value.action_id,
                candidate_ids=dataset.candidate_ids,
                action_ids=OPPONENT_DEVIATION_ACTION_IDS,
            )
            rows.append(
                (row_id, surface.root_id, features, value.empirical_opponent_value)
            )
    return tuple(rows)


def _fit_weights(
    rows: Sequence[Sequence[tuple[str, float]]],
    targets: Sequence[float],
    *,
    normalizer: NamedFeatureNormalizer,
    regularization: float,
) -> tuple[float, ...]:
    matrix = torch.tensor(
        [normalizer.transform(row) for row in rows], dtype=torch.float64
    )
    matrix = torch.cat(
        (torch.ones((len(matrix), 1), dtype=torch.float64), matrix), dim=1
    )
    target = torch.tensor(targets, dtype=torch.float64)
    penalty = torch.eye(matrix.shape[1], dtype=torch.float64)
    penalty[0, 0] = 0.0
    weights = torch.linalg.solve(
        matrix.T @ matrix + regularization * penalty,
        matrix.T @ target,
    )
    return tuple(float(value) for value in weights.tolist())


def _predict_weights(
    weights: Sequence[float],
    normalizer: NamedFeatureNormalizer,
    features: Sequence[tuple[str, float]],
) -> float:
    values = normalizer.transform(features)
    prediction = weights[0] + sum(
        weight * value for weight, value in zip(weights[1:], values, strict=True)
    )
    return max(-1.0, min(1.0, float(prediction)))


def train_hypothesis_conditioned_value_model(
    dataset: PublicRootAlignedValueDataset,
) -> HypothesisConditionedValueModel:
    rows = _training_rows(dataset, "train")
    train_root_ids = tuple(sorted({root_id for _, root_id, _, _ in rows}))
    expected_roots = len(dataset.candidate_ids)
    if len(train_root_ids) != expected_roots:
        raise ValueError("aligned model requires one training root per candidate")
    metrics = []
    for regularization in ALIGNED_VALUE_REGULARIZATION_GRID:
        squared_errors = []
        for held_root in train_root_ids:
            fold_train = tuple(row for row in rows if row[1] != held_root)
            fold_test = tuple(row for row in rows if row[1] == held_root)
            fold_fingerprint = _fingerprint(
                [(row_id, root_id, features) for row_id, root_id, features, _ in fold_train]
            )
            normalizer = NamedFeatureNormalizer.fit(
                [features for _, _, features, _ in fold_train],
                fit_rows_fingerprint=fold_fingerprint,
            )
            weights = _fit_weights(
                [features for _, _, features, _ in fold_train],
                [target for _, _, _, target in fold_train],
                normalizer=normalizer,
                regularization=regularization,
            )
            squared_errors.extend(
                (
                    _predict_weights(weights, normalizer, features) - target
                )
                ** 2
                for _, _, features, target in fold_test
            )
        metrics.append(
            RootCrossValidationMetric(
                regularization=regularization,
                leave_public_root_out_rmse=math.sqrt(
                    sum(squared_errors) / len(squared_errors)
                ),
            )
        )
    selected = min(
        metrics,
        key=lambda item: (item.leave_public_root_out_rmse, item.regularization),
    ).regularization
    rows_fingerprint = _fingerprint(
        [(row_id, root_id, features, target) for row_id, root_id, features, target in rows]
    )
    normalizer = NamedFeatureNormalizer.fit(
        [features for _, _, features, _ in rows],
        fit_rows_fingerprint=rows_fingerprint,
    )
    weights = _fit_weights(
        [features for _, _, features, _ in rows],
        [target for _, _, _, target in rows],
        normalizer=normalizer,
        regularization=selected,
    )
    return HypothesisConditionedValueModel(
        source_dataset_fingerprint=dataset.fingerprint,
        deck_fingerprint=dataset.deck_fingerprint,
        candidate_ids=dataset.candidate_ids,
        action_ids=OPPONENT_DEVIATION_ACTION_IDS,
        normalizer=normalizer,
        regularization_grid=ALIGNED_VALUE_REGULARIZATION_GRID,
        cross_validation_metrics=tuple(metrics),
        selected_regularization=selected,
        weights=weights,
        train_root_ids=train_root_ids,
        train_row_ids=tuple(row_id for row_id, _, _, _ in rows),
        train_rows_fingerprint=rows_fingerprint,
    )


def calibrate_aligned_value_model(
    dataset: PublicRootAlignedValueDataset,
    model: HypothesisConditionedValueModel,
    *,
    target_coverage: float = ALIGNED_VALUE_TARGET_COVERAGE,
) -> AlignedValueCalibration:
    if model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("aligned calibration model/dataset mismatch")
    if not 0 < target_coverage < 1:
        raise ValueError("aligned calibration target coverage is invalid")
    action_metrics = []
    all_validation_truths = []
    all_validation_predictions = []
    all_validation_covered = []
    for action_id in OPPONENT_DEVIATION_ACTION_IDS:
        by_split: dict[str, list[tuple[float, float]]] = {
            "calibration": [],
            "validation": [],
        }
        for surface in dataset.surfaces:
            if surface.split not in by_split:
                continue
            value = next(item for item in surface.action_values if item.action_id == action_id)
            prediction = model.predict(
                surface.public_features,
                surface.candidate_id_hypothesis_input,
                action_id,
            )
            by_split[surface.split].append(
                (value.empirical_opponent_value, prediction)
            )
        calibration_rows = by_split["calibration"]
        validation_rows = by_split["validation"]
        residuals = sorted(abs(truth - prediction) for truth, prediction in calibration_rows)
        rank = min(len(residuals), math.ceil((len(residuals) + 1) * target_coverage))
        radius = residuals[rank - 1]

        def covered(truth: float, prediction: float) -> bool:
            return max(-1.0, prediction - radius) - 1e-12 <= truth <= min(
                1.0, prediction + radius
            ) + 1e-12

        calibration_covered = [covered(*row) for row in calibration_rows]
        validation_covered = [covered(*row) for row in validation_rows]
        widths = [
            min(1.0, prediction + radius) - max(-1.0, prediction - radius)
            for _, prediction in validation_rows
        ]
        action_metrics.append(
            AlignedActionIntervalMetric(
                action_id=action_id,
                calibration_examples=len(calibration_rows),
                validation_examples=len(validation_rows),
                conformal_rank=rank,
                conformal_radius=radius,
                calibration_coverage=sum(calibration_covered) / len(calibration_covered),
                validation_coverage=sum(validation_covered) / len(validation_covered),
                validation_mean_interval_width=sum(widths) / len(widths),
                validation_rmse=math.sqrt(
                    sum((prediction - truth) ** 2 for truth, prediction in validation_rows)
                    / len(validation_rows)
                ),
                validation_zero_baseline_rmse=math.sqrt(
                    sum(truth**2 for truth, _ in validation_rows)
                    / len(validation_rows)
                ),
            )
        )
        all_validation_truths.extend(truth for truth, _ in validation_rows)
        all_validation_predictions.extend(prediction for _, prediction in validation_rows)
        all_validation_covered.extend(validation_covered)
    validation_rmse = math.sqrt(
        sum(
            (prediction - truth) ** 2
            for truth, prediction in zip(
                all_validation_truths, all_validation_predictions, strict=True
            )
        )
        / len(all_validation_truths)
    )
    zero_rmse = math.sqrt(
        sum(truth**2 for truth in all_validation_truths) / len(all_validation_truths)
    )
    validation_coverage = sum(all_validation_covered) / len(all_validation_covered)
    empirical_gate = all(
        (
            dataset.public_root_alignment_verified,
            dataset.collection_disjoint_roots_verified,
            validation_coverage >= target_coverage,
            validation_rmse <= zero_rmse,
            dataset.maximum_rollout_standard_error <= 0.70,
        )
    )
    return AlignedValueCalibration(
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        target_coverage=target_coverage,
        action_metrics=tuple(action_metrics),
        validation_coverage=validation_coverage,
        validation_rmse=validation_rmse,
        validation_zero_baseline_rmse=zero_rmse,
        maximum_rollout_standard_error=dataset.maximum_rollout_standard_error,
        public_root_alignment_gate_passed=dataset.public_root_alignment_verified,
        collection_disjoint_gate_passed=dataset.collection_disjoint_roots_verified,
        empirical_conditional_model_gate_passed=empirical_gate,
    )


def run_phase6_aligned_counterfactual_values(
    *,
    project_root: str | Path | None = None,
    source_tactical_blueprint_path: str | Path | None = None,
    source_plan_blueprint_path: str | Path | None = None,
    source_phase6p_evidence_path: str | Path | None = None,
    particle_variants: int = ALIGNED_VALUE_PARTICLE_VARIANTS,
    rollouts_per_action: int = ALIGNED_VALUE_ROLLOUTS_PER_ACTION,
    maximum_live_selections: int = 900,
    maximum_capture_attempts: int = 8,
    maximum_rollout_transitions: int = 512,
    maximum_tactical_transitions: int = 512,
    seed: int = PHASE6_ALIGNED_VALUE_SEED,
) -> Phase6AlignedValueArtifacts:
    if particle_variants != ALIGNED_VALUE_PARTICLE_VARIANTS:
        raise ValueError("Phase 6Z requires exactly three allocations per public root")
    if min(
        rollouts_per_action,
        maximum_live_selections,
        maximum_capture_attempts,
        maximum_rollout_transitions,
        maximum_tactical_transitions,
    ) <= 0:
        raise ValueError("Phase 6Z limits must be positive")
    if rollouts_per_action < 2:
        raise ValueError("Phase 6Z requires repeated terminal rollouts")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
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
        or root / "artifacts" / "phase6_native_plan_iterations" / "phase6p_evidence.json"
    ).resolve()
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
        raise ValueError("Phase 6Z plan model/evidence mismatch")
    if plan_artifact.source_tactical_blueprint_fingerprint != tactical_artifact.fingerprint:
        raise ValueError("Phase 6Z plan/tactical provenance mismatch")
    bridge = NativeBlueprintFeatureBridge()
    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import (  # type: ignore[import-not-found]
        battle_finish,
        battle_select,
        battle_start,
    )

    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    profile_by_id = {
        profile.archetype: profile for profile in OFFICIAL_PANEL_DECKS + (target,)
    }
    if set(tactical_artifact.candidate_ids) != set(profile_by_id):
        raise ValueError("Phase 6Z candidate catalog/native profiles mismatch")
    candidates = tuple(profile_by_id[name] for name in tactical_artifact.candidate_ids)
    if tactical_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6Z tactical blueprint/deck mismatch")
    if plan_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6Z plan blueprint/deck mismatch")

    roots: list[AlignedPublicRoot] = []
    surfaces: list[AlignedAllocationSurface] = []
    native_transitions = 0
    macro_transitions = 0
    tactical_transitions = 0
    runtime_fingerprint: str | None = None
    basics = _runtime_basics(runtime)
    for split in ALIGNED_VALUE_SPLITS:
        style, ordinal, reverse = _ROOT_CAPTURE_CONFIG[split]
        for candidate_index, candidate in enumerate(candidates):
            battle_active = False
            try:
                capture_error: RuntimeError | None = None
                for capture_attempt in range(1, maximum_capture_attempts + 1):
                    try:
                        observation, captured_turn, selection_count = (
                            _capture_later_player_1_root(
                                battle_start,
                                battle_select,
                                battle_finish,
                                target=target,
                                candidate=candidate,
                                style=style,
                                ordinal=ordinal,
                                reverse=(
                                    reverse
                                    ^ bool(candidate_index % 2)
                                    ^ bool((capture_attempt - 1) % 2)
                                ),
                                maximum_live_selections=maximum_live_selections,
                            )
                        )
                        break
                    except RuntimeError as exc:
                        capture_error = exc
                else:
                    raise RuntimeError(
                        "Phase 6Z exhausted live-root capture attempts"
                    ) from capture_error
                battle_active = True
                public_state_fingerprint = _fingerprint(
                    sanitize_official_public_state(observation)
                )
                completions, completion_metadata = _aligned_completions(
                    observation,
                    split=split,
                    candidate=candidate,
                    target=target,
                    basic_card_ids=basics,
                    variants=particle_variants,
                    seed=seed,
                )
                with OfficialNativeSearchGame(
                    observation,
                    completions,
                    runtime_directory=runtime,
                ) as native_game:
                    if runtime_fingerprint is None:
                        runtime_fingerprint = native_game.provenance.ruleset_fingerprint
                    elif runtime_fingerprint != native_game.provenance.ruleset_fingerprint:
                        raise RuntimeError("Phase 6Z runtime fingerprint drifted")
                    root_surfaces: list[AlignedAllocationSurface] = []
                    for outcome in native_game.chance_outcomes(native_game.root_state()):
                        scenario = native_game.transition(
                            native_game.root_state(), None, chance_outcome=outcome
                        )
                        completion = native_game.completion(scenario.hypothesis_id or "")
                        variant, allocation_fingerprint = completion_metadata[
                            completion.hypothesis_id
                        ]
                        macro = OfficialNativePlanGame(
                            native_game,
                            scenario,
                            bridge=bridge,
                            tactical_provider=tactical_provider,
                            maximum_tactical_transitions=maximum_tactical_transitions,
                        )
                        if macro.player_to_act(macro.root_state()) != 1:
                            raise RuntimeError(
                                "Phase 6Z captured root is not a player-1 plan decision"
                            )
                        state = macro.root_state()
                        information_set = macro.information_set(state, player=1)
                        belief = macro.public_belief(state, player=1)
                        distribution: ActionDistribution = plan_provider.strategy(
                            information_set, belief
                        )
                        distribution.validate_support(information_set)
                        if tuple(
                            action.action_id for action in information_set.legal_actions
                        ) != OPPONENT_DEVIATION_ACTION_IDS:
                            raise RuntimeError("Phase 6Z plan support drifted")
                        private_fingerprint = (
                            information_set.private_observation_fingerprint
                        )
                        if private_fingerprint is None:
                            raise RuntimeError(
                                "Phase 6Z information set lacks a private digest"
                            )
                        rollout_seeds = tuple(
                            _seed(
                                seed,
                                "aligned-paired-rollout",
                                split,
                                candidate.archetype,
                                rollout,
                            )
                            for rollout in range(rollouts_per_action)
                        )
                        action_values = []
                        for action in information_set.legal_actions:
                            child = macro.transition(state, action)
                            forced_tactical = (
                                child.tactical_transitions - state.tactical_transitions
                            )
                            returns = []
                            depths = []
                            rollout_macro_transitions = 0
                            for rollout_seed in rollout_seeds:
                                value, transitions, terminal_depth = (
                                    _rollout_average_blueprint(
                                        macro,
                                        child,
                                        plan_provider,
                                        value_player=1,
                                        rng=random.Random(rollout_seed),
                                        maximum_macro_transitions=(
                                            maximum_rollout_transitions
                                        ),
                                    )
                                )
                                returns.append(value)
                                depths.append(terminal_depth)
                                rollout_macro_transitions += transitions
                            action_values.append(
                                OpponentPlanActionValue(
                                    action_id=action.action_id,
                                    blueprint_probability=distribution.probability(
                                        action.action_id
                                    ),
                                    terminal_returns=tuple(returns),
                                    empirical_opponent_value=statistics.mean(returns),
                                    rollout_standard_error=(
                                        statistics.stdev(returns)
                                        / math.sqrt(len(returns))
                                    ),
                                    forced_action_tactical_transitions=forced_tactical,
                                    rollout_macro_transitions=(
                                        rollout_macro_transitions
                                    ),
                                    terminal_native_depth_minimum=min(depths),
                                    terminal_native_depth_maximum=max(depths),
                                )
                            )
                        root_id = belief.public_history_fingerprint
                        values = tuple(action_values)
                        surface_id = _fingerprint(
                            {
                                "root": root_id,
                                "completion": completion.fingerprint,
                                "allocation": allocation_fingerprint,
                                "actions": [value.to_dict() for value in values],
                                "rollout_seeds": rollout_seeds,
                            }
                        )
                        root_surfaces.append(
                            AlignedAllocationSurface(
                                surface_id=surface_id,
                                root_id=root_id,
                                split=split,
                                candidate_id_hypothesis_input=candidate.archetype,
                                allocation_variant=variant,
                                completion_fingerprint=completion.fingerprint,
                                hidden_allocation_fingerprint=allocation_fingerprint,
                                source_game_fingerprint=macro.fingerprint,
                                source_state_id=state.state_id,
                                information_set_id=information_set.info_set_id,
                                public_belief_fingerprint=belief.fingerprint,
                                public_history_fingerprint=(
                                    belief.public_history_fingerprint
                                ),
                                private_observation_fingerprint=private_fingerprint,
                                public_state_fingerprint=public_state_fingerprint,
                                public_features=belief.public_features,
                                action_values=values,
                            )
                        )
                        macro_transitions += macro.macro_transition_count
                        tactical_transitions += macro.tactical_transition_count
                    native_transitions += native_game.transition_count
                    public_signatures = {
                        (
                            surface.root_id,
                            surface.public_state_fingerprint,
                            surface.public_history_fingerprint,
                            surface.information_set_id,
                            surface.public_belief_fingerprint,
                            surface.public_features,
                        )
                        for surface in root_surfaces
                    }
                    if len(public_signatures) != 1:
                        raise RuntimeError(
                            "Phase 6Z allocation variants do not share a public root"
                        )
                    first = root_surfaces[0]
                    roots.append(
                        AlignedPublicRoot(
                            root_id=first.root_id,
                            split=split,
                            candidate_id=candidate.archetype,
                            baseline_style=style,
                            requested_player_1_main_ordinal=ordinal,
                            capture_attempt=capture_attempt,
                            captured_turn=captured_turn,
                            live_selection_count=selection_count,
                            public_state_fingerprint=public_state_fingerprint,
                            public_history_fingerprint=(
                                first.public_history_fingerprint
                            ),
                            information_set_id=first.information_set_id,
                            public_belief_fingerprint=(
                                first.public_belief_fingerprint
                            ),
                            public_features=first.public_features,
                            allocation_surface_ids=tuple(
                                surface.surface_id for surface in root_surfaces
                            ),
                        )
                    )
                    surfaces.extend(root_surfaces)
            finally:
                if battle_active:
                    battle_finish()
    if runtime_fingerprint is None:
        raise RuntimeError("Phase 6Z did not execute any native roots")
    provisional = PublicRootAlignedValueDataset(
        runtime_fingerprint=runtime_fingerprint,
        deck_fingerprint=target.fingerprint,
        source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
        source_plan_blueprint_fingerprint=plan_artifact.fingerprint,
        candidate_ids=tactical_artifact.candidate_ids,
        particle_variants_per_root=particle_variants,
        rollouts_per_action=rollouts_per_action,
        native_transition_count=native_transitions,
        macro_transition_count=macro_transitions,
        tactical_transition_count=tactical_transitions,
        roots=tuple(roots),
        surfaces=tuple(surfaces),
        public_root_alignment_verified=True,
        collection_disjoint_roots_verified=(
            len({root.root_id for root in roots}) == len(roots)
        ),
    )
    serialized = provisional.to_dict()
    raw_private = _raw_private_key_count(serialized)
    if raw_private != 0:
        raise RuntimeError("Phase 6Z serialized a forbidden raw private key")
    dataset = provisional
    restored = PublicRootAlignedValueDataset.from_dict(
        json.loads(json.dumps(serialized))
    )
    if restored.fingerprint != dataset.fingerprint:
        raise RuntimeError("Phase 6Z dataset roundtrip drifted")
    model = train_hypothesis_conditioned_value_model(dataset)
    calibration = calibrate_aligned_value_model(dataset, model)
    root_counts = {
        split: sum(root.split == split for root in dataset.roots)
        for split in ALIGNED_VALUE_SPLITS
    }
    claims = (
        ("live_root_capture", "genuine_official_battle_execution"),
        ("descendant_transitions", "genuine_official_search_execution"),
        ("hidden_completion_prior", "supplied_research_abstraction"),
        ("turn_plan_actions", "five_action_research_abstraction"),
        ("value_model", "hypothesis_conditioned_ridge_abstraction"),
        ("acting_player_hand_support", "fixed_within_each_public_root"),
        ("upstream_official_reach", "not_available"),
        ("counterfactual_value_bounds", "not_calibrated"),
        ("safe_resolver", "not_enabled"),
        ("official_strategy_control", "disabled"),
        ("pokemon_strength", "not_claimed"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6AlignedValueEvidence(
        benchmark_id="phase6z-public-root-aligned-values-v1",
        status=(
            "phase6z_aligned_conditional_model_validated_partial"
            if calibration.empirical_conditional_model_gate_passed
            else "phase6z_aligned_substrate_validated_model_rejected_partial"
        ),
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=len(dataset.roots),
        surface_count=len(dataset.surfaces),
        action_target_count=dataset.action_target_count,
        terminal_rollout_count=dataset.terminal_rollout_count,
        native_transition_count=dataset.native_transition_count,
        train_root_count=root_counts["train"],
        calibration_root_count=root_counts["calibration"],
        validation_root_count=root_counts["validation"],
        validation_coverage=calibration.validation_coverage,
        validation_rmse=calibration.validation_rmse,
        validation_zero_baseline_rmse=(
            calibration.validation_zero_baseline_rmse
        ),
        maximum_rollout_standard_error=(
            calibration.maximum_rollout_standard_error
        ),
        public_root_alignment_gate_passed=(
            calibration.public_root_alignment_gate_passed
        ),
        collection_disjoint_gate_passed=(
            calibration.collection_disjoint_gate_passed
        ),
        empirical_conditional_model_gate_passed=(
            calibration.empirical_conditional_model_gate_passed
        ),
        native_counterfactual_values_calibrated=False,
        safe_resolver_enabled=False,
        official_strategy_control_enabled=False,
        claims=claims,
    )
    return Phase6AlignedValueArtifacts(dataset, model, calibration, evidence)


def main() -> None:
    print(run_phase6_aligned_counterfactual_values().evidence.to_json())


if __name__ == "__main__":
    main()
