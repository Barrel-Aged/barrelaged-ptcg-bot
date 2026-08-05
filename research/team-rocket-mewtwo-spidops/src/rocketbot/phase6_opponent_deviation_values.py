"""Phase 6X native opponent-plan deviation-value substrate.

For each supplied opponent hidden-allocation hypothesis, this module reaches an
opponent turn-plan decision, forces each of the five abstract plans, and rolls
out the frozen average-plan blueprint with paired random seeds. The resulting
conditional action returns are useful supervision for a future
hypothesis-conditioned value model. They are not reach-weighted opponent
counterfactual values and cannot enable safe resolving.
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
from typing import Any, Callable, Mapping, Sequence

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .counterfactual_reach import (
    CounterfactualReachTrace,
    sample_first_player_decision_reach,
)
from .game_theory import ActionDistribution, InformationSetActor
from .native_blueprint_features import (
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeBlueprintFeatureBridge,
)
from .native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanBlueprintProvider,
    restore_native_plan_blueprint,
)
from .native_plan_game import (
    NATIVE_PLAN_ACTIONS,
    NativePlanGameState,
    OfficialNativePlanGame,
)
from .native_regret_blueprint import (
    NativeRegretBlueprintArtifact,
    restore_native_regret_blueprint,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
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


PHASE6_OPPONENT_DEVIATION_SCHEMA_VERSION = 1
PHASE6_OPPONENT_DEVIATION_SEED = 6_030
OPPONENT_DEVIATION_PARTICLE_VARIANTS = 2
OPPONENT_DEVIATION_ROLLOUTS_PER_ACTION = 10
OPPONENT_DEVIATION_ACTION_IDS = tuple(action.action_id for action in NATIVE_PLAN_ACTIONS)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big"
    )


def _sample_identifier(
    probabilities: Sequence[tuple[str, float]],
    rng: random.Random,
) -> str:
    draw = rng.random()
    cumulative = 0.0
    for identifier, probability in probabilities:
        cumulative += probability
        if draw <= cumulative + 1e-15:
            return identifier
    return probabilities[-1][0]


@dataclass(frozen=True, slots=True)
class OpponentPlanActionValue:
    action_id: str
    blueprint_probability: float
    terminal_returns: tuple[float, ...]
    empirical_opponent_value: float
    rollout_standard_error: float
    forced_action_tactical_transitions: int
    rollout_macro_transitions: int
    terminal_native_depth_minimum: int
    terminal_native_depth_maximum: int

    def __post_init__(self) -> None:
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("opponent deviation action is outside plan support")
        if not math.isfinite(self.blueprint_probability) or not (
            0 <= self.blueprint_probability <= 1
        ):
            raise ValueError("opponent deviation blueprint probability is invalid")
        if len(self.terminal_returns) < 2 or any(
            not math.isfinite(value) or not -1 <= value <= 1
            for value in self.terminal_returns
        ):
            raise ValueError("opponent deviation terminal returns are invalid")
        if not math.isclose(
            self.empirical_opponent_value,
            statistics.mean(self.terminal_returns),
            abs_tol=1e-12,
        ):
            raise ValueError("opponent deviation mean does not match its returns")
        expected_se = statistics.stdev(self.terminal_returns) / math.sqrt(
            len(self.terminal_returns)
        )
        if not math.isclose(self.rollout_standard_error, expected_se, abs_tol=1e-12):
            raise ValueError("opponent deviation standard error is inconsistent")
        if min(
            self.forced_action_tactical_transitions,
            self.rollout_macro_transitions,
        ) < 0:
            raise ValueError("opponent deviation transition accounting is invalid")
        if min(
            self.terminal_native_depth_minimum,
            self.terminal_native_depth_maximum,
        ) <= 0 or self.terminal_native_depth_minimum > self.terminal_native_depth_maximum:
            raise ValueError("opponent deviation terminal depths are invalid")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["terminal_returns"] = list(self.terminal_returns)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentPlanActionValue":
        return cls(
            action_id=str(raw["action_id"]),
            blueprint_probability=float(raw["blueprint_probability"]),
            terminal_returns=tuple(float(value) for value in raw["terminal_returns"]),
            empirical_opponent_value=float(raw["empirical_opponent_value"]),
            rollout_standard_error=float(raw["rollout_standard_error"]),
            forced_action_tactical_transitions=int(
                raw["forced_action_tactical_transitions"]
            ),
            rollout_macro_transitions=int(raw["rollout_macro_transitions"]),
            terminal_native_depth_minimum=int(raw["terminal_native_depth_minimum"]),
            terminal_native_depth_maximum=int(raw["terminal_native_depth_maximum"]),
        )


@dataclass(frozen=True, slots=True)
class OpponentDeviationSurface:
    surface_id: str
    source_game_fingerprint: str
    source_state_id: str
    information_set_id: str
    public_belief_fingerprint: str
    public_history_fingerprint: str
    private_observation_fingerprint: str
    particle_id: str
    particle_fingerprint: str
    candidate_id_supervision_only: str
    split: str
    variant: int
    public_features: tuple[tuple[str, float], ...]
    action_values: tuple[OpponentPlanActionValue, ...]
    blueprint_opponent_value: float
    best_response_action_id: str
    best_response_opponent_value: float
    empirical_deviation_gain: float
    empirical_action_span: float
    acting_player: int = 1
    value_player: int = 1
    source_descendant_official_engine_executed: bool = True
    conditional_on_sampled_reach: bool = True
    opponent_reach_weighted: bool = False
    exact_counterfactual_values: bool = False
    selected_hidden_truth_identifier_input: bool = False
    raw_private_card_ids_persisted: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "surface_id",
            "source_game_fingerprint",
            "source_state_id",
            "public_belief_fingerprint",
            "public_history_fingerprint",
            "private_observation_fingerprint",
            "particle_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not all(
            (
                self.information_set_id,
                self.particle_id,
                self.candidate_id_supervision_only,
            )
        ):
            raise ValueError("opponent deviation identities are incomplete")
        if self.split not in {"calibration", "validation"} or self.variant not in (0, 1):
            raise ValueError("opponent deviation split/variant is invalid")
        if self.acting_player != 1 or self.value_player != 1:
            raise ValueError("opponent deviation surface must value player 1 at player 1's turn")
        if tuple(name for name, _ in self.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
            raise ValueError("opponent deviation public feature schema mismatch")
        if tuple(item.action_id for item in self.action_values) != OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("opponent deviation action support/order is invalid")
        if not math.isclose(
            sum(item.blueprint_probability for item in self.action_values),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("opponent deviation blueprint distribution is not normalized")
        expected_blueprint = sum(
            item.blueprint_probability * item.empirical_opponent_value
            for item in self.action_values
        )
        if not math.isclose(self.blueprint_opponent_value, expected_blueprint, abs_tol=1e-12):
            raise ValueError("opponent deviation blueprint value is inconsistent")
        best = max(
            self.action_values,
            key=lambda item: (item.empirical_opponent_value, item.action_id),
        )
        if self.best_response_action_id != best.action_id or not math.isclose(
            self.best_response_opponent_value,
            best.empirical_opponent_value,
            abs_tol=1e-12,
        ):
            raise ValueError("opponent deviation best response is inconsistent")
        expected_gain = best.empirical_opponent_value - expected_blueprint
        expected_span = max(item.empirical_opponent_value for item in self.action_values) - min(
            item.empirical_opponent_value for item in self.action_values
        )
        if not math.isclose(self.empirical_deviation_gain, expected_gain, abs_tol=1e-12):
            raise ValueError("opponent deviation gain is inconsistent")
        if not math.isclose(self.empirical_action_span, expected_span, abs_tol=1e-12):
            raise ValueError("opponent deviation action span is inconsistent")
        if self.empirical_deviation_gain < -1e-12 or self.empirical_action_span < 0:
            raise ValueError("opponent deviation empirical metrics are invalid")
        if any(
            (
                not self.source_descendant_official_engine_executed,
                not self.conditional_on_sampled_reach,
                self.opponent_reach_weighted,
                self.exact_counterfactual_values,
                self.selected_hidden_truth_identifier_input,
                self.raw_private_card_ids_persisted,
            )
        ):
            raise ValueError("opponent deviation surface promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def maximum_rollout_standard_error(self) -> float:
        return max(item.rollout_standard_error for item in self.action_values)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OPPONENT_DEVIATION_SCHEMA_VERSION,
            "surface_id": self.surface_id,
            "source": {
                "game_fingerprint": self.source_game_fingerprint,
                "state_id": self.source_state_id,
                "information_set_id": self.information_set_id,
                "public_belief_fingerprint": self.public_belief_fingerprint,
                "public_history_fingerprint": self.public_history_fingerprint,
                "private_observation_fingerprint": self.private_observation_fingerprint,
                "source_descendant_official_engine_executed": (
                    self.source_descendant_official_engine_executed
                ),
            },
            "particle": {
                "particle_id": self.particle_id,
                "particle_fingerprint": self.particle_fingerprint,
                "candidate_id_supervision_only": self.candidate_id_supervision_only,
                "split": self.split,
                "variant": self.variant,
                "selected_hidden_truth_identifier_input": (
                    self.selected_hidden_truth_identifier_input
                ),
            },
            "value_surface": {
                "acting_player": self.acting_player,
                "value_player": self.value_player,
                "actions": [item.to_dict() for item in self.action_values],
                "blueprint_opponent_value": self.blueprint_opponent_value,
                "best_response_action_id": self.best_response_action_id,
                "best_response_opponent_value": self.best_response_opponent_value,
                "empirical_deviation_gain": self.empirical_deviation_gain,
                "empirical_action_span": self.empirical_action_span,
                "conditional_on_sampled_reach": self.conditional_on_sampled_reach,
                "opponent_reach_weighted": self.opponent_reach_weighted,
                "exact_counterfactual_values": self.exact_counterfactual_values,
            },
            "features": {
                "public": [list(item) for item in self.public_features],
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentDeviationSurface":
        source = raw["source"]
        particle = raw["particle"]
        values = raw["value_surface"]
        features = raw["features"]
        result = cls(
            surface_id=str(raw["surface_id"]),
            source_game_fingerprint=str(source["game_fingerprint"]),
            source_state_id=str(source["state_id"]),
            information_set_id=str(source["information_set_id"]),
            public_belief_fingerprint=str(source["public_belief_fingerprint"]),
            public_history_fingerprint=str(source["public_history_fingerprint"]),
            private_observation_fingerprint=str(source["private_observation_fingerprint"]),
            particle_id=str(particle["particle_id"]),
            particle_fingerprint=str(particle["particle_fingerprint"]),
            candidate_id_supervision_only=str(particle["candidate_id_supervision_only"]),
            split=str(particle["split"]),
            variant=int(particle["variant"]),
            public_features=tuple(
                (str(name), float(value)) for name, value in features["public"]
            ),
            action_values=tuple(
                OpponentPlanActionValue.from_dict(item) for item in values["actions"]
            ),
            blueprint_opponent_value=float(values["blueprint_opponent_value"]),
            best_response_action_id=str(values["best_response_action_id"]),
            best_response_opponent_value=float(values["best_response_opponent_value"]),
            empirical_deviation_gain=float(values["empirical_deviation_gain"]),
            empirical_action_span=float(values["empirical_action_span"]),
            acting_player=int(values["acting_player"]),
            value_player=int(values["value_player"]),
            source_descendant_official_engine_executed=bool(
                source["source_descendant_official_engine_executed"]
            ),
            conditional_on_sampled_reach=bool(values["conditional_on_sampled_reach"]),
            opponent_reach_weighted=bool(values["opponent_reach_weighted"]),
            exact_counterfactual_values=bool(values["exact_counterfactual_values"]),
            selected_hidden_truth_identifier_input=bool(
                particle["selected_hidden_truth_identifier_input"]
            ),
            raw_private_card_ids_persisted=bool(features["raw_private_card_ids_persisted"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("opponent deviation surface fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OpponentDeviationDataset:
    runtime_fingerprint: str
    deck_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    candidate_ids: tuple[str, ...]
    particle_family_fingerprint: str
    rollouts_per_action: int
    native_transition_count: int
    macro_transition_count: int
    tactical_transition_count: int
    surfaces: tuple[OpponentDeviationSurface, ...]
    common_random_numbers_across_actions: bool = True
    source_descendants_official_engine_executed: bool = True
    conditional_action_values_only: bool = True
    opponent_reach_weighted: bool = False
    exact_counterfactual_targets: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
            "particle_family_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.candidate_ids or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("opponent deviation candidates must be unique")
        if self.rollouts_per_action < 2 or min(
            self.native_transition_count,
            self.macro_transition_count,
            self.tactical_transition_count,
        ) < 0:
            raise ValueError("opponent deviation dataset accounting is invalid")
        expected = {
            (candidate_id, variant)
            for candidate_id in self.candidate_ids
            for variant in range(OPPONENT_DEVIATION_PARTICLE_VARIANTS)
        }
        observed = {
            (item.candidate_id_supervision_only, item.variant) for item in self.surfaces
        }
        if observed != expected or len(self.surfaces) != len(expected):
            raise ValueError("opponent deviation candidate/variant support is incomplete")
        if len({item.surface_id for item in self.surfaces}) != len(self.surfaces):
            raise ValueError("opponent deviation surface IDs must be unique")
        if len({item.particle_fingerprint for item in self.surfaces}) != len(self.surfaces):
            raise ValueError("opponent deviation particles must be unique")
        if any(
            len(action.terminal_returns) != self.rollouts_per_action
            for surface in self.surfaces
            for action in surface.action_values
        ):
            raise ValueError("opponent deviation rollout count drifted")
        if any(
            (
                not self.common_random_numbers_across_actions,
                not self.source_descendants_official_engine_executed,
                not self.conditional_action_values_only,
                self.opponent_reach_weighted,
                self.exact_counterfactual_targets,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("opponent deviation dataset promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def terminal_rollout_count(self) -> int:
        return len(self.surfaces) * len(OPPONENT_DEVIATION_ACTION_IDS) * self.rollouts_per_action

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OPPONENT_DEVIATION_SCHEMA_VERSION,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "source_plan_blueprint_fingerprint": self.source_plan_blueprint_fingerprint,
            "candidate_ids": list(self.candidate_ids),
            "particle_family_fingerprint": self.particle_family_fingerprint,
            "rollouts_per_action": self.rollouts_per_action,
            "accounting": {
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "tactical_transition_count": self.tactical_transition_count,
                "terminal_rollout_count": self.terminal_rollout_count,
            },
            "surfaces": [item.to_dict() for item in self.surfaces],
            "claims": {
                "common_random_numbers_across_actions": (
                    self.common_random_numbers_across_actions
                ),
                "source_descendants_official_engine_executed": (
                    self.source_descendants_official_engine_executed
                ),
                "conditional_action_values_only": self.conditional_action_values_only,
                "opponent_reach_weighted": self.opponent_reach_weighted,
                "exact_counterfactual_targets": self.exact_counterfactual_targets,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentDeviationDataset":
        accounting = raw["accounting"]
        claims = raw["claims"]
        result = cls(
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(raw["source_plan_blueprint_fingerprint"]),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            particle_family_fingerprint=str(raw["particle_family_fingerprint"]),
            rollouts_per_action=int(raw["rollouts_per_action"]),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            tactical_transition_count=int(accounting["tactical_transition_count"]),
            surfaces=tuple(OpponentDeviationSurface.from_dict(item) for item in raw["surfaces"]),
            common_random_numbers_across_actions=bool(
                claims["common_random_numbers_across_actions"]
            ),
            source_descendants_official_engine_executed=bool(
                claims["source_descendants_official_engine_executed"]
            ),
            conditional_action_values_only=bool(claims["conditional_action_values_only"]),
            opponent_reach_weighted=bool(claims["opponent_reach_weighted"]),
            exact_counterfactual_targets=bool(claims["exact_counterfactual_targets"]),
            raw_private_card_ids_persisted=int(claims["raw_private_card_ids_persisted"]),
        )
        if accounting.get("terminal_rollout_count") != result.terminal_rollout_count:
            raise ValueError("opponent deviation terminal rollout accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("opponent deviation dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class OpponentDeviationSplitMetrics:
    split: str
    surfaces: int
    candidate_strata: int
    action_targets: int
    nontrivial_action_surface_fraction: float
    mean_action_span: float
    maximum_action_span: float
    mean_empirical_deviation_gain: float
    maximum_empirical_deviation_gain: float
    maximum_rollout_standard_error: float
    blueprint_top_action_matches_empirical_best_fraction: float
    best_response_action_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.split not in {"calibration", "validation"}:
            raise ValueError("opponent deviation metric split is invalid")
        if min(self.surfaces, self.candidate_strata, self.action_targets) <= 0:
            raise ValueError("opponent deviation metric accounting is invalid")
        if tuple(name for name, _ in self.best_response_action_counts) != (
            OPPONENT_DEVIATION_ACTION_IDS
        ):
            raise ValueError("opponent deviation best-action counts have invalid support")
        if sum(value for _, value in self.best_response_action_counts) != self.surfaces:
            raise ValueError("opponent deviation best-action counts do not sum")
        for field_name in (
            "nontrivial_action_surface_fraction",
            "mean_action_span",
            "maximum_action_span",
            "mean_empirical_deviation_gain",
            "maximum_empirical_deviation_gain",
            "maximum_rollout_standard_error",
            "blueprint_top_action_matches_empirical_best_fraction",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if any(
            value > 1
            for value in (
                self.nontrivial_action_surface_fraction,
                self.blueprint_top_action_matches_empirical_best_fraction,
            )
        ):
            raise ValueError("opponent deviation fraction exceeds one")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["best_response_action_counts"] = [
            [name, count] for name, count in self.best_response_action_counts
        ]
        return result


@dataclass(frozen=True, slots=True)
class OpponentDeviationCalibration:
    dataset_fingerprint: str
    minimum_candidate_strata: int
    minimum_rollouts_per_action: int
    maximum_rollout_standard_error_threshold: float
    minimum_nontrivial_action_span: float
    minimum_nontrivial_surface_fraction: float
    calibration: OpponentDeviationSplitMetrics
    validation: OpponentDeviationSplitMetrics
    conditional_deviation_substrate_gate_passed: bool
    opponent_counterfactual_value_gate_passed: bool = False
    safe_resolver_compatible: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        if min(self.minimum_candidate_strata, self.minimum_rollouts_per_action) <= 0:
            raise ValueError("opponent deviation calibration minimums are invalid")
        if min(
            self.maximum_rollout_standard_error_threshold,
            self.minimum_nontrivial_action_span,
            self.minimum_nontrivial_surface_fraction,
        ) < 0 or self.minimum_nontrivial_surface_fraction > 1:
            raise ValueError("opponent deviation calibration thresholds are invalid")
        expected_gate = all(
            (
                self.calibration.candidate_strata >= self.minimum_candidate_strata,
                self.validation.candidate_strata >= self.minimum_candidate_strata,
                self.calibration.maximum_rollout_standard_error
                <= self.maximum_rollout_standard_error_threshold,
                self.validation.maximum_rollout_standard_error
                <= self.maximum_rollout_standard_error_threshold,
                self.calibration.nontrivial_action_surface_fraction
                >= self.minimum_nontrivial_surface_fraction,
                self.validation.nontrivial_action_surface_fraction
                >= self.minimum_nontrivial_surface_fraction,
            )
        )
        if self.conditional_deviation_substrate_gate_passed != expected_gate:
            raise ValueError("opponent deviation substrate gate is inconsistent")
        if self.opponent_counterfactual_value_gate_passed or self.safe_resolver_compatible:
            raise ValueError("conditional deviation targets cannot promote safety")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OPPONENT_DEVIATION_SCHEMA_VERSION,
            "dataset_fingerprint": self.dataset_fingerprint,
            "thresholds": {
                "minimum_candidate_strata": self.minimum_candidate_strata,
                "minimum_rollouts_per_action": self.minimum_rollouts_per_action,
                "maximum_rollout_standard_error": (
                    self.maximum_rollout_standard_error_threshold
                ),
                "minimum_nontrivial_action_span": self.minimum_nontrivial_action_span,
                "minimum_nontrivial_surface_fraction": (
                    self.minimum_nontrivial_surface_fraction
                ),
            },
            "calibration": self.calibration.to_dict(),
            "validation": self.validation.to_dict(),
            "gates": {
                "conditional_deviation_substrate": (
                    self.conditional_deviation_substrate_gate_passed
                ),
                "opponent_counterfactual_value": (
                    self.opponent_counterfactual_value_gate_passed
                ),
                "safe_resolver_compatible": self.safe_resolver_compatible,
            },
            "scope": {
                "conditional_on_sampled_reach": True,
                "opponent_reach_weighted": False,
                "exact_counterfactual_value": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentDeviationCalibration":
        thresholds = raw["thresholds"]
        gates = raw["gates"]

        def split(item: Mapping[str, Any]) -> OpponentDeviationSplitMetrics:
            values = dict(item)
            values["best_response_action_counts"] = tuple(
                (str(name), int(count))
                for name, count in item["best_response_action_counts"]
            )
            return OpponentDeviationSplitMetrics(**values)

        result = cls(
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            minimum_candidate_strata=int(thresholds["minimum_candidate_strata"]),
            minimum_rollouts_per_action=int(thresholds["minimum_rollouts_per_action"]),
            maximum_rollout_standard_error_threshold=float(
                thresholds["maximum_rollout_standard_error"]
            ),
            minimum_nontrivial_action_span=float(
                thresholds["minimum_nontrivial_action_span"]
            ),
            minimum_nontrivial_surface_fraction=float(
                thresholds["minimum_nontrivial_surface_fraction"]
            ),
            calibration=split(raw["calibration"]),
            validation=split(raw["validation"]),
            conditional_deviation_substrate_gate_passed=bool(
                gates["conditional_deviation_substrate"]
            ),
            opponent_counterfactual_value_gate_passed=bool(
                gates["opponent_counterfactual_value"]
            ),
            safe_resolver_compatible=bool(gates["safe_resolver_compatible"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("opponent deviation calibration fingerprint mismatch")
        return result


def _split_metrics(
    surfaces: Sequence[OpponentDeviationSurface],
    *,
    split: str,
    minimum_nontrivial_action_span: float,
) -> OpponentDeviationSplitMetrics:
    selected = tuple(item for item in surfaces if item.split == split)
    if not selected:
        raise ValueError("opponent deviation split is empty")
    counts = {name: 0 for name in OPPONENT_DEVIATION_ACTION_IDS}
    for item in selected:
        counts[item.best_response_action_id] += 1
    blueprint_matches = []
    for item in selected:
        blueprint_top = max(
            item.action_values,
            key=lambda action: (action.blueprint_probability, action.action_id),
        ).action_id
        blueprint_matches.append(blueprint_top == item.best_response_action_id)
    return OpponentDeviationSplitMetrics(
        split=split,
        surfaces=len(selected),
        candidate_strata=len(
            {item.candidate_id_supervision_only for item in selected}
        ),
        action_targets=len(selected) * len(OPPONENT_DEVIATION_ACTION_IDS),
        nontrivial_action_surface_fraction=sum(
            item.empirical_action_span >= minimum_nontrivial_action_span
            for item in selected
        )
        / len(selected),
        mean_action_span=statistics.mean(item.empirical_action_span for item in selected),
        maximum_action_span=max(item.empirical_action_span for item in selected),
        mean_empirical_deviation_gain=statistics.mean(
            item.empirical_deviation_gain for item in selected
        ),
        maximum_empirical_deviation_gain=max(
            item.empirical_deviation_gain for item in selected
        ),
        maximum_rollout_standard_error=max(
            item.maximum_rollout_standard_error for item in selected
        ),
        blueprint_top_action_matches_empirical_best_fraction=(
            sum(blueprint_matches) / len(blueprint_matches)
        ),
        best_response_action_counts=tuple(
            (name, counts[name]) for name in OPPONENT_DEVIATION_ACTION_IDS
        ),
    )


def calibrate_opponent_deviation_dataset(
    dataset: OpponentDeviationDataset,
    *,
    minimum_candidate_strata: int = 6,
    minimum_rollouts_per_action: int = OPPONENT_DEVIATION_ROLLOUTS_PER_ACTION,
    maximum_rollout_standard_error: float = 0.35,
    minimum_nontrivial_action_span: float = 0.10,
    minimum_nontrivial_surface_fraction: float = 0.50,
) -> OpponentDeviationCalibration:
    if dataset.rollouts_per_action < minimum_rollouts_per_action:
        raise ValueError("opponent deviation dataset has too few rollouts per action")
    calibration = _split_metrics(
        dataset.surfaces,
        split="calibration",
        minimum_nontrivial_action_span=minimum_nontrivial_action_span,
    )
    validation = _split_metrics(
        dataset.surfaces,
        split="validation",
        minimum_nontrivial_action_span=minimum_nontrivial_action_span,
    )
    gate = all(
        (
            calibration.candidate_strata >= minimum_candidate_strata,
            validation.candidate_strata >= minimum_candidate_strata,
            calibration.maximum_rollout_standard_error
            <= maximum_rollout_standard_error,
            validation.maximum_rollout_standard_error <= maximum_rollout_standard_error,
            calibration.nontrivial_action_surface_fraction
            >= minimum_nontrivial_surface_fraction,
            validation.nontrivial_action_surface_fraction
            >= minimum_nontrivial_surface_fraction,
        )
    )
    return OpponentDeviationCalibration(
        dataset_fingerprint=dataset.fingerprint,
        minimum_candidate_strata=minimum_candidate_strata,
        minimum_rollouts_per_action=minimum_rollouts_per_action,
        maximum_rollout_standard_error_threshold=maximum_rollout_standard_error,
        minimum_nontrivial_action_span=minimum_nontrivial_action_span,
        minimum_nontrivial_surface_fraction=minimum_nontrivial_surface_fraction,
        calibration=calibration,
        validation=validation,
        conditional_deviation_substrate_gate_passed=gate,
    )


@dataclass(frozen=True, slots=True)
class Phase6OpponentDeviationEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    calibration_fingerprint: str
    surface_count: int
    action_target_count: int
    terminal_rollout_count: int
    native_transition_count: int
    candidate_strata: int
    rollouts_per_action: int
    validation_nontrivial_action_surface_fraction: float
    validation_mean_action_span: float
    validation_mean_empirical_deviation_gain: float
    validation_maximum_rollout_standard_error: float
    validation_blueprint_top_action_matches_empirical_best_fraction: float
    conditional_deviation_substrate_gate_passed: bool
    opponent_counterfactual_values_calibrated: bool
    public_belief_calibrated: bool
    safe_resolver_enabled: bool
    official_strategy_control_enabled: bool
    raw_private_card_ids_persisted: int
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _validate_digest(self.dataset_fingerprint, "dataset_fingerprint")
        _validate_digest(self.calibration_fingerprint, "calibration_fingerprint")
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6X evidence identity is missing")
        if min(
            self.surface_count,
            self.action_target_count,
            self.terminal_rollout_count,
            self.native_transition_count,
            self.candidate_strata,
            self.rollouts_per_action,
        ) <= 0:
            raise ValueError("Phase 6X evidence accounting is invalid")
        if any(
            (
                self.opponent_counterfactual_values_calibrated,
                self.public_belief_calibrated,
                self.safe_resolver_enabled,
                self.official_strategy_control_enabled,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6X evidence cannot enable safety or private IDs")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OPPONENT_DEVIATION_SCHEMA_VERSION,
            **asdict(self),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6X opponent deviation-value evidence

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Hypothesis-conditioned surfaces | {self.surface_count} |
| Forced plan-action targets | {self.action_target_count} |
| Genuine terminal continuations | {self.terminal_rollout_count} |
| Genuine native transitions | {self.native_transition_count} |
| Rollouts per forced action | {self.rollouts_per_action} |
| Validation nontrivial surface fraction | {self.validation_nontrivial_action_surface_fraction:.3%} |
| Validation mean action span | {self.validation_mean_action_span:.6f} |
| Validation mean empirical deviation gain | {self.validation_mean_empirical_deviation_gain:.6f} |
| Validation maximum rollout SE | {self.validation_maximum_rollout_standard_error:.6f} |
| Blueprint-top / empirical-best agreement | {self.validation_blueprint_top_action_matches_empirical_best_fraction:.3%} |
| Conditional deviation substrate gate | **{str(self.conditional_deviation_substrate_gate_passed).lower()}** |
| Opponent counterfactual values calibrated | **false** |

The forced plans and future blueprint rollouts are research abstractions over
genuine native descendants. Values are conditional on sampled reach and are not
opponent-reach-weighted counterfactual values. No resolver or official strategy
control is enabled.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6OpponentDeviationEvidence":
        fields = {
            name: raw[name]
            for name in cls.__dataclass_fields__
            if name != "claims"
        }
        result = cls(
            **fields,
            claims=tuple((str(key), str(value)) for key, value in raw["claims"].items()),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6X evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6OpponentDeviationArtifacts:
    dataset: OpponentDeviationDataset
    calibration: OpponentDeviationCalibration
    evidence: Phase6OpponentDeviationEvidence


def _deviation_particles(
    observation: Mapping[str, Any],
    *,
    target: KaggleDeckProfile,
    candidates: Sequence[KaggleDeckProfile],
    basic_card_ids: Sequence[int],
    variants: int,
    seed: int,
) -> tuple[tuple[HiddenZoneCompletion, ...], dict[str, tuple[str, int, str]]]:
    if variants != OPPONENT_DEVIATION_PARTICLE_VARIANTS:
        raise ValueError("Phase 6X requires exactly one calibration and validation variant")
    probability = 1.0 / (len(candidates) * variants)
    particles = []
    metadata: dict[str, tuple[str, int, str]] = {}
    for candidate in candidates:
        for variant in range(variants):
            split = "calibration" if variant == 0 else "validation"
            identifier = f"{candidate.archetype}:opponent-deviation-{variant}:{split}"
            particle = sample_hidden_zone_completion(
                observation,
                hypothesis_id=identifier,
                probability=probability,
                allocation_seed=_seed(seed, candidate.fingerprint, variant),
                your_candidate_id=target.archetype,
                opponent_candidate_id=candidate.archetype,
                your_archetype_id=target.archetype,
                opponent_archetype_id=candidate.archetype,
                your_style_id="unknown",
                opponent_style_id="unknown",
                your_card_ids=target.card_ids,
                opponent_card_ids=candidate.card_ids,
                your_deck_fingerprint=target.fingerprint,
                opponent_deck_fingerprint=candidate.fingerprint,
                basic_card_ids=basic_card_ids,
                source=(
                    "Phase 6X public-consistent hidden allocation for conditional "
                    "opponent plan-deviation values"
                ),
            )
            particles.append(particle)
            metadata[identifier] = (candidate.archetype, variant, split)
    if len({item.fingerprint for item in particles}) != len(particles):
        raise RuntimeError("Phase 6X allocation particles are not distinct")
    return tuple(particles), metadata


def _sample_chance(game: OfficialNativePlanGame, state: NativePlanGameState, rng):
    outcomes = game.chance_outcomes(state)
    selected = _sample_identifier(
        tuple((item.outcome_id, item.probability) for item in outcomes),
        rng,
    )
    return next(item for item in outcomes if item.outcome_id == selected)


def _average_plan_action(
    game: OfficialNativePlanGame,
    state: NativePlanGameState,
    provider: NativePlanBlueprintProvider,
    rng: random.Random,
):
    player = game.player_to_act(state)
    if player not in (0, 1):
        raise RuntimeError("average-plan sampling state lacks a player")
    information_set = game.information_set(state, player=player)
    belief = game.public_belief(state, player=player)
    distribution = provider.strategy(information_set, belief)
    distribution.validate_support(information_set)
    selected = _sample_identifier(
        tuple(
            (action.action_id, distribution.probability(action.action_id))
            for action in information_set.legal_actions
        ),
        rng,
    )
    return next(
        action for action in information_set.legal_actions if action.action_id == selected
    )


def _reach_opponent_decision(
    game: OfficialNativePlanGame,
    provider: NativePlanBlueprintProvider,
    *,
    rng: random.Random,
    maximum_macro_transitions: int,
    redaction_salt: str,
) -> tuple[NativePlanGameState, int, CounterfactualReachTrace]:
    state, trace = sample_first_player_decision_reach(
        game,
        provider,
        target_player=1,
        rng=rng,
        maximum_transitions=maximum_macro_transitions,
        redaction_salt=redaction_salt,
        source_descendants_official_engine_executed=True,
    )
    return state, len(trace.steps), trace


def _rollout_average_blueprint(
    game: OfficialNativePlanGame,
    start: NativePlanGameState,
    provider: NativePlanBlueprintProvider,
    *,
    value_player: int,
    rng: random.Random,
    maximum_macro_transitions: int,
) -> tuple[float, int, int]:
    state = start
    transitions = 0
    while not state.terminal:
        if transitions >= maximum_macro_transitions:
            raise RuntimeError("Phase 6X terminal continuation exceeded its budget")
        if state.actor is InformationSetActor.CHANCE:
            state = game.transition(
                state,
                None,
                chance_outcome=_sample_chance(game, state, rng),
            )
        else:
            state = game.transition(
                state,
                _average_plan_action(game, state, provider, rng),
            )
        transitions += 1
    return game.terminal_utility(state, value_player), transitions, state.native_state.depth


def run_phase6_opponent_deviation_values(
    *,
    project_root: str | Path | None = None,
    source_tactical_blueprint_path: str | Path | None = None,
    source_plan_blueprint_path: str | Path | None = None,
    source_phase6p_evidence_path: str | Path | None = None,
    particle_variants: int = OPPONENT_DEVIATION_PARTICLE_VARIANTS,
    rollouts_per_action: int = OPPONENT_DEVIATION_ROLLOUTS_PER_ACTION,
    maximum_reach_transitions: int = 16,
    maximum_rollout_transitions: int = 512,
    maximum_tactical_transitions: int = 512,
    seed: int = PHASE6_OPPONENT_DEVIATION_SEED,
    reach_observer: Callable[
        [OpponentDeviationSurface, CounterfactualReachTrace, float], None
    ]
    | None = None,
) -> Phase6OpponentDeviationArtifacts:
    if min(
        particle_variants,
        rollouts_per_action,
        maximum_reach_transitions,
        maximum_rollout_transitions,
        maximum_tactical_transitions,
    ) <= 0:
        raise ValueError("Phase 6X limits must be positive")
    if particle_variants != OPPONENT_DEVIATION_PARTICLE_VARIANTS:
        raise ValueError("Phase 6X requires two allocation variants")
    if rollouts_per_action < 2:
        raise ValueError("Phase 6X requires repeated forced-action rollouts")
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
    plan_provider = restore_native_plan_blueprint(plan_artifact)
    phase6p = Phase6NativePlanIterationsEvidence.from_dict(
        json.loads(phase6p_path.read_text(encoding="utf-8"))
    )
    if phase6p.final_model_fingerprint != plan_artifact.fingerprint:
        raise ValueError("Phase 6X plan model/evidence mismatch")
    if plan_artifact.source_tactical_blueprint_fingerprint != tactical_artifact.fingerprint:
        raise ValueError("Phase 6X plan/tactical provenance mismatch")
    bridge = NativeBlueprintFeatureBridge()
    runtime_text = str(runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_start  # type: ignore[import-not-found]

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
        raise ValueError("Phase 6X candidate catalog/native profiles mismatch")
    candidates = tuple(profile_by_id[name] for name in tactical_artifact.candidate_ids)
    if tactical_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6X tactical blueprint/deck mismatch")
    if plan_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6X plan blueprint/deck mismatch")
    observation, start = battle_start(list(target.card_ids), list(candidates[0].card_ids))
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6X decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        particles, metadata = _deviation_particles(
            observation,
            target=target,
            candidates=candidates,
            basic_card_ids=_runtime_basics(runtime),
            variants=particle_variants,
            seed=seed,
        )
        particle_family_fingerprint = _fingerprint(
            {
                "particles": [item.fingerprint for item in particles],
                "metadata": metadata,
                "conditional_action_values": True,
                "opponent_reach_weighted": False,
            }
        )
        with OfficialNativeSearchGame(
            observation,
            particles,
            runtime_directory=runtime,
        ) as native_game:
            surfaces = []
            macro_games = []
            for particle_index, outcome in enumerate(
                native_game.chance_outcomes(native_game.root_state())
            ):
                scenario = native_game.transition(
                    native_game.root_state(),
                    None,
                    chance_outcome=outcome,
                )
                completion = native_game.completion(scenario.hypothesis_id or "")
                candidate_id, variant, split = metadata[completion.hypothesis_id]
                macro = OfficialNativePlanGame(
                    native_game,
                    scenario,
                    bridge=bridge,
                    tactical_provider=tactical_provider,
                    maximum_tactical_transitions=maximum_tactical_transitions,
                )
                macro_games.append(macro)
                opponent_state, reach_transitions, reach_trace = _reach_opponent_decision(
                    macro,
                    plan_provider,
                    rng=random.Random(_seed(seed, "reach", particle_index)),
                    maximum_macro_transitions=maximum_reach_transitions,
                    redaction_salt=_fingerprint(
                        {
                            "seed": seed,
                            "particle_index": particle_index,
                            "particle": completion.fingerprint,
                        }
                    ),
                )
                information_set = macro.information_set(opponent_state, player=1)
                belief = macro.public_belief(opponent_state, player=1)
                distribution: ActionDistribution = plan_provider.strategy(
                    information_set,
                    belief,
                )
                distribution.validate_support(information_set)
                if tuple(action.action_id for action in information_set.legal_actions) != (
                    OPPONENT_DEVIATION_ACTION_IDS
                ):
                    raise RuntimeError("Phase 6X opponent plan support drifted")
                private_fingerprint = information_set.private_observation_fingerprint
                if private_fingerprint is None:
                    raise RuntimeError("Phase 6X opponent information set lacks private digest")
                action_values = []
                rollout_seeds = tuple(
                    _seed(seed, "paired-rollout", particle_index, rollout)
                    for rollout in range(rollouts_per_action)
                )
                for action in information_set.legal_actions:
                    child = macro.transition(opponent_state, action)
                    forced_tactical = (
                        child.tactical_transitions - opponent_state.tactical_transitions
                    )
                    returns = []
                    depths = []
                    rollout_transitions = 0
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
                        rollout_transitions += transitions
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
                            rollout_macro_transitions=rollout_transitions,
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
                    key=lambda item: (item.empirical_opponent_value, item.action_id),
                )
                span = max(item.empirical_opponent_value for item in values) - min(
                    item.empirical_opponent_value for item in values
                )
                surface_id = _fingerprint(
                    {
                        "state": opponent_state.state_id,
                        "particle": completion.fingerprint,
                        "blueprint": distribution.probabilities,
                        "actions": [item.to_dict() for item in values],
                        "rollout_seeds": rollout_seeds,
                        "reach_transitions": reach_transitions,
                    }
                )
                surface = OpponentDeviationSurface(
                    surface_id=surface_id,
                    source_game_fingerprint=macro.fingerprint,
                    source_state_id=opponent_state.state_id,
                    information_set_id=information_set.info_set_id,
                    public_belief_fingerprint=belief.fingerprint,
                    public_history_fingerprint=belief.public_history_fingerprint,
                    private_observation_fingerprint=private_fingerprint,
                    particle_id=completion.hypothesis_id,
                    particle_fingerprint=completion.fingerprint,
                    candidate_id_supervision_only=candidate_id,
                    split=split,
                    variant=variant,
                    public_features=belief.public_features,
                    action_values=values,
                    blueprint_opponent_value=blueprint_value,
                    best_response_action_id=best.action_id,
                    best_response_opponent_value=best.empirical_opponent_value,
                    empirical_deviation_gain=(
                        best.empirical_opponent_value - blueprint_value
                    ),
                    empirical_action_span=span,
                )
                surfaces.append(surface)
                if reach_observer is not None:
                    reach_observer(surface, reach_trace, completion.probability)
            provisional = OpponentDeviationDataset(
                runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                deck_fingerprint=target.fingerprint,
                source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
                source_plan_blueprint_fingerprint=plan_artifact.fingerprint,
                candidate_ids=tactical_artifact.candidate_ids,
                particle_family_fingerprint=particle_family_fingerprint,
                rollouts_per_action=rollouts_per_action,
                native_transition_count=native_game.transition_count,
                macro_transition_count=sum(game.macro_transition_count for game in macro_games),
                tactical_transition_count=sum(
                    game.tactical_transition_count for game in macro_games
                ),
                surfaces=tuple(surfaces),
            )
            serialized = provisional.to_dict()
            raw_private = _raw_private_key_count(serialized)
            if raw_private != 0:
                raise RuntimeError("Phase 6X serialized a forbidden raw private key")
            dataset = provisional
            restored = OpponentDeviationDataset.from_dict(
                json.loads(json.dumps(serialized))
            )
            if restored.fingerprint != dataset.fingerprint:
                raise RuntimeError("Phase 6X dataset roundtrip drifted")
            calibration = calibrate_opponent_deviation_dataset(dataset)
            claims = (
                ("source_descendants", "genuine_official_search_execution"),
                ("forced_plan_actions", "research_abstraction"),
                ("future_strategy", "frozen_average_plan_blueprint_abstraction"),
                ("hidden_allocations", "supplied_public_consistent_particles"),
                ("paired_rollout_seeds", "common_random_numbers_across_actions"),
                ("value_target", "sampled_reach_conditional_not_counterfactual"),
                (
                    "conditional_deviation_substrate_gate",
                    str(calibration.conditional_deviation_substrate_gate_passed).lower(),
                ),
                ("opponent_counterfactual_value_bounds", "not_built"),
                ("safe_resolver", "not_enabled"),
                ("official_strategy_control", "disabled"),
                ("pokemon_strength", "not_claimed"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            evidence = Phase6OpponentDeviationEvidence(
                benchmark_id="phase6x-native-opponent-deviation-values-v1",
                status=(
                    "phase6x_conditional_deviation_substrate_validated_partial"
                    if calibration.conditional_deviation_substrate_gate_passed
                    else "phase6x_conditional_deviation_substrate_rejected_partial"
                ),
                dataset_fingerprint=dataset.fingerprint,
                calibration_fingerprint=calibration.fingerprint,
                surface_count=len(dataset.surfaces),
                action_target_count=(
                    len(dataset.surfaces) * len(OPPONENT_DEVIATION_ACTION_IDS)
                ),
                terminal_rollout_count=dataset.terminal_rollout_count,
                native_transition_count=dataset.native_transition_count,
                candidate_strata=len(dataset.candidate_ids),
                rollouts_per_action=dataset.rollouts_per_action,
                validation_nontrivial_action_surface_fraction=(
                    calibration.validation.nontrivial_action_surface_fraction
                ),
                validation_mean_action_span=calibration.validation.mean_action_span,
                validation_mean_empirical_deviation_gain=(
                    calibration.validation.mean_empirical_deviation_gain
                ),
                validation_maximum_rollout_standard_error=(
                    calibration.validation.maximum_rollout_standard_error
                ),
                validation_blueprint_top_action_matches_empirical_best_fraction=(
                    calibration.validation.blueprint_top_action_matches_empirical_best_fraction
                ),
                conditional_deviation_substrate_gate_passed=(
                    calibration.conditional_deviation_substrate_gate_passed
                ),
                opponent_counterfactual_values_calibrated=False,
                public_belief_calibrated=False,
                safe_resolver_enabled=False,
                official_strategy_control_enabled=False,
                raw_private_card_ids_persisted=raw_private,
                claims=claims,
            )
            return Phase6OpponentDeviationArtifacts(dataset, calibration, evidence)
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_opponent_deviation_values().evidence.to_json())


if __name__ == "__main__":
    main()
