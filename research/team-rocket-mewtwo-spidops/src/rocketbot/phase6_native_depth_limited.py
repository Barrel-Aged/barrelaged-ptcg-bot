"""Phase 6R: depth-limited native turn-plan variance-control evidence.

This module deliberately stops before counterfactual-value calibration.  It
first checks the cutoff wrapper on the exactly enumerable ``RocketPrizeDuel``.
It then runs short-horizon outcome-sampling trajectories whose descendants are
executed by the licensed native engine, but whose plan actions, hidden-state
prior, cutoff values, and learning targets remain research abstractions.

The live cutoff value is a clamped Phase 6K scalar with the full [-1, 1]
utility interval and zero confidence.  Consequently this artifact cannot
enable counterfactual bounds, safe resolving, exploitation, official strategy
control, or any performance claim.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Mapping

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .depth_limited_game import (
    DepthLimitedInformationSetGame,
    ExactProfileLeafValueProvider,
)
from .exact_game import RocketPrizeDuel, analytic_equilibrium_profile
from .game_theory import InformationSetActor
from .native_blueprint_features import (
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeBlueprintFeatureBridge,
)
from .native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanBlueprintProvider,
    restore_native_plan_blueprint,
)
from .native_plan_game import (
    NATIVE_PLAN_ACTION_FEATURE_NAMES,
    NATIVE_PLAN_ACTIONS,
    OfficialNativePlanGame,
)
from .native_plan_leaf import UncalibratedNativePlanLeafValueProvider
from .native_regret_blueprint import (
    NativeRegretBlueprintArtifact,
    restore_native_regret_blueprint,
)
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    rocket_compatibility_profile,
)
from .official_search_game import OfficialNativeSearchGame
from .outcome_sampling import (
    OutcomeRegretReservoir,
    OutcomeSamplingConfig,
    OutcomeSamplingRegretSample,
    OutcomeSamplingStrategySample,
    OutcomeSamplingTraversal,
    OutcomeStrategyReservoir,
)
from .outcome_sampling_reference import audit_outcome_sampling_expectation
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _completions, _runtime_basics
from .phase6_native_plan_iterations import Phase6NativePlanIterationsEvidence


PHASE6_NATIVE_DEPTH_LIMITED_SCHEMA_VERSION = 1
PHASE6_NATIVE_DEPTH_LIMITED_SEED = 6_024


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


def _sample_feature_contract(
    sample: OutcomeSamplingRegretSample | OutcomeSamplingStrategySample,
) -> None:
    """Validate plan feature schemas while requiring abstract target provenance."""

    if tuple(name for name, _ in sample.public_features) != NATIVE_PUBLIC_FEATURE_NAMES:
        raise ValueError("depth-limited public feature schema mismatch")
    if tuple(name for name, _ in sample.private_features) != NATIVE_PRIVATE_FEATURE_NAMES:
        raise ValueError("depth-limited private feature schema mismatch")
    expected_ids = tuple(action.action_id for action in NATIVE_PLAN_ACTIONS)
    if tuple(action_id for action_id, _ in sample.action_features) != expected_ids:
        raise ValueError("depth-limited plan action support mismatch")
    if any(
        tuple(name for name, _ in features) != NATIVE_PLAN_ACTION_FEATURE_NAMES
        for _, features in sample.action_features
    ):
        raise ValueError("depth-limited plan action feature schema mismatch")
    if sample.source_official_engine_executed:
        raise ValueError("cutoff-valued samples cannot claim official terminal execution")
    if sample.hidden_completion_prior_official:
        raise ValueError("depth-limited samples cannot promote the supplied prior")


@dataclass(frozen=True, slots=True)
class DepthLimitedExactAuditEvidence:
    maximum_source_depth_delta: int
    traversals: int
    audited_regret_entries: int
    maximum_absolute_error: float
    full_reference_fingerprint: str
    exact_leaf_count: int

    def __post_init__(self) -> None:
        if min(
            self.maximum_source_depth_delta,
            self.traversals,
            self.audited_regret_entries,
            self.exact_leaf_count,
        ) <= 0:
            raise ValueError("depth-limited exact audit accounting must be positive")
        if self.maximum_absolute_error < 0 or not math.isfinite(
            self.maximum_absolute_error
        ):
            raise ValueError("depth-limited exact audit error is invalid")
        _validate_digest(self.full_reference_fingerprint, "full_reference_fingerprint")


def _enumerated_wrapper_paths(
    game: DepthLimitedInformationSetGame,
    profile,
    *,
    traverser: int,
    exploration: float,
) -> tuple[tuple[float, dict[str, str], dict[str, str]], ...]:
    """Enumerate sample paths using wrapper IDs, including cutoff terminals."""

    result: list[tuple[float, dict[str, str], dict[str, str]]] = []

    def visit(state, probability, forced_chance, forced_actions) -> None:
        if state.terminal:
            result.append((probability, forced_chance, forced_actions))
            return
        if state.actor is InformationSetActor.CHANCE:
            for outcome in game.chance_outcomes(state):
                next_chance = dict(forced_chance)
                next_chance[state.state_id] = outcome.outcome_id
                visit(
                    game.transition(state, None, chance_outcome=outcome),
                    probability * outcome.probability,
                    next_chance,
                    forced_actions,
                )
            return
        player = game.player_to_act(state)
        if player not in (0, 1):
            raise RuntimeError("enumerated wrapper decision lacks a player")
        information_set = game.information_set(state, player=player)
        current = profile.strategy(information_set)
        uniform = 1.0 / len(information_set.legal_actions)
        for action in information_set.legal_actions:
            current_probability = current.probability(action.action_id)
            sample_probability = (
                (1.0 - exploration) * current_probability + exploration * uniform
                if player == traverser
                else current_probability
            )
            if sample_probability <= 0:
                continue
            previous = forced_actions.get(information_set.info_set_id)
            if previous is not None and previous != action.action_id:
                raise RuntimeError("one exact wrapper path revisits an information set")
            next_actions = dict(forced_actions)
            next_actions[information_set.info_set_id] = action.action_id
            visit(
                game.transition(state, action),
                probability * sample_probability,
                forced_chance,
                next_actions,
            )

    visit(game.root_state(), 1.0, {}, {})
    if not math.isclose(sum(item[0] for item in result), 1.0, abs_tol=1e-12):
        raise RuntimeError("enumerated depth-limited paths do not sum to one")
    return tuple(result)


def audit_depth_limited_exact_expectation(
    *, maximum_source_depth_delta: int = 3
) -> DepthLimitedExactAuditEvidence:
    """Prove that exact leaves preserve pre-cutoff regret expectations."""

    game = RocketPrizeDuel()
    profile = analytic_equilibrium_profile(game)
    full = audit_outcome_sampling_expectation(game, profile)
    exact = {
        (row.traverser, row.information_set_id, row.action_id): (
            row.exact_immediate_counterfactual_regret
        )
        for row in full.rows
    }
    wrapper = DepthLimitedInformationSetGame(
        game,
        ExactProfileLeafValueProvider(game, profile),
        maximum_source_depth_delta=maximum_source_depth_delta,
    )
    traversal = OutcomeSamplingTraversal(
        OutcomeSamplingConfig(
            exploration=full.exploration,
            maximum_transitions=32,
            maximum_depth=32,
        )
    )
    weighted: dict[tuple[int, str, str], float] = defaultdict(float)
    traversals = 0
    for traverser in (0, 1):
        for probability, forced_chance, forced_actions in _enumerated_wrapper_paths(
            wrapper,
            profile,
            traverser=traverser,
            exploration=full.exploration,
        ):
            result = traversal.run(
                wrapper,
                lambda information_set, belief: profile.strategy(information_set),
                traverser=traverser,
                iteration=1,
                rng=random.Random(0),
                forced_chance=forced_chance,
                forced_actions=forced_actions,
            )
            traversals += 1
            if result.source_official_engine_executed:
                raise RuntimeError("exact cutoff traversal promoted official execution")
            for sample in result.regret_samples:
                for action_id, regret in sample.sampled_regrets:
                    weighted[(traverser, sample.information_set_id, action_id)] += (
                        probability * regret
                    )
    if len(weighted) < 4:
        raise RuntimeError("exact cutoff audit covered too few regret entries")
    maximum_error = max(abs(value - exact[key]) for key, value in weighted.items())
    if maximum_error > 1e-12:
        raise RuntimeError(
            f"depth-limited exact expectation audit failed: {maximum_error:.3e}"
        )
    if not wrapper.leaf_estimates or any(
        not item.calibrated
        or item.lower != item.value
        or item.upper != item.value
        or item.official_terminal_result
        for item in wrapper.leaf_estimates
    ):
        raise RuntimeError("exact cutoff leaves failed their calibration contract")
    return DepthLimitedExactAuditEvidence(
        maximum_source_depth_delta=maximum_source_depth_delta,
        traversals=traversals,
        audited_regret_entries=len(weighted),
        maximum_absolute_error=maximum_error,
        full_reference_fingerprint=full.fingerprint,
        exact_leaf_count=len(wrapper.leaf_estimates),
    )


@dataclass(frozen=True, slots=True)
class NativeDepthLimitedPlanDataset:
    native_game_fingerprint: str
    runtime_fingerprint: str
    deck_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    source_plan_blueprint_fingerprint: str
    source_macro_game_fingerprints: tuple[str, ...]
    depth_limited_game_fingerprints: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    iterations: int
    exploration: float
    maximum_source_depth_delta: int
    player_0_regrets: OutcomeRegretReservoir
    player_1_regrets: OutcomeRegretReservoir
    average_strategy: OutcomeStrategyReservoir
    trajectory_count: int
    cutoff_trajectory_count: int
    official_terminal_trajectory_count: int
    leaf_estimate_count: int
    native_transition_count: int
    macro_transition_count: int
    tactical_transition_count: int
    source_descendant_official_engine_executed: bool = True
    target_boundary_official_engine_executed: bool = False
    hidden_completion_prior_official: bool = False
    leaf_values_calibrated: bool = False
    counterfactual_values_calibrated: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "native_game_fingerprint",
            "runtime_fingerprint",
            "deck_fingerprint",
            "source_tactical_blueprint_fingerprint",
            "source_plan_blueprint_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.candidate_ids or not (
            len(self.source_macro_game_fingerprints)
            == len(self.depth_limited_game_fingerprints)
            == len(self.candidate_ids)
        ):
            raise ValueError("depth-limited games/candidates must align")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("depth-limited candidate IDs must be unique")
        for value in self.source_macro_game_fingerprints + self.depth_limited_game_fingerprints:
            _validate_digest(value, "game_fingerprint")
        if self.iterations <= 0 or not 0 < self.exploration <= 1:
            raise ValueError("depth-limited sampling metadata is invalid")
        if self.maximum_source_depth_delta <= 0:
            raise ValueError("depth-limited source horizon must be positive")
        if min(
            self.player_0_regrets.seen,
            self.player_1_regrets.seen,
            self.average_strategy.seen,
            self.trajectory_count,
            self.cutoff_trajectory_count,
            self.leaf_estimate_count,
            self.native_transition_count,
            self.macro_transition_count,
            self.tactical_transition_count,
        ) <= 0:
            raise ValueError("depth-limited dataset accounting must be positive")
        if (
            self.cutoff_trajectory_count + self.official_terminal_trajectory_count
            != self.trajectory_count
            or self.leaf_estimate_count != self.cutoff_trajectory_count
        ):
            raise ValueError("depth-limited terminal/cutoff accounting disagrees")
        allowed_games = set(self.depth_limited_game_fingerprints)
        for sample in (
            self.player_0_regrets.samples
            + self.player_1_regrets.samples
            + self.average_strategy.samples
        ):
            _sample_feature_contract(sample)
            if sample.game_fingerprint not in allowed_games:
                raise ValueError("depth-limited sample belongs to an unknown wrapper")
        if any(
            (
                not self.source_descendant_official_engine_executed,
                self.target_boundary_official_engine_executed,
                self.hidden_completion_prior_official,
                self.leaf_values_calibrated,
                self.counterfactual_values_calibrated,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("depth-limited dataset promoted an invalid capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_DEPTH_LIMITED_SCHEMA_VERSION,
            "native_game_fingerprint": self.native_game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "source_plan_blueprint_fingerprint": self.source_plan_blueprint_fingerprint,
            "source_macro_game_fingerprints": list(self.source_macro_game_fingerprints),
            "depth_limited_game_fingerprints": list(
                self.depth_limited_game_fingerprints
            ),
            "candidate_ids": list(self.candidate_ids),
            "iterations": self.iterations,
            "exploration": self.exploration,
            "maximum_source_depth_delta": self.maximum_source_depth_delta,
            "player_0_regrets": self.player_0_regrets.to_dict(),
            "player_1_regrets": self.player_1_regrets.to_dict(),
            "average_strategy": self.average_strategy.to_dict(),
            "accounting": {
                "trajectory_count": self.trajectory_count,
                "cutoff_trajectory_count": self.cutoff_trajectory_count,
                "official_terminal_trajectory_count": (
                    self.official_terminal_trajectory_count
                ),
                "leaf_estimate_count": self.leaf_estimate_count,
                "native_transition_count": self.native_transition_count,
                "macro_transition_count": self.macro_transition_count,
                "tactical_transition_count": self.tactical_transition_count,
            },
            "claims": {
                "source_descendant_official_engine_executed": (
                    self.source_descendant_official_engine_executed
                ),
                "target_boundary_official_engine_executed": (
                    self.target_boundary_official_engine_executed
                ),
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "leaf_values_calibrated": self.leaf_values_calibrated,
                "counterfactual_values_calibrated": (
                    self.counterfactual_values_calibrated
                ),
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
                "turn_plan_actions_official": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeDepthLimitedPlanDataset":
        if int(raw.get("schema_version", -1)) != PHASE6_NATIVE_DEPTH_LIMITED_SCHEMA_VERSION:
            raise ValueError("unsupported depth-limited dataset schema")
        accounting = raw["accounting"]
        claims = raw["claims"]
        result = cls(
            native_game_fingerprint=str(raw["native_game_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(
                raw["source_plan_blueprint_fingerprint"]
            ),
            source_macro_game_fingerprints=tuple(
                str(value) for value in raw["source_macro_game_fingerprints"]
            ),
            depth_limited_game_fingerprints=tuple(
                str(value) for value in raw["depth_limited_game_fingerprints"]
            ),
            candidate_ids=tuple(str(value) for value in raw["candidate_ids"]),
            iterations=int(raw["iterations"]),
            exploration=float(raw["exploration"]),
            maximum_source_depth_delta=int(raw["maximum_source_depth_delta"]),
            player_0_regrets=OutcomeRegretReservoir.from_dict(raw["player_0_regrets"]),
            player_1_regrets=OutcomeRegretReservoir.from_dict(raw["player_1_regrets"]),
            average_strategy=OutcomeStrategyReservoir.from_dict(
                raw["average_strategy"]
            ),
            trajectory_count=int(accounting["trajectory_count"]),
            cutoff_trajectory_count=int(accounting["cutoff_trajectory_count"]),
            official_terminal_trajectory_count=int(
                accounting["official_terminal_trajectory_count"]
            ),
            leaf_estimate_count=int(accounting["leaf_estimate_count"]),
            native_transition_count=int(accounting["native_transition_count"]),
            macro_transition_count=int(accounting["macro_transition_count"]),
            tactical_transition_count=int(accounting["tactical_transition_count"]),
            source_descendant_official_engine_executed=bool(
                claims["source_descendant_official_engine_executed"]
            ),
            target_boundary_official_engine_executed=bool(
                claims["target_boundary_official_engine_executed"]
            ),
            hidden_completion_prior_official=bool(
                claims["hidden_completion_prior_official"]
            ),
            leaf_values_calibrated=bool(claims["leaf_values_calibrated"]),
            counterfactual_values_calibrated=bool(
                claims["counterfactual_values_calibrated"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("depth-limited dataset fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6NativeDepthLimitedEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    source_phase6p_evidence_fingerprint: str
    source_plan_blueprint_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    exact_audit: DepthLimitedExactAuditEvidence
    trajectory_count: int
    player_0_regret_samples: int
    player_1_regret_samples: int
    strategy_samples: int
    cutoff_trajectory_count: int
    official_terminal_trajectory_count: int
    raw_regret_median_absolute: float
    raw_regret_p95_absolute: float
    raw_regret_maximum_absolute: float
    source_phase6p_raw_regret_p95_absolute: float
    source_phase6p_raw_regret_maximum_absolute: float
    p95_ratio_to_phase6p: float
    maximum_ratio_to_phase6p: float
    source_descendant_official_engine_executed: bool
    target_boundary_official_engine_executed: bool
    leaf_values_calibrated: bool
    counterfactual_bounds_built: bool
    safe_resolver_executed: bool
    exploitation_executed: bool
    official_strategy_control_enabled: bool
    raw_private_card_ids_persisted: int
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6R evidence identity is invalid")
        for field_name in (
            "dataset_fingerprint",
            "source_phase6p_evidence_fingerprint",
            "source_plan_blueprint_fingerprint",
            "source_tactical_blueprint_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if min(
            self.trajectory_count,
            self.player_0_regret_samples,
            self.player_1_regret_samples,
            self.strategy_samples,
            self.cutoff_trajectory_count,
        ) <= 0:
            raise ValueError("Phase 6R evidence accounting must be positive")
        if (
            self.cutoff_trajectory_count + self.official_terminal_trajectory_count
            != self.trajectory_count
        ):
            raise ValueError("Phase 6R cutoff/terminal accounting disagrees")
        for field_name in (
            "raw_regret_median_absolute",
            "raw_regret_p95_absolute",
            "raw_regret_maximum_absolute",
            "source_phase6p_raw_regret_p95_absolute",
            "source_phase6p_raw_regret_maximum_absolute",
            "p95_ratio_to_phase6p",
            "maximum_ratio_to_phase6p",
        ):
            value = float(getattr(self, field_name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if any(
            (
                not self.source_descendant_official_engine_executed,
                self.target_boundary_official_engine_executed,
                self.leaf_values_calibrated,
                self.counterfactual_bounds_built,
                self.safe_resolver_executed,
                self.exploitation_executed,
                self.official_strategy_control_enabled,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6R promoted an invalid execution/capability claim")
        if dict(self.claims).get("phase6_complete") != "false":
            raise ValueError("Phase 6R cannot mark Phase 6 complete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_DEPTH_LIMITED_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_phase6p_evidence_fingerprint": (
                self.source_phase6p_evidence_fingerprint
            ),
            "source_plan_blueprint_fingerprint": self.source_plan_blueprint_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "exact_audit": asdict(self.exact_audit),
            "sampling": {
                "trajectory_count": self.trajectory_count,
                "player_0_regret_samples": self.player_0_regret_samples,
                "player_1_regret_samples": self.player_1_regret_samples,
                "strategy_samples": self.strategy_samples,
                "cutoff_trajectory_count": self.cutoff_trajectory_count,
                "official_terminal_trajectory_count": (
                    self.official_terminal_trajectory_count
                ),
            },
            "variance": {
                "raw_regret_median_absolute": self.raw_regret_median_absolute,
                "raw_regret_p95_absolute": self.raw_regret_p95_absolute,
                "raw_regret_maximum_absolute": self.raw_regret_maximum_absolute,
                "source_phase6p_raw_regret_p95_absolute": (
                    self.source_phase6p_raw_regret_p95_absolute
                ),
                "source_phase6p_raw_regret_maximum_absolute": (
                    self.source_phase6p_raw_regret_maximum_absolute
                ),
                "p95_ratio_to_phase6p": self.p95_ratio_to_phase6p,
                "maximum_ratio_to_phase6p": self.maximum_ratio_to_phase6p,
                "comparison_is_controlled_performance_estimate": False,
            },
            "execution": {
                "source_descendant_official_engine_executed": (
                    self.source_descendant_official_engine_executed
                ),
                "target_boundary_official_engine_executed": (
                    self.target_boundary_official_engine_executed
                ),
                "leaf_values_calibrated": self.leaf_values_calibrated,
                "counterfactual_bounds_built": self.counterfactual_bounds_built,
                "safe_resolver_executed": self.safe_resolver_executed,
                "exploitation_executed": self.exploitation_executed,
                "official_strategy_control_enabled": (
                    self.official_strategy_control_enabled
                ),
                "raw_private_card_ids_persisted": (
                    self.raw_private_card_ids_persisted
                ),
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6NativeDepthLimitedEvidence":
        if int(raw.get("schema_version", -1)) != PHASE6_NATIVE_DEPTH_LIMITED_SCHEMA_VERSION:
            raise ValueError("unsupported Phase 6R evidence schema")
        exact = raw["exact_audit"]
        sampling = raw["sampling"]
        variance = raw["variance"]
        execution = raw["execution"]
        result = cls(
            benchmark_id=str(raw["benchmark_id"]),
            status=str(raw["status"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            source_phase6p_evidence_fingerprint=str(
                raw["source_phase6p_evidence_fingerprint"]
            ),
            source_plan_blueprint_fingerprint=str(
                raw["source_plan_blueprint_fingerprint"]
            ),
            source_tactical_blueprint_fingerprint=str(
                raw["source_tactical_blueprint_fingerprint"]
            ),
            exact_audit=DepthLimitedExactAuditEvidence(
                maximum_source_depth_delta=int(exact["maximum_source_depth_delta"]),
                traversals=int(exact["traversals"]),
                audited_regret_entries=int(exact["audited_regret_entries"]),
                maximum_absolute_error=float(exact["maximum_absolute_error"]),
                full_reference_fingerprint=str(exact["full_reference_fingerprint"]),
                exact_leaf_count=int(exact["exact_leaf_count"]),
            ),
            trajectory_count=int(sampling["trajectory_count"]),
            player_0_regret_samples=int(sampling["player_0_regret_samples"]),
            player_1_regret_samples=int(sampling["player_1_regret_samples"]),
            strategy_samples=int(sampling["strategy_samples"]),
            cutoff_trajectory_count=int(sampling["cutoff_trajectory_count"]),
            official_terminal_trajectory_count=int(
                sampling["official_terminal_trajectory_count"]
            ),
            raw_regret_median_absolute=float(
                variance["raw_regret_median_absolute"]
            ),
            raw_regret_p95_absolute=float(variance["raw_regret_p95_absolute"]),
            raw_regret_maximum_absolute=float(
                variance["raw_regret_maximum_absolute"]
            ),
            source_phase6p_raw_regret_p95_absolute=float(
                variance["source_phase6p_raw_regret_p95_absolute"]
            ),
            source_phase6p_raw_regret_maximum_absolute=float(
                variance["source_phase6p_raw_regret_maximum_absolute"]
            ),
            p95_ratio_to_phase6p=float(variance["p95_ratio_to_phase6p"]),
            maximum_ratio_to_phase6p=float(variance["maximum_ratio_to_phase6p"]),
            source_descendant_official_engine_executed=bool(
                execution["source_descendant_official_engine_executed"]
            ),
            target_boundary_official_engine_executed=bool(
                execution["target_boundary_official_engine_executed"]
            ),
            leaf_values_calibrated=bool(execution["leaf_values_calibrated"]),
            counterfactual_bounds_built=bool(
                execution["counterfactual_bounds_built"]
            ),
            safe_resolver_executed=bool(execution["safe_resolver_executed"]),
            exploitation_executed=bool(execution["exploitation_executed"]),
            official_strategy_control_enabled=bool(
                execution["official_strategy_control_enabled"]
            ),
            raw_private_card_ids_persisted=int(
                execution["raw_private_card_ids_persisted"]
            ),
            claims=tuple(
                (str(name), str(value)) for name, value in raw["claims"].items()
            ),
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("Phase 6R evidence fingerprint mismatch")
        return result

    def to_markdown(self) -> str:
        return f"""# Phase 6R depth-limited native plan evidence

Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

The exact microgame cutoff audit covered {self.exact_audit.audited_regret_entries}
pre-cutoff regret entries over {self.exact_audit.traversals} enumerated sample
paths. Maximum expectation error was
`{self.exact_audit.maximum_absolute_error:.3e}`.

The live probe completed {self.trajectory_count} trajectories, with
{self.cutoff_trajectory_count} uncalibrated cutoff leaves and
{self.official_terminal_trajectory_count} genuine terminal outcomes. It
collected {self.player_0_regret_samples} / {self.player_1_regret_samples}
player regret samples and {self.strategy_samples} strategy samples.

Raw absolute regret median / p95 / maximum was
`{self.raw_regret_median_absolute:.6g}` / `{self.raw_regret_p95_absolute:.6g}` /
`{self.raw_regret_maximum_absolute:.6g}`. The p95 and maximum ratios to the
earlier Phase 6P artifact were `{self.p95_ratio_to_phase6p:.3e}` and
`{self.maximum_ratio_to_phase6p:.3e}`. This is a deterministic wiring probe,
not a controlled performance estimate.

## Claim boundary

Genuine native descendants were executed, but the plan boundary, supplied
hidden completion prior, cutoff target, and persisted samples are abstractions.
The leaf values have confidence zero and the complete `[-1, 1]` interval. No
counterfactual bounds, safe resolver, exploitation overlay, official strategy
control, Kaggle submission, equilibrium claim, or strength claim is enabled.
"""


@dataclass(frozen=True, slots=True)
class Phase6NativeDepthLimitedArtifacts:
    dataset: NativeDepthLimitedPlanDataset
    evidence: Phase6NativeDepthLimitedEvidence


def _bounded_plan_baseline(macro_game: OfficialNativePlanGame):
    def baseline(information_set, belief, value_player, acting_player):
        raw = macro_game.plan_value_baseline(
            information_set, belief, value_player, acting_player
        )
        return {
            action_id: max(-1.0, min(1.0, float(value)))
            for action_id, value in raw.items()
        }

    return baseline


def _learned_plan_strategy(provider: NativePlanBlueprintProvider):
    def strategy(information_set, belief):
        player = 0 if information_set.actor is InformationSetActor.SELF else 1
        return provider.current_strategy(information_set, belief, player=player)

    return strategy


def run_phase6_native_depth_limited(
    *,
    project_root: str | Path | None = None,
    source_tactical_blueprint_path: str | Path | None = None,
    source_plan_blueprint_path: str | Path | None = None,
    source_phase6p_evidence_path: str | Path | None = None,
    iterations: int = 3,
    exploration: float = 0.25,
    maximum_source_depth_delta: int = 6,
    reservoir_capacity: int = 4_096,
    maximum_tactical_transitions: int = 512,
    seed: int = PHASE6_NATIVE_DEPTH_LIMITED_SEED,
) -> Phase6NativeDepthLimitedArtifacts:
    """Run an exact audit and a live, uncalibrated cutoff variance probe."""

    if min(
        iterations,
        maximum_source_depth_delta,
        reservoir_capacity,
        maximum_tactical_transitions,
    ) <= 0 or not 0 < exploration <= 1:
        raise ValueError("Phase 6R limits are invalid")
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
        raise ValueError("Phase 6R plan blueprint does not match Phase 6P evidence")
    if (
        plan_artifact.source_tactical_blueprint_fingerprint
        != tactical_artifact.fingerprint
    ):
        raise ValueError("Phase 6R plan/tactical blueprint provenance mismatch")
    exact_audit = audit_depth_limited_exact_expectation()
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
    if tactical_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6R tactical blueprint/deck fingerprint mismatch")
    if plan_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6R plan blueprint/deck fingerprint mismatch")
    candidates: tuple[KaggleDeckProfile, ...] = OFFICIAL_PANEL_DECKS + (target,)
    observation, start = battle_start(
        list(target.card_ids), list(OFFICIAL_PANEL_DECKS[0].card_ids)
    )
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6R decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        completions = _completions(
            observation,
            target=target,
            candidates=candidates,
            basic_card_ids=_runtime_basics(runtime),
        )
        with OfficialNativeSearchGame(
            observation,
            completions,
            runtime_directory=runtime,
        ) as native_game:
            macro_games: list[OfficialNativePlanGame] = []
            wrappers: list[DepthLimitedInformationSetGame] = []
            for outcome in native_game.chance_outcomes(native_game.root_state()):
                scenario = native_game.transition(
                    native_game.root_state(), None, chance_outcome=outcome
                )
                macro = OfficialNativePlanGame(
                    native_game,
                    scenario,
                    bridge=bridge,
                    tactical_provider=tactical_provider,
                    maximum_tactical_transitions=maximum_tactical_transitions,
                )
                macro_games.append(macro)
                wrappers.append(
                    DepthLimitedInformationSetGame(
                        macro,
                        UncalibratedNativePlanLeafValueProvider(macro),
                        maximum_source_depth_delta=maximum_source_depth_delta,
                    )
                )
            if len(wrappers) != len(candidates):
                raise RuntimeError("Phase 6R completion candidates/wrappers disagree")
            regrets = (
                OutcomeRegretReservoir(reservoir_capacity, seed=_seed(seed, "p0")),
                OutcomeRegretReservoir(reservoir_capacity, seed=_seed(seed, "p1")),
            )
            strategies = OutcomeStrategyReservoir(
                2 * reservoir_capacity, seed=_seed(seed, "strategy")
            )
            traversal = OutcomeSamplingTraversal(
                OutcomeSamplingConfig(
                    exploration=exploration,
                    maximum_transitions=maximum_source_depth_delta + 2,
                    maximum_depth=maximum_source_depth_delta + 1,
                )
            )
            learned_strategy = _learned_plan_strategy(plan_provider)
            trajectory_count = 0
            cutoff_count = 0
            official_terminal_count = 0
            for iteration in range(1, iterations + 1):
                for candidate_index, (macro_game, wrapper) in enumerate(
                    zip(macro_games, wrappers, strict=True)
                ):
                    baseline = _bounded_plan_baseline(macro_game)
                    for traverser in (0, 1):
                        leaf_before = len(wrapper.leaf_estimates)
                        result = traversal.run(
                            wrapper,
                            learned_strategy,
                            traverser=traverser,
                            iteration=iteration,
                            rng=random.Random(
                                _seed(seed, iteration, candidate_index, traverser)
                            ),
                            baseline=baseline,
                        )
                        trajectory_count += 1
                        leaf_after = len(wrapper.leaf_estimates)
                        if leaf_after == leaf_before + 1:
                            cutoff_count += 1
                        elif leaf_after == leaf_before:
                            official_terminal_count += 1
                        else:
                            raise RuntimeError("Phase 6R emitted multiple leaves per path")
                        if result.source_official_engine_executed:
                            raise RuntimeError("Phase 6R cutoff sample claimed official result")
                        for sample in result.regret_samples:
                            regrets[traverser].add(sample)
                        for sample in result.strategy_samples:
                            strategies.add(sample)
            leaf_estimates = tuple(
                estimate for wrapper in wrappers for estimate in wrapper.leaf_estimates
            )
            if any(
                estimate.calibrated
                or estimate.confidence != 0.0
                or estimate.lower != -1.0
                or estimate.upper != 1.0
                or estimate.official_terminal_result
                for estimate in leaf_estimates
            ):
                raise RuntimeError("Phase 6R live leaf promoted calibration/official status")
            provisional = NativeDepthLimitedPlanDataset(
                native_game_fingerprint=native_game.fingerprint,
                runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                deck_fingerprint=target.fingerprint,
                source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
                source_plan_blueprint_fingerprint=plan_artifact.fingerprint,
                source_macro_game_fingerprints=tuple(
                    game.fingerprint for game in macro_games
                ),
                depth_limited_game_fingerprints=tuple(
                    game.fingerprint for game in wrappers
                ),
                candidate_ids=tuple(candidate.archetype for candidate in candidates),
                iterations=iterations,
                exploration=exploration,
                maximum_source_depth_delta=maximum_source_depth_delta,
                player_0_regrets=regrets[0],
                player_1_regrets=regrets[1],
                average_strategy=strategies,
                trajectory_count=trajectory_count,
                cutoff_trajectory_count=cutoff_count,
                official_terminal_trajectory_count=official_terminal_count,
                leaf_estimate_count=len(leaf_estimates),
                native_transition_count=native_game.transition_count,
                macro_transition_count=sum(
                    game.macro_transition_count for game in macro_games
                ),
                tactical_transition_count=sum(
                    game.tactical_transition_count for game in macro_games
                ),
            )
            serialized = provisional.to_dict()
            raw_private = _raw_private_key_count(serialized)
            if raw_private != 0:
                raise RuntimeError("Phase 6R serialized a forbidden raw private key")
            dataset = provisional
            restored = NativeDepthLimitedPlanDataset.from_dict(
                json.loads(json.dumps(serialized))
            )
            if restored.fingerprint != dataset.fingerprint:
                raise RuntimeError("Phase 6R dataset roundtrip drifted")
            raw_values = sorted(
                abs(value)
                for sample in regrets[0].samples + regrets[1].samples
                for _, value in sample.sampled_regrets
            )
            if not raw_values:
                raise RuntimeError("Phase 6R emitted no raw regret targets")
            p95_index = min(len(raw_values) - 1, int(0.95 * (len(raw_values) - 1)))
            p95 = raw_values[p95_index]
            maximum = raw_values[-1]
            p95_ratio = p95 / phase6p.raw_regret_p95_absolute
            maximum_ratio = maximum / phase6p.raw_regret_maximum_absolute
            claims = (
                ("exact_cutoff_expectation", "validated_on_non_official_microgame"),
                ("native_descendant_transitions", "genuine_official_search_execution"),
                ("turn_plan_game", "research_abstraction"),
                ("hidden_completion_prior", "supplied_uncalibrated_abstraction"),
                ("cutoff_leaf", "phase6k_scalar_clamped_uncalibrated"),
                ("variance_comparison", "deterministic_wiring_probe_not_performance_estimate"),
                ("counterfactual_value_bounds", "not_built"),
                ("safe_resolver", "not_enabled"),
                ("exploitation_overlay", "not_enabled"),
                ("official_strategy_control", "disabled"),
                ("equilibrium_convergence", "not_validated"),
                ("pokemon_strength", "not_claimed"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            evidence = Phase6NativeDepthLimitedEvidence(
                benchmark_id="phase6r-depth-limited-native-plan-v1",
                status="phase6r_depth_limited_variance_control_validated_partial",
                dataset_fingerprint=dataset.fingerprint,
                source_phase6p_evidence_fingerprint=phase6p.fingerprint,
                source_plan_blueprint_fingerprint=plan_artifact.fingerprint,
                source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
                exact_audit=exact_audit,
                trajectory_count=trajectory_count,
                player_0_regret_samples=regrets[0].seen,
                player_1_regret_samples=regrets[1].seen,
                strategy_samples=strategies.seen,
                cutoff_trajectory_count=cutoff_count,
                official_terminal_trajectory_count=official_terminal_count,
                raw_regret_median_absolute=statistics.median(raw_values),
                raw_regret_p95_absolute=p95,
                raw_regret_maximum_absolute=maximum,
                source_phase6p_raw_regret_p95_absolute=(
                    phase6p.raw_regret_p95_absolute
                ),
                source_phase6p_raw_regret_maximum_absolute=(
                    phase6p.raw_regret_maximum_absolute
                ),
                p95_ratio_to_phase6p=p95_ratio,
                maximum_ratio_to_phase6p=maximum_ratio,
                source_descendant_official_engine_executed=True,
                target_boundary_official_engine_executed=False,
                leaf_values_calibrated=False,
                counterfactual_bounds_built=False,
                safe_resolver_executed=False,
                exploitation_executed=False,
                official_strategy_control_enabled=False,
                raw_private_card_ids_persisted=raw_private,
                claims=claims,
            )
            return Phase6NativeDepthLimitedArtifacts(dataset, evidence)
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_depth_limited().evidence.to_json())
