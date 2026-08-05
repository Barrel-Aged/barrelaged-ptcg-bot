"""Phase 6AI causal branch-conditioned expected-value experiment.

The official engine executes complete source matches, snapshot replay, forced
plan descendants, and terminal continuations.  The supplied hidden allocation
law, five-plan boundary, public branch summarization, neural model, intervals,
and all downstream interpretations remain research abstractions.
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
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from .beliefs import OPPONENT_INTENTS
from .game_theory import InformationSetActor
from .native_blueprint_features import native_observer_public_feature_items
from .neural_blueprint import TensorArtifact
from .official_bridge import sanitize_official_public_state
from .phase6_expected_value_surfaces import (
    PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH,
    PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR,
    PHASE6_EXPECTED_VALUE_MEAN_CI_Z,
    PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE,
    PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT,
    PHASE6_EXPECTED_VALUE_POLICY_PROTOCOL,
    PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION,
    PHASE6_EXPECTED_VALUE_TARGET_PLAYER,
    ExpectedValueDataset,
    ExpectedValueRoot,
    _build_expected_value_root,
    _capture_complete_match_late_root,
    _load_expected_value_sources,
    _protocol_by_variant,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_opponent_deviation_values import OPPONENT_DEVIATION_ACTION_IDS
from .trajectory_neural import NamedFeatureNormalizer


PHASE6_BRANCH_VALUE_SCHEMA_VERSION = 1
PHASE6_BRANCH_VALUE_SEED = 6_049
PHASE6_BRANCH_VALUE_BOOTSTRAP_SEED = 6_050
PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES = 10_000
PHASE6_BRANCH_VALUE_COVERAGE = 0.90
PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR = 0.98
PHASE6_BRANCH_VALUE_HIDDEN_GRID = (32, 64)
PHASE6_BRANCH_VALUE_WEIGHT_DECAY_GRID = (1e-3, 1e-2, 1e-1)
PHASE6_BRANCH_VALUE_CONTRAST_WEIGHT_GRID = (1.0, 4.0)
PHASE6_BRANCH_VALUE_MEAN_BLEND_GRID = (0.50, 0.75, 1.0)
PHASE6_BRANCH_VALUE_ADVANTAGE_SCALE_GRID = (0.10, 0.25, 0.50, 1.0)
PHASE6_BRANCH_VALUE_EPOCHS = 100
PHASE6_BRANCH_VALUE_LEARNING_RATE = 0.01

PHASE6_BRANCH_PUBLIC_SCALARS = (
    "turn",
    "turn_action_count",
    "own_board_count",
    "own_board_hp",
    "own_hand_count",
    "own_prize_count",
    "own_deck_count",
    "opponent_board_count",
    "opponent_board_hp",
    "opponent_hand_count",
    "opponent_prize_count",
    "opponent_deck_count",
    "opponent_public_revealed_copies",
)
PHASE6_BRANCH_PUBLIC_BUCKET_PREFIXES = (
    "own_public_card_bucket_",
    "opponent_public_card_bucket_",
    "stadium_card_bucket_",
)
PHASE6_BRANCH_DELTA_NAMES = PHASE6_BRANCH_PUBLIC_SCALARS + tuple(
    f"{prefix}l1" for prefix in PHASE6_BRANCH_PUBLIC_BUCKET_PREFIXES
)
PHASE6_BRANCH_ACTORS = ("self", "opponent", "chance", "terminal")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _public_heuristic(features: Mapping[str, float]) -> float:
    prize_advantage = (
        float(features["opponent_prize_count"])
        - float(features["own_prize_count"])
    ) / 6.0
    hp_advantage = (
        float(features["own_board_hp"])
        - float(features["opponent_board_hp"])
    ) / 600.0
    board_advantage = (
        float(features["own_board_count"])
        - float(features["opponent_board_count"])
    ) / 6.0
    return math.tanh(
        2.0 * prize_advantage + 0.35 * hp_advantage + 0.20 * board_advantage
    )


def _public_delta_items(
    before: Sequence[tuple[str, float]],
    after: Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    left = dict(before)
    right = dict(after)
    if set(left) != set(right):
        raise ValueError("Phase 6AI public branch feature schemas differ")
    values: list[tuple[str, float]] = [
        (name, float(right[name] - left[name]))
        for name in PHASE6_BRANCH_PUBLIC_SCALARS
    ]
    for prefix in PHASE6_BRANCH_PUBLIC_BUCKET_PREFIXES:
        names = tuple(sorted(name for name in left if name.startswith(prefix)))
        if len(names) != 32:
            raise ValueError("Phase 6AI public card bucket schema drifted")
        values.append(
            (
                f"{prefix}l1",
                float(sum(abs(right[name] - left[name]) for name in names)),
            )
        )
    result = tuple(values)
    if tuple(name for name, _ in result) != PHASE6_BRANCH_DELTA_NAMES:
        raise RuntimeError("Phase 6AI public delta order drifted")
    return result


@dataclass(frozen=True, slots=True)
class BranchActionObservation:
    allocation_index: int
    action_id: str
    source_state_id: str
    child_state_id: str
    child_public_state_fingerprint: str
    public_deltas: tuple[tuple[str, float], ...]
    causal_branch_heuristic: float
    forced_tactical_transitions: int
    immediate_terminal: bool
    next_actor: str
    official_native_transition_executed: bool = True
    hidden_completion_truth_selected_as_feature: bool = False
    post_branch_private_features_persisted: int = 0

    def __post_init__(self) -> None:
        if self.allocation_index not in range(PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT):
            raise ValueError("Phase 6AI allocation index is invalid")
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AI action id is invalid")
        for name in (
            "source_state_id",
            "child_state_id",
            "child_public_state_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if tuple(name for name, _ in self.public_deltas) != PHASE6_BRANCH_DELTA_NAMES:
            raise ValueError("Phase 6AI branch delta schema/order is invalid")
        for name, value in self.public_deltas:
            _finite(value, name)
        if not -1 <= _finite(self.causal_branch_heuristic, "causal_branch_heuristic") <= 1:
            raise ValueError("Phase 6AI branch heuristic is outside [-1, 1]")
        if self.forced_tactical_transitions < 0:
            raise ValueError("Phase 6AI tactical transition count is negative")
        if self.next_actor not in PHASE6_BRANCH_ACTORS:
            raise ValueError("Phase 6AI next actor is invalid")
        if self.immediate_terminal != (self.next_actor == "terminal"):
            raise ValueError("Phase 6AI terminal/actor metadata disagree")
        if any(
            (
                not self.official_native_transition_executed,
                self.hidden_completion_truth_selected_as_feature,
                self.post_branch_private_features_persisted != 0,
            )
        ):
            raise ValueError("Phase 6AI branch observation promoted private/abstract state")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "allocation_index": self.allocation_index,
            "action_id": self.action_id,
            "source_state_id": self.source_state_id,
            "child_state_id": self.child_state_id,
            "child_public_state_fingerprint": self.child_public_state_fingerprint,
            "public_deltas": dict(self.public_deltas),
            "causal_branch_heuristic": self.causal_branch_heuristic,
            "forced_tactical_transitions": self.forced_tactical_transitions,
            "immediate_terminal": self.immediate_terminal,
            "next_actor": self.next_actor,
            "claims": {
                "official_native_transition_executed": (
                    self.official_native_transition_executed
                ),
                "hidden_completion_truth_selected_as_feature": (
                    self.hidden_completion_truth_selected_as_feature
                ),
                "post_branch_private_features_persisted": (
                    self.post_branch_private_features_persisted
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchActionObservation":
        claims = raw["claims"]
        result = cls(
            allocation_index=int(raw["allocation_index"]),
            action_id=str(raw["action_id"]),
            source_state_id=str(raw["source_state_id"]),
            child_state_id=str(raw["child_state_id"]),
            child_public_state_fingerprint=str(raw["child_public_state_fingerprint"]),
            public_deltas=tuple(
                (name, float(raw["public_deltas"][name]))
                for name in PHASE6_BRANCH_DELTA_NAMES
            ),
            causal_branch_heuristic=float(raw["causal_branch_heuristic"]),
            forced_tactical_transitions=int(raw["forced_tactical_transitions"]),
            immediate_terminal=bool(raw["immediate_terminal"]),
            next_actor=str(raw["next_actor"]),
            official_native_transition_executed=bool(
                claims["official_native_transition_executed"]
            ),
            hidden_completion_truth_selected_as_feature=bool(
                claims["hidden_completion_truth_selected_as_feature"]
            ),
            post_branch_private_features_persisted=int(
                claims["post_branch_private_features_persisted"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AI branch observation fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class BranchConditionedRoot:
    value_root: ExpectedValueRoot
    branch_observations: tuple[BranchActionObservation, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            (allocation, action)
            for allocation in range(PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT)
            for action in OPPONENT_DEVIATION_ACTION_IDS
        )
        actual = tuple(
            (item.allocation_index, item.action_id)
            for item in self.branch_observations
        )
        if actual != expected:
            raise ValueError("Phase 6AI branch support/order is incomplete")
        by_allocation = {
            allocation: {
                action: next(
                    item
                    for item in self.branch_observations
                    if item.allocation_index == allocation and item.action_id == action
                )
                for action in OPPONENT_DEVIATION_ACTION_IDS
            }
            for allocation in range(PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT)
        }
        for surface in self.value_root.surfaces:
            for estimate in surface.action_values:
                branch = by_allocation[surface.allocation_index][estimate.action_id]
                if (
                    branch.forced_tactical_transitions
                    != estimate.forced_action_tactical_transitions
                ):
                    raise ValueError("Phase 6AI branch/value transition counts disagree")
        serialized = self.to_dict(include_fingerprint=False)
        if _raw_private_key_count(serialized) != 0:
            raise ValueError("Phase 6AI root persisted a forbidden raw private key")

    @property
    def root_id(self) -> str:
        return self.value_root.root_id

    @property
    def match_id(self) -> str:
        return self.value_root.match_id

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_VALUE_SCHEMA_VERSION,
            "value_root": self.value_root.to_dict(),
            "branch_observations": [
                item.to_dict() for item in self.branch_observations
            ],
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchConditionedRoot":
        result = cls(
            value_root=ExpectedValueRoot.from_dict(raw["value_root"]),
            branch_observations=tuple(
                BranchActionObservation.from_dict(item)
                for item in raw["branch_observations"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AI root fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class BranchConditionedDataset:
    source_phase6ah_design_dataset_fingerprint: str
    design_root_ids: tuple[str, ...]
    seed: int
    match_namespace: str
    value_dataset: ExpectedValueDataset
    roots: tuple[BranchConditionedRoot, ...]
    phase6ah_roots_reused: int = 0
    public_branch_features_only: bool = True
    hidden_completion_truth_selected_as_feature: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        _validate_digest(
            self.source_phase6ah_design_dataset_fingerprint,
            "source_phase6ah_design_dataset_fingerprint",
        )
        if self.seed != PHASE6_BRANCH_VALUE_SEED or self.match_namespace != "phase6ai":
            raise ValueError("Phase 6AI frozen seed/namespace drifted")
        if len(self.design_root_ids) != 180 or len(set(self.design_root_ids)) != 180:
            raise ValueError("Phase 6AI design-root ledger is incomplete")
        if len(self.roots) != 180 or len({item.root_id for item in self.roots}) != 180:
            raise ValueError("Phase 6AI fresh root panel is incomplete")
        if tuple(item.value_root for item in self.roots) != self.value_dataset.roots:
            raise ValueError("Phase 6AI branch/value root order differs")
        if set(self.design_root_ids) & {item.root_id for item in self.roots}:
            raise ValueError("Phase 6AI reused a Phase 6AH design root")
        if any(not item.match_id.startswith("phase6ai-") for item in self.roots):
            raise ValueError("Phase 6AI root escaped its namespace")
        if any(
            (
                self.phase6ah_roots_reused != 0,
                not self.public_branch_features_only,
                self.hidden_completion_truth_selected_as_feature,
                self.raw_private_card_ids_persisted != 0,
            )
        ):
            raise ValueError("Phase 6AI dataset promoted its abstraction boundary")
        if _raw_private_key_count(self.to_dict(include_fingerprint=False)) != 0:
            raise ValueError("Phase 6AI dataset persisted a forbidden raw private key")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def terminal_rollout_count(self) -> int:
        return self.value_dataset.terminal_rollout_count

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_VALUE_SCHEMA_VERSION,
            "source_phase6ah_design_dataset_fingerprint": (
                self.source_phase6ah_design_dataset_fingerprint
            ),
            "design_root_ids": list(self.design_root_ids),
            "seed": self.seed,
            "match_namespace": self.match_namespace,
            "value_dataset": self.value_dataset.to_dict(),
            "roots": [item.to_dict() for item in self.roots],
            "claims": {
                "phase6ah_roots_reused": self.phase6ah_roots_reused,
                "public_branch_features_only": self.public_branch_features_only,
                "hidden_completion_truth_selected_as_feature": (
                    self.hidden_completion_truth_selected_as_feature
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchConditionedDataset":
        claims = raw["claims"]
        result = cls(
            source_phase6ah_design_dataset_fingerprint=str(
                raw["source_phase6ah_design_dataset_fingerprint"]
            ),
            design_root_ids=tuple(str(value) for value in raw["design_root_ids"]),
            seed=int(raw["seed"]),
            match_namespace=str(raw["match_namespace"]),
            value_dataset=ExpectedValueDataset.from_dict(raw["value_dataset"]),
            roots=tuple(BranchConditionedRoot.from_dict(item) for item in raw["roots"]),
            phase6ah_roots_reused=int(claims["phase6ah_roots_reused"]),
            public_branch_features_only=bool(claims["public_branch_features_only"]),
            hidden_completion_truth_selected_as_feature=bool(
                claims["hidden_completion_truth_selected_as_feature"]
            ),
            raw_private_card_ids_persisted=int(
                claims["raw_private_card_ids_persisted"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AI dataset fingerprint mismatch")
        return result


def _branch_actor(child: Any, macro: Any) -> str:
    if child.terminal:
        return "terminal"
    if child.actor is InformationSetActor.CHANCE:
        return "chance"
    player = macro.player_to_act(child)
    if player == PHASE6_EXPECTED_VALUE_TARGET_PLAYER:
        return "self"
    if player == 1 - PHASE6_EXPECTED_VALUE_TARGET_PLAYER:
        return "opponent"
    raise ValueError("Phase 6AI branch has no valid next actor")


def _make_value_dataset(sources: Any, roots: Sequence[ExpectedValueRoot]) -> ExpectedValueDataset:
    return ExpectedValueDataset(
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
        roots=tuple(roots),
    )


def _build_branch_conditioned_root(
    capture: Any,
    *,
    sources: Any,
    candidate: Any,
    intent: Any,
    policy_variant: str,
    split: str,
    replica: int,
    match_id: str,
    seed: int,
    maximum_rollout_transitions: int,
    maximum_tactical_transitions: int,
    experiment_label: str = "Phase 6AI",
) -> BranchConditionedRoot:
    observations: list[BranchActionObservation] = []

    def observe_branch(
        allocation_index: int,
        action: Any,
        state: Any,
        child: Any,
        macro: Any,
        belief: Any,
    ) -> None:
        before = tuple(
            (str(name), float(value))
            for name, value in belief.public_features
        )
        after = native_observer_public_feature_items(
            child.native_state.observation,
            belief.opponent_range,
            observer_player=PHASE6_EXPECTED_VALUE_TARGET_PLAYER,
        )
        observations.append(
            BranchActionObservation(
                allocation_index=allocation_index,
                action_id=action.action_id,
                source_state_id=state.state_id,
                child_state_id=child.state_id,
                child_public_state_fingerprint=_fingerprint(
                    sanitize_official_public_state(child.native_state.observation)
                ),
                public_deltas=_public_delta_items(before, after),
                causal_branch_heuristic=_public_heuristic(dict(after)),
                forced_tactical_transitions=(
                    child.tactical_transitions - state.tactical_transitions
                ),
                immediate_terminal=child.terminal,
                next_actor=_branch_actor(child, macro),
            )
        )

    value_root = _build_expected_value_root(
        capture,
        sources=sources,
        candidate=candidate,
        intent=intent,
        policy_variant=policy_variant,
        split=split,
        replica=replica,
        match_id=match_id,
        seed=seed,
        maximum_rollout_transitions=maximum_rollout_transitions,
        maximum_tactical_transitions=maximum_tactical_transitions,
        experiment_label=experiment_label,
        action_branch_observer=observe_branch,
    )
    result = BranchConditionedRoot(
        value_root=value_root,
        branch_observations=tuple(observations),
    )
    if BranchConditionedRoot.from_dict(json.loads(result.to_json())) != result:
        raise RuntimeError("Phase 6AI root roundtrip changed values")
    return result


def collect_branch_conditioned_dataset(
    *,
    project_root: str | Path | None = None,
    existing_roots: Sequence[BranchConditionedRoot] = (),
    root_observer: Callable[[BranchConditionedRoot, int, int], None] | None = None,
    maximum_live_selections: int = PHASE6_EXPECTED_VALUE_MAXIMUM_LIVE_SELECTIONS,
    maximum_capture_attempts: int = PHASE6_EXPECTED_VALUE_MAXIMUM_CAPTURE_ATTEMPTS,
    maximum_rollout_transitions: int = PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    maximum_tactical_transitions: int = PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    seed: int = PHASE6_BRANCH_VALUE_SEED,
) -> BranchConditionedDataset:
    if min(
        maximum_live_selections,
        maximum_capture_attempts,
        maximum_rollout_transitions,
        maximum_tactical_transitions,
        seed,
    ) <= 0 or seed != PHASE6_BRANCH_VALUE_SEED:
        raise ValueError("Phase 6AI collection limits/seed are invalid")
    sources = _load_expected_value_sources(project_root)
    design_path = (
        sources.root
        / "artifacts"
        / "phase6_expected_value_surfaces"
        / "expected_value_surface_dataset.json"
    )
    design = ExpectedValueDataset.from_dict(
        json.loads(design_path.read_text(encoding="utf-8"))
    )
    runtime_text = str(sources.runtime)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

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
            item.value_root.policy_variant_id_supervision_only,
            item.value_root.candidate_id_hypothesis_input,
            item.value_root.intent_id_supervision_only,
            item.value_root.replica,
        ): item
        for item in materialized
    }
    if len(by_key) != len(materialized) or not set(by_key).issubset(expected_keys):
        raise ValueError("Phase 6AI resume roots are duplicate/outside the frozen grid")
    candidate_by_id = {item.archetype: item for item in sources.candidates}
    intent_by_id = {item.value: item for item in OPPONENT_INTENTS}
    total = len(expected_keys)
    for key in expected_keys:
        if key in by_key:
            continue
        variant, candidate_id, intent_id, replica = key
        split, _ = _protocol_by_variant()[variant]
        candidate = candidate_by_id[candidate_id]
        intent = intent_by_id[intent_id]
        match_id = f"phase6ai-{split}-{variant}-{candidate_id}-{intent_id}-r{replica}"
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
                branch_root = _build_branch_conditioned_root(
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
                by_key[key] = branch_root
                if root_observer is not None:
                    root_observer(branch_root, len(by_key), total)
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                "Phase 6AI exhausted capture attempts for "
                f"{variant}/{candidate_id}/{intent_id}/r{replica}"
            ) from last_error
    roots = tuple(by_key[key] for key in expected_keys)
    value_dataset = _make_value_dataset(
        sources,
        tuple(item.value_root for item in roots),
    )
    dataset = BranchConditionedDataset(
        source_phase6ah_design_dataset_fingerprint=design.fingerprint,
        design_root_ids=tuple(sorted(item.root_id for item in design.roots)),
        seed=seed,
        match_namespace="phase6ai",
        value_dataset=value_dataset,
        roots=roots,
    )
    restored = BranchConditionedDataset.from_dict(
        json.loads(dataset.to_json())
    )
    if restored != dataset:
        raise RuntimeError("Phase 6AI dataset roundtrip changed values")
    return dataset


@dataclass(frozen=True, slots=True)
class BranchValueActionTarget:
    action_id: str
    branch_features: tuple[tuple[str, float], ...]
    branch_heuristic_value: float
    terminal_returns: tuple[float, ...]
    empirical_player0_value: float
    target_standard_error: float

    def __post_init__(self) -> None:
        if self.action_id not in OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AI target action is invalid")
        names = tuple(name for name, _ in self.branch_features)
        if not names or len(names) != len(set(names)):
            raise ValueError("Phase 6AI branch target feature schema is invalid")
        if any(
            forbidden in name
            for name in names
            for forbidden in (
                "intent_id",
                "policy_variant",
                "supervision_only",
                "terminal_return",
                "rollout_depth",
                "hidden_completion",
            )
        ):
            raise ValueError("Phase 6AI branch target leaked supervision/future truth")
        if len(self.terminal_returns) != (
            PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT
            * PHASE6_EXPECTED_VALUE_ROLLOUTS_PER_PARTICLE_ACTION
        ):
            raise ValueError("Phase 6AI target return count is invalid")
        if any(value not in (-1.0, 0.0, 1.0) for value in self.terminal_returns):
            raise ValueError("Phase 6AI terminal utility is invalid")
        mean = statistics.mean(self.terminal_returns)
        se = statistics.stdev(self.terminal_returns) / math.sqrt(
            len(self.terminal_returns)
        )
        if not math.isclose(self.empirical_player0_value, mean, abs_tol=1e-12):
            raise ValueError("Phase 6AI target mean is inconsistent")
        if not math.isclose(self.target_standard_error, se, abs_tol=1e-12):
            raise ValueError("Phase 6AI target SE is inconsistent")
        if not -1 <= self.branch_heuristic_value <= 1:
            raise ValueError("Phase 6AI branch heuristic is invalid")


@dataclass(frozen=True, slots=True)
class BranchValueRootExample:
    root_id: str
    match_id: str
    split: str
    calibration_partition: str
    candidate_id_hypothesis_input: str
    intent_id_supervision_only: str
    policy_variant_id_supervision_only: str
    root_features: tuple[tuple[str, float], ...]
    root_heuristic_value: float
    actions: tuple[BranchValueActionTarget, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.root_id, "root_id")
        if not self.match_id or self.split not in {"train", "calibration", "validation"}:
            raise ValueError("Phase 6AI example identity/split is invalid")
        names = tuple(name for name, _ in self.root_features)
        if not names or len(names) != len(set(names)):
            raise ValueError("Phase 6AI root feature schema is invalid")
        if any(
            forbidden in name
            for name in names
            for forbidden in ("intent_id", "policy_variant", "supervision_only")
        ):
            raise ValueError("Phase 6AI root features leaked an audit-only label")
        if tuple(item.action_id for item in self.actions) != OPPONENT_DEVIATION_ACTION_IDS:
            raise ValueError("Phase 6AI example lacks complete ordered action support")
        if not -1 <= self.root_heuristic_value <= 1:
            raise ValueError("Phase 6AI root heuristic is invalid")


def _root_feature_items(
    root: ExpectedValueRoot,
    candidate_ids: Sequence[str],
) -> tuple[tuple[str, float], ...]:
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
    )


def _aggregated_branch_features(
    root: BranchConditionedRoot,
    action_id: str,
) -> tuple[tuple[tuple[str, float], ...], float]:
    observations = tuple(
        item for item in root.branch_observations if item.action_id == action_id
    )
    if len(observations) != PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT:
        raise ValueError("Phase 6AI action lacks both allocation branches")
    values: list[tuple[str, float]] = []
    for delta_name in PHASE6_BRANCH_DELTA_NAMES:
        samples = [dict(item.public_deltas)[delta_name] for item in observations]
        values.extend(
            (
                (f"public_delta_mean::{delta_name}", statistics.mean(samples)),
                (f"public_delta_range::{delta_name}", max(samples) - min(samples)),
            )
        )
    tactical = [float(item.forced_tactical_transitions) for item in observations]
    values.extend(
        (
            ("forced_tactical_transitions_mean", statistics.mean(tactical)),
            ("forced_tactical_transitions_range", max(tactical) - min(tactical)),
            (
                "immediate_terminal_rate",
                statistics.mean(float(item.immediate_terminal) for item in observations),
            ),
        )
    )
    for actor in PHASE6_BRANCH_ACTORS:
        values.append(
            (
                f"next_actor_rate::{actor}",
                statistics.mean(float(item.next_actor == actor) for item in observations),
            )
        )
    blueprint = dict(root.value_root.blueprint_probabilities)[action_id]
    values.append(("blueprint_probability", blueprint))
    values.extend(
        (
            f"plan_action::{candidate_action}",
            1.0 if action_id == candidate_action else 0.0,
        )
        for candidate_action in OPPONENT_DEVIATION_ACTION_IDS
    )
    branch_heuristic = statistics.mean(
        item.causal_branch_heuristic for item in observations
    )
    return tuple(values), branch_heuristic


def build_branch_value_examples(
    dataset: BranchConditionedDataset,
) -> tuple[BranchValueRootExample, ...]:
    return build_branch_value_examples_from_roots(
        dataset.roots,
        dataset.value_dataset.candidate_ids,
        expected_count=180,
    )


def build_branch_value_examples_from_roots(
    roots: Sequence[BranchConditionedRoot],
    candidate_ids: Sequence[str],
    *,
    expected_count: int | None = None,
) -> tuple[BranchValueRootExample, ...]:
    examples = []
    for branch_root in roots:
        root = branch_root.value_root
        estimates = {
            action_id: tuple(
                estimate
                for surface in root.surfaces
                for estimate in surface.action_values
                if estimate.action_id == action_id
            )
            for action_id in OPPONENT_DEVIATION_ACTION_IDS
        }
        action_targets = []
        for action_id in OPPONENT_DEVIATION_ACTION_IDS:
            action_estimates = estimates[action_id]
            if len(action_estimates) != PHASE6_EXPECTED_VALUE_PARTICLES_PER_ROOT:
                raise ValueError("Phase 6AI value root lacks both action estimates")
            terminal_returns = tuple(
                value
                for estimate in action_estimates
                for value in estimate.terminal_returns
            )
            features, branch_heuristic = _aggregated_branch_features(
                branch_root,
                action_id,
            )
            action_targets.append(
                BranchValueActionTarget(
                    action_id=action_id,
                    branch_features=features,
                    branch_heuristic_value=branch_heuristic,
                    terminal_returns=terminal_returns,
                    empirical_player0_value=statistics.mean(terminal_returns),
                    target_standard_error=(
                        statistics.stdev(terminal_returns)
                        / math.sqrt(len(terminal_returns))
                    ),
                )
            )
        example = BranchValueRootExample(
            root_id=root.root_id,
            match_id=root.match_id,
            split=root.split,
            calibration_partition=root.calibration_partition,
            candidate_id_hypothesis_input=root.candidate_id_hypothesis_input,
            intent_id_supervision_only=root.intent_id_supervision_only,
            policy_variant_id_supervision_only=(
                root.policy_variant_id_supervision_only
            ),
            root_features=_root_feature_items(
                root,
                candidate_ids,
            ),
            root_heuristic_value=_public_heuristic(dict(root.public_features)),
            actions=tuple(action_targets),
        )
        examples.append(example)
    if expected_count is not None and len(examples) != expected_count:
        raise RuntimeError("Phase 6AI example count drifted")
    if len({item.root_id for item in examples}) != len(examples):
        raise RuntimeError("Phase 6AI example accounting drifted")
    if len({tuple(name for name, _ in item.root_features) for item in examples}) != 1:
        raise ValueError("Phase 6AI root feature schemas differ")
    if len(
        {
            tuple(name for name, _ in action.branch_features)
            for item in examples
            for action in item.actions
        }
    ) != 1:
        raise ValueError("Phase 6AI branch feature schemas differ")
    return tuple(examples)


class _BranchDuelingNetwork(nn.Module):
    def __init__(self, root_width: int, branch_width: int, hidden_dim: int) -> None:
        super().__init__()
        branch_hidden = max(8, hidden_dim // 2)
        self.root_input = nn.Linear(root_width, hidden_dim, dtype=torch.float64)
        self.root_hidden = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float64)
        self.branch_input = nn.Linear(
            branch_width,
            branch_hidden,
            dtype=torch.float64,
        )
        self.advantage_hidden = nn.Linear(
            hidden_dim + branch_hidden,
            hidden_dim,
            dtype=torch.float64,
        )
        self.mean_output = nn.Linear(hidden_dim, 1, dtype=torch.float64)
        self.advantage_output = nn.Linear(hidden_dim, 1, dtype=torch.float64)

    def forward(self, root_features: Tensor, branch_features: Tensor) -> tuple[Tensor, Tensor]:
        root = torch.tanh(self.root_input(root_features))
        root = torch.tanh(self.root_hidden(root))
        count, actions, width = branch_features.shape
        branch = torch.tanh(
            self.branch_input(branch_features.reshape(count * actions, width))
        ).reshape(count, actions, -1)
        expanded = root[:, None, :].expand(-1, actions, -1)
        advantage = torch.tanh(
            self.advantage_hidden(torch.cat((expanded, branch), dim=-1))
        )
        advantage = torch.tanh(self.advantage_output(advantage).squeeze(-1))
        advantage = advantage - advantage.mean(dim=1, keepdim=True)
        mean = torch.tanh(self.mean_output(root).squeeze(-1))
        return mean, advantage


def _network_tensors(network: _BranchDuelingNetwork) -> tuple[TensorArtifact, ...]:
    return tuple(
        TensorArtifact(
            name=name,
            shape=tuple(tensor.shape),
            values=tuple(float(value) for value in tensor.detach().reshape(-1).tolist()),
        )
        for name, tensor in network.state_dict().items()
    )


def _restore_network(
    root_normalizer: NamedFeatureNormalizer,
    branch_normalizer: NamedFeatureNormalizer,
    hidden_dim: int,
    tensors: Sequence[TensorArtifact],
) -> _BranchDuelingNetwork:
    network = _BranchDuelingNetwork(
        len(root_normalizer.names),
        len(branch_normalizer.names),
        hidden_dim,
    )
    templates = network.state_dict()
    supplied = {item.name: item for item in tensors}
    if set(templates) != set(supplied):
        raise ValueError("Phase 6AI tensor names do not match the architecture")
    restored = {}
    for name, template in templates.items():
        artifact = supplied[name]
        if artifact.shape != tuple(template.shape):
            raise ValueError(f"Phase 6AI tensor shape mismatch: {name}")
        restored[name] = torch.tensor(
            artifact.values,
            dtype=torch.float64,
        ).reshape(artifact.shape)
    network.load_state_dict(restored, strict=True)
    network.eval()
    return network


def _example_tensors(
    examples: Sequence[BranchValueRootExample],
    root_normalizer: NamedFeatureNormalizer,
    branch_normalizer: NamedFeatureNormalizer,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    root_features = torch.tensor(
        [root_normalizer.transform(item.root_features) for item in examples],
        dtype=torch.float64,
    )
    branch_features = torch.tensor(
        [
            [branch_normalizer.transform(action.branch_features) for action in item.actions]
            for item in examples
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [
            [action.empirical_player0_value for action in item.actions]
            for item in examples
        ],
        dtype=torch.float64,
    )
    root_heuristics = torch.tensor(
        [item.root_heuristic_value for item in examples],
        dtype=torch.float64,
    )
    return root_features, branch_features, targets, root_heuristics


def _train_network(
    examples: Sequence[BranchValueRootExample],
    root_normalizer: NamedFeatureNormalizer,
    branch_normalizer: NamedFeatureNormalizer,
    *,
    hidden_dim: int,
    weight_decay: float,
    contrast_weight: float,
    seed: int,
) -> tuple[_BranchDuelingNetwork, float, float]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = _BranchDuelingNetwork(
        len(root_normalizer.names),
        len(branch_normalizer.names),
        hidden_dim,
    )
    root, branch, targets, _ = _example_tensors(
        examples,
        root_normalizer,
        branch_normalizer,
    )
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=PHASE6_BRANCH_VALUE_LEARNING_RATE,
        weight_decay=weight_decay,
    )

    def loss_value() -> Tensor:
        mean, advantages = network(root, branch)
        predictions = torch.clamp(mean[:, None] + advantages, -1.0, 1.0)
        target_mean = targets.mean(dim=1)
        prediction_contrast = predictions - predictions.mean(dim=1, keepdim=True)
        target_contrast = targets - targets.mean(dim=1, keepdim=True)
        return torch.mean((mean - target_mean) ** 2) + contrast_weight * torch.mean(
            (prediction_contrast - target_contrast) ** 2
        )

    with torch.no_grad():
        initial = float(loss_value())
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for _ in range(PHASE6_BRANCH_VALUE_EPOCHS):
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
        raise RuntimeError("Phase 6AI grouped training loss did not decrease")
    network.eval()
    return network, initial, final


def _flatten(values: Sequence[Sequence[float]]) -> list[float]:
    return [float(value) for group in values for value in group]


def _targets(examples: Sequence[BranchValueRootExample]) -> list[list[float]]:
    return [
        [action.empirical_player0_value for action in item.actions]
        for item in examples
    ]


def _root_heuristic_predictions(
    examples: Sequence[BranchValueRootExample],
) -> list[list[float]]:
    return [
        [item.root_heuristic_value] * len(OPPONENT_DEVIATION_ACTION_IDS)
        for item in examples
    ]


def _branch_heuristic_predictions(
    examples: Sequence[BranchValueRootExample],
) -> list[list[float]]:
    return [
        [action.branch_heuristic_value for action in item.actions]
        for item in examples
    ]


def _mse(targets: Sequence[Sequence[float]], predictions: Sequence[Sequence[float]]) -> float:
    left = _flatten(targets)
    right = _flatten(predictions)
    if not left or len(left) != len(right):
        raise ValueError("Phase 6AI MSE vectors are empty/misaligned")
    return statistics.mean((prediction - target) ** 2 for target, prediction in zip(left, right, strict=True))


def _contrast_root_errors(
    targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    if not targets or len(targets) != len(predictions):
        raise ValueError("Phase 6AI contrast groups are empty/misaligned")
    errors = []
    for target, prediction in zip(targets, predictions, strict=True):
        if len(target) != len(OPPONENT_DEVIATION_ACTION_IDS) or len(prediction) != len(target):
            raise ValueError("Phase 6AI contrast group lacks five actions")
        target_mean = statistics.mean(target)
        prediction_mean = statistics.mean(prediction)
        errors.append(
            statistics.mean(
                ((p - prediction_mean) - (t - target_mean)) ** 2
                for t, p in zip(target, prediction, strict=True)
            )
        )
    return tuple(errors)


def _contrast_mse(
    targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> float:
    return statistics.mean(_contrast_root_errors(targets, predictions))


def _network_predictions(
    network: _BranchDuelingNetwork,
    examples: Sequence[BranchValueRootExample],
    root_normalizer: NamedFeatureNormalizer,
    branch_normalizer: NamedFeatureNormalizer,
    *,
    mean_blend: float,
    advantage_scale: float,
) -> list[list[float]]:
    root, branch, _, root_heuristic = _example_tensors(
        examples,
        root_normalizer,
        branch_normalizer,
    )
    with torch.no_grad():
        mean, advantages = network(root, branch)
        mean = mean_blend * mean + (1.0 - mean_blend) * root_heuristic
        predictions = torch.clamp(
            mean[:, None] + advantage_scale * advantages,
            -1.0,
            1.0,
        )
    return [
        [float(value) for value in row]
        for row in predictions.tolist()
    ]


@dataclass(frozen=True, slots=True)
class BranchValueSelectionMetric:
    hidden_dim: int
    weight_decay: float
    contrast_weight: float
    mean_blend: float
    advantage_scale: float
    initial_training_loss: float
    final_training_loss: float
    selection_mse: float
    selection_zero_mse: float
    selection_root_heuristic_mse: float
    selection_branch_heuristic_mse: float
    selection_action_contrast_mse: float
    selection_zero_action_contrast_mse: float
    selection_branch_action_contrast_mse: float
    selection_score: float

    def __post_init__(self) -> None:
        if self.hidden_dim not in PHASE6_BRANCH_VALUE_HIDDEN_GRID:
            raise ValueError("Phase 6AI hidden width is unsupported")
        if self.weight_decay not in PHASE6_BRANCH_VALUE_WEIGHT_DECAY_GRID:
            raise ValueError("Phase 6AI weight decay is unsupported")
        if self.contrast_weight not in PHASE6_BRANCH_VALUE_CONTRAST_WEIGHT_GRID:
            raise ValueError("Phase 6AI contrast weight is unsupported")
        if self.mean_blend not in PHASE6_BRANCH_VALUE_MEAN_BLEND_GRID:
            raise ValueError("Phase 6AI mean blend is unsupported")
        if self.advantage_scale not in PHASE6_BRANCH_VALUE_ADVANTAGE_SCALE_GRID:
            raise ValueError("Phase 6AI advantage scale is unsupported")
        for name, value in asdict(self).items():
            if name == "hidden_dim":
                continue
            if float(value) < 0 or not math.isfinite(float(value)):
                raise ValueError(f"Phase 6AI selection metric {name} is invalid")
        if not self.final_training_loss < self.initial_training_loss:
            raise ValueError("Phase 6AI training loss did not decrease")


@dataclass(frozen=True, slots=True)
class BranchValueModel:
    source_dataset_fingerprint: str
    root_normalizer: NamedFeatureNormalizer
    branch_normalizer: NamedFeatureNormalizer
    seed: int
    epochs: int
    learning_rate: float
    selected_hidden_dim: int
    selected_weight_decay: float
    selected_contrast_weight: float
    selected_mean_blend: float
    selected_advantage_scale: float
    selection_metrics: tuple[BranchValueSelectionMetric, ...]
    train_root_ids: tuple[str, ...]
    calibration_selection_root_ids: tuple[str, ...]
    tensors: tuple[TensorArtifact, ...]
    grouped_five_action_objective: bool = True
    public_branch_features_only: bool = True
    audit_labels_as_features: bool = False
    terminal_rollout_results_as_features: bool = False
    official_engine_executed: bool = False
    equilibrium_trained: bool = False
    counterfactual_value_provider: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.source_dataset_fingerprint, "source_dataset_fingerprint")
        if self.seed != PHASE6_BRANCH_VALUE_SEED or self.epochs != PHASE6_BRANCH_VALUE_EPOCHS:
            raise ValueError("Phase 6AI frozen model configuration drifted")
        if self.learning_rate != PHASE6_BRANCH_VALUE_LEARNING_RATE:
            raise ValueError("Phase 6AI learning rate drifted")
        if not self.selection_metrics or not self.tensors:
            raise ValueError("Phase 6AI model diagnostics/tensors are empty")
        selected = min(
            self.selection_metrics,
            key=lambda item: (
                item.selection_score,
                item.selection_mse,
                item.selection_action_contrast_mse,
                item.hidden_dim,
                item.weight_decay,
                item.contrast_weight,
                -item.mean_blend,
                item.advantage_scale,
            ),
        )
        if (
            self.selected_hidden_dim,
            self.selected_weight_decay,
            self.selected_contrast_weight,
            self.selected_mean_blend,
            self.selected_advantage_scale,
        ) != (
            selected.hidden_dim,
            selected.weight_decay,
            selected.contrast_weight,
            selected.mean_blend,
            selected.advantage_scale,
        ):
            raise ValueError("Phase 6AI selected model record is inconsistent")
        if not self.train_root_ids or not self.calibration_selection_root_ids:
            raise ValueError("Phase 6AI model split roots are empty")
        if set(self.train_root_ids) & set(self.calibration_selection_root_ids):
            raise ValueError("Phase 6AI training/selection roots overlap")
        if any(
            (
                not self.grouped_five_action_objective,
                not self.public_branch_features_only,
                self.audit_labels_as_features,
                self.terminal_rollout_results_as_features,
                self.official_engine_executed,
                self.equilibrium_trained,
                self.counterfactual_value_provider,
            )
        ):
            raise ValueError("Phase 6AI model promoted its abstraction boundary")
        _restore_network(
            self.root_normalizer,
            self.branch_normalizer,
            self.selected_hidden_dim,
            self.tensors,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_VALUE_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "root_normalizer": self.root_normalizer.to_dict(),
            "branch_normalizer": self.branch_normalizer.to_dict(),
            "seed": self.seed,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "selected": {
                "hidden_dim": self.selected_hidden_dim,
                "weight_decay": self.selected_weight_decay,
                "contrast_weight": self.selected_contrast_weight,
                "mean_blend": self.selected_mean_blend,
                "advantage_scale": self.selected_advantage_scale,
            },
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
                "grouped_five_action_objective": self.grouped_five_action_objective,
                "public_branch_features_only": self.public_branch_features_only,
                "audit_labels_as_features": self.audit_labels_as_features,
                "terminal_rollout_results_as_features": (
                    self.terminal_rollout_results_as_features
                ),
                "official_engine_executed": self.official_engine_executed,
                "equilibrium_trained": self.equilibrium_trained,
                "counterfactual_value_provider": self.counterfactual_value_provider,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchValueModel":
        selected = raw["selected"]
        claims = raw["claims"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            root_normalizer=NamedFeatureNormalizer.from_dict(raw["root_normalizer"]),
            branch_normalizer=NamedFeatureNormalizer.from_dict(raw["branch_normalizer"]),
            seed=int(raw["seed"]),
            epochs=int(raw["epochs"]),
            learning_rate=float(raw["learning_rate"]),
            selected_hidden_dim=int(selected["hidden_dim"]),
            selected_weight_decay=float(selected["weight_decay"]),
            selected_contrast_weight=float(selected["contrast_weight"]),
            selected_mean_blend=float(selected["mean_blend"]),
            selected_advantage_scale=float(selected["advantage_scale"]),
            selection_metrics=tuple(
                BranchValueSelectionMetric(**dict(item))
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
            grouped_five_action_objective=bool(
                claims["grouped_five_action_objective"]
            ),
            public_branch_features_only=bool(claims["public_branch_features_only"]),
            audit_labels_as_features=bool(claims["audit_labels_as_features"]),
            terminal_rollout_results_as_features=bool(
                claims["terminal_rollout_results_as_features"]
            ),
            official_engine_executed=bool(claims["official_engine_executed"]),
            equilibrium_trained=bool(claims["equilibrium_trained"]),
            counterfactual_value_provider=bool(
                claims["counterfactual_value_provider"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AI model fingerprint mismatch")
        return result


class BranchValueProvider:
    def __init__(self, model: BranchValueModel) -> None:
        self.model = model
        self.network = _restore_network(
            model.root_normalizer,
            model.branch_normalizer,
            model.selected_hidden_dim,
            model.tensors,
        )

    def predict_examples(
        self,
        examples: Sequence[BranchValueRootExample],
    ) -> list[list[float]]:
        return _network_predictions(
            self.network,
            examples,
            self.model.root_normalizer,
            self.model.branch_normalizer,
            mean_blend=self.model.selected_mean_blend,
            advantage_scale=self.model.selected_advantage_scale,
        )

    def predict_example(self, example: BranchValueRootExample) -> tuple[float, ...]:
        return tuple(self.predict_examples((example,))[0])


def restore_branch_value_model(model: BranchValueModel) -> BranchValueProvider:
    return BranchValueProvider(model)


def train_branch_value_model(
    dataset: BranchConditionedDataset,
) -> BranchValueModel:
    examples = build_branch_value_examples(dataset)
    train = tuple(item for item in examples if item.split == "train")
    selection = tuple(
        item
        for item in examples
        if item.split == "calibration" and item.calibration_partition == "selection"
    )
    if len(train) != 72 or len(selection) != 36:
        raise ValueError("Phase 6AI train/selection root counts drifted")
    root_normalizer = NamedFeatureNormalizer.fit(
        [item.root_features for item in train],
        fit_rows_fingerprint=_fingerprint([item.root_id for item in train]),
    )
    branch_rows = [action.branch_features for item in train for action in item.actions]
    branch_normalizer = NamedFeatureNormalizer.fit(
        branch_rows,
        fit_rows_fingerprint=_fingerprint(
            [(item.root_id, action.action_id) for item in train for action in item.actions]
        ),
    )
    selection_targets = _targets(selection)
    zero = [[0.0] * len(OPPONENT_DEVIATION_ACTION_IDS) for _ in selection]
    root_heuristic = _root_heuristic_predictions(selection)
    branch_heuristic = _branch_heuristic_predictions(selection)
    zero_mse = _mse(selection_targets, zero)
    root_mse = _mse(selection_targets, root_heuristic)
    branch_mse = _mse(selection_targets, branch_heuristic)
    zero_contrast = _contrast_mse(selection_targets, zero)
    branch_contrast = _contrast_mse(selection_targets, branch_heuristic)
    candidates: list[tuple[BranchValueSelectionMetric, _BranchDuelingNetwork]] = []
    for hidden_dim in PHASE6_BRANCH_VALUE_HIDDEN_GRID:
        for weight_decay in PHASE6_BRANCH_VALUE_WEIGHT_DECAY_GRID:
            for contrast_weight in PHASE6_BRANCH_VALUE_CONTRAST_WEIGHT_GRID:
                candidate_seed = (
                    PHASE6_BRANCH_VALUE_SEED
                    + hidden_dim
                    + int(weight_decay * 100_000)
                    + int(contrast_weight * 10)
                )
                network, initial, final = _train_network(
                    train,
                    root_normalizer,
                    branch_normalizer,
                    hidden_dim=hidden_dim,
                    weight_decay=weight_decay,
                    contrast_weight=contrast_weight,
                    seed=candidate_seed,
                )
                for mean_blend in PHASE6_BRANCH_VALUE_MEAN_BLEND_GRID:
                    for advantage_scale in PHASE6_BRANCH_VALUE_ADVANTAGE_SCALE_GRID:
                        predictions = _network_predictions(
                            network,
                            selection,
                            root_normalizer,
                            branch_normalizer,
                            mean_blend=mean_blend,
                            advantage_scale=advantage_scale,
                        )
                        mse = _mse(selection_targets, predictions)
                        contrast = _contrast_mse(selection_targets, predictions)
                        point_reference = max(
                            min(zero_mse, root_mse, branch_mse),
                            1e-12,
                        )
                        contrast_reference = max(
                            min(zero_contrast, branch_contrast),
                            1e-12,
                        )
                        metric = BranchValueSelectionMetric(
                            hidden_dim=hidden_dim,
                            weight_decay=weight_decay,
                            contrast_weight=contrast_weight,
                            mean_blend=mean_blend,
                            advantage_scale=advantage_scale,
                            initial_training_loss=initial,
                            final_training_loss=final,
                            selection_mse=mse,
                            selection_zero_mse=zero_mse,
                            selection_root_heuristic_mse=root_mse,
                            selection_branch_heuristic_mse=branch_mse,
                            selection_action_contrast_mse=contrast,
                            selection_zero_action_contrast_mse=zero_contrast,
                            selection_branch_action_contrast_mse=branch_contrast,
                            selection_score=max(
                                mse / point_reference,
                                contrast
                                / (
                                    PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR
                                    * contrast_reference
                                ),
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
            item[0].contrast_weight,
            -item[0].mean_blend,
            item[0].advantage_scale,
        ),
    )
    model = BranchValueModel(
        source_dataset_fingerprint=dataset.fingerprint,
        root_normalizer=root_normalizer,
        branch_normalizer=branch_normalizer,
        seed=PHASE6_BRANCH_VALUE_SEED,
        epochs=PHASE6_BRANCH_VALUE_EPOCHS,
        learning_rate=PHASE6_BRANCH_VALUE_LEARNING_RATE,
        selected_hidden_dim=selected_metric.hidden_dim,
        selected_weight_decay=selected_metric.weight_decay,
        selected_contrast_weight=selected_metric.contrast_weight,
        selected_mean_blend=selected_metric.mean_blend,
        selected_advantage_scale=selected_metric.advantage_scale,
        selection_metrics=tuple(metric for metric, _ in candidates),
        train_root_ids=tuple(sorted(item.root_id for item in train)),
        calibration_selection_root_ids=tuple(
            sorted(item.root_id for item in selection)
        ),
        tensors=_network_tensors(selected_network),
    )
    restored = BranchValueModel.from_dict(json.loads(model.to_json()))
    if restored != model:
        raise RuntimeError("Phase 6AI model roundtrip changed values")
    return model


@dataclass(frozen=True, slots=True)
class BranchValuePointMetric:
    role: str
    root_count: int
    row_count: int
    mse: float
    zero_mse: float
    root_heuristic_mse: float
    branch_heuristic_mse: float
    action_contrast_mse: float
    zero_action_contrast_mse: float
    branch_action_contrast_mse: float
    mean_selected_action_value: float
    branch_heuristic_selected_action_value: float
    oracle_action_value: float
    mean_selected_action_regret: float
    pairwise_concordance: float

    def __post_init__(self) -> None:
        if not self.role or min(self.root_count, self.row_count) <= 0:
            raise ValueError("Phase 6AI point metric identity/count is invalid")
        for name, value in asdict(self).items():
            if name in {"role", "root_count", "row_count"}:
                continue
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Phase 6AI point metric {name} is nonfinite")
            if name.endswith("mse") or name.endswith("regret"):
                if value < -1e-12:
                    raise ValueError(f"Phase 6AI point metric {name} is negative")
            elif name == "pairwise_concordance" and not 0 <= value <= 1:
                raise ValueError("Phase 6AI concordance is invalid")
            elif name.endswith("action_value") and not -1 <= value <= 1:
                raise ValueError(f"Phase 6AI point metric {name} is outside utility range")


@dataclass(frozen=True, slots=True)
class BranchValueSliceMetric:
    dimension: str
    slice_id: str
    validation_row_count: int
    validation_mse: float
    validation_coverage: float
    validation_mean_width: float

    def __post_init__(self) -> None:
        if self.dimension not in {"action", "candidate", "intent"} or not self.slice_id:
            raise ValueError("Phase 6AI slice identity is invalid")
        if self.validation_row_count <= 0 or self.validation_mse < 0:
            raise ValueError("Phase 6AI slice count/MSE is invalid")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("Phase 6AI slice coverage is invalid")
        if not 0 <= self.validation_mean_width <= 2:
            raise ValueError("Phase 6AI slice width is invalid")


def _pairwise_concordance(
    targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
) -> float:
    correct = 0.0
    count = 0
    for target, prediction in zip(targets, predictions, strict=True):
        for left in range(len(target)):
            for right in range(left + 1, len(target)):
                target_delta = target[left] - target[right]
                predicted_delta = prediction[left] - prediction[right]
                if target_delta == 0:
                    continue
                count += 1
                if predicted_delta == 0:
                    correct += 0.5
                elif (target_delta > 0) == (predicted_delta > 0):
                    correct += 1.0
    return 0.5 if count == 0 else correct / count


def _point_metric(
    role: str,
    examples: Sequence[BranchValueRootExample],
    predictions: Sequence[Sequence[float]],
) -> BranchValuePointMetric:
    targets = _targets(examples)
    zero = [[0.0] * len(OPPONENT_DEVIATION_ACTION_IDS) for _ in examples]
    root_heuristic = _root_heuristic_predictions(examples)
    branch_heuristic = _branch_heuristic_predictions(examples)
    selected_values = []
    branch_values = []
    oracle_values = []
    regrets = []
    for target, predicted, branch in zip(
        targets,
        predictions,
        branch_heuristic,
        strict=True,
    ):
        selected_index = max(range(len(predicted)), key=lambda index: (predicted[index], -index))
        branch_index = max(range(len(branch)), key=lambda index: (branch[index], -index))
        oracle = max(target)
        selected_values.append(target[selected_index])
        branch_values.append(target[branch_index])
        oracle_values.append(oracle)
        regrets.append(oracle - target[selected_index])
    return BranchValuePointMetric(
        role=role,
        root_count=len(examples),
        row_count=len(examples) * len(OPPONENT_DEVIATION_ACTION_IDS),
        mse=_mse(targets, predictions),
        zero_mse=_mse(targets, zero),
        root_heuristic_mse=_mse(targets, root_heuristic),
        branch_heuristic_mse=_mse(targets, branch_heuristic),
        action_contrast_mse=_contrast_mse(targets, predictions),
        zero_action_contrast_mse=_contrast_mse(targets, zero),
        branch_action_contrast_mse=_contrast_mse(targets, branch_heuristic),
        mean_selected_action_value=statistics.mean(selected_values),
        branch_heuristic_selected_action_value=statistics.mean(branch_values),
        oracle_action_value=statistics.mean(oracle_values),
        mean_selected_action_regret=statistics.mean(regrets),
        pairwise_concordance=_pairwise_concordance(targets, predictions),
    )


def _bootstrap_contrast_lower_bounds(
    targets: Sequence[Sequence[float]],
    predictions: Sequence[Sequence[float]],
    branch_predictions: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    model_errors = _contrast_root_errors(targets, predictions)
    zero_errors = _contrast_root_errors(
        targets,
        [[0.0] * len(OPPONENT_DEVIATION_ACTION_IDS) for _ in targets],
    )
    branch_errors = _contrast_root_errors(targets, branch_predictions)
    zero_improvements = [
        baseline - model
        for baseline, model in zip(zero_errors, model_errors, strict=True)
    ]
    branch_improvements = [
        baseline - model
        for baseline, model in zip(branch_errors, model_errors, strict=True)
    ]
    rng = random.Random(PHASE6_BRANCH_VALUE_BOOTSTRAP_SEED)
    zero_means = []
    branch_means = []
    for _ in range(PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES):
        indices = [rng.randrange(len(targets)) for _ in targets]
        zero_means.append(statistics.mean(zero_improvements[index] for index in indices))
        branch_means.append(statistics.mean(branch_improvements[index] for index in indices))
    zero_means.sort()
    branch_means.sort()
    lower_index = max(
        0,
        math.ceil(
            (1.0 - PHASE6_BRANCH_VALUE_COVERAGE)
            * PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES
        )
        - 1,
    )
    zero_lower = zero_means[lower_index]
    branch_lower = branch_means[lower_index]
    return zero_lower, branch_lower, min(zero_lower, branch_lower)


def _conformal_radius(
    examples: Sequence[BranchValueRootExample],
    predictions: Sequence[Sequence[float]],
) -> float:
    residuals = sorted(
        abs(prediction - action.empirical_player0_value)
        + PHASE6_EXPECTED_VALUE_MEAN_CI_Z * action.target_standard_error
        for example, group in zip(examples, predictions, strict=True)
        for action, prediction in zip(example.actions, group, strict=True)
    )
    if not residuals:
        raise ValueError("Phase 6AI conformal residual set is empty")
    rank = min(
        len(residuals),
        math.ceil((len(residuals) + 1) * PHASE6_BRANCH_VALUE_COVERAGE),
    )
    return min(2.0, residuals[rank - 1])


def _interval_result(
    action: BranchValueActionTarget,
    prediction: float,
    radius: float,
) -> tuple[bool, float]:
    predicted_lower = max(-1.0, prediction - radius)
    predicted_upper = min(1.0, prediction + radius)
    target_lower = max(
        -1.0,
        action.empirical_player0_value
        - PHASE6_EXPECTED_VALUE_MEAN_CI_Z * action.target_standard_error,
    )
    target_upper = min(
        1.0,
        action.empirical_player0_value
        + PHASE6_EXPECTED_VALUE_MEAN_CI_Z * action.target_standard_error,
    )
    covered = (
        predicted_lower <= target_lower + 1e-12
        and target_upper <= predicted_upper + 1e-12
    )
    return covered, predicted_upper - predicted_lower


@dataclass(frozen=True, slots=True)
class BranchValueCalibration:
    source_dataset_fingerprint: str
    source_model_fingerprint: str
    calibration_bound_root_ids: tuple[str, ...]
    validation_root_ids: tuple[str, ...]
    conformal_radius: float
    point_metrics: tuple[BranchValuePointMetric, ...]
    slice_metrics: tuple[BranchValueSliceMetric, ...]
    maximum_target_standard_error: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    zero_contrast_bootstrap_lower_bound: float
    branch_contrast_bootstrap_lower_bound: float
    minimum_contrast_bootstrap_lower_bound: float
    sampling_precision_gate_passed: bool
    point_mse_gate_passed: bool
    contrast_relative_gate_passed: bool
    contrast_bootstrap_gate_passed: bool
    interval_gate_passed: bool
    implementation_holdout_gate_passed: bool
    action_sensitive_point_model_validated: bool
    conditional_expected_mean_model_admitted: bool
    counterfactual_value_compatible: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("source_dataset_fingerprint", "source_model_fingerprint"):
            _validate_digest(getattr(self, name), name)
        if not self.calibration_bound_root_ids or not self.validation_root_ids:
            raise ValueError("Phase 6AI bound/validation root ledgers are empty")
        if set(self.calibration_bound_root_ids) & set(self.validation_root_ids):
            raise ValueError("Phase 6AI bound/validation roots overlap")
        if tuple(item.role for item in self.point_metrics) != (
            "calibration_selection",
            "calibration_bounds",
            "validation_untouched",
        ):
            raise ValueError("Phase 6AI point metrics are incomplete")
        if not self.slice_metrics or not 0 <= self.conformal_radius <= 2:
            raise ValueError("Phase 6AI interval diagnostics are invalid")
        validation = self.point_metrics[-1]
        expected_sampling = self.maximum_target_standard_error <= (
            PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
        )
        expected_point = validation.mse < min(
            validation.zero_mse,
            validation.root_heuristic_mse,
            validation.branch_heuristic_mse,
        )
        expected_contrast = validation.action_contrast_mse < (
            PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR
            * min(
                validation.zero_action_contrast_mse,
                validation.branch_action_contrast_mse,
            )
        )
        expected_bootstrap = self.minimum_contrast_bootstrap_lower_bound > 0
        expected_interval = (
            self.validation_coverage >= PHASE6_BRANCH_VALUE_COVERAGE
            and self.validation_mean_width
            <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
            and self.minimum_validation_slice_coverage
            >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
        )
        if self.sampling_precision_gate_passed != expected_sampling:
            raise ValueError("Phase 6AI sampling gate is inconsistent")
        if self.point_mse_gate_passed != expected_point:
            raise ValueError("Phase 6AI point gate is inconsistent")
        if self.contrast_relative_gate_passed != expected_contrast:
            raise ValueError("Phase 6AI contrast gate is inconsistent")
        if self.contrast_bootstrap_gate_passed != expected_bootstrap:
            raise ValueError("Phase 6AI bootstrap gate is inconsistent")
        if self.interval_gate_passed != expected_interval:
            raise ValueError("Phase 6AI interval gate is inconsistent")
        expected_point_validation = all(
            (
                expected_sampling,
                expected_point,
                expected_contrast,
                expected_bootstrap,
                self.implementation_holdout_gate_passed,
            )
        )
        if self.action_sensitive_point_model_validated != expected_point_validation:
            raise ValueError("Phase 6AI action-sensitive point admission is inconsistent")
        if self.conditional_expected_mean_model_admitted != (
            expected_point_validation and expected_interval
        ):
            raise ValueError("Phase 6AI conditional-mean admission is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AI calibration promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_VALUE_SCHEMA_VERSION,
            "source_dataset_fingerprint": self.source_dataset_fingerprint,
            "source_model_fingerprint": self.source_model_fingerprint,
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
            "bootstrap": {
                "resamples": PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES,
                "zero_contrast_lower_bound": (
                    self.zero_contrast_bootstrap_lower_bound
                ),
                "branch_contrast_lower_bound": (
                    self.branch_contrast_bootstrap_lower_bound
                ),
                "minimum_lower_bound": self.minimum_contrast_bootstrap_lower_bound,
            },
            "gates": {
                "sampling_precision_gate_passed": self.sampling_precision_gate_passed,
                "point_mse_gate_passed": self.point_mse_gate_passed,
                "contrast_relative_gate_passed": self.contrast_relative_gate_passed,
                "contrast_bootstrap_gate_passed": self.contrast_bootstrap_gate_passed,
                "interval_gate_passed": self.interval_gate_passed,
                "implementation_holdout_gate_passed": (
                    self.implementation_holdout_gate_passed
                ),
                "action_sensitive_point_model_validated": (
                    self.action_sensitive_point_model_validated
                ),
                "conditional_expected_mean_model_admitted": (
                    self.conditional_expected_mean_model_admitted
                ),
                "counterfactual_value_compatible": self.counterfactual_value_compatible,
                "equilibrium_convergence_validated": (
                    self.equilibrium_convergence_validated
                ),
                "safe_resolver_enabled": self.safe_resolver_enabled,
                "opponent_exploitation_enabled": self.opponent_exploitation_enabled,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BranchValueCalibration":
        bootstrap = raw["bootstrap"]
        if int(bootstrap["resamples"]) != PHASE6_BRANCH_VALUE_BOOTSTRAP_RESAMPLES:
            raise ValueError("Phase 6AI bootstrap count drifted")
        gates = raw["gates"]
        result = cls(
            source_dataset_fingerprint=str(raw["source_dataset_fingerprint"]),
            source_model_fingerprint=str(raw["source_model_fingerprint"]),
            calibration_bound_root_ids=tuple(
                str(value) for value in raw["calibration_bound_root_ids"]
            ),
            validation_root_ids=tuple(str(value) for value in raw["validation_root_ids"]),
            conformal_radius=float(raw["conformal_radius"]),
            point_metrics=tuple(
                BranchValuePointMetric(**dict(item)) for item in raw["point_metrics"]
            ),
            slice_metrics=tuple(
                BranchValueSliceMetric(**dict(item)) for item in raw["slice_metrics"]
            ),
            maximum_target_standard_error=float(raw["maximum_target_standard_error"]),
            validation_coverage=float(raw["validation_coverage"]),
            validation_mean_width=float(raw["validation_mean_width"]),
            minimum_validation_slice_coverage=float(
                raw["minimum_validation_slice_coverage"]
            ),
            zero_contrast_bootstrap_lower_bound=float(
                bootstrap["zero_contrast_lower_bound"]
            ),
            branch_contrast_bootstrap_lower_bound=float(
                bootstrap["branch_contrast_lower_bound"]
            ),
            minimum_contrast_bootstrap_lower_bound=float(
                bootstrap["minimum_lower_bound"]
            ),
            sampling_precision_gate_passed=bool(
                gates["sampling_precision_gate_passed"]
            ),
            point_mse_gate_passed=bool(gates["point_mse_gate_passed"]),
            contrast_relative_gate_passed=bool(
                gates["contrast_relative_gate_passed"]
            ),
            contrast_bootstrap_gate_passed=bool(
                gates["contrast_bootstrap_gate_passed"]
            ),
            interval_gate_passed=bool(gates["interval_gate_passed"]),
            implementation_holdout_gate_passed=bool(
                gates["implementation_holdout_gate_passed"]
            ),
            action_sensitive_point_model_validated=bool(
                gates["action_sensitive_point_model_validated"]
            ),
            conditional_expected_mean_model_admitted=bool(
                gates["conditional_expected_mean_model_admitted"]
            ),
            counterfactual_value_compatible=bool(
                gates["counterfactual_value_compatible"]
            ),
            equilibrium_convergence_validated=bool(
                gates["equilibrium_convergence_validated"]
            ),
            safe_resolver_enabled=bool(gates["safe_resolver_enabled"]),
            opponent_exploitation_enabled=bool(
                gates["opponent_exploitation_enabled"]
            ),
            official_strategy_control_enabled=bool(
                gates["official_strategy_control_enabled"]
            ),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AI calibration fingerprint mismatch")
        return result


def calibrate_branch_value_model(
    dataset: BranchConditionedDataset,
    model: BranchValueModel,
) -> BranchValueCalibration:
    if model.source_dataset_fingerprint != dataset.fingerprint:
        raise ValueError("Phase 6AI model/dataset mismatch")
    examples = build_branch_value_examples(dataset)
    selection = tuple(
        item
        for item in examples
        if item.split == "calibration" and item.calibration_partition == "selection"
    )
    bounds = tuple(
        item
        for item in examples
        if item.split == "calibration" and item.calibration_partition == "bounds"
    )
    validation = tuple(item for item in examples if item.split == "validation")
    if tuple(map(len, (selection, bounds, validation))) != (36, 36, 36):
        raise ValueError("Phase 6AI calibration/validation root counts drifted")
    provider = restore_branch_value_model(model)
    selection_predictions = provider.predict_examples(selection)
    bound_predictions = provider.predict_examples(bounds)
    validation_predictions = provider.predict_examples(validation)
    radius = _conformal_radius(bounds, bound_predictions)
    covered: list[bool] = []
    widths: list[float] = []
    for example, predictions in zip(validation, validation_predictions, strict=True):
        for action, prediction in zip(example.actions, predictions, strict=True):
            item_covered, width = _interval_result(action, prediction, radius)
            covered.append(item_covered)
            widths.append(width)
    slice_metrics = []
    flat = [
        (example, action, prediction)
        for example, predictions in zip(validation, validation_predictions, strict=True)
        for action, prediction in zip(example.actions, predictions, strict=True)
    ]
    slice_specs = (
        ("action", OPPONENT_DEVIATION_ACTION_IDS, lambda example, action: action.action_id),
        (
            "candidate",
            dataset.value_dataset.candidate_ids,
            lambda example, action: example.candidate_id_hypothesis_input,
        ),
        (
            "intent",
            dataset.value_dataset.intent_ids,
            lambda example, action: example.intent_id_supervision_only,
        ),
    )
    for dimension, identifiers, selector in slice_specs:
        for identifier in identifiers:
            indices = [
                index
                for index, (example, action, _) in enumerate(flat)
                if selector(example, action) == identifier
            ]
            if not indices:
                raise RuntimeError(f"Phase 6AI empty slice: {dimension}/{identifier}")
            slice_metrics.append(
                BranchValueSliceMetric(
                    dimension=dimension,
                    slice_id=identifier,
                    validation_row_count=len(indices),
                    validation_mse=statistics.mean(
                        (
                            flat[index][2]
                            - flat[index][1].empirical_player0_value
                        )
                        ** 2
                        for index in indices
                    ),
                    validation_coverage=(
                        sum(covered[index] for index in indices) / len(indices)
                    ),
                    validation_mean_width=statistics.mean(
                        widths[index] for index in indices
                    ),
                )
            )
    metrics = (
        _point_metric("calibration_selection", selection, selection_predictions),
        _point_metric("calibration_bounds", bounds, bound_predictions),
        _point_metric("validation_untouched", validation, validation_predictions),
    )
    validation_metric = metrics[-1]
    zero_lower, branch_lower, minimum_lower = _bootstrap_contrast_lower_bounds(
        _targets(validation),
        validation_predictions,
        _branch_heuristic_predictions(validation),
    )
    maximum_se = max(
        action.target_standard_error
        for example in examples
        for action in example.actions
    )
    overall_coverage = sum(covered) / len(covered)
    overall_width = statistics.mean(widths)
    minimum_slice = min(item.validation_coverage for item in slice_metrics)
    sampling_gate = maximum_se <= PHASE6_EXPECTED_VALUE_MAXIMUM_TARGET_STANDARD_ERROR
    point_gate = validation_metric.mse < min(
        validation_metric.zero_mse,
        validation_metric.root_heuristic_mse,
        validation_metric.branch_heuristic_mse,
    )
    contrast_gate = validation_metric.action_contrast_mse < (
        PHASE6_BRANCH_VALUE_CONTRAST_RELATIVE_FACTOR
        * min(
            validation_metric.zero_action_contrast_mse,
            validation_metric.branch_action_contrast_mse,
        )
    )
    bootstrap_gate = minimum_lower > 0
    interval_gate = (
        overall_coverage >= PHASE6_BRANCH_VALUE_COVERAGE
        and overall_width <= PHASE6_EXPECTED_VALUE_MAXIMUM_INTERVAL_WIDTH
        and minimum_slice >= PHASE6_EXPECTED_VALUE_MINIMUM_SLICE_COVERAGE
    )
    train_roots = {item.root_id for item in examples if item.split == "train"}
    selection_roots = {item.root_id for item in selection}
    bound_roots = {item.root_id for item in bounds}
    validation_roots = {item.root_id for item in validation}
    implementation_gate = (
        not any(
            (
                train_roots & selection_roots,
                train_roots & bound_roots,
                train_roots & validation_roots,
                selection_roots & bound_roots,
                selection_roots & validation_roots,
                bound_roots & validation_roots,
            )
        )
        and {item.policy_variant_id_supervision_only for item in validation}
        == {"action_first"}
        and {item.policy_variant_id_supervision_only for item in bounds}
        == {"hand_first"}
        and set(model.train_root_ids) == train_roots
        and set(model.calibration_selection_root_ids) == selection_roots
    )
    point_admitted = all(
        (
            sampling_gate,
            point_gate,
            contrast_gate,
            bootstrap_gate,
            implementation_gate,
        )
    )
    calibration = BranchValueCalibration(
        source_dataset_fingerprint=dataset.fingerprint,
        source_model_fingerprint=model.fingerprint,
        calibration_bound_root_ids=tuple(sorted(bound_roots)),
        validation_root_ids=tuple(sorted(validation_roots)),
        conformal_radius=radius,
        point_metrics=metrics,
        slice_metrics=tuple(slice_metrics),
        maximum_target_standard_error=maximum_se,
        validation_coverage=overall_coverage,
        validation_mean_width=overall_width,
        minimum_validation_slice_coverage=minimum_slice,
        zero_contrast_bootstrap_lower_bound=zero_lower,
        branch_contrast_bootstrap_lower_bound=branch_lower,
        minimum_contrast_bootstrap_lower_bound=minimum_lower,
        sampling_precision_gate_passed=sampling_gate,
        point_mse_gate_passed=point_gate,
        contrast_relative_gate_passed=contrast_gate,
        contrast_bootstrap_gate_passed=bootstrap_gate,
        interval_gate_passed=interval_gate,
        implementation_holdout_gate_passed=implementation_gate,
        action_sensitive_point_model_validated=point_admitted,
        conditional_expected_mean_model_admitted=(point_admitted and interval_gate),
    )
    restored = BranchValueCalibration.from_dict(json.loads(calibration.to_json()))
    if restored != calibration:
        raise RuntimeError("Phase 6AI calibration roundtrip changed values")
    return calibration


@dataclass(frozen=True, slots=True)
class Phase6BranchValueEvidence:
    status: str
    dataset_fingerprint: str
    model_fingerprint: str
    calibration_fingerprint: str
    root_count: int
    branch_observation_count: int
    surface_count: int
    action_target_count: int
    terminal_rollout_count: int
    live_selection_count: int
    native_transition_count: int
    validation_mse: float
    validation_zero_mse: float
    validation_root_heuristic_mse: float
    validation_branch_heuristic_mse: float
    validation_action_contrast_mse: float
    validation_zero_action_contrast_mse: float
    validation_branch_action_contrast_mse: float
    minimum_contrast_bootstrap_lower_bound: float
    validation_coverage: float
    validation_mean_width: float
    minimum_validation_slice_coverage: float
    maximum_target_standard_error: float
    sampling_precision_gate_passed: bool
    action_sensitive_point_model_validated: bool
    conditional_expected_mean_model_admitted: bool
    counterfactual_value_compatible: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in ("dataset_fingerprint", "model_fingerprint", "calibration_fingerprint"):
            _validate_digest(getattr(self, name), name)
        if (
            self.root_count != 180
            or self.branch_observation_count != 1_800
            or self.surface_count != 360
            or self.action_target_count != 900
            or self.terminal_rollout_count != 14_400
        ):
            raise ValueError("Phase 6AI evidence accounting is incomplete")
        expected_status = (
            "fresh_branch_conditional_expected_mean_admitted_downstream_fail_closed"
            if self.conditional_expected_mean_model_admitted
            else (
                "fresh_action_sensitive_point_signal_validated_intervals_rejected_downstream_fail_closed"
                if self.action_sensitive_point_model_validated
                else "fresh_branch_conditioned_values_rejected_downstream_fail_closed"
            )
        )
        if self.status != expected_status:
            raise ValueError("Phase 6AI evidence status is inconsistent")
        if any(
            (
                self.counterfactual_value_compatible,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AI evidence promoted downstream control")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BRANCH_VALUE_SCHEMA_VERSION,
            **asdict(self),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6BranchValueEvidence":
        values = dict(raw)
        values.pop("schema_version", None)
        fingerprint = values.pop("fingerprint", None)
        result = cls(**values)
        if fingerprint not in (None, result.fingerprint):
            raise ValueError("Phase 6AI evidence fingerprint mismatch")
        return result

    def to_markdown(self) -> str:
        return f"""# Phase 6AI causal branch-conditioned values

Status: `{self.status}`  
Evidence fingerprint: `{self.fingerprint}`

| Metric | Result |
|---|---:|
| Complete official matches / fresh roots | {self.root_count} / {self.root_count} |
| Immediate native branch observations | {self.branch_observation_count} |
| Supplied-allocation surfaces / root-action targets | {self.surface_count} / {self.action_target_count} |
| Genuine terminal native continuations | {self.terminal_rollout_count} |
| Genuine live selections / native transitions | {self.live_selection_count} / {self.native_transition_count} |
| Validation neural / zero / root heuristic / branch heuristic MSE | `{self.validation_mse:.6f}` / `{self.validation_zero_mse:.6f}` / `{self.validation_root_heuristic_mse:.6f}` / `{self.validation_branch_heuristic_mse:.6f}` |
| Validation contrast neural / zero / branch heuristic MSE | `{self.validation_action_contrast_mse:.6f}` / `{self.validation_zero_action_contrast_mse:.6f}` / `{self.validation_branch_action_contrast_mse:.6f}` |
| Minimum 90% bootstrap contrast-improvement lower bound | `{self.minimum_contrast_bootstrap_lower_bound:.6f}` |
| Validation mean-CI coverage / width / worst slice | `{self.validation_coverage:.6f}` / `{self.validation_mean_width:.6f}` / `{self.minimum_validation_slice_coverage:.6f}` |
| Maximum target SE | `{self.maximum_target_standard_error:.6f}` |
| Sampling precision / action-sensitive point / conditional mean | **{str(self.sampling_precision_gate_passed).lower()} / {str(self.action_sensitive_point_model_validated).lower()} / {str(self.conditional_expected_mean_model_admitted).lower()}** |
| Counterfactual / equilibrium / safe resolver / exploitation / control | **false / false / false / false / false** |

Complete matches, legal live selections, retained snapshot replay, immediate
branches, and terminal descendants are genuine official-engine execution. The
allocation law, plan boundary, branch aggregation, model, bootstrap, and
intervals are abstractions. No Kaggle submission or strength claim is enabled.
"""


@dataclass(frozen=True, slots=True)
class Phase6BranchValueArtifacts:
    dataset: BranchConditionedDataset
    model: BranchValueModel
    calibration: BranchValueCalibration
    evidence: Phase6BranchValueEvidence


def run_phase6_branch_conditioned_values(
    *,
    project_root: str | Path | None = None,
    source_dataset: BranchConditionedDataset | None = None,
    existing_roots: Sequence[BranchConditionedRoot] = (),
    root_observer: Callable[[BranchConditionedRoot, int, int], None] | None = None,
) -> Phase6BranchValueArtifacts:
    dataset = source_dataset or collect_branch_conditioned_dataset(
        project_root=project_root,
        existing_roots=existing_roots,
        root_observer=root_observer,
    )
    model = train_branch_value_model(dataset)
    calibration = calibrate_branch_value_model(dataset, model)
    validation = calibration.point_metrics[-1]
    admitted = calibration.conditional_expected_mean_model_admitted
    point = calibration.action_sensitive_point_model_validated
    evidence = Phase6BranchValueEvidence(
        status=(
            "fresh_branch_conditional_expected_mean_admitted_downstream_fail_closed"
            if admitted
            else (
                "fresh_action_sensitive_point_signal_validated_intervals_rejected_downstream_fail_closed"
                if point
                else "fresh_branch_conditioned_values_rejected_downstream_fail_closed"
            )
        ),
        dataset_fingerprint=dataset.fingerprint,
        model_fingerprint=model.fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        root_count=len(dataset.roots),
        branch_observation_count=sum(
            len(item.branch_observations) for item in dataset.roots
        ),
        surface_count=sum(
            len(item.value_root.surfaces) for item in dataset.roots
        ),
        action_target_count=len(dataset.roots) * len(OPPONENT_DEVIATION_ACTION_IDS),
        terminal_rollout_count=dataset.terminal_rollout_count,
        live_selection_count=dataset.value_dataset.live_selection_count,
        native_transition_count=dataset.value_dataset.native_transition_count,
        validation_mse=validation.mse,
        validation_zero_mse=validation.zero_mse,
        validation_root_heuristic_mse=validation.root_heuristic_mse,
        validation_branch_heuristic_mse=validation.branch_heuristic_mse,
        validation_action_contrast_mse=validation.action_contrast_mse,
        validation_zero_action_contrast_mse=(
            validation.zero_action_contrast_mse
        ),
        validation_branch_action_contrast_mse=(
            validation.branch_action_contrast_mse
        ),
        minimum_contrast_bootstrap_lower_bound=(
            calibration.minimum_contrast_bootstrap_lower_bound
        ),
        validation_coverage=calibration.validation_coverage,
        validation_mean_width=calibration.validation_mean_width,
        minimum_validation_slice_coverage=(
            calibration.minimum_validation_slice_coverage
        ),
        maximum_target_standard_error=calibration.maximum_target_standard_error,
        sampling_precision_gate_passed=calibration.sampling_precision_gate_passed,
        action_sensitive_point_model_validated=point,
        conditional_expected_mean_model_admitted=admitted,
    )
    restored = Phase6BranchValueEvidence.from_dict(json.loads(evidence.to_json()))
    if restored != evidence:
        raise RuntimeError("Phase 6AI evidence roundtrip changed values")
    return Phase6BranchValueArtifacts(
        dataset=dataset,
        model=model,
        calibration=calibration,
        evidence=evidence,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "phase6_branch_conditioned_values"
        ),
    )
    parser.add_argument("--reuse-source", action="store_true")
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    dataset_path = target / "branch_conditioned_dataset.json"
    if args.reuse_source:
        dataset = BranchConditionedDataset.from_dict(
            json.loads(dataset_path.read_text(encoding="utf-8"))
        )
    else:
        dataset = collect_branch_conditioned_dataset()
    artifacts = run_phase6_branch_conditioned_values(source_dataset=dataset)
    target.mkdir(parents=True, exist_ok=True)
    (target / "branch_conditioned_dataset.json").write_text(
        artifacts.dataset.to_json() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "branch_value_model.json").write_text(
        artifacts.model.to_json() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "branch_value_calibration.json").write_text(
        artifacts.calibration.to_json() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "phase6ai_evidence.json").write_text(
        artifacts.evidence.to_json() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (target / "phase6ai_evidence.md").write_text(
        artifacts.evidence.to_markdown(),
        encoding="utf-8",
        newline="\n",
    )
    print(artifacts.evidence.to_json())


if __name__ == "__main__":
    main()
