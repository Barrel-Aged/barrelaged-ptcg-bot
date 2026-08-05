"""Phase 6K native counterfactual-regret blueprint-candidate evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .game_theory import AbstractAction, ActionDistribution, InformationSetActor
from .native_blueprint_features import (
    NATIVE_ACTION_FEATURE_NAMES,
    NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT,
    NATIVE_PRIVATE_FEATURE_NAMES,
    NATIVE_PUBLIC_FEATURE_NAMES,
    NativeBlueprintFeatureBridge,
    NativeBlueprintView,
)
from .native_regret_blueprint import (
    DEFAULT_NATIVE_REGRET_SEED,
    NativeRegretBlueprintArtifact,
    NativeRegretBlueprintProvider,
    NativeRegretDataset,
    NativeRegretSplitMetrics,
    NativeRegretTrainingRun,
    NativeRegretTrainingSample,
    native_regret_sample_view,
    restore_native_regret_blueprint,
    train_native_regret_blueprint,
)
from .official_evaluation import OFFICIAL_PANEL_DECKS, KaggleDeckProfile, rocket_compatibility_profile
from .official_search_game import OfficialNativeSearchGame, OfficialSearchGameState
from .phase6_native_game import _completions, _runtime_basics
from .trajectory_neural import (
    LoggedTrajectoryBootstrapProvider,
    TrajectoryBootstrapArtifact,
    restore_logged_trajectory_bootstrap,
)


PHASE6_NATIVE_BLUEPRINT_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*items: Any) -> int:
    digest = sha256(_canonical_json(items).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _finite(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


class _BootstrapNativePolicy:
    def __init__(self, provider: LoggedTrajectoryBootstrapProvider) -> None:
        self.provider = provider

    def strategy(self, view: NativeBlueprintView) -> ActionDistribution:
        prediction = self.provider.predict(
            public_features=view.public_belief.public_features,
            private_features=view.information_set.private_features,
            legal_actions=view.information_set.legal_actions,
        )
        return ActionDistribution.normalized(dict(prediction.action_probabilities))


class _RegretNativePolicy:
    def __init__(self, provider: NativeRegretBlueprintProvider) -> None:
        self.provider = provider

    def strategy(self, view: NativeBlueprintView) -> ActionDistribution:
        return self.provider.strategy(view.information_set, view.public_belief)


def _exploring_distribution(
    policy: Any,
    view: NativeBlueprintView,
    *,
    exploration: float,
) -> tuple[tuple[str, float], ...]:
    if not 0 <= exploration < 1:
        raise ValueError("native regret exploration must be in [0, 1)")
    base = policy.strategy(view)
    base.validate_support(view.information_set)
    probabilities = base.as_dict()
    uniform = 1.0 / len(view.information_set.legal_actions)
    return tuple(
        (
            action.action_id,
            (1.0 - exploration) * probabilities[action.action_id]
            + exploration * uniform,
        )
        for action in view.information_set.legal_actions
    )


def _sample_identifier(
    probabilities: Sequence[tuple[str, float]],
    rng: random.Random,
) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for identifier, probability in probabilities:
        cumulative += probability
        if threshold <= cumulative + 1e-15:
            return identifier
    return probabilities[-1][0]


def _sample_chance(
    game: OfficialNativeSearchGame,
    state: OfficialSearchGameState,
    rng: random.Random,
) -> OfficialSearchGameState:
    outcomes = game.chance_outcomes(state)
    identifier = _sample_identifier(
        tuple((outcome.outcome_id, outcome.probability) for outcome in outcomes),
        rng,
    )
    outcome = next(item for item in outcomes if item.outcome_id == identifier)
    return game.transition(state, None, chance_outcome=outcome)


def _raw_action(
    game: OfficialNativeSearchGame,
    state: OfficialSearchGameState,
    action: AbstractAction,
) -> AbstractAction:
    matches = [
        candidate
        for candidate in game.legal_actions(state)
        if candidate.action_id == action.action_id
        and candidate.official_option_indices == action.official_option_indices
    ]
    if len(matches) != 1:
        raise RuntimeError("enriched native action did not map to exactly one raw action")
    return matches[0]


def _policy_transition(
    game: OfficialNativeSearchGame,
    state: OfficialSearchGameState,
    *,
    bridge: NativeBlueprintFeatureBridge,
    policy: Any,
    exploration: float,
    rng: random.Random,
) -> OfficialSearchGameState:
    if state.actor is InformationSetActor.CHANCE:
        return _sample_chance(game, state, rng)
    view = bridge.view(game, state)
    distribution = _exploring_distribution(policy, view, exploration=exploration)
    identifier = _sample_identifier(distribution, rng)
    action = next(
        item for item in view.information_set.legal_actions if item.action_id == identifier
    )
    return game.transition(state, _raw_action(game, state, action))


def _rollout(
    game: OfficialNativeSearchGame,
    state: OfficialSearchGameState,
    *,
    utility_player: int,
    bridge: NativeBlueprintFeatureBridge,
    policy: Any,
    exploration: float,
    rng: random.Random,
    maximum_depth: int,
) -> tuple[float, int]:
    transitions = 0
    while not state.terminal and transitions < maximum_depth:
        state = _policy_transition(
            game,
            state,
            bridge=bridge,
            policy=policy,
            exploration=exploration,
            rng=rng,
        )
        transitions += 1
    if not state.terminal:
        raise RuntimeError("native regret terminal rollout exceeded depth cap")
    return game.terminal_utility(state, utility_player), transitions


@dataclass(frozen=True, slots=True)
class _RawBranchSample:
    round_index: int
    candidate_id: str
    player: int
    state_depth: int
    view: NativeBlueprintView
    behavior_strategy: tuple[tuple[str, float], ...]
    action_values: tuple[tuple[str, float], ...]
    rollouts_per_action: int
    native_transition_count: int


def _evaluate_information_set(
    game: OfficialNativeSearchGame,
    state: OfficialSearchGameState,
    *,
    round_index: int,
    candidate_id: str,
    bridge: NativeBlueprintFeatureBridge,
    policy: Any,
    behavior_exploration: float,
    continuation_exploration: float,
    rollouts_per_action: int,
    maximum_rollout_depth: int,
    seed: int,
) -> _RawBranchSample:
    view = bridge.view(game, state)
    behavior = _exploring_distribution(
        policy,
        view,
        exploration=behavior_exploration,
    )
    before = game.transition_count
    values = []
    for action in view.information_set.legal_actions:
        outcomes = []
        for rollout_index in range(rollouts_per_action):
            child = game.transition(state, _raw_action(game, state, action))
            utility, _ = _rollout(
                game,
                child,
                utility_player=int(state.player_index),
                bridge=bridge,
                policy=policy,
                exploration=continuation_exploration,
                rng=random.Random(
                    _seed(
                        seed,
                        "branch_rollout",
                        round_index,
                        candidate_id,
                        state.player_index,
                        view.information_set.info_set_id,
                        rollout_index,
                    )
                ),
                maximum_depth=maximum_rollout_depth,
            )
            outcomes.append(utility)
        values.append((action.action_id, sum(outcomes) / len(outcomes)))
    transitions = game.transition_count - before
    return _RawBranchSample(
        round_index=round_index,
        candidate_id=candidate_id,
        player=int(state.player_index),
        state_depth=state.depth,
        view=view,
        behavior_strategy=behavior,
        action_values=tuple(values),
        rollouts_per_action=rollouts_per_action,
        native_transition_count=transitions,
    )


def _collect_round(
    game: OfficialNativeSearchGame,
    *,
    round_index: int,
    candidates: Sequence[KaggleDeckProfile],
    bridge: NativeBlueprintFeatureBridge,
    policy: Any,
    minimum_branch_depth: int,
    maximum_branch_actions: int,
    behavior_exploration: float,
    continuation_exploration: float,
    rollouts_per_action: int,
    maximum_rollout_depth: int,
    seed: int,
) -> tuple[_RawBranchSample, ...]:
    chance_root = game.root_state()
    outcomes = game.chance_outcomes(chance_root)
    if len(outcomes) != len(candidates):
        raise RuntimeError("candidate/chance-root count mismatch")
    records = []
    for outcome, candidate in zip(outcomes, candidates, strict=True):
        state = game.transition(chance_root, None, chance_outcome=outcome)
        found: set[int] = set()
        spine_rng = random.Random(
            _seed(seed, "spine", round_index, candidate.archetype)
        )
        for _ in range(maximum_rollout_depth):
            if state.terminal or found == {0, 1}:
                break
            if state.actor is not InformationSetActor.CHANCE:
                player = int(state.player_index)
                actions = game.legal_actions(state)
                if (
                    player not in found
                    and state.depth >= minimum_branch_depth
                    and 2 <= len(actions) <= maximum_branch_actions
                ):
                    records.append(
                        _evaluate_information_set(
                            game,
                            state,
                            round_index=round_index,
                            candidate_id=candidate.archetype,
                            bridge=bridge,
                            policy=policy,
                            behavior_exploration=behavior_exploration,
                            continuation_exploration=continuation_exploration,
                            rollouts_per_action=rollouts_per_action,
                            maximum_rollout_depth=maximum_rollout_depth,
                            seed=seed,
                        )
                    )
                    found.add(player)
            state = _policy_transition(
                game,
                state,
                bridge=bridge,
                policy=policy,
                exploration=behavior_exploration,
                rng=spine_rng,
            )
        if found != {0, 1}:
            raise RuntimeError(
                f"native regret round did not find both players for {candidate.archetype}"
            )
    return tuple(records)


def _initial_split_map(records: Sequence[_RawBranchSample]) -> dict[str, str]:
    identifiers = sorted(
        {record.view.information_set.info_set_id for record in records},
        key=lambda value: _fingerprint({"native_regret_split": value}),
    )
    if len(identifiers) < 3:
        raise RuntimeError("native regret collection needs at least three information sets")
    train_count = min(len(identifiers) - 2, max(1, round(0.60 * len(identifiers))))
    remaining = len(identifiers) - train_count
    calibration_count = max(1, remaining // 2)
    if train_count + calibration_count >= len(identifiers):
        calibration_count = 1
    return {
        identifier: (
            "train"
            if index < train_count
            else "calibration"
            if index < train_count + calibration_count
            else "validation"
        )
        for index, identifier in enumerate(identifiers)
    }


@dataclass(slots=True)
class _RegretAccumulator:
    action_ids: tuple[str, ...]
    cumulative_regrets: list[float]
    strategy_sum: list[float]
    visits: int = 0


def _regret_matching(values: Sequence[float]) -> tuple[float, ...]:
    positive = tuple(max(0.0, float(value)) for value in values)
    total = sum(positive)
    if total <= 1e-12:
        return tuple(1.0 / len(positive) for _ in positive)
    return tuple(value / total for value in positive)


def _materialize_round(
    records: Sequence[_RawBranchSample],
    *,
    split_map: dict[str, str],
    accumulators: dict[str, _RegretAccumulator],
) -> tuple[NativeRegretTrainingSample, ...]:
    grouped: dict[str, list[_RawBranchSample]] = {}
    for record in records:
        grouped.setdefault(record.view.information_set.info_set_id, []).append(record)
    result = []
    for identifier, group in grouped.items():
        actions = group[0].view.information_set.legal_actions
        action_ids = tuple(action.action_id for action in actions)
        if any(
            tuple(action.action_id for action in item.view.information_set.legal_actions)
            != action_ids
            for item in group
        ):
            raise RuntimeError("one native information set has inconsistent action support")
        behavior = [0.0] * len(action_ids)
        values = [0.0] * len(action_ids)
        for item in group:
            behavior_by_id = dict(item.behavior_strategy)
            values_by_id = dict(item.action_values)
            for index, action_id in enumerate(action_ids):
                behavior[index] += behavior_by_id[action_id] / len(group)
                values[index] += values_by_id[action_id] / len(group)
        baseline = sum(probability * value for probability, value in zip(behavior, values, strict=True))
        instantaneous = [value - baseline for value in values]
        accumulator = accumulators.get(identifier)
        if accumulator is None:
            accumulator = _RegretAccumulator(
                action_ids=action_ids,
                cumulative_regrets=[0.0] * len(action_ids),
                strategy_sum=[0.0] * len(action_ids),
            )
            accumulators[identifier] = accumulator
        if accumulator.action_ids != action_ids:
            raise RuntimeError("native regret support changed for one information set")
        accumulator.cumulative_regrets = [
            previous + regret
            for previous, regret in zip(
                accumulator.cumulative_regrets,
                instantaneous,
                strict=True,
            )
        ]
        current_target = _regret_matching(accumulator.cumulative_regrets)
        accumulator.strategy_sum = [
            previous + probability
            for previous, probability in zip(
                accumulator.strategy_sum,
                current_target,
                strict=True,
            )
        ]
        accumulator.visits += 1
        average_target = tuple(
            value / accumulator.visits for value in accumulator.strategy_sum
        )
        split = split_map.setdefault(identifier, "train")
        for item in group:
            item_values = dict(item.action_values)
            item_behavior = dict(item.behavior_strategy)
            item_baseline = sum(
                item_behavior[action_id] * item_values[action_id]
                for action_id in action_ids
            )
            item_baseline = max(-1.0, min(1.0, item_baseline))
            private_fingerprint = (
                item.view.information_set.private_observation_fingerprint
            )
            if private_fingerprint is None:
                raise RuntimeError("native regret view lacks private fingerprint")
            result.append(
                NativeRegretTrainingSample(
                    round_index=item.round_index,
                    source_candidate_id_evaluation_only=item.candidate_id,
                    player=item.player,
                    depth=item.state_depth,
                    split=split,
                    information_set_id=identifier,
                    public_belief_fingerprint=item.view.public_belief.fingerprint,
                    private_observation_fingerprint=private_fingerprint,
                    public_features=item.view.public_belief.public_features,
                    private_features=item.view.information_set.private_features,
                    actions=actions,
                    behavior_strategy=tuple(
                        (action_id, item_behavior[action_id]) for action_id in action_ids
                    ),
                    action_values=tuple(
                        (action_id, item_values[action_id]) for action_id in action_ids
                    ),
                    instantaneous_regrets=tuple(
                        (action_id, regret)
                        for action_id, regret in zip(
                            action_ids,
                            instantaneous,
                            strict=True,
                        )
                    ),
                    cumulative_regrets=tuple(
                        (action_id, regret)
                        for action_id, regret in zip(
                            action_ids,
                            accumulator.cumulative_regrets,
                            strict=True,
                        )
                    ),
                    target_strategy=tuple(
                        (action_id, probability)
                        for action_id, probability in zip(
                            action_ids,
                            average_target,
                            strict=True,
                        )
                    ),
                    baseline_value=item_baseline,
                    rollouts_per_action=item.rollouts_per_action,
                    native_transition_count=item.native_transition_count,
                )
            )
    return tuple(result)


def _dataset(
    *,
    game: OfficialNativeSearchGame,
    deck_fingerprint: str,
    bootstrap_fingerprint: str,
    seed: int,
    training_rounds: int,
    samples: Sequence[NativeRegretTrainingSample],
) -> NativeRegretDataset:
    return NativeRegretDataset(
        game_fingerprint=game.fingerprint,
        runtime_fingerprint=game.ruleset_fingerprint,
        deck_fingerprint=deck_fingerprint,
        source_bootstrap_fingerprint=bootstrap_fingerprint,
        feature_schema_fingerprint=NATIVE_BLUEPRINT_FEATURE_SCHEMA_FINGERPRINT,
        sampling_seed=seed,
        training_rounds=training_rounds,
        samples=tuple(samples),
    )


def _roundtrip_delta(
    provider: NativeRegretBlueprintProvider,
    restored: NativeRegretBlueprintProvider,
    dataset: NativeRegretDataset,
) -> float:
    maximum = 0.0
    for sample in dataset.samples:
        information_set, belief = native_regret_sample_view(sample)
        left = provider.predict(information_set, belief)
        right = restored.predict(information_set, belief)
        left_strategy = left.strategy.as_dict()
        right_strategy = right.strategy.as_dict()
        maximum = max(
            maximum,
            *(abs(left_strategy[key] - right_strategy[key]) for key in left_strategy),
            *(
                abs(value - dict(right.advantages)[key])
                for key, value in left.advantages
            ),
            abs(left.state_value - right.state_value),
            *(
                abs(value - dict(right.opponent_range)[key])
                for key, value in left.opponent_range
            ),
        )
    return maximum


def _raw_private_key_count(value: Any) -> int:
    forbidden = {
        "cardId",
        "card_id",
        "card_ids",
        "own_hand",
        "own_hand_card_ids",
        "opponent_hand",
        "opponent_hand_card_ids",
        "your_deck",
        "opponent_deck",
        "your_prize",
        "opponent_prize",
        "prize_ids",
        "hand_ids",
    }
    if isinstance(value, Mapping):
        return sum(
            int(str(key) in forbidden) + _raw_private_key_count(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_raw_private_key_count(item) for item in value)
    return 0


@dataclass(frozen=True, slots=True)
class Phase6NativeBlueprintEvidence:
    benchmark_id: str
    status: str
    runtime_fingerprint: str
    game_fingerprint: str
    deck_fingerprint: str
    source_bootstrap_fingerprint: str
    dataset_fingerprint: str
    model_fingerprint: str
    feature_schema_fingerprint: str
    training_rounds: int
    sample_count: int
    information_set_count: int
    candidate_count: int
    player_0_samples: int
    player_1_samples: int
    split_counts: tuple[tuple[str, int], ...]
    native_branch_transition_count: int
    rollouts_per_action: int
    root_policy_view_invariant_across_completions: bool
    source_official_engine_executed: bool
    model_official_engine_executed: bool
    selected_hidden_truth_in_policy_features: bool
    raw_private_card_ids_persisted: int
    initial_training_loss: float
    final_training_loss: float
    gradient_parameter_count: int
    initialized_parameter_count: int
    model_roundtrip_maximum_prediction_delta: float
    split_metrics: tuple[NativeRegretSplitMetrics, ...]
    strategy_changed_from_behavior_samples: int
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_fingerprint",
            "game_fingerprint",
            "deck_fingerprint",
            "source_bootstrap_fingerprint",
            "dataset_fingerprint",
            "model_fingerprint",
            "feature_schema_fingerprint",
        ):
            value = getattr(self, field_name)
            if len(value) != 64:
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        if self.status not in {
            "phase6k_native_regret_blueprint_candidate_validated_partial",
            "failed",
        }:
            raise ValueError("unknown Phase 6K status")
        if min(
            self.training_rounds,
            self.sample_count,
            self.information_set_count,
            self.candidate_count,
            self.player_0_samples,
            self.player_1_samples,
            self.native_branch_transition_count,
            self.rollouts_per_action,
            self.gradient_parameter_count,
            self.initialized_parameter_count,
        ) <= 0:
            raise ValueError("Phase 6K evidence counts must be positive")
        if tuple(name for name, _ in self.split_counts) != (
            "train",
            "calibration",
            "validation",
        ):
            raise ValueError("Phase 6K split counts must be ordered")
        if sum(value for _, value in self.split_counts) != self.sample_count:
            raise ValueError("Phase 6K split counts do not cover samples")
        if not self.source_official_engine_executed or self.model_official_engine_executed:
            raise ValueError("Phase 6K provenance boundary is invalid")
        if self.selected_hidden_truth_in_policy_features:
            raise ValueError("Phase 6K policy features cannot contain selected truth")
        if self.raw_private_card_ids_persisted != 0:
            raise ValueError("Phase 6K cannot persist raw private card IDs")
        for field_name in (
            "initial_training_loss",
            "final_training_loss",
            "model_roundtrip_maximum_prediction_delta",
        ):
            if _finite(getattr(self, field_name), field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if tuple(metric.split for metric in self.split_metrics) != (
            "train",
            "calibration",
            "validation",
        ):
            raise ValueError("Phase 6K split metrics must be ordered")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_BLUEPRINT_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "runtime_fingerprint": self.runtime_fingerprint,
            "game_fingerprint": self.game_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "source_bootstrap_fingerprint": self.source_bootstrap_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "training": {
                "rounds": self.training_rounds,
                "samples": self.sample_count,
                "information_sets": self.information_set_count,
                "candidate_count": self.candidate_count,
                "player_0_samples": self.player_0_samples,
                "player_1_samples": self.player_1_samples,
                "split_counts": dict(self.split_counts),
                "native_branch_transitions": self.native_branch_transition_count,
                "rollouts_per_action": self.rollouts_per_action,
                "initial_loss": self.initial_training_loss,
                "final_loss": self.final_training_loss,
                "gradient_parameter_count": self.gradient_parameter_count,
                "initialized_parameter_count": self.initialized_parameter_count,
                "strategy_changed_from_behavior_samples": (
                    self.strategy_changed_from_behavior_samples
                ),
            },
            "checks": {
                "root_policy_view_invariant_across_completions": (
                    self.root_policy_view_invariant_across_completions
                ),
                "selected_hidden_truth_in_policy_features": (
                    self.selected_hidden_truth_in_policy_features
                ),
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
                "model_roundtrip_maximum_prediction_delta": (
                    self.model_roundtrip_maximum_prediction_delta
                ),
            },
            "provenance": {
                "source_official_engine_executed": (
                    self.source_official_engine_executed
                ),
                "model_official_engine_executed": self.model_official_engine_executed,
            },
            "split_metrics": [asdict(item) for item in self.split_metrics],
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        metrics = "\n".join(
            (
                f"| {item.split} | {item.sample_count} | {item.information_set_count} | "
                f"{item.policy_cross_entropy:.6f} | {item.advantage_rmse:.6f} | "
                f"{item.value_rmse:.6f} | {item.mean_candidate_deviation_incentive:.6f} | "
                f"{item.mean_behavior_deviation_incentive:.6f} | "
                f"{item.exact_legal_support_rate:.3f} |"
            )
            for item in self.split_metrics
        )
        claims = "\n".join(f"| {name} | {value} |" for name, value in self.claims)
        return f"""# Phase 6K native regret blueprint candidate

Benchmark: `{self.benchmark_id}`  
Evidence fingerprint: `{self.fingerprint}`  
Dataset fingerprint: `{self.dataset_fingerprint}`  
Model fingerprint: `{self.model_fingerprint}`  
Status: **{self.status}**

The action values below come from terminal descendant paths genuinely executed
by the licensed native search engine. The hidden-completion prior, derived
dataset, neural model, and strategy are abstract research objects. This is an
equilibrium-oriented sampled-regret training milestone, not a Nash-convergence,
safety, official-agent, or strength claim.

- {self.training_rounds} regret collection/training rounds;
- {self.sample_count} samples / {self.information_set_count} information sets;
- {self.player_0_samples} player-0 and {self.player_1_samples} player-1 samples;
- {self.native_branch_transition_count:,} genuine native branch transitions;
- train loss {self.initial_training_loss:.6f} -> {self.final_training_loss:.6f};
- exact restore max drift {self.model_roundtrip_maximum_prediction_delta:.3e}; and
- {self.raw_private_card_ids_persisted} raw private card IDs persisted.

| Split | Samples | Info sets | Policy CE | Advantage RMSE | Value RMSE | Candidate deviation | Behavior deviation | Legal support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{metrics}

Sampled deviation incentives are diagnostics on the collected branch values,
not exploitability or NashConv for the full game.

## Claim boundary

| Claim | Status |
|---|---|
{claims}
"""


@dataclass(slots=True)
class Phase6NativeBlueprintArtifacts:
    evidence: Phase6NativeBlueprintEvidence
    dataset: NativeRegretDataset
    training: NativeRegretTrainingRun


def _load_bootstrap(project_root: Path) -> TrajectoryBootstrapArtifact:
    path = (
        project_root
        / "artifacts"
        / "phase6_trajectory_neural"
        / "logged_trajectory_bootstrap.json"
    )
    return TrajectoryBootstrapArtifact.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def run_phase6_native_regret_blueprint(
    *,
    project_root: str | Path | None = None,
    seed: int = DEFAULT_NATIVE_REGRET_SEED,
    training_rounds: int = 2,
    epochs_per_round: int = 40,
    learning_rate: float = 0.003,
    minimum_branch_depth: int = 10,
    maximum_branch_actions: int = 6,
    behavior_exploration: float = 0.10,
    continuation_exploration: float = 0.05,
    rollouts_per_action: int = 1,
    maximum_rollout_depth: int = 2_000,
) -> Phase6NativeBlueprintArtifacts:
    if training_rounds <= 0 or epochs_per_round <= 0:
        raise ValueError("Phase 6K rounds/epochs must be positive")
    if maximum_branch_actions < 2 or rollouts_per_action <= 0:
        raise ValueError("Phase 6K branch/rollout counts are invalid")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    path_text = str(runtime)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    from cg.game import battle_finish, battle_start  # type: ignore[import-not-found]

    bootstrap_artifact = _load_bootstrap(root)
    bootstrap_provider = restore_logged_trajectory_bootstrap(bootstrap_artifact)
    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    candidates = OFFICIAL_PANEL_DECKS + (target,)
    observation, start = battle_start(
        list(target.card_ids),
        list(OFFICIAL_PANEL_DECKS[0].card_ids),
    )
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6K root decks: "
            f"player={start.errorPlayer}, error={start.errorType}"
        )
    try:
        completions = _completions(
            observation,
            target=target,
            candidates=candidates,
            basic_card_ids=_runtime_basics(runtime),
        )
        bridge = NativeBlueprintFeatureBridge()
        with OfficialNativeSearchGame(
            observation,
            completions,
            runtime_directory=runtime,
        ) as game:
            roots = [
                game.transition(game.root_state(), None, chance_outcome=outcome)
                for outcome in game.chance_outcomes(game.root_state())
            ]
            root_views = [bridge.view(game, state) for state in roots]
            root_invariant = (
                len({view.public_belief.fingerprint for view in root_views}) == 1
                and len({view.information_set.info_set_id for view in root_views}) == 1
            )
            if tuple(bootstrap_artifact.public_normalizer.names) != NATIVE_PUBLIC_FEATURE_NAMES:
                raise RuntimeError("Phase 6K public feature/bootstrap schema mismatch")
            if tuple(bootstrap_artifact.private_normalizer.names) != NATIVE_PRIVATE_FEATURE_NAMES:
                raise RuntimeError("Phase 6K private feature/bootstrap schema mismatch")
            if tuple(bootstrap_artifact.action_normalizer.names) != NATIVE_ACTION_FEATURE_NAMES:
                raise RuntimeError("Phase 6K action feature/bootstrap schema mismatch")

            split_map: dict[str, str] = {}
            accumulators: dict[str, _RegretAccumulator] = {}
            samples: list[NativeRegretTrainingSample] = []
            policy: Any = _BootstrapNativePolicy(bootstrap_provider)
            training: NativeRegretTrainingRun | None = None
            for round_index in range(1, training_rounds + 1):
                raw = _collect_round(
                    game,
                    round_index=round_index,
                    candidates=candidates,
                    bridge=bridge,
                    policy=policy,
                    minimum_branch_depth=minimum_branch_depth,
                    maximum_branch_actions=maximum_branch_actions,
                    behavior_exploration=behavior_exploration,
                    continuation_exploration=continuation_exploration,
                    rollouts_per_action=rollouts_per_action,
                    maximum_rollout_depth=maximum_rollout_depth,
                    seed=seed,
                )
                if round_index == 1:
                    split_map = _initial_split_map(raw)
                samples.extend(
                    _materialize_round(
                        raw,
                        split_map=split_map,
                        accumulators=accumulators,
                    )
                )
                current_dataset = _dataset(
                    game=game,
                    deck_fingerprint=target.fingerprint,
                    bootstrap_fingerprint=bootstrap_artifact.fingerprint,
                    seed=seed,
                    training_rounds=round_index,
                    samples=samples,
                )
                training = train_native_regret_blueprint(
                    current_dataset,
                    bootstrap_artifact,
                    seed=seed,
                    epochs=epochs_per_round,
                    learning_rate=learning_rate,
                )
                policy = _RegretNativePolicy(training.provider)
            if training is None:
                raise AssertionError("Phase 6K training loop did not run")
            dataset = _dataset(
                game=game,
                deck_fingerprint=target.fingerprint,
                bootstrap_fingerprint=bootstrap_artifact.fingerprint,
                seed=seed,
                training_rounds=training_rounds,
                samples=samples,
            )
            if training.artifact.dataset_fingerprint != dataset.fingerprint:
                raise RuntimeError("final Phase 6K model/dataset mismatch")
            restored_artifact = NativeRegretBlueprintArtifact.from_dict(
                json.loads(training.artifact.to_json())
            )
            restored = restore_native_regret_blueprint(restored_artifact)
            roundtrip_delta = _roundtrip_delta(
                training.provider,
                restored,
                dataset,
            )
            restored_dataset = NativeRegretDataset.from_dict(
                json.loads(dataset.to_json())
            )
            if restored_dataset.fingerprint != dataset.fingerprint:
                raise RuntimeError("Phase 6K dataset roundtrip mismatch")
            split_counts = tuple(
                (split, sum(sample.split == split for sample in dataset.samples))
                for split in ("train", "calibration", "validation")
            )
            player_counts = tuple(
                sum(sample.player == player for sample in dataset.samples)
                for player in (0, 1)
            )
            candidates_seen = {
                sample.source_candidate_id_evaluation_only for sample in dataset.samples
            }
            branch_transitions = sum(
                sample.native_transition_count for sample in dataset.samples
            )
            raw_private_key_count = _raw_private_key_count(dataset.to_dict())
            required_gradient_prefixes = (
                "public_encoder.",
                "private_encoder.",
                "state_encoder.",
                "action_encoder.",
                "average_policy_",
                "advantage_",
                "value_head.",
            )
            gradients_cover = all(
                any(name.startswith(prefix) for name in training.report.gradient_parameter_names)
                for prefix in required_gradient_prefixes
            )
            gates = (
                root_invariant,
                len(candidates_seen) == len(candidates) == 6,
                player_counts[0] > 0,
                player_counts[1] > 0,
                all(count > 0 for _, count in split_counts),
                branch_transitions > 0,
                training.report.final_training_loss
                < training.report.initial_training_loss,
                gradients_cover,
                roundtrip_delta <= 1e-12,
                all(
                    metric.exact_legal_support_rate == 1.0
                    for metric in training.report.split_metrics
                ),
                training.artifact.equilibrium_objective_trained,
                not training.artifact.equilibrium_convergence_validated,
                not training.artifact.deployable_blueprint,
                not training.artifact.model_official_engine_executed,
                raw_private_key_count == 0,
            )
            status = (
                "phase6k_native_regret_blueprint_candidate_validated_partial"
                if all(gates)
                else "failed"
            )
            claims = (
                ("native_branch_descendants", "genuine_official_search_execution"),
                ("hidden_completion_prior", "supplied_uncalibrated_abstraction"),
                ("feature_initialization", "phase6i_checkpoint_copied"),
                ("training_objective", "terminal_branch_sampled_regret_matching"),
                ("deep_cfr", "not_claimed"),
                ("equilibrium_convergence", "not_validated"),
                ("full_game_exploitability", "not_computable_from_this_sample"),
                ("counterfactual_value_bounds", "not_calibrated"),
                ("safe_resolver", "not_compatible_or_deployed"),
                ("opponent_exploitation", "disabled"),
                ("official_agent_strategy_control", "disabled"),
                ("exact_deck", "false_watchtower_compatibility_proxy"),
                ("strength_claim", "none"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            evidence = Phase6NativeBlueprintEvidence(
                benchmark_id="phase6k-native-regret-blueprint-candidate-v1",
                status=status,
                runtime_fingerprint=game.ruleset_fingerprint,
                game_fingerprint=game.fingerprint,
                deck_fingerprint=target.fingerprint,
                source_bootstrap_fingerprint=bootstrap_artifact.fingerprint,
                dataset_fingerprint=dataset.fingerprint,
                model_fingerprint=training.artifact.fingerprint,
                feature_schema_fingerprint=bridge.fingerprint,
                training_rounds=training_rounds,
                sample_count=len(dataset.samples),
                information_set_count=len(
                    {sample.information_set_id for sample in dataset.samples}
                ),
                candidate_count=len(candidates_seen),
                player_0_samples=player_counts[0],
                player_1_samples=player_counts[1],
                split_counts=split_counts,
                native_branch_transition_count=branch_transitions,
                rollouts_per_action=rollouts_per_action,
                root_policy_view_invariant_across_completions=root_invariant,
                source_official_engine_executed=True,
                model_official_engine_executed=False,
                selected_hidden_truth_in_policy_features=False,
                raw_private_card_ids_persisted=raw_private_key_count,
                initial_training_loss=training.report.initial_training_loss,
                final_training_loss=training.report.final_training_loss,
                gradient_parameter_count=len(
                    training.report.gradient_parameter_names
                ),
                initialized_parameter_count=len(
                    training.report.initialized_parameter_names
                ),
                model_roundtrip_maximum_prediction_delta=roundtrip_delta,
                split_metrics=training.report.split_metrics,
                strategy_changed_from_behavior_samples=(
                    training.report.strategy_changed_from_behavior_samples
                ),
                claims=claims,
            )
            if evidence.status == "failed":
                raise RuntimeError("Phase 6K native regret blueprint failed a gate")
            return Phase6NativeBlueprintArtifacts(
                evidence=evidence,
                dataset=dataset,
                training=training,
            )
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_regret_blueprint().evidence.to_json())


if __name__ == "__main__":
    main()
