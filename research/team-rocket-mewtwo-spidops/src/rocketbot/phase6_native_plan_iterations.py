"""Phase 6P: iterative neural outcome sampling on the native turn-plan game.

Phase 6O proved that nontrivial both-player samples can be collected. This
runner closes one further algorithmic wiring gap: after every sampling round it
fits fresh regret/average-strategy networks over the cumulative reservoirs, and
the next round samples from those player-specific regret-matching strategies.

The turn-plan game, tactical executor, completion prior, estimator targets, and
models remain abstractions. Iteration does not imply convergence, calibrated
counterfactual values, safe resolving, official strategy control, or strength.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .game_theory import InformationSetActor
from .native_blueprint_features import NativeBlueprintFeatureBridge
from .native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanBlueprintProvider,
    NativePlanOutcomeDataset,
    NativePlanTrainingReport,
    native_plan_sample_view,
    restore_native_plan_blueprint,
    train_native_plan_blueprint,
)
from .native_plan_game import OfficialNativePlanGame
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
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _completions, _runtime_basics


PHASE6_NATIVE_PLAN_ITERATIONS_SCHEMA_VERSION = 1
PHASE6_NATIVE_PLAN_ITERATIONS_SEED = 6_023


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big"
    )


def _mean_policy_drift(
    before: NativePlanBlueprintProvider,
    after: NativePlanBlueprintProvider,
    samples: tuple[OutcomeSamplingRegretSample, ...],
    *,
    player: int,
) -> float:
    values = []
    for sample in samples:
        information_set, belief = native_plan_sample_view(sample)
        first = before.current_strategy(information_set, belief, player=player)
        second = after.current_strategy(information_set, belief, player=player)
        values.append(
            0.5
            * sum(
                abs(first.probability(action_id) - second.probability(action_id))
                for action_id in first.as_dict()
            )
        )
    return sum(values) / len(values) if values else 0.0


def _mean_average_drift(
    before: NativePlanBlueprintProvider,
    after: NativePlanBlueprintProvider,
    samples: tuple[OutcomeSamplingStrategySample, ...],
) -> float:
    values = []
    for sample in samples:
        information_set, belief = native_plan_sample_view(sample)
        first = before.strategy(information_set, belief)
        second = after.strategy(information_set, belief)
        values.append(
            0.5
            * sum(
                abs(first.probability(action_id) - second.probability(action_id))
                for action_id in first.as_dict()
            )
        )
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class NativePlanIterationEvidence:
    iteration: int
    completed_trajectories: int
    player_0_samples_added: int
    player_1_samples_added: int
    strategy_samples_added: int
    cumulative_player_0_samples: int
    cumulative_player_1_samples: int
    cumulative_strategy_samples: int
    nonzero_regret_actions_added: int
    initial_training_loss: float
    final_training_loss: float
    player_0_mean_strategy_drift: float
    player_1_mean_strategy_drift: float
    average_plan_mean_strategy_drift: float
    dataset_fingerprint: str
    model_fingerprint: str

    def __post_init__(self) -> None:
        if min(
            self.iteration,
            self.completed_trajectories,
            self.player_0_samples_added,
            self.player_1_samples_added,
            self.strategy_samples_added,
            self.cumulative_player_0_samples,
            self.cumulative_player_1_samples,
            self.cumulative_strategy_samples,
            self.nonzero_regret_actions_added,
        ) <= 0:
            raise ValueError("Phase 6P iteration accounting must be positive")
        for name in (
            "initial_training_loss",
            "final_training_loss",
            "player_0_mean_strategy_drift",
            "player_1_mean_strategy_drift",
            "average_plan_mean_strategy_drift",
        ):
            value = float(getattr(self, name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("dataset_fingerprint", "model_fingerprint"):
            value = getattr(self, name)
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NativePlanIterationEvidence":
        return cls(
            iteration=int(raw["iteration"]),
            completed_trajectories=int(raw["completed_trajectories"]),
            player_0_samples_added=int(raw["player_0_samples_added"]),
            player_1_samples_added=int(raw["player_1_samples_added"]),
            strategy_samples_added=int(raw["strategy_samples_added"]),
            cumulative_player_0_samples=int(raw["cumulative_player_0_samples"]),
            cumulative_player_1_samples=int(raw["cumulative_player_1_samples"]),
            cumulative_strategy_samples=int(raw["cumulative_strategy_samples"]),
            nonzero_regret_actions_added=int(raw["nonzero_regret_actions_added"]),
            initial_training_loss=float(raw["initial_training_loss"]),
            final_training_loss=float(raw["final_training_loss"]),
            player_0_mean_strategy_drift=float(raw["player_0_mean_strategy_drift"]),
            player_1_mean_strategy_drift=float(raw["player_1_mean_strategy_drift"]),
            average_plan_mean_strategy_drift=float(
                raw["average_plan_mean_strategy_drift"]
            ),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            model_fingerprint=str(raw["model_fingerprint"]),
        )


@dataclass(frozen=True, slots=True)
class Phase6NativePlanIterationsEvidence:
    benchmark_id: str
    status: str
    initial_plan_model_fingerprint: str
    final_dataset_fingerprint: str
    final_model_fingerprint: str
    native_game_fingerprint: str
    runtime_fingerprint: str
    rounds: tuple[NativePlanIterationEvidence, ...]
    native_transitions: int
    macro_transitions: int
    tactical_transitions: int
    raw_regret_median_absolute: float
    raw_regret_p95_absolute: float
    raw_regret_maximum_absolute: float
    maximum_restore_drift: float
    raw_private_card_ids_persisted: int
    source_official_engine_executed: bool
    iterative_strategy_updates_executed: bool
    equilibrium_convergence_validated: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status or not self.rounds:
            raise ValueError("Phase 6P evidence identity is invalid")
        if tuple(item.iteration for item in self.rounds) != tuple(
            range(1, len(self.rounds) + 1)
        ):
            raise ValueError("Phase 6P rounds must be complete and ordered")
        for name in (
            "initial_plan_model_fingerprint",
            "final_dataset_fingerprint",
            "final_model_fingerprint",
            "native_game_fingerprint",
            "runtime_fingerprint",
        ):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if min(
            self.native_transitions,
            self.macro_transitions,
            self.tactical_transitions,
        ) <= 0:
            raise ValueError("Phase 6P transition accounting must be positive")
        for name in (
            "raw_regret_median_absolute",
            "raw_regret_p95_absolute",
            "raw_regret_maximum_absolute",
            "maximum_restore_drift",
        ):
            value = float(getattr(self, name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            self.raw_private_card_ids_persisted != 0
            or not self.source_official_engine_executed
            or not self.iterative_strategy_updates_executed
            or self.equilibrium_convergence_validated
        ):
            raise ValueError("Phase 6P provenance/capability boundary is invalid")
        if dict(self.claims).get("phase6_complete") != "false":
            raise ValueError("Phase 6P cannot mark Phase 6 complete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_PLAN_ITERATIONS_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "initial_plan_model_fingerprint": self.initial_plan_model_fingerprint,
            "final_dataset_fingerprint": self.final_dataset_fingerprint,
            "final_model_fingerprint": self.final_model_fingerprint,
            "native_game_fingerprint": self.native_game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "rounds": [asdict(item) for item in self.rounds],
            "transitions": {
                "native": self.native_transitions,
                "macro": self.macro_transitions,
                "tactical": self.tactical_transitions,
            },
            "variance": {
                "raw_regret_median_absolute": self.raw_regret_median_absolute,
                "raw_regret_p95_absolute": self.raw_regret_p95_absolute,
                "raw_regret_maximum_absolute": self.raw_regret_maximum_absolute,
                "neural_target": "train_split_scale_asinh",
            },
            "maximum_restore_drift": self.maximum_restore_drift,
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any]
    ) -> "Phase6NativePlanIterationsEvidence":
        if int(raw.get("schema_version", -1)) != PHASE6_NATIVE_PLAN_ITERATIONS_SCHEMA_VERSION:
            raise ValueError("unsupported Phase 6P evidence schema")
        transitions = raw["transitions"]
        variance = raw["variance"]
        provenance = raw["provenance"]
        claims = tuple(
            (str(name), str(value)) for name, value in raw["claims"].items()
        )
        result = cls(
            benchmark_id=str(raw["benchmark_id"]),
            status=str(raw["status"]),
            initial_plan_model_fingerprint=str(raw["initial_plan_model_fingerprint"]),
            final_dataset_fingerprint=str(raw["final_dataset_fingerprint"]),
            final_model_fingerprint=str(raw["final_model_fingerprint"]),
            native_game_fingerprint=str(raw["native_game_fingerprint"]),
            runtime_fingerprint=str(raw["runtime_fingerprint"]),
            rounds=tuple(
                NativePlanIterationEvidence.from_dict(dict(item))
                for item in raw["rounds"]
            ),
            native_transitions=int(transitions["native"]),
            macro_transitions=int(transitions["macro"]),
            tactical_transitions=int(transitions["tactical"]),
            raw_regret_median_absolute=float(
                variance["raw_regret_median_absolute"]
            ),
            raw_regret_p95_absolute=float(variance["raw_regret_p95_absolute"]),
            raw_regret_maximum_absolute=float(
                variance["raw_regret_maximum_absolute"]
            ),
            maximum_restore_drift=float(raw["maximum_restore_drift"]),
            raw_private_card_ids_persisted=int(
                provenance["raw_private_card_ids_persisted"]
            ),
            source_official_engine_executed=bool(
                provenance["source_official_engine_executed"]
            ),
            iterative_strategy_updates_executed=(
                dict(claims).get("iterative_collection")
                == "previous_regret_models_drive_next_round"
            ),
            equilibrium_convergence_validated=(
                dict(claims).get("equilibrium_convergence") == "validated"
            ),
            claims=claims,
        )
        expected = raw.get("fingerprint")
        if expected is not None and str(expected) != result.fingerprint:
            raise ValueError("Phase 6P evidence fingerprint mismatch")
        return result

    def to_markdown(self) -> str:
        rows = "\n".join(
            "| {0.iteration} | {0.completed_trajectories} | {0.player_0_samples_added} | "
            "{0.player_1_samples_added} | {0.strategy_samples_added} | "
            "{0.initial_training_loss:.6f} → {0.final_training_loss:.6f} | "
            "{0.player_0_mean_strategy_drift:.4f} / "
            "{0.player_1_mean_strategy_drift:.4f} / "
            "{0.average_plan_mean_strategy_drift:.4f} |".format(item)
            for item in self.rounds
        )
        claims = "\n".join(f"| {name} | {value} |" for name, value in self.claims)
        return f"""# Phase 6P iterative native turn-plan candidate

Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

Each round samples from the previous round's player-specific regret-matching
models, then refits player-0 regret, player-1 regret, and average-plan models on
the cumulative memories. Genuine native descendant execution is kept separate
from the abstract plan game, supplied hidden prior, targets, and networks.

| Round | Trajectories | P0 added | P1 added | Strategy added | Loss | Mean TV drift P0 / P1 / average |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

Native / macro / tactical transitions: {self.native_transitions} /
{self.macro_transitions} / {self.tactical_transitions}.  
Raw private-card fields persisted: {self.raw_private_card_ids_persisted}.

Raw absolute regret median / p95 / maximum:
{self.raw_regret_median_absolute:.6g} / {self.raw_regret_p95_absolute:.6g} /
{self.raw_regret_maximum_absolute:.6g}. This tail blocks value calibration and
safe-resolver admission. Checkpoint restore drift is
{self.maximum_restore_drift:.3e}.

## Claim boundary

| Claim | Status |
|---|---|
{claims}
"""


@dataclass(frozen=True, slots=True)
class Phase6NativePlanIterationsArtifacts:
    dataset: NativePlanOutcomeDataset
    model: NativePlanBlueprintArtifact
    report: NativePlanTrainingReport
    evidence: Phase6NativePlanIterationsEvidence


def run_phase6_native_plan_iterations(
    *,
    project_root: str | Path | None = None,
    source_tactical_blueprint_path: str | Path | None = None,
    initial_plan_blueprint_path: str | Path | None = None,
    rounds: int = 3,
    exploration: float = 0.25,
    reservoir_capacity: int = 12_288,
    maximum_macro_transitions: int = 256,
    maximum_tactical_transitions: int = 512,
    epochs_per_round: int = 20,
    learning_rate: float = 2e-3,
    seed: int = PHASE6_NATIVE_PLAN_ITERATIONS_SEED,
) -> Phase6NativePlanIterationsArtifacts:
    if min(
        rounds,
        reservoir_capacity,
        maximum_macro_transitions,
        maximum_tactical_transitions,
        epochs_per_round,
    ) <= 0 or not 0 < exploration <= 1 or learning_rate <= 0:
        raise ValueError("Phase 6P limits are invalid")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    tactical_path = Path(
        source_tactical_blueprint_path
        or root / "artifacts" / "phase6_native_blueprint" / "native_regret_blueprint.json"
    ).resolve()
    initial_path = Path(
        initial_plan_blueprint_path
        or root / "artifacts" / "phase6_native_plan_blueprint" / "native_plan_blueprint.json"
    ).resolve()
    tactical_artifact = NativeRegretBlueprintArtifact.from_dict(
        json.loads(tactical_path.read_text(encoding="utf-8"))
    )
    tactical_provider = restore_native_regret_blueprint(tactical_artifact)
    initial_artifact = NativePlanBlueprintArtifact.from_dict(
        json.loads(initial_path.read_text(encoding="utf-8"))
    )
    current_provider = restore_native_plan_blueprint(initial_artifact)
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
        raise ValueError("Phase 6P tactical blueprint/deck fingerprint mismatch")
    if initial_artifact.deck_fingerprint != target.fingerprint:
        raise ValueError("Phase 6P initial plan blueprint/deck fingerprint mismatch")
    if (
        initial_artifact.source_tactical_blueprint_fingerprint
        != tactical_artifact.fingerprint
    ):
        raise ValueError("Phase 6P initial plan/tactical provenance mismatch")
    candidates: tuple[KaggleDeckProfile, ...] = OFFICIAL_PANEL_DECKS + (target,)
    observation, start = battle_start(
        list(target.card_ids), list(OFFICIAL_PANEL_DECKS[0].card_ids)
    )
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6P decks: "
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
            macro_games = []
            macro_fingerprints = []
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
                macro_fingerprints.append(macro.fingerprint)
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
                    maximum_transitions=maximum_macro_transitions,
                    maximum_depth=maximum_macro_transitions,
                )
            )
            round_evidence = []
            final_dataset: NativePlanOutcomeDataset | None = None
            final_report: NativePlanTrainingReport | None = None
            for iteration in range(1, rounds + 1):
                before_counts = (regrets[0].seen, regrets[1].seen, strategies.seen)
                added_p0: list[OutcomeSamplingRegretSample] = []
                added_p1: list[OutcomeSamplingRegretSample] = []
                added_strategy: list[OutcomeSamplingStrategySample] = []

                def learned_strategy(information_set, belief):
                    player = (
                        0
                        if information_set.actor is InformationSetActor.SELF
                        else 1
                    )
                    return current_provider.current_strategy(
                        information_set, belief, player=player
                    )

                for candidate_index, macro_game in enumerate(macro_games):
                    for traverser in (0, 1):
                        result = traversal.run(
                            macro_game,
                            learned_strategy,
                            traverser=traverser,
                            iteration=iteration,
                            rng=random.Random(
                                _seed(seed, iteration, candidate_index, traverser)
                            ),
                            baseline=macro_game.plan_value_baseline,
                        )
                        for sample in result.regret_samples:
                            regrets[traverser].add(sample)
                            (added_p0 if traverser == 0 else added_p1).append(sample)
                        for sample in result.strategy_samples:
                            strategies.add(sample)
                            added_strategy.append(sample)
                final_dataset = NativePlanOutcomeDataset(
                    native_game_fingerprint=native_game.fingerprint,
                    runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                    deck_fingerprint=target.fingerprint,
                    source_tactical_blueprint_fingerprint=tactical_artifact.fingerprint,
                    macro_game_fingerprints=tuple(macro_fingerprints),
                    candidate_ids=tuple(candidate.archetype for candidate in candidates),
                    iterations=iteration,
                    exploration=exploration,
                    player_0_regrets=regrets[0],
                    player_1_regrets=regrets[1],
                    average_strategy=strategies,
                    native_transition_count=native_game.transition_count,
                    macro_transition_count=sum(
                        item.macro_transition_count for item in macro_games
                    ),
                    tactical_transition_count=sum(
                        item.tactical_transition_count for item in macro_games
                    ),
                )
                training = train_native_plan_blueprint(
                    final_dataset,
                    tactical_provider,
                    epochs=epochs_per_round,
                    learning_rate=learning_rate,
                    seed=seed + iteration,
                )
                next_provider = training.provider
                report = training.report
                nonzero = sum(
                    abs(value) > 1e-12
                    for sample in tuple(added_p0) + tuple(added_p1)
                    for _, value in sample.sampled_regrets
                )
                item = NativePlanIterationEvidence(
                    iteration=iteration,
                    completed_trajectories=2 * len(candidates),
                    player_0_samples_added=regrets[0].seen - before_counts[0],
                    player_1_samples_added=regrets[1].seen - before_counts[1],
                    strategy_samples_added=strategies.seen - before_counts[2],
                    cumulative_player_0_samples=regrets[0].seen,
                    cumulative_player_1_samples=regrets[1].seen,
                    cumulative_strategy_samples=strategies.seen,
                    nonzero_regret_actions_added=nonzero,
                    initial_training_loss=report.initial_training_loss,
                    final_training_loss=report.final_training_loss,
                    player_0_mean_strategy_drift=_mean_policy_drift(
                        current_provider,
                        next_provider,
                        tuple(added_p0),
                        player=0,
                    ),
                    player_1_mean_strategy_drift=_mean_policy_drift(
                        current_provider,
                        next_provider,
                        tuple(added_p1),
                        player=1,
                    ),
                    average_plan_mean_strategy_drift=_mean_average_drift(
                        current_provider, next_provider, tuple(added_strategy)
                    ),
                    dataset_fingerprint=final_dataset.fingerprint,
                    model_fingerprint=training.artifact.fingerprint,
                )
                if not (
                    item.final_training_loss < item.initial_training_loss
                    and min(
                        item.player_0_mean_strategy_drift,
                        item.player_1_mean_strategy_drift,
                        item.average_plan_mean_strategy_drift,
                    )
                    > 0.0
                ):
                    raise RuntimeError(
                        f"Phase 6P iteration {iteration} did not update all models"
                    )
                round_evidence.append(item)
                current_provider = next_provider
                final_report = report
            assert final_dataset is not None and final_report is not None
            serialized = final_dataset.to_dict()
            restored_dataset = NativePlanOutcomeDataset.from_dict(
                json.loads(json.dumps(serialized))
            )
            if restored_dataset.fingerprint != final_dataset.fingerprint:
                raise RuntimeError("Phase 6P final dataset roundtrip drifted")
            raw_private = _raw_private_key_count(serialized)
            if raw_private != 0:
                raise RuntimeError("Phase 6P serialized a forbidden raw private key")
            claims = (
                ("native_descendant_transitions", "genuine_official_search_execution"),
                ("turn_plan_game", "research_abstraction"),
                ("hidden_completion_prior", "supplied_uncalibrated_abstraction"),
                ("iterative_collection", "previous_regret_models_drive_next_round"),
                ("average_strategy", "separate_iteration_indexed_memory_and_model"),
                ("neural_target", "train_split_asinh_changes_regret_magnitude"),
                ("equilibrium_convergence", "not_validated"),
                ("counterfactual_value_bounds", "not_calibrated"),
                ("safe_resolver", "not_enabled"),
                ("official_strategy_control", "disabled"),
                ("pokemon_strength", "not_claimed"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            evidence = Phase6NativePlanIterationsEvidence(
                benchmark_id="phase6p-iterative-native-plan-v1",
                status="phase6p_iterative_native_plan_candidate_validated_partial",
                initial_plan_model_fingerprint=initial_artifact.fingerprint,
                final_dataset_fingerprint=final_dataset.fingerprint,
                final_model_fingerprint=current_provider.artifact.fingerprint,
                native_game_fingerprint=native_game.fingerprint,
                runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                rounds=tuple(round_evidence),
                native_transitions=native_game.transition_count,
                macro_transitions=sum(
                    item.macro_transition_count for item in macro_games
                ),
                tactical_transitions=sum(
                    item.tactical_transition_count for item in macro_games
                ),
                raw_regret_median_absolute=final_report.raw_regret_median_absolute,
                raw_regret_p95_absolute=final_report.raw_regret_p95_absolute,
                raw_regret_maximum_absolute=final_report.raw_regret_maximum_absolute,
                maximum_restore_drift=final_report.maximum_restore_drift,
                raw_private_card_ids_persisted=raw_private,
                source_official_engine_executed=True,
                iterative_strategy_updates_executed=True,
                equilibrium_convergence_validated=False,
                claims=claims,
            )
            return Phase6NativePlanIterationsArtifacts(
                final_dataset,
                current_provider.artifact,
                final_report,
                evidence,
            )
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_plan_iterations().evidence.to_json())


if __name__ == "__main__":
    main()
