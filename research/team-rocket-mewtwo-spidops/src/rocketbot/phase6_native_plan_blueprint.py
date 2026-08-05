"""Phase 6O: native turn-plan outcome sampling and neural candidate evidence."""

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
from .native_blueprint_features import NativeBlueprintFeatureBridge
from .native_plan_blueprint import (
    NativePlanBlueprintArtifact,
    NativePlanOutcomeDataset,
    NativePlanTrainingReport,
    NativePlanTrainingRun,
    train_native_plan_blueprint,
)
from .native_plan_game import OfficialNativePlanGame, heuristic_plan_strategy
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
    OutcomeSamplingTraversal,
    OutcomeStrategyReservoir,
)
from .phase6_native_external_sampling import _raw_private_key_count
from .phase6_native_game import _completions, _runtime_basics


PHASE6_NATIVE_PLAN_BLUEPRINT_SCHEMA_VERSION = 1
PHASE6_NATIVE_PLAN_BLUEPRINT_SEED = 6_022


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _seed(*items: Any) -> int:
    return int.from_bytes(
        sha256(_canonical_json(items).encode("utf-8")).digest()[:8], "big"
    )


@dataclass(frozen=True, slots=True)
class Phase6NativePlanBlueprintEvidence:
    benchmark_id: str
    status: str
    dataset_fingerprint: str
    model_fingerprint: str
    native_game_fingerprint: str
    runtime_fingerprint: str
    source_tactical_blueprint_fingerprint: str
    candidate_count: int
    sampling_iterations: int
    completed_trajectories: int
    player_0_trajectories: int
    player_1_trajectories: int
    native_transitions: int
    macro_transitions: int
    tactical_transitions: int
    player_0_regret_samples: int
    player_1_regret_samples: int
    strategy_samples: int
    nonzero_regret_actions: int
    raw_regret_median_absolute: float
    raw_regret_p95_absolute: float
    raw_regret_maximum_absolute: float
    initial_training_loss: float
    final_training_loss: float
    maximum_restore_drift: float
    split_counts: tuple[tuple[str, tuple[int, int, int]], ...]
    raw_private_card_ids_persisted: int
    source_official_engine_executed: bool
    hidden_completion_prior_official: bool
    equilibrium_convergence_validated: bool
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6O evidence identity is invalid")
        for field_name in (
            "dataset_fingerprint",
            "model_fingerprint",
            "native_game_fingerprint",
            "runtime_fingerprint",
            "source_tactical_blueprint_fingerprint",
        ):
            value = getattr(self, field_name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        if min(
            self.candidate_count,
            self.sampling_iterations,
            self.completed_trajectories,
            self.player_0_trajectories,
            self.player_1_trajectories,
            self.native_transitions,
            self.macro_transitions,
            self.tactical_transitions,
            self.player_0_regret_samples,
            self.player_1_regret_samples,
            self.strategy_samples,
            self.nonzero_regret_actions,
        ) <= 0:
            raise ValueError("Phase 6O accounting must be positive")
        for field_name in (
            "raw_regret_median_absolute",
            "raw_regret_p95_absolute",
            "raw_regret_maximum_absolute",
            "initial_training_loss",
            "final_training_loss",
            "maximum_restore_drift",
        ):
            value = getattr(self, field_name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")
        if tuple(name for name, _ in self.split_counts) != (
            "train",
            "calibration",
            "validation",
        ):
            raise ValueError("Phase 6O split counts must be ordered and complete")
        if any(min(counts) <= 0 for _, counts in self.split_counts):
            raise ValueError("Phase 6O every split/head must contain samples")
        if (
            self.raw_private_card_ids_persisted != 0
            or not self.source_official_engine_executed
            or self.hidden_completion_prior_official
            or self.equilibrium_convergence_validated
        ):
            raise ValueError("Phase 6O provenance/capability boundary is invalid")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6O claim names must be unique")
        if dict(self.claims).get("phase6_complete") != "false":
            raise ValueError("Phase 6O cannot mark Phase 6 complete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NATIVE_PLAN_BLUEPRINT_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "native_game_fingerprint": self.native_game_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_tactical_blueprint_fingerprint": (
                self.source_tactical_blueprint_fingerprint
            ),
            "sampling": {
                "candidate_count": self.candidate_count,
                "iterations": self.sampling_iterations,
                "completed_trajectories": self.completed_trajectories,
                "player_0_trajectories": self.player_0_trajectories,
                "player_1_trajectories": self.player_1_trajectories,
                "native_transitions": self.native_transitions,
                "macro_transitions": self.macro_transitions,
                "tactical_transitions": self.tactical_transitions,
                "player_0_regret_samples": self.player_0_regret_samples,
                "player_1_regret_samples": self.player_1_regret_samples,
                "strategy_samples": self.strategy_samples,
                "nonzero_regret_actions": self.nonzero_regret_actions,
            },
            "variance": {
                "raw_regret_median_absolute": self.raw_regret_median_absolute,
                "raw_regret_p95_absolute": self.raw_regret_p95_absolute,
                "raw_regret_maximum_absolute": self.raw_regret_maximum_absolute,
                "neural_target": "train_split_scale_asinh",
            },
            "training": {
                "initial_loss": self.initial_training_loss,
                "final_loss": self.final_training_loss,
                "maximum_restore_drift": self.maximum_restore_drift,
                "split_counts": {
                    name: {
                        "player_0_regret": counts[0],
                        "player_1_regret": counts[1],
                        "average_strategy": counts[2],
                    }
                    for name, counts in self.split_counts
                },
            },
            "provenance": {
                "source_official_engine_executed": self.source_official_engine_executed,
                "hidden_completion_prior_official": self.hidden_completion_prior_official,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        claims = "\n".join(f"| {name} | {value} |" for name, value in self.claims)
        splits = "\n".join(
            f"| {name} | {counts[0]} | {counts[1]} | {counts[2]} |"
            for name, counts in self.split_counts
        )
        return f"""# Phase 6O native turn-plan outcome blueprint candidate

Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

Phase 6O moves the game-theoretic unit of choice from procedural option indices
to one of five persistent plans per turn. A plan-conditioned tactical executor
maps each abstract plan to legal selections; every descendant transition runs
through the licensed native search engine. The turn-plan actions, executor,
fixed hidden completion, sampled regrets, and neural models remain abstractions.

| Check | Result |
|---|---:|
| Hidden-completion candidates | {self.candidate_count} |
| Sampling iterations / trajectories | {self.sampling_iterations} / {self.completed_trajectories} |
| Player-0 / player-1 trajectories | {self.player_0_trajectories} / {self.player_1_trajectories} |
| Native / macro / tactical transitions | {self.native_transitions} / {self.macro_transitions} / {self.tactical_transitions} |
| Player-0 / player-1 regret samples | {self.player_0_regret_samples} / {self.player_1_regret_samples} |
| Average-strategy samples | {self.strategy_samples} |
| Nonzero sampled-regret actions | {self.nonzero_regret_actions} |
| Median / p95 / max absolute raw regret | {self.raw_regret_median_absolute:.6g} / {self.raw_regret_p95_absolute:.6g} / {self.raw_regret_maximum_absolute:.6g} |
| Training loss | {self.initial_training_loss:.8f} -> {self.final_training_loss:.8f} |
| Maximum checkpoint restore drift | {self.maximum_restore_drift:.3e} |
| Raw private card IDs persisted | {self.raw_private_card_ids_persisted} |

The heavy tail is material, not hidden: neural regret targets use a
train-split-only `asinh` transform. That stabilizes this wiring experiment but
changes regret magnitude and prevents an equilibrium/convergence claim.

## Information-set-disjoint splits

| Split | Player 0 regret | Player 1 regret | Average strategy |
|---|---:|---:|---:|
{splits}

## Claim boundary

| Claim | Status |
|---|---|
{claims}
"""


@dataclass(frozen=True, slots=True)
class Phase6NativePlanBlueprintArtifacts:
    dataset: NativePlanOutcomeDataset
    model: NativePlanBlueprintArtifact
    report: NativePlanTrainingReport
    evidence: Phase6NativePlanBlueprintEvidence


def run_phase6_native_plan_blueprint(
    *,
    project_root: str | Path | None = None,
    source_blueprint_path: str | Path | None = None,
    sampling_iterations: int = 3,
    exploration: float = 0.2,
    reservoir_capacity: int = 8_192,
    maximum_macro_transitions: int = 256,
    maximum_tactical_transitions: int = 512,
    epochs: int = 30,
    learning_rate: float = 2e-3,
    seed: int = PHASE6_NATIVE_PLAN_BLUEPRINT_SEED,
) -> Phase6NativePlanBlueprintArtifacts:
    if min(
        sampling_iterations,
        reservoir_capacity,
        maximum_macro_transitions,
        maximum_tactical_transitions,
        epochs,
    ) <= 0 or not 0 < exploration <= 1 or learning_rate <= 0:
        raise ValueError("Phase 6O limits are invalid")
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    runtime = root / "work" / "official_kaggle" / "sample_submission"
    blueprint_path = Path(
        source_blueprint_path
        or root / "artifacts" / "phase6_native_blueprint" / "native_regret_blueprint.json"
    ).resolve()
    source_artifact = NativeRegretBlueprintArtifact.from_dict(
        json.loads(blueprint_path.read_text(encoding="utf-8"))
    )
    source_provider = restore_native_regret_blueprint(source_artifact)
    bridge = NativeBlueprintFeatureBridge()
    path_text = str(runtime)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    from cg.game import battle_finish, battle_start  # type: ignore[import-not-found]

    target = rocket_compatibility_profile(
        resolve_official_deck(
            substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
            fail_on_unresolved=True,
        )
    )
    candidates: tuple[KaggleDeckProfile, ...] = OFFICIAL_PANEL_DECKS + (target,)
    observation, start = battle_start(
        list(target.card_ids), list(OFFICIAL_PANEL_DECKS[0].card_ids)
    )
    if observation is None:
        raise RuntimeError(
            f"official engine rejected Phase 6O decks: "
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
            regrets = (
                OutcomeRegretReservoir(reservoir_capacity, seed=_seed(seed, "p0")),
                OutcomeRegretReservoir(reservoir_capacity, seed=_seed(seed, "p1")),
            )
            strategies = OutcomeStrategyReservoir(
                2 * reservoir_capacity,
                seed=_seed(seed, "strategy"),
            )
            macro_fingerprints = []
            completed = [0, 0]
            macro_transitions = 0
            tactical_transitions = 0
            traversal = OutcomeSamplingTraversal(
                OutcomeSamplingConfig(
                    exploration=exploration,
                    maximum_transitions=maximum_macro_transitions,
                    maximum_depth=maximum_macro_transitions,
                )
            )
            for candidate_index, (candidate, outcome) in enumerate(
                zip(
                    candidates,
                    native_game.chance_outcomes(native_game.root_state()),
                    strict=True,
                )
            ):
                scenario = native_game.transition(
                    native_game.root_state(),
                    None,
                    chance_outcome=outcome,
                )
                macro_game = OfficialNativePlanGame(
                    native_game,
                    scenario,
                    bridge=bridge,
                    tactical_provider=source_provider,
                    maximum_tactical_transitions=maximum_tactical_transitions,
                )
                macro_fingerprints.append(macro_game.fingerprint)
                for iteration in range(1, sampling_iterations + 1):
                    for traverser in (0, 1):
                        result = traversal.run(
                            macro_game,
                            heuristic_plan_strategy,
                            traverser=traverser,
                            iteration=iteration,
                            rng=random.Random(
                                _seed(seed, candidate_index, iteration, traverser)
                            ),
                            baseline=macro_game.plan_value_baseline,
                        )
                        completed[traverser] += 1
                        macro_transitions += result.transitions
                        for sample in result.regret_samples:
                            regrets[traverser].add(sample)
                        for sample in result.strategy_samples:
                            strategies.add(sample)
                tactical_transitions += macro_game.tactical_transition_count
            dataset = NativePlanOutcomeDataset(
                native_game_fingerprint=native_game.fingerprint,
                runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                deck_fingerprint=target.fingerprint,
                source_tactical_blueprint_fingerprint=source_artifact.fingerprint,
                macro_game_fingerprints=tuple(macro_fingerprints),
                candidate_ids=tuple(candidate.archetype for candidate in candidates),
                iterations=sampling_iterations,
                exploration=exploration,
                player_0_regrets=regrets[0],
                player_1_regrets=regrets[1],
                average_strategy=strategies,
                native_transition_count=native_game.transition_count,
                macro_transition_count=macro_transitions,
                tactical_transition_count=tactical_transitions,
            )
            serialized = dataset.to_dict()
            restored_dataset = NativePlanOutcomeDataset.from_dict(
                json.loads(json.dumps(serialized))
            )
            if restored_dataset.fingerprint != dataset.fingerprint:
                raise RuntimeError("Phase 6O dataset roundtrip drifted")
            if _raw_private_key_count(serialized) != 0:
                raise RuntimeError("Phase 6O serialized a forbidden raw private key")
            training: NativePlanTrainingRun = train_native_plan_blueprint(
                dataset,
                source_provider,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
            )
            report = training.report
            split_counts = tuple(
                (
                    item.split,
                    (
                        item.player_0_samples,
                        item.player_1_samples,
                        item.strategy_samples,
                    ),
                )
                for item in report.split_metrics
            )
            nonzero = sum(
                abs(value) > 1e-12
                for sample in regrets[0].samples + regrets[1].samples
                for _, value in sample.sampled_regrets
            )
            claims = (
                ("native_descendant_transitions", "genuine_official_search_execution"),
                ("turn_plan_actions", "research_abstraction_not_official_options"),
                ("tactical_executor", "phase6k_behavior_conditioned_by_plan_bias"),
                ("hidden_completion_prior", "supplied_research_abstraction"),
                ("sampling", "primary_source_aligned_outcome_mccfr_estimator"),
                ("variance", "heavy_tailed_raw_regrets_reported"),
                ("neural_target", "train_split_asinh_transform_changes_magnitude"),
                ("separate_models", "player_regret_heads_and_average_plan_head"),
                ("equilibrium_convergence", "not_validated"),
                ("counterfactual_value_bounds", "not_calibrated_full_game"),
                ("safe_resolver", "not_deployed_official"),
                ("official_strategy_control", "disabled"),
                ("pokemon_strength", "not_claimed"),
                ("kaggle_submission", "not_attempted"),
                ("phase6_complete", "false"),
            )
            expected_parameter_names = {
                f"{prefix}.{name}"
                for prefix, network in training.provider.networks.items()
                for name, _ in network.named_parameters()
            }
            gates = (
                completed == [len(candidates) * sampling_iterations] * 2,
                regrets[0].seen > 0,
                regrets[1].seen > 0,
                strategies.seen > 0,
                nonzero > 0,
                report.final_training_loss < report.initial_training_loss,
                set(report.gradient_parameter_names) == expected_parameter_names,
                report.maximum_restore_drift == 0.0,
                all(min(counts) > 0 for _, counts in split_counts),
                _raw_private_key_count(serialized) == 0,
            )
            status = (
                "phase6o_native_turn_plan_blueprint_validated_partial"
                if all(gates)
                else "failed"
            )
            evidence = Phase6NativePlanBlueprintEvidence(
                benchmark_id="phase6o-native-turn-plan-blueprint-v1",
                status=status,
                dataset_fingerprint=dataset.fingerprint,
                model_fingerprint=training.artifact.fingerprint,
                native_game_fingerprint=native_game.fingerprint,
                runtime_fingerprint=native_game.provenance.ruleset_fingerprint,
                source_tactical_blueprint_fingerprint=source_artifact.fingerprint,
                candidate_count=len(candidates),
                sampling_iterations=sampling_iterations,
                completed_trajectories=sum(completed),
                player_0_trajectories=completed[0],
                player_1_trajectories=completed[1],
                native_transitions=native_game.transition_count,
                macro_transitions=macro_transitions,
                tactical_transitions=tactical_transitions,
                player_0_regret_samples=regrets[0].seen,
                player_1_regret_samples=regrets[1].seen,
                strategy_samples=strategies.seen,
                nonzero_regret_actions=nonzero,
                raw_regret_median_absolute=report.raw_regret_median_absolute,
                raw_regret_p95_absolute=report.raw_regret_p95_absolute,
                raw_regret_maximum_absolute=report.raw_regret_maximum_absolute,
                initial_training_loss=report.initial_training_loss,
                final_training_loss=report.final_training_loss,
                maximum_restore_drift=report.maximum_restore_drift,
                split_counts=split_counts,
                raw_private_card_ids_persisted=_raw_private_key_count(serialized),
                source_official_engine_executed=True,
                hidden_completion_prior_official=False,
                equilibrium_convergence_validated=False,
                claims=claims,
            )
            if evidence.status == "failed":
                failed = [index for index, passed in enumerate(gates) if not passed]
                raise RuntimeError(f"Phase 6O failed declared gates {failed}")
            return Phase6NativePlanBlueprintArtifacts(
                dataset,
                training.artifact,
                report,
                evidence,
            )
    finally:
        battle_finish()


def main() -> None:
    print(run_phase6_native_plan_blueprint().evidence.to_json())


if __name__ == "__main__":
    main()
