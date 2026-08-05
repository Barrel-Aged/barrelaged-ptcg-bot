"""Phase 6N evidence: scalable outcome-sampling MCCFR reference."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any

from .exact_game import RocketPrizeDuel, exact_profile_metrics, uniform_profile
from .outcome_sampling_reference import (
    OUTCOME_SAMPLING_PRIMARY_SOURCES,
    OutcomeSamplingExpectationAudit,
    OutcomeSamplingMCCFRSolver,
    OutcomeSamplingTrainingResult,
    audit_outcome_sampling_expectation,
)


PHASE6_OUTCOME_REFERENCE_SCHEMA_VERSION = 1
PHASE6_OUTCOME_REFERENCE_SEED = 6_021


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase6OutcomeReferenceEvidence:
    benchmark_id: str
    status: str
    seed: int
    exploration: float
    expectation_audit_fingerprint: str
    maximum_expectation_error: float
    enumerated_trajectories: int
    training_iterations: int
    training_traversals: int
    training_transitions: int
    training_regret_samples: int
    training_strategy_samples: int
    solver_fingerprint: str
    uniform_nash_conv: float
    trained_nash_conv: float
    trained_exploitability: float
    trained_expected_value_player_0: float
    primary_sources: tuple[str, ...]
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status or self.seed < 0:
            raise ValueError("Phase 6N evidence identity/seed is invalid")
        if not 0 < self.exploration <= 1:
            raise ValueError("Phase 6N exploration is invalid")
        for field_name in ("expectation_audit_fingerprint", "solver_fingerprint"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        for field_name in (
            "maximum_expectation_error",
            "uniform_nash_conv",
            "trained_nash_conv",
            "trained_exploitability",
            "trained_expected_value_player_0",
        ):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if min(
            self.enumerated_trajectories,
            self.training_iterations,
            self.training_traversals,
            self.training_transitions,
            self.training_regret_samples,
            self.training_strategy_samples,
        ) <= 0:
            raise ValueError("Phase 6N evidence accounting must be positive")
        if self.primary_sources != OUTCOME_SAMPLING_PRIMARY_SOURCES:
            raise ValueError("Phase 6N must bind reviewed primary sources")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6N claim names must be unique")
        if dict(self.claims).get("phase6_complete") != "false":
            raise ValueError("Phase 6N cannot mark Phase 6 complete")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_OUTCOME_REFERENCE_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "seed": self.seed,
            "exploration": self.exploration,
            "expectation": {
                "audit_fingerprint": self.expectation_audit_fingerprint,
                "maximum_absolute_error": self.maximum_expectation_error,
                "enumerated_trajectories": self.enumerated_trajectories,
                "nonzero_pre_action_baseline": True,
            },
            "training": {
                "iterations": self.training_iterations,
                "traversals": self.training_traversals,
                "transitions": self.training_transitions,
                "regret_samples": self.training_regret_samples,
                "strategy_samples": self.training_strategy_samples,
                "solver_fingerprint": self.solver_fingerprint,
            },
            "metrics": {
                "uniform_nash_conv": self.uniform_nash_conv,
                "trained_nash_conv": self.trained_nash_conv,
                "trained_exploitability": self.trained_exploitability,
                "trained_expected_value_player_0": (
                    self.trained_expected_value_player_0
                ),
            },
            "primary_sources": list(self.primary_sources),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        claims = "\n".join(f"| {name} | {value} |" for name, value in self.claims)
        sources = "\n".join(f"- {source}" for source in self.primary_sources)
        return f"""# Phase 6N outcome-sampling MCCFR reference

Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

This milestone adds a linear-cost terminal-trajectory estimator for native-scale
research. The zero-baseline recursion is generalized with a pre-action baseline;
an arbitrary nonzero baseline is used in the exhaustive audit to prove that the
correction remains unbiased on the abstract `RocketPrizeDuel`.

| Check | Result |
|---|---:|
| Enumerated sampled trajectories | {self.enumerated_trajectories} |
| Maximum expectation error | {self.maximum_expectation_error:.3e} |
| Exploration | {self.exploration:.3f} |
| Training iterations / traversals | {self.training_iterations} / {self.training_traversals} |
| Training transitions | {self.training_transitions} |
| Regret / strategy samples | {self.training_regret_samples} / {self.training_strategy_samples} |
| Uniform NashConv | {self.uniform_nash_conv:.9f} |
| Trained NashConv | {self.trained_nash_conv:.9f} |
| Trained exploitability | {self.trained_exploitability:.9f} |
| Trained player-0 value | {self.trained_expected_value_player_0:.9f} |

Outcome sampling is much higher variance than external sampling. These exact
metrics apply only to the enumerable microgame and do not certify a native
neural strategy.

## Primary methodology sources

{sources}

## Claim boundary

| Claim | Status |
|---|---|
{claims}
"""


@dataclass(frozen=True, slots=True)
class Phase6OutcomeReferenceArtifacts:
    audit: OutcomeSamplingExpectationAudit
    training: OutcomeSamplingTrainingResult
    evidence: Phase6OutcomeReferenceEvidence


def run_phase6_outcome_sampling_reference(
    *,
    training_iterations: int = 20_000,
    exploration: float = 0.6,
    seed: int = PHASE6_OUTCOME_REFERENCE_SEED,
) -> Phase6OutcomeReferenceArtifacts:
    if training_iterations <= 0:
        raise ValueError("Phase 6N training iterations must be positive")
    game = RocketPrizeDuel()
    uniform = uniform_profile(game)
    baseline = exact_profile_metrics(game, uniform)
    audit = audit_outcome_sampling_expectation(
        game,
        uniform,
        exploration=exploration,
    )
    training = OutcomeSamplingMCCFRSolver(
        game,
        exploration=exploration,
    ).train(training_iterations, seed=seed)
    metrics = training.metrics
    gates = (
        audit.maximum_absolute_error < 2e-12,
        len(audit.rows) == 24,
        audit.trajectory_count_player_0 > 0,
        audit.trajectory_count_player_1 > 0,
        metrics.nash_conv < 0.08,
        metrics.nash_conv < baseline.nash_conv * 0.15,
        not audit.official_engine_executed,
        not training.official_engine_executed,
        not training.full_game_equilibrium_convergence,
    )
    status = (
        "phase6n_outcome_sampling_reference_validated_partial"
        if all(gates)
        else "failed"
    )
    claims = (
        ("algorithm_reference", "outcome_sampling_mccfr_primary_source_aligned"),
        ("baseline_correction", "exact_nonzero_baseline_expectation_validated"),
        ("trajectory_cost", "one_complete_terminal_history_per_traverser"),
        ("microgame_convergence", "exact_nashconv_reported_high_variance"),
        ("native_transition_execution", "not_in_this_reference_artifact"),
        ("native_full_game_equilibrium", "not_validated"),
        ("deep_cfr", "not_claimed_outcome_sampling_is_distinct"),
        ("counterfactual_value_bounds", "not_calibrated_full_game"),
        ("safe_resolver", "not_deployed_official"),
        ("pokemon_strength", "not_claimed"),
        ("kaggle_submission", "not_attempted"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6OutcomeReferenceEvidence(
        benchmark_id="phase6n-outcome-sampling-reference-v1",
        status=status,
        seed=seed,
        exploration=exploration,
        expectation_audit_fingerprint=audit.fingerprint,
        maximum_expectation_error=audit.maximum_absolute_error,
        enumerated_trajectories=(
            audit.trajectory_count_player_0 + audit.trajectory_count_player_1
        ),
        training_iterations=training.iterations,
        training_traversals=training.traversals,
        training_transitions=training.transitions,
        training_regret_samples=training.regret_samples,
        training_strategy_samples=training.strategy_samples,
        solver_fingerprint=training.solver_fingerprint,
        uniform_nash_conv=baseline.nash_conv,
        trained_nash_conv=metrics.nash_conv,
        trained_exploitability=metrics.exploitability,
        trained_expected_value_player_0=metrics.expected_value_player_0,
        primary_sources=OUTCOME_SAMPLING_PRIMARY_SOURCES,
        claims=claims,
    )
    if evidence.status == "failed":
        raise RuntimeError("Phase 6N outcome-sampling reference failed a gate")
    return Phase6OutcomeReferenceArtifacts(audit, training, evidence)


def main() -> None:
    print(run_phase6_outcome_sampling_reference().evidence.to_json())


if __name__ == "__main__":
    main()
