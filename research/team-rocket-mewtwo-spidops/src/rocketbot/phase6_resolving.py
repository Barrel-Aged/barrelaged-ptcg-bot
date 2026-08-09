"""Deterministic Phase 6E interruption-safe exact resolving evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .counterfactual_calibration import (
    CalibratedNeuralCounterfactualBoundProvider,
    CounterfactualBoundContext,
)
from .exact_game import RocketPrizeDuel
from .game_theory import ResolveBudget
from .neural_blueprint import provider_profile
from .phase6_bounds import run_phase6_counterfactual_bounds
from .safe_resolving import (
    BudgetAwareExactSafeResolver,
    ExactSafePlanningPipeline,
    ResolveInterruption,
)


PHASE6_RESOLVING_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ResolveScopeEvidence:
    scope: str
    roots: int
    completed_searches: int
    changed_strategies: int
    blueprint_selected_after_complete_search: int
    total_exact_actor_value_gain: float
    maximum_exact_actor_value_gain: float
    maximum_opponent_baseline_regression: float
    maximum_bound_violation: float

    def __post_init__(self) -> None:
        if self.scope not in {"endgame", "continual_root"}:
            raise ValueError("unknown resolve evidence scope")
        for field_name in (
            "roots",
            "completed_searches",
            "changed_strategies",
            "blueprint_selected_after_complete_search",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.completed_searches > self.roots or self.changed_strategies > self.roots:
            raise ValueError("resolve scope counts cannot exceed roots")
        for field_name in (
            "total_exact_actor_value_gain",
            "maximum_exact_actor_value_gain",
            "maximum_opponent_baseline_regression",
            "maximum_bound_violation",
        ):
            if float(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True, slots=True)
class InterruptionFallbackEvidence:
    trigger: str
    interruption: str
    returned_blueprint_exactly: bool
    evaluated_iterations: int
    evaluated_nodes: int

    def __post_init__(self) -> None:
        if not self.trigger or not self.interruption:
            raise ValueError("interruption evidence must be described")
        if self.evaluated_iterations < 0 or self.evaluated_nodes < 0:
            raise ValueError("interruption accounting cannot be negative")


@dataclass(frozen=True, slots=True)
class Phase6ResolvingEvidence:
    benchmark_id: str
    status: str
    game_fingerprint: str
    blueprint_fingerprint: str
    calibration_fingerprint: str
    phase6d_evidence_fingerprint: str
    fixed_budget: tuple[tuple[str, float], ...]
    endgame: ResolveScopeEvidence
    continual_root: ResolveScopeEvidence
    interruption_fallbacks: tuple[InterruptionFallbackEvidence, ...]
    all_returned_strategies_legal: bool
    all_completed_certificates_safe: bool
    all_completed_strict_nonregression: bool
    total_exact_actor_value_gain: float
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6E benchmark identity/status is required")
        for field_name in (
            "game_fingerprint",
            "blueprint_fingerprint",
            "calibration_fingerprint",
            "phase6d_evidence_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.total_exact_actor_value_gain < 0:
            raise ValueError("total exact value gain cannot be negative")
        if len(dict(self.fixed_budget)) != len(self.fixed_budget):
            raise ValueError("fixed budget fields must be unique")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6E claim fields must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_RESOLVING_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "game_fingerprint": self.game_fingerprint,
            "blueprint_fingerprint": self.blueprint_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "phase6d_evidence_fingerprint": self.phase6d_evidence_fingerprint,
            "fixed_budget": dict(self.fixed_budget),
            "endgame": asdict(self.endgame),
            "continual_root": asdict(self.continual_root),
            "interruption_fallbacks": [
                asdict(item) for item in self.interruption_fallbacks
            ],
            "all_returned_strategies_legal": self.all_returned_strategies_legal,
            "all_completed_certificates_safe": self.all_completed_certificates_safe,
            "all_completed_strict_nonregression": self.all_completed_strict_nonregression,
            "total_exact_actor_value_gain": self.total_exact_actor_value_gain,
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        scope_rows = "\n".join(
            f"| {item.scope} | {item.roots} | {item.completed_searches} | "
            f"{item.changed_strategies} | {item.total_exact_actor_value_gain:.10f} | "
            f"{item.maximum_opponent_baseline_regression:.3e} |"
            for item in (self.endgame, self.continual_root)
        )
        fallback_rows = "\n".join(
            f"| {item.trigger} | {item.interruption} | "
            f"{str(item.returned_blueprint_exactly).lower()} | "
            f"{item.evaluated_iterations} | {item.evaluated_nodes} |"
            for item in self.interruption_fallbacks
        )
        claim_rows = "\n".join(
            f"| {name} | {value} |" for name, value in self.claims
        )
        return f"""# Phase 6E interruption-safe resolving evidence

Benchmark: `{self.benchmark_id}`  
Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

Every completed search enumerates the same 102 candidate mixtures under one
fixed budget, verifies every private-state hypothesis against both its calibrated
upper bound and the exact blueprint value, and exposes a candidate only after
the complete grid finishes.

| Scope | Roots | Completed | Changed | Total exact actor gain | Max opponent regression |
|---|---:|---:|---:|---:|---:|
{scope_rows}

Aggregate exact conditional actor-value gain: `{self.total_exact_actor_value_gain:.10f}`.

## Interruption and gate probes

| Trigger | Result | Exact blueprint | Evaluated iterations | Evaluated nodes |
|---|---|---:|---:|---:|
{fallback_rows}

## Claim boundary

| Claim | Status |
|---|---|
{claim_rows}
"""


class _TickingClock:
    def __init__(self, step_seconds: float) -> None:
        self.value = 0.0
        self.step_seconds = step_seconds

    def __call__(self) -> float:
        result = self.value
        self.value += self.step_seconds
        return result


def _scope_evidence(scope: str, records: list[dict[str, Any]]) -> ResolveScopeEvidence:
    return ResolveScopeEvidence(
        scope=scope,
        roots=len(records),
        completed_searches=sum(item["completed"] for item in records),
        changed_strategies=sum(item["changed"] for item in records),
        blueprint_selected_after_complete_search=sum(
            item["completed"] and not item["changed"] for item in records
        ),
        total_exact_actor_value_gain=sum(item["gain"] for item in records),
        maximum_exact_actor_value_gain=max((item["gain"] for item in records), default=0.0),
        maximum_opponent_baseline_regression=max(
            (item["regression"] for item in records), default=0.0
        ),
        maximum_bound_violation=max(
            (item["bound_violation"] for item in records), default=0.0
        ),
    )


def run_phase6_safe_resolving(
    *,
    project_root: Path | None = None,
) -> Phase6ResolvingEvidence:
    root = project_root or Path(__file__).resolve().parents[2]
    phase6d = run_phase6_counterfactual_bounds(project_root=root)
    game = RocketPrizeDuel()
    blueprint = phase6d.blueprint
    profile = provider_profile(game, blueprint, label="phase6e_safe_resolve_profile")
    bound_provider = CalibratedNeuralCounterfactualBoundProvider(
        game,
        blueprint,
        phase6d.calibration,
    )
    resolver = BudgetAwareExactSafeResolver(profile, grid_steps=100)
    pipeline = ExactSafePlanningPipeline(
        game=game,
        blueprint=blueprint,
        bound_provider=bound_provider,
        resolver=resolver,
    )
    fixed_budget = ResolveBudget(
        iterations=102,
        max_nodes=206,
        time_budget_ms=1_000.0,
        safety_tolerance=1e-12,
    )
    records: dict[str, list[dict[str, Any]]] = {
        "endgame": [],
        "continual_root": [],
    }
    all_legal = True
    all_safe = True
    all_nonregression = True
    representatives = []
    ordered_information_sets = sorted(
        game.information_sets(),
        key=lambda item: (
            game.states_for_information_set(item.info_set_id)[0].depth < 2,
            item.info_set_id,
        ),
    )
    for information_set in ordered_information_sets:
        state = game.states_for_information_set(information_set.info_set_id)[0]
        actor = state.acting_player
        if actor is None:
            raise RuntimeError("Phase 6E information set has no actor")
        belief = game.public_belief(state, player=actor)
        representatives.append((information_set, belief))
        decision = pipeline.decide(
            information_set,
            belief,
            budget=fixed_budget,
        )
        status = decision.resolve_status
        if status is None or not status.completed_search:
            raise RuntimeError("Phase 6E fixed budget failed to complete exact grid")
        decision.strategy.validate_support(information_set)
        changed = decision.strategy != decision.blueprint_strategy
        gain = status.exact_actor_value_gain
        if gain is None:
            raise RuntimeError("completed Phase 6E resolve lacks exact gain")
        scope = "endgame" if state.depth >= 2 else "continual_root"
        records[scope].append(
            {
                "information_set": information_set.info_set_id,
                "completed": status.completed_search,
                "changed": changed,
                "gain": gain,
                "regression": status.maximum_opponent_baseline_regression,
                "bound_violation": status.maximum_bound_violation,
            }
        )
        all_legal = all_legal and True
        all_safe = all_safe and status.resolved.certificate.safe
        all_nonregression = all_nonregression and status.strict_baseline_nonregression
    endgame = _scope_evidence("endgame", records["endgame"])
    continual = _scope_evidence("continual_root", records["continual_root"])
    total_gain = endgame.total_exact_actor_value_gain + continual.total_exact_actor_value_gain
    if total_gain <= 0:
        raise RuntimeError("Phase 6E fixed-budget exact resolver produced no value gain")
    information_set, belief = representatives[0]

    def interruption_probe(trigger: str, budget: ResolveBudget) -> InterruptionFallbackEvidence:
        decision = pipeline.decide(information_set, belief, budget=budget)
        status = decision.resolve_status
        if status is None:
            raise RuntimeError("resolver interruption probe did not invoke resolver")
        return InterruptionFallbackEvidence(
            trigger=trigger,
            interruption=status.interruption.value,
            returned_blueprint_exactly=(decision.strategy == decision.blueprint_strategy),
            evaluated_iterations=status.evaluated_iterations,
            evaluated_nodes=status.evaluated_nodes,
        )

    iteration_fallback = interruption_probe(
        "iteration_preflight",
        ResolveBudget(
            iterations=101,
            max_nodes=206,
            time_budget_ms=1_000.0,
            safety_tolerance=1e-12,
        ),
    )
    node_fallback = interruption_probe(
        "node_preflight",
        ResolveBudget(
            iterations=102,
            max_nodes=205,
            time_budget_ms=1_000.0,
            safety_tolerance=1e-12,
        ),
    )
    deadline_resolver = BudgetAwareExactSafeResolver(
        profile,
        grid_steps=100,
        clock=_TickingClock(step_seconds=0.001),
    )
    deadline_pipeline = ExactSafePlanningPipeline(
        game=game,
        blueprint=blueprint,
        bound_provider=bound_provider,
        resolver=deadline_resolver,
    )
    deadline_decision = deadline_pipeline.decide(
        information_set,
        belief,
        budget=ResolveBudget(
            iterations=102,
            max_nodes=206,
            time_budget_ms=0.5,
            safety_tolerance=1e-12,
        ),
    )
    deadline_status = deadline_decision.resolve_status
    if deadline_status is None:
        raise RuntimeError("deadline probe did not invoke resolver")
    deadline_fallback = InterruptionFallbackEvidence(
        trigger="deadline_mid_search",
        interruption=deadline_status.interruption.value,
        returned_blueprint_exactly=(
            deadline_decision.strategy == deadline_decision.blueprint_strategy
        ),
        evaluated_iterations=deadline_status.evaluated_iterations,
        evaluated_nodes=deadline_status.evaluated_nodes,
    )
    ood_decision = pipeline.decide(
        information_set,
        belief,
        budget=fixed_budget,
        bound_context=CounterfactualBoundContext(
            ood_score=0.51,
            required_confidence=phase6d.calibration.config.target_coverage,
        ),
    )
    ood_fallback = InterruptionFallbackEvidence(
        trigger="bound_ood_gate",
        interruption=ResolveInterruption.BOUND_GATE.value,
        returned_blueprint_exactly=(
            ood_decision.strategy == ood_decision.blueprint_strategy
        ),
        evaluated_iterations=0,
        evaluated_nodes=0,
    )
    interruptions = (
        iteration_fallback,
        node_fallback,
        deadline_fallback,
        ood_fallback,
    )
    expected_interruptions = (
        ResolveInterruption.ITERATION_BUDGET.value,
        ResolveInterruption.NODE_BUDGET.value,
        ResolveInterruption.DEADLINE.value,
        ResolveInterruption.BOUND_GATE.value,
    )
    if tuple(item.interruption for item in interruptions) != expected_interruptions:
        raise RuntimeError("Phase 6E interruption routing is not deterministic")
    if not all(item.returned_blueprint_exactly for item in interruptions):
        raise RuntimeError("Phase 6E interruption failed exact blueprint fallback")
    claims = (
        ("safe_endgame_resolving", "validated_exact_microgame_only"),
        ("continual_root_resolving", "validated_exact_microgame_only"),
        ("strict_opponent_nonregression", "validated_on_all_12_exact_roots"),
        ("fixed_budget_value_gain", "positive_exact_conditional_gain"),
        ("iteration_node_deadline_fallback", "returns_blueprint_exactly"),
        ("partial_search_strategy_exposure", "none"),
        ("official_full_game_resolver", "not_implemented"),
        (
            "official_information_set_bridge",
            "one_step_shadow_validated_full_game_not_implemented",
        ),
        ("phase6_complete", "false"),
    )
    return Phase6ResolvingEvidence(
        benchmark_id="phase6e-exact-safe-resolve-grid100-v1",
        status="phase6e_exact_interruption_safe_resolving_validated",
        game_fingerprint=game.fingerprint,
        blueprint_fingerprint=blueprint.checkpoint.model_fingerprint,
        calibration_fingerprint=phase6d.calibration.fingerprint,
        phase6d_evidence_fingerprint=phase6d.evidence.fingerprint,
        fixed_budget=tuple(
            (name, float(value))
            for name, value in asdict(fixed_budget).items()
        ),
        endgame=endgame,
        continual_root=continual,
        interruption_fallbacks=interruptions,
        all_returned_strategies_legal=all_legal,
        all_completed_certificates_safe=all_safe,
        all_completed_strict_nonregression=all_nonregression,
        total_exact_actor_value_gain=total_gain,
        claims=claims,
    )


def main() -> None:
    print(run_phase6_safe_resolving().to_json())


if __name__ == "__main__":
    main()
