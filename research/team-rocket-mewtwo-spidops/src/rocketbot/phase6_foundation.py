"""Deterministic Phase 6A-6B evidence: public replay data and exact game theory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .competition import KAGGLE_WATCHTOWER_SUBSTITUTION, resolve_official_deck
from .deck import load_default_deck
from .exact_game import (
    ANALYTIC_GAME_VALUE_PLAYER_0,
    DuelCard,
    ExactGridSafeResolver,
    ExactProfileMetrics,
    ExactTabularBlueprintProvider,
    RocketPrizeDuel,
    TabularCFRSolver,
    analytic_equilibrium_profile,
    deal_count,
    evaluate_profile_from_state,
    exact_counterfactual_bounds,
    exact_profile_metrics,
    information_set_action_count,
    terminal_state_count,
    uniform_profile,
)
from .game_theory import ResolveBudget, SafeResolveRequest
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    rocket_compatibility_profile,
)
from .public_replay import (
    OfficialReplayConverter,
    PublicReplayDataset,
    build_public_replay_dataset,
)


PHASE6_FOUNDATION_SCHEMA_VERSION = 1
PHASE6_FOUNDATION_SEED = 20260726


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SafeResolveEvidence:
    information_set_id: str
    baseline_strategy: tuple[tuple[str, float], ...]
    resolved_strategy: tuple[tuple[str, float], ...]
    baseline_actor_value: float
    resolved_actor_value: float
    actor_value_gain: float
    maximum_counterfactual_value_violation: float
    bound_confidence: float
    safe: bool
    candidate_strategies: int
    expanded_hypothesis_nodes: int
    method: str

    def __post_init__(self) -> None:
        for field_name in (
            "baseline_actor_value",
            "resolved_actor_value",
            "actor_value_gain",
            "maximum_counterfactual_value_violation",
            "bound_confidence",
        ):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if not self.information_set_id or not self.method:
            raise ValueError("safe resolve evidence identity/method is required")
        if self.candidate_strategies <= 0 or self.expanded_hypothesis_nodes <= 0:
            raise ValueError("safe resolver must record positive exact work")
        if self.maximum_counterfactual_value_violation < 0:
            raise ValueError("counterfactual violation cannot be negative")


@dataclass(frozen=True, slots=True)
class Phase6FoundationEvidence:
    benchmark_id: str
    source_phase6_evidence_fingerprint: str
    public_replay_dataset_fingerprint: str
    public_replay_summary: tuple[tuple[str, Any], ...]
    game_name: str
    game_fingerprint: str
    game_boundary_kind: str
    official_engine_executed_by_game: bool
    game_state_count: int
    terminal_state_count: int
    information_set_count: int
    information_set_action_count: int
    chance_deal_count: int
    analytic_metrics: ExactProfileMetrics
    uniform_metrics: ExactProfileMetrics
    cfr_iterations: int
    cfr_solver_fingerprint: str
    cfr_metrics: ExactProfileMetrics
    cfr_value_error: float
    safe_resolve: SafeResolveEvidence
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.game_name:
            raise ValueError("foundation evidence identity is required")
        for field_name in (
            "source_phase6_evidence_fingerprint",
            "public_replay_dataset_fingerprint",
            "game_fingerprint",
            "cfr_solver_fingerprint",
        ):
            value = getattr(self, field_name)
            if len(value) != 64:
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        if self.game_boundary_kind != "abstract_information_set_game":
            raise ValueError("exact validation game must be labeled abstract")
        if self.official_engine_executed_by_game:
            raise ValueError("abstract exact game cannot claim official execution")
        if min(
            self.game_state_count,
            self.terminal_state_count,
            self.information_set_count,
            self.information_set_action_count,
            self.chance_deal_count,
            self.cfr_iterations,
        ) <= 0:
            raise ValueError("foundation game/CFR counts must be positive")
        if self.cfr_value_error < 0 or not math.isfinite(self.cfr_value_error):
            raise ValueError("CFR value error must be finite and non-negative")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def status(self) -> str:
        claims = dict(self.claims)
        local_required = (
            "public_replay_conversion",
            "known_hidden_state_leakage",
            "range_supervision_labels",
            "exact_validation_game_boundary",
            "analytic_equilibrium_reference",
            "tabular_cfr_reference",
            "exact_microgame_safe_resolve",
        )
        return (
            "phase6ab_foundation_validated"
            if all(claims[name].startswith("validated") or claims[name].startswith("zero_") for name in local_required)
            else "failed"
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_FOUNDATION_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "source_phase6_evidence_fingerprint": self.source_phase6_evidence_fingerprint,
            "public_replay_dataset_fingerprint": self.public_replay_dataset_fingerprint,
            "public_replay_summary": dict(self.public_replay_summary),
            "game": {
                "name": self.game_name,
                "fingerprint": self.game_fingerprint,
                "boundary_kind": self.game_boundary_kind,
                "official_engine_executed": self.official_engine_executed_by_game,
                "state_count": self.game_state_count,
                "terminal_state_count": self.terminal_state_count,
                "information_set_count": self.information_set_count,
                "information_set_action_count": self.information_set_action_count,
                "chance_deal_count": self.chance_deal_count,
            },
            "analytic_reference": asdict(self.analytic_metrics),
            "uniform_baseline": asdict(self.uniform_metrics),
            "tabular_cfr": {
                "iterations": self.cfr_iterations,
                "solver_fingerprint": self.cfr_solver_fingerprint,
                "metrics": asdict(self.cfr_metrics),
                "value_error_from_analytic": self.cfr_value_error,
            },
            "safe_resolve": asdict(self.safe_resolve),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        summary = dict(self.public_replay_summary)
        lines = [
            "# Phase 6A-6B game-theoretic foundation evidence",
            "",
            f"Benchmark: `{self.benchmark_id}`  ",
            f"Evidence fingerprint: `{self.fingerprint}`  ",
            f"Status: **{self.status}**",
            "",
            "## Official replay conversion",
            "",
            f"- Source matches: **{summary['matches']}**; isolated player-perspective episodes: **{summary['episodes']}**.",
            f"- Public frames: **{summary['frames']}**; acting-observer decision frames: **{summary['decision_frames']}**.",
            f"- Hidden card identities removed from events: **{summary['redacted_events']}**.",
            f"- Supervision-only later reveals: **{summary['supervision_only_reveal_targets']}**.",
            f"- Unknown event types: **{len(summary['unknown_event_types'])}**; leakage audit validated: **{summary['validated']}**.",
            "- These frames derive from genuine official-engine replays. Conversion does not make the exact validation game official.",
            "",
            "## Exact abstract game",
            "",
            f"`{self.game_name}` has **{self.game_state_count}** states, **{self.terminal_state_count}** terminal states, "
            f"**{self.information_set_count}** information sets, and **{self.chance_deal_count}** private deals.",
            "It is explicitly abstract and Pokemon-themed only; `official_engine_executed` is `false`.",
            "",
            "| Profile | P0 value | NashConv | Exploitability |",
            "|---|---:|---:|---:|",
            f"| Uniform | {self.uniform_metrics.expected_value_player_0:+.8f} | {self.uniform_metrics.nash_conv:.8f} | {self.uniform_metrics.exploitability:.8f} |",
            f"| Analytic equilibrium | {self.analytic_metrics.expected_value_player_0:+.8f} | {self.analytic_metrics.nash_conv:.8e} | {self.analytic_metrics.exploitability:.8e} |",
            f"| CFR ({self.cfr_iterations:,}) | {self.cfr_metrics.expected_value_player_0:+.8f} | {self.cfr_metrics.nash_conv:.8f} | {self.cfr_metrics.exploitability:.8f} |",
            "",
            f"Analytic value: **{ANALYTIC_GAME_VALUE_PLAYER_0:+.8f}**; CFR absolute value error: **{self.cfr_value_error:.8e}**.",
            "",
            "## Exact microgame safe re-solve",
            "",
            f"At `{self.safe_resolve.information_set_id}`, exact enumeration changed "
            f"`{dict(self.safe_resolve.baseline_strategy)}` to `{dict(self.safe_resolve.resolved_strategy)}`.",
            f"Conditional actor value changed from **{self.safe_resolve.baseline_actor_value:+.6f}** "
            f"to **{self.safe_resolve.resolved_actor_value:+.6f}** while maximum measured opponent-bound violation was "
            f"**{self.safe_resolve.maximum_counterfactual_value_violation:.3e}**.",
            "This certificate applies only to the exactly enumerable microgame and specified conditional bounds.",
            "",
            "## Claim boundary",
            "",
            "| Claim | Status |",
            "|---|---|",
        ]
        lines.extend(f"| {claim} | {status} |" for claim, status in self.claims)
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class Phase6FoundationArtifacts:
    public_replays: PublicReplayDataset
    evidence: Phase6FoundationEvidence


def _load_phase6_evidence(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "matches" not in raw:
        raise ValueError("Phase 6 evidence JSON is invalid")
    return raw


def _public_replays(
    *,
    project_root: Path,
    phase6_evidence: Mapping[str, Any],
) -> PublicReplayDataset:
    runtime = project_root / "work" / "official_kaggle" / "sample_submission"
    engine_sha = _file_sha256(runtime / "cg" / "cg.dll")
    converter = OfficialReplayConverter()
    target_resolution = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    target_profile = rocket_compatibility_profile(target_resolution)
    profiles = {profile.label: profile for profile in OFFICIAL_PANEL_DECKS + (target_profile,)}
    records = {str(record["match_id"]): record for record in phase6_evidence["matches"]}
    episodes = []
    replay_directory = project_root / "artifacts" / "phase6_evidence" / "replays"
    for match_id in phase6_evidence["replay_review_queue"]:
        record = records[str(match_id)]
        opponent = profiles[str(record["opponent_label"])]
        agent_seat = int(record["agent_seat"])
        decks = (
            (target_profile.fingerprint, opponent.fingerprint)
            if agent_seat == 0
            else (opponent.fingerprint, target_profile.fingerprint)
        )
        replay_text = (replay_directory / f"{match_id}.json").read_text(encoding="utf-8")
        episodes.extend(
            converter.convert_match(
                replay_text,
                match_id=str(match_id),
                engine_sha256=engine_sha,
                card_pool_sha256=str(phase6_evidence["external"]["card_pool_sha256"]),
                environment_fingerprint=str(phase6_evidence["environment_fingerprint"]),
                deck_fingerprints=decks,
            )
        )
    dataset = build_public_replay_dataset(episodes)
    if not dataset.validated:
        raise RuntimeError("public replay leakage audit failed")
    if dataset.summary["unknown_event_types"]:
        raise RuntimeError("unknown official replay event types require redaction review")
    return dataset


def _conditional_actor_value(
    game: RocketPrizeDuel,
    profile: Any,
    states: tuple[Any, ...],
    belief: Any,
    *,
    actor: int,
) -> float:
    by_hypothesis = {}
    for state in states:
        opponent_card = state.cards[1 - actor]
        by_hypothesis[f"opponent:{opponent_card.label}"] = state
    return sum(
        hypothesis.weight
        * evaluate_profile_from_state(
            game,
            profile,
            by_hypothesis[hypothesis.hypothesis_id],
            player=actor,
        )
        for hypothesis in belief.opponent_range.hypotheses
    )


def run_phase6_foundation(
    *,
    project_root: str | Path | None = None,
    cfr_iterations: int = 20_000,
) -> Phase6FoundationArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    phase6_path = root / "artifacts" / "phase6_evidence" / "phase6_evaluation.json"
    phase6 = _load_phase6_evidence(phase6_path)
    public_dataset = _public_replays(project_root=root, phase6_evidence=phase6)

    game = RocketPrizeDuel()
    uniform = uniform_profile(game)
    analytic = analytic_equilibrium_profile(game)
    uniform_metrics = exact_profile_metrics(game, uniform)
    analytic_metrics = exact_profile_metrics(game, analytic)
    cfr = TabularCFRSolver(game).train(cfr_iterations)
    cfr_value_error = abs(
        cfr.metrics.expected_value_player_0 - ANALYTIC_GAME_VALUE_PLAYER_0
    )

    # A genuine bounded local re-solve on the exact microgame. It begins with
    # the deliberately weak uniform profile and may improve only if no opponent
    # private-card hypothesis receives more value than its exact baseline bound.
    state = next(
        item
        for item in game.all_states()
        if item.cards[0] is DuelCard.SPIDOPS and item.history == ()
    )
    info_set = game.information_set(state, player=0)
    belief = game.public_belief(state, player=0)
    provider = ExactTabularBlueprintProvider(
        game,
        uniform,
        deck_fingerprint=load_default_deck().fingerprint,
        algorithm="uniform_validation_baseline_for_exact_safe_resolve",
        training_steps=1,
    )
    baseline = provider.strategy(info_set, belief)
    bounds = exact_counterfactual_bounds(
        provider,
        info_set,
        belief,
        opponent_player=1,
        half_width=0.0,
    )
    request = SafeResolveRequest(
        information_set=info_set,
        belief=belief,
        baseline_strategy=baseline,
        opponent_value_bounds=bounds,
        budget=ResolveBudget(
            iterations=101,
            max_nodes=1_000,
            time_budget_ms=1_000.0,
            safety_tolerance=1e-12,
        ),
    )
    resolved = ExactGridSafeResolver(uniform, grid_steps=100).resolve(game, request)
    resolved_profile = uniform.with_strategy(
        info_set.info_set_id,
        resolved.strategy,
        label="exact_safe_resolved_uniform_profile",
    )
    states = game.states_for_information_set(info_set.info_set_id)
    baseline_actor_value = _conditional_actor_value(
        game, uniform, states, belief, actor=0
    )
    resolved_actor_value = _conditional_actor_value(
        game, resolved_profile, states, belief, actor=0
    )
    safe_evidence = SafeResolveEvidence(
        information_set_id=info_set.info_set_id,
        baseline_strategy=baseline.probabilities,
        resolved_strategy=resolved.strategy.probabilities,
        baseline_actor_value=baseline_actor_value,
        resolved_actor_value=resolved_actor_value,
        actor_value_gain=resolved_actor_value - baseline_actor_value,
        maximum_counterfactual_value_violation=(
            resolved.certificate.max_counterfactual_value_violation
        ),
        bound_confidence=resolved.certificate.confidence,
        safe=resolved.certificate.safe,
        candidate_strategies=resolved.iterations,
        expanded_hypothesis_nodes=resolved.expanded_nodes,
        method=resolved.certificate.method,
    )

    analytic_valid = (
        abs(analytic_metrics.expected_value_player_0 - ANALYTIC_GAME_VALUE_PLAYER_0)
        <= 1e-12
        and analytic_metrics.nash_conv <= 1e-12
    )
    cfr_valid = (
        cfr.metrics.nash_conv <= 0.005
        and cfr_value_error <= 1e-4
        and cfr.metrics.nash_conv < uniform_metrics.nash_conv * 0.02
    )
    resolve_valid = (
        safe_evidence.safe
        and safe_evidence.maximum_counterfactual_value_violation <= 1e-12
        and safe_evidence.actor_value_gain > 0
    )
    claims = (
        ("official_truth_replay_source", "validated_source_from_native_engine"),
        ("public_replay_conversion", "validated_allowlist_and_roundtrip"),
        ("known_hidden_state_leakage", "zero_detected_by_structural_audit"),
        ("range_supervision_labels", "validated_available_after_reveal_only"),
        ("exact_validation_game_boundary", "validated_abstract_non_official"),
        (
            "analytic_equilibrium_reference",
            "validated" if analytic_valid else "failed",
        ),
        ("tabular_cfr_reference", "validated" if cfr_valid else "failed"),
        (
            "exact_microgame_safe_resolve",
            "validated_microgame_only" if resolve_valid else "failed",
        ),
        (
            "neural_game_theoretic_blueprint",
            "trained_exact_microgame_only_full_game_not_trained",
        ),
        (
            "official_information_set_bridge",
            "one_step_shadow_validated_full_game_not_implemented",
        ),
        ("official_full_game_safe_resolve", "not_implemented"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6FoundationEvidence(
        benchmark_id=f"phase6ab-public-replay-exact-game-cfr-{cfr_iterations}",
        source_phase6_evidence_fingerprint=str(phase6["fingerprint"]),
        public_replay_dataset_fingerprint=public_dataset.fingerprint,
        public_replay_summary=tuple(sorted(public_dataset.summary.items())),
        game_name=game.provenance.name,
        game_fingerprint=game.fingerprint,
        game_boundary_kind=game.provenance.kind.value,
        official_engine_executed_by_game=game.provenance.official_engine_executed,
        game_state_count=len(game.all_states()),
        terminal_state_count=terminal_state_count(game),
        information_set_count=len(game.information_sets()),
        information_set_action_count=information_set_action_count(game),
        chance_deal_count=deal_count(game),
        analytic_metrics=analytic_metrics,
        uniform_metrics=uniform_metrics,
        cfr_iterations=cfr.iterations,
        cfr_solver_fingerprint=cfr.solver_fingerprint,
        cfr_metrics=cfr.metrics,
        cfr_value_error=cfr_value_error,
        safe_resolve=safe_evidence,
        claims=claims,
    )
    if evidence.status != "phase6ab_foundation_validated":
        raise RuntimeError("Phase 6A-6B foundation failed a declared gate")
    return Phase6FoundationArtifacts(public_replays=public_dataset, evidence=evidence)


def main() -> None:
    print(run_phase6_foundation().evidence.to_json())


if __name__ == "__main__":
    main()
