"""Phase 6C exact-game neural blueprint comparison and checkpoint evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .deck import load_default_deck
from .exact_game import (
    RocketPrizeDuel,
    TabularCFRSolver,
    analytic_equilibrium_profile,
    exact_profile_metrics,
    uniform_profile,
)
from .neural_blueprint import (
    NeuralBlueprintArtifact,
    NeuralBlueprintTrainingReport,
    NeuralBlueprintTrainingRun,
    provider_profile,
    regret_matching_targets_from_profile,
    restore_neural_duel_blueprint,
    train_neural_duel_blueprint,
)


PHASE6_NEURAL_SCHEMA_VERSION = 1
PHASE6_NEURAL_SEED = 20260726


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointRoundtripEvidence:
    strategy_mode: str
    artifact_fingerprint: str
    maximum_strategy_probability_delta: float
    maximum_counterfactual_value_delta: float
    exact_metrics_nash_conv_delta: float

    def __post_init__(self) -> None:
        if self.strategy_mode not in {"policy", "regret"}:
            raise ValueError("checkpoint roundtrip mode is invalid")
        if len(self.artifact_fingerprint) != 64:
            raise ValueError("checkpoint artifact fingerprint is invalid")
        for field_name in (
            "maximum_strategy_probability_delta",
            "maximum_counterfactual_value_delta",
            "exact_metrics_nash_conv_delta",
        ):
            value = getattr(self, field_name)
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Phase6NeuralEvidence:
    benchmark_id: str
    foundation_evidence_fingerprint: str
    public_replay_dataset_fingerprint: str
    game_fingerprint: str
    deck_fingerprint: str
    cfr_iterations: int
    seed: int
    policy_candidate: NeuralBlueprintTrainingReport
    regret_candidate: NeuralBlueprintTrainingReport
    policy_roundtrip: CheckpointRoundtripEvidence
    regret_roundtrip: CheckpointRoundtripEvidence
    selected_strategy_mode: str
    selected_artifact_fingerprint: str
    selected_nash_conv: float
    selected_exploitability: float
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or self.selected_strategy_mode not in {"policy", "regret"}:
            raise ValueError("Phase 6C evidence identity/selection is invalid")
        for field_name in (
            "foundation_evidence_fingerprint",
            "public_replay_dataset_fingerprint",
            "game_fingerprint",
            "deck_fingerprint",
            "selected_artifact_fingerprint",
        ):
            if len(getattr(self, field_name)) != 64:
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        if self.cfr_iterations <= 0 or self.selected_nash_conv < 0 or self.selected_exploitability < 0:
            raise ValueError("Phase 6C counts/metrics are invalid")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def status(self) -> str:
        claims = dict(self.claims)
        required = (
            "public_belief_policy_candidate",
            "regret_matching_candidate",
            "selected_checkpoint_roundtrip",
            "selected_exact_game_exploitability",
        )
        return (
            "phase6c_neural_microgame_blueprint_validated"
            if all(claims[name].startswith("validated") for name in required)
            else "failed"
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_NEURAL_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "foundation_evidence_fingerprint": self.foundation_evidence_fingerprint,
            "public_replay_dataset_fingerprint": self.public_replay_dataset_fingerprint,
            "game_fingerprint": self.game_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "cfr_iterations": self.cfr_iterations,
            "seed": self.seed,
            "candidates": {
                "policy": asdict(self.policy_candidate),
                "regret": asdict(self.regret_candidate),
            },
            "roundtrip": {
                "policy": asdict(self.policy_roundtrip),
                "regret": asdict(self.regret_roundtrip),
            },
            "selected": {
                "strategy_mode": self.selected_strategy_mode,
                "artifact_fingerprint": self.selected_artifact_fingerprint,
                "nash_conv": self.selected_nash_conv,
                "exploitability": self.selected_exploitability,
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines = [
            "# Phase 6C neural microgame blueprint evidence",
            "",
            f"Benchmark: `{self.benchmark_id}`  ",
            f"Evidence fingerprint: `{self.fingerprint}`  ",
            f"Status: **{self.status}**",
            "",
            "Both candidates consume the same public-belief/information-set schema and are evaluated by exact pure best responses over the complete non-official microgame.",
            "",
            "| Candidate | Training target | Epochs | Policy L1 | Value RMSE | NashConv | Exploitability |",
            "|---|---|---:|---:|---:|---:|---:|",
            (
                f"| Policy | Analytic equilibrium distribution | {self.policy_candidate.epochs:,} | "
                f"{self.policy_candidate.policy_l1_to_teacher:.8f} | {self.policy_candidate.conditional_value_rmse:.8f} | "
                f"{self.policy_candidate.metrics.nash_conv:.8f} | {self.policy_candidate.metrics.exploitability:.8f} |"
            ),
            (
                f"| Regret matching | Tabular CFR average strategy | {self.regret_candidate.epochs:,} | "
                f"{self.regret_candidate.policy_l1_to_teacher:.8f} | {self.regret_candidate.conditional_value_rmse:.8f} | "
                f"{self.regret_candidate.metrics.nash_conv:.8f} | {self.regret_candidate.metrics.exploitability:.8f} |"
            ),
            "",
            f"Selected: **{self.selected_strategy_mode}**; artifact `{self.selected_artifact_fingerprint}`.",
            "",
            "The policy candidate is selected by exact NashConv. The regret-matching candidate validates a distinct neural inference parameterization; its trainer is supervised from tabular CFR and is not represented as Deep CFR.",
            "",
            "## Checkpoint roundtrip",
            "",
            "| Candidate | Max strategy delta | Max CF-value delta | NashConv delta |",
            "|---|---:|---:|---:|",
            f"| Policy | {self.policy_roundtrip.maximum_strategy_probability_delta:.3e} | {self.policy_roundtrip.maximum_counterfactual_value_delta:.3e} | {self.policy_roundtrip.exact_metrics_nash_conv_delta:.3e} |",
            f"| Regret | {self.regret_roundtrip.maximum_strategy_probability_delta:.3e} | {self.regret_roundtrip.maximum_counterfactual_value_delta:.3e} | {self.regret_roundtrip.exact_metrics_nash_conv_delta:.3e} |",
            "",
            "## Claim boundary",
            "",
            "| Claim | Status |",
            "|---|---|",
        ]
        lines.extend(f"| {name} | {status} |" for name, status in self.claims)
        return "\n".join(lines) + "\n"


@dataclass(slots=True)
class Phase6NeuralArtifacts:
    evidence: Phase6NeuralEvidence
    policy_run: NeuralBlueprintTrainingRun
    regret_run: NeuralBlueprintTrainingRun

    @property
    def selected_artifact(self) -> NeuralBlueprintArtifact:
        return (
            self.policy_run.artifact
            if self.evidence.selected_strategy_mode == "policy"
            else self.regret_run.artifact
        )


def _roundtrip(
    game: RocketPrizeDuel,
    run: NeuralBlueprintTrainingRun,
) -> CheckpointRoundtripEvidence:
    encoded = run.artifact.to_dict()
    artifact = NeuralBlueprintArtifact.from_dict(json.loads(json.dumps(encoded)))
    restored = restore_neural_duel_blueprint(game, artifact)
    restored_profile = provider_profile(
        game,
        restored,
        label=run.profile.label,
    )
    maximum_strategy_delta = 0.0
    maximum_value_delta = 0.0
    for information_set in game.information_sets():
        state = game.states_for_information_set(information_set.info_set_id)[0]
        actor = state.acting_player
        assert actor is not None
        belief = game.public_belief(state, player=actor)
        original_strategy = run.provider.strategy(information_set, belief)
        restored_strategy = restored.strategy(information_set, belief)
        maximum_strategy_delta = max(
            maximum_strategy_delta,
            *(
                abs(original_strategy.probability(action.action_id) - restored_strategy.probability(action.action_id))
                for action in information_set.legal_actions
            ),
        )
        original_values = dict(
            run.provider.counterfactual_values(
                information_set, belief, player=actor
            ).values
        )
        restored_values = dict(
            restored.counterfactual_values(
                information_set, belief, player=actor
            ).values
        )
        maximum_value_delta = max(
            maximum_value_delta,
            *(abs(original_values[key] - value) for key, value in restored_values.items()),
        )
    restored_metrics = exact_profile_metrics(game, restored_profile)
    return CheckpointRoundtripEvidence(
        strategy_mode=run.report.strategy_mode,
        artifact_fingerprint=artifact.fingerprint,
        maximum_strategy_probability_delta=maximum_strategy_delta,
        maximum_counterfactual_value_delta=maximum_value_delta,
        exact_metrics_nash_conv_delta=abs(
            restored_metrics.nash_conv - run.report.metrics.nash_conv
        ),
    )


def _load_foundation(path: Path) -> Mapping[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "phase6ab_foundation_validated":
        raise ValueError("Phase 6C requires validated Phase 6A-6B evidence")
    return result


def run_phase6_neural_blueprint(
    *,
    project_root: str | Path | None = None,
    cfr_iterations: int = 20_000,
    policy_epochs: int = 2_000,
    regret_epochs: int = 4_000,
    seed: int = PHASE6_NEURAL_SEED,
) -> Phase6NeuralArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    foundation = _load_foundation(
        root
        / "artifacts"
        / "phase6_game_theory_foundation"
        / "phase6ab_evidence.json"
    )
    game = RocketPrizeDuel()
    if foundation["game"]["fingerprint"] != game.fingerprint:
        raise ValueError("Phase 6C game fingerprint does not match foundation")
    deck_fingerprint = load_default_deck().fingerprint
    solver = TabularCFRSolver(game)
    cfr = solver.train(cfr_iterations)
    analytic = analytic_equilibrium_profile(game)
    policy_run = train_neural_duel_blueprint(
        game,
        analytic,
        regret_matching_targets_from_profile(game, analytic),
        deck_fingerprint=deck_fingerprint,
        strategy_mode="policy",
        seed=seed,
        epochs=policy_epochs,
        hidden_dim=32,
        learning_rate=0.03,
    )
    regret_run = train_neural_duel_blueprint(
        game,
        cfr.profile,
        regret_matching_targets_from_profile(game, cfr.profile),
        deck_fingerprint=deck_fingerprint,
        strategy_mode="regret",
        seed=seed,
        epochs=regret_epochs,
        hidden_dim=32,
        learning_rate=0.03,
    )
    policy_roundtrip = _roundtrip(game, policy_run)
    regret_roundtrip = _roundtrip(game, regret_run)
    selected = min(
        (policy_run, regret_run),
        key=lambda run: (
            run.report.metrics.nash_conv,
            run.report.conditional_value_rmse,
            run.report.strategy_mode,
        ),
    )
    uniform_nash = exact_profile_metrics(game, uniform_profile(game)).nash_conv
    policy_valid = (
        policy_run.report.metrics.nash_conv < 0.001
        and policy_run.report.metrics.nash_conv < uniform_nash * 0.01
        and policy_run.report.conditional_value_rmse < 0.001
    )
    regret_valid = (
        regret_run.report.metrics.nash_conv < 0.005
        and regret_run.report.metrics.nash_conv < uniform_nash * 0.01
        and regret_run.report.conditional_value_rmse < 0.001
    )
    roundtrip_valid = all(
        evidence.maximum_strategy_probability_delta == 0.0
        and evidence.maximum_counterfactual_value_delta == 0.0
        and evidence.exact_metrics_nash_conv_delta == 0.0
        for evidence in (policy_roundtrip, regret_roundtrip)
    )
    selected_valid = selected.report.metrics.nash_conv < 0.001
    claims = (
        (
            "public_belief_policy_candidate",
            "validated_exact_microgame" if policy_valid else "failed",
        ),
        (
            "regret_matching_candidate",
            "validated_exact_microgame" if regret_valid else "failed",
        ),
        (
            "selected_checkpoint_roundtrip",
            "validated_exact" if roundtrip_valid else "failed",
        ),
        (
            "selected_exact_game_exploitability",
            "validated_below_0.001_nash_conv" if selected_valid else "failed",
        ),
        ("neural_game_theoretic_blueprint", "validated_exact_microgame_only"),
        ("full_game_neural_blueprint", "not_trained"),
        (
            "official_information_set_bridge",
            "one_step_shadow_validated_full_game_not_implemented",
        ),
        ("official_full_game_safe_resolve", "not_implemented"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6NeuralEvidence(
        benchmark_id=(
            f"phase6c-neural-duel-cfr{cfr_iterations}-"
            f"policy{policy_epochs}-regret{regret_epochs}"
        ),
        foundation_evidence_fingerprint=str(foundation["fingerprint"]),
        public_replay_dataset_fingerprint=str(
            foundation["public_replay_dataset_fingerprint"]
        ),
        game_fingerprint=game.fingerprint,
        deck_fingerprint=deck_fingerprint,
        cfr_iterations=cfr_iterations,
        seed=seed,
        policy_candidate=policy_run.report,
        regret_candidate=regret_run.report,
        policy_roundtrip=policy_roundtrip,
        regret_roundtrip=regret_roundtrip,
        selected_strategy_mode=selected.report.strategy_mode,
        selected_artifact_fingerprint=selected.artifact.fingerprint,
        selected_nash_conv=selected.report.metrics.nash_conv,
        selected_exploitability=selected.report.metrics.exploitability,
        claims=claims,
    )
    if evidence.status != "phase6c_neural_microgame_blueprint_validated":
        raise RuntimeError("Phase 6C neural microgame blueprint failed a declared gate")
    return Phase6NeuralArtifacts(
        evidence=evidence,
        policy_run=policy_run,
        regret_run=regret_run,
    )


def main() -> None:
    print(run_phase6_neural_blueprint().evidence.to_json())


if __name__ == "__main__":
    main()
