"""Official-engine Phase 6 integration-baseline benchmark and replay triage."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .competition import (
    AblationMode,
    CompetitionAgentConfig,
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    Phase6CompetitionAgent,
    resolve_official_deck,
)
from .competition_search import OfficialCgSearchAdvisor
from .deck import load_default_deck
from .evaluation import ExternalEvidence, OfficialMatchRecord, Phase6EvaluationReport
from .official_evaluation import (
    OFFICIAL_PANEL_DECKS,
    KaggleDeckProfile,
    OfficialMatchRunner,
    PriorityBaseline,
    rocket_compatibility_profile,
)
from .replay_review import ReplayReview, review_official_replay


PHASE6_SEED = 20260726
BENCHMARK_DECISION_BUDGET_MS = 250.0
BENCHMARK_SEARCH_BUDGET_MS = 50.0
BENCHMARK_CRITICALITY_THRESHOLD = 0.45


@dataclass(frozen=True, slots=True)
class Phase6BenchmarkArtifacts:
    report: Phase6EvaluationReport
    reviews: tuple[ReplayReview, ...]
    replay_data: tuple[tuple[str, str], ...]
    deck_validation: tuple[tuple[str, bool, int, int], ...]

    def summary(self) -> dict[str, Any]:
        exact = resolve_official_deck()
        proxy = resolve_official_deck(substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION)
        claims = dict(self.report.claims)
        return {
            "status": (
                "ok" if all(value == "validated" for value in claims.values()) else "external_blocked"
            ),
            "software_status": "ok"
            if claims["official_engine_execution"] == "validated"
            and claims["legal_action_safety"] == "validated"
            else "failed",
            "external_status": {
                "exact_deck": claims["exact_naic_deck_on_current_kaggle_pool"],
                "leaderboard": claims["new_kaggle_leaderboard_movement"],
                "expert_review": claims["human_expert_replay_review"],
                "superhuman_strength": claims["superhuman_strength"],
            },
            "planning_status": {
                "heuristic_and_sampled_rollout": "integration_baseline_not_final",
                "neural_blueprint": claims["neural_game_theoretic_blueprint"],
                "safe_resolving": claims["safe_game_theoretic_resolving"],
                "official_bridge": claims["official_game_theoretic_bridge"],
            },
            "deck": {
                "exact_resolution": exact.to_dict(),
                "compatibility_resolution": proxy.to_dict(),
                "validation": [
                    {
                        "profile": label,
                        "accepted": accepted,
                        "error_player": error_player,
                        "error_type": error_type,
                    }
                    for label, accepted, error_player, error_type in self.deck_validation
                ],
            },
            "evaluation": self.report.to_dict(include_matches=False),
            "replay_reviews": [review.to_dict() for review in self.reviews],
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_official_runtime() -> Path:
    return _project_root() / "work" / "official_kaggle" / "sample_submission"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _card_pool_metadata(runtime: Path) -> tuple[Path, int, str]:
    csv_path = runtime.parent / "EN_Card_Data.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"official card CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        unique_ids = {int(row["Card ID"]) for row in csv.DictReader(handle)}
    return csv_path, len(unique_ids), _file_sha256(csv_path)


def _environment_fingerprint(
    runtime: Path,
    panel: tuple[KaggleDeckProfile, ...],
    repetitions: int,
) -> str:
    payload = {
        "schema": 1,
        "native_library_sha256": _file_sha256(runtime / "cg" / "cg.dll"),
        "panel": [profile.fingerprint for profile in panel],
        "styles": ["tempo", "attack_rush", "control"],
        "ablations": [mode.value for mode in AblationMode],
        "repetitions": repetitions,
        "benchmark_seed": PHASE6_SEED,
        "selection_cap": 2000,
        "decision_budget_ms": BENCHMARK_DECISION_BUDGET_MS,
        "search_budget_ms": BENCHMARK_SEARCH_BUDGET_MS,
        "criticality_threshold": BENCHMARK_CRITICALITY_THRESHOLD,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_phase6_benchmark(
    *,
    runtime_directory: str | Path | None = None,
    repetitions: int = 1,
    max_review_replays: int = 6,
) -> Phase6BenchmarkArtifacts:
    """Run the full official-engine panel and return serializable evidence."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if max_review_replays <= 0:
        raise ValueError("max_review_replays must be positive")
    runtime = Path(runtime_directory or default_official_runtime()).resolve()
    runner = OfficialMatchRunner(runtime)
    exact_deck = load_default_deck()
    exact_resolution = resolve_official_deck()
    compatibility = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    target_profile = rocket_compatibility_profile(compatibility)
    panel = OFFICIAL_PANEL_DECKS + (target_profile,)
    validation = tuple(
        (profile.label, *runner.validate_deck(profile)) for profile in panel
    )
    if not all(accepted for _, accepted, _, _ in validation):
        rejected = [label for label, accepted, _, _ in validation if not accepted]
        raise RuntimeError(f"official engine rejected panel decks: {rejected}")

    records: list[OfficialMatchRecord] = []
    replay_candidates: list[tuple[OfficialMatchRecord, ReplayReview, str]] = []
    styles = ("tempo", "attack_rush", "control")
    match_number = 0
    for repetition in range(repetitions):
        for mode in AblationMode:
            for opponent_profile in panel:
                for style in styles:
                    for target_seat in (0, 1):
                        match_number += 1
                        advisor = (
                            OfficialCgSearchAdvisor(
                                compatibility,
                                rollout_steps=5,
                                value_scale=0.40,
                            )
                            if mode is AblationMode.FULL
                            else None
                        )
                        agent = Phase6CompetitionAgent(
                            compatibility,
                            config=CompetitionAgentConfig(
                                ablation=mode,
                                decision_budget_ms=BENCHMARK_DECISION_BUDGET_MS,
                                search_budget_ms=BENCHMARK_SEARCH_BUDGET_MS,
                                criticality_threshold=BENCHMARK_CRITICALITY_THRESHOLD,
                            ),
                            search_advisor=advisor,
                        )
                        opponent = PriorityBaseline(
                            opponent_profile.card_ids,
                            style=style,
                            reverse=bool((PHASE6_SEED + match_number + repetition) % 2),
                        )
                        match_id = (
                            f"phase6-r{repetition}-{mode.value}-"
                            f"{opponent_profile.archetype}-{style}-s{target_seat}"
                        )
                        capture = mode is AblationMode.FULL
                        execution = runner.run_match(
                            match_id=match_id,
                            target_agent=agent,
                            target_deck=target_profile,
                            opponent_agent=opponent,
                            opponent_deck=opponent_profile,
                            opponent_style=style,
                            target_seat=target_seat,
                            capture_replay=capture,
                        )
                        records.append(execution.record)
                        if capture:
                            review = review_official_replay(
                                replay_data=execution.replay_data,
                                record=execution.record,
                                telemetry=tuple(agent.memory.telemetry),
                            )
                            replay_candidates.append(
                                (execution.record, review, execution.replay_data)
                            )

    replay_candidates.sort(
        key=lambda item: (
            item[0].score,
            item[0].opponent_prizes_remaining - item[0].own_prizes_remaining,
            -item[0].belief_nll,
            -item[0].search_changes,
            item[0].match_id,
        )
    )
    selected_replays = tuple(replay_candidates[:max_review_replays])
    csv_path, unique_ids, card_pool_hash = _card_pool_metadata(runtime)
    environment_fingerprint = _environment_fingerprint(runtime, panel, repetitions)
    external = ExternalEvidence(
        simulator_source="Official pokemon-tcg-ai-battle competition download",
        simulator_file_date="2026-07-15",
        card_pool_unique_ids=unique_ids,
        card_pool_sha256=card_pool_hash,
        exact_deck_playable=exact_resolution.playable,
        incompatibility=(
            "Prism Tower (CRI 80) is in the exact NAIC 2026 list but absent from "
            "the current 1,267-ID TEF-POR Kaggle engine."
        ),
        prior_related_submission="HelbrosRocketSpidopsProV3.tar.gz",
        prior_related_public_score=571.8,
        submission_status="blocked_exact_deck_card_pool",
        leaderboard_observed_at="2026-07-26",
    )
    report = Phase6EvaluationReport.build(
        benchmark_id=f"official-cg-phase6-r{repetitions}",
        benchmark_seed=PHASE6_SEED,
        deck_fingerprint=exact_deck.fingerprint,
        environment_fingerprint=environment_fingerprint,
        matches=records,
        external=external,
        replay_review_queue=(item[0].match_id for item in selected_replays),
        failure_modes=(
            "The exact NAIC list cannot enter the current Kaggle engine because Prism Tower is absent.",
            "The local Watchtower profile is an explicit compatibility substitution, not the requested exact deck.",
            "Four non-sample opponent decks are legality probes, not tournament-optimized expert agents.",
            "The official battle API exposes no match-seed control, so engine shuffles cannot be paired exactly across ablations.",
            "The submission search prior defaults to Abomasnow and can be misspecified against other archetypes.",
            "The heuristic policy and sampled rollout advisor are integration baselines, not the final strategic architecture.",
            "The neural blueprint, safe resolver, and official information-set bridge are interfaces only and are not trained or deployed.",
            "Automated replay triage is complete; human expert review is still external and pending.",
            "No new Kaggle submission was made because doing so would require changing the exact deck.",
        ),
    )
    return Phase6BenchmarkArtifacts(
        report=report,
        reviews=tuple(item[1] for item in selected_replays),
        replay_data=tuple((item[0].match_id, item[2]) for item in selected_replays),
        deck_validation=validation,
    )


def run_phase6_demo(
    *,
    runtime_directory: str | Path | None = None,
    repetitions: int = 1,
) -> dict[str, Any]:
    return run_phase6_benchmark(
        runtime_directory=runtime_directory,
        repetitions=repetitions,
    ).summary()


def main() -> None:
    print(json.dumps(run_phase6_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
