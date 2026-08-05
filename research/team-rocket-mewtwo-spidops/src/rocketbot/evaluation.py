"""Reproducible multidimensional Phase 6 evaluation records and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from statistics import fmean
from typing import Any, Iterable, Mapping

from .competition import STRATEGIC_AXIS_NAMES


EVALUATION_SCHEMA_VERSION = 1


def _finite(value: float, field_name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    materialized = tuple(values)
    return fmean(materialized) if materialized else default


def wilson_interval(successes: float, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval, allowing draws to contribute half a success."""

    if trials <= 0:
        return (0.0, 1.0)
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


@dataclass(frozen=True, slots=True)
class OfficialMatchRecord:
    """One target-agent match through the competition's native engine."""

    match_id: str
    agent_label: str
    ablation: str
    opponent_label: str
    opponent_archetype: str
    opponent_style: str
    agent_seat: int
    winner: int
    turns: int
    selections: int
    decision_count: int
    runner_fallback_count: int
    illegal_action_count: int
    soft_timeout_count: int
    mean_latency_ms: float
    max_latency_ms: float
    action_counts: tuple[tuple[str, int], ...]
    lethal_opportunities: int
    lethal_conversions: int
    first_attack_turn: int | None
    first_prize_turn: int | None
    own_prizes_remaining: int
    opponent_prizes_remaining: int
    mean_own_board: float
    mean_opponent_board: float
    mean_own_hp: float
    mean_opponent_hp: float
    mean_own_hand: float
    mean_option_count: float
    strategic_values: tuple[float, ...]
    strategic_weights: tuple[float, ...]
    strategic_aggregate: float
    belief_nll: float
    belief_brier: float
    search_invocations: int
    search_changes: int
    mean_search_bonus: float
    engine_error: str | None = None

    def __post_init__(self) -> None:
        if not self.match_id or not self.agent_label or not self.opponent_label:
            raise ValueError("match identity fields must be non-empty")
        if self.agent_seat not in (0, 1):
            raise ValueError("agent_seat must be 0 or 1")
        if self.winner not in (0, 1, 2, -1):
            raise ValueError("winner must be player 0, player 1, draw, or unfinished")
        for field_name in (
            "turns",
            "selections",
            "decision_count",
            "runner_fallback_count",
            "illegal_action_count",
            "soft_timeout_count",
            "lethal_opportunities",
            "lethal_conversions",
            "own_prizes_remaining",
            "opponent_prizes_remaining",
            "search_invocations",
            "search_changes",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.lethal_conversions > self.lethal_opportunities:
            raise ValueError("lethal conversions cannot exceed opportunities")
        if self.search_changes > self.search_invocations:
            raise ValueError("search changes cannot exceed invocations")
        if len(self.strategic_values) != len(STRATEGIC_AXIS_NAMES):
            raise ValueError("match record requires six strategic values")
        if len(self.strategic_weights) != len(STRATEGIC_AXIS_NAMES):
            raise ValueError("match record requires six strategic weights")
        if not math.isclose(sum(self.strategic_weights), 1.0, abs_tol=1e-6):
            raise ValueError("strategic weights must sum to one")
        for field_name in (
            "mean_latency_ms",
            "max_latency_ms",
            "mean_own_board",
            "mean_opponent_board",
            "mean_own_hp",
            "mean_opponent_hp",
            "mean_own_hand",
            "mean_option_count",
            "strategic_aggregate",
            "belief_nll",
            "belief_brier",
            "mean_search_bonus",
        ):
            _finite(getattr(self, field_name), field_name)

    @property
    def score(self) -> float:
        if self.winner == self.agent_seat:
            return 1.0
        if self.winner == 2:
            return 0.5
        return 0.0

    @property
    def won(self) -> bool:
        return self.winner == self.agent_seat

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action_counts"] = dict(self.action_counts)
        result["strategic_values"] = dict(
            zip(STRATEGIC_AXIS_NAMES, self.strategic_values, strict=True)
        )
        result["strategic_weights"] = dict(
            zip(STRATEGIC_AXIS_NAMES, self.strategic_weights, strict=True)
        )
        result["score"] = self.score
        return result


@dataclass(frozen=True, slots=True)
class EvaluationSlice:
    label: str
    matches: int
    wins: int
    draws: int
    score_rate: float
    confidence_low: float
    confidence_high: float
    mean_prize_margin: float
    illegal_actions: int
    soft_timeouts: int

    @classmethod
    def from_matches(
        cls, label: str, matches: Iterable[OfficialMatchRecord]
    ) -> "EvaluationSlice":
        records = tuple(matches)
        scores = sum(record.score for record in records)
        low, high = wilson_interval(scores, len(records))
        return cls(
            label=label,
            matches=len(records),
            wins=sum(record.won for record in records),
            draws=sum(record.winner == 2 for record in records),
            score_rate=scores / len(records) if records else 0.0,
            confidence_low=low,
            confidence_high=high,
            mean_prize_margin=_mean(
                record.opponent_prizes_remaining - record.own_prizes_remaining
                for record in records
            ),
            illegal_actions=sum(record.illegal_action_count for record in records),
            soft_timeouts=sum(record.soft_timeout_count for record in records),
        )


@dataclass(frozen=True, slots=True)
class TacticalEvaluation:
    decisions: int
    illegal_action_rate: float
    fallback_rate: float
    soft_timeout_rate: float
    lethal_conversion_rate: float
    mean_latency_ms: float
    max_latency_ms: float
    action_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class StrategicEvaluation:
    axis_means: tuple[tuple[str, float], ...]
    aggregate_mean: float
    mean_prize_margin: float
    mean_first_attack_turn: float
    mean_first_prize_turn: float
    mean_board_advantage: float
    mean_resource_hand: float
    mean_optionality: float


@dataclass(frozen=True, slots=True)
class BeliefEvaluation:
    mean_nll: float
    mean_brier: float
    evaluated_matches: int


@dataclass(frozen=True, slots=True)
class SearchEvaluation:
    invocations: int
    changed_actions: int
    change_rate: float
    mean_advice_bonus: float


@dataclass(frozen=True, slots=True)
class ExternalEvidence:
    simulator_source: str
    simulator_file_date: str
    card_pool_unique_ids: int
    card_pool_sha256: str
    exact_deck_playable: bool
    incompatibility: str | None
    prior_related_submission: str | None
    prior_related_public_score: float | None
    submission_status: str
    leaderboard_observed_at: str


@dataclass(frozen=True, slots=True)
class Phase6EvaluationReport:
    """Complete evidence bundle, including explicit claims that remain blocked."""

    schema_version: int
    benchmark_id: str
    benchmark_seed: int
    deck_fingerprint: str
    environment_fingerprint: str
    matches: tuple[OfficialMatchRecord, ...]
    tactical: TacticalEvaluation
    strategic: StrategicEvaluation
    belief: BeliefEvaluation
    search: SearchEvaluation
    by_ablation: tuple[EvaluationSlice, ...]
    by_archetype: tuple[EvaluationSlice, ...]
    by_style: tuple[EvaluationSlice, ...]
    external: ExternalEvidence
    replay_review_queue: tuple[str, ...]
    failure_modes: tuple[str, ...]
    claims: tuple[tuple[str, str], ...]

    @classmethod
    def build(
        cls,
        *,
        benchmark_id: str,
        benchmark_seed: int,
        deck_fingerprint: str,
        environment_fingerprint: str,
        matches: Iterable[OfficialMatchRecord],
        external: ExternalEvidence,
        replay_review_queue: Iterable[str] = (),
        failure_modes: Iterable[str] = (),
    ) -> "Phase6EvaluationReport":
        records = tuple(matches)
        if not records:
            raise ValueError("Phase 6 report requires at least one official match")
        decisions = sum(record.decision_count for record in records)
        action_counts: dict[str, int] = {}
        for record in records:
            for name, count in record.action_counts:
                action_counts[name] = action_counts.get(name, 0) + count
        lethal_opportunities = sum(record.lethal_opportunities for record in records)
        tactical = TacticalEvaluation(
            decisions=decisions,
            illegal_action_rate=(
                sum(record.illegal_action_count for record in records) / max(1, decisions)
            ),
            fallback_rate=(
                sum(record.runner_fallback_count for record in records) / max(1, decisions)
            ),
            soft_timeout_rate=(
                sum(record.soft_timeout_count for record in records) / max(1, decisions)
            ),
            lethal_conversion_rate=(
                sum(record.lethal_conversions for record in records)
                / max(1, lethal_opportunities)
            ),
            mean_latency_ms=_mean(record.mean_latency_ms for record in records),
            max_latency_ms=max(record.max_latency_ms for record in records),
            action_counts=tuple(sorted(action_counts.items())),
        )
        axis_means = tuple(
            (
                axis,
                _mean(record.strategic_values[index] for record in records),
            )
            for index, axis in enumerate(STRATEGIC_AXIS_NAMES)
        )
        attack_turns = tuple(
            record.first_attack_turn
            for record in records
            if record.first_attack_turn is not None
        )
        prize_turns = tuple(
            record.first_prize_turn
            for record in records
            if record.first_prize_turn is not None
        )
        strategic = StrategicEvaluation(
            axis_means=axis_means,
            aggregate_mean=_mean(record.strategic_aggregate for record in records),
            mean_prize_margin=_mean(
                record.opponent_prizes_remaining - record.own_prizes_remaining
                for record in records
            ),
            mean_first_attack_turn=_mean(attack_turns, default=-1.0),
            mean_first_prize_turn=_mean(prize_turns, default=-1.0),
            mean_board_advantage=_mean(
                record.mean_own_board - record.mean_opponent_board for record in records
            ),
            mean_resource_hand=_mean(record.mean_own_hand for record in records),
            mean_optionality=_mean(record.mean_option_count for record in records),
        )
        belief = BeliefEvaluation(
            mean_nll=_mean(record.belief_nll for record in records),
            mean_brier=_mean(record.belief_brier for record in records),
            evaluated_matches=len(records),
        )
        search_invocations = sum(record.search_invocations for record in records)
        search_changes = sum(record.search_changes for record in records)
        search = SearchEvaluation(
            invocations=search_invocations,
            changed_actions=search_changes,
            change_rate=search_changes / max(1, search_invocations),
            mean_advice_bonus=_mean(
                record.mean_search_bonus
                for record in records
                if record.search_invocations > 0
            ),
        )

        def grouped(field_name: str) -> tuple[EvaluationSlice, ...]:
            labels = sorted({str(getattr(record, field_name)) for record in records})
            return tuple(
                EvaluationSlice.from_matches(
                    label,
                    (record for record in records if str(getattr(record, field_name)) == label),
                )
                for label in labels
            )

        claims = (
            (
                "official_engine_execution",
                "validated" if all(record.engine_error is None for record in records) else "failed",
            ),
            (
                "legal_action_safety",
                "validated"
                if sum(record.illegal_action_count for record in records) == 0
                else "failed",
            ),
            (
                "exact_naic_deck_on_current_kaggle_pool",
                "validated" if external.exact_deck_playable else "blocked",
            ),
            (
                "new_kaggle_leaderboard_movement",
                "validated" if external.submission_status == "scored" else "blocked",
            ),
            ("human_expert_replay_review", "pending_external_review"),
            ("neural_game_theoretic_blueprint", "interface_only_untrained"),
            ("safe_game_theoretic_resolving", "blueprint_identity_fallback_only"),
            ("official_game_theoretic_bridge", "not_implemented"),
            ("superhuman_strength", "not_established"),
        )
        return cls(
            schema_version=EVALUATION_SCHEMA_VERSION,
            benchmark_id=benchmark_id,
            benchmark_seed=benchmark_seed,
            deck_fingerprint=deck_fingerprint,
            environment_fingerprint=environment_fingerprint,
            matches=records,
            tactical=tactical,
            strategic=strategic,
            belief=belief,
            search=search,
            by_ablation=grouped("ablation"),
            by_archetype=grouped("opponent_archetype"),
            by_style=grouped("opponent_style"),
            external=external,
            replay_review_queue=tuple(replay_review_queue),
            failure_modes=tuple(failure_modes),
            claims=claims,
        )

    def to_dict(self, *, include_matches: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["tactical"]["action_counts"] = dict(self.tactical.action_counts)
        result["strategic"]["axis_means"] = dict(self.strategic.axis_means)
        result["claims"] = dict(self.claims)
        if include_matches:
            result["matches"] = [record.to_dict() for record in self.matches]
        else:
            result.pop("matches", None)
        result["fingerprint"] = self.fingerprint
        return result

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        overall = EvaluationSlice.from_matches("overall", self.matches)
        claims = dict(self.claims)
        lines = [
            "# Phase 6 evaluation evidence",
            "",
            f"Benchmark: `{self.benchmark_id}`  ",
            f"Evidence fingerprint: `{self.fingerprint}`  ",
            f"Official-engine matches: **{overall.matches}**  ",
            f"Score rate: **{overall.score_rate:.1%}** "
            f"(95% Wilson {overall.confidence_low:.1%}–{overall.confidence_high:.1%})  ",
            f"Illegal action rate: **{self.tactical.illegal_action_rate:.3%}**  ",
            f"Maximum measured decision latency: **{self.tactical.max_latency_ms:.3f} ms**",
            "",
            "## Component ablations",
            "",
            "| Mode | Matches | Wins | Score rate | Prize margin |",
            "|---|---:|---:|---:|---:|",
        ]
        for item in self.by_ablation:
            lines.append(
                f"| {item.label} | {item.matches} | {item.wins} | "
                f"{item.score_rate:.1%} | {item.mean_prize_margin:+.2f} |"
            )
        lines.extend(
            [
                "",
                "## Robustness by archetype",
                "",
                "| Archetype | Matches | Score rate | 95% interval | Illegal |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in self.by_archetype:
            lines.append(
                f"| {item.label} | {item.matches} | {item.score_rate:.1%} | "
                f"{item.confidence_low:.1%}–{item.confidence_high:.1%} | "
                f"{item.illegal_actions} |"
            )
        lines.extend(
            [
                "",
                "## Strategic and belief diagnostics",
                "",
                "| Axis | Mean value |",
                "|---|---:|",
            ]
        )
        for axis, value in self.strategic.axis_means:
            lines.append(f"| {axis} | {value:+.3f} |")
        lines.extend(
            [
                "",
                f"Belief NLL: **{self.belief.mean_nll:.4f}**; "
                f"Brier: **{self.belief.mean_brier:.4f}**.  ",
                f"Search invoked **{self.search.invocations}** times and changed "
                f"**{self.search.changed_actions}** selections.",
                "",
                "## External status and claim boundary",
                "",
                f"- Exact deck playable in current Kaggle pool: "
                f"**{self.external.exact_deck_playable}**.",
            ]
        )
        if self.external.incompatibility:
            lines.append(f"- Incompatibility: {self.external.incompatibility}")
        if self.external.prior_related_public_score is not None:
            lines.append(
                f"- Prior related account submission: "
                f"`{self.external.prior_related_submission}` scored "
                f"**{self.external.prior_related_public_score:.1f}**; this is context, "
                "not evidence for the new agent."
            )
        lines.append(f"- New submission status: **{self.external.submission_status}**.")
        lines.append("")
        lines.append("| Claim | Status |")
        lines.append("|---|---|")
        for claim, status in self.claims:
            lines.append(f"| {claim} | {status} |")
        if self.failure_modes:
            lines.extend(["", "## Known failure modes", ""])
            lines.extend(f"- {item}" for item in self.failure_modes)
        if self.replay_review_queue:
            lines.extend(["", "## Replay review queue", ""])
            lines.extend(f"- `{item}`" for item in self.replay_review_queue)
        return "\n".join(lines) + "\n"
