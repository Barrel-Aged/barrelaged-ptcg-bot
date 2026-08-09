"""Automated replay triage and a stable rubric for later human expert review."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from statistics import fmean
from typing import Any, Mapping, Sequence

from .competition import DecisionTelemetry, TEAM_ROCKET_POKEMON
from .evaluation import OfficialMatchRecord


@dataclass(frozen=True, slots=True)
class ReplayIssue:
    category: str
    severity: str
    turn: int | None
    evidence: str
    review_question: str


@dataclass(frozen=True, slots=True)
class ReplayReview:
    match_id: str
    target_seat: int
    frame_count: int
    result: int
    result_reason: int | None
    issues: tuple[ReplayIssue, ...]
    critical_turns: tuple[int, ...]
    human_review_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"# Replay review: {self.match_id}",
            "",
            f"Target seat: **{self.target_seat}**  ",
            f"Result: **{self.result}**; reason: **{self.result_reason}**  ",
            f"Frames: **{self.frame_count}**  ",
            f"Human expert status: **{self.human_review_status}**",
            "",
            "## Automated triage",
            "",
        ]
        if not self.issues:
            lines.append("No deterministic rubric issue was triggered.")
        for issue in self.issues:
            turn = "n/a" if issue.turn is None else str(issue.turn)
            lines.extend(
                [
                    f"### {issue.category} ({issue.severity}, turn {turn})",
                    "",
                    issue.evidence,
                    "",
                    f"Expert question: {issue.review_question}",
                    "",
                ]
            )
        lines.extend(
            [
                "## Expert rubric",
                "",
                "Score each item 0–2 and cite the exact frame/turn:",
                "",
                "- Setup preservation and four-Team-Rocket activation timing.",
                "- Trainer/search ordering before irreversible discard or shuffle.",
                "- Energy placement, Charging Up use, and Erasure Ball discard economy.",
                "- Evolution, retreat, gust, and knockout sequencing.",
                "- Prize map and liabilities placed on the Bench.",
                "- Optionality retained after the chosen line.",
                "- Whether the archetype belief and any search intervention were justified.",
                "",
                "This automated triage is not a human expert endorsement.",
            ]
        )
        return "\n".join(lines) + "\n"


def review_official_replay(
    *,
    replay_data: str,
    record: OfficialMatchRecord,
    telemetry: Sequence[DecisionTelemetry],
) -> ReplayReview:
    raw = json.loads(replay_data)
    if not isinstance(raw, list) or not raw:
        raise ValueError("official replay must be a non-empty frame list")
    frames = tuple(frame for frame in raw if isinstance(frame, Mapping))
    final = frames[-1]
    current = final.get("current", {})
    result = int(current.get("result", record.winner))
    result_reason: int | None = None
    for log in final.get("logs", ()):
        if isinstance(log, Mapping) and log.get("type") == "Result":
            value = log.get("reason")
            result_reason = None if value is None else int(value)

    issues: list[ReplayIssue] = []
    critical_turns: set[int] = set()
    target_won = result == record.agent_seat
    if not target_won:
        reason_text = {
            1: "opponent took all Prize cards",
            2: "target could not draw at turn start",
            3: "target had no Active Pokémon",
            4: "a card effect ended the game",
        }.get(result_reason, "unknown terminal reason")
        issues.append(
            ReplayIssue(
                category="terminal_loss",
                severity="high",
                turn=record.turns,
                evidence=f"The target lost via {reason_text} with prize margin "
                f"{record.opponent_prizes_remaining - record.own_prizes_remaining:+d}.",
                review_question="Which earlier irreversible decision first made this loss likely?",
            )
        )
        critical_turns.add(record.turns)

    setup_frame = next(
        (
            frame
            for frame in frames
            if int(frame.get("current", {}).get("turn", 0)) >= 2
        ),
        None,
    )
    if setup_frame is not None:
        player = setup_frame.get("current", {}).get("players", ())[record.agent_seat]
        board = list(player.get("active", ())) + list(player.get("bench", ()))
        rocket_count = sum(
            isinstance(card, Mapping) and int(card.get("id", -1)) in TEAM_ROCKET_POKEMON
            for card in board
        )
        if rocket_count < 4:
            turn = int(setup_frame.get("current", {}).get("turn", 0))
            issues.append(
                ReplayIssue(
                    category="setup_activation",
                    severity="medium",
                    turn=turn,
                    evidence=f"Only {rocket_count} Team Rocket Pokémon were in play at the "
                    "first post-setup checkpoint; Mewtwo requires four to attack.",
                    review_question="Did the opening hand permit a faster four-Rocket board?",
                )
            )
            critical_turns.add(turn)

    if record.lethal_opportunities > record.lethal_conversions:
        issues.append(
            ReplayIssue(
                category="lethal_conversion",
                severity="high",
                turn=None,
                evidence=(
                    f"Converted {record.lethal_conversions}/{record.lethal_opportunities} "
                    "static lethal opportunities."
                ),
                review_question="Were missed attacks truly lethal after effects, Weakness, and prevention?",
            )
        )
    if record.first_attack_turn is None or record.first_attack_turn > 7:
        issues.append(
            ReplayIssue(
                category="tempo",
                severity="medium",
                turn=record.first_attack_turn,
                evidence=f"First attack turn was {record.first_attack_turn}.",
                review_question="Could setup/search/attachment ordering have attacked earlier safely?",
            )
        )
    if record.mean_option_count < 2.0:
        issues.append(
            ReplayIssue(
                category="optionality",
                severity="medium",
                turn=None,
                evidence=f"Mean legal option count was only {record.mean_option_count:.2f}.",
                review_question="Did the policy prematurely consume flexible cards or bench slots?",
            )
        )
    searched = tuple(item for item in telemetry if item.search_used)
    changed = tuple(item for item in searched if item.search_changed_action)
    if searched and not changed:
        issues.append(
            ReplayIssue(
                category="search_efficiency",
                severity="low",
                turn=None,
                evidence=f"Search ran {len(searched)} times without changing a selection.",
                review_question="Should the gate threshold rise for this matchup/state family?",
            )
        )
    if changed:
        critical_turns.update(item.turn for item in changed)
    if telemetry:
        mean_entropy = fmean(item.belief_entropy for item in telemetry)
        if mean_entropy > 1.7:
            issues.append(
                ReplayIssue(
                    category="belief_uncertainty",
                    severity="low",
                    turn=None,
                    evidence=f"Mean archetype entropy remained high ({mean_entropy:.3f} nats).",
                    review_question="Which public actions should have separated the live archetypes?",
                )
            )

    return ReplayReview(
        match_id=record.match_id,
        target_seat=record.agent_seat,
        frame_count=len(frames),
        result=result,
        result_reason=result_reason,
        issues=tuple(issues),
        critical_turns=tuple(sorted(critical_turns)),
    )

