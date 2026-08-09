"""Reproduce packaged decisions and summarize the first internal-arena sample."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOT_NAME = "HelbrosMewtwoSpidopsGTWTV1"
OPTION_NAMES = {
    0: "number",
    1: "yes",
    2: "no",
    3: "card",
    4: "tool_card",
    5: "energy_card",
    6: "energy",
    7: "play",
    8: "attach",
    9: "evolve",
    10: "ability",
    11: "discard",
    12: "retreat",
    13: "attack",
    14: "end",
    15: "skill",
    16: "special_condition",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analyze(replay_directory: Path, extracted: Path, output: Path) -> dict[str, Any]:
    replay_paths = sorted(replay_directory.resolve().glob("*.jsonl"))
    if not replay_paths:
        raise FileNotFoundError(f"no arena replays in {replay_directory}")
    extracted_text = str(extracted.resolve())
    if extracted_text not in sys.path:
        sys.path.insert(0, extracted_text)
    packaged_main = importlib.import_module("main")

    games: list[dict[str, Any]] = []
    objectives: Counter[str] = Counter()
    plans: Counter[str] = Counter()
    selected_option_types: Counter[str] = Counter()
    total_decisions = 0
    total_reproduced = 0
    total_deferred = 0
    total_neural_teacher_disagreements = 0
    total_main_neural_teacher_disagreements = 0
    total_safe_attempts = 0
    total_safe_accepts = 0
    total_exceptions = 0
    confidence_values: list[float] = []

    for path in replay_paths:
        rows = _load_rows(path)
        meta = rows[0]["meta"]
        players = list(meta["players"])
        if BOT_NAME not in players:
            raise AssertionError(f"{path.name} does not contain {BOT_NAME}")
        bot_seat = players.index(BOT_NAME)
        packaged_main.BOT.reset()
        decisions = 0
        reproduced = 0
        teacher_differences = 0
        main_teacher_differences = 0
        attacks = 0
        ends = 0
        last_observation: dict[str, Any] | None = None
        for row in rows[1:]:
            if "obs" not in row:
                continue
            last_observation = row["obs"]
            if int(row["p"]) != bot_seat:
                continue
            observation = row["obs"]
            recorded = list(row["action"])
            teacher = list(packaged_main.TACTICAL_PROVIDER.act(observation))
            reproduced_action = list(packaged_main.BOT.choose(observation))
            decisions += 1
            reproduced += int(reproduced_action == recorded)
            teacher_differs = teacher != recorded
            teacher_differences += int(teacher_differs)
            context = int((observation.get("select") or {}).get("context", -1))
            main_teacher_differences += int(teacher_differs and context == 0)
            options = (observation.get("select") or {}).get("option") or []
            for index in recorded:
                option_type = int(options[index].get("type", -1))
                name = OPTION_NAMES.get(option_type, f"unknown_{option_type}")
                selected_option_types[name] += 1
                attacks += int(name == "attack")
                ends += int(name == "end")
        result = next(row["result"] for row in reversed(rows) if "result" in row)
        summary = packaged_main.BOT.summary()
        telemetry = summary["telemetry"]
        for row in telemetry:
            objectives[str(row["strategic_objective"])] += 1
            plans[str(row["plan"])] += 1
            confidence_values.append(
                float(row["tactical_policy_minimum_prefix_confidence"])
            )
        state = (last_observation or {}).get("current") or {}
        state_players = state.get("players") or [{}, {}]
        own = state_players[bot_seat] if len(state_players) > bot_seat else {}
        opponent_seat = 1 - bot_seat
        opponent = (
            state_players[opponent_seat]
            if len(state_players) > opponent_seat
            else {}
        )
        final_own_board = list(own.get("active") or []) + list(own.get("bench") or [])
        final_opponent_board = list(opponent.get("active") or []) + list(
            opponent.get("bench") or []
        )
        game = {
            "file": path.name,
            "opponent": players[opponent_seat],
            "bot_seat": bot_seat,
            "winner": int(result["winner"]),
            "bot_win": int(result["winner"]) == bot_seat,
            "steps": int(result["steps"]),
            "bot_spent_seconds": float(result["spent"][bot_seat]),
            "decisions": decisions,
            "decisions_reproduced": reproduced,
            "teacher_deferences": int(summary["tactical_teacher_deference_count"]),
            "recorded_actions_differing_from_teacher": teacher_differences,
            "main_actions_differing_from_teacher": main_teacher_differences,
            "safe_resolve_attempts": sum(
                bool(row["safe_resolve_attempted"]) for row in telemetry
            ),
            "safe_resolve_accepts": sum(
                bool(row["safe_resolve_accepted"]) for row in telemetry
            ),
            "controller_exceptions": int(summary["exception_fallback_count"]),
            "attacks": attacks,
            "end_actions": ends,
            "final_own_prizes": len(own.get("prize") or []),
            "final_opponent_prizes": len(opponent.get("prize") or []),
            "final_own_board_ids": [
                int(card.get("id", -1)) for card in final_own_board if card
            ],
            "final_opponent_board_ids": [
                int(card.get("id", -1)) for card in final_opponent_board if card
            ],
        }
        games.append(game)
        total_decisions += decisions
        total_reproduced += reproduced
        total_deferred += game["teacher_deferences"]
        total_neural_teacher_disagreements += teacher_differences
        total_main_neural_teacher_disagreements += main_teacher_differences
        total_safe_attempts += game["safe_resolve_attempts"]
        total_safe_accepts += game["safe_resolve_accepts"]
        total_exceptions += game["controller_exceptions"]

    wins = sum(bool(game["bot_win"]) for game in games)
    losses = len(games) - wins
    report = {
        "schema_version": 1,
        "bot": BOT_NAME,
        "replay_directory": str(replay_directory.resolve()),
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(games),
        "arena_execution_confirmed": True,
        "recorded_decisions": total_decisions,
        "deterministically_reproduced_decisions": total_reproduced,
        "decision_reproduction_rate": total_reproduced / max(total_decisions, 1),
        "controller_exceptions_on_reproduction": total_exceptions,
        "tactical_teacher_deferences": total_deferred,
        "tactical_teacher_deference_rate": total_deferred
        / max(total_decisions, 1),
        "recorded_actions_differing_from_teacher": total_neural_teacher_disagreements,
        "main_actions_differing_from_teacher": total_main_neural_teacher_disagreements,
        "safe_resolve_attempts": total_safe_attempts,
        "safe_resolve_accepts": total_safe_accepts,
        "mean_neural_prefix_confidence": statistics.fmean(confidence_values),
        "selected_option_types": dict(selected_option_types),
        "strategic_objectives": dict(objectives),
        "selected_plans": dict(plans),
        "mean_bot_spent_seconds_per_game": statistics.fmean(
            float(game["bot_spent_seconds"]) for game in games
        ),
        "loss_review": {
            "losses_with_zero_attacks": sum(
                not game["bot_win"] and int(game["attacks"]) == 0 for game in games
            ),
            "losses_with_more_prizes_remaining_than_opponent": sum(
                not game["bot_win"]
                and int(game["final_own_prizes"])
                > int(game["final_opponent_prizes"])
                for game in games
            ),
            "interpretation": (
                "The archive executes deterministically without fallback errors; "
                "the weak result is policy quality under the arena field, not a "
                "packaging or legality failure. Neural-controlled actions that "
                "differ from the tactical provider are the first retraining target."
            ),
        },
        "per_game": games,
        "claim_boundary": (
            "observed internal-arena replay sample only; no superhuman or "
            "generalization claim"
        ),
    }
    if total_reproduced != total_decisions or total_exceptions:
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replays",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "arena_gtwt",
    )
    parser.add_argument(
        "--extracted",
        type=Path,
        default=PROJECT_ROOT / "work" / "final_archive_audit_20260727_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "internal_arena"
            / "HelbrosMewtwoSpidopsGTWTV1_replay_analysis.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.replays, args.extracted, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
