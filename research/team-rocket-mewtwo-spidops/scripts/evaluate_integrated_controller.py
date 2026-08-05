"""Fresh official-engine evaluation for the Phase 6 integrated controller."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.competition import (  # noqa: E402
    CompetitionAgentConfig,
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    OptionType,
    Phase6CompetitionAgent,
    SelectionContract,
    _in_play,
    _player,
    _safe_int,
    _state,
    resolve_official_deck,
)
from rocketbot.integrated_planner import (  # noqa: E402
    ARCHETYPE_NAMES,
    INTENT_NAMES,
    IntegratedBeliefPlannerAgent,
    IntegratedPlannerConfig,
)
from rocketbot.official_evaluation import OFFICIAL_PANEL_DECKS, PriorityBaseline  # noqa: E402


DEFAULT_RUNTIME = PROJECT_ROOT / "work" / "official_kaggle" / "sample_submission"
DEFAULT_TEACHER = Path(
    r"C:\Users\jhos4\PokemonDocs\_local_archive_2026-07-26\evidence"
    r"\rocket_spidops_mewtwo_20260719\package_pro_v3_extracted"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "integrated_evaluation"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


class TacticalTeacher:
    def __init__(self, teacher_root: Path, deck: Sequence[int]) -> None:
        teacher_text = str(teacher_root.resolve())
        if teacher_text not in sys.path:
            sys.path.insert(0, teacher_text)
        from bot.rocket_spidops import RocketSpidopsPolicy  # type: ignore[import-not-found]

        self.policy_type = RocketSpidopsPolicy
        self.deck = tuple(deck)
        self.policy = self.policy_type(self.deck)

    def reset(self) -> None:
        self.policy = self.policy_type(self.deck)

    def choose(self, observation: Mapping[str, Any]) -> list[int]:
        return list(self.policy.act(observation))


def _integrated(
    deck: Any,
    model: Path,
    teacher_root: Path,
    ablation: str = "full",
) -> IntegratedBeliefPlannerAgent:
    return IntegratedBeliefPlannerAgent(
        deck,
        model,
        telemetry_limit=4_096,
        planner_config=IntegratedPlannerConfig(ablation=ablation),
        tactical_teacher=TacticalTeacher(teacher_root, deck.card_ids),
    )


def _telemetry_summary(agent: Any, *, true_archetype: str | None, true_intent: str | None) -> dict[str, Any]:
    if not isinstance(agent, IntegratedBeliefPlannerAgent):
        memory = getattr(agent, "memory", None)
        return {
            "controller": type(agent).__name__,
            "fallback_count": int(getattr(memory, "fallback_count", 0)) if memory is not None else 0,
        }
    rows = list(agent.memory.telemetry)
    archetype_nll: list[float] = []
    archetype_brier: list[float] = []
    intent_nll: list[float] = []
    intent_brier: list[float] = []
    if true_archetype in ARCHETYPE_NAMES:
        for row in rows:
            posterior = row["archetype_range"]
            archetype_nll.append(-math.log(max(float(posterior[true_archetype]), 1e-12)))
            archetype_brier.append(
                sum((float(posterior[name]) - float(name == true_archetype)) ** 2 for name in ARCHETYPE_NAMES)
            )
    if true_intent in INTENT_NAMES:
        for row in rows:
            posterior = row["intent_range"]
            intent_nll.append(-math.log(max(float(posterior[true_intent]), 1e-12)))
            intent_brier.append(
                sum((float(posterior[name]) - float(name == true_intent)) ** 2 for name in INTENT_NAMES)
            )
    return {
        "controller": "IntegratedBeliefPlannerAgent",
        "model_fingerprint": agent.planner.fingerprint,
        "ablation": agent.planner_config.ablation,
        "decision_count": len(rows),
        "exception_fallback_count": agent.memory.exception_fallback_count,
        "exception_types": dict(agent.memory.exception_types),
        "search_gate_attempts": sum(bool(row["safe_resolve_attempted"]) for row in rows),
        "safe_resolve_accepts": sum(bool(row["safe_resolve_accepted"]) for row in rows),
        "tactical_teacher_deferences": agent.memory.tactical_teacher_deference_count,
        "mean_search_gate_probability": statistics.fmean(float(row["search_gate_probability"]) for row in rows) if rows else 0.0,
        "strategic_objectives": dict(Counter(str(row["strategic_objective"]) for row in rows)),
        "plans": dict(Counter(str(row["plan"]) for row in rows)),
        "mean_archetype_nll": statistics.fmean(archetype_nll) if archetype_nll else None,
        "mean_archetype_brier": statistics.fmean(archetype_brier) if archetype_brier else None,
        "mean_intent_nll": statistics.fmean(intent_nll) if intent_nll else None,
        "mean_intent_brier": statistics.fmean(intent_brier) if intent_brier else None,
        "final_hidden_hypothesis_count": len(agent.memory.hidden_hypotheses),
        "deception_alternative_retained": any(
            row.get("play_style") == "deceptive" for row in agent.memory.hidden_hypotheses
        ),
    }


def run_match(
    *,
    battle_start: Any,
    battle_select: Any,
    battle_finish: Any,
    match_id: str,
    focal_label: str,
    focal_agent: Any,
    focal_deck: Sequence[int],
    opponent_label: str,
    opponent_agent: Any,
    opponent_deck: Sequence[int],
    focal_seat: int,
    opponent_archetype: str | None,
    opponent_intent: str | None,
    capture: bool,
    replay_directory: Path,
    maximum_selections: int = 2_000,
) -> dict[str, Any]:
    focal_agent.reset()
    opponent_agent.reset()
    agents = [focal_agent, opponent_agent]
    labels = [focal_label, opponent_label]
    decks = [tuple(focal_deck), tuple(opponent_deck)]
    if focal_seat == 1:
        agents.reverse()
        labels.reverse()
        decks.reverse()
    observation, start = battle_start(list(decks[0]), list(decks[1]))
    if observation is None:
        raise RuntimeError(
            f"official engine rejected {match_id}: player={start.errorPlayer} error={start.errorType}"
        )
    selections = 0
    illegal = 0
    engine_error: str | None = None
    latencies: list[float] = []
    action_counts: Counter[str] = Counter()
    lethal_opportunities = 0
    lethal_conversions = 0
    board_samples: list[tuple[float, float, float, float]] = []
    replay_rows: list[dict[str, Any]] = [
        {"meta": {"match_id": match_id, "players": labels, "decks": [list(deck) for deck in decks], "focal_seat": focal_seat}}
    ]
    try:
        while (
            observation is not None
            and _safe_int(_state(observation).get("result"), -1) == -1
            and selections < maximum_selections
        ):
            actor = _safe_int(_state(observation).get("yourIndex"), 0)
            is_focal = actor == focal_seat
            contract = SelectionContract.from_observation(observation)
            lethal_indices: set[int] = set()
            if is_focal and isinstance(focal_agent, IntegratedBeliefPlannerAgent):
                opposing = _in_play(_player(observation, 1 - actor))[:1]
                if opposing:
                    hp = _safe_int(opposing[0].get("hp"), 9_999)
                    for index, option in enumerate(observation["select"]["option"]):
                        if _safe_int(option.get("type"), -1) == int(OptionType.ATTACK):
                            if focal_agent._tactical_prior._expected_attack_damage(
                                observation, _safe_int(option.get("attackId"), -1)
                            ) >= hp:
                                lethal_indices.add(index)
                    lethal_opportunities += int(bool(lethal_indices))
                own_board = _in_play(_player(observation, actor))
                opposing_board = _in_play(_player(observation, 1 - actor))
                board_samples.append(
                    (
                        float(len(own_board)),
                        float(len(opposing_board)),
                        float(sum(_safe_int(card.get("hp"), 0) for card in own_board)),
                        float(sum(_safe_int(card.get("hp"), 0) for card in opposing_board)),
                    )
                )
            started = perf_counter()
            try:
                action = list(contract.validate(agents[actor].choose(observation)))
            except Exception:
                illegal += int(is_focal)
                action = list(contract.fallback())
            elapsed = (perf_counter() - started) * 1_000.0
            if is_focal:
                latencies.append(elapsed)
                for index in action:
                    option_type = _safe_int(observation["select"]["option"][index].get("type"), -1)
                    try:
                        action_counts[OptionType(option_type).name.lower()] += 1
                    except ValueError:
                        action_counts[f"unknown_{option_type}"] += 1
                lethal_conversions += int(bool(lethal_indices.intersection(action)))
            if capture:
                replay_rows.append({"p": actor, "obs": deepcopy(observation), "action": list(action)})
            observation = battle_select(action)
            selections += 1
        if selections >= maximum_selections:
            engine_error = f"selection cap reached ({maximum_selections})"
    except Exception as exc:
        engine_error = f"{type(exc).__name__}: {exc}"
    finally:
        final_state = _state(observation or {})
        winner = _safe_int(final_state.get("result"), -1)
        turns = _safe_int(final_state.get("turn"), 0)
        if capture:
            replay_rows.append({"result": {"winner": winner, "turns": turns, "selections": selections, "engine_error": engine_error}})
            replay_directory.mkdir(parents=True, exist_ok=True)
            replay_path = replay_directory / f"{match_id}.jsonl"
            replay_path.write_text(
                "".join(_canonical(row) + "\n" for row in replay_rows),
                encoding="utf-8",
                newline="\n",
            )
        else:
            replay_path = None
        battle_finish()
    telemetry = _telemetry_summary(
        focal_agent,
        true_archetype=opponent_archetype,
        true_intent=opponent_intent,
    )
    return {
        "match_id": match_id,
        "focal_label": focal_label,
        "opponent_label": opponent_label,
        "focal_seat": focal_seat,
        "winner": winner,
        "focal_win": winner == focal_seat,
        "focal_utility": 1.0 if winner == focal_seat else -1.0 if winner in (0, 1) else 0.0,
        "turns": turns,
        "selections": selections,
        "runner_illegal_fallbacks": illegal,
        "engine_error": engine_error,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "max_latency_ms": max(latencies, default=0.0),
        "action_counts": dict(action_counts),
        "lethal_opportunities": lethal_opportunities,
        "lethal_conversions": lethal_conversions,
        "mean_own_board": statistics.fmean(row[0] for row in board_samples) if board_samples else 0.0,
        "mean_opponent_board": statistics.fmean(row[1] for row in board_samples) if board_samples else 0.0,
        "mean_own_board_hp": statistics.fmean(row[2] for row in board_samples) if board_samples else 0.0,
        "mean_opponent_board_hp": statistics.fmean(row[3] for row in board_samples) if board_samples else 0.0,
        "telemetry": telemetry,
        "replay": replay_path.name if replay_path else None,
    }


def _suite_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    wins = sum(bool(record["focal_win"]) for record in records)
    utilities = [float(record["focal_utility"]) for record in records]
    mean_utility = statistics.fmean(utilities) if utilities else 0.0
    standard_error = (
        statistics.stdev(utilities) / math.sqrt(len(utilities)) if len(utilities) > 1 else 0.0
    )
    return {
        "games": len(records),
        "wins": wins,
        "losses": sum(record["winner"] in (0, 1) and not record["focal_win"] for record in records),
        "unresolved": sum(record["winner"] not in (0, 1) for record in records),
        "win_rate": wins / max(len(records), 1),
        "win_rate_wilson_95": list(_wilson(wins, len(records))),
        "mean_utility": mean_utility,
        "utility_normal_95": [mean_utility - 1.96 * standard_error, mean_utility + 1.96 * standard_error],
        "runner_illegal_fallbacks": sum(int(record["runner_illegal_fallbacks"]) for record in records),
        "controller_exception_fallbacks": sum(int(record["telemetry"].get("exception_fallback_count", 0)) for record in records),
        "engine_errors": sum(record["engine_error"] is not None for record in records),
        "mean_latency_ms": statistics.fmean(float(record["mean_latency_ms"]) for record in records) if records else 0.0,
        "max_latency_ms": max((float(record["max_latency_ms"]) for record in records), default=0.0),
        "search_gate_attempts": sum(int(record["telemetry"].get("search_gate_attempts", 0)) for record in records),
        "safe_resolve_accepts": sum(int(record["telemetry"].get("safe_resolve_accepts", 0)) for record in records),
        "tactical_teacher_deferences": sum(int(record["telemetry"].get("tactical_teacher_deferences", 0)) for record in records),
        "deception_alternative_games": sum(bool(record["telemetry"].get("deception_alternative_retained", False)) for record in records),
        "lethal_opportunities": sum(int(record["lethal_opportunities"]) for record in records),
        "lethal_conversions": sum(int(record["lethal_conversions"]) for record in records),
    }


def evaluate(
    *,
    runtime: Path,
    model: Path,
    teacher_root: Path,
    output: Path,
) -> dict[str, Any]:
    runtime_text = str(runtime.resolve())
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

    output.mkdir(parents=True, exist_ok=True)
    replay_directory = output / "replays"
    deck = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    all_records: list[dict[str, Any]] = []
    suites: dict[str, list[dict[str, Any]]] = {}

    def execute(suite: str, **kwargs: Any) -> None:
        records = suites.setdefault(suite, [])
        record = run_match(
            battle_start=battle_start,
            battle_select=battle_select,
            battle_finish=battle_finish,
            replay_directory=replay_directory,
            **kwargs,
        )
        records.append(record)
        all_records.append(record)
        print(_canonical({key: record[key] for key in ("match_id", "focal_win", "turns", "engine_error")}), flush=True)

    style_to_intent = {
        "tempo": "setup",
        "attack_rush": "prize_race",
        "control": "lock_disruption",
    }
    capture_budget = 12
    for replica in range(2):
        for panel in OFFICIAL_PANEL_DECKS:
            for style in ("tempo", "attack_rush", "control"):
                for seat in (0, 1):
                    match_id = f"robust-{panel.archetype}-{style}-s{seat}-r{replica}"
                    execute(
                        "robustness_panel",
                        match_id=match_id,
                        focal_label="integrated_full",
                        focal_agent=_integrated(deck, model, teacher_root),
                        focal_deck=deck.card_ids,
                        opponent_label=f"{panel.archetype}_{style}",
                        opponent_agent=PriorityBaseline(panel.card_ids, style=style, reverse=bool(replica)),
                        opponent_deck=panel.card_ids,
                        focal_seat=seat,
                        opponent_archetype=panel.archetype,
                        opponent_intent=style_to_intent[style],
                        capture=capture_budget > 0,
                    )
                    capture_budget -= int(capture_budget > 0)

    for index in range(40):
        seat = index % 2
        heuristic = Phase6CompetitionAgent(
            deck,
            config=CompetitionAgentConfig(decision_budget_ms=1_000.0, search_budget_ms=0.0),
        )
        execute(
            "head_to_head_legacy_heuristic",
            match_id=f"legacy-h2h-{index:03d}",
            focal_label="integrated_full",
            focal_agent=_integrated(deck, model, teacher_root),
            focal_deck=deck.card_ids,
            opponent_label="legacy_phase6_heuristic",
            opponent_agent=heuristic,
            opponent_deck=deck.card_ids,
            focal_seat=seat,
            opponent_archetype="rocket_mewtwo",
            opponent_intent=None,
            capture=False,
        )

    for index in range(40):
        seat = index % 2
        execute(
            "head_to_head_tactical_teacher",
            match_id=f"teacher-h2h-{index:03d}",
            focal_label="integrated_full",
            focal_agent=_integrated(deck, model, teacher_root),
            focal_deck=deck.card_ids,
            opponent_label="frozen_prov3_tactical_teacher",
            opponent_agent=TacticalTeacher(teacher_root, deck.card_ids),
            opponent_deck=deck.card_ids,
            focal_seat=seat,
            opponent_archetype="rocket_mewtwo",
            opponent_intent=None,
            capture=False,
        )

    for ablation in ("no_beliefs", "no_plan", "no_relational", "no_meta", "no_safe_resolve"):
        for index in range(20):
            seat = index % 2
            execute(
                f"ablation_{ablation}",
                match_id=f"ablation-{ablation}-{index:03d}",
                focal_label="integrated_full",
                focal_agent=_integrated(deck, model, teacher_root),
                focal_deck=deck.card_ids,
                opponent_label=f"integrated_{ablation}",
                opponent_agent=_integrated(deck, model, teacher_root, ablation),
                opponent_deck=deck.card_ids,
                focal_seat=seat,
                opponent_archetype="rocket_mewtwo",
                opponent_intent=None,
                capture=False,
            )

    summaries = {name: _suite_summary(records) for name, records in suites.items()}
    belief_rows = suites["robustness_panel"]
    summaries["belief_calibration"] = {
        "archetype_nll": statistics.fmean(
            float(record["telemetry"]["mean_archetype_nll"])
            for record in belief_rows
            if record["telemetry"]["mean_archetype_nll"] is not None
        ),
        "archetype_brier": statistics.fmean(
            float(record["telemetry"]["mean_archetype_brier"])
            for record in belief_rows
            if record["telemetry"]["mean_archetype_brier"] is not None
        ),
        "intent_nll": statistics.fmean(
            float(record["telemetry"]["mean_intent_nll"])
            for record in belief_rows
            if record["telemetry"]["mean_intent_nll"] is not None
        ),
        "intent_brier": statistics.fmean(
            float(record["telemetry"]["mean_intent_brier"])
            for record in belief_rows
            if record["telemetry"]["mean_intent_brier"] is not None
        ),
    }
    model_raw = json.loads(model.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_fingerprint": model_raw["fingerprint"],
        "model_sha256": sha256(model.read_bytes()).hexdigest(),
        "deck_resolution": deck.to_dict(),
        "official_engine_execution": True,
        "game_count": len(all_records),
        "summaries": summaries,
        "records_path": "official_match_records.json",
        "replay_count": sum(record["replay"] is not None for record in all_records),
        "claim_boundary": "local official-engine strength estimate; not internal-arena or Kaggle performance",
    }
    report["fingerprint"] = _fingerprint(report)
    (output / "official_match_records.json").write_text(
        json.dumps(all_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "integrated_evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "integrated_controller" / "integrated_planner.json",
    )
    parser.add_argument("--teacher-root", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(
        runtime=args.runtime,
        model=args.model,
        teacher_root=args.teacher_root,
        output=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
