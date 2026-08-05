#!/usr/bin/env python3
"""Run genuine-second-seat official panels with causal-overlay telemetry."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)


class FirstPlayerCapture:
    def __init__(self) -> None:
        self.first_player = None

    def write(self, payload: str) -> int:
        row = json.loads(payload)
        value = ((row.get("obs") or {}).get("current") or {}).get("firstPlayer")
        if self.first_player is None and value in (0, 1):
            self.first_player = int(value)
        return len(payload)

    def flush(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--opponent-name", default="opponent")
    parser.add_argument("--sim-dir", type=Path)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--maximum-attempts", type=int, default=480)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--action-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("ptcg_local_evaluator_causal", path.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_deck(path: Path) -> list[int]:
    deck = [int(value) for value in (path / "deck.csv").read_text().split()]
    if len(deck) != 60:
        raise ValueError(f"{path}: deck does not contain 60 cards")
    return deck


def telemetry(stderr: str) -> dict[str, int]:
    roots = interventions = overlay_errors = 0
    for line in stderr.splitlines():
        try:
            row = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            continue
        decision = row.get("randomized_causal_overlay")
        if isinstance(decision, dict):
            roots += int(bool(decision.get("used")))
            interventions += int(bool(decision.get("changed")))
        overlay_errors += int("randomized_causal_overlay_error" in row)
    return {
        "causal_overlay_roots": roots,
        "causal_overlay_interventions": interventions,
        "causal_overlay_errors": overlay_errors,
        "tracebacks": stderr.count("Traceback"),
    }


def main() -> int:
    args = parse_args()
    if args.games <= 0 or args.maximum_attempts < args.games:
        raise ValueError("invalid game/attempt count")
    evaluator = load_evaluator(args.evaluator)
    candidate = args.candidate.resolve()
    opponent = args.opponent.resolve()
    sim_dir = (args.sim_dir or candidate).resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_root = output.with_suffix(".logs")
    log_root.mkdir(parents=True, exist_ok=True)
    specs = [
        {
            "name": args.candidate_name,
            "dir": candidate,
            "env": {"PTCG_POLICY": "mcts"},
            "deck": read_deck(candidate),
        },
        {
            "name": args.opponent_name,
            "dir": opponent,
            "env": {"PTCG_POLICY": "heuristic"},
            "deck": read_deck(opponent),
        },
    ]
    run_args = SimpleNamespace(
        max_steps=args.max_steps,
        action_timeout_s=args.action_timeout_seconds,
    )
    battle_api = evaluator._import_sim(sim_dir)
    accepted = []
    attempted = 0
    with output.open("w", encoding="utf-8") as sink:
        for attempt in range(1, args.maximum_attempts + 1):
            attempted = attempt
            capture = FirstPlayerCapture()
            candidate_seat = 1
            record = evaluator._run_game(
                attempt,
                specs,
                [1, 0],
                run_args,
                battle_api,
                log_root,
                capture,
            )
            stderr_path = (
                log_root / f"game_{attempt:04d}" / f"{args.candidate_name}.stderr.log"
            )
            stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
            record.update(telemetry(stderr))
            record["first_player"] = capture.first_player
            record["candidate_seat"] = candidate_seat
            record["candidate_started_second"] = bool(
                capture.first_player in (0, 1) and capture.first_player != candidate_seat
            )
            if not record["candidate_started_second"]:
                continue
            accepted.append(record)
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")
            sink.flush()
            if len(accepted) == 1 or len(accepted) % 10 == 0 or len(accepted) == args.games:
                print(
                    json.dumps(
                        {
                            "accepted": len(accepted),
                            "of": args.games,
                            "winner": record["winner"],
                            "interventions": record["causal_overlay_interventions"],
                            "reason": record["reason"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if len(accepted) >= args.games:
                break
    if len(accepted) != args.games:
        raise RuntimeError(f"collected only {len(accepted)} genuine-second-seat games")
    wins = sum(record["winner"] == args.candidate_name for record in accepted)
    losses = sum(record["winner"] == args.opponent_name for record in accepted)
    draws = len(accepted) - wins - losses
    summary = {
        "schema": "rocket-causal-second-seat-full-match-panel/v1",
        "candidate": str(candidate),
        "opponent": str(opponent),
        "accepted_games": len(accepted),
        "attempted_games": attempted,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisive_win_rate": wins / max(1, wins + losses),
        "causal_overlay_roots": sum(row["causal_overlay_roots"] for row in accepted),
        "causal_overlay_interventions": sum(
            row["causal_overlay_interventions"] for row in accepted
        ),
        "games_with_intervention": sum(
            row["causal_overlay_interventions"] > 0 for row in accepted
        ),
        "causal_overlay_errors": sum(row["causal_overlay_errors"] for row in accepted),
        "engine_or_agent_errors": sum(row["reason"] != "terminal" for row in accepted),
        "tracebacks": sum(row["tracebacks"] for row in accepted),
        "mean_steps": sum(row["steps"] for row in accepted) / len(accepted),
        "output": str(output),
        "claim_boundary": "fresh local official-engine panel; not an arena strength claim",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
