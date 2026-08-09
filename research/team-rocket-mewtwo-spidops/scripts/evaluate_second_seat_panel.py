#!/usr/bin/env python3
"""Run a fixed number of genuine official-engine games with agent A going second."""

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
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--maximum-attempts", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--action-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("ptcg_local_evaluator", path.resolve())
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
    all_attempts = []
    with output.open("w", encoding="utf-8") as sink:
        for attempt in range(1, args.maximum_attempts + 1):
            capture = FirstPlayerCapture()
            # The official local engine assigns the first turn to seat 0. Put
            # the focal candidate in seat 1 and still verify the observation.
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
            stderr_path = log_root / f"game_{attempt:04d}" / f"{args.candidate_name}.stderr.log"
            stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
            record["first_player"] = capture.first_player
            record["candidate_seat"] = candidate_seat
            record["candidate_started_second"] = (
                capture.first_player in (0, 1)
                and capture.first_player != candidate_seat
            )
            record["semantic_rule_interventions"] = stderr.count('"semantic_rule"')
            record["candidate_tracebacks"] = stderr.count("Traceback")
            all_attempts.append(record)
            if record["candidate_started_second"]:
                accepted.append(record)
                sink.write(json.dumps(record, separators=(",", ":")) + "\n")
                sink.flush()
                print(
                    json.dumps(
                        {
                            "accepted": len(accepted),
                            "of": args.games,
                            "attempt": attempt,
                            "winner": record["winner"],
                            "interventions": record["semantic_rule_interventions"],
                            "reason": record["reason"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if len(accepted) >= args.games:
                    break
    if len(accepted) < args.games:
        raise RuntimeError(
            f"collected only {len(accepted)} second-seat games in {len(all_attempts)} attempts"
        )
    wins = sum(record["winner"] == args.candidate_name for record in accepted)
    losses = sum(record["winner"] == args.opponent_name for record in accepted)
    draws = len(accepted) - wins - losses
    summary = {
        "schema": "rocket-second-seat-full-match-panel/v1",
        "candidate": str(candidate),
        "opponent": str(opponent),
        "accepted_games": len(accepted),
        "attempted_games": len(all_attempts),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisive_win_rate": wins / max(1, wins + losses),
        "semantic_rule_interventions": sum(
            record["semantic_rule_interventions"] for record in accepted
        ),
        "games_with_intervention": sum(
            record["semantic_rule_interventions"] > 0 for record in accepted
        ),
        "engine_or_agent_errors": sum(
            record["reason"] != "terminal" for record in accepted
        ),
        "candidate_tracebacks": sum(
            record["candidate_tracebacks"] for record in accepted
        ),
        "mean_steps": sum(record["steps"] for record in accepted) / len(accepted),
        "output": str(output),
        "claim_boundary": "fresh local official-engine panel; not an arena strength claim",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
