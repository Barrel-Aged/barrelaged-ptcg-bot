#!/usr/bin/env python3
"""Verify the strict N5 evaluator exactly matches the packaged NNAgent action path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from collect_rocket_blueprint_n5_panel import StrictGreedyAgent
from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
)
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
DEFAULT_OPPONENTS = (
    PROJECT_ROOT / "work" / "arena_bundles" / "AshMarnieGeneralV110_extracted",
    PROJECT_ROOT / "work" / "arena_bundles" / "HelbrosMarnieGeneralRecoveryV4_extracted",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--opponent", action="append", type=Path)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--games-per-opponent", type=int, default=4)
    parser.add_argument("--seed", type=int, default=788021)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games_per_opponent <= 0:
        raise ValueError("parity game count must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    model_path = args.model.resolve() if args.model else candidate / "model.pt"
    opponents = tuple(path.resolve() for path in (args.opponent or DEFAULT_OPPONENTS))
    sys.path.insert(0, str(candidate))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import NNAgent, load_policy  # type: ignore

    deck = read_deck(candidate)
    library_path = candidate / "deck_library.json"
    priors = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else None
    model = load_policy(model_path, map_location="cpu").cpu().eval()
    decisions = games = 0
    opponent_rows = []
    for opponent_path in opponents:
        opponent = OpponentWorker(opponent_path, args.evaluator.resolve(), 30.0)
        opponent_games = opponent_decisions = 0
        try:
            for _ in range(args.games_per_opponent):
                opponent.reset()
                strict = StrictGreedyAgent(model, deck, priors)
                packaged = NNAgent(
                    model,
                    greedy=True,
                    deck_ids=deck,
                    forbid_end_when_attack=True,
                    opponent_deck_priors=priors,
                )
                observation, start = battle_start(opponent.deck, deck)
                if observation is None:
                    raise RuntimeError(f"parity engine start failed: {start.errorPlayer}/{start.errorType}")
                try:
                    for _step in range(args.max_decisions):
                        current = observation.get("current") or {}
                        result = int(current.get("result") if current.get("result") is not None else -1)
                        if result != -1:
                            break
                        selection = observation.get("select")
                        actor = int(current.get("yourIndex") if current.get("yourIndex") is not None else -1)
                        if actor == 1:
                            strict_action = strict(observation)
                            packaged_action = packaged(observation)
                            if strict_action != packaged_action:
                                raise AssertionError(
                                    f"N5 runtime parity mismatch at decision {decisions}: strict={strict_action}, packaged={packaged_action}"
                                )
                            action = strict_action
                            decisions += 1
                            opponent_decisions += 1
                        elif actor == 0:
                            action = opponent.act(observation)
                        else:
                            raise RuntimeError(f"unexpected parity actor: {actor}")
                        if not legal(list(action), selection):
                            raise RuntimeError(f"illegal parity action: {action}")
                        observation = battle_select(list(action))
                    else:
                        raise RuntimeError("parity game exceeded decision limit")
                finally:
                    battle_finish()
                games += 1
                opponent_games += 1
        finally:
            opponent.close()
        opponent_rows.append({"opponent": opponent_path.name, "games": opponent_games, "matched_decisions": opponent_decisions})
    report = {
        "schema": "rocket-blueprint-n5-runtime-parity/v1",
        "official_engine_execution": True,
        "candidate": str(candidate),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "games": games,
        "matched_decisions": decisions,
        "mismatched_decisions": 0,
        "opponents": opponent_rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
