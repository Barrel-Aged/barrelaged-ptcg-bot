#!/usr/bin/env python3
"""Collect an interleaved neural-plan panel in the official local engine."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from time import perf_counter, time
from typing import Any, Mapping

import numpy as np
import torch

from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "candidates"
    / "scale_value_calibrated_v1"
)
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
DEFAULT_ARMS = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "neural_plan_n1"
    / "development_arms"
    / "arms.json"
)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
LOCKED_ARMS = (
    "zero_control",
    "seed_781101",
    "seed_781102",
    "seed_781103",
    "seed_781104",
    "mean_all4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("smoke", "development", "validation"), required=True
    )
    parser.add_argument("--games-per-arm", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def load_arms(path: Path, *, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-neural-plan-development-arms/v1":
        raise ValueError("unexpected neural-plan arm manifest")
    if manifest.get("protocol") != "rocket-neural-plan-residual/v1":
        raise ValueError("arm manifest protocol mismatch")
    if manifest.get("deck_sha256") != DECK_SHA256:
        raise ValueError("arm manifest deck mismatch")
    if manifest.get("base_model_sha256") != MODEL_SHA256:
        raise ValueError("arm manifest blueprint mismatch")
    arms = []
    for arm in manifest.get("arms") or ():
        source = Path(str(arm["path"])).resolve()
        if not source.exists() or sha256(source) != str(arm["sha256"]):
            raise ValueError(f"arm artifact missing or changed: {source}")
        arms.append({"name": str(arm["name"]), "path": source, "sha256": str(arm["sha256"])})
    names = tuple(arm["name"] for arm in arms)
    if len(set(names)) != len(names):
        raise ValueError("arm names are not unique")
    if split == "development" and names != LOCKED_ARMS:
        raise ValueError(f"development arms differ from lock: {names}")
    if "zero_control" not in names:
        raise ValueError("panel requires zero_control")
    return manifest, arms


def public_metrics(observation: Mapping[str, Any]) -> dict[str, Any]:
    from train_neural_plan_residual import board_snapshot

    snapshot = board_snapshot(observation, own_seat=1)
    players = ((observation.get("current") or {}).get("players") or ())
    own = players[1] if len(players) == 2 else {}
    board = [
        pokemon
        for zone in ("active", "bench")
        for pokemon in (own or {}).get(zone) or ()
        if isinstance(pokemon, Mapping)
    ]
    return {
        "mewtwo_on_board": any(int(pokemon.get("id") or 0) == 431 for pokemon in board),
        "ready_attacker_count": len(snapshot["ready"]),
    }


def play_game(
    engine: tuple[Any, Any, Any],
    own_deck: list[int],
    own_agent: Any,
    opponent: OpponentWorker,
    *,
    max_decisions: int,
) -> dict[str, Any]:
    battle_start, battle_select, battle_finish = engine
    opponent.reset()
    own_agent.reset_game()
    observation, start = battle_start(opponent.deck, own_deck)
    if observation is None:
        raise RuntimeError(
            f"official engine start failed: {start.errorPlayer}/{start.errorType}"
        )
    first_player: int | None = None
    ever_mewtwo = False
    ever_second_ready = False
    max_ready = 0
    started = perf_counter()
    try:
        for step in range(max_decisions):
            current = observation.get("current") or {}
            observed_first = current.get("firstPlayer")
            if observed_first in (0, 1):
                first_player = int(observed_first)
                if first_player != 0:
                    raise RuntimeError("panel did not put opponent first")
            metrics = public_metrics(observation)
            ever_mewtwo = ever_mewtwo or bool(metrics["mewtwo_on_board"])
            ever_second_ready = ever_second_ready or bool(metrics["ready_attacker_count"] >= 2)
            max_ready = max(max_ready, int(metrics["ready_attacker_count"]))
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                if first_player != 0:
                    raise RuntimeError("official game ended without verified second seat")
                telemetry = own_agent.telemetry_snapshot()
                return {
                    "result": result,
                    "win": int(result == 1),
                    "draw": int(result == 2),
                    "first_player": first_player,
                    "started_second": True,
                    "decisions": step,
                    "plan_decisions": int(telemetry["decisions"]),
                    "intervention_decisions": int(telemetry["intervention_decisions"]),
                    "games_with_intervention": int(telemetry["intervention_decisions"] > 0),
                    "exception_fallbacks": int(telemetry["exception_fallbacks"]),
                    "final_plan_values": telemetry["plan_values"],
                    "final_meta_weights": telemetry["meta_weights"],
                    "ever_mewtwo_on_board": ever_mewtwo,
                    "ever_second_ready_attacker": ever_second_ready,
                    "max_ready_attackers": max_ready,
                    "duration_seconds": perf_counter() - started,
                }
            selection = observation.get("select")
            if not isinstance(selection, Mapping):
                raise RuntimeError("official engine emitted no live selection")
            actor = int(current.get("yourIndex") or 0)
            action = own_agent(observation) if actor == 1 else opponent.act(dict(observation))
            if not legal(list(action), dict(selection)):
                raise RuntimeError(f"illegal panel action actor={actor}: {action}")
            observation = battle_select(list(action))
        raise RuntimeError(f"panel game exceeded {max_decisions} decisions")
    finally:
        battle_finish()


def main() -> int:
    args = parse_args()
    if args.games_per_arm <= 0 or args.max_decisions <= 0:
        raise ValueError("game and decision counts must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    opponent_path = args.opponent.resolve()
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing existing panel output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    arm_manifest, arms = load_arms(args.arms.resolve(), split=args.split)
    if sha256(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("candidate deck is not the frozen qualifying deck")
    if sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("candidate model is not the frozen blueprint")
    sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore
    from neural_plan_residual_runtime import NeuralPlanResidualAgent

    own_deck = read_deck(candidate)
    net = load_policy(candidate / "model.pt").cpu().eval()
    library_path = candidate / "deck_library.json"
    deck_library = (
        json.loads(library_path.read_text(encoding="utf-8"))
        if library_path.exists()
        else None
    )
    agents = {
        arm["name"]: NeuralPlanResidualAgent(
            net,
            arm["path"],
            deck_ids=own_deck,
            opponent_deck_priors=deck_library,
            forbid_end_when_attack=True,
            expected_base_model_sha256=MODEL_SHA256,
            expected_deck_sha256=DECK_SHA256,
        )
        for arm in arms
    }
    opponent = OpponentWorker(
        opponent_path, args.evaluator.resolve(), args.action_timeout_seconds
    )
    schedule = [arm["name"] for arm in arms for _ in range(args.games_per_arm)]
    rng = random.Random(args.seed)
    rng.shuffle(schedule)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    started = time()
    engine = (battle_start, battle_select, battle_finish)
    try:
        with output.open("x", encoding="utf-8") as sink:
            for game_index, arm_name in enumerate(schedule, start=1):
                row = play_game(
                    engine,
                    own_deck,
                    agents[arm_name],
                    opponent,
                    max_decisions=args.max_decisions,
                )
                row.update(
                    {
                        "schema": "rocket-neural-plan-panel-match/v1",
                        "protocol": "rocket-neural-plan-residual/v1",
                        "split": args.split,
                        "game_index": game_index,
                        "schedule_seed": args.seed,
                        "arm": arm_name,
                        "arm_sha256": next(
                            arm["sha256"] for arm in arms if arm["name"] == arm_name
                        ),
                        "opponent": opponent_path.name,
                        "opponent_deck_sha256": sha256(opponent_path / "deck.csv"),
                        "opponent_model_sha256": sha256(opponent_path / "model.pt"),
                    }
                )
                sink.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                sink.flush()
                counts[arm_name] += 1
                wins[arm_name] += int(row["win"])
                if game_index == 1 or game_index % 20 == 0 or game_index == len(schedule):
                    print(
                        json.dumps(
                            {
                                "completed": game_index,
                                "target": len(schedule),
                                "arm": arm_name,
                                "arm_games": counts[arm_name],
                                "arm_wins": wins[arm_name],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        opponent.close()
    manifest = {
        "schema": "rocket-neural-plan-panel-shard/v1",
        "protocol": "rocket-neural-plan-residual/v1",
        "split": args.split,
        "sealed": False,
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
        "candidate": str(candidate),
        "deck_sha256": DECK_SHA256,
        "base_model_sha256": MODEL_SHA256,
        "arms_manifest": str(args.arms.resolve()),
        "arms_manifest_sha256": sha256(args.arms.resolve()),
        "arms": arm_manifest["arms"],
        "opponent": str(opponent_path),
        "opponent_deck_sha256": sha256(opponent_path / "deck.csv"),
        "opponent_model_sha256": sha256(opponent_path / "model.pt"),
        "schedule_seed": args.seed,
        "games_per_arm": args.games_per_arm,
        "rows": len(schedule),
        "counts": dict(counts),
        "wins": dict(wins),
        "elapsed_seconds": time() - started,
        "claim_boundary": "development data; not an arena strength claim",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
