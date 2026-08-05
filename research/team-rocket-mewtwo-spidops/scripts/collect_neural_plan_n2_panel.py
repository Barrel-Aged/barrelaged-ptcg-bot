#!/usr/bin/env python3
"""Collect locked N2 ensemble arms in complete official-engine games."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from time import time
from typing import Any

import numpy as np
import torch

from collect_neural_plan_panel import play_game
from train_rank1_second_seat_ppo import OpponentWorker, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
)
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
DEFAULT_ARMS = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "neural_plan_n2" / "arms" / "arms.json"
)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
LOCKED_ARMS = (
    "zero_control",
    "mean_logits_025",
    "mean_logits_050",
    "mean_logits_100",
    "consensus_3of4",
    "consensus_4of4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "development", "validation"), required=True)
    parser.add_argument("--games-per-arm", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def load_arms(path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-neural-plan-ensemble-arms/v1":
        raise ValueError("unexpected N2 arm manifest")
    if manifest.get("protocol") != "rocket-neural-plan-ensemble/v1":
        raise ValueError("N2 arm protocol mismatch")
    if manifest.get("deck_sha256") != DECK_SHA256 or manifest.get("base_model_sha256") != MODEL_SHA256:
        raise ValueError("N2 frozen substrate mismatch")
    arms = []
    for arm in manifest.get("arms") or ():
        source = Path(str(arm["path"])).resolve()
        if not source.exists() or sha256(source) != str(arm["sha256"]):
            raise ValueError(f"N2 arm missing or changed: {source}")
        arms.append({"name": str(arm["name"]), "path": source, "sha256": str(arm["sha256"])})
    names = tuple(arm["name"] for arm in arms)
    if split == "development" and names != LOCKED_ARMS:
        raise ValueError(f"N2 development arms differ from lock: {names}")
    if "zero_control" not in names or len(names) != len(set(names)):
        raise ValueError("N2 arms require one unique zero_control")
    return manifest, arms


def main() -> int:
    args = parse_args()
    if args.games_per_arm <= 0 or args.max_decisions <= 0:
        raise ValueError("game and decision counts must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"refusing existing N2 output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    arm_manifest, arms = load_arms(args.arms.resolve(), args.split)
    candidate = CANDIDATE.resolve()
    opponent_path = args.opponent.resolve()
    if sha256(candidate / "deck.csv") != DECK_SHA256 or sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("N2 candidate substrate changed")
    sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore
    from neural_plan_ensemble_runtime import NeuralPlanEnsembleAgent

    own_deck = read_deck(candidate)
    net = load_policy(candidate / "model.pt").cpu().eval()
    library_path = candidate / "deck_library.json"
    deck_library = json.loads(library_path.read_text(encoding="utf-8"))
    agents = {
        arm["name"]: NeuralPlanEnsembleAgent(
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
    opponent = OpponentWorker(opponent_path, EVALUATOR, args.action_timeout_seconds)
    schedule = [arm["name"] for arm in arms for _ in range(args.games_per_arm)]
    random.Random(args.seed).shuffle(schedule)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    engine = (battle_start, battle_select, battle_finish)
    started = time()
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
                        "schema": "rocket-neural-plan-ensemble-panel-match/v1",
                        "protocol": "rocket-neural-plan-ensemble/v1",
                        "split": args.split,
                        "game_index": game_index,
                        "schedule_seed": args.seed,
                        "arm": arm_name,
                        "arm_sha256": next(arm["sha256"] for arm in arms if arm["name"] == arm_name),
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
        "schema": "rocket-neural-plan-ensemble-panel-shard/v1",
        "protocol": "rocket-neural-plan-ensemble/v1",
        "split": args.split,
        "sealed": False,
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
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
        "claim_boundary": f"{args.split} official-engine evidence; not an arena strength claim",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
