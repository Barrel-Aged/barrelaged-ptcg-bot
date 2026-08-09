#!/usr/bin/env python3
"""N5 near-greedy PPO fine-tune of the Rocket relational blueprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import time
from typing import Any

import numpy as np
import torch

from train_rank1_second_seat_ppo import (
    OpponentWorker,
    assign_gae,
    ppo_update,
    read_deck,
    rollout_game,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
)
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
OPPONENTS = (
    PROJECT_ROOT / "work" / "arena_bundles" / "AshMarnieGeneralV110_extracted",
    PROJECT_ROOT / "work" / "arena_bundles" / "HelbrosMarnieGeneralRecoveryV4_extracted",
)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--rollout-games", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--block-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--clip", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.30)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--kl-coef", type=float, default=0.20)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=2_000)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    for name in ("iterations", "rollout_games", "ppo_epochs", "batch_size", "threads", "max_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.rollout_games % len(OPPONENTS):
        raise ValueError("rollout games must divide equally across opponents")
    if not (0.0 < args.temperature <= 0.5):
        raise ValueError("N5 requires near-greedy temperature in (0, 0.5]")


def main() -> int:
    args = parse_args()
    validate(args)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty N5 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    candidate = CANDIDATE.resolve()
    if sha256(candidate / "deck.csv") != DECK_SHA256 or sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("N5 frozen deck/model changed")
    sys.path.insert(0, str(candidate))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import feat_to_tensors, load_policy, save_policy  # type: ignore

    model = load_policy(candidate / "model.pt", map_location="cpu").cpu()
    reference = load_policy(candidate / "model.pt", map_location="cpu").cpu().eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.logit_head.parameters():
        parameter.requires_grad_(True)
    for parameter in model.value.parameters():
        parameter.requires_grad_(True)
    for parameter in model.encoder.layers[-1].parameters():
        parameter.requires_grad_(True)
    head_parameters = list(model.logit_head.parameters()) + list(model.value.parameters())
    block_parameters = list(model.encoder.layers[-1].parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.head_learning_rate},
            {"params": block_parameters, "lr": args.block_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    deck = read_deck(candidate)
    library_path = candidate / "deck_library.json"
    priors = json.loads(library_path.read_text(encoding="utf-8"))
    opponents = [path.resolve() for path in OPPONENTS]
    workers = {
        path.name: OpponentWorker(path, EVALUATOR, args.timeout) for path in opponents
    }
    opponent_hashes = {
        path.name: {
            "deck_sha256": sha256(path / "deck.csv"),
            "model_sha256": sha256(path / "model.pt"),
        }
        for path in opponents
    }
    engine = (battle_start, battle_select, battle_finish)
    rng = random.Random(args.seed)
    history = []
    total_games = 0
    started = time()
    try:
        for iteration in range(1, args.iterations + 1):
            schedule = []
            per_opponent = args.rollout_games // len(opponents)
            for path in opponents:
                schedule.extend([path.name] * per_opponent)
            rng.shuffle(schedule)
            model.eval()
            episodes = []
            steps = []
            wins = {name: 0 for name in workers}
            games = {name: 0 for name in workers}
            iteration_started = time()
            for game_index, opponent_name in enumerate(schedule, start=1):
                episode = rollout_game(
                    engine,
                    deck,
                    workers[opponent_name],
                    model,
                    features,
                    feat_to_tensors,
                    Belief,
                    priors,
                    greedy=False,
                    temperature=args.temperature,
                    collect=True,
                    max_steps=args.max_steps,
                )
                assign_gae(episode.steps, episode.reward, args.gamma, args.gae_lambda)
                episodes.append(episode)
                steps.extend(episode.steps)
                games[opponent_name] += 1
                wins[opponent_name] += int(episode.reward > 0)
                total_games += 1
                if game_index == 1 or game_index % 16 == 0 or game_index == len(schedule):
                    print(
                        json.dumps(
                            {
                                "iteration": iteration,
                                "rollout_game": game_index,
                                "rollout_target": len(schedule),
                                "opponent": opponent_name,
                                "reward": episode.reward,
                                "actions": len(episode.steps),
                                "total_games": total_games,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            update = ppo_update(model, reference, optimizer, steps, args, torch.device("cpu"))
            checkpoint = output / f"iteration_{iteration:02d}.pt"
            save_policy(model, checkpoint)
            record = {
                "iteration": iteration,
                "games": games,
                "wins": wins,
                "actions": len(steps),
                "update": update,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
                "elapsed_seconds": time() - iteration_started,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    final_path = output / "model.pt"
    save_policy(model, final_path)
    report = {
        "schema": "rocket-greedy-compatible-ppo-training/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "official_engine_execution": True,
        "hidden_information_used_by_actor_or_critic": False,
        "deck_sha256": DECK_SHA256,
        "initial_model_sha256": MODEL_SHA256,
        "opponents": [str(path) for path in opponents],
        "opponent_hashes": opponent_hashes,
        "training_seed": args.seed,
        "hyperparameters": {
            key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, bool))
        },
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "iterations": history,
        "total_training_games": total_games,
        "final_artifact": final_path.name,
        "final_artifact_sha256": sha256(final_path),
        "checkpoint_selection": "final iteration only",
        "elapsed_seconds": time() - started,
        "claim_boundary": "training telemetry only; not validation or arena strength evidence",
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, "seed": args.seed, "games": total_games, "artifact": str(final_path), "sha256": report["final_artifact_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
