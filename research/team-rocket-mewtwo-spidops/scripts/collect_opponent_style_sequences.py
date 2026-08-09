#!/usr/bin/env python3
"""Collect public/own-visible token sequences for opponent-style inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from time import time

import numpy as np
import torch

from collect_neural_plan_panel import play_game
from train_rank1_second_seat_ppo import OpponentWorker, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
)
ZERO_ARM = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "neural_plan_n2" / "arms" / "zero_control.pt"
)
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--label", type=int, choices=(0, 1), required=True)
    parser.add_argument("--label-name", choices=("v110", "recovery_v4"), required=True)
    parser.add_argument("--split", choices=("train", "calibration", "validation"), required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-mode",
        choices=("global_token", "public_event"),
        default="global_token",
    )
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games <= 0:
        raise ValueError("games must be positive")
    expected = {0: "v110", 1: "recovery_v4"}
    if expected[args.label] != args.label_name:
        raise ValueError("style label and label-name disagree")
    output = args.output.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing style dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    candidate = CANDIDATE.resolve()
    opponent_path = args.opponent.resolve()
    if sha256(candidate / "deck.csv") != DECK_SHA256 or sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("style collection substrate changed")
    sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore
    own_deck = read_deck(candidate)
    net = load_policy(candidate / "model.pt").cpu().eval()
    if args.feature_mode == "public_event":
        from public_event_style_runtime import (
            EventTrackingEnsembleAgent as AgentType,
            FEATURE_WIDTH,
        )
    else:
        from neural_plan_ensemble_runtime import NeuralPlanEnsembleAgent as AgentType

        FEATURE_WIDTH = int(net.cfg["h"])

    deck_library = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    agent = AgentType(
        net,
        ZERO_ARM,
        deck_ids=own_deck,
        opponent_deck_priors=deck_library,
        forbid_end_when_attack=True,
        expected_base_model_sha256=MODEL_SHA256,
        expected_deck_sha256=DECK_SHA256,
    )
    opponent = OpponentWorker(opponent_path, EVALUATOR, args.action_timeout_seconds)
    engine = (battle_start, battle_select, battle_finish)
    episodes = []
    started = time()
    try:
        for game_index in range(1, args.games + 1):
            result = play_game(
                engine,
                own_deck,
                agent,
                opponent,
                max_decisions=args.max_decisions,
            )
            feature_history = (
                agent.public_event_feature_history
                if args.feature_mode == "public_event"
                else agent.global_token_history
            )
            if not feature_history:
                raise RuntimeError("style episode has no own-decision tokens")
            tokens = torch.stack(feature_history, dim=0).to(torch.float16)
            episodes.append(
                {
                    "tokens": tokens,
                    "label": int(args.label),
                    "label_name": args.label_name,
                    "game_index": game_index,
                    "win": int(result["win"]),
                    "result": int(result["result"]),
                    "started_second": bool(result["started_second"]),
                    "exception_fallbacks": int(result["exception_fallbacks"]),
                }
            )
            if game_index == 1 or game_index % 25 == 0 or game_index == args.games:
                print(
                    json.dumps(
                        {
                            "split": args.split,
                            "style": args.label_name,
                            "completed": game_index,
                            "target": args.games,
                            "last_sequence": int(tokens.shape[0]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        opponent.close()
    payload = {
        "schema": (
            "rocket-public-event-style-sequences/v1"
            if args.feature_mode == "public_event"
            else "rocket-opponent-style-sequences/v1"
        ),
        "protocol": (
            "rocket-public-event-style-gate/v1"
            if args.feature_mode == "public_event"
            else "rocket-opponent-style-gate/v1"
        ),
        "split": args.split,
        "label": args.label,
        "label_name": args.label_name,
        "feature": (
            "explicit_public_opponent_events_and_board"
            if args.feature_mode == "public_event"
            else "frozen_blueprint_global_token_sequence"
        ),
        "global_width": int(FEATURE_WIDTH),
        "official_engine_execution": True,
        "hidden_information_used": False,
        "deck_sha256": DECK_SHA256,
        "base_model_sha256": MODEL_SHA256,
        "zero_arm_sha256": sha256(ZERO_ARM),
        "opponent": str(opponent_path),
        "opponent_deck_sha256": sha256(opponent_path / "deck.csv"),
        "opponent_model_sha256": sha256(opponent_path / "model.pt"),
        "seed": args.seed,
        "episodes": episodes,
    }
    torch.save(payload, output)
    manifest = {
        key: value
        for key, value in payload.items()
        if key != "episodes"
    }
    manifest.update(
        {
            "schema": "rocket-opponent-style-sequence-shard/v1",
            "games": len(episodes),
            "tokens": sum(int(episode["tokens"].shape[0]) for episode in episodes),
            "all_started_second": all(bool(episode["started_second"]) for episode in episodes),
            "exception_fallbacks": sum(int(episode["exception_fallbacks"]) for episode in episodes),
            "artifact": str(output),
            "artifact_sha256": sha256(output),
            "elapsed_seconds": time() - started,
            "claim_boundary": "style-classifier dataset; not strength evidence",
        }
    )
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
