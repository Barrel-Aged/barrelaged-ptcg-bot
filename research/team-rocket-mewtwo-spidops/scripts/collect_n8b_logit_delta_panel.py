#!/usr/bin/env python3
"""Collect one locked N8B official-engine development shard."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from time import time
from typing import Any

import numpy as np
import torch

from collect_rocket_blueprint_n5_panel import optional_sha256, play_game, tree_sha256
from train_rank1_second_seat_ppo import OpponentWorker, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "scale_value_calibrated_v1"
)
DEFAULT_ARMS = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n8b_recovery_logit_delta" / "development_arms" / "arms.json"
)
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
ORDER = ("control", "delta_050", "delta_100", "delta_150", "delta_200")
LOCKED_SEEDS = {796101, 796102, 796201, 796202}
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=BASE)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games-per-arm", type=int, default=120)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def load_arms(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-n8b-logit-delta-arms/v1":
        raise ValueError("unexpected N8B arm manifest")
    if manifest.get("protocol") != "rocket-recovery-logit-delta-transfer/v1":
        raise ValueError("N8B protocol mismatch")
    if manifest.get("deck_sha256") != DECK_SHA256:
        raise ValueError("N8B deck provenance mismatch")
    arms = []
    for item in manifest.get("arms") or ():
        model_path = Path(str(item["path"])).resolve()
        if not model_path.exists() or digest(model_path) != item.get("sha256"):
            raise ValueError(f"N8B arm missing or changed: {model_path}")
        arms.append(
            {
                "name": str(item["name"]),
                "alpha": float(item["alpha"]),
                "path": model_path,
                "sha256": str(item["sha256"]),
            }
        )
    if tuple(item["name"] for item in arms) != ORDER:
        raise ValueError("N8B arm order differs from lock")
    return manifest, arms


def main() -> int:
    args = parse_args()
    if args.seed not in LOCKED_SEEDS or args.games_per_arm != 120:
        raise ValueError("N8B panel seed or game count differs from lock")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    opponent_path = args.opponent.resolve()
    output = args.output.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing N8B shard: {output}")
    if digest(candidate / "deck.csv") != DECK_SHA256 or digest(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("N8B frozen candidate changed")
    arm_manifest, arms = load_arms(args.arms.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(candidate))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = read_deck(candidate)
    priors = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    models = {
        item["name"]: load_policy(item["path"], map_location="cpu").cpu().eval()
        for item in arms
    }
    opponent_fingerprint = {
        "opponent_deck_sha256": sha256(opponent_path / "deck.csv"),
        "opponent_model_sha256": optional_sha256(opponent_path / "model.pt"),
        "opponent_main_sha256": sha256(opponent_path / "main.py"),
        "opponent_bundle_sha256": tree_sha256(opponent_path),
    }
    opponent = OpponentWorker(
        opponent_path, args.evaluator.resolve(), args.action_timeout_seconds
    )
    schedule = [item["name"] for item in arms for _ in range(args.games_per_arm)]
    random.Random(args.seed).shuffle(schedule)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    started = time()
    engine = (battle_start, battle_select, battle_finish)
    try:
        with output.open("x", encoding="utf-8") as sink:
            for game_index, arm_name in enumerate(schedule, start=1):
                row = play_game(
                    engine, deck, models[arm_name], priors, opponent, 1, args.max_decisions
                )
                arm = next(item for item in arms if item["name"] == arm_name)
                row.update(
                    {
                        "schema": "rocket-n8b-development-panel-match/v1",
                        "protocol": "rocket-recovery-logit-delta-transfer/v1",
                        "game_index": game_index,
                        "schedule_seed": args.seed,
                        "arm": arm_name,
                        "alpha": arm["alpha"],
                        "arm_sha256": arm["sha256"],
                        "opponent": opponent_path.name,
                        **opponent_fingerprint,
                    }
                )
                sink.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                sink.flush()
                counts[arm_name] += 1
                wins[arm_name] += int(row["win"])
                if game_index == 1 or game_index % 50 == 0 or game_index == len(schedule):
                    print(
                        json.dumps(
                            {
                                "completed": game_index,
                                "target": len(schedule),
                                "seed": args.seed,
                                "opponent": opponent_path.name,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        opponent.close()
    manifest = {
        "schema": "rocket-n8b-development-panel-shard/v1",
        "protocol": "rocket-recovery-logit-delta-transfer/v1",
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
        "strict_no_fallback_execution": True,
        "collector_sha256": digest(Path(__file__).resolve()),
        "candidate": str(candidate),
        "candidate_bundle_sha256": tree_sha256(candidate),
        "deck_sha256": DECK_SHA256,
        "initial_model_sha256": MODEL_SHA256,
        "arms_manifest": str(args.arms.resolve()),
        "arms_manifest_sha256": digest(args.arms.resolve()),
        "arms": arm_manifest["arms"],
        "opponent": str(opponent_path),
        **opponent_fingerprint,
        "schedule_seed": args.seed,
        "own_seat": 1,
        "games_per_arm": args.games_per_arm,
        "rows": len(schedule),
        "counts": dict(counts),
        "wins": dict(wins),
        "output_sha256": digest(output),
        "elapsed_seconds": time() - started,
        "deployment_authority": False,
    }
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(output), "counts": dict(counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

