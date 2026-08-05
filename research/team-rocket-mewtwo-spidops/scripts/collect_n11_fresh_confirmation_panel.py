#!/usr/bin/env python3
"""Collect one locked N11 fresh official-engine confirmation shard."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from time import time

import numpy as np
import torch

import collect_n10_confirmation_panel as n10
from collect_rocket_blueprint_n5_panel import optional_sha256, play_game, tree_sha256
from train_rank1_second_seat_ppo import OpponentWorker, read_deck, sha256


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
ARMS = ROOT / "artifacts/team_rocket_recovery/n10_epoch4_confirmation/arms.json"
EVALUATOR = Path(r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py")
ORDER = ("control", "candidate")
LOCKED_SEEDS = {800101, 800102, 800201, 800202}
SEED_OPPONENT = {
    800101: "AshMarnieGeneralV110_extracted",
    800102: "AshMarnieGeneralV110_extracted",
    800201: "HelbrosMarnieGeneralRecoveryV4_extracted",
    800202: "HelbrosMarnieGeneralRecoveryV4_extracted",
}
N10_COLLECTOR_SHA = "99882f0cdc4e722a03f45b1bf54bf2cf71e963ab0c1395b5e41b429b91f9a589"
ARMS_SHA = "d60ab7cbd53a49cc9a924f8a9b8d7445937f7db4b9c80d90ef7a04c007bdf443"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=BASE)
    parser.add_argument("--arms", type=Path, default=ARMS)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games-per-arm", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if n10.digest(Path(n10.__file__).resolve()) != N10_COLLECTOR_SHA:
        raise ValueError("frozen N10 collector dependency changed")
    if n10.digest(args.arms.resolve()) != ARMS_SHA:
        raise ValueError("N11 arm manifest changed")
    if args.seed not in LOCKED_SEEDS or args.games_per_arm != 400:
        raise ValueError("N11 panel seed/count differs from lock")
    candidate = args.candidate.resolve()
    opponent_path = args.opponent.resolve()
    if opponent_path.name != SEED_OPPONENT[args.seed]:
        raise ValueError("N11 seed assigned to wrong opponent")
    output = args.output.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing N11 shard: {output}")
    if n10.digest(candidate / "model.pt") != CONTROL_SHA or n10.digest(candidate / "deck.csv") != DECK_SHA:
        raise ValueError("N11 substrate changed")
    arm_manifest, arms = n10.load_arms(args.arms.resolve())
    if tuple(item["name"] for item in arms) != ORDER:
        raise ValueError("N11 arm order changed")
    if {item["sha256"] for item in arms} != {CONTROL_SHA, CANDIDATE_SHA}:
        raise ValueError("N11 model hashes changed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(candidate))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = read_deck(candidate)
    priors = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    models = {item["name"]: load_policy(item["path"], map_location="cpu").cpu().eval() for item in arms}
    fingerprint = {
        "opponent_deck_sha256": sha256(opponent_path / "deck.csv"),
        "opponent_model_sha256": optional_sha256(opponent_path / "model.pt"),
        "opponent_main_sha256": sha256(opponent_path / "main.py"),
        "opponent_bundle_sha256": tree_sha256(opponent_path),
    }
    opponent = OpponentWorker(opponent_path, EVALUATOR, args.action_timeout_seconds)
    schedule = [item["name"] for item in arms for _ in range(args.games_per_arm)]
    random.Random(args.seed).shuffle(schedule)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    started = time()
    try:
        with output.open("x", encoding="utf-8") as sink:
            for game_index, arm in enumerate(schedule, start=1):
                row = play_game((battle_start, battle_select, battle_finish), deck, models[arm], priors, opponent, 1, args.max_decisions)
                arm_row = next(item for item in arms if item["name"] == arm)
                row.update({
                    "schema": "rocket-n11-confirmation-match/v1",
                    "protocol": "rocket-n11-fresh-epoch4-confirmation/v1",
                    "game_index": game_index,
                    "schedule_seed": args.seed,
                    "arm": arm,
                    "arm_sha256": arm_row["sha256"],
                    "opponent": opponent_path.name,
                    **fingerprint,
                })
                sink.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                sink.flush()
                counts[arm] += 1
                wins[arm] += int(row["win"])
                if game_index == 1 or game_index % 50 == 0 or game_index == len(schedule):
                    print(json.dumps({"completed": game_index, "target": len(schedule), "seed": args.seed}), flush=True)
    finally:
        opponent.close()
    manifest = {
        "schema": "rocket-n11-confirmation-shard/v1",
        "protocol": "rocket-n11-fresh-epoch4-confirmation/v1",
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
        "strict_no_fallback_execution": True,
        "collector_sha256": n10.digest(Path(__file__).resolve()),
        "collector_dependency_sha256": N10_COLLECTOR_SHA,
        "candidate": str(candidate),
        "candidate_bundle_sha256": tree_sha256(candidate),
        "deck_sha256": DECK_SHA,
        "initial_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "arms_manifest": str(args.arms.resolve()),
        "arms_manifest_sha256": n10.digest(args.arms.resolve()),
        "arms": arm_manifest["arms"],
        "opponent": str(opponent_path),
        **fingerprint,
        "schedule_seed": args.seed,
        "own_seat": 1,
        "games_per_arm": args.games_per_arm,
        "rows": len(schedule),
        "counts": dict(counts),
        "wins": dict(wins),
        "output_sha256": n10.digest(output),
        "elapsed_seconds": time() - started,
        "deployment_authority": False,
    }
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(output), "counts": dict(counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

