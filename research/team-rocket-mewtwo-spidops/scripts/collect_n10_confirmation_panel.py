#!/usr/bin/env python3
"""Collect one locked N10 official-engine confirmation shard."""

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


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
DEFAULT_ARMS = ROOT / "artifacts/team_rocket_recovery/n10_epoch4_confirmation/arms.json"
EVALUATOR = Path(r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py")
ORDER = ("control", "candidate")
LOCKED_SEEDS = {799101, 799102, 799201, 799202}
SEED_OPPONENT = {
    799101: "AshMarnieGeneralV110_extracted",
    799102: "AshMarnieGeneralV110_extracted",
    799201: "HelbrosMarnieGeneralRecoveryV4_extracted",
    799202: "HelbrosMarnieGeneralRecoveryV4_extracted",
}
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arms(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-n10-confirmation-arms/v1" or manifest.get("protocol") != "rocket-n10-independent-epoch4-confirmation/v1":
        raise ValueError("unexpected N10 arm manifest")
    if manifest.get("deck_sha256") != DECK_SHA or manifest.get("initial_model_sha256") != CONTROL_SHA or manifest.get("candidate_model_sha256") != CANDIDATE_SHA:
        raise ValueError("N10 arm provenance mismatch")
    arms = []
    for item in manifest.get("arms") or ():
        model_path = Path(str(item["path"])).resolve()
        if not model_path.exists() or digest(model_path) != item.get("sha256"):
            raise ValueError(f"N10 arm missing/changed: {model_path}")
        arms.append({"name": str(item["name"]), "path": model_path, "sha256": str(item["sha256"])})
    if tuple(item["name"] for item in arms) != ORDER:
        raise ValueError("N10 arm order changed")
    return manifest, arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=BASE)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games-per-arm", type=int, default=400)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.seed not in LOCKED_SEEDS or args.games_per_arm != 400:
        raise ValueError("N10 panel seed/count differs from lock")
    candidate = args.candidate.resolve()
    opponent_path = args.opponent.resolve()
    if opponent_path.name != SEED_OPPONENT[args.seed]:
        raise ValueError("N10 panel seed assigned to wrong opponent")
    output = args.output.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing N10 panel shard: {output}")
    if digest(candidate / "model.pt") != CONTROL_SHA or digest(candidate / "deck.csv") != DECK_SHA:
        raise ValueError("N10 panel substrate changed")
    arm_manifest, arms = load_arms(args.arms.resolve())
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
                    "schema": "rocket-n10-confirmation-match/v1",
                    "protocol": "rocket-n10-independent-epoch4-confirmation/v1",
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
        "schema": "rocket-n10-confirmation-shard/v1",
        "protocol": "rocket-n10-independent-epoch4-confirmation/v1",
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
        "strict_no_fallback_execution": True,
        "collector_sha256": digest(Path(__file__).resolve()),
        "candidate": str(candidate),
        "candidate_bundle_sha256": tree_sha256(candidate),
        "deck_sha256": DECK_SHA,
        "initial_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "arms_manifest": str(args.arms.resolve()),
        "arms_manifest_sha256": digest(args.arms.resolve()),
        "arms": arm_manifest["arms"],
        "opponent": str(opponent_path),
        **fingerprint,
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

