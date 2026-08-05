#!/usr/bin/env python3
"""Verify every N7 arm against the exact packaged NNAgent action path."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from collect_rocket_blueprint_n5_panel import StrictGreedyAgent
from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2"
)
DEFAULT_ARMS = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n7_marnie_second_awr" / "development_arms" / "arms.json"
)
DEFAULT_OUTPUT = DEFAULT_ARMS.parent / "runtime_parity.json"
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
DEFAULT_OPPONENTS = (
    PROJECT_ROOT / "work" / "arena_bundles" / "AshMarnieGeneralV110_extracted",
    PROJECT_ROOT / "work" / "arena_bundles" / "HelbrosMarnieGeneralRecoveryV4_extracted",
)
ORDER = ("control", "epoch_01", "epoch_02", "epoch_03", "epoch_04")
SEED = 794501
GAMES_PER_OPPONENT = 4
MINIMUM_DECISIONS = 2_000
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=BASE)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    arms_path = args.arms.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N7 parity output: {output}")
    if digest(candidate / "model.pt") != MODEL_SHA256 or digest(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("N7 parity source changed")
    manifest = json.loads(arms_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-n7-development-arms/v1":
        raise ValueError("unexpected N7 arm manifest")
    arms = list(manifest.get("arms") or [])
    if tuple(item["name"] for item in arms) != ORDER:
        raise ValueError("N7 parity arm order changed")
    for item in arms:
        if digest(Path(str(item["path"]))) != item["sha256"]:
            raise ValueError(f"N7 parity arm changed: {item['name']}")
    sys.path.insert(0, str(candidate))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import NNAgent, load_policy  # type: ignore

    deck = read_deck(candidate)
    library_path = candidate / "deck_library.json"
    priors = json.loads(library_path.read_text(encoding="utf-8"))
    models = {
        item["name"]: load_policy(Path(str(item["path"])), map_location="cpu").eval()
        for item in arms
    }
    counts: Counter[str] = Counter()
    total_decisions = 0
    for item in arms:
        arm_name = str(item["name"])
        for opponent_path in DEFAULT_OPPONENTS:
            opponent = OpponentWorker(opponent_path.resolve(), args.evaluator.resolve(), 30.0)
            try:
                for _ in range(GAMES_PER_OPPONENT):
                    opponent.reset()
                    strict = StrictGreedyAgent(models[arm_name], deck, priors)
                    packaged = NNAgent(
                        models[arm_name],
                        greedy=True,
                        deck_ids=deck,
                        forbid_end_when_attack=True,
                        opponent_deck_priors=priors,
                    )
                    observation, start = battle_start(opponent.deck, deck)
                    if observation is None:
                        raise RuntimeError(
                            f"N7 parity start failed: {start.errorPlayer}/{start.errorType}"
                        )
                    try:
                        for _step in range(args.max_decisions):
                            current = observation.get("current") or {}
                            result = int(
                                current.get("result")
                                if current.get("result") is not None else -1
                            )
                            if result != -1:
                                break
                            selection = observation.get("select")
                            actor = int(
                                current.get("yourIndex")
                                if current.get("yourIndex") is not None else -1
                            )
                            if actor == 1:
                                strict_action = strict(observation)
                                packaged_action = packaged(observation)
                                if strict_action != packaged_action:
                                    raise AssertionError(
                                        f"N7 runtime parity mismatch: arm={arm_name} "
                                        f"strict={strict_action} packaged={packaged_action}"
                                    )
                                action = strict_action
                                counts[f"{arm_name}:decisions"] += 1
                                total_decisions += 1
                            elif actor == 0:
                                action = opponent.act(observation)
                            else:
                                raise RuntimeError(f"unexpected N7 parity actor: {actor}")
                            if not legal(list(action), selection):
                                raise RuntimeError(f"illegal N7 parity action: {action}")
                            observation = battle_select(list(action))
                        else:
                            raise RuntimeError("N7 parity game exceeded decision limit")
                    finally:
                        battle_finish()
                    counts[f"{arm_name}:games"] += 1
            finally:
                opponent.close()
    if total_decisions < MINIMUM_DECISIONS:
        raise ValueError(
            f"N7 parity decision floor not met: {total_decisions} < {MINIMUM_DECISIONS}"
        )
    report = {
        "schema": "rocket-n7-runtime-parity/v1",
        "protocol": "N7_ARENA_MARNIE_SECOND_AWR_PROTOCOL.md",
        "training_output_amendment": "N7_TRAINING_OUTPUT_RECOVERY_AMENDMENT.md",
        "official_engine_execution": True,
        "candidate": str(candidate),
        "deck_sha256": DECK_SHA256,
        "arms_manifest": str(arms_path),
        "arms_manifest_sha256": digest(arms_path),
        "verifier_sha256": digest(Path(__file__).resolve()),
        "seed": SEED,
        "games_per_opponent_per_arm": GAMES_PER_OPPONENT,
        "opponents": [str(path.resolve()) for path in DEFAULT_OPPONENTS],
        "total_matched_decisions": total_decisions,
        "total_mismatched_decisions": 0,
        "arms": {
            item["name"]: {
                "model_sha256": item["sha256"],
                "games": counts[f"{item['name']}:games"],
                "matched_decisions": counts[f"{item['name']}:decisions"],
                "mismatched_decisions": 0,
            }
            for item in arms
        },
        "development_only": True,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
