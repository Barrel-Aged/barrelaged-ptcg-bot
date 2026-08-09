#!/usr/bin/env python3
"""Verify both N15 arms against the exact packaged NNAgent action path."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from collect_rocket_blueprint_n5_panel import StrictGreedyAgent, tree_sha256
from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/development_arms/candidate_bundle"
ARMS = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/development_arms/arms.json"
OUTPUT = ARMS.parent / "runtime_parity.json"
EVALUATOR = Path(r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py")
OPPONENTS = (
    ROOT / "work/arena_bundles/AshMarnieGeneralV110_extracted",
    ROOT / "work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted",
)
ORDER = ("control", "candidate")
SEED = 815001
GAMES_PER_OPPONENT_ARM = 4
MINIMUM_DECISIONS = 600

ARMS_SHA = "e46197852d7e88d069b54d9ecc5cc978b5d0d5b0c731cb34409a9d88c1fa34ef"
BUNDLE_SHA = "982fa183a4ed963bda201c5dd01b55cf316e37848797441ef34d4848e0c7ff74"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arms() -> list[dict[str, str]]:
    if digest(ARMS) != ARMS_SHA:
        raise ValueError("N15 arms manifest changed")
    manifest = json.loads(ARMS.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rocket-n15-development-arms/v1"
        or manifest.get("protocol") != "rocket-n15-prospective-terminal-consolidation/v1"
        or manifest.get("candidate_bundle_sha256") != BUNDLE_SHA
    ):
        raise ValueError("N15 arm provenance changed")
    arms = list(manifest.get("arms") or ())
    if tuple(item.get("name") for item in arms) != ORDER:
        raise ValueError("N15 arm order changed")
    for item in arms:
        path = Path(str(item["path"])).resolve()
        if digest(path) != item.get("sha256"):
            raise ValueError(f"N15 model changed: {item['name']}")
    return arms


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N15 runtime verifier accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N15 parity output: {OUTPUT}")
    if tree_sha256(BUNDLE) != BUNDLE_SHA:
        raise ValueError("N15 candidate bundle changed")
    if digest(BUNDLE / "deck.csv") != DECK_SHA:
        raise ValueError("N15 deck changed")
    if digest(BUNDLE / "soldier/net_v10.py") != RUNTIME_SHA:
        raise ValueError("N15 residual runtime changed")
    arms = load_arms()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    sys.path.insert(0, str(BUNDLE))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import NNAgent, load_policy  # type: ignore

    deck = read_deck(BUNDLE)
    priors = json.loads((BUNDLE / "deck_library.json").read_text(encoding="utf-8"))
    models = {
        str(item["name"]): load_policy(Path(str(item["path"])), map_location="cpu").cpu().eval()
        for item in arms
    }
    counts: Counter[str] = Counter()
    for item in arms:
        arm = str(item["name"])
        for opponent_path in OPPONENTS:
            opponent = OpponentWorker(opponent_path, EVALUATOR, 30.0)
            try:
                for _game in range(GAMES_PER_OPPONENT_ARM):
                    opponent.reset()
                    strict = StrictGreedyAgent(models[arm], deck, priors)
                    packaged = NNAgent(
                        models[arm],
                        greedy=True,
                        deck_ids=deck,
                        forbid_end_when_attack=True,
                        opponent_deck_priors=priors,
                    )
                    observation, start = battle_start(opponent.deck, deck)
                    if observation is None:
                        raise RuntimeError(
                            f"N15 parity start failed: {start.errorPlayer}/{start.errorType}"
                        )
                    try:
                        for _step in range(2_000):
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
                                packaged_action = list(packaged(observation))
                                if strict_action != packaged_action:
                                    raise AssertionError(
                                        "N15 runtime mismatch "
                                        f"arm={arm}: {strict_action}/{packaged_action}"
                                    )
                                action = strict_action
                                counts[f"{arm}:decisions"] += 1
                            elif actor == 0:
                                action = opponent.act(observation)
                            else:
                                raise RuntimeError(f"unexpected N15 actor: {actor}")
                            if not legal(list(action), selection):
                                raise RuntimeError(f"illegal N15 parity action: {action}")
                            observation = battle_select(list(action))
                        else:
                            raise RuntimeError("N15 parity game exceeded decision limit")
                    finally:
                        battle_finish()
                    counts[f"{arm}:games"] += 1
            finally:
                opponent.close()
    total = sum(counts[f"{arm}:decisions"] for arm in ORDER)
    if total < MINIMUM_DECISIONS:
        raise ValueError(f"N15 parity decision floor not met: {total}")
    report = {
        "schema": "rocket-n15-runtime-parity/v1",
        "protocol": "rocket-n15-prospective-terminal-consolidation/v1",
        "official_engine_execution": True,
        "candidate_bundle": str(BUNDLE.resolve()),
        "candidate_bundle_sha256": BUNDLE_SHA,
        "deck_sha256": DECK_SHA,
        "arms_manifest": str(ARMS.resolve()),
        "arms_manifest_sha256": ARMS_SHA,
        "verifier_sha256": digest(Path(__file__).resolve()),
        "seed": SEED,
        "games_per_opponent_per_arm": GAMES_PER_OPPONENT_ARM,
        "opponents": [str(path.resolve()) for path in OPPONENTS],
        "total_matched_decisions": total,
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
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
