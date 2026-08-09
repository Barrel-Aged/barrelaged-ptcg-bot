#!/usr/bin/env python3
"""Verify both N17 arms against the packaged NNAgent on official games."""

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
BUNDLE = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/development_arms/candidate_bundle"
ARMS = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/development_arms/arms.json"
OUTPUT = ARMS.parent / "runtime_parity.json"
EVALUATOR = Path(r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py")
OPPONENTS = (
    ROOT / "work/arena_bundles/AshMarnieGeneralV110_extracted",
    ROOT / "work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted",
)
ORDER = ("control", "candidate")
SEED = 817000
GAMES_PER_OPPONENT_ARM = 4
MINIMUM_DECISIONS = 600

ARMS_SHA = "28ac3ef5c0748f9b28faef4e9d071f809855c9d78fcdd951ea1f4aa38132b8bc"
BUNDLE_SHA = "a7d02c5bc39428f570317c704893a103c6f1225491f1b72c84312b980ef50dd3"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
HELPER_SHA = "f3cb21219efbf82c94a1bebe1d90b712fcd02cb28037397a02176d2ce5047f6b"
WORKER_SHA = "a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85"
EVALUATOR_SHA = "d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303"
OPPONENT_TREE_SHA = {
    "AshMarnieGeneralV110_extracted": "3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b",
    "HelbrosMarnieGeneralRecoveryV4_extracted": "caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arms() -> list[dict[str, str]]:
    if digest(ARMS) != ARMS_SHA:
        raise ValueError("N17 arms manifest changed")
    manifest = json.loads(ARMS.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rocket-n17-development-arms/v1"
        or manifest.get("protocol") != "rocket-n17-replicated-target-residual/v1"
        or manifest.get("candidate_bundle_sha256") != BUNDLE_SHA
        or manifest.get("deck_sha256") != DECK_SHA
        or manifest.get("control_model_sha256") != CONTROL_SHA
        or manifest.get("candidate_model_sha256") != CANDIDATE_SHA
    ):
        raise ValueError("N17 arm provenance changed")
    arms = list(manifest.get("arms") or ())
    if tuple(item.get("name") for item in arms) != ORDER:
        raise ValueError("N17 arm order changed")
    for item in arms:
        if digest(Path(str(item["path"])).resolve()) != item.get("sha256"):
            raise ValueError(f"N17 model changed: {item['name']}")
    return arms


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N17 runtime verifier accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N17 parity output: {OUTPUT}")
    checks = {
        ROOT / "scripts/collect_rocket_blueprint_n5_panel.py": HELPER_SHA,
        ROOT / "scripts/train_rank1_second_seat_ppo.py": WORKER_SHA,
        EVALUATOR: EVALUATOR_SHA,
        BUNDLE / "deck.csv": DECK_SHA,
        BUNDLE / "model.pt": CANDIDATE_SHA,
        BUNDLE / "soldier/net_v10.py": RUNTIME_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N17 parity input changed: {path}")
    if tree_sha256(BUNDLE) != BUNDLE_SHA:
        raise ValueError("N17 candidate bundle changed")
    for path in OPPONENTS:
        if tree_sha256(path) != OPPONENT_TREE_SHA[path.name]:
            raise ValueError(f"N17 parity opponent changed: {path.name}")
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
                        models[arm], greedy=True, deck_ids=deck,
                        forbid_end_when_attack=True, opponent_deck_priors=priors,
                    )
                    observation, start = battle_start(opponent.deck, deck)
                    if observation is None:
                        raise RuntimeError(
                            f"N17 parity start failed: {start.errorPlayer}/{start.errorType}"
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
                                        f"N17 runtime mismatch {arm}: "
                                        f"{strict_action}/{packaged_action}"
                                    )
                                action = strict_action
                                counts[f"{arm}:decisions"] += 1
                            elif actor == 0:
                                action = opponent.act(observation)
                            else:
                                raise RuntimeError(f"unexpected N17 actor: {actor}")
                            if not legal(list(action), selection):
                                raise RuntimeError(f"illegal N17 parity action: {action}")
                            observation = battle_select(list(action))
                        else:
                            raise RuntimeError("N17 parity game exceeded decision limit")
                    finally:
                        battle_finish()
                    counts[f"{arm}:games"] += 1
            finally:
                opponent.close()
    total = sum(counts[f"{arm}:decisions"] for arm in ORDER)
    if total < MINIMUM_DECISIONS:
        raise ValueError(f"N17 parity decision floor not met: {total}")
    report = {
        "schema": "rocket-n17-runtime-parity/v1",
        "protocol": "rocket-n17-replicated-target-residual/v1",
        "official_engine_execution": True,
        "candidate_bundle": str(BUNDLE.resolve()),
        "candidate_bundle_sha256": BUNDLE_SHA,
        "deck_sha256": DECK_SHA,
        "arms_manifest": str(ARMS.resolve()),
        "arms_manifest_sha256": ARMS_SHA,
        "verifier_sha256": digest(Path(__file__).resolve()),
        "helper_sha256": HELPER_SHA,
        "worker_sha256": WORKER_SHA,
        "evaluator_sha256": EVALUATOR_SHA,
        "seed": SEED,
        "games_per_opponent_per_arm": GAMES_PER_OPPONENT_ARM,
        "opponents": [str(path.resolve()) for path in OPPONENTS],
        "opponent_bundle_sha256": {
            str(path.resolve()): OPPONENT_TREE_SHA[path.name] for path in OPPONENTS
        },
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
