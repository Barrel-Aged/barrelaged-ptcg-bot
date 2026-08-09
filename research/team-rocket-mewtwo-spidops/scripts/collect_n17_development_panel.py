#!/usr/bin/env python3
"""Collect one locked N17 official-engine development shard."""

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
BUNDLE = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/development_arms/candidate_bundle"
ARMS = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/development_arms/arms.json"
PARITY = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/development_arms/runtime_parity.json"
PANEL = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/development_panel"
EVALUATOR = Path(r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py")
ORDER = ("control", "candidate")
SEED_OPPONENT = {
    817101: "AshMarnieGeneralV110_extracted",
    817102: "AshMarnieGeneralV110_extracted",
    817201: "HelbrosMarnieGeneralRecoveryV4_extracted",
    817202: "HelbrosMarnieGeneralRecoveryV4_extracted",
}
SEED_OUTPUT = {
    817101: "v110_01.jsonl",
    817102: "v110_02.jsonl",
    817201: "recovery_01.jsonl",
    817202: "recovery_02.jsonl",
}
OPPONENT_HASHES = {
    "AshMarnieGeneralV110_extracted": {
        "opponent_deck_sha256": "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b",
        "opponent_model_sha256": "552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b",
        "opponent_main_sha256": "8c87db1a011acf3cc5a614159d1ac6b6bf56aa7286f0f8a7148bb8e2aa20a6d1",
        "opponent_bundle_sha256": "3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b",
    },
    "HelbrosMarnieGeneralRecoveryV4_extracted": {
        "opponent_deck_sha256": "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b",
        "opponent_model_sha256": "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c",
        "opponent_main_sha256": "d077ff28d8a267bf3f2c28fb26e0ae798dedeac5c1d9274627cc8c9e45243533",
        "opponent_bundle_sha256": "caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4",
    },
}

ARMS_SHA = "28ac3ef5c0748f9b28faef4e9d071f809855c9d78fcdd951ea1f4aa38132b8bc"
BUNDLE_SHA = "a7d02c5bc39428f570317c704893a103c6f1225491f1b72c84312b980ef50dd3"
PARITY_SHA = "c913449822c8ed573fe78b371ca237b564e6402424c4486a156bdd4069f719cf"
HELPER_SHA = "f3cb21219efbf82c94a1bebe1d90b712fcd02cb28037397a02176d2ce5047f6b"
WORKER_SHA = "a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85"
EVALUATOR_SHA = "d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
GAMES_PER_ARM = 120


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arms() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if digest(ARMS) != ARMS_SHA:
        raise ValueError("N17 arms manifest changed")
    manifest = json.loads(ARMS.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "rocket-n17-development-arms/v1"
        or manifest.get("protocol") != "rocket-n17-replicated-target-residual/v1"
        or manifest.get("deck_sha256") != DECK_SHA
        or manifest.get("control_model_sha256") != CONTROL_SHA
        or manifest.get("candidate_model_sha256") != CANDIDATE_SHA
        or manifest.get("candidate_bundle_sha256") != BUNDLE_SHA
    ):
        raise ValueError("N17 arm provenance changed")
    arms = []
    for item in manifest.get("arms") or ():
        model = Path(str(item["path"])).resolve()
        if digest(model) != item.get("sha256"):
            raise ValueError(f"N17 arm changed: {item.get('name')}")
        arms.append({"name": str(item["name"]), "path": model, "sha256": str(item["sha256"])})
    if tuple(item["name"] for item in arms) != ORDER:
        raise ValueError("N17 arm order changed")
    return manifest, arms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed not in SEED_OPPONENT:
        raise ValueError("N17 panel seed differs from lock")
    opponent_path = args.opponent.resolve()
    output = args.output.resolve()
    if opponent_path.name != SEED_OPPONENT[args.seed]:
        raise ValueError("N17 seed assigned to wrong opponent")
    if output != (PANEL / SEED_OUTPUT[args.seed]).resolve():
        raise ValueError("N17 seed assigned to wrong output")
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing N17 shard: {output}")
    checks = {
        ROOT / "scripts/collect_rocket_blueprint_n5_panel.py": HELPER_SHA,
        ROOT / "scripts/train_rank1_second_seat_ppo.py": WORKER_SHA,
        EVALUATOR: EVALUATOR_SHA,
        PARITY: PARITY_SHA,
        BUNDLE / "deck.csv": DECK_SHA,
        BUNDLE / "model.pt": CANDIDATE_SHA,
        BUNDLE / "soldier/net_v10.py": RUNTIME_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N17 collector input changed: {path}")
    if tree_sha256(BUNDLE) != BUNDLE_SHA:
        raise ValueError("N17 candidate bundle changed")
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if int(parity["total_mismatched_decisions"]) != 0 or int(parity["total_matched_decisions"]) < 600:
        raise ValueError("N17 packaged runtime parity failed")
    arm_manifest, arms = load_arms()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BUNDLE))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = read_deck(BUNDLE)
    priors = json.loads((BUNDLE / "deck_library.json").read_text(encoding="utf-8"))
    models = {item["name"]: load_policy(item["path"], map_location="cpu").cpu().eval() for item in arms}
    fingerprint = {
        "opponent_deck_sha256": sha256(opponent_path / "deck.csv"),
        "opponent_model_sha256": optional_sha256(opponent_path / "model.pt"),
        "opponent_main_sha256": sha256(opponent_path / "main.py"),
        "opponent_bundle_sha256": tree_sha256(opponent_path),
    }
    if fingerprint != OPPONENT_HASHES[opponent_path.name]:
        raise ValueError(f"N17 opponent bundle changed: {opponent_path.name}")
    opponent = OpponentWorker(opponent_path, EVALUATOR, 30.0)
    schedule = [item["name"] for item in arms for _ in range(GAMES_PER_ARM)]
    random.Random(args.seed).shuffle(schedule)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    started = time()
    try:
        with output.open("x", encoding="utf-8") as sink:
            for game_index, arm in enumerate(schedule, start=1):
                row = play_game(
                    (battle_start, battle_select, battle_finish), deck, models[arm],
                    priors, opponent, 1, 2_000,
                )
                arm_row = next(item for item in arms if item["name"] == arm)
                row.update({
                    "schema": "rocket-n17-development-panel-match/v1",
                    "protocol": "rocket-n17-replicated-target-residual/v1",
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
                if game_index == 1 or game_index % 40 == 0 or game_index == len(schedule):
                    print(
                        json.dumps(
                            {"completed": game_index, "target": len(schedule), "seed": args.seed},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        opponent.close()
    manifest = {
        "schema": "rocket-n17-development-panel-shard/v1",
        "protocol": "rocket-n17-replicated-target-residual/v1",
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
        "strict_no_fallback_execution": True,
        "collector_sha256": digest(Path(__file__).resolve()),
        "collector_helper_sha256": HELPER_SHA,
        "opponent_worker_sha256": WORKER_SHA,
        "evaluator_sha256": EVALUATOR_SHA,
        "candidate_bundle": str(BUNDLE.resolve()),
        "candidate_bundle_sha256": BUNDLE_SHA,
        "deck_sha256": DECK_SHA,
        "control_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "runtime_parity_sha256": PARITY_SHA,
        "arms_manifest": str(ARMS.resolve()),
        "arms_manifest_sha256": ARMS_SHA,
        "arms": arm_manifest["arms"],
        "opponent": str(opponent_path),
        **fingerprint,
        "schedule_seed": args.seed,
        "own_seat": 1,
        "games_per_arm": GAMES_PER_ARM,
        "rows": len(schedule),
        "counts": dict(counts),
        "wins": dict(wins),
        "output_sha256": digest(output),
        "elapsed_seconds": time() - started,
        "deployment_authority": False,
    }
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"complete": True, "output": str(output), "counts": dict(counts)},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
