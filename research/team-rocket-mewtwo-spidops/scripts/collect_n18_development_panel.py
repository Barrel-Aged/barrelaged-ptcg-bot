#!/usr/bin/env python3
"""Collect one locked N18 packaged-agent official-engine development shard."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter, time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "development_arms/arms.json"
)
RUNTIME_AUDIT = ARMS.parent / "runtime_audit.json"
PANEL = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "development_panel"
)
PROTOCOL = ROOT / "N18_CROSS_DECK_LEADER_TRANSFER_PROTOCOL.md"
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)

ARMS_SHA = "57ab74eaaf05aa2f4e36a09f1555aef4eadf1d84dd0415c59376de036bfbed18"
RUNTIME_AUDIT_SHA = "93c2e64764108ee78415b13ce3de3bf6e0c7bc8256b44804a002081e802f0b93"
PROTOCOL_SHA = "b64bd56fb0c1593f2310b636723c83c7c08596f5dbb63c3d9e848015b3c25805"
WORKER_SHA = "a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85"
HELPER_SHA = "f3cb21219efbf82c94a1bebe1d90b712fcd02cb28037397a02176d2ce5047f6b"
EVALUATOR_SHA = "d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"

MATCH_ARMS = ("control", "live_residual", "leader_base", "leader_residual")
GAMES_PER_ARM = 120
MAX_DECISIONS = 2_000
ACTION_TIMEOUT_SECONDS = 30.0

LEADER = ROOT / "work/n18/bam"
V110 = ROOT / "work/arena_bundles/AshMarnieGeneralV110_extracted"
RECOVERY = ROOT / "work/arena_bundles/HelbrosMarnieGeneralRecoveryV4_extracted"

SEED_CONFIG = {
    818101: (LEADER, 0, "leader_seat0.jsonl"),
    818102: (LEADER, 1, "leader_seat1.jsonl"),
    818201: (V110, 0, "v110_seat0.jsonl"),
    818202: (V110, 1, "v110_seat1.jsonl"),
    818301: (RECOVERY, 0, "recovery_seat0.jsonl"),
    818302: (RECOVERY, 1, "recovery_seat1.jsonl"),
}

OPPONENT_FINGERPRINTS = {
    str(LEADER.resolve()): {
        "opponent_tree_sha256": "1a78564bb3304fc4c968e14cb01245eba2c232fe32594c0cfdb563a01a76e828",
        "opponent_deck_sha256": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
        "opponent_model_sha256": "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c",
        "opponent_main_sha256": "0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b",
        "opponent_residual_sha256": "5df9bd6e9f2d733092b0653765b4d1d3f082a488cfe68fe31f23c94f3aa43f4f",
    },
    str(V110.resolve()): {
        "opponent_tree_sha256": "3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b",
        "opponent_deck_sha256": "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b",
        "opponent_model_sha256": "552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b",
        "opponent_main_sha256": "8c87db1a011acf3cc5a614159d1ac6b6bf56aa7286f0f8a7148bb8e2aa20a6d1",
        "opponent_residual_sha256": None,
    },
    str(RECOVERY.resolve()): {
        "opponent_tree_sha256": "caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4",
        "opponent_deck_sha256": "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b",
        "opponent_model_sha256": "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c",
        "opponent_main_sha256": "d077ff28d8a267bf3f2c28fb26e0ae798dedeac5c1d9274627cc8c9e45243533",
        "opponent_residual_sha256": "25ee55a36d35d771997de61e61f71395f882178668317b8b5ffee4ccd4ff181c",
    },
}


sys.path.insert(0, str((ROOT / "scripts").resolve()))
from collect_rocket_blueprint_n5_panel import tree_sha256  # noqa: E402
from train_rank1_second_seat_ppo import (  # noqa: E402
    OpponentWorker,
    legal,
    read_deck,
    sha256,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_digest(path: Path) -> str | None:
    return digest(path) if path.exists() else None


def fingerprint(bundle: Path) -> dict[str, Any]:
    return {
        "opponent_tree_sha256": tree_sha256(bundle),
        "opponent_deck_sha256": digest(bundle / "deck.csv"),
        "opponent_model_sha256": digest(bundle / "model.pt"),
        "opponent_main_sha256": digest(bundle / "main.py"),
        "opponent_residual_sha256": optional_digest(bundle / "history_residual.pt"),
    }


def play_game(
    engine: tuple[Any, Any, Any],
    own: OpponentWorker,
    opponent: OpponentWorker,
    own_seat: int,
) -> dict[str, Any]:
    battle_start, battle_select, battle_finish = engine
    own.reset()
    opponent.reset()
    player_decks = (own.deck, opponent.deck) if own_seat == 0 else (opponent.deck, own.deck)
    observation, start = battle_start(*player_decks)
    if observation is None:
        raise RuntimeError(
            f"N18 official engine start failed: {start.errorPlayer}/{start.errorType}"
        )
    started = perf_counter()
    first_player: int | None = None
    own_decisions = 0
    opponent_decisions = 0
    try:
        for decision in range(MAX_DECISIONS):
            current = observation.get("current") or {}
            observed_first = current.get("firstPlayer")
            if observed_first in (0, 1):
                first_player = int(observed_first)
                if first_player != 0:
                    raise RuntimeError("N18 official engine changed the first-player convention")
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                if first_player != 0:
                    raise RuntimeError("N18 match ended without first-player verification")
                return {
                    "result": result,
                    "win": int(result == own_seat),
                    "draw": int(result == 2),
                    "first_player": first_player,
                    "own_seat": own_seat,
                    "started_first": own_seat == 0,
                    "started_second": own_seat == 1,
                    "decisions": decision,
                    "own_decisions": own_decisions,
                    "opponent_decisions": opponent_decisions,
                    "duration_seconds": perf_counter() - started,
                    "illegal_actions": 0,
                    "exception_fallbacks": 0,
                }
            selection = observation.get("select")
            if not isinstance(selection, dict):
                raise RuntimeError("N18 official engine emitted no live selection")
            actor = int(current.get("yourIndex") if current.get("yourIndex") is not None else -1)
            if actor == own_seat:
                action = own.act(dict(observation))
                own_decisions += 1
            elif actor == 1 - own_seat:
                action = opponent.act(dict(observation))
                opponent_decisions += 1
            else:
                raise RuntimeError(f"N18 official engine emitted invalid acting player: {actor}")
            if not legal(action, selection):
                raise RuntimeError(f"N18 packaged agent returned illegal action: {action}")
            observation = battle_select(list(action))
        raise RuntimeError(f"N18 game exceeded {MAX_DECISIONS} decisions")
    finally:
        battle_finish()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.seed not in SEED_CONFIG:
        raise ValueError("N18 development seed is not preregistered")
    opponent_path, own_seat, output_name = SEED_CONFIG[args.seed]
    output = (PANEL / output_name).resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing N18 shard: {output}")

    checks = {
        ARMS: ARMS_SHA,
        RUNTIME_AUDIT: RUNTIME_AUDIT_SHA,
        PROTOCOL: PROTOCOL_SHA,
        ROOT / "scripts/train_rank1_second_seat_ppo.py": WORKER_SHA,
        ROOT / "scripts/collect_rocket_blueprint_n5_panel.py": HELPER_SHA,
        EVALUATOR: EVALUATOR_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N18 frozen collector input changed: {path}")

    manifest = json.loads(ARMS.read_text(encoding="utf-8"))
    all_arms = {item["name"]: item for item in manifest.get("arms") or ()}
    if any(name not in all_arms for name in MATCH_ARMS):
        raise ValueError("N18 match arm missing")
    match_arms = [all_arms[name] for name in MATCH_ARMS]
    for item in all_arms.values():
        bundle = Path(item["bundle"])
        if tree_sha256(bundle) != item["bundle_tree_sha256"]:
            raise ValueError(f"N18 arm tree changed: {item['name']}")
        if sha256(bundle / "deck.csv") != DECK_SHA:
            raise ValueError(f"N18 arm deck changed: {item['name']}")
    control_bundle = Path(all_arms["control"]["bundle"])

    expected_opponent = OPPONENT_FINGERPRINTS[str(opponent_path.resolve())]
    actual_opponent = fingerprint(opponent_path)
    if actual_opponent != expected_opponent:
        raise ValueError(f"N18 opponent changed: {opponent_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    sys.path.insert(0, str(control_bundle.resolve()))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore  # noqa: E402

    engine = (battle_start, battle_select, battle_finish)
    own_workers: dict[str, OpponentWorker] = {}
    opponent = OpponentWorker(opponent_path, EVALUATOR, ACTION_TIMEOUT_SECONDS)
    schedule = [name for name in MATCH_ARMS for _ in range(GAMES_PER_ARM)]
    random.Random(args.seed).shuffle(schedule)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    draws: Counter[str] = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time()
    try:
        for item in match_arms:
            own_workers[item["name"]] = OpponentWorker(
                Path(item["bundle"]), EVALUATOR, ACTION_TIMEOUT_SECONDS
            )
        with output.open("x", encoding="utf-8") as sink:
            for game_index, arm_name in enumerate(schedule, start=1):
                row = play_game(engine, own_workers[arm_name], opponent, own_seat)
                arm = all_arms[arm_name]
                row.update(
                    {
                        "schema": "rocket-n18-development-match/v1",
                        "protocol": "rocket-n18-cross-deck-leader-transfer/v1",
                        "game_index": game_index,
                        "schedule_seed": args.seed,
                        "arm": arm_name,
                        "arm_bundle_tree_sha256": arm["bundle_tree_sha256"],
                        "arm_model_sha256": arm["model_sha256"],
                        "arm_main_sha256": arm["main_sha256"],
                        "arm_history_residual_sha256": arm["history_residual_sha256"],
                        "exact_team_rocket_deck_sha256": DECK_SHA,
                        "opponent": opponent_path.name,
                        **actual_opponent,
                    }
                )
                sink.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                sink.flush()
                counts[arm_name] += 1
                wins[arm_name] += int(row["win"])
                draws[arm_name] += int(row["draw"])
                if game_index == 1 or game_index % 40 == 0 or game_index == len(schedule):
                    print(
                        json.dumps(
                            {
                                "completed": game_index,
                                "target": len(schedule),
                                "seed": args.seed,
                                "own_seat": own_seat,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    finally:
        opponent.close()
        for worker in own_workers.values():
            worker.close()

    manifest_data = {
        "schema": "rocket-n18-development-shard/v1",
        "protocol": "rocket-n18-cross-deck-leader-transfer/v1",
        "official_engine_execution": True,
        "canonical_engine_bundle": str(control_bundle.resolve()),
        "canonical_engine_bundle_tree_sha256": all_arms["control"]["bundle_tree_sha256"],
        "packaged_worker_execution": True,
        "strict_legality": True,
        "hidden_information_used_by_actor": False,
        "collector_sha256": digest(Path(__file__).resolve()),
        "worker_sha256": WORKER_SHA,
        "evaluator_sha256": EVALUATOR_SHA,
        "protocol_sha256": PROTOCOL_SHA,
        "arms_manifest_sha256": ARMS_SHA,
        "runtime_audit_sha256": RUNTIME_AUDIT_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "arms": match_arms,
        "opponent": str(opponent_path.resolve()),
        **actual_opponent,
        "schedule_seed": args.seed,
        "own_seat": own_seat,
        "games_per_arm": GAMES_PER_ARM,
        "rows": len(schedule),
        "counts": dict(counts),
        "wins": dict(wins),
        "draws": dict(draws),
        "output": str(output),
        "output_sha256": digest(output),
        "elapsed_seconds": time() - started,
        "development_authority": False,
        "deployment_authority": False,
    }
    sidecar.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "output": str(output),
                "rows": len(schedule),
                "counts": dict(counts),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
