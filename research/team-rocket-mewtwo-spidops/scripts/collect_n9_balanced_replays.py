#!/usr/bin/env python3
"""Collect one locked shard of exact-control N9 arena-compatible replays."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from time import perf_counter, time

import numpy as np
import torch

from collect_rocket_blueprint_n5_panel import StrictGreedyAgent, tree_sha256
from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2"
)
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
BOT = "N9RocketControl"
LOCKED_SEEDS = {797101, 797102, 797201, 797202}
SEED_OPPONENT = {
    797101: "AshMarnieGeneralV110_extracted",
    797102: "AshMarnieGeneralV110_extracted",
    797201: "HelbrosMarnieGeneralRecoveryV4_extracted",
    797202: "HelbrosMarnieGeneralRecoveryV4_extracted",
}
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
OPPONENT_MODEL_SHA256 = {
    "AshMarnieGeneralV110_extracted": "552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b",
    "HelbrosMarnieGeneralRecoveryV4_extracted": "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c",
}
OPPONENT_DECK_SHA256 = "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_tree_digest(paths: list[Path]) -> str:
    result = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        name = path.name.encode("utf-8")
        result.update(len(name).to_bytes(4, "big"))
        result.update(name)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                result.update(chunk)
    return result.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed not in LOCKED_SEEDS or args.games != 300:
        raise ValueError("N9 seed/game count differs from protocol")
    candidate = CANDIDATE.resolve()
    opponent_path = args.opponent.resolve()
    if opponent_path.name != SEED_OPPONENT[args.seed]:
        raise ValueError("N9 seed assigned to a different opponent")
    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing existing N9 shard: {output}")
    if sha256(candidate / "model.pt") != MODEL_SHA256 or sha256(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("N9 frozen Rocket substrate changed")
    if sha256(opponent_path / "model.pt") != OPPONENT_MODEL_SHA256[opponent_path.name]:
        raise ValueError("N9 opponent model changed")
    if sha256(opponent_path / "deck.csv") != OPPONENT_DECK_SHA256:
        raise ValueError("N9 opponent deck changed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(candidate))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = read_deck(candidate)
    priors = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    model = load_policy(candidate / "model.pt", map_location="cpu").cpu().eval()
    opponent = OpponentWorker(opponent_path, EVALUATOR, args.action_timeout_seconds)
    replay_paths: list[Path] = []
    game_rows = []
    wins = losses = draws = decisions = 0
    started = time()
    try:
        for game_index in range(1, args.games + 1):
            opponent.reset()
            agent = StrictGreedyAgent(model, deck, priors)
            observation, start = battle_start(opponent.deck, deck)
            if observation is None:
                raise RuntimeError(f"N9 battle_start failed: {start.errorPlayer}/{start.errorType}")
            name = f"n9_{opponent_path.name}_{args.seed}_{game_index:04d}.jsonl"
            final_path = output / name
            temporary = output / f".{name}.partial"
            if final_path.exists() or temporary.exists():
                raise FileExistsError(f"N9 replay collision: {final_path}")
            first_player: int | None = None
            result = -1
            game_decisions = 0
            game_started = perf_counter()
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as sink:
                    meta = {
                        "meta": {
                            "schema": "rocket-n9-exact-replay/v1",
                            "protocol": "rocket-balanced-exact-pcgrad-awr/v1",
                            "players": [opponent_path.name, BOT],
                            "decks": [opponent.deck, deck],
                            "seed": args.seed,
                            "game_index": game_index,
                        }
                    }
                    sink.write(json.dumps(meta, separators=(",", ":"), sort_keys=True) + "\n")
                    for decision_index in range(args.max_decisions):
                        current = observation.get("current") or {}
                        observed_first = current.get("firstPlayer")
                        if observed_first in (0, 1):
                            first_player = int(observed_first)
                            if first_player != 0:
                                raise RuntimeError("N9 official engine changed first-player seat")
                        result = int(current.get("result") if current.get("result") is not None else -1)
                        if result != -1:
                            game_decisions = decision_index
                            break
                        selection = observation.get("select")
                        if not isinstance(selection, dict):
                            raise RuntimeError("N9 official engine emitted no live selection")
                        actor = int(current.get("yourIndex") if current.get("yourIndex") is not None else -1)
                        if actor == 1:
                            action = agent(observation)
                        elif actor == 0:
                            action = opponent.act(dict(observation))
                        else:
                            raise RuntimeError(f"unexpected N9 acting player: {actor}")
                        if not legal(list(action), selection):
                            raise RuntimeError(f"illegal N9 action actor={actor}: {action}")
                        sink.write(
                            json.dumps(
                                {"p": actor, "obs": deepcopy(observation), "action": list(action)},
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        observation = battle_select(list(action))
                    else:
                        raise RuntimeError("N9 game exceeded decision limit")
                    if first_player != 0 or result not in (0, 1, 2):
                        raise RuntimeError("N9 terminal/seat marker was not verified")
                    winner = result if result in (0, 1) else None
                    sink.write(
                        json.dumps(
                            {
                                "result": {
                                    "winner": winner,
                                    "reason": "result",
                                    "steps": game_decisions,
                                    "seed": args.seed,
                                    "game_index": game_index,
                                }
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    )
                os.replace(temporary, final_path)
            finally:
                battle_finish()
                if temporary.exists():
                    temporary.unlink()
            replay_paths.append(final_path)
            decisions += game_decisions
            wins += int(result == 1)
            losses += int(result == 0)
            draws += int(result == 2)
            game_rows.append(
                {
                    "file": name,
                    "sha256": digest(final_path),
                    "result": result,
                    "decisions": game_decisions,
                    "duration_seconds": perf_counter() - game_started,
                }
            )
            if game_index == 1 or game_index % 25 == 0 or game_index == args.games:
                print(
                    json.dumps(
                        {
                            "opponent": opponent_path.name,
                            "seed": args.seed,
                            "completed": game_index,
                            "target": args.games,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        opponent.close()
    manifest = {
        "schema": "rocket-n9-exact-replay-shard/v1",
        "protocol": "rocket-balanced-exact-pcgrad-awr/v1",
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
        "strict_no_fallback_execution": True,
        "collector_sha256": digest(Path(__file__).resolve()),
        "candidate": str(candidate),
        "candidate_bundle_sha256": tree_sha256(candidate),
        "recorded_bot": BOT,
        "model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "opponent": str(opponent_path),
        "opponent_bundle_sha256": tree_sha256(opponent_path),
        "opponent_model_sha256": OPPONENT_MODEL_SHA256[opponent_path.name],
        "opponent_deck_sha256": OPPONENT_DECK_SHA256,
        "seed": args.seed,
        "games": len(replay_paths),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisions": decisions,
        "replay_tree_sha256": replay_tree_digest(replay_paths),
        "replays": game_rows,
        "elapsed_seconds": time() - started,
        "development_only": True,
        "deployment_authority": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "games": len(replay_paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
