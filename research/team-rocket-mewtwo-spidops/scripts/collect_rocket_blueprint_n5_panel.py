#!/usr/bin/env python3
"""Collect a blinded, interleaved N5 greedy-policy panel in the official engine."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from time import perf_counter, time
from typing import Any, Mapping

import numpy as np
import torch

from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
)
DEFAULT_ARMS = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "development_arms" / "arms.json"
)
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
LOCKED_ARMS = (
    "control",
    "seed_788101",
    "seed_788102",
    "seed_788103",
    "seed_788104",
    "mean_all4",
)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "development", "validation", "regression"), required=True)
    parser.add_argument("--own-seat", type=int, choices=(0, 1), default=1)
    parser.add_argument("--games-per-arm", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def optional_sha256(path: Path) -> str | None:
    return sha256(path) if path.exists() else None


def tree_sha256(root: Path) -> str:
    """Hash immutable bundle contents while excluding interpreter caches."""
    import hashlib

    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix.lower() not in (".pyc", ".pyo")
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_arms(path: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    allowed_schemas = {"rocket-blueprint-n5-development-arms/v1"}
    if split in ("validation", "regression"):
        allowed_schemas.add("rocket-blueprint-n5-validation-arms/v1")
    if manifest.get("schema") not in allowed_schemas:
        raise ValueError("unexpected N5 arm manifest")
    if manifest.get("protocol") != "rocket-greedy-compatible-ppo/v1":
        raise ValueError("N5 arm protocol mismatch")
    if manifest.get("deck_sha256") != DECK_SHA256 or manifest.get("initial_model_sha256") != MODEL_SHA256:
        raise ValueError("N5 frozen provenance mismatch")
    arms: list[dict[str, Any]] = []
    for item in manifest.get("arms") or ():
        source = Path(str(item["path"])).resolve()
        if not source.exists() or sha256(source) != str(item["sha256"]):
            raise ValueError(f"N5 arm missing or changed: {source}")
        arms.append({"name": str(item["name"]), "path": source, "sha256": str(item["sha256"])})
    names = tuple(item["name"] for item in arms)
    if len(set(names)) != len(names) or "control" not in names:
        raise ValueError(f"invalid N5 arm names: {names}")
    if split == "development" and names != LOCKED_ARMS:
        raise ValueError(f"development arm order differs from lock: {names}")
    if split in ("validation", "regression") and names != ("control", "candidate"):
        raise ValueError(f"validation arm order differs from lock: {names}")
    return manifest, arms


class StrictGreedyAgent:
    """The packaged NNAgent greedy path, with exceptions exposed instead of hidden."""

    def __init__(self, model: torch.nn.Module, deck: list[int], priors: dict[str, Any] | None) -> None:
        from soldier.belief import Belief

        self.model = model
        self.belief = Belief(deck, opponent_deck_priors=priors)

    @torch.no_grad()
    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        from soldier import features
        from soldier.net import attack_safe_logits, feat_to_tensors, select_autoregressive

        selection = observation["select"]
        options = selection["option"]
        count = len(options)
        maximum = min(int(selection["maxCount"]), count)
        if maximum == 0:
            return []
        base = features.extract(
            observation,
            belief=self.belief,
            slot_link=bool(getattr(self.model, "slot_link", False)),
        )
        tensors = feat_to_tensors(base, "cpu")
        logits, _ = self.model(tensors)
        if maximum == 1:
            logits = attack_safe_logits(logits, options)
            return [int(torch.argmax(logits))]
        return list(select_autoregressive(self.model, base, maximum, sample=False, device="cpu"))


def play_game(
    engine: tuple[Any, Any, Any],
    deck: list[int],
    model: torch.nn.Module,
    priors: dict[str, Any] | None,
    opponent: OpponentWorker,
    own_seat: int,
    max_decisions: int,
) -> dict[str, Any]:
    battle_start, battle_select, battle_finish = engine
    opponent.reset()
    agent = StrictGreedyAgent(model, deck, priors)
    player_decks = (deck, opponent.deck) if own_seat == 0 else (opponent.deck, deck)
    observation, start = battle_start(*player_decks)
    if observation is None:
        raise RuntimeError(f"official engine start failed: {start.errorPlayer}/{start.errorType}")
    started = perf_counter()
    own_decisions = 0
    first_player: int | None = None
    try:
        for decision in range(max_decisions):
            current = observation.get("current") or {}
            observed_first = current.get("firstPlayer")
            if observed_first in (0, 1):
                first_player = int(observed_first)
                if first_player != 0:
                    raise RuntimeError("N5 official engine did not preserve locked first seat")
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                if first_player != 0:
                    raise RuntimeError("N5 game ended without a verified first-player marker")
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
                    "duration_seconds": perf_counter() - started,
                    "exception_fallbacks": 0,
                }
            selection = observation.get("select")
            if not isinstance(selection, dict):
                raise RuntimeError("official engine emitted no live selection")
            actor = int(current.get("yourIndex") if current.get("yourIndex") is not None else -1)
            if actor == own_seat:
                action = agent(observation)
                own_decisions += 1
            elif actor == 1 - own_seat:
                action = opponent.act(dict(observation))
            else:
                raise RuntimeError(f"unexpected N5 acting player: {actor}")
            if not legal(list(action), selection):
                raise RuntimeError(f"illegal N5 panel action actor={actor}: {action}")
            observation = battle_select(list(action))
        raise RuntimeError(f"N5 game exceeded {max_decisions} decisions")
    finally:
        battle_finish()


def main() -> int:
    args = parse_args()
    if args.games_per_arm <= 0 or args.max_decisions <= 0:
        raise ValueError("N5 panel counts must be positive")
    if args.split in ("development", "validation") and args.own_seat != 1:
        raise ValueError("locked N5 development/validation requires Rocket in seat 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    opponent_path = args.opponent.resolve()
    output = args.output.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"refusing existing N5 panel output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    arms_path = args.arms.resolve()
    arm_manifest, arms = load_arms(arms_path, args.split)
    if sha256(candidate / "deck.csv") != DECK_SHA256 or sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("N5 frozen candidate changed")
    candidate_bundle_sha256 = tree_sha256(candidate)

    sys.path.insert(0, str(candidate))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = read_deck(candidate)
    library_path = candidate / "deck_library.json"
    priors = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else None
    models = {item["name"]: load_policy(item["path"], map_location="cpu").cpu().eval() for item in arms}
    opponent_fingerprint = {
        "opponent_deck_sha256": sha256(opponent_path / "deck.csv"),
        "opponent_model_sha256": optional_sha256(opponent_path / "model.pt"),
        "opponent_main_sha256": sha256(opponent_path / "main.py"),
        "opponent_bundle_sha256": tree_sha256(opponent_path),
    }
    opponent = OpponentWorker(opponent_path, args.evaluator.resolve(), args.action_timeout_seconds)
    schedule = [item["name"] for item in arms for _ in range(args.games_per_arm)]
    random.Random(args.seed).shuffle(schedule)
    counts: Counter[str] = Counter()
    wins: Counter[str] = Counter()
    started = time()
    engine = (battle_start, battle_select, battle_finish)
    try:
        with output.open("x", encoding="utf-8") as sink:
            for game_index, arm_name in enumerate(schedule, start=1):
                row = play_game(engine, deck, models[arm_name], priors, opponent, args.own_seat, args.max_decisions)
                arm = next(item for item in arms if item["name"] == arm_name)
                row.update(
                    {
                        "schema": "rocket-blueprint-n5-panel-match/v1",
                        "protocol": "rocket-greedy-compatible-ppo/v1",
                        "split": args.split,
                        "game_index": game_index,
                        "schedule_seed": args.seed,
                        "arm": arm_name,
                        "arm_sha256": arm["sha256"],
                        "opponent": opponent_path.name,
                        **opponent_fingerprint,
                    }
                )
                sink.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                sink.flush()
                counts[arm_name] += 1
                wins[arm_name] += int(row["win"])
                if game_index == 1 or game_index % 20 == 0 or game_index == len(schedule):
                    progress = {"completed": game_index, "target": len(schedule), "split": args.split}
                    if args.split in ("smoke", "regression"):
                        progress.update({"arm": arm_name, "arm_games": counts[arm_name]})
                    print(json.dumps(progress, sort_keys=True), flush=True)
    finally:
        opponent.close()

    manifest = {
        "schema": "rocket-blueprint-n5-panel-shard/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "split": args.split,
        "sealed": args.split == "validation",
        "official_engine_execution": True,
        "hidden_information_used_by_actor": False,
        "strict_no_fallback_execution": True,
        "candidate": str(candidate),
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "deck_sha256": DECK_SHA256,
        "initial_model_sha256": MODEL_SHA256,
        "arms_manifest": str(arms_path),
        "arms_manifest_sha256": sha256(arms_path),
        "arms": arm_manifest["arms"],
        "opponent": str(opponent_path),
        **opponent_fingerprint,
        "schedule_seed": args.seed,
        "own_seat": args.own_seat,
        "games_per_arm": args.games_per_arm,
        "rows": len(schedule),
        "counts": dict(counts),
        "output_sha256": sha256(output),
        "elapsed_seconds": time() - started,
        "claim_boundary": "official local-engine evidence; not an arena strength claim",
    }
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(output), "manifest": str(sidecar), "counts": dict(counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
