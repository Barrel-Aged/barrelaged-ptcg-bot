#!/usr/bin/env python3
"""Collect one locked Target-80 randomized-support replay shard.

The Rocket actor is the exact Wobbuffet-free Mimikyu/Maximum-Belt package:
both public hand-authored guards remain deterministic, while otherwise
plausible one-choice alternatives are selected by a fully logged behavior
mixture. Games execute in the bundled official engine. This collector never
uses opponent hidden cards and has no fallback path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter, time
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from collect_rocket_blueprint_n5_panel import tree_sha256
from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_ROOT / "TARGET80_RANDOMIZED_SUPPORT_ADVANTAGE_V1_PROTOCOL.md"
CANDIDATE = (
    PROJECT_ROOT
    / "work"
    / "target80_candidates"
    / "mimikyu_maxbelt_neural_control_v1"
)
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)

SCHEMA = "target80-randomized-support-replay/v1"
MANIFEST_SCHEMA = "target80-randomized-support-shard/v1"
PROTOCOL_NAME = "target80-randomized-support-advantage/v1"
BOT = "Target80MimikyuMaxBeltRandomSupportV1"
EPSILON = 0.15
MIN_REFERENCE_SUPPORT = 0.03

CANDIDATE_TREE_SHA256 = "30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f"
CANDIDATE_DECK_SHA256 = "4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea"
CANDIDATE_MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_MAIN_SHA256 = "80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6"

OPPONENTS = {
    "marnie": {
        "path": PROJECT_ROOT / "work" / "arena_bundles" / "AshV178MarnieFullBest_20260801_extracted",
        "tree": "49bb3353ab42fded18e32e38f154ad1a001d60c5852383045ff4f437be36efe1",
        "deck": "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d",
        "main": "0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b",
        "index": 1,
    },
    "starmie": {
        "path": PROJECT_ROOT / "work" / "arena_bundles" / "SparkStarmieFroslassCFV13_extracted",
        "tree": "fec4a911a29f18522cdc2dcd018831370760ee1d9452131f6ed4420b73c06a1e",
        "deck": "60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61",
        "main": "3d570642a9294bca0abea9587786b4a9059a42d1c777c462631e131cde24e87e",
        "index": 2,
    },
    "alakazam": {
        "path": PROJECT_ROOT / "work" / "arena_bundles_20260804" / "VioletAlakazamMKV93_extracted",
        "tree": "d24e5e225504383d01226d6c3a5d83ce7a388f2bfac4749b896bdc5a0e4daf9e",
        "deck": "4e56a20249fad14097d06aaee2a241f355b90ef9bef19307f95b47798d0a2a3d",
        "main": "c324c943d2dfe62d28c9a404e264e6d25eabeea296e37f9575ee1231a76bbec4",
        "index": 3,
    },
}

SPLIT_COUNTS = {"smoke": 2, "train": 120, "validation": 40, "test": 40}
SPLIT_PREFIX = {"smoke": 850, "train": 851, "validation": 852, "test": 853}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_tree_digest(paths: Sequence[Path]) -> str:
    result = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        name = path.name.encode("utf-8")
        result.update(len(name).to_bytes(4, "big"))
        result.update(name)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                result.update(chunk)
    return result.hexdigest()


def expected_seed(split: str, opponent_key: str, own_seat: int) -> int:
    if split == "smoke" and opponent_key != "marnie":
        raise ValueError("V1 smoke is locked to the Marnie opponent")
    return SPLIT_PREFIX[split] * 1000 + int(OPPONENTS[opponent_key]["index"]) * 100 + own_seat + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLIT_COUNTS), required=True)
    parser.add_argument("--opponent", choices=tuple(OPPONENTS), required=True)
    parser.add_argument("--own-seat", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def _load_candidate_main(candidate: Path) -> ModuleType:
    name = "target80_randomized_support_candidate_main"
    spec = importlib.util.spec_from_file_location(name, candidate / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not load frozen candidate main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _finite_json_floats(values: torch.Tensor) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values.detach().cpu().reshape(-1)]


def _one_hot(size: int, index: int) -> list[float]:
    result = [0.0] * size
    result[index] = 1.0
    return result


def hidden_hand_boundary(observation: Mapping[str, Any], actor: int) -> bool:
    current = observation.get("current") or {}
    players = current.get("players") or []
    return (
        actor in (0, 1)
        and len(players) == 2
        and isinstance(players[actor], Mapping)
        and isinstance(players[1 - actor], Mapping)
        and players[1 - actor].get("hand") is None
    )


class ExploratoryAgent:
    """Exact packaged control with logged, supported one-choice interventions."""

    def __init__(
        self,
        candidate_main: ModuleType,
        model: torch.nn.Module,
        deck: list[int],
        priors: dict[str, Any] | None,
        rng: random.Random,
    ) -> None:
        from soldier.belief import Belief

        self.candidate_main = candidate_main
        self.model = model
        self.belief = Belief(deck, opponent_deck_priors=priors)
        self.rng = rng
        self.last_decision: dict[str, Any] | None = None

    def _guard(self, observation: Mapping[str, Any]) -> tuple[str, list[int]] | None:
        for name, function_name in (
            ("early_spidops_guard", "_early_spidops_search_override"),
            ("bangle_lethal_guard", "_spidops_bangle_lethal_override"),
        ):
            action = getattr(self.candidate_main, function_name)(dict(observation))
            if action is not None:
                return name, [int(value) for value in action]
        return None

    @torch.no_grad()
    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        from soldier import features
        from soldier.net import attack_safe_logits, feat_to_tensors, select_autoregressive

        selection = observation.get("select") or {}
        options = selection.get("option") or []
        count = len(options)
        minimum = min(max(0, int(selection.get("minCount") or 0)), count)
        maximum = min(max(0, int(selection.get("maxCount") or 0)), count)
        selection_type = int(selection.get("type") if selection.get("type") is not None else -1)
        self.last_decision = None

        if maximum == 0:
            action: list[int] = []
            self.last_decision = {
                "schema": "target80-supported-behavior/v1",
                "reference_source": "empty",
                "selection_type": selection_type,
                "minimum": minimum,
                "maximum": maximum,
                "option_count": count,
                "chosen": action,
                "chosen_propensity": 1.0,
                "eligible": False,
                "explored": False,
                "attack_guard_applied": False,
                "reference_logits": [],
                "reference_probabilities": [],
                "behavior_probabilities": [],
                "reference_argmax": None,
                "supported_alternatives": [],
            }
            return action

        guarded = self._guard(observation)
        if guarded is not None:
            guard_name, action = guarded
            if len(action) != 1:
                raise RuntimeError(f"packaged guard returned non-single action: {action}")
            self.last_decision = {
                "schema": "target80-supported-behavior/v1",
                "reference_source": guard_name,
                "selection_type": selection_type,
                "minimum": minimum,
                "maximum": maximum,
                "option_count": count,
                "chosen": action,
                "chosen_propensity": 1.0,
                "eligible": False,
                "explored": False,
                "attack_guard_applied": False,
                "reference_logits": None,
                "reference_probabilities": None,
                "behavior_probabilities": _one_hot(count, action[0]),
                "reference_argmax": action[0],
                "supported_alternatives": [],
            }
            return action

        base = features.extract(
            observation,
            belief=self.belief,
            slot_link=bool(getattr(self.model, "slot_link", False)),
        )
        if maximum != 1:
            action = list(
                select_autoregressive(
                    self.model,
                    base,
                    maximum,
                    sample=False,
                    device="cpu",
                )
            )
            self.last_decision = {
                "schema": "target80-supported-behavior/v1",
                "reference_source": "neural_autoregressive_greedy",
                "selection_type": selection_type,
                "minimum": minimum,
                "maximum": maximum,
                "option_count": count,
                "chosen": action,
                "chosen_propensity": 1.0,
                "eligible": False,
                "explored": False,
                "attack_guard_applied": False,
                "reference_logits": None,
                "reference_probabilities": None,
                "behavior_probabilities": None,
                "reference_argmax": None,
                "supported_alternatives": [],
            }
            return action

        tensors = feat_to_tensors(base, "cpu")
        logits, _ = self.model(tensors)
        raw_logits = logits.detach().cpu().reshape(-1)
        if len(raw_logits) != count:
            raise RuntimeError(f"model emitted {len(raw_logits)} logits for {count} options")
        attack_present = any(option.get("type") == 13 for option in options)
        end_indices = [index for index, option in enumerate(options) if option.get("type") == 14]
        logits = attack_safe_logits(logits, options).detach().cpu().reshape(-1)
        finite = torch.isfinite(logits)
        if not bool(finite.any()):
            raise RuntimeError("no finite attack-safe action logits")
        probabilities = torch.zeros_like(logits, dtype=torch.float64)
        probabilities[finite] = torch.softmax(logits[finite].to(torch.float64), dim=0)
        baseline = int(torch.argmax(logits))
        supported = [
            index
            for index in range(count)
            if index != baseline
            and bool(finite[index])
            and float(probabilities[index]) >= MIN_REFERENCE_SUPPORT
        ]
        eligible = selection_type != 9 and bool(supported)
        behavior = [0.0] * count
        explored = False
        chosen = baseline
        if eligible:
            behavior[baseline] = 1.0 - EPSILON
            alternative_mass = sum(float(probabilities[index]) for index in supported)
            if not alternative_mass > 0.0:
                raise RuntimeError("supported alternatives have zero reference mass")
            for index in supported:
                behavior[index] = EPSILON * float(probabilities[index]) / alternative_mass
            if self.rng.random() < EPSILON:
                explored = True
                threshold = self.rng.random() * alternative_mass
                cumulative = 0.0
                chosen = supported[-1]
                for index in supported:
                    cumulative += float(probabilities[index])
                    if threshold <= cumulative:
                        chosen = index
                        break
        else:
            behavior[baseline] = 1.0
        attack_guard_applied = attack_present and bool(end_indices)
        action = [chosen]
        self.last_decision = {
            "schema": "target80-supported-behavior/v1",
            "reference_source": "neural_supported_mixture" if eligible else "neural_greedy",
            "selection_type": selection_type,
            "minimum": minimum,
            "maximum": maximum,
            "option_count": count,
            "chosen": action,
            "chosen_propensity": float(behavior[chosen]),
            "eligible": eligible,
            "explored": explored,
            "epsilon": EPSILON,
            "minimum_reference_support": MIN_REFERENCE_SUPPORT,
            "attack_guard_applied": attack_guard_applied,
            "reference_logits": _finite_json_floats(logits),
            "raw_reference_logits": _finite_json_floats(raw_logits),
            "reference_probabilities": [float(value) for value in probabilities],
            "behavior_probabilities": behavior,
            "reference_argmax": baseline,
            "supported_alternatives": supported,
        }
        return action


def audit_behavior_record(
    record: Mapping[str, Any], action: Sequence[int], selection: Mapping[str, Any]
) -> None:
    if list(record.get("chosen") or []) != list(action):
        raise RuntimeError("behavior record does not match applied action")
    if not legal(list(action), dict(selection)):
        raise RuntimeError("behavior record contains an illegal action")
    count = len(selection.get("option") or [])
    if int(record.get("option_count") or 0) != count:
        raise RuntimeError("behavior option count changed")
    propensity = float(record.get("chosen_propensity") or 0.0)
    if not math.isfinite(propensity) or not propensity > 0.0 or propensity > 1.0 + 1e-12:
        raise RuntimeError("invalid chosen propensity")
    behavior = record.get("behavior_probabilities")
    if behavior is not None:
        if len(behavior) != count or any(not math.isfinite(float(value)) or float(value) < 0 for value in behavior):
            raise RuntimeError("invalid behavior probability vector")
        if abs(sum(float(value) for value in behavior) - 1.0) > 1e-6:
            raise RuntimeError("behavior probabilities do not sum to one")
        if len(action) == 1 and abs(float(behavior[action[0]]) - propensity) > 1e-9:
            raise RuntimeError("chosen propensity differs from behavior vector")
    reference = record.get("reference_probabilities")
    if reference is not None:
        if len(reference) != count or any(not math.isfinite(float(value)) or float(value) < 0 for value in reference):
            raise RuntimeError("invalid reference probability vector")
        if abs(sum(float(value) for value in reference) - 1.0) > 1e-6:
            raise RuntimeError("reference probabilities do not sum to one")
        for index in record.get("supported_alternatives") or []:
            if float(reference[int(index)]) + 1e-12 < MIN_REFERENCE_SUPPORT:
                raise RuntimeError("unsupported alternative entered behavior mixture")
    if record.get("attack_guard_applied"):
        options = selection.get("option") or []
        if not any(option.get("type") == 13 for option in options):
            raise RuntimeError("attack guard logged without ATTACK")
        for index, option in enumerate(options):
            if option.get("type") == 14 and behavior is not None and float(behavior[index]) != 0.0:
                raise RuntimeError("END retained behavior support while ATTACK was legal")


def verify_frozen_inputs(opponent_key: str) -> tuple[Path, dict[str, Any]]:
    candidate = CANDIDATE.resolve()
    opponent_spec = OPPONENTS[opponent_key]
    opponent = Path(opponent_spec["path"]).resolve()
    required = (candidate, opponent, EVALUATOR.resolve(), PROTOCOL.resolve())
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    if tree_sha256(candidate) != CANDIDATE_TREE_SHA256:
        raise ValueError("frozen Rocket bundle tree changed")
    if sha256(candidate / "deck.csv") != CANDIDATE_DECK_SHA256:
        raise ValueError("frozen Rocket deck changed")
    if sha256(candidate / "model.pt") != CANDIDATE_MODEL_SHA256:
        raise ValueError("frozen Rocket model changed")
    if sha256(candidate / "main.py") != CANDIDATE_MAIN_SHA256:
        raise ValueError("frozen Rocket main changed")
    if tree_sha256(opponent) != opponent_spec["tree"]:
        raise ValueError(f"frozen {opponent_key} bundle tree changed")
    if sha256(opponent / "deck.csv") != opponent_spec["deck"]:
        raise ValueError(f"frozen {opponent_key} deck changed")
    if sha256(opponent / "main.py") != opponent_spec["main"]:
        raise ValueError(f"frozen {opponent_key} main changed")
    deck = read_deck(candidate)
    if len(deck) != 60 or deck.count(432) != 0 or deck.count(434) != 1:
        raise ValueError("Rocket deck is not the locked Wobbuffet-free Mimikyu list")
    if deck.count(1158) != 1 or deck.count(1175) != 1:
        raise ValueError("Rocket deck lost the locked Maximum Belt/Brave Bangle tools")
    return opponent, opponent_spec


def main() -> int:
    args = parse_args()
    opponent_path, opponent_spec = verify_frozen_inputs(args.opponent)
    split = str(args.split)
    games = SPLIT_COUNTS[split]
    shard_seed = expected_seed(split, args.opponent, args.own_seat)
    output = args.output.resolve()
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing existing V1 shard: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)

    random.seed(shard_seed)
    np.random.seed(shard_seed)
    torch.manual_seed(shard_seed)
    torch.set_num_threads(1)
    sys.path.insert(0, str(CANDIDATE.resolve()))
    candidate_main = _load_candidate_main(CANDIDATE.resolve())
    model = candidate_main._NET.cpu().eval()
    deck = [int(value) for value in candidate_main._DECK]
    priors = candidate_main._OPPONENT_DECK_PRIORS
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier.engine import supports_seeded_battles  # type: ignore

    if supports_seeded_battles():
        raise RuntimeError("protocol recorded unseeded native engine, but seeded extension is now present")

    opponent = OpponentWorker(opponent_path, EVALUATOR.resolve(), args.action_timeout_seconds)
    replay_paths: list[Path] = []
    game_rows: list[dict[str, Any]] = []
    wins = losses = draws = decisions = own_decisions = 0
    eligible = explored = guarded = hidden_audits = 0
    selection_types: Counter[str] = Counter()
    started = time()
    try:
        for game_index in range(1, games + 1):
            opponent.reset()
            exploration_seed = shard_seed * 1_000_000 + game_index
            agent = ExploratoryAgent(
                candidate_main,
                model,
                deck,
                priors,
                random.Random(exploration_seed),
            )
            player_decks = (deck, opponent.deck) if args.own_seat == 0 else (opponent.deck, deck)
            observation, start = battle_start(*player_decks)
            if observation is None:
                raise RuntimeError(f"official battle_start failed: {start.errorPlayer}/{start.errorType}")
            name = f"target80_rs_v1_{split}_{args.opponent}_s{args.own_seat}_{game_index:04d}.jsonl"
            final_path = output / name
            temporary = output / f".{name}.partial"
            game_started = perf_counter()
            first_player: int | None = None
            result = -1
            game_decisions = game_own_decisions = game_eligible = game_explored = game_guarded = 0
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as sink:
                    meta = {
                        "meta": {
                            "schema": SCHEMA,
                            "protocol": PROTOCOL_NAME,
                            "split": split,
                            "players": [BOT if args.own_seat == 0 else opponent_path.name,
                                        opponent_path.name if args.own_seat == 0 else BOT],
                            "decks": player_decks,
                            "opponent_key": args.opponent,
                            "own_seat": args.own_seat,
                            "shard_behavior_seed": shard_seed,
                            "exploration_seed": exploration_seed,
                            "engine_seed": None,
                            "engine_seeded": False,
                            "game_index": game_index,
                        }
                    }
                    sink.write(json.dumps(meta, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
                    for decision_index in range(args.max_decisions):
                        current = observation.get("current") or {}
                        observed_first = current.get("firstPlayer")
                        if observed_first in (0, 1):
                            first_player = int(observed_first)
                            if first_player != 0:
                                raise RuntimeError("official engine changed the first-player seat")
                        result = int(current.get("result") if current.get("result") is not None else -1)
                        if result != -1:
                            game_decisions = decision_index
                            break
                        selection = observation.get("select")
                        if not isinstance(selection, dict):
                            raise RuntimeError("official engine emitted no live selection")
                        actor = int(current.get("yourIndex") if current.get("yourIndex") is not None else -1)
                        behavior: dict[str, Any] | None = None
                        if actor == args.own_seat:
                            if not hidden_hand_boundary(observation, actor):
                                raise RuntimeError("Rocket actor observation crossed hidden-hand boundary")
                            hidden_audits += 1
                            action = agent(observation)
                            behavior = agent.last_decision
                            if behavior is None:
                                raise RuntimeError("Rocket action has no behavior record")
                            audit_behavior_record(behavior, action, selection)
                            own_decisions += 1
                            game_own_decisions += 1
                            selection_types[str(behavior["selection_type"])] += 1
                            if behavior.get("eligible"):
                                eligible += 1
                                game_eligible += 1
                            if behavior.get("explored"):
                                explored += 1
                                game_explored += 1
                            if str(behavior.get("reference_source") or "").endswith("_guard"):
                                guarded += 1
                                game_guarded += 1
                        elif actor == 1 - args.own_seat:
                            action = opponent.act(dict(observation))
                        else:
                            raise RuntimeError(f"unexpected acting player: {actor}")
                        if not legal(list(action), selection):
                            raise RuntimeError(f"illegal official-engine action actor={actor}: {action}")
                        row: dict[str, Any] = {
                            "p": actor,
                            "obs": deepcopy(observation),
                            "action": list(action),
                        }
                        if behavior is not None:
                            row["behavior"] = behavior
                        sink.write(json.dumps(row, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
                        observation = battle_select(list(action))
                    else:
                        raise RuntimeError("game exceeded decision limit")
                    if first_player != 0 or result not in (0, 1, 2):
                        raise RuntimeError("terminal result or first-player marker was not verified")
                    sink.write(
                        json.dumps(
                            {
                                "result": {
                                    "winner": result if result in (0, 1) else None,
                                    "raw_result": result,
                                    "own_win": int(result == args.own_seat),
                                    "draw": int(result == 2),
                                    "steps": game_decisions,
                                    "own_decisions": game_own_decisions,
                                    "eligible": game_eligible,
                                    "explored": game_explored,
                                    "guarded": game_guarded,
                                    "shard_behavior_seed": shard_seed,
                                    "exploration_seed": exploration_seed,
                                    "game_index": game_index,
                                }
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                            allow_nan=False,
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
            wins += int(result == args.own_seat)
            losses += int(result == 1 - args.own_seat)
            draws += int(result == 2)
            game_rows.append(
                {
                    "file": name,
                    "sha256": digest(final_path),
                    "raw_result": result,
                    "own_win": int(result == args.own_seat),
                    "draw": int(result == 2),
                    "decisions": game_decisions,
                    "own_decisions": game_own_decisions,
                    "eligible": game_eligible,
                    "explored": game_explored,
                    "guarded": game_guarded,
                    "duration_seconds": perf_counter() - game_started,
                }
            )
            if game_index == 1 or game_index % 20 == 0 or game_index == games:
                print(
                    json.dumps(
                        {
                            "split": split,
                            "opponent": args.opponent,
                            "own_seat": args.own_seat,
                            "completed": game_index,
                            "target": games,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        opponent.close()

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": digest(PROTOCOL.resolve()),
        "collector_sha256": digest(Path(__file__).resolve()),
        "official_engine_execution": True,
        "native_engine_seeded": False,
        "hidden_information_used_by_actor": False,
        "strict_no_fallback_execution": True,
        "development_only": True,
        "deployment_authority": False,
        "split": split,
        "candidate": str(CANDIDATE.resolve()),
        "candidate_bundle_sha256": tree_sha256(CANDIDATE.resolve()),
        "candidate_deck_sha256": sha256(CANDIDATE.resolve() / "deck.csv"),
        "candidate_model_sha256": sha256(CANDIDATE.resolve() / "model.pt"),
        "candidate_main_sha256": sha256(CANDIDATE.resolve() / "main.py"),
        "opponent_key": args.opponent,
        "opponent": str(opponent_path),
        "opponent_bundle_sha256": tree_sha256(opponent_path),
        "opponent_deck_sha256": opponent_spec["deck"],
        "opponent_main_sha256": opponent_spec["main"],
        "own_seat": args.own_seat,
        "shard_behavior_seed": shard_seed,
        "epsilon": EPSILON,
        "minimum_reference_support": MIN_REFERENCE_SUPPORT,
        "games": len(replay_paths),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisions": decisions,
        "own_decisions": own_decisions,
        "hidden_boundary_audits": hidden_audits,
        "eligible_decisions": eligible,
        "explored_decisions": explored,
        "guarded_decisions": guarded,
        "selection_type_counts": dict(sorted(selection_types.items())),
        "replay_tree_sha256": replay_tree_digest(replay_paths),
        "replays": game_rows,
        "elapsed_seconds": time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "games": len(replay_paths),
                "eligible": eligible,
                "explored": explored,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

