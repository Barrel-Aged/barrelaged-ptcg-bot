#!/usr/bin/env python3
"""Collect action-level causal values at live official-engine roots.

The collector plays complete local matches through the packaged official
``cg`` runtime.  At selected Team Rocket decision roots it reads the native
visualizer's transient full state, checks it against the normal redacted agent
observation, and uses that realized hidden completion only to initialize the
licensed ``search_begin`` branch API.  Every candidate is then rolled out from
the same root.  Raw hidden cards are never serialized or supplied to a policy.

These samples are exact for the realized root but are not an information-set
expectation or an equilibrium certificate.  They are intended for conservative
paired policy improvement with complete-game train/calibration/test splits.
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter, time
from typing import Any, Callable, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SELECT = 0
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs"
    r"\ptcg_agent_lab\tools\evaluate.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_neural_resolver_safe_tie_v3"),
    )
    parser.add_argument(
        "--own-candidate",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"),
    )
    parser.add_argument(
        "--opponent",
        type=Path,
        default=(PROJECT_ROOT / "work" / "arena_bundles" / "AshMarnieGeneralV110_extracted"),
    )
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--opponent-timeout", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "live_truth_causal_v1.jsonl"),
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=615_021)
    parser.add_argument("--seat", choices=("first", "second", "balanced", "any"), default="second")
    parser.add_argument("--outcome", choices=("all", "win", "loss"), default="all")
    parser.add_argument("--roots-per-game", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--rollouts-per-candidate", type=int, default=2)
    parser.add_argument("--earliest-turn", type=int, default=1)
    parser.add_argument("--latest-turn", type=int, default=8)
    parser.add_argument("--max-live-decisions", type=int, default=2_000)
    parser.add_argument("--max-rollout-decisions", type=int, default=600)
    parser.add_argument("--root-budget-seconds", type=float, default=60.0)
    parser.add_argument(
        "--common-random-coins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose native coin chance nodes and use one shared fair sequence per rollout.",
    )
    parser.add_argument(
        "--include-observation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "games",
        "roots_per_game",
        "candidates",
        "rollouts_per_candidate",
        "max_live_decisions",
        "max_rollout_decisions",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.root_budget_seconds <= 0:
        raise ValueError("--root-budget-seconds must be positive")
    if args.opponent_timeout <= 0:
        raise ValueError("--opponent-timeout must be positive")
    if args.earliest_turn < 0 or args.latest_turn < args.earliest_turn:
        raise ValueError("turn window is invalid")
    manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError(f"refusing to overwrite causal artifact: {args.output}")


def read_deck(path: Path) -> list[int]:
    values = path.read_text(encoding="utf-8").replace(",", "\n").split()
    deck = [int(value) for value in values]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(deck)}")
    return deck


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _card_ids(items: Any, name: str) -> list[int]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        raise ValueError(f"truth zone {name} is not a card sequence")
    result = []
    for item in items:
        if not isinstance(item, Mapping) or int(item.get("id") or 0) <= 0:
            raise ValueError(f"truth zone {name} contains a hidden or invalid card")
        result.append(int(item["id"]))
    return result


def _pokemon_public(item: Any) -> Any:
    if item is None:
        return None
    if not isinstance(item, Mapping):
        return {"invalid": type(item).__name__}
    return {
        "id": int(item.get("id") or 0),
        "hp": int(item.get("hp") or 0),
        "max_hp": int(item.get("maxHp") or 0),
        "energies": tuple(int(value) for value in item.get("energies") or ()),
        "energy_cards": tuple(
            int(card.get("id") or 0)
            for card in item.get("energyCards") or ()
            if isinstance(card, Mapping)
        ),
        "tools": tuple(
            int(card.get("id") or 0)
            for card in item.get("tools") or ()
            if isinstance(card, Mapping)
        ),
        "pre_evolution": tuple(
            int(card.get("id") or 0)
            for card in item.get("preEvolution") or ()
            if isinstance(card, Mapping)
        ),
    }


def _public_parity_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    current = observation.get("current") or {}
    players = current.get("players") or ()
    public_players = []
    for player in players:
        player = player if isinstance(player, Mapping) else {}
        public_players.append(
            {
                "active": tuple(_pokemon_public(item) for item in player.get("active") or ()),
                "bench": tuple(_pokemon_public(item) for item in player.get("bench") or ()),
                "deck_count": int(player.get("deckCount") or 0),
                "hand_count": int(player.get("handCount") or 0),
                "prize_count": len(player.get("prize") or ()),
                "discard": tuple(
                    int(card.get("id") or 0)
                    for card in player.get("discard") or ()
                    if isinstance(card, Mapping)
                ),
            }
        )
    return {
        "turn": int(current.get("turn") or 0),
        "turn_action_count": int(current.get("turnActionCount") or 0),
        "your_index": int(current.get("yourIndex") or 0),
        "first_player": int(current.get("firstPlayer") if current.get("firstPlayer") is not None else -1),
        "result": int(current.get("result") if current.get("result") is not None else -1),
        "players": tuple(public_players),
        "stadium": tuple(
            (int(card.get("id") or 0), int(card.get("playerIndex") or 0))
            for card in current.get("stadium") or ()
            if isinstance(card, Mapping)
        ),
    }


def _truth_completion(
    observation: Mapping[str, Any],
    truth_frame: Mapping[str, Any],
    *,
    manual_coin: bool,
) -> dict[str, Any]:
    if _public_parity_view(observation) != _public_parity_view(truth_frame):
        raise RuntimeError("visualizer/live public-state parity failed")
    live = observation.get("current") or {}
    truth = truth_frame.get("current") or {}
    live_players = live.get("players") or ()
    truth_players = truth.get("players") or ()
    root_player = int(live.get("yourIndex") or 0)
    if root_player not in (0, 1) or len(live_players) != 2 or len(truth_players) != 2:
        raise RuntimeError("truth root does not contain two valid players")
    mine = truth_players[root_player]
    theirs = truth_players[1 - root_player]
    live_theirs = live_players[1 - root_player]
    hidden_active: list[int] = []
    live_active = live_theirs.get("active") or ()
    if live_active and live_active[0] is None:
        hidden_active = _card_ids(theirs.get("active") or (), "opponent active")
    completion = {
        "your_deck": _card_ids(mine.get("deck") or (), "acting deck"),
        "your_prize": _card_ids(mine.get("prize") or (), "acting prize"),
        "opponent_deck": _card_ids(theirs.get("deck") or (), "opponent deck"),
        "opponent_prize": _card_ids(theirs.get("prize") or (), "opponent prize"),
        "opponent_hand": _card_ids(theirs.get("hand") or (), "opponent hand"),
        "opponent_active": hidden_active,
        "manual_coin": bool(manual_coin),
    }
    expected = (
        int(live_players[root_player].get("deckCount") or 0),
        len(live_players[root_player].get("prize") or ()),
        int(live_players[1 - root_player].get("deckCount") or 0),
        len(live_players[1 - root_player].get("prize") or ()),
        int(live_players[1 - root_player].get("handCount") or 0),
        len(hidden_active),
    )
    observed = (
        len(completion["your_deck"]),
        len(completion["your_prize"]),
        len(completion["opponent_deck"]),
        len(completion["opponent_prize"]),
        len(completion["opponent_hand"]),
        len(completion["opponent_active"]),
    )
    if expected != observed:
        raise RuntimeError(f"truth/live hidden-zone count mismatch: {observed} != {expected}")
    return completion


def _completion_public_digest(completion: Mapping[str, Any]) -> str:
    return fingerprint({name: len(value) if isinstance(value, list) else value for name, value in completion.items()})


def _valid_choice(observation: Mapping[str, Any], action: Sequence[int]) -> bool:
    select = observation.get("select") or {}
    options = select.get("option") or ()
    low = int(select.get("minCount") or 0)
    high = min(int(select.get("maxCount") or 0), len(options))
    return (
        isinstance(action, Sequence)
        and not isinstance(action, (str, bytes, bytearray))
        and low <= len(action) <= high
        and len(set(action)) == len(action)
        and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
    )


def option_summary(option: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(option, Mapping):
        return {"raw_type": type(option).__name__}
    current = observation.get("current") or {}
    players = current.get("players") or ()
    me = int(current.get("yourIndex") or 0)
    index = option.get("index")
    hand_card_id = None
    if 0 <= me < len(players) and isinstance(index, int):
        hand = players[me].get("hand") or ()
        if 0 <= index < len(hand) and isinstance(hand[index], Mapping):
            hand_card_id = hand[index].get("id")
    return {
        "type": option.get("type"),
        "index": option.get("index"),
        "area": option.get("area"),
        "in_play_area": option.get("inPlayArea"),
        "in_play_index": option.get("inPlayIndex"),
        "player_index": option.get("playerIndex"),
        "attack_id": option.get("attackId"),
        "hand_card_id": hand_card_id,
    }


def utility(result: int, root_player: int) -> float:
    if result == 2:
        return 0.0
    if result not in (0, 1):
        raise ValueError(f"invalid result {result}")
    return 1.0 if result == root_player else -1.0


def _mean_and_se(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, math.inf
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, math.inf
    return mean, statistics.stdev(values) / math.sqrt(len(values))


class _WorkerAgent:
    """Adapt an isolated packaged arena policy to the local agent protocol."""

    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.belief: Any = None

    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        return self.worker.act(dict(observation))


class _GuardedAgent:
    """Apply the candidate's public-state deployment guards before its net."""

    def __init__(
        self,
        blueprint: Any,
        guards: Sequence[tuple[str, Callable[[dict[str, Any]], Any]]],
    ) -> None:
        self.blueprint = blueprint
        self.guards = tuple(guards)

    @property
    def belief(self) -> Any:
        return self.blueprint.belief

    @belief.setter
    def belief(self, value: Any) -> None:
        self.blueprint.belief = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self.blueprint, name)

    def __call__(self, observation: Mapping[str, Any]) -> list[int]:
        public_observation = dict(observation)
        for _name, guard in self.guards:
            action = guard(public_observation)
            if action is not None:
                return list(action)
        return list(self.blueprint(public_observation))


def _load_public_guards(candidate: Path) -> tuple[tuple[str, Callable[..., Any]], ...]:
    """Load only the known public-state guards from a packaged candidate.

    Importing a candidate also loads its frozen model, so preserve the caller's
    random state and import path.  The returned functions retain their module
    globals and therefore execute exactly as they do in the packaged entrypoint.
    """

    main_path = candidate.resolve() / "main.py"
    if not main_path.is_file():
        return ()
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    import_path = list(sys.path)
    try:
        module_name = f"rocket_causal_guards_{sha256(main_path)[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load candidate entrypoint: {main_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        names = (
            "_early_spidops_search_override",
            "_spidops_bangle_lethal_override",
        )
        return tuple(
            (name, function)
            for name in names
            if callable(function := getattr(module, name, None))
        )
    finally:
        sys.path[:] = import_path
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


class OfficialCausalRuntime:
    def __init__(
        self,
        runtime: Path,
        own: Path,
        opponent: Path,
        candidates: int,
        *,
        evaluator: Path,
        opponent_timeout: float,
    ):
        runtime = runtime.resolve()
        if str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        from cg.api import search_begin, search_end, search_step, to_observation_class  # type: ignore
        from soldier.engine import battle_finish, battle_select, battle_start, visualize_data  # type: ignore
        from soldier.net import NNAgent, load_policy  # type: ignore
        from soldier.neural_search import NeuralRolloutAgent  # type: ignore
        from train_rank1_second_seat_ppo import OpponentWorker  # type: ignore

        self.battle_start = battle_start
        self.battle_select = battle_select
        self.battle_finish = battle_finish
        self.visualize_data = visualize_data
        self.search_begin = search_begin
        self.search_end = search_end
        self.search_step = search_step
        self.to_observation_class = to_observation_class
        self.agent_type = NNAgent
        self.teacher_type = NeuralRolloutAgent
        self.own_path = own.resolve()
        self.opponent_path = opponent.resolve()
        self.own_deck = read_deck(self.own_path / "deck.csv")
        self.opponent_deck = read_deck(self.opponent_path / "deck.csv")
        self.own_model = load_policy(self.own_path / "model.pt").eval()
        opponent_model_path = self.opponent_path / "model.pt"
        self.opponent_model = (
            load_policy(opponent_model_path).eval()
            if opponent_model_path.is_file()
            else None
        )
        self.opponent_worker = (
            None
            if self.opponent_model is not None
            else OpponentWorker(self.opponent_path, evaluator, opponent_timeout)
        )
        self.opponent_execution = (
            "in_process_neural_blueprint"
            if self.opponent_model is not None
            else "isolated_packaged_worker"
        )
        self.evaluator_path = evaluator.resolve()
        self.guards = _load_public_guards(self.own_path)
        self.candidates = int(candidates)
        library_path = self.own_path / "deck_library.json"
        self.deck_library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else {}
        self.deck_library["causal_target_opponent"] = self.opponent_deck

    def new_agents(
        self,
        beliefs: tuple[Any, Any] | None = None,
        *,
        opponent_history: Sequence[tuple[Mapping[str, Any], Sequence[int]]] = (),
    ) -> tuple[Any, Any]:
        own_blueprint = self.agent_type(
            self.own_model,
            greedy=True,
            deck_ids=self.own_deck,
            forbid_end_when_attack=True,
            opponent_deck_priors=self.deck_library,
        )
        own = _GuardedAgent(own_blueprint, self.guards) if self.guards else own_blueprint
        if self.opponent_worker is not None:
            self.opponent_worker.reset()
            for history_index, (historic_observation, historic_action) in enumerate(opponent_history):
                replayed_action = self.opponent_worker.act(copy.deepcopy(dict(historic_observation)))
                if replayed_action != list(historic_action):
                    raise RuntimeError(
                        "packaged opponent history replay diverged at "
                        f"call {history_index}: {replayed_action} != {list(historic_action)}"
                    )
            opponent = _WorkerAgent(self.opponent_worker)
        else:
            opponent = self.agent_type(
                self.opponent_model,
                greedy=True,
                deck_ids=self.opponent_deck,
                forbid_end_when_attack=True,
                opponent_deck_priors=self.deck_library,
            )
        if beliefs is not None:
            own.belief = copy.deepcopy(beliefs[0])
            opponent.belief = copy.deepcopy(beliefs[1])
        return own, opponent

    def close(self) -> None:
        if self.opponent_worker is not None:
            self.opponent_worker.close()
            self.opponent_worker = None

    def capture_match(
        self,
        *,
        seat_mode: str,
        accepted_game_index: int,
        roots_per_game: int,
        earliest_turn: int,
        latest_turn: int,
        max_decisions: int,
        common_random_coins: bool,
        capture_truth: bool = True,
    ) -> dict[str, Any]:
        if seat_mode == "first":
            own_seat = 0
        elif seat_mode == "second":
            own_seat = 1
        elif seat_mode == "balanced":
            own_seat = 0 if accepted_game_index % 2 == 1 else 1
        else:
            own_seat = (accepted_game_index - 1) % 2
        decks = (
            (self.own_deck, self.opponent_deck)
            if own_seat == 0
            else (self.opponent_deck, self.own_deck)
        )
        observation, start = self.battle_start(*decks)
        if observation is None:
            raise RuntimeError(f"battle start failed: player={start.errorPlayer} type={start.errorType}")
        first_player: int | None = None
        started_first: bool | None = None
        own_agent, opponent_agent = self.new_agents()
        agents = (
            (own_agent, opponent_agent)
            if own_seat == 0
            else (opponent_agent, own_agent)
        )
        roots = []
        opponent_history: list[tuple[dict[str, Any], list[int]]] = []
        retained_turns: set[int] = set()
        started = perf_counter()
        try:
            for step in range(max_decisions):
                current = observation.get("current") or {}
                observed_first_player = int(
                    current.get("firstPlayer")
                    if current.get("firstPlayer") is not None
                    else -1
                )
                if first_player is None and observed_first_player in (0, 1):
                    first_player = observed_first_player
                    started_first = first_player == own_seat
                    desired = (
                        seat_mode == "any"
                        or (seat_mode == "first" and started_first)
                        or (seat_mode == "second" and not started_first)
                        or (
                            seat_mode == "balanced"
                            and started_first == (accepted_game_index % 2 == 1)
                        )
                    )
                    if not desired:
                        return {"seat_skipped": True}
                result = int(current.get("result") if current.get("result") is not None else -1)
                if result != -1:
                    if first_player is None or started_first is None:
                        raise RuntimeError("official match terminated before first player was declared")
                    return {
                        "seat_skipped": False,
                        "result": result,
                        "own_seat": own_seat,
                        "first_player": first_player,
                        "started_first": started_first,
                        "decisions": step,
                        "duration_seconds": perf_counter() - started,
                        "roots": roots,
                    }
                actor = int(current.get("yourIndex"))
                action = agents[actor](observation)
                if not _valid_choice(observation, action):
                    raise RuntimeError(f"policy returned illegal live action at step {step}: {action}")
                select = observation.get("select") or {}
                options = select.get("option") or ()
                turn = int(current.get("turn") or 0)
                if (
                    actor == own_seat
                    and len(roots) < roots_per_game
                    and turn not in retained_turns
                    and earliest_turn <= turn <= latest_turn
                    and select.get("type") == MAIN_SELECT
                    and select.get("minCount") == 1
                    and select.get("maxCount") == 1
                    and len(options) >= 2
                    and len(action) == 1
                    and observation.get("search_begin_input") is not None
                ):
                    teacher = self.teacher_type(
                        own_agent,
                        own_model=self.own_model,
                        # Candidate proposal is an own-model-only operation.
                        # A packaged code opponent may not expose a model file.
                        opponent_model=self.opponent_model or self.own_model,
                        own_deck=self.own_deck,
                        deck_library=self.deck_library,
                        candidates=self.candidates,
                    )
                    candidate_indices = teacher._candidate_indices(observation, int(action[0]))
                    if len(candidate_indices) >= 2:
                        root = {
                            "step": step,
                            "turn": turn,
                            "turn_action_count": current.get("turnActionCount"),
                            "baseline_action": list(action),
                            "candidate_indices": list(candidate_indices),
                            "candidate_options": {
                                str(index): option_summary(options[index], observation)
                                for index in candidate_indices
                            },
                            "observation": copy.deepcopy(observation),
                            "observation_fingerprint": fingerprint(observation),
                            "_beliefs": (
                                copy.deepcopy(own_agent.belief),
                                copy.deepcopy(opponent_agent.belief),
                            ),
                            "_opponent_history": copy.deepcopy(opponent_history),
                            "_own_seat": own_seat,
                        }
                        if capture_truth:
                            frames = json.loads(self.visualize_data())
                            if not isinstance(frames, list) or not frames or not isinstance(frames[-1], dict):
                                raise RuntimeError("official visualizer returned no truth frame")
                            completion = _truth_completion(
                                observation,
                                frames[-1],
                                manual_coin=common_random_coins,
                            )
                            root.update(
                                {
                                    "truth_frame_fingerprint": fingerprint(frames[-1]),
                                    "truth_completion_count_fingerprint": _completion_public_digest(completion),
                                    "_completion": completion,
                                }
                            )
                        roots.append(root)
                        retained_turns.add(turn)
                if actor != own_seat and self.opponent_worker is not None:
                    opponent_history.append((copy.deepcopy(observation), list(action)))
                observation = self.battle_select(list(action))
            raise RuntimeError(f"live match exceeded {max_decisions} decisions")
        finally:
            self.battle_finish()

    def _rollout(
        self,
        branch: Any,
        *,
        root_player: int,
        beliefs: tuple[Any, Any],
        opponent_history: Sequence[tuple[Mapping[str, Any], Sequence[int]]],
        max_decisions: int,
        deadline: float,
        coin_seed: int,
        common_random_coins: bool,
    ) -> tuple[float, bool, int, int]:
        agents = self.new_agents(beliefs, opponent_history=opponent_history)
        actors = (
            agents
            if root_player == 0
            else (agents[1], agents[0])
        )
        coin_rng = random.Random(coin_seed)
        coin_flips = 0
        for step in range(max_decisions):
            raw = asdict(branch.observation)
            current = raw.get("current") or {}
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                return utility(result, root_player), True, step, coin_flips
            if perf_counter() >= deadline or raw.get("select") is None:
                break
            actor = int(current.get("yourIndex") or 0)
            select = raw.get("select") or {}
            if common_random_coins and int(select.get("context") or -1) == 46:
                want_type = 1 if coin_rng.getrandbits(1) else 2
                matches = [
                    index
                    for index, option in enumerate(select.get("option") or ())
                    if isinstance(option, Mapping) and int(option.get("type") or -1) == want_type
                ]
                if len(matches) != 1:
                    raise RuntimeError("manual coin prompt did not expose one head/tail option")
                action = [matches[0]]
                coin_flips += 1
            else:
                action = actors[actor](raw)
            if not _valid_choice(raw, action):
                raise RuntimeError(f"rollout policy returned illegal action: {action}")
            branch = self.search_step(branch.searchId, action)
        return 0.0, False, max_decisions, coin_flips

    def evaluate_root(
        self,
        root: Mapping[str, Any],
        *,
        rollouts_per_candidate: int,
        max_rollout_decisions: int,
        budget_seconds: float,
        shuffle_seed: int,
        common_random_coins: bool,
    ) -> dict[str, Any]:
        deadline = perf_counter() + budget_seconds
        candidates = tuple(int(index) for index in root["candidate_indices"])
        samples = {index: [] for index in candidates}
        terminal = Counter()
        depths = {index: [] for index in candidates}
        coin_flips = {index: [] for index in candidates}
        errors: list[str] = []
        search_root = None
        try:
            search_root = self.search_begin(
                self.to_observation_class(root["observation"]),
                **root["_completion"],
            )
            for rollout in range(rollouts_per_candidate):
                order = list(candidates)
                random.Random(shuffle_seed + 1009 * rollout).shuffle(order)
                for index in order:
                    if perf_counter() >= deadline:
                        break
                    try:
                        child = self.search_step(search_root.searchId, [index])
                        value, reached_terminal, depth, flips = self._rollout(
                            child,
                            root_player=int(root["_own_seat"]),
                            beliefs=root["_beliefs"],
                            opponent_history=root["_opponent_history"],
                            max_decisions=max_rollout_decisions,
                            deadline=deadline,
                            coin_seed=shuffle_seed + 65_537 * rollout,
                            common_random_coins=common_random_coins,
                        )
                        if reached_terminal:
                            samples[index].append(value)
                            terminal[index] += 1
                            depths[index].append(depth)
                            coin_flips[index].append(flips)
                    except Exception as error:
                        errors.append(f"candidate={index}: {type(error).__name__}: {str(error)[:180]}")
                if perf_counter() >= deadline:
                    break
        finally:
            if search_root is not None:
                self.search_end()
        complete = min((len(values) for values in samples.values()), default=0)
        means = {index: _mean_and_se(values)[0] for index, values in samples.items() if values}
        standard_errors = {index: _mean_and_se(values)[1] for index, values in samples.items() if values}
        baseline = int(root["baseline_action"][0])
        baseline_mean = means.get(baseline)
        best = max(means, key=lambda index: (means[index], -index)) if baseline_mean is not None else baseline
        deltas = {index: value - baseline_mean for index, value in means.items()} if baseline_mean is not None else {}
        return {
            "baseline": baseline,
            "best": best,
            "changed": bool(best != baseline and deltas.get(best, 0.0) > 0.0),
            "samples": {str(index): values for index, values in samples.items()},
            "means": {str(index): value for index, value in means.items()},
            "standard_errors": {
                str(index): value if math.isfinite(value) else None
                for index, value in standard_errors.items()
            },
            "deltas": {str(index): value for index, value in deltas.items()},
            "terminal": {str(index): terminal[index] for index in candidates},
            "depths": {str(index): values for index, values in depths.items()},
            "coin_flips": {str(index): values for index, values in coin_flips.items()},
            "complete_rollout_sets": complete,
            "fully_complete": complete >= rollouts_per_candidate,
            "errors": errors,
        }


def _outcome_matches(mode: str, value: float) -> bool:
    return mode == "all" or (mode == "win" and value > 0.0) or (mode == "loss" and value < 0.0)


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    runtime = OfficialCausalRuntime(
        args.runtime,
        args.own_candidate,
        args.opponent,
        args.candidates,
        evaluator=args.evaluator,
        opponent_timeout=args.opponent_timeout,
    )
    atexit.register(runtime.close)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    accepted_games = 0
    attempts = 0
    started = time()
    with output.open("x", encoding="utf-8") as sink:
        while accepted_games < args.games:
            attempts += 1
            capture = runtime.capture_match(
                seat_mode=args.seat,
                accepted_game_index=accepted_games + 1,
                roots_per_game=args.roots_per_game,
                earliest_turn=args.earliest_turn,
                latest_turn=args.latest_turn,
                max_decisions=args.max_live_decisions,
                common_random_coins=args.common_random_coins,
            )
            if capture.get("seat_skipped"):
                counters["seat_skips"] += 1
                continue
            baseline_utility = utility(int(capture["result"]), int(capture["own_seat"]))
            if not _outcome_matches(args.outcome, baseline_utility):
                counters["outcome_skips"] += 1
                continue
            accepted_games += 1
            counters["games"] += 1
            counters["baseline_wins"] += int(baseline_utility > 0)
            counters["baseline_losses"] += int(baseline_utility < 0)
            counters["games_without_roots"] += int(not capture["roots"])
            for root_index, root in enumerate(capture["roots"]):
                evaluation = runtime.evaluate_root(
                    root,
                    rollouts_per_candidate=args.rollouts_per_candidate,
                    max_rollout_decisions=args.max_rollout_decisions,
                    budget_seconds=args.root_budget_seconds,
                    shuffle_seed=args.seed + 100_003 * accepted_games + root_index,
                    common_random_coins=args.common_random_coins,
                )
                counters["roots"] += 1
                counters["changed"] += int(evaluation["changed"])
                counters["fully_complete_roots"] += int(evaluation["fully_complete"])
                counters["rollout_errors"] += len(evaluation["errors"])
                row = {
                    "schema": "rocket-live-truth-causal-action/v1",
                    "game_id": accepted_games,
                    "capture_attempt": attempts,
                    "root_index": root_index,
                    "opponent": args.opponent.resolve().name,
                    "own_seat": capture["own_seat"],
                    "first_player": capture["first_player"],
                    "started_first": capture["started_first"],
                    "baseline_game_result": capture["result"],
                    "baseline_game_utility": baseline_utility,
                    "baseline_game_decisions": capture["decisions"],
                    "step": root["step"],
                    "turn": root["turn"],
                    "turn_action_count": root["turn_action_count"],
                    "baseline_action": root["baseline_action"],
                    "candidate_indices": root["candidate_indices"],
                    "candidate_options": root["candidate_options"],
                    "observation_fingerprint": root["observation_fingerprint"],
                    "truth_frame_fingerprint": root["truth_frame_fingerprint"],
                    "truth_completion_count_fingerprint": root["truth_completion_count_fingerprint"],
                    **evaluation,
                }
                if args.include_observation:
                    row["observation"] = root["observation"]
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                sink.flush()
            print(
                json.dumps(
                    {
                        "game": accepted_games,
                        "of": args.games,
                        "baseline_utility": baseline_utility,
                        "roots": len(capture["roots"]),
                        "total_roots": counters["roots"],
                        "changed": counters["changed"],
                        "complete": counters["fully_complete_roots"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = {
        "schema": "rocket-live-truth-causal-manifest/v1",
        "method": "transient official visualizer truth plus licensed native one-action branches",
        "genuine_official_engine_source_matches": True,
        "genuine_official_engine_descendant_transitions": True,
        "realized_hidden_state_official_sample": True,
        "raw_hidden_truth_serialized": False,
        "hidden_truth_policy_input": False,
        "public_information_set_expected_value": False,
        "equilibrium_or_safety_certificate": False,
        "counterfactual_scope": "conditional on one realized hidden state per independently reached root",
        "common_random_coin_sequences": args.common_random_coins,
        "own_public_deployment_guards": [name for name, _function in runtime.guards],
        "opponent_execution": runtime.opponent_execution,
        "opponent_observation_history_reconstructed_at_branches": True,
        "opponent_history_actions_exactly_reproduced": True,
        "code_opponent_branch_limitation": (
            None
            if runtime.opponent_execution == "in_process_neural_blueprint"
            else (
                "The isolated packaged opponent is reset at each branch, then every earlier "
                "source-match opponent observation is replayed and its action is required to "
                "match exactly. Wall-clock timing and process-external state are not replayed."
            )
        ),
        "own_candidate": str(args.own_candidate.resolve()),
        "opponent": str(args.opponent.resolve()),
        "own_model_sha256": sha256(args.own_candidate.resolve() / "model.pt"),
        "opponent_model_sha256": (
            sha256(args.opponent.resolve() / "model.pt")
            if (args.opponent.resolve() / "model.pt").is_file()
            else None
        ),
        "evaluator": str(runtime.evaluator_path),
        "evaluator_sha256": sha256(runtime.evaluator_path),
        "own_deck_sha256": sha256(args.own_candidate.resolve() / "deck.csv"),
        "opponent_deck_sha256": sha256(args.opponent.resolve() / "deck.csv"),
        "output": str(output),
        "output_sha256": sha256(output),
        "attempted_matches": attempts,
        "counts": dict(counters),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "elapsed_seconds": time() - started,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    runtime.close()
    atexit.unregister(runtime.close)
    print(json.dumps({"complete": True, "counts": manifest["counts"], "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
