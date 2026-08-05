#!/usr/bin/env python3
"""Train a persistent neural plan residual from complete official outcomes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional

from train_rank1_second_seat_ppo import OpponentWorker, legal, read_deck, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "candidates"
    / "scale_value_calibrated_v1"
)
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
DEFAULT_OPPONENTS = (
    PROJECT_ROOT / "work" / "arena_bundles" / "AshMarnieGeneralV110_extracted",
    PROJECT_ROOT
    / "work"
    / "arena_bundles"
    / "HelbrosMarnieGeneralRecoveryV4_extracted",
)


@dataclass
class PickRecord:
    option_tokens: torch.Tensor
    base_logits: torch.Tensor
    action: int
    old_logp: float


@dataclass
class DecisionRecord:
    global_token: torch.Tensor
    picks: list[PickRecord]
    old_value: float
    advantage: float = 0.0
    normalized_advantage: float = 0.0
    return_: float = 0.0


@dataclass
class EpisodeMetrics:
    setup_by_turn_six: bool = False
    first_attack_own_turn: int | None = None
    ever_second_ready: bool = False
    max_ready_attackers: int = 0
    ever_mewtwo_with_backup: bool = False
    avoidable_active_attachments: int = 0
    terminal_prizes_remaining: int = 6


@dataclass
class Episode:
    decisions: list[DecisionRecord]
    reward: float
    opponent: str
    engine_steps: int
    metrics: EpisodeMetrics
    strategic_target: torch.Tensor
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--opponent", type=Path, action="append", default=[])
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--rollout-games", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--clip", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.35)
    parser.add_argument("--strategic-coef", type=float, default=0.15)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--kl-coef", type=float, default=0.10)
    parser.add_argument("--residual-coef", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=1.10)
    parser.add_argument("--memory", type=int, default=128)
    parser.add_argument("--option-width", type=int, default=64)
    parser.add_argument("--max-abs-bias", type=float, default=2.5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=2_000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "iterations",
        "rollout_games",
        "ppo_epochs",
        "batch_episodes",
        "memory",
        "option_width",
        "threads",
        "max_steps",
    )
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "learning_rate",
        "gamma",
        "gae_lambda",
        "temperature",
        "max_abs_bias",
        "timeout",
    ):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not (0 < args.clip < 1):
        raise ValueError("PPO clip must be in (0, 1)")
    opponents = args.opponent or list(DEFAULT_OPPONENTS)
    if args.rollout_games % len(opponents):
        raise ValueError("rollout-games must be divisible by opponent count")


def raw_deck_sha256(path: Path) -> str:
    return sha256(path)


def _energy_cards(pokemon: Mapping[str, Any]) -> list[int]:
    cards = [
        int(card.get("id") or 0)
        for card in pokemon.get("energyCards") or ()
        if isinstance(card, Mapping)
    ]
    if cards:
        return cards
    return [15 if int(value) == 11 else int(value) for value in pokemon.get("energies") or ()]


def _ready(card_id: int, energy_cards: Sequence[int], rocket_count: int) -> bool:
    rocket = sum(card_id_ == 15 for card_id_ in energy_cards)
    grass = sum(card_id_ == 1 for card_id_ in energy_cards)
    psychic = sum(card_id_ == 5 for card_id_ in energy_cards) + 2 * rocket
    total = len(energy_cards) + rocket
    if card_id == 431:
        return rocket_count >= 4 and psychic >= 2 and total >= 3
    if card_id == 401:
        return grass >= 1 and total >= 2
    if card_id == 400:
        return grass >= 1
    if card_id == 432:
        return psychic >= 1 and total >= 2
    if card_id == 463:
        return total >= 1
    return False


def board_snapshot(observation: Mapping[str, Any], own_seat: int = 1) -> dict[str, Any]:
    current = observation.get("current") or {}
    players = current.get("players") or ()
    if len(players) != 2:
        return {
            "rocket_count": 0,
            "ready": [],
            "unready_bench": [],
            "prizes": 6,
        }
    own = players[own_seat] or {}
    board: list[tuple[int, int, Mapping[str, Any]]] = []
    board.extend(
        (4, index, pokemon)
        for index, pokemon in enumerate(own.get("active") or ())
        if isinstance(pokemon, Mapping)
    )
    board.extend(
        (5, index, pokemon)
        for index, pokemon in enumerate(own.get("bench") or ())
        if isinstance(pokemon, Mapping)
    )
    rocket_ids = {400, 401, 414, 431, 432, 463}
    attacker_ids = {400, 401, 431, 432, 463}
    rocket_count = sum(int(pokemon.get("id") or 0) in rocket_ids for _, _, pokemon in board)
    ready = [
        (area, index, pokemon)
        for area, index, pokemon in board
        if int(pokemon.get("id") or 0) in attacker_ids
        and _ready(
            int(pokemon.get("id") or 0),
            _energy_cards(pokemon),
            rocket_count,
        )
    ]
    unready_bench = [
        (area, index, pokemon)
        for area, index, pokemon in board
        if area == 5
        and int(pokemon.get("id") or 0) in attacker_ids
        and not _ready(
            int(pokemon.get("id") or 0),
            _energy_cards(pokemon),
            rocket_count,
        )
    ]
    return {
        "rocket_count": rocket_count,
        "ready": ready,
        "unready_bench": unready_bench,
        "prizes": len(own.get("prize") or ()),
    }


def update_public_metrics(
    metrics: EpisodeMetrics,
    observation: Mapping[str, Any],
) -> None:
    current = observation.get("current") or {}
    turn = int(current.get("turn") or 0)
    snapshot = board_snapshot(observation)
    if turn <= 6 and snapshot["rocket_count"] >= 4:
        metrics.setup_by_turn_six = True
    ready = snapshot["ready"]
    metrics.max_ready_attackers = max(metrics.max_ready_attackers, len(ready))
    metrics.ever_second_ready = metrics.ever_second_ready or len(ready) >= 2
    ready_ids = [int(pokemon.get("id") or 0) for _, _, pokemon in ready]
    if 431 in ready_ids and any(card_id != 431 for card_id in ready_ids):
        metrics.ever_mewtwo_with_backup = True
    metrics.terminal_prizes_remaining = int(snapshot["prizes"])


def record_selected_action_metrics(
    metrics: EpisodeMetrics,
    observation: Mapping[str, Any],
    action: Sequence[int],
    features_module: Any,
) -> None:
    selection = observation.get("select") or {}
    options = selection.get("option") or ()
    current = observation.get("current") or {}
    turn = int(current.get("turn") or 0)
    if any(
        int(options[index].get("type") or 0) == 13 for index in action
    ) and metrics.first_attack_own_turn is None:
        metrics.first_attack_own_turn = max(1, turn // 2)
    snapshot = board_snapshot(observation)
    active_ready = any(area == 4 for area, _, _ in snapshot["ready"])
    if active_ready and snapshot["unready_bench"]:
        me = int(current.get("yourIndex") or 0)
        for index in action:
            option = options[index]
            card_id = int(features_module._opt_card_id(option, current, me, selection) or 0)
            if (
                int(option.get("type") or 0) == 8
                and card_id in (1, 5, 15)
                and int(option.get("inPlayArea") or 0) == 4
            ):
                metrics.avoidable_active_attachments += 1


def strategic_target(metrics: EpisodeMetrics) -> torch.Tensor:
    setup = float(metrics.setup_by_turn_six)
    if metrics.first_attack_own_turn is None:
        tempo = 0.0
    else:
        tempo = max(0.0, 1.0 - (metrics.first_attack_own_turn - 1) / 5.0)
    if metrics.ever_second_ready:
        resource = 1.0 if metrics.avoidable_active_attachments == 0 else 0.5
    else:
        resource = 0.0
    prize = min(1.0, max(0.0, (6 - metrics.terminal_prizes_remaining) / 6.0))
    board = min(1.0, metrics.max_ready_attackers / 2.0)
    optionality = float(metrics.ever_mewtwo_with_backup)
    values = torch.tensor(
        [setup, tempo, resource, prize, board, optionality],
        dtype=torch.float32,
    )
    return 2.0 * values - 1.0


@torch.no_grad()
def choose_action(
    net: torch.nn.Module,
    residual: torch.nn.Module,
    memory: torch.Tensor,
    observation: Mapping[str, Any],
    belief: Any,
    features_module: Any,
    encode_public_tokens: Any,
    *,
    temperature: float,
) -> tuple[list[int], DecisionRecord, torch.Tensor]:
    selection = observation.get("select") or {}
    options = selection.get("option") or ()
    high = min(int(selection.get("maxCount") or 0), len(options))
    if high == 0:
        raise RuntimeError("choose_action called on empty selection")
    base_feature = features_module.extract(
        observation,
        belief=belief,
        slot_link=bool(getattr(net, "slot_link", False)),
    )
    picked: list[int] = []
    records: list[PickRecord] = []
    memory_token: torch.Tensor | None = None
    for _ in range(high):
        feature = (
            base_feature
            if high == 1
            else features_module.mark_selection(base_feature, picked, high)
        )
        global_token, option_tokens, base_logits = encode_public_tokens(
            net,
            feature,
            forbid_end_when_attack=True,
            options=options,
        )
        if memory_token is None:
            memory_token = global_token
        residual_logits = residual.option_residual(option_tokens, memory)
        combined = base_logits + residual_logits
        distribution = torch.distributions.Categorical(
            logits=combined / temperature
        )
        action_tensor = distribution.sample()
        action = int(action_tensor)
        records.append(
            PickRecord(
                option_tokens=option_tokens.detach().cpu(),
                base_logits=base_logits.detach().cpu(),
                action=action,
                old_logp=float(distribution.log_prob(action_tensor)),
            )
        )
        picked.append(action)
    if memory_token is None:
        raise RuntimeError("no neural plan memory token")
    value, _, _ = residual.values(memory)
    decision = DecisionRecord(
        global_token=memory_token.detach().cpu(),
        picks=records,
        old_value=float(value),
    )
    next_memory = residual.advance(memory_token, memory).detach()
    return picked, decision, next_memory


def assign_gae(
    episode: Episode,
    *,
    gamma: float,
    gae_lambda: float,
) -> None:
    advantage = 0.0
    next_value = 0.0
    for index in reversed(range(len(episode.decisions))):
        decision = episode.decisions[index]
        terminal_reward = episode.reward if index == len(episode.decisions) - 1 else 0.0
        delta = terminal_reward + gamma * next_value - decision.old_value
        advantage = delta + gamma * gae_lambda * advantage
        decision.advantage = float(advantage)
        decision.return_ = float(advantage + decision.old_value)
        next_value = decision.old_value


def rollout_game(
    engine: tuple[Any, Any, Any],
    own_deck: list[int],
    opponent: OpponentWorker,
    opponent_name: str,
    net: torch.nn.Module,
    residual: torch.nn.Module,
    features_module: Any,
    belief_class: Any,
    deck_library: Mapping[str, Any] | None,
    encode_public_tokens: Any,
    *,
    temperature: float,
    max_steps: int,
) -> Episode:
    battle_start, battle_select, battle_finish = engine
    opponent.reset()
    observation, start = battle_start(opponent.deck, own_deck)
    if observation is None:
        raise RuntimeError(
            f"official engine battle_start failed: {start.errorPlayer}/{start.errorType}"
        )
    belief = belief_class(own_deck, opponent_deck_priors=deck_library)
    memory = residual.initial_memory()
    decisions: list[DecisionRecord] = []
    metrics = EpisodeMetrics()
    started = time.perf_counter()
    winner: int | None = None
    steps = 0
    try:
        for steps in range(max_steps):
            current = observation.get("current") or {}
            first_player = current.get("firstPlayer")
            if first_player in (0, 1) and int(first_player) != 0:
                raise RuntimeError("training match did not place opponent first")
            update_public_metrics(metrics, observation)
            result = int(current.get("result") if current.get("result") is not None else -1)
            if result != -1:
                winner = None if result == 2 else result
                break
            selection = observation.get("select")
            if not isinstance(selection, Mapping):
                raise RuntimeError("official engine emitted no live selection")
            actor = int(current.get("yourIndex") or 0)
            if actor == 1:
                action, decision, memory = choose_action(
                    net,
                    residual,
                    memory,
                    observation,
                    belief,
                    features_module,
                    encode_public_tokens,
                    temperature=temperature,
                )
                decisions.append(decision)
                record_selected_action_metrics(
                    metrics,
                    observation,
                    action,
                    features_module,
                )
            elif actor == 0:
                action = opponent.act(dict(observation))
            else:
                raise RuntimeError(f"unexpected actor {actor}")
            if not legal(list(action), dict(selection)):
                raise RuntimeError(f"illegal training action actor={actor}: {action}")
            observation = battle_select(list(action))
        else:
            raise RuntimeError(f"training game exceeded {max_steps} decisions")
    finally:
        battle_finish()
    reward = 0.0 if winner is None else (1.0 if winner == 1 else -1.0)
    episode = Episode(
        decisions=decisions,
        reward=reward,
        opponent=opponent_name,
        engine_steps=steps,
        metrics=metrics,
        strategic_target=strategic_target(metrics),
        duration_seconds=time.perf_counter() - started,
    )
    return episode


def normalize_advantages(episodes: Sequence[Episode]) -> None:
    decisions = [decision for episode in episodes for decision in episode.decisions]
    values = np.asarray([decision.advantage for decision in decisions], dtype=np.float32)
    mean = float(values.mean())
    scale = float(values.std()) + 1.0e-8
    for decision in decisions:
        decision.normalized_advantage = (decision.advantage - mean) / scale


def ppo_update(
    residual: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, float]:
    normalize_advantages(episodes)
    totals: dict[str, float] = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "strategic_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "residual_l2": 0.0,
        "clip_fraction": 0.0,
        "saturation_fraction": 0.0,
        "gradient_norm": 0.0,
        "updates": 0.0,
    }
    residual.train()
    indices = list(range(len(episodes)))
    for _ in range(args.ppo_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), args.batch_episodes):
            selected = [episodes[index] for index in indices[start : start + args.batch_episodes]]
            policy_terms: list[torch.Tensor] = []
            value_terms: list[torch.Tensor] = []
            strategic_terms: list[torch.Tensor] = []
            entropy_terms: list[torch.Tensor] = []
            kl_terms: list[torch.Tensor] = []
            residual_terms: list[torch.Tensor] = []
            clipped_terms: list[torch.Tensor] = []
            saturated_terms: list[torch.Tensor] = []
            for episode in selected:
                memory = residual.initial_memory()
                target = episode.strategic_target
                for decision in episode.decisions:
                    overall, axes, _ = residual.values(memory)
                    value_terms.append(
                        torch_functional.smooth_l1_loss(
                            overall,
                            torch.tensor(decision.return_, dtype=torch.float32),
                        )
                    )
                    strategic_terms.append(
                        torch_functional.mse_loss(axes, target)
                    )
                    advantage = torch.tensor(
                        decision.normalized_advantage,
                        dtype=torch.float32,
                    )
                    for pick in decision.picks:
                        residual_logits = residual.option_residual(
                            pick.option_tokens,
                            memory,
                        )
                        combined = pick.base_logits + residual_logits
                        distribution = torch.distributions.Categorical(
                            logits=combined / args.temperature
                        )
                        base_distribution = torch.distributions.Categorical(
                            logits=pick.base_logits / args.temperature
                        )
                        action = torch.tensor(pick.action, dtype=torch.long)
                        new_logp = distribution.log_prob(action)
                        old_logp = torch.tensor(pick.old_logp, dtype=torch.float32)
                        ratio = torch.exp(new_logp - old_logp)
                        unclipped = ratio * advantage
                        clipped = torch.clamp(
                            ratio,
                            1.0 - args.clip,
                            1.0 + args.clip,
                        ) * advantage
                        policy_terms.append(-torch.minimum(unclipped, clipped))
                        entropy_terms.append(distribution.entropy())
                        kl_terms.append(
                            torch.distributions.kl_divergence(
                                base_distribution,
                                distribution,
                            )
                        )
                        finite_residual = residual_logits[torch.isfinite(pick.base_logits)]
                        residual_terms.append(finite_residual.square().mean())
                        clipped_terms.append(
                            ((ratio < 1.0 - args.clip) | (ratio > 1.0 + args.clip)).float()
                        )
                        saturated_terms.append(
                            (finite_residual.abs() >= residual.max_abs_bias - 1.0e-4)
                            .float()
                            .mean()
                        )
                    memory = residual.advance(decision.global_token, memory)
            if not policy_terms:
                raise RuntimeError("PPO batch contains no policy decisions")
            policy_loss = torch.stack(policy_terms).mean()
            value_loss = torch.stack(value_terms).mean()
            strategic_loss = torch.stack(strategic_terms).mean()
            entropy = torch.stack(entropy_terms).mean()
            kl = torch.stack(kl_terms).mean()
            residual_l2 = torch.stack(residual_terms).mean()
            loss = (
                policy_loss
                + args.value_coef * value_loss
                + args.strategic_coef * strategic_loss
                - args.entropy_coef * entropy
                + args.kl_coef * kl
                + args.residual_coef * residual_l2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                residual.parameters(),
                1.0,
            )
            optimizer.step()
            for name, value in (
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("strategic_loss", strategic_loss),
                ("entropy", entropy),
                ("kl", kl),
                ("residual_l2", residual_l2),
                ("clip_fraction", torch.stack(clipped_terms).mean()),
                ("saturation_fraction", torch.stack(saturated_terms).mean()),
            ):
                totals[name] += float(value.detach())
            totals["gradient_norm"] += float(gradient_norm)
            totals["updates"] += 1.0
    update_count = max(1.0, totals.pop("updates"))
    return {name: value / update_count for name, value in totals.items()}


def checkpoint(
    residual: torch.nn.Module,
    output: Path,
    *,
    base_model_hash: str,
    deck_hash: str,
    seed: int,
    iteration: int,
    provenance: Mapping[str, Any],
    residual_payload: Any,
) -> str:
    payload = residual_payload(
        residual,
        base_model_sha256=base_model_hash,
        deck_sha256=deck_hash,
        training_seed=seed,
        iteration=iteration,
        provenance=provenance,
    )
    torch.save(payload, output)
    return sha256(output)


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    candidate = args.candidate.resolve()
    opponents = [path.resolve() for path in (args.opponent or list(DEFAULT_OPPONENTS))]
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty neural-plan output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy  # type: ignore
    from neural_plan_residual_runtime import (
        NeuralPlanResidual,
        encode_public_tokens,
        residual_payload,
    )

    own_deck = read_deck(candidate)
    deck_hash = raw_deck_sha256(candidate / "deck.csv")
    base_model_hash = sha256(candidate / "model.pt")
    expected_deck = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
    expected_model = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
    if deck_hash != expected_deck or base_model_hash != expected_model:
        raise RuntimeError("N1 requires the frozen qualifying deck and blueprint")
    net = load_policy(candidate / "model.pt").cpu().eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    residual = NeuralPlanResidual(
        global_width=int(net.cfg["h"]),
        option_width=args.option_width,
        memory=args.memory,
        max_abs_bias=args.max_abs_bias,
    )
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    library_path = candidate / "deck_library.json"
    deck_library = (
        json.loads(library_path.read_text(encoding="utf-8"))
        if library_path.exists()
        else None
    )
    workers = {
        path.name: OpponentWorker(path, args.evaluator.resolve(), args.timeout)
        for path in opponents
    }
    opponent_hashes = {
        path.name: {
            "deck_sha256": sha256(path / "deck.csv"),
            "model_sha256": sha256(path / "model.pt"),
        }
        for path in opponents
    }
    rng = random.Random(args.seed)
    engine = (battle_start, battle_select, battle_finish)
    started = time.time()
    history: list[dict[str, Any]] = []
    total_games = 0
    try:
        for iteration in range(1, args.iterations + 1):
            schedule = []
            per_opponent = args.rollout_games // len(opponents)
            for path in opponents:
                schedule.extend([path.name] * per_opponent)
            rng.shuffle(schedule)
            residual.eval()
            episodes: list[Episode] = []
            wins: dict[str, int] = {name: 0 for name in workers}
            games: dict[str, int] = {name: 0 for name in workers}
            decisions = 0
            iteration_started = time.time()
            for game_index, opponent_name in enumerate(schedule, start=1):
                episode = rollout_game(
                    engine,
                    own_deck,
                    workers[opponent_name],
                    opponent_name,
                    net,
                    residual,
                    features,
                    Belief,
                    deck_library,
                    encode_public_tokens,
                    temperature=args.temperature,
                    max_steps=args.max_steps,
                )
                assign_gae(
                    episode,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )
                episodes.append(episode)
                games[opponent_name] += 1
                wins[opponent_name] += int(episode.reward > 0)
                decisions += len(episode.decisions)
                total_games += 1
                if game_index == 1 or game_index % 16 == 0 or game_index == len(schedule):
                    print(
                        json.dumps(
                            {
                                "iteration": iteration,
                                "rollout_game": game_index,
                                "rollout_target": len(schedule),
                                "opponent": opponent_name,
                                "reward": episode.reward,
                                "decisions": len(episode.decisions),
                                "total_training_games": total_games,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            update = ppo_update(residual, optimizer, episodes, args, rng)
            path = output / f"iteration_{iteration:02d}.pt"
            provenance = {
                "protocol": "rocket-neural-plan-residual/v1",
                "games_through_iteration": total_games,
                "opponents": list(workers),
            }
            checkpoint_hash = checkpoint(
                residual,
                path,
                base_model_hash=base_model_hash,
                deck_hash=deck_hash,
                seed=args.seed,
                iteration=iteration,
                provenance=provenance,
                residual_payload=residual_payload,
            )
            record = {
                "iteration": iteration,
                "rollout_games": len(episodes),
                "wins": wins,
                "games": games,
                "decisions": decisions,
                "update": update,
                "checkpoint": path.name,
                "checkpoint_sha256": checkpoint_hash,
                "elapsed_seconds": time.time() - iteration_started,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        for worker in workers.values():
            worker.close()

    final_path = output / "neural_plan_residual.pt"
    final_hash = checkpoint(
        residual,
        final_path,
        base_model_hash=base_model_hash,
        deck_hash=deck_hash,
        seed=args.seed,
        iteration=args.iterations,
        provenance={
            "protocol": "rocket-neural-plan-residual/v1",
            "games": total_games,
            "checkpoint_selection": "final_iteration_only",
            "opponents": list(workers),
        },
        residual_payload=residual_payload,
    )
    report = {
        "schema": "rocket-neural-plan-residual-training/v1",
        "protocol": "rocket-neural-plan-residual/v1",
        "official_engine_execution": True,
        "hidden_information_used_by_actor_or_critic": False,
        "search_branches_used": False,
        "candidate": str(candidate),
        "deck_sha256": deck_hash,
        "base_model_sha256": base_model_hash,
        "opponents": [str(path) for path in opponents],
        "opponent_hashes": opponent_hashes,
        "training_seed": args.seed,
        "hyperparameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool))
        },
        "trainable_parameters": sum(parameter.numel() for parameter in residual.parameters()),
        "frozen_blueprint_parameters": sum(parameter.numel() for parameter in net.parameters()),
        "iterations": history,
        "total_training_games": total_games,
        "final_artifact": final_path.name,
        "final_artifact_sha256": final_hash,
        "checkpoint_selection": "final iteration; no in-training performance selection",
        "elapsed_seconds": time.time() - started,
        "claim_boundary": "training telemetry only; not validation or arena strength evidence",
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "seed": args.seed,
                "total_training_games": total_games,
                "final_artifact": str(final_path),
                "final_artifact_sha256": final_hash,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> int:
    args = parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
