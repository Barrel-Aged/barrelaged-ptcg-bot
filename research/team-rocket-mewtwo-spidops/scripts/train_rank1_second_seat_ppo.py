#!/usr/bin/env python3
"""Constrained seat-conditioned PPO against a frozen official-engine opponent.

This is deliberately narrow.  It improves the Rocket policy's observed live
weakness when starting second, while anchoring every update to the broad
ScaleV2 blueprint.  Only the relational action head and value head are
trainable by default; the state/option transformer, public-belief features,
legality handling, and attack-safe end-turn guard remain frozen.

The opponent runs in a separate process so its ``soldier`` package cannot
collide with the candidate's package.  Games themselves execute through the
bundled official ``cg`` engine.  No hidden opponent cards are supplied to the
actor or critic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import random
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as Fnn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "candidates"
    / "leader_bc_scale_v2"
)
DEFAULT_OPPONENT = (
    PROJECT_ROOT
    / "work"
    / "arena_bundles"
    / "AshMarnieGeneralV110_extracted"
)
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rank1_second_ppo"
)


@dataclass
class Step:
    feat: dict[str, np.ndarray]
    action: int
    old_logp: float
    old_value: float
    advantage: float = 0.0
    return_: float = 0.0


@dataclass
class Episode:
    steps: list[Step]
    reward: float
    winner: int | None
    decisions: int
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--opponent",
        dest="opponents",
        type=Path,
        action="append",
        help=(
            "Frozen official-engine opponent bundle. Repeat for round-robin "
            "multi-opponent training; defaults to the historical Marnie bundle."
        ),
    )
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rollout-games", type=int, default=24)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--clip", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.35)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--kl-coef", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--seat",
        choices=("first", "second"),
        default="second",
        help="Seat occupied by the trainable Rocket policy (default: second).",
    )
    parser.add_argument(
        "--train-last-block",
        action="store_true",
        help="Also train the final transformer block (off by default).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(bundle: Path) -> list[int]:
    values = [
        int(value)
        for value in (bundle / "deck.csv")
        .read_text(encoding="utf-8")
        .replace(",", "\n")
        .split()
    ]
    if len(values) != 60:
        raise ValueError(f"{bundle}: expected 60 cards, found {len(values)}")
    return values


def legal(choice: list[int], selection: dict[str, Any]) -> bool:
    options = selection.get("option") or []
    minimum = min(max(0, int(selection.get("minCount") or 0)), len(options))
    maximum = min(max(0, int(selection.get("maxCount") or 0)), len(options))
    return (
        minimum <= len(choice) <= maximum
        and len(choice) == len(set(choice))
        and all(isinstance(value, int) and 0 <= value < len(options) for value in choice)
    )


class OpponentWorker:
    """Line-delimited JSON worker around the arena evaluator's process host."""

    def __init__(self, bundle: Path, evaluator: Path, timeout: float) -> None:
        self.bundle = bundle.resolve()
        self.deck = read_deck(self.bundle)
        self.timeout = timeout
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stderr = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="rocket_rank1_ppo_opponent_",
            suffix=".log",
            delete=False,
            encoding="utf-8",
        )
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(evaluator.resolve()),
                "--worker-agent-dir",
                str(self.bundle),
            ],
            cwd=str(self.bundle),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if line.strip():
                    self._queue.put(line.strip())
        finally:
            self._queue.put(None)

    def _request(self, observation: dict[str, Any]) -> list[int]:
        if self._proc.poll() is not None:
            raise RuntimeError(f"opponent worker exited: {self.stderr_tail()}")
        assert self._proc.stdin is not None
        self._proc.stdin.write(
            json.dumps({"obs": observation}, separators=(",", ":")) + "\n"
        )
        self._proc.stdin.flush()
        try:
            line = self._queue.get(timeout=self.timeout)
        except queue.Empty as error:
            raise RuntimeError("opponent worker timed out") from error
        if line is None:
            raise RuntimeError(f"opponent worker exited: {self.stderr_tail()}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("traceback") or self.stderr_tail())
        return list(response.get("action") or [])

    def reset(self) -> None:
        deck = self._request({"select": None})
        if deck != self.deck:
            raise RuntimeError("opponent deck handshake changed")

    def act(self, observation: dict[str, Any]) -> list[int]:
        return self._request(observation)

    def stderr_tail(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read()[-4000:]
        except Exception:
            return ""

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                assert self._proc.stdin is not None
                self._proc.stdin.write('{"cmd":"stop"}\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._stderr.close()


def condition_attack_safety(
    feat: dict[str, np.ndarray], selection: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Encode the runtime attack guard in the train/eval action mask."""
    options = selection.get("option") or []
    if not any(option.get("type") == 13 for option in options):
        return feat
    end_indices = [
        index for index, option in enumerate(options) if option.get("type") == 14
    ]
    if not end_indices:
        return feat
    conditioned = {key: value.copy() for key, value in feat.items()}
    available = conditioned.get(
        "opt_avail", np.ones(len(options), dtype=np.float32)
    ).copy()
    available[end_indices] = 0.0
    conditioned["opt_avail"] = available
    return conditioned


def choose_action(
    model: torch.nn.Module,
    observation: dict[str, Any],
    belief: Any,
    features: Any,
    feat_to_tensors: Any,
    *,
    greedy: bool,
    temperature: float,
    collect: bool,
) -> tuple[list[int], list[Step]]:
    selection = observation["select"]
    options = selection.get("option") or []
    minimum = min(max(0, int(selection.get("minCount") or 0)), len(options))
    maximum = min(max(0, int(selection.get("maxCount") or 0)), len(options))
    if maximum == 0:
        return [], []
    base = features.extract(
        observation,
        belief=belief,
        slot_link=bool(getattr(model, "slot_link", False)),
    )
    base = condition_attack_safety(base, selection)
    picked: list[int] = []
    trajectory: list[Step] = []
    target_count = maximum
    for _ in range(target_count):
        feat = features.mark_selection(base, picked, target_count)
        tensors = feat_to_tensors(feat, "cpu")
        with torch.no_grad():
            logits, value = model(tensors)
            distribution = torch.distributions.Categorical(
                logits=logits / max(temperature, 1.0e-6)
            )
            action_tensor = (
                torch.argmax(logits)
                if greedy
                else distribution.sample()
            )
            action = int(action_tensor)
            logp = float(distribution.log_prob(action_tensor))
        if collect:
            trajectory.append(
                Step(
                    feat=feat,
                    action=action,
                    old_logp=logp,
                    old_value=float(value),
                )
            )
        picked.append(action)
    if len(picked) < minimum:
        raise RuntimeError("sampled fewer actions than the selection minimum")
    return picked, trajectory


def rollout_game(
    engine: tuple[Any, Any, Any],
    our_deck: list[int],
    opponent: OpponentWorker,
    model: torch.nn.Module,
    features: Any,
    feat_to_tensors: Any,
    belief_class: Any,
    opponent_priors: dict[str, Any] | None,
    own_seat: int,
    *,
    greedy: bool,
    temperature: float,
    collect: bool,
    max_steps: int,
) -> Episode:
    battle_start, battle_select, battle_finish = engine
    opponent.reset()
    decks = (our_deck, opponent.deck) if own_seat == 0 else (opponent.deck, our_deck)
    observation, _ = battle_start(*decks)
    if observation is None:
        raise RuntimeError("official engine battle_start failed")
    belief = belief_class(our_deck, opponent_deck_priors=opponent_priors)
    steps: list[Step] = []
    decision_count = 0
    started = time.perf_counter()
    winner: int | None = None
    try:
        for _ in range(max_steps):
            current = observation.get("current") or {}
            result = current.get("result", -1)
            if result != -1:
                winner = None if int(result) == 2 else int(result)
                break
            selection = observation.get("select")
            if not isinstance(selection, dict):
                raise RuntimeError("official engine emitted no live selection")
            actor = current.get("yourIndex")
            if actor == own_seat:
                choice, decision_steps = choose_action(
                    model,
                    observation,
                    belief,
                    features,
                    feat_to_tensors,
                    greedy=greedy,
                    temperature=temperature,
                    collect=collect,
                )
                steps.extend(decision_steps)
                decision_count += 1
            elif actor == 1 - own_seat:
                choice = opponent.act(observation)
            else:
                raise RuntimeError(f"unexpected acting player: {actor!r}")
            if not legal(choice, selection):
                raise RuntimeError(
                    f"illegal action from player {actor}: {choice} for {selection}"
                )
            observation = battle_select(choice)
        else:
            raise RuntimeError(f"game exceeded {max_steps} engine decisions")
    finally:
        battle_finish()
    reward = 0.0 if winner is None else (1.0 if winner == own_seat else -1.0)
    return Episode(
        steps=steps,
        reward=reward,
        winner=winner,
        decisions=decision_count,
        duration_seconds=time.perf_counter() - started,
    )


def assign_gae(
    steps: list[Step], reward: float, gamma: float, gae_lambda: float
) -> None:
    advantage = 0.0
    next_value = 0.0
    for index in reversed(range(len(steps))):
        terminal_reward = reward if index == len(steps) - 1 else 0.0
        delta = terminal_reward + gamma * next_value - steps[index].old_value
        advantage = delta + gamma * gae_lambda * advantage
        steps[index].advantage = float(advantage)
        steps[index].return_ = float(advantage + steps[index].old_value)
        next_value = steps[index].old_value


def collate(steps: list[Step], device: torch.device) -> dict[str, torch.Tensor]:
    """Pad V110/v10 variable-card and variable-option feature tensors."""
    feats = [step.feat for step in steps]
    batch_count = len(feats)
    poke_count = max(feat["poke_feats"].shape[0] for feat in feats)
    option_count = max(feat["opt_feats"].shape[0] for feat in feats)

    def maximum(key: str) -> int:
        return max(feat[key].shape[0] for feat in feats)

    global_features = np.zeros(
        (batch_count, feats[0]["global"].shape[0]), dtype=np.float32
    )
    poke_features = np.zeros(
        (batch_count, poke_count, feats[0]["poke_feats"].shape[1]),
        dtype=np.float32,
    )
    poke_card = np.zeros((batch_count, poke_count), dtype=np.int64)
    poke_mask = np.zeros((batch_count, poke_count), dtype=np.float32)
    option_features = np.zeros(
        (batch_count, option_count, feats[0]["opt_feats"].shape[1]),
        dtype=np.float32,
    )
    option_card = np.zeros((batch_count, option_count), dtype=np.int64)
    option_attack = np.zeros((batch_count, option_count), dtype=np.int64)
    option_target = np.zeros((batch_count, option_count), dtype=np.int64)
    option_mask = np.zeros((batch_count, option_count), dtype=np.float32)

    def zone(key: str) -> tuple[np.ndarray, np.ndarray]:
        size = maximum(key)
        return (
            np.zeros((batch_count, size), dtype=np.int64),
            np.zeros((batch_count, size), dtype=np.float32),
        )

    hand_card, hand_mask = zone("hand_card")
    my_discard_card, my_discard_mask = zone("my_discard_card")
    opp_discard_card, opp_discard_mask = zone("opp_discard_card")
    my_unseen_card, my_unseen_mask = zone("my_unseen_card")
    opp_unseen_card, opp_unseen_mask = zone("opp_unseen_card")
    known_prized_card, known_prized_mask = zone("known_prized_card")
    stadium_card = np.zeros(batch_count, dtype=np.int64)
    effect_card = np.zeros(batch_count, dtype=np.int64)
    belief_global = np.zeros(
        (batch_count, feats[0]["belief_global"].shape[0]), dtype=np.float32
    )
    option_unseen = np.zeros((batch_count, option_count, 1), dtype=np.float32)
    belief_prized = np.zeros(
        (batch_count, feats[0]["belief_prized"].shape[0]), dtype=np.float32
    )

    for row, feat in enumerate(feats):
        pokes = feat["poke_feats"].shape[0]
        options = feat["opt_feats"].shape[0]
        global_features[row] = feat["global"]
        poke_features[row, :pokes] = feat["poke_feats"]
        poke_card[row, :pokes] = feat["poke_card"]
        poke_mask[row, : max(pokes - 1, 0)] = 1.0
        option_features[row, :options] = feat["opt_feats"]
        option_card[row, :options] = feat["opt_card"]
        option_attack[row, :options] = feat["opt_attack"]
        option_target[row, :options] = feat["opt_target"]
        option_mask[row, :options] = feat.get(
            "opt_avail", np.ones(options, dtype=np.float32)
        )
        for key, values, masks in (
            ("hand_card", hand_card, hand_mask),
            ("my_discard_card", my_discard_card, my_discard_mask),
            ("opp_discard_card", opp_discard_card, opp_discard_mask),
            ("my_unseen_card", my_unseen_card, my_unseen_mask),
            ("opp_unseen_card", opp_unseen_card, opp_unseen_mask),
            ("known_prized_card", known_prized_card, known_prized_mask),
        ):
            count = feat[key].shape[0]
            values[row, :count] = feat[key]
            masks[row, :count] = (feat[key] != 0).astype(np.float32)
        stadium_card[row] = int(feat["stadium_card"][0])
        effect_card[row] = int(feat["effect_card"][0])
        belief_global[row] = feat["belief_global"]
        option_unseen[row, :options] = feat["opt_unseen"]
        belief_prized[row] = feat["belief_prized"]

    def tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=device)

    return {
        "global": tensor(global_features, torch.float32),
        "poke_feats": tensor(poke_features, torch.float32),
        "poke_card": tensor(poke_card, torch.int64),
        "poke_mask": tensor(poke_mask, torch.float32),
        "opt_feats": tensor(option_features, torch.float32),
        "opt_card": tensor(option_card, torch.int64),
        "opt_attack": tensor(option_attack, torch.int64),
        "opt_target": tensor(option_target, torch.int64),
        "opt_mask": tensor(option_mask, torch.float32),
        "hand_card": tensor(hand_card, torch.int64),
        "hand_mask": tensor(hand_mask, torch.float32),
        "my_discard_card": tensor(my_discard_card, torch.int64),
        "my_discard_mask": tensor(my_discard_mask, torch.float32),
        "opp_discard_card": tensor(opp_discard_card, torch.int64),
        "opp_discard_mask": tensor(opp_discard_mask, torch.float32),
        "stadium_card": tensor(stadium_card, torch.int64),
        "my_unseen_card": tensor(my_unseen_card, torch.int64),
        "my_unseen_mask": tensor(my_unseen_mask, torch.float32),
        "opp_unseen_card": tensor(opp_unseen_card, torch.int64),
        "opp_unseen_mask": tensor(opp_unseen_mask, torch.float32),
        "belief_global": tensor(belief_global, torch.float32),
        "opt_unseen": tensor(option_unseen, torch.float32),
        "known_prized_card": tensor(known_prized_card, torch.int64),
        "known_prized_mask": tensor(known_prized_mask, torch.float32),
        "effect_card": tensor(effect_card, torch.int64),
        "belief_prized": tensor(belief_prized, torch.float32),
    }


def ppo_update(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: list[Step],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    advantages = np.asarray([step.advantage for step in steps], dtype=np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    order = np.arange(len(steps))
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
        "updates": 0.0,
    }
    model.train()
    for _ in range(args.ppo_epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            selected = [steps[int(index)] for index in indices]
            batch = collate(selected, device)
            actions = torch.as_tensor(
                [step.action for step in selected], dtype=torch.long, device=device
            )
            old_logp = torch.as_tensor(
                [step.old_logp for step in selected],
                dtype=torch.float32,
                device=device,
            )
            returns = torch.as_tensor(
                [step.return_ for step in selected],
                dtype=torch.float32,
                device=device,
            )
            adv = torch.as_tensor(
                advantages[indices], dtype=torch.float32, device=device
            )
            logits, values = model.forward_batch(batch)
            with torch.no_grad():
                reference_logits, _ = reference.forward_batch(batch)
            distribution = torch.distributions.Categorical(
                logits=logits / args.temperature
            )
            reference_distribution = torch.distributions.Categorical(
                logits=reference_logits / args.temperature
            )
            new_logp = distribution.log_prob(actions)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * adv
            clipped = torch.clamp(
                ratio, 1.0 - args.clip, 1.0 + args.clip
            ) * adv
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = Fnn.smooth_l1_loss(values, returns)
            entropy = distribution.entropy().mean()
            kl = torch.distributions.kl_divergence(
                reference_distribution, distribution
            ).mean()
            loss = (
                policy_loss
                + args.value_coef * value_loss
                - args.entropy_coef * entropy
                + args.kl_coef * kl
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                1.0,
            )
            optimizer.step()
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["kl"] += float(kl.detach())
            totals["clip_fraction"] += float(
                ((ratio < 1.0 - args.clip) | (ratio > 1.0 + args.clip))
                .float()
                .mean()
            )
            totals["updates"] += 1.0
    count = max(1.0, totals.pop("updates"))
    return {key: value / count for key, value in totals.items()}


def evaluate_policy(
    count: int,
    engine: tuple[Any, Any, Any],
    deck: list[int],
    opponents: list[OpponentWorker],
    model: torch.nn.Module,
    features: Any,
    feat_to_tensors: Any,
    belief_class: Any,
    opponent_priors: dict[str, Any] | None,
    own_seat: int,
    max_steps: int,
) -> dict[str, float | int]:
    model.eval()
    wins = draws = decisions = 0
    durations: list[float] = []
    for game in range(count):
        opponent = opponents[game % len(opponents)]
        episode = rollout_game(
            engine,
            deck,
            opponent,
            model,
            features,
            feat_to_tensors,
            belief_class,
            opponent_priors,
            own_seat,
            greedy=True,
            temperature=1.0,
            collect=False,
            max_steps=max_steps,
        )
        wins += int(episode.reward > 0)
        draws += int(episode.reward == 0)
        decisions += episode.decisions
        durations.append(episode.duration_seconds)
        print(
            f"eval {game + 1:03d}/{count}: reward={episode.reward:+.0f} "
            f"decisions={episode.decisions} seconds={episode.duration_seconds:.2f}",
            flush=True,
        )
    return {
        "games": count,
        "wins": wins,
        "losses": count - wins - draws,
        "draws": draws,
        "win_rate": wins / max(1, count),
        "decisions": decisions,
        "mean_game_seconds": float(np.mean(durations)) if durations else 0.0,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    candidate = args.candidate.resolve()
    opponent_paths = [
        path.resolve() for path in (args.opponents or [DEFAULT_OPPONENT])
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(candidate))

    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import feat_to_tensors, load_policy, save_policy  # type: ignore

    model_path = candidate / "model.pt"
    model = load_policy(model_path, map_location=device).to(device)
    reference = load_policy(model_path, map_location=device).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.logit_head.parameters():
        parameter.requires_grad_(True)
    for parameter in model.value.parameters():
        parameter.requires_grad_(True)
    if args.train_last_block:
        for parameter in model.encoder.layers[-1].parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    deck = read_deck(candidate)
    deck_library_path = candidate / "deck_library.json"
    opponent_priors = (
        json.loads(deck_library_path.read_text(encoding="utf-8"))
        if deck_library_path.exists()
        else None
    )
    engine = (battle_start, battle_select, battle_finish)
    opponents = [
        OpponentWorker(path, args.evaluator, args.timeout)
        for path in opponent_paths
    ]
    own_seat = 0 if args.seat == "first" else 1
    started = time.time()
    history: list[dict[str, Any]] = []
    best_win_rate = -1.0
    best_path = output / "best_model.pt"
    try:
        baseline = evaluate_policy(
            args.eval_games,
            engine,
            deck,
            opponents,
            model,
            features,
            feat_to_tensors,
            Belief,
            opponent_priors,
            own_seat,
            args.max_steps,
        )
        print(json.dumps({"baseline": baseline}, sort_keys=True), flush=True)
        for iteration in range(1, args.iterations + 1):
            model.eval()
            episodes: list[Episode] = []
            rollout_steps: list[Step] = []
            rollout_wins = rollout_draws = 0
            for game in range(args.rollout_games):
                opponent = opponents[game % len(opponents)]
                episode = rollout_game(
                    engine,
                    deck,
                    opponent,
                    model,
                    features,
                    feat_to_tensors,
                    Belief,
                    opponent_priors,
                    own_seat,
                    greedy=False,
                    temperature=args.temperature,
                    collect=True,
                    max_steps=args.max_steps,
                )
                assign_gae(
                    episode.steps,
                    episode.reward,
                    args.gamma,
                    args.gae_lambda,
                )
                episodes.append(episode)
                rollout_steps.extend(episode.steps)
                rollout_wins += int(episode.reward > 0)
                rollout_draws += int(episode.reward == 0)
                print(
                    f"iter {iteration} rollout {game + 1:03d}/{args.rollout_games}: "
                    f"reward={episode.reward:+.0f} steps={len(episode.steps)}",
                    flush=True,
                )
            update = ppo_update(
                model, reference, optimizer, rollout_steps, args, device
            )
            evaluation = evaluate_policy(
                args.eval_games,
                engine,
                deck,
                opponents,
                model,
                features,
                feat_to_tensors,
                Belief,
                opponent_priors,
                own_seat,
                args.max_steps,
            )
            checkpoint = output / f"iteration_{iteration:02d}.pt"
            save_policy(model, checkpoint)
            record = {
                "iteration": iteration,
                "rollout": {
                    "games": args.rollout_games,
                    "wins": rollout_wins,
                    "losses": args.rollout_games - rollout_wins - rollout_draws,
                    "draws": rollout_draws,
                    "steps": len(rollout_steps),
                },
                "update": update,
                "evaluation": evaluation,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if float(evaluation["win_rate"]) > best_win_rate:
                best_win_rate = float(evaluation["win_rate"])
                save_policy(model, best_path)
    finally:
        for opponent in opponents:
            opponent.close()

    report = {
        "schema_version": 2,
        "method": "seat_conditioned_constrained_ppo",
        "official_engine_execution": True,
        "actor_information": "public observation plus own private zones",
        "opponent_hidden_hand_used_by_actor_or_critic": False,
        "candidate": str(candidate),
        "candidate_model_sha256": sha256(model_path),
        "candidate_deck_sha256": hashlib.sha256(
            ",".join(str(card) for card in deck).encode("ascii")
        ).hexdigest(),
        "opponents": [
            {
                "path": str(path),
                "model_sha256": (
                    sha256(path / "model.pt")
                    if (path / "model.pt").is_file()
                    else None
                ),
            }
            for path in opponent_paths
        ],
        "seat": own_seat,
        "seat_role": args.seat,
        "opponent_count": len(opponent_paths),
        "first_player_expected": 0,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_last_block": bool(args.train_last_block),
        "hyperparameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (int, float, str, bool))
        },
        "baseline": baseline,
        "iterations": history,
        "best_model": best_path.name,
        "best_model_sha256": sha256(best_path),
        "best_observed_eval_win_rate": best_win_rate,
        "elapsed_seconds": time.time() - started,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    report = train(args)
    print(
        json.dumps(
            {
                "complete": True,
                "best_observed_eval_win_rate": report[
                    "best_observed_eval_win_rate"
                ],
                "best_model": str(Path(args.output).resolve() / "best_model.pt"),
                "elapsed_seconds": report["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
