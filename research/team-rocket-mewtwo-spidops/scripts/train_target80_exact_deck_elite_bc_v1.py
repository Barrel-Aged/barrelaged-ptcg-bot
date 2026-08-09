#!/usr/bin/env python3
"""Exact-deck elite behavioral cloning for the Target-80 Rocket candidate.

The official ``cg`` engine generates stochastic trajectories with the exact
zero-Wobbuffet Hancang deck.  Only decisions from winning trajectories are
used as supervised labels.  Updates remain anchored to the frozen starting
policy with a forward KL penalty, and the value function is never trained.

This is a development experiment, not a claim of game-theoretic solving or
hidden-hand inference.  Actors receive only the official public observation
plus their own private zones.  Opponents execute in isolated evaluator worker
processes so bundle imports cannot collide.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "work"
    / "target80_candidates"
    / "hancang_wobbfree_scale_control_v1"
)
DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs"
    r"\ptcg_agent_lab\tools\evaluate.py"
)
EXPECTED_DECK_FILE_SHA256 = (
    "0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8"
)
EXPECTED_MODEL_SHA256 = (
    "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
)
WOBBUFFET_CARD_ID = 432

sys.path.insert(0, str(SCRIPTS_ROOT))
from train_rank1_second_seat_ppo import (  # noqa: E402
    Episode,
    OpponentWorker,
    Step,
    collate,
    read_deck,
    rollout_game,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--opponent",
        dest="opponents",
        type=Path,
        action="append",
        required=True,
        help="Frozen opponent bundle; repeat for the locked round robin.",
    )
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rollout-games-per-opponent", type=int, default=28)
    parser.add_argument("--eval-games-per-opponent", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--kl-coef", type=float, default=0.10)
    parser.add_argument("--entropy-coef", type=float, default=5.0e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=806050)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--seat",
        choices=("first", "second"),
        default="first",
        help="Seat occupied by Rocket. The live arena convention is first.",
    )
    parser.add_argument(
        "--train-last-block",
        action="store_true",
        help="Train the final relational-transformer block as well as logit_head.",
    )
    parser.add_argument(
        "--expected-deck-sha256",
        default=EXPECTED_DECK_FILE_SHA256,
        help="Hard identity lock for the exact 60-card deck.csv file.",
    )
    parser.add_argument(
        "--expected-model-sha256",
        default=EXPECTED_MODEL_SHA256,
        help="Hard identity lock for the frozen starting checkpoint.",
    )
    return parser.parse_args()


def semantic_tree_sha256(root: Path) -> str:
    """Hash bundle semantics while excluding runtime-generated cache markers."""
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix.lower() not in (".pyc", ".pyo")
            and path.name != ".src_mtime"
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


def bundle_identity(path: Path) -> dict[str, Any]:
    selected = {}
    for name in (
        "deck.csv",
        "main.py",
        "model.pt",
        "cf_model_opp_abomasnow_seat1_single.pt",
    ):
        file_path = path / name
        if file_path.is_file():
            selected[name] = sha256(file_path)
    return {
        "path": str(path),
        "semantic_tree_sha256": semantic_tree_sha256(path),
        "selected_file_sha256": selected,
    }


def trainable_parameters(
    model: torch.nn.Module, train_last_block: bool
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.logit_head.parameters():
        parameter.requires_grad_(True)
    if train_last_block:
        if not hasattr(model, "encoder") or not model.encoder.layers:
            raise ValueError("model has no final encoder block to train")
        for parameter in model.encoder.layers[-1].parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_mask: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Cross entropy whose smoothing mass covers legal options only."""
    legal = option_mask.bool()
    # ``forward_batch`` represents padded options as -inf.  Replacing those
    # values before log_softmax is important for backward stability: masking a
    # later ``0 * -inf`` expression with torch.where can still leave NaN
    # gradients in the unselected branch.
    safe_logits = logits.masked_fill(~legal, -1.0e9)
    log_probabilities = torch_functional.log_softmax(safe_logits, dim=1)
    negative_log_likelihood = -log_probabilities.gather(
        1, labels[:, None]
    ).squeeze(1)
    if smoothing <= 0.0:
        return negative_log_likelihood.mean()
    smooth_loss = -log_probabilities.masked_fill(~legal, 0.0).sum(dim=1)
    smooth_loss = smooth_loss / legal.sum(dim=1).clamp(min=1)
    return (
        (1.0 - smoothing) * negative_log_likelihood + smoothing * smooth_loss
    ).mean()


def masked_reference_kl(
    logits: torch.Tensor,
    reference_logits: torch.Tensor,
    option_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward KL(reference || candidate) on the shared legal support."""
    legal = option_mask.bool()
    safe_logits = logits.masked_fill(~legal, -1.0e9)
    safe_reference_logits = reference_logits.masked_fill(~legal, -1.0e9)
    log_probability = torch_functional.log_softmax(safe_logits, dim=1)
    reference_log_probability = torch_functional.log_softmax(
        safe_reference_logits, dim=1
    )
    reference_probability = reference_log_probability.exp()
    terms = torch.where(
        legal,
        reference_probability * (reference_log_probability - log_probability),
        torch.zeros_like(log_probability),
    )
    return terms.sum(dim=1).mean()


def masked_entropy(
    logits: torch.Tensor, option_mask: torch.Tensor
) -> torch.Tensor:
    legal = option_mask.bool()
    safe_logits = logits.masked_fill(~legal, -1.0e9)
    log_probability = torch_functional.log_softmax(safe_logits, dim=1)
    probability = log_probability.exp()
    terms = torch.where(
        legal,
        -probability * log_probability,
        torch.zeros_like(log_probability),
    )
    return terms.sum(dim=1).mean()


def useful_steps(episode: Episode) -> list[Step]:
    """Drop forced one-option decisions, which carry no tactical signal."""
    if episode.reward <= 0:
        return []
    answer = []
    for step in episode.steps:
        availability = step.feat.get("opt_avail")
        legal_count = int(np.sum(availability >= 0.5)) if availability is not None else 0
        if legal_count > 1:
            answer.append(step)
    return answer


def balanced_winner_steps(
    by_opponent: dict[str, list[Step]], rng: random.Random
) -> tuple[list[Step], dict[str, int]]:
    """Sample equal opponent mass without increasing the original data volume."""
    available = [name for name, steps in by_opponent.items() if steps]
    if not available:
        raise RuntimeError("no winning multi-option steps were collected")
    total = sum(len(by_opponent[name]) for name in available)
    balanced = [
        rng.choice(by_opponent[available[index % len(available)]])
        for index in range(total)
    ]
    rng.shuffle(balanced)
    counts = {name: 0 for name in by_opponent}
    for index in range(total):
        counts[available[index % len(available)]] += 1
    return balanced, counts


def update_policy(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: list[Step],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    totals = {
        "cross_entropy": 0.0,
        "kl": 0.0,
        "entropy": 0.0,
        "loss": 0.0,
        "updates": 0,
    }
    order = np.arange(len(steps))
    model.train()
    for _ in range(args.epochs):
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            selected = [steps[int(index)] for index in indices]
            batch = collate(selected, device)
            labels = torch.as_tensor(
                [step.action for step in selected],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model.forward_batch(batch)
            with torch.no_grad():
                reference_logits, _ = reference.forward_batch(batch)
            cross_entropy = masked_cross_entropy(
                logits,
                labels,
                batch["opt_mask"],
                args.label_smoothing,
            )
            kl = masked_reference_kl(
                logits, reference_logits, batch["opt_mask"]
            )
            entropy = masked_entropy(logits, batch["opt_mask"])
            loss = (
                cross_entropy
                + args.kl_coef * kl
                - args.entropy_coef * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "non-finite elite-BC loss before optimizer step"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
            )
            optimizer.step()
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
                if parameter.requires_grad
            ):
                raise FloatingPointError(
                    "non-finite trainable parameter after optimizer step"
                )
            totals["cross_entropy"] += float(cross_entropy.detach())
            totals["kl"] += float(kl.detach())
            totals["entropy"] += float(entropy.detach())
            totals["loss"] += float(loss.detach())
            totals["updates"] += 1
    updates = max(1, int(totals["updates"]))
    return {
        "training_steps": len(steps),
        "updates": int(totals["updates"]),
        "mean_cross_entropy": totals["cross_entropy"] / updates,
        "mean_kl": totals["kl"] / updates,
        "mean_entropy": totals["entropy"] / updates,
        "mean_loss": totals["loss"] / updates,
    }


def evaluate_matrix(
    engine: tuple[Any, Any, Any],
    deck: list[int],
    opponents: list[tuple[str, OpponentWorker]],
    model: torch.nn.Module,
    features: Any,
    feat_to_tensors: Any,
    belief_class: Any,
    opponent_priors: dict[str, Any] | None,
    own_seat: int,
    games_per_opponent: int,
    max_steps: int,
) -> dict[str, Any]:
    model.eval()
    per_opponent: dict[str, dict[str, Any]] = {}
    total_wins = total_losses = total_draws = total_decisions = 0
    total_seconds = 0.0
    for name, worker in opponents:
        wins = losses = draws = decisions = 0
        seconds = 0.0
        for game in range(games_per_opponent):
            episode = rollout_game(
                engine,
                deck,
                worker,
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
            losses += int(episode.reward < 0)
            draws += int(episode.reward == 0)
            decisions += episode.decisions
            seconds += episode.duration_seconds
            print(
                f"eval {name} {game + 1:02d}/{games_per_opponent}: "
                f"reward={episode.reward:+.0f}",
                flush=True,
            )
        per_opponent[name] = {
            "games": games_per_opponent,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / max(1, games_per_opponent),
            "decisions": decisions,
            "mean_game_seconds": seconds / max(1, games_per_opponent),
        }
        total_wins += wins
        total_losses += losses
        total_draws += draws
        total_decisions += decisions
        total_seconds += seconds
    games = games_per_opponent * len(opponents)
    return {
        "games": games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "win_rate": total_wins / max(1, games),
        "decisions": total_decisions,
        "mean_game_seconds": total_seconds / max(1, games),
        "minimum_opponent_wins": min(
            result["wins"] for result in per_opponent.values()
        ),
        "per_opponent": per_opponent,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(
            f"refusing to reuse output directory: {args.output.resolve()}"
        )
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.rollout_games_per_opponent < 1:
        raise ValueError("rollout-games-per-opponent must be positive")
    if args.eval_games_per_opponent < 1:
        raise ValueError("eval-games-per-opponent must be positive")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if len(args.opponents) < 2:
        raise ValueError("the robustness experiment requires at least two opponents")
    names = [path.resolve().name for path in args.opponents]
    if len(names) != len(set(names)):
        raise ValueError("opponent bundle names must be unique")


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    python_rng = random.Random(args.seed + 17)
    numpy_rng = np.random.default_rng(args.seed + 29)
    device = torch.device("cpu")

    candidate = args.candidate.resolve()
    opponent_paths = [path.resolve() for path in args.opponents]
    evaluator = args.evaluator.resolve()
    output = args.output.resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    model_path = candidate / "model.pt"
    deck_path = candidate / "deck.csv"
    if sha256(deck_path) != args.expected_deck_sha256:
        raise RuntimeError("candidate deck identity lock failed")
    if sha256(model_path) != args.expected_model_sha256:
        raise RuntimeError("candidate model identity lock failed")
    deck = read_deck(candidate)
    if WOBBUFFET_CARD_ID in deck:
        raise RuntimeError("zero-Wobbuffet hard constraint failed")
    for path in opponent_paths:
        read_deck(path)

    identities_before = {
        "candidate": bundle_identity(candidate),
        "opponents": [bundle_identity(path) for path in opponent_paths],
    }
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(candidate))

    from cg.game import battle_finish, battle_select, battle_start  # type: ignore
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import feat_to_tensors, load_policy, save_policy  # type: ignore

    model = load_policy(model_path, map_location=device).to(device)
    reference = load_policy(model_path, map_location=device).to(device).eval()
    if int(getattr(model, "mem", 0)) != 0:
        raise ValueError("this trainer currently requires a memory-free policy")
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainable = trainable_parameters(model, args.train_last_block)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    deck_library_path = candidate / "deck_library.json"
    opponent_priors = (
        json.loads(deck_library_path.read_text(encoding="utf-8"))
        if deck_library_path.is_file()
        else None
    )
    engine = (battle_start, battle_select, battle_finish)
    workers = [
        (path.name, OpponentWorker(path, evaluator, args.timeout))
        for path in opponent_paths
    ]
    own_seat = 0 if args.seat == "first" else 1
    started = time.time()
    baseline: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    best_key = (-1, -1)
    best_path = output / "best_model.pt"
    try:
        baseline = evaluate_matrix(
            engine,
            deck,
            workers,
            model,
            features,
            feat_to_tensors,
            Belief,
            opponent_priors,
            own_seat,
            args.eval_games_per_opponent,
            args.max_steps,
        )
        print(json.dumps({"baseline_complete": baseline}, sort_keys=True), flush=True)
        for iteration in range(1, args.iterations + 1):
            model.eval()
            by_opponent: dict[str, list[Step]] = {
                name: [] for name, _ in workers
            }
            rollout_summary: dict[str, dict[str, int]] = {
                name: {
                    "games": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "winning_multioption_steps": 0,
                }
                for name, _ in workers
            }
            for name, worker in workers:
                for game in range(args.rollout_games_per_opponent):
                    episode = rollout_game(
                        engine,
                        deck,
                        worker,
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
                    selected = useful_steps(episode)
                    by_opponent[name].extend(selected)
                    summary = rollout_summary[name]
                    summary["games"] += 1
                    summary["wins"] += int(episode.reward > 0)
                    summary["losses"] += int(episode.reward < 0)
                    summary["draws"] += int(episode.reward == 0)
                    summary["winning_multioption_steps"] += len(selected)
                    print(
                        f"iteration {iteration} rollout {name} "
                        f"{game + 1:02d}/{args.rollout_games_per_opponent}: "
                        f"reward={episode.reward:+.0f} retained={len(selected)}",
                        flush=True,
                    )
            training_steps, sampled_by_opponent = balanced_winner_steps(
                by_opponent, python_rng
            )
            update = update_policy(
                model,
                reference,
                optimizer,
                training_steps,
                args,
                device,
                numpy_rng,
            )
            evaluation = evaluate_matrix(
                engine,
                deck,
                workers,
                model,
                features,
                feat_to_tensors,
                Belief,
                opponent_priors,
                own_seat,
                args.eval_games_per_opponent,
                args.max_steps,
            )
            checkpoint = output / f"iteration_{iteration:02d}.pt"
            save_policy(model, checkpoint)
            selection_key = (
                int(evaluation["wins"]),
                int(evaluation["minimum_opponent_wins"]),
            )
            record = {
                "iteration": iteration,
                "rollout": rollout_summary,
                "sampled_training_steps_by_opponent": sampled_by_opponent,
                "update": update,
                "evaluation": evaluation,
                "selection_key": list(selection_key),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
            }
            history.append(record)
            print(json.dumps({"iteration_complete": record}, sort_keys=True), flush=True)
            if selection_key > best_key:
                best_key = selection_key
                save_policy(model, best_path)
    finally:
        for _, worker in workers:
            worker.close()

    if baseline is None or not best_path.is_file():
        raise RuntimeError("training did not produce a complete baseline and checkpoint")
    identities_after = {
        "candidate": bundle_identity(candidate),
        "opponents": [bundle_identity(path) for path in opponent_paths],
    }
    if identities_after != identities_before:
        raise RuntimeError("an input bundle changed semantically during training")

    report = {
        "schema_version": 1,
        "method": "exact_deck_elite_behavioral_cloning_with_frozen_blueprint_kl",
        "official_engine_execution": True,
        "official_engine_scope": "trajectory generation and greedy evaluation",
        "abstract_scope": "winning-trajectory selection and supervised policy update",
        "game_theoretic_solver": False,
        "actor_information": "public observation plus own private zones",
        "opponent_hidden_hand_used_by_actor_or_trainer": False,
        "candidate": str(candidate),
        "candidate_deck_file_sha256": sha256(deck_path),
        "candidate_model_sha256": sha256(model_path),
        "candidate_zero_wobbuffet": WOBBUFFET_CARD_ID not in deck,
        "input_identities_before": identities_before,
        "input_identities_after": identities_after,
        "opponent_order": [name for name, _ in workers],
        "seat": own_seat,
        "seat_role": args.seat,
        "first_player_expected": 0,
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "train_last_block": bool(args.train_last_block),
        "hyperparameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (int, float, str, bool))
        },
        "baseline": baseline,
        "iterations": history,
        "best_selection_key": list(best_key),
        "best_model": best_path.name,
        "best_model_sha256": sha256(best_path),
        "elapsed_seconds": time.time() - started,
    }
    report_path = output / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "complete": True,
                "best_selection_key": report["best_selection_key"],
                "best_model": str(best_path),
                "report": str(report_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
