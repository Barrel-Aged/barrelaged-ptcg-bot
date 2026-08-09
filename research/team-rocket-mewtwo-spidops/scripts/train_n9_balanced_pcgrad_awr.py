#!/usr/bin/env python3
"""Train the preregistered N9 shared action head with balanced PCGrad AWR."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Any, Iterable

import numpy as np
import torch

try:
    from train_live_outcome_awr import (
        WeightedExample,
        collate_like_live_agent,
        deployment_logits,
        extract_examples,
        metrics,
        read_game,
        reference_parity,
    )
    from train_public_leader_v110_bc import configure_trainable
except ModuleNotFoundError:  # pragma: no cover - module-style test import
    from scripts.train_live_outcome_awr import (
        WeightedExample,
        collate_like_live_agent,
        deployment_logits,
        extract_examples,
        metrics,
        read_game,
        reference_parity,
    )
    from scripts.train_public_leader_v110_bc import configure_trainable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2"
)
DEFAULT_REPLAYS = PROJECT_ROOT / "work" / "n9_balanced_exact_replays"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n9_balanced_pcgrad_awr" / "training_v1"
)
BOT = "N9RocketControl"
GROUPS = (
    "AshMarnieGeneralV110_extracted",
    "HelbrosMarnieGeneralRecoveryV4_extracted",
)
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
SEED = 797001
TRAINABLE_KEYS = ("logit_head.bias", "logit_head.weight")


@dataclass(frozen=True)
class N9Game:
    path: Path
    opponent: str
    won: bool
    draw: bool

    @property
    def group(self) -> str:
        return self.opponent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--group-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--kl-coef", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    locked = {
        "epochs": 4,
        "group_batch_size": 64,
        "learning_rate": 3.0e-6,
        "weight_decay": 1.0e-4,
        "temperature": 0.5,
        "kl_coef": 0.25,
        "threads": 1,
        "seed": SEED,
    }
    for key, expected in locked.items():
        if getattr(args, key) != expected:
            raise ValueError(f"N9 locked argument changed: {key}")


def validate_parity(path: Path, candidate: Path, replays: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "rocket-n9-exact-agent-parity/v1":
        raise ValueError("unexpected N9 parity schema")
    expected = {
        "candidate": str(candidate),
        "replays": str(replays),
        "reference_model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "recorded_bot": BOT,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"N9 parity provenance mismatch: {key}")
    totals = report.get("totals") or {}
    if int(report.get("replay_files", -1)) != 1200:
        raise ValueError("N9 parity replay count changed")
    if int(totals.get("games_with_mismatch", -1)) != 0 or int(totals.get("mismatched", -1)) != 0:
        raise ValueError("N9 contains off-policy replay actions")
    return {"path": str(path), "sha256": digest(path), **report}


def load_games(replays: Path) -> tuple[list[N9Game], dict[str, int]]:
    games = []
    counts = {group: 0 for group in GROUPS}
    for path in sorted(replays.rglob("*.jsonl")):
        info, _ = read_game(path, BOT)
        if info.opponent not in GROUPS or info.seat != 1 or info.started_first:
            raise ValueError(f"N9 replay identity/seat mismatch: {path}")
        counts[info.opponent] += 1
        games.append(N9Game(path, info.opponent, info.won, info.draw))
    if counts != {group: 600 for group in GROUPS}:
        raise ValueError(f"N9 opponent counts changed: {counts}")
    return games, counts


def split_games(games: list[N9Game]) -> dict[str, list[N9Game]]:
    result: dict[str, list[N9Game]] = {"train": [], "validation": [], "test": []}
    for group in GROUPS:
        decisive = sorted(
            (game for game in games if game.opponent == group and not game.draw),
            key=lambda game: game.path.name,
        )
        for index, game in enumerate(decisive):
            bucket = index % 10
            split = "train" if bucket <= 7 else "validation" if bucket == 8 else "test"
            result[split].append(game)
    if any(not values for values in result.values()):
        raise ValueError("N9 split is empty")
    return result


def group_rates(games: list[N9Game]) -> dict[str, float]:
    global_rate = sum(game.won for game in games) / len(games)
    rates = {"__global__": global_rate}
    for group in GROUPS:
        subset = [game for game in games if game.opponent == group]
        rates[group] = (sum(game.won for game in subset) + 5.0 * global_rate) / (len(subset) + 5.0)
    return rates


def apply_balanced_weights(
    examples: list[WeightedExample], rates: dict[str, float], temperature: float
) -> dict[str, float]:
    for example in examples:
        baseline = rates[example.group]
        example.advantage = float(example.won) - baseline
        example.weight *= math.exp(example.advantage / temperature)
    totals = {
        group: sum(example.weight for example in examples if example.group == group)
        for group in GROUPS
    }
    if any(value <= 0 for value in totals.values()):
        raise ValueError("N9 group has no positive training mass")
    target = len(examples) / len(GROUPS)
    for example in examples:
        example.weight *= target / totals[example.group]
    return {
        group: sum(example.weight for example in examples if example.group == group)
        for group in GROUPS
    }


def repeated_order(length: int, target: int, seed: int) -> list[int]:
    if length <= 0 or target <= 0:
        raise ValueError("invalid N9 group order size")
    output = []
    cycle = 0
    while len(output) < target:
        values = list(range(length))
        random.Random(f"{seed}:{cycle}").shuffle(values)
        output.extend(values)
        cycle += 1
    return output[:target]


def dot(left: Iterable[torch.Tensor], right: Iterable[torch.Tensor]) -> torch.Tensor:
    return sum((a * b).sum() for a, b in zip(left, right))


def symmetric_pcgrad(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...]
) -> tuple[tuple[torch.Tensor, ...], bool, float]:
    cross = dot(first, second)
    norm_first = dot(first, first)
    norm_second = dot(second, second)
    cosine = float(cross / torch.sqrt(norm_first * norm_second).clamp_min(1.0e-20))
    if float(cross) >= 0.0:
        return tuple((a + b) * 0.5 for a, b in zip(first, second)), False, cosine
    projected_first = tuple(
        a - cross / norm_second.clamp_min(1.0e-20) * b for a, b in zip(first, second)
    )
    projected_second = tuple(
        b - cross / norm_first.clamp_min(1.0e-20) * a for a, b in zip(first, second)
    )
    return (
        tuple((a + b) * 0.5 for a, b in zip(projected_first, projected_second)),
        True,
        cosine,
    )


def group_objective(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    selected: list[WeightedExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, availability = collate_like_live_agent(selected, device)
    labels = torch.tensor([example.action for example in selected], dtype=torch.long, device=device)
    weights = torch.tensor([example.weight for example in selected], device=device)
    logits, _ = model.forward_batch(batch)
    with torch.no_grad():
        reference_logits, _ = reference.forward_batch(batch)
    logits = deployment_logits(logits, availability, selected)
    reference_logits = deployment_logits(reference_logits, availability, selected)
    logp = torch.log_softmax(logits, dim=1)
    chosen = -logp.gather(1, labels[:, None]).squeeze(1)
    awr = (chosen * weights).sum() / weights.sum().clamp_min(1.0e-12)
    ref_probability = torch.softmax(reference_logits, dim=1)
    ref_logp = torch.log_softmax(reference_logits, dim=1)
    kl = (ref_probability * (ref_logp - logp)).sum(1).mean()
    return awr, kl


def changed_keys(reference: torch.nn.Module, model: torch.nn.Module) -> list[str]:
    before = reference.state_dict()
    after = model.state_dict()
    return sorted(key for key in before if not torch.equal(before[key], after[key]))


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    candidate = CANDIDATE.resolve()
    replays = args.replays.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N9 training output: {output}")
    if digest(candidate / "model.pt") != MODEL_SHA256 or digest(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("N9 frozen substrate changed")
    parity = validate_parity(args.parity_report.resolve(), candidate, replays)
    sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    deck = [int(value) for value in (candidate / "deck.csv").read_text().split()]
    priors = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    games, source_counts = load_games(replays)
    splits = split_games(games)
    rates = group_rates(splits["train"])
    reference = load_policy(candidate / "model.pt", map_location=device).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    data: dict[str, list[WeightedExample]] = {}
    extraction = {}
    parity_by_split = {}
    group_mass = {}
    for split, split_games_list in splits.items():
        data[split], counts = extract_examples(
            split_games_list, BOT, deck, priors, Belief, features
        )
        group_mass[split] = apply_balanced_weights(data[split], rates, args.temperature)
        extraction[split] = dict(counts)
        parity_by_split[split] = reference_parity(
            reference, data[split], args.group_batch_size * 2, device
        )
        if int(parity_by_split[split]["mismatched"]):
            raise ValueError(f"N9 feature/batch parity failed: {split}")
    by_group = {
        group: [example for example in data["train"] if example.group == group]
        for group in GROUPS
    }
    if any(not examples for examples in by_group.values()):
        raise ValueError("N9 training group has no examples")
    model = copy.deepcopy(reference)
    trainable = configure_trainable(model, "head")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if tuple(sorted(trainable_names)) != TRAINABLE_KEYS:
        raise ValueError(f"N9 trainable scope changed: {trainable_names}")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output.mkdir(parents=True, exist_ok=False)
    baseline_metrics = {
        split: {
            group: metrics(
                reference,
                reference,
                [example for example in examples if example.group == group],
                args.group_batch_size * 2,
                device,
            )
            for group in GROUPS
        }
        for split, examples in data.items()
    }
    history = []
    started = monotonic()
    target_length = max(len(examples) for examples in by_group.values())
    for epoch in range(1, args.epochs + 1):
        orders = {
            group: repeated_order(
                len(by_group[group]), target_length, args.seed + epoch * 100 + index
            )
            for index, group in enumerate(GROUPS)
        }
        # The entire feature trunk is frozen, so keep deployment-time eval
        # semantics (notably dropout-off) while differentiating the head.
        model.eval()
        conflicts = batches = 0
        cosines = []
        awr_totals = {group: 0.0 for group in GROUPS}
        kl_total = 0.0
        for start in range(0, target_length, args.group_batch_size):
            selected = {
                group: [
                    by_group[group][index]
                    for index in orders[group][start : start + args.group_batch_size]
                ]
                for group in GROUPS
            }
            objectives = {
                group: group_objective(model, reference, selected[group], device)
                for group in GROUPS
            }
            gradients = {
                group: torch.autograd.grad(
                    objectives[group][0], trainable, retain_graph=True
                )
                for group in GROUPS
            }
            robust, conflict, cosine = symmetric_pcgrad(
                gradients[GROUPS[0]], gradients[GROUPS[1]]
            )
            kl = 0.5 * (objectives[GROUPS[0]][1] + objectives[GROUPS[1]][1])
            kl_gradient = torch.autograd.grad(kl, trainable)
            optimizer.zero_grad(set_to_none=True)
            for parameter, robust_gradient, anchor_gradient in zip(
                trainable, robust, kl_gradient
            ):
                parameter.grad = robust_gradient + args.kl_coef * anchor_gradient
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            batches += 1
            conflicts += int(conflict)
            cosines.append(cosine)
            for group in GROUPS:
                awr_totals[group] += float(objectives[group][0].detach())
            kl_total += float(kl.detach())
        changed = changed_keys(reference, model)
        if tuple(changed) != TRAINABLE_KEYS:
            raise ValueError(f"N9 changed unexpected tensors after epoch {epoch}: {changed}")
        epoch_path = output / f"epoch_{epoch:02d}.pt"
        save_policy(model, epoch_path)
        validation = {
            group: metrics(
                model,
                reference,
                [example for example in data["validation"] if example.group == group],
                args.group_batch_size * 2,
                device,
            )
            for group in GROUPS
        }
        test = {
            group: metrics(
                model,
                reference,
                [example for example in data["test"] if example.group == group],
                args.group_batch_size * 2,
                device,
            )
            for group in GROUPS
        }
        row = {
            "epoch": epoch,
            "model": str(epoch_path),
            "model_sha256": digest(epoch_path),
            "batches": batches,
            "pcgrad_conflicts": conflicts,
            "pcgrad_conflict_rate": conflicts / max(1, batches),
            "mean_gradient_cosine": float(np.mean(cosines)),
            "mean_awr": {group: awr_totals[group] / max(1, batches) for group in GROUPS},
            "mean_reference_kl": kl_total / max(1, batches),
            "validation": validation,
            "test": test,
            "changed_tensors": changed,
            "elapsed_seconds": monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    save_policy(model, output / "model.pt")
    report = {
        "schema": "rocket-n9-balanced-pcgrad-awr-training/v1",
        "protocol": "rocket-balanced-exact-pcgrad-awr/v1",
        "official_engine_source": True,
        "runtime_opponent_identity_input": False,
        "hidden_information_used": False,
        "candidate": str(candidate),
        "model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "replays": str(replays),
        "source_counts": source_counts,
        "parity_report": parity,
        "split": {
            name: {
                "games": len(values),
                "wins": sum(game.won for game in values),
                "by_opponent": {
                    group: sum(game.opponent == group for game in values)
                    for group in GROUPS
                },
                "files": [str(game.path) for game in values],
            }
            for name, values in splits.items()
        },
        "group_train_win_rates": rates,
        "group_example_mass": group_mass,
        "extraction": extraction,
        "reference_action_parity": parity_by_split,
        "trainable_names": trainable_names,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "hyperparameters": {
            "seed": args.seed,
            "epochs": args.epochs,
            "group_batch_size": args.group_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "kl_coef": args.kl_coef,
            "threads": args.threads,
        },
        "baseline_metrics": baseline_metrics,
        "epochs": history,
        "checkpoint_selection": "none from offline metrics; all four epochs are development arms",
        "final_epoch_alias": 4,
        "final_model_sha256": digest(output / "model.pt"),
        "trainer_sha256": digest(Path(__file__).resolve()),
        "elapsed_seconds": monotonic() - started,
        "development_only": True,
        "deployment_authority": False,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
