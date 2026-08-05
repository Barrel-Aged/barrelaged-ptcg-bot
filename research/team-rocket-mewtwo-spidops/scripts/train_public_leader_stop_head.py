"""Train a conservative optional-selection stop head on public leader wins.

The V110 runtime historically selected exactly ``maxCount`` options.  This
script keeps the Scale V2 action policy frozen and learns only a linear stop
head over its relational global token.  Labels use the correctly shifted
Kaggle action alignment.  Attack-bearing selections are excluded because the
runtime's existing attack-safety contract deliberately disables early stop.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_public_leader_v110_bc import collate  # noqa: E402


@dataclass
class StopExample:
    feat: dict[str, np.ndarray]
    stop: int
    game: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "leader_bc_stop_v1"
        ),
    )
    parser.add_argument(
        "--replays", type=Path, default=PROJECT_ROOT / "work" / "kaggle_leader_54834745"
    )
    parser.add_argument(
        "--frozen-split-summary",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "leader_bc"
            / "last"
            / "training_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "leader_stop_v1"
        ),
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--minimum-precision", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--threads", type=int, default=6)
    return parser.parse_args()


def aligned_winner_rows(path: Path) -> list[tuple[dict[str, Any], list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rewards = payload.get("rewards") or []
    if len(rewards) != 2 or rewards[0] == rewards[1]:
        return []
    winner = int(rewards[1] > rewards[0])
    previous: list[dict[str, Any] | None] = [None, None]
    rows: list[tuple[dict[str, Any], list[int]]] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for actor, agent in enumerate(step[:2]):
            if not isinstance(agent, dict):
                continue
            observation = agent.get("observation")
            action = agent.get("action")
            decision = previous[actor]
            if (
                actor == winner
                and isinstance(decision, dict)
                and decision.get("select") is not None
                and isinstance(action, list)
                and (
                    bool(action)
                    or int(((decision.get("select") or {}).get("minCount")) or 0)
                    == 0
                )
            ):
                selection = decision["select"]
                options = selection.get("option") or []
                lower = int(selection.get("minCount") or 0)
                upper = min(int(selection.get("maxCount") or 0), len(options))
                legal = (
                    len(action) == len(set(action))
                    and lower <= len(action) <= upper
                    and all(
                        isinstance(index, int) and 0 <= index < len(options)
                        for index in action
                    )
                )
                if legal:
                    rows.append((decision, [int(index) for index in action]))
            if isinstance(observation, dict):
                previous[actor] = observation
    return rows


def make_splits(replays: Path, summary: Path) -> dict[str, list[Path]]:
    paths = sorted(replays.resolve().glob("episode_*.json"))
    by_name = {path.name: path for path in paths}
    frozen = json.loads(summary.resolve().read_text(encoding="utf-8"))["split_games"]
    validation_names = list(frozen["validation"])
    test_names = list(frozen["test"])
    missing = sorted((set(validation_names) | set(test_names)) - set(by_name))
    if missing:
        raise ValueError(f"Frozen stop-head holdouts are unavailable: {missing}")
    held_out = set(validation_names) | set(test_names)
    return {
        "train": [path for path in paths if path.name not in held_out],
        "validation": [by_name[name] for name in validation_names],
        "test": [by_name[name] for name in test_names],
    }


def build_examples(
    paths: list[Path],
    deck: list[int],
    deck_library: Any,
    belief_class: Any,
    features: Any,
) -> tuple[list[StopExample], Counter[str]]:
    examples: list[StopExample] = []
    counts: Counter[str] = Counter()
    for path in paths:
        belief = belief_class(deck, opponent_deck_priors=deck_library)
        for observation, action in aligned_winner_rows(path):
            selection = observation["select"]
            options = selection.get("option") or []
            lower = int(selection.get("minCount") or 0)
            upper = min(int(selection.get("maxCount") or 0), len(options))
            try:
                # Extract on every actual decision so belief/prize state evolves
                # exactly as it does in the shipped sequential runtime.
                base = features.extract(observation, belief=belief, slot_link=True)
            except Exception as error:
                counts[f"feature_error:{type(error).__name__}"] += 1
                continue
            if upper <= lower:
                counts["mandatory_only"] += 1
                continue
            if any(option.get("type") == 13 for option in options):
                counts["attack_guard_excluded"] += 1
                continue
            selected: list[int] = []
            for position in range(upper):
                if position >= lower:
                    conditioned = features.mark_selection(base, selected, upper)
                    should_stop = int(position >= len(action))
                    examples.append(StopExample(conditioned, should_stop, path.name))
                    counts["positive" if should_stop else "negative"] += 1
                    if should_stop:
                        break
                if position < len(action):
                    selected.append(action[position])
                else:
                    break
    counts["examples"] = len(examples)
    return examples, counts


@torch.no_grad()
def embed_examples(
    model: torch.nn.Module,
    examples: list[StopExample],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks: list[torch.Tensor] = []
    labels: list[int] = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        batch = collate(selected, device)
        model.forward_batch(batch)
        chunks.append(model.last_global_token_batch.detach().cpu())
        labels.extend(example.stop for example in selected)
    return torch.cat(chunks), torch.as_tensor(labels, dtype=torch.float32)


def binary_metrics(probabilities: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    truth = labels.bool()
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    true_negative = int((~predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "examples": int(labels.numel()),
        "threshold": float(threshold),
        "accuracy": (true_positive + true_negative) / max(int(labels.numel()), 1),
        "precision": precision,
        "recall": recall,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def select_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    minimum_precision: float,
) -> tuple[float, list[dict[str, Any]]]:
    table = [
        binary_metrics(probabilities, labels, step / 100.0)
        for step in range(5, 100)
    ]
    eligible = [row for row in table if row["precision"] >= minimum_precision]
    if not eligible:
        eligible = table
    # Prefer maximum recall subject to the safety precision floor. Ties choose
    # higher accuracy and then the higher (more conservative) threshold.
    best = max(
        eligible,
        key=lambda row: (row["recall"], row["accuracy"], row["threshold"]),
    )
    return float(best["threshold"]), table


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    candidate = args.candidate.resolve()
    sys.path.insert(0, str(candidate))

    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import build_net, save_policy  # type: ignore

    deck = [
        int(line)
        for line in (candidate / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:60]
    if len(deck) != 60:
        raise ValueError(f"Expected 60 cards, got {len(deck)}")
    deck_library_path = candidate / "deck_library.json"
    deck_library = json.loads(deck_library_path.read_text(encoding="utf-8"))

    checkpoint = torch.load(candidate / "model.pt", map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError("Stop training requires a configured relational checkpoint")
    config = dict(checkpoint["config"])
    config.update(stop=True, stop_threshold=0.5)
    model = build_net(config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if set(missing) != {"stop_head.weight", "stop_head.bias"} or unexpected:
        raise ValueError(f"Unexpected warm-load delta: missing={missing}, unexpected={unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert model.stop_head is not None
    for parameter in model.stop_head.parameters():
        parameter.requires_grad = True

    splits = make_splits(args.replays, args.frozen_split_summary)
    extraction: dict[str, dict[str, int]] = {}
    embedded: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    started = time.monotonic()
    for name, paths in splits.items():
        examples, counts = build_examples(paths, deck, deck_library, Belief, features)
        if not examples:
            raise ValueError(f"No stop examples in {name}")
        extraction[name] = dict(counts)
        embedded[name] = embed_examples(
            model, examples, args.feature_batch_size, device
        )
        print(
            json.dumps(
                {
                    "stage": "embed",
                    "split": name,
                    "examples": len(examples),
                    "positive": counts["positive"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        model.stop_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_x, train_y = embedded["train"]
    validation_x, validation_y = embedded["validation"]
    order = list(range(train_y.numel()))
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(order)
        model.stop_head.train()
        total_loss = 0.0
        for start in range(0, len(order), args.train_batch_size):
            indexes = torch.as_tensor(
                order[start : start + args.train_batch_size], dtype=torch.long
            )
            logits = model.stop_head(train_x[indexes]).squeeze(-1)
            loss = torch_functional.binary_cross_entropy_with_logits(
                logits, train_y[indexes]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * int(indexes.numel())
        model.stop_head.eval()
        with torch.no_grad():
            validation_logits = model.stop_head(validation_x).squeeze(-1)
            validation_loss = float(
                torch_functional.binary_cross_entropy_with_logits(
                    validation_logits, validation_y
                )
            )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(order), 1),
            "validation_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.stop_head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise AssertionError("Stop head never produced a checkpoint")
    model.stop_head.load_state_dict(best_state)
    model.stop_head.eval()

    with torch.no_grad():
        validation_probabilities = torch.sigmoid(
            model.stop_head(validation_x).squeeze(-1)
        )
    threshold, threshold_table = select_threshold(
        validation_probabilities, validation_y, args.minimum_precision
    )
    model.stop_threshold = threshold
    model.cfg["stop_threshold"] = threshold
    metrics: dict[str, dict[str, Any]] = {}
    for name, (inputs, labels) in embedded.items():
        with torch.no_grad():
            probabilities = torch.sigmoid(model.stop_head(inputs).squeeze(-1))
        metrics[name] = binary_metrics(probabilities, labels, threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.pt"
    save_policy(model, model_path)
    summary = {
        "schema_version": 1,
        "source_checkpoint": str((candidate / "model.pt").resolve()),
        "replays": str(args.replays.resolve()),
        "frozen_split_summary": str(args.frozen_split_summary.resolve()),
        "attack_guard_excluded": True,
        "runtime_contract": (
            "stop is considered only after minCount selections and never when "
            "the attack-safety guard sees an attack option"
        ),
        "split_games": {
            name: [path.name for path in paths] for name, paths in splits.items()
        },
        "extraction": extraction,
        "training": {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "history": history,
            "elapsed_seconds": time.monotonic() - started,
        },
        "calibration": {
            "minimum_validation_precision": args.minimum_precision,
            "selected_threshold": threshold,
            "validation_threshold_table": threshold_table,
        },
        "always_continue_accuracy": {
            name: float((labels == 0).float().mean())
            for name, (_, labels) in embedded.items()
        },
        "metrics": metrics,
        "model": str(model_path.resolve()),
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
