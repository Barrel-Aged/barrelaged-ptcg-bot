"""Train a conservative recurrent residual on winning Rocket replay histories.

The frozen ScaleV2 policy is a useful state-only blueprint, but its public
leader imitation accuracy leaves a large sequencing gap.  This trainer keeps
that blueprint frozen and learns only a small GRU whose memory is driven by
the public global token.  The GRU emits an option-interaction residual for
single-choice decisions.  A validation-calibrated margin gate preserves the
blueprint unless the recurrent branch is sufficiently decisive.

Kaggle episode actions are paired with the same seat's *previous* observation.
The parser below preserves that required shift and advances recurrent memory
on every non-empty decision, including trivial and multi-select states that do
not contribute a policy label.
"""

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
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "candidates"
    / "leader_bc_scale_v2"
)
DEFAULT_REPLAYS = PROJECT_ROOT / "work" / "kaggle_leader_54834745"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "history_residual_rocket"
)


@dataclass
class RawDecision:
    feature: dict[str, np.ndarray]
    label: int
    forbidden: tuple[int, ...]
    game: str


@dataclass
class EncodedDecision:
    global_token: torch.Tensor
    option_token: torch.Tensor
    base_logits: torch.Tensor
    label: int
    forbidden: tuple[int, ...]


class HistoryResidual(nn.Module):
    """Public-history memory and a low-rank option interaction."""

    def __init__(self, global_width: int, option_width: int, memory: int) -> None:
        super().__init__()
        self.memory = int(memory)
        self.gru = nn.GRUCell(global_width, memory)
        self.query = nn.Linear(memory, option_width)
        self.outcome = nn.Sequential(
            nn.Linear(memory, memory), nn.ReLU(), nn.Linear(memory, 1)
        )


class ResidualTrainer(nn.Module):
    def __init__(
        self,
        global_width: int,
        option_width: int,
        memory: int,
        projection: torch.Tensor,
    ) -> None:
        super().__init__()
        self.residual = HistoryResidual(global_width, option_width, memory)
        self.projection = nn.Parameter(projection.clone())
        nn.init.normal_(self.residual.query.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.residual.query.bias)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--memory", type=int, default=128)
    parser.add_argument("--option-width", type=int, default=64)
    parser.add_argument("--residual-penalty", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument(
        "--opponent-family", choices=("all", "marnie"), default="all"
    )
    parser.add_argument(
        "--seat-role", choices=("all", "first", "second"), default="all"
    )
    return parser.parse_args()


def shifted_winner_observations(path: Path) -> list[tuple[dict[str, Any], list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rewards = payload.get("rewards") or []
    if len(rewards) != 2 or rewards[0] == rewards[1]:
        return []
    winner = int(rewards[1] > rewards[0])
    previous: list[dict[str, Any] | None] = [None, None]
    result: list[tuple[dict[str, Any], list[int]]] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for actor, agent in enumerate(step[:2]):
            if not isinstance(agent, dict):
                continue
            decision_observation = previous[actor]
            action = agent.get("action")
            if (
                actor == winner
                and isinstance(decision_observation, dict)
                and decision_observation.get("select") is not None
                and isinstance(action, list)
            ):
                result.append((decision_observation, action))
            observation = agent.get("observation")
            if isinstance(observation, dict):
                previous[actor] = observation
    return result


def raw_game_sequence(
    path: Path,
    *,
    deck: list[int],
    deck_library: Any,
    belief_class: Any,
    features_module: Any,
) -> tuple[list[RawDecision], dict[str, int]]:
    belief = belief_class(deck, opponent_deck_priors=deck_library)
    sequence: list[RawDecision] = []
    counts = {
        "observations": 0,
        "single_labels": 0,
        "multi_or_optional": 0,
        "feature_errors": 0,
    }
    for observation, action in shifted_winner_observations(path):
        selection = observation.get("select") or {}
        options = selection.get("option") or []
        high = min(int(selection.get("maxCount") or 0), len(options))
        if high <= 0:
            continue
        try:
            feature = features_module.extract(
                observation, belief=belief, slot_link=True
            )
        except Exception:
            counts["feature_errors"] += 1
            continue

        counts["observations"] += 1
        attack_exists = any(option.get("type") == 13 for option in options)
        forbidden = tuple(
            index
            for index, option in enumerate(options)
            if attack_exists and option.get("type") == 14
        )
        label = -1
        if (
            high == 1
            and len(options) > 1
            and len(action) == 1
            and isinstance(action[0], int)
            and 0 <= action[0] < len(options)
            and int(action[0]) not in forbidden
        ):
            label = int(action[0])
            counts["single_labels"] += 1
        else:
            counts["multi_or_optional"] += 1
        sequence.append(
            RawDecision(
                feature=feature,
                label=label,
                forbidden=forbidden,
                game=path.name,
            )
        )
    return sequence, counts


def token_batch(model: nn.Module, batch: dict[str, torch.Tensor]):
    return model._tokens(
        batch["global"],
        batch["belief_global"],
        batch["hand_card"],
        batch["my_discard_card"],
        batch["opp_discard_card"],
        batch["stadium_card"],
        batch["my_unseen_card"],
        batch["opp_unseen_card"],
        batch["poke_feats"],
        batch["poke_card"],
        batch["opt_feats"],
        batch["opt_unseen"],
        batch["opt_card"],
        batch["opt_attack"],
        batch["opt_target"],
        batch["poke_mask"],
        batch["opt_mask"],
        batch.get("hand_mask"),
        batch.get("my_discard_mask"),
        batch.get("opp_discard_mask"),
        batch.get("my_unseen_mask"),
        batch.get("opp_unseen_mask"),
        kpc=batch.get("known_prized_card") if model.v4 else None,
        kpm=batch.get("known_prized_mask") if model.v4 else None,
        efc=batch.get("effect_card") if model.v4 else None,
        bpz=batch.get("belief_prized") if model.v4 else None,
        ctc=batch.get("context_card") if model.v5 else None,
        pwk=batch.get("poke_weak") if model.v5 else None,
    )


@torch.no_grad()
def encode_sequences(
    model: nn.Module,
    raw_sequences: dict[str, list[RawDecision]],
    *,
    collate: Any,
    example_class: Any,
    batch_size: int,
) -> dict[str, list[EncodedDecision]]:
    flat: list[RawDecision] = []
    positions: dict[str, tuple[int, int]] = {}
    for game, sequence in raw_sequences.items():
        positions[game] = (len(flat), len(sequence))
        flat.extend(sequence)

    encoded_flat: list[EncodedDecision] = []
    model.eval()
    for start in range(0, len(flat), batch_size):
        selected = flat[start : start + batch_size]
        examples = [
            example_class(item.feature, max(item.label, 0), item.game)
            for item in selected
        ]
        batch = collate(examples, torch.device("cpu"))
        global_token, option_token = token_batch(model, batch)
        base_logits = model.logit_head(option_token).squeeze(-1)
        for row, item in enumerate(selected):
            option_count = int(item.feature["opt_card"].shape[0])
            logits = base_logits[row, :option_count].detach().cpu().clone()
            if item.forbidden:
                logits[list(item.forbidden)] = -1.0e9
            encoded_flat.append(
                EncodedDecision(
                    global_token=global_token[row].detach().cpu().clone(),
                    option_token=option_token[row, :option_count].detach().cpu().clone(),
                    base_logits=logits,
                    label=item.label,
                    forbidden=item.forbidden,
                )
            )
        if start and start % (batch_size * 25) == 0:
            print(
                json.dumps({"encoded": start, "total": len(flat)}),
                flush=True,
            )

    return {
        game: encoded_flat[start : start + count]
        for game, (start, count) in positions.items()
    }


def sequence_loss(
    trainer: ResidualTrainer,
    sequence: list[EncodedDecision],
    residual_penalty: float,
) -> tuple[torch.Tensor | None, int]:
    memory = torch.zeros(1, trainer.residual.memory)
    losses: list[torch.Tensor] = []
    for decision in sequence:
        option = decision.option_token @ trainer.projection
        residual_logits = (
            option @ trainer.residual.query(memory).squeeze(0)
            / math.sqrt(option.shape[-1])
        )
        logits = decision.base_logits + residual_logits
        if decision.forbidden:
            logits = logits.clone()
            logits[list(decision.forbidden)] = -1.0e9
        if decision.label >= 0:
            target = torch.tensor([decision.label], dtype=torch.long)
            policy_loss = torch_functional.cross_entropy(logits.unsqueeze(0), target)
            losses.append(
                policy_loss + residual_penalty * residual_logits.square().mean()
            )
        memory = trainer.residual.gru(
            decision.global_token.unsqueeze(0), memory
        )
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), len(losses)


@torch.no_grad()
def collect_predictions(
    trainer: ResidualTrainer,
    sequences: dict[str, list[EncodedDecision]],
) -> list[dict[str, float | int | str]]:
    trainer.eval()
    rows: list[dict[str, float | int | str]] = []
    for game, sequence in sequences.items():
        memory = torch.zeros(1, trainer.residual.memory)
        for decision in sequence:
            option = decision.option_token @ trainer.projection
            residual_logits = (
                option @ trainer.residual.query(memory).squeeze(0)
                / math.sqrt(option.shape[-1])
            )
            combined = decision.base_logits + residual_logits
            if decision.forbidden:
                combined = combined.clone()
                combined[list(decision.forbidden)] = -1.0e9
            if decision.label >= 0:
                base_index = int(decision.base_logits.argmax())
                residual_index = int(combined.argmax())
                advantage = (
                    float(combined[residual_index] - combined[base_index])
                    if residual_index != base_index
                    else 0.0
                )
                rows.append(
                    {
                        "game": game,
                        "label": decision.label,
                        "base": base_index,
                        "residual": residual_index,
                        "advantage": advantage,
                        "nll": float(
                            torch_functional.cross_entropy(
                                combined.unsqueeze(0),
                                torch.tensor([decision.label]),
                            )
                        ),
                    }
                )
            memory = trainer.residual.gru(
                decision.global_token.unsqueeze(0), memory
            )
    return rows


def metrics(rows: list[dict[str, float | int | str]], threshold: float) -> dict[str, Any]:
    total = len(rows)
    base_correct = 0
    raw_correct = 0
    gated_correct = 0
    changed = 0
    gated = 0
    changed_correct = 0
    for row in rows:
        label = int(row["label"])
        base = int(row["base"])
        residual = int(row["residual"])
        use = residual != base and float(row["advantage"]) >= threshold
        chosen = residual if use else base
        base_correct += int(base == label)
        raw_correct += int(residual == label)
        gated_correct += int(chosen == label)
        changed += int(residual != base)
        gated += int(use)
        changed_correct += int(use and residual == label)
    return {
        "examples": total,
        "threshold": threshold,
        "base_top1": base_correct / max(total, 1),
        "raw_residual_top1": raw_correct / max(total, 1),
        "gated_top1": gated_correct / max(total, 1),
        "raw_change_rate": changed / max(total, 1),
        "gated_change_rate": gated / max(total, 1),
        "gated_change_precision": changed_correct / max(gated, 1),
        "mean_residual_nll": sum(float(row["nll"]) for row in rows) / max(total, 1),
    }


def select_threshold(rows: list[dict[str, float | int | str]]) -> tuple[float, list[dict[str, Any]]]:
    advantages = sorted(
        {
            max(0.0, float(row["advantage"]))
            for row in rows
            if int(row["residual"]) != int(row["base"])
        }
    )
    candidates = [0.0]
    if advantages:
        for quantile in np.linspace(0.0, 1.0, 41):
            candidates.append(float(np.quantile(advantages, quantile)))
        candidates.append(max(advantages) + 1.0e-6)
    candidates = sorted(set(candidates))
    table = [metrics(rows, threshold) for threshold in candidates]
    best = max(
        table,
        key=lambda row: (
            float(row["gated_top1"]),
            -float(row["gated_change_rate"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), table


def main() -> dict[str, Any]:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))

    candidate = args.candidate.resolve()
    replay_root = args.replays.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(candidate))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy  # type: ignore
    import train_public_leader_v110_bc as behavior_cloning

    deck = [
        int(line)
        for line in (candidate / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:60]
    if len(deck) != 60:
        raise ValueError(f"expected 60 cards, got {len(deck)}")
    deck_library_path = candidate / "deck_library.json"
    deck_library = (
        json.loads(deck_library_path.read_text(encoding="utf-8"))
        if deck_library_path.exists()
        else None
    )
    provenance_path = candidate / "SCALE_TRAINING_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if args.opponent_family == "all" and args.seat_role == "all":
        split_names = provenance["split_games"]
    else:
        eligible = [
            path
            for path in sorted(replay_root.glob("episode_*.json"))
            if behavior_cloning.matches_family(path, args.opponent_family)
            and behavior_cloning.matches_seat_role(path, args.seat_role)
        ]
        if len(eligible) < 20:
            raise ValueError(
                f"targeted residual needs at least 20 games, found {len(eligible)}"
            )
        targeted_split = behavior_cloning.deterministic_split(eligible, args.seed)
        split_names = {
            split: [path.name for path in paths]
            for split, paths in targeted_split.items()
        }

    raw_sequences: dict[str, list[RawDecision]] = {}
    extraction_totals = {
        "observations": 0,
        "single_labels": 0,
        "multi_or_optional": 0,
        "feature_errors": 0,
    }
    all_names = [
        name
        for split in ("train", "validation", "test")
        for name in split_names[split]
    ]
    started = time.monotonic()
    for index, name in enumerate(all_names, 1):
        sequence, counts = raw_game_sequence(
            replay_root / name,
            deck=deck,
            deck_library=deck_library,
            belief_class=Belief,
            features_module=features,
        )
        raw_sequences[name] = sequence
        for key, value in counts.items():
            extraction_totals[key] += value
        if index % 50 == 0:
            print(
                json.dumps(
                    {
                        "feature_games": index,
                        "total_games": len(all_names),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    blueprint = load_policy(candidate / "model.pt", map_location="cpu").eval()
    encoded = encode_sequences(
        blueprint,
        raw_sequences,
        collate=behavior_cloning.collate,
        example_class=behavior_cloning.Example,
        batch_size=args.batch_size,
    )
    splits = {
        split: {name: encoded[name] for name in split_names[split]}
        for split in ("train", "validation", "test")
    }

    global_width = int(next(iter(next(iter(encoded.values())))).global_token.shape[0])
    generator = torch.Generator().manual_seed(args.seed)
    random_projection = torch.randn(
        global_width, args.option_width, generator=generator
    )
    projection, _ = torch.linalg.qr(random_projection, mode="reduced")
    trainer = ResidualTrainer(
        global_width,
        args.option_width,
        args.memory,
        projection,
    )
    optimizer = torch.optim.AdamW(
        trainer.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_nll = math.inf
    stale = 0
    train_names = list(splits["train"])
    for epoch in range(1, args.epochs + 1):
        trainer.train()
        random.Random(args.seed + epoch).shuffle(train_names)
        epoch_loss = 0.0
        epoch_labels = 0
        for name in train_names:
            loss, labels = sequence_loss(
                trainer,
                splits["train"][name],
                args.residual_penalty,
            )
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * labels
            epoch_labels += labels

        validation_rows = collect_predictions(trainer, splits["validation"])
        validation_nll = sum(float(row["nll"]) for row in validation_rows) / max(
            len(validation_rows), 1
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_labels, 1),
            "validation_nll": validation_nll,
            "validation_raw_top1": metrics(validation_rows, 0.0)[
                "raw_residual_top1"
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_nll < best_validation_nll - 1.0e-5:
            best_validation_nll = validation_nll
            best_state = copy.deepcopy(trainer.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    trainer.load_state_dict(best_state)
    prediction_rows = {
        split: collect_predictions(trainer, sequences)
        for split, sequences in splits.items()
    }
    threshold, threshold_table = select_threshold(prediction_rows["validation"])
    final_metrics = {
        split: metrics(rows, threshold)
        for split, rows in prediction_rows.items()
    }

    payload = {
        "schema": "rocket-history-residual-policy/v1",
        "state_dict": trainer.residual.state_dict(),
        "global_width": global_width,
        "option_width": args.option_width,
        "memory": args.memory,
        "projection": trainer.projection.detach().cpu(),
        "gate_threshold": threshold,
        "base": str(candidate / "model.pt"),
        "base_sha256": hashlib.sha256((candidate / "model.pt").read_bytes()).hexdigest(),
        "training_seed": args.seed,
    }
    model_path = output / "history_residual.pt"
    torch.save(payload, model_path)
    report = {
        "schema_version": 1,
        "candidate": str(candidate),
        "replays": str(replay_root),
        "opponent_family": args.opponent_family,
        "seat_role": args.seat_role,
        "game_counts": {split: len(names) for split, names in split_names.items()},
        "extraction": extraction_totals,
        "training": {
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "memory": args.memory,
            "option_width": args.option_width,
            "residual_penalty": args.residual_penalty,
            "history": history,
        },
        "gate": {
            "threshold": threshold,
            "selection_split": "validation",
            "threshold_table": threshold_table,
        },
        "metrics": final_metrics,
        "artifact": str(model_path),
        "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
