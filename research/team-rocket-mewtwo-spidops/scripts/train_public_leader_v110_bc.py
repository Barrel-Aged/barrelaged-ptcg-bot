"""Fine-tune the V110 relational policy on correctly aligned public leader wins.

Kaggle episode rows store each submitted action next to the observation that
resulted from that action.  Consequently, an action is paired with the same
seat's *previous* observation.  Treating row-local observation/action fields as
a pair silently trains on the wrong state; this script makes the one-step
alignment explicit and tests every target against the previous selection's
legal option count.

The shipped V110 autoregressive head always emits ``maxCount`` selections, so
optional empty selections and early stops are reported but excluded.  Every
fully supported multi-select action is unrolled into conditional next-choice
examples using ``features.mark_selection``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as torch_functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "v110_transfer"
)
DEFAULT_REPLAYS = PROJECT_ROOT / "work" / "kaggle_leader_54834745"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "leader_bc"


@dataclass(frozen=True)
class Decision:
    observation: dict[str, Any]
    action: tuple[int, ...]


@dataclass
class Example:
    feat: dict[str, np.ndarray]
    action: int
    game: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("head", "last", "full"),
        default="last",
        help="Train logit head only, last transformer block + head, or policy trunk.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument(
        "--focal-team",
        help=(
            "Train on the uniquely named team's decisions in every replay, including losses. "
            "When omitted, preserve the original winner-only behavior."
        ),
    )
    parser.add_argument(
        "--frozen-split-summary",
        type=Path,
        help=(
            "Reuse validation/test game names from an earlier training_summary.json; "
            "all other matching replays become training games."
        ),
    )
    parser.add_argument(
        "--opponent-family",
        choices=("all", "marnie", "starmie"),
        default="all",
        help="Optionally fine-tune only on wins against a public-card family.",
    )
    parser.add_argument(
        "--seat-role",
        choices=("all", "first", "second"),
        default="all",
        help="Optionally fine-tune only on wins where Rocket started first/second.",
    )
    parser.add_argument(
        "--marnie-train-repeat",
        type=int,
        default=1,
        help=(
            "Repeat training examples from publicly identifiable Marnie-family "
            "games without changing validation/test metrics."
        ),
    )
    parser.add_argument(
        "--second-train-repeat",
        type=int,
        default=1,
        help=(
            "Repeat training examples from games Rocket won from the second seat "
            "without changing validation/test metrics."
        ),
    )
    return parser.parse_args()


def replay_decisions(
    path: Path, focal_team: str | None = None
) -> tuple[list[Decision], Counter[str]]:
    """Return winning-seat decisions with Kaggle's required one-step shift."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rewards = payload.get("rewards") or []
    if len(rewards) != 2:
        return [], Counter({"non_decisive_game": 1})
    if focal_team:
        teams = list((payload.get("info") or {}).get("TeamNames") or [])
        seats = [index for index, name in enumerate(teams) if name == focal_team]
        if len(seats) != 1:
            return [], Counter({"focal_team_not_unique": 1})
        actor_of_interest = seats[0]
    else:
        if rewards[0] == rewards[1]:
            return [], Counter({"non_decisive_game": 1})
        actor_of_interest = int(rewards[1] > rewards[0])
    previous_observations: list[dict[str, Any] | None] = [None, None]
    decisions: list[Decision] = []
    counts: Counter[str] = Counter()
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for actor, agent in enumerate(step[:2]):
            if not isinstance(agent, dict):
                continue
            observation = agent.get("observation")
            action = agent.get("action")
            decision_observation = previous_observations[actor]
            if (
                actor == actor_of_interest
                and isinstance(decision_observation, dict)
                and decision_observation.get("select") is not None
                and isinstance(action, list)
                and (
                    bool(action)
                    or int(
                        ((decision_observation.get("select") or {}).get("minCount"))
                        or 0
                    )
                    == 0
                )
            ):
                selection = decision_observation["select"]
                options = selection.get("option") or []
                option_count = len(options)
                lower = int(selection.get("minCount") or 0)
                upper = min(int(selection.get("maxCount") or 0), option_count)
                legal = (
                    len(action) == len(set(action))
                    and lower <= len(action) <= upper
                    and all(
                        isinstance(index, int) and 0 <= index < option_count
                        for index in action
                    )
                )
                if not legal:
                    counts["invalid"] += 1
                elif option_count <= 1:
                    counts["trivial"] += 1
                elif upper <= 0 or not action:
                    counts["unsupported_empty"] += 1
                elif len(action) != upper:
                    counts["unsupported_early_stop"] += 1
                else:
                    decisions.append(
                        Decision(decision_observation, tuple(int(index) for index in action))
                    )
                    counts["supported_decisions"] += 1
                    counts["supported_choice_targets"] += len(action)
            if isinstance(observation, dict):
                previous_observations[actor] = observation
    return decisions, counts


def deterministic_split(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    ordered = sorted(paths)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    train_end = max(1, int(count * 0.8))
    validation_end = max(train_end + 1, int(count * 0.9))
    validation_end = min(validation_end, count)
    return {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }


def public_opponent_card_ids(path: Path) -> set[int]:
    """Collect only opponent cards visible to the winning Rocket seat."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rewards = payload.get("rewards") or []
    if len(rewards) != 2 or rewards[0] == rewards[1]:
        return set()
    winner = int(rewards[1] > rewards[0])
    visible: set[int] = set()

    def walk(value: Any) -> Iterable[int]:
        if isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif isinstance(value, dict):
            if isinstance(value.get("id"), int):
                yield int(value["id"])
            for key in ("preEvolution", "tools", "energyCards"):
                yield from walk(value.get(key) or [])

    for step in payload.get("steps") or []:
        if not isinstance(step, list) or len(step) <= winner:
            continue
        agent = step[winner]
        if not isinstance(agent, dict):
            continue
        current = (agent.get("observation") or {}).get("current") or {}
        players = current.get("players") or []
        your_index = current.get("yourIndex")
        if your_index not in (0, 1) or len(players) != 2:
            continue
        opponent = players[1 - int(your_index)] or {}
        for zone in ("active", "bench", "discard"):
            visible.update(walk(opponent.get(zone) or []))
    return visible


def matches_family(path: Path, family: str) -> bool:
    if family == "all":
        return True
    visible = public_opponent_card_ids(path)
    if family == "marnie":
        return bool(
            visible & {646, 647, 648}
            or 1259 in visible
            or {112, 860} <= visible
        )
    if family == "starmie":
        # Staryu / Mega Starmie ex are definitive public identifiers.  The
        # extra IDs cover the Froslass support line used by the arena build,
        # but never classify a game from private information.
        return bool(visible & {131, 132, 133, 1030, 1031})
    raise ValueError(f"Unsupported opponent family: {family}")


def matches_seat_role(path: Path, seat_role: str) -> bool:
    if seat_role == "all":
        return True
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rewards = payload.get("rewards") or []
    if len(rewards) != 2 or rewards[0] == rewards[1]:
        return False
    winner = int(rewards[1] > rewards[0])
    for step in payload.get("steps") or []:
        if not isinstance(step, list) or len(step) <= winner:
            continue
        agent = step[winner]
        if not isinstance(agent, dict):
            continue
        current = (agent.get("observation") or {}).get("current") or {}
        first_player = current.get("firstPlayer")
        your_index = current.get("yourIndex")
        if first_player in (0, 1) and your_index in (0, 1):
            started_first = int(first_player) == int(your_index)
            return started_first if seat_role == "first" else not started_first
    return False


def load_examples(
    paths: Iterable[Path],
    deck: list[int],
    deck_library: Any,
    belief_class: Any,
    features_module: Any,
    focal_team: str | None = None,
) -> tuple[list[Example], Counter[str]]:
    examples: list[Example] = []
    total: Counter[str] = Counter()
    for path in paths:
        decisions, counts = replay_decisions(path, focal_team=focal_team)
        total.update(counts)
        belief = belief_class(deck, opponent_deck_priors=deck_library)
        for decision in decisions:
            try:
                base = features_module.extract(
                    decision.observation, belief=belief, slot_link=True
                )
                upper = min(
                    int(decision.observation["select"].get("maxCount") or 0),
                    len(decision.observation["select"].get("option") or []),
                )
                selected: list[int] = []
                for action in decision.action:
                    conditioned = features_module.mark_selection(base, selected, upper)
                    examples.append(Example(conditioned, action, path.name))
                    selected.append(action)
            except Exception as error:  # Keep corrupt public rows auditable.
                total[f"feature_error:{type(error).__name__}"] += 1
    total["examples"] = len(examples)
    return examples, total


def collate(examples: list[Example], device: torch.device) -> dict[str, torch.Tensor]:
    """Pad variable-sized V110 feature dictionaries for ``forward_batch``."""
    batch_count = len(examples)
    feats = [example.feat for example in examples]
    poke_count = max(feat["poke_feats"].shape[0] for feat in feats)
    option_count = max(feat["opt_feats"].shape[0] for feat in feats)
    hand_count = max(feat["hand_card"].shape[0] for feat in feats)
    my_discard_count = max(feat["my_discard_card"].shape[0] for feat in feats)
    opponent_discard_count = max(feat["opp_discard_card"].shape[0] for feat in feats)
    my_unseen_count = max(feat["my_unseen_card"].shape[0] for feat in feats)
    opponent_unseen_count = max(feat["opp_unseen_card"].shape[0] for feat in feats)
    known_prized_count = max(feat["known_prized_card"].shape[0] for feat in feats)

    global_features = np.zeros(
        (batch_count, feats[0]["global"].shape[0]), dtype=np.float32
    )
    poke_features = np.zeros(
        (batch_count, poke_count, feats[0]["poke_feats"].shape[1]), dtype=np.float32
    )
    poke_card = np.zeros((batch_count, poke_count), dtype=np.int64)
    poke_mask = np.zeros((batch_count, poke_count), dtype=np.float32)
    option_features = np.zeros(
        (batch_count, option_count, feats[0]["opt_feats"].shape[1]), dtype=np.float32
    )
    option_card = np.zeros((batch_count, option_count), dtype=np.int64)
    option_attack = np.zeros((batch_count, option_count), dtype=np.int64)
    option_target = np.zeros((batch_count, option_count), dtype=np.int64)
    option_mask = np.zeros((batch_count, option_count), dtype=np.float32)

    def integer_zone(size: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((batch_count, size), dtype=np.int64),
            np.zeros((batch_count, size), dtype=np.float32),
        )

    hand_card, hand_mask = integer_zone(hand_count)
    my_discard_card, my_discard_mask = integer_zone(my_discard_count)
    opponent_discard_card, opponent_discard_mask = integer_zone(opponent_discard_count)
    my_unseen_card, my_unseen_mask = integer_zone(my_unseen_count)
    opponent_unseen_card, opponent_unseen_mask = integer_zone(opponent_unseen_count)
    known_prized_card, known_prized_mask = integer_zone(known_prized_count)
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
            ("opp_discard_card", opponent_discard_card, opponent_discard_mask),
            ("my_unseen_card", my_unseen_card, my_unseen_mask),
            ("opp_unseen_card", opponent_unseen_card, opponent_unseen_mask),
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
        "opp_discard_card": tensor(opponent_discard_card, torch.int64),
        "opp_discard_mask": tensor(opponent_discard_mask, torch.float32),
        "stadium_card": tensor(stadium_card, torch.int64),
        "my_unseen_card": tensor(my_unseen_card, torch.int64),
        "my_unseen_mask": tensor(my_unseen_mask, torch.float32),
        "opp_unseen_card": tensor(opponent_unseen_card, torch.int64),
        "opp_unseen_mask": tensor(opponent_unseen_mask, torch.float32),
        "belief_global": tensor(belief_global, torch.float32),
        "opt_unseen": tensor(option_unseen, torch.float32),
        "known_prized_card": tensor(known_prized_card, torch.int64),
        "known_prized_mask": tensor(known_prized_mask, torch.float32),
        "effect_card": tensor(effect_card, torch.int64),
        "belief_prized": tensor(belief_prized, torch.float32),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    examples: list[Example],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    correct = 0
    total_loss = 0.0
    total = 0
    for start in range(0, len(examples), batch_size):
        selected = examples[start : start + batch_size]
        batch = collate(selected, device)
        labels = torch.as_tensor(
            [example.action for example in selected], dtype=torch.long, device=device
        )
        logits, _ = model.forward_batch(batch)
        loss = torch_functional.cross_entropy(logits, labels, reduction="sum")
        total_loss += float(loss)
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += len(selected)
    return {
        "examples": total,
        "loss": total_loss / max(total, 1),
        "top1": correct / max(total, 1),
    }


def configure_trainable(model: torch.nn.Module, mode: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if mode == "head":
        prefixes = ("logit_head.",)
    elif mode == "last":
        last_index = len(model.encoder.layers) - 1
        prefixes = ("logit_head.", f"encoder.layers.{last_index}.")
    else:
        prefixes = (
            "card_emb.",
            "atk_emb.",
            "glob_proj.",
            "poke_proj.",
            "opt_proj.",
            "type_emb.",
            "encoder.",
            "logit_head.",
        )
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    option_mask: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Cross entropy whose smoothing mass covers legal options only."""
    log_probabilities = torch_functional.log_softmax(logits, dim=1)
    negative_log_likelihood = -log_probabilities.gather(1, labels[:, None]).squeeze(1)
    if smoothing <= 0.0:
        return negative_log_likelihood.mean()
    legal = option_mask.bool()
    smooth_loss = -log_probabilities.masked_fill(~legal, 0.0).sum(dim=1)
    smooth_loss = smooth_loss / legal.sum(dim=1).clamp(min=1)
    return ((1.0 - smoothing) * negative_log_likelihood + smoothing * smooth_loss).mean()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.marnie_train_repeat < 1 or args.second_train_repeat < 1:
        raise ValueError("Training repeat factors must be positive integers")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")

    candidate = args.candidate.resolve()
    sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    deck = [
        int(line)
        for line in (candidate / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:60]
    if len(deck) != 60:
        raise ValueError(f"Expected a 60-card deck, got {len(deck)}")
    deck_library_path = candidate / "deck_library.json"
    deck_library = (
        json.loads(deck_library_path.read_text(encoding="utf-8"))
        if deck_library_path.exists()
        else None
    )
    replay_paths = [
        path
        for path in sorted(
            set(args.replays.resolve().glob("episode_*.json"))
            | set(args.replays.resolve().glob("episode-*-replay.json"))
        )
        if matches_family(path, args.opponent_family)
        and matches_seat_role(path, args.seat_role)
    ]
    if args.max_games > 0:
        replay_paths = replay_paths[: args.max_games]
    if len(replay_paths) < 3:
        raise ValueError(f"Need at least three replay files, found {len(replay_paths)}")
    if args.frozen_split_summary:
        frozen = json.loads(
            args.frozen_split_summary.resolve().read_text(encoding="utf-8")
        )["split_games"]
        by_name = {path.name: path for path in replay_paths}
        requested_validation = list(frozen["validation"])
        requested_test = list(frozen["test"])
        validation_names = [name for name in requested_validation if name in by_name]
        test_names = [name for name in requested_test if name in by_name]
        missing_frozen = sorted(
            (set(requested_validation) | set(requested_test)) - set(by_name)
        )
        if (
            missing_frozen
            and args.opponent_family == "all"
            and args.seat_role == "all"
        ):
            raise ValueError(f"Frozen split games are unavailable: {missing_frozen}")
        if not validation_names or not test_names:
            raise ValueError(
                "Frozen split has no validation or test games after opponent-family filtering"
            )
        held_out = set(validation_names) | set(test_names)
        splits = {
            "train": [path for path in replay_paths if path.name not in held_out],
            "validation": [by_name[name] for name in validation_names],
            "test": [by_name[name] for name in test_names],
        }
    else:
        splits = deterministic_split(replay_paths, args.seed)

    data: dict[str, list[Example]] = {}
    extraction: dict[str, dict[str, int]] = {}
    for split_name, split_paths in splits.items():
        data[split_name], split_counts = load_examples(
            split_paths, deck, deck_library, Belief, features, args.focal_team
        )
        extraction[split_name] = dict(split_counts)
        if not data[split_name]:
            raise ValueError(f"No supported examples in {split_name} split")

    # Reweight only the optimization stream.  Metrics and frozen held-out
    # splits remain unweighted, so gains cannot be manufactured by duplicating
    # the targeted examples during evaluation.
    marnie_train_games = {
        path.name for path in splits["train"] if matches_family(path, "marnie")
    }
    second_train_games = {
        path.name for path in splits["train"] if matches_seat_role(path, "second")
    }
    training_examples: list[Example] = []
    weighted_example_count_by_group: Counter[str] = Counter()
    for example in data["train"]:
        is_marnie = example.game in marnie_train_games
        is_second = example.game in second_train_games
        repeat = (
            args.marnie_train_repeat if is_marnie else 1
        ) * (
            args.second_train_repeat if is_second else 1
        )
        training_examples.extend([example] * repeat)
        weighted_example_count_by_group[
            f"marnie={is_marnie},second={is_second}"
        ] += repeat

    source_model = load_policy(candidate / "model.pt", map_location=device).to(device)
    base_metrics = {
        name: evaluate(source_model, examples, args.batch_size, device)
        for name, examples in data.items()
    }
    model = copy.deepcopy(source_model)
    trainable = configure_trainable(model, args.mode)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    trainable_count = sum(parameter.numel() for parameter in trainable)

    output = args.output.resolve() / args.mode
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "model.pt"
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    order = list(range(len(training_examples)))
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(order)
        epoch_loss = 0.0
        seen = 0
        for start in range(0, len(order), args.batch_size):
            indexes = order[start : start + args.batch_size]
            selected = [training_examples[index] for index in indexes]
            batch = collate(selected, device)
            labels = torch.as_tensor(
                [example.action for example in selected],
                dtype=torch.long,
                device=device,
            )
            logits, _ = model.forward_batch(batch)
            loss = masked_cross_entropy(
                logits, labels, batch["opt_mask"], args.label_smoothing
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(selected)
            seen += len(selected)

        validation_metrics = evaluate(
            model, data["validation"], args.batch_size, device
        )
        row = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(seen, 1),
            "validation_loss": validation_metrics["loss"],
            "validation_top1": validation_metrics["top1"],
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if float(validation_metrics["loss"]) < best_validation_loss - 1.0e-5:
            best_validation_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            stale_epochs = 0
            save_policy(model, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    best_model = load_policy(best_path, map_location=device).to(device)
    tuned_metrics = {
        name: evaluate(best_model, examples, args.batch_size, device)
        for name, examples in data.items()
    }
    source_sha256 = hashlib.sha256((candidate / "model.pt").read_bytes()).hexdigest()
    tuned_sha256 = hashlib.sha256(best_path.read_bytes()).hexdigest()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "candidate": str(candidate),
            "model_sha256": source_sha256,
            "replays": str(args.replays.resolve()),
            "game_count": len(replay_paths),
            "opponent_family": args.opponent_family,
            "seat_role": args.seat_role,
            "frozen_split_summary": (
                str(args.frozen_split_summary.resolve())
                if args.frozen_split_summary
                else None
            ),
            "focal_team": args.focal_team,
            "deck_card_count": len(deck),
        },
        "alignment": "action paired with same seat's previous observation",
        "unsupported_policy_cases": (
            "optional empty selections and early stops are excluded because V110 "
            "always emits maxCount choices"
        ),
        "split_games": {
            name: [path.name for path in paths] for name, paths in splits.items()
        },
        "extraction": extraction,
        "training": {
            "mode": args.mode,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "trainable_parameters": trainable_count,
            "unweighted_train_examples": len(data["train"]),
            "weighted_train_examples": len(training_examples),
            "marnie_train_repeat": args.marnie_train_repeat,
            "second_train_repeat": args.second_train_repeat,
            "marnie_train_games": len(marnie_train_games),
            "second_train_games": len(second_train_games),
            "weighted_example_count_by_group": dict(
                weighted_example_count_by_group
            ),
            "elapsed_seconds": time.monotonic() - started,
            "history": history,
        },
        "base_metrics": base_metrics,
        "tuned_metrics": tuned_metrics,
        "tuned_model": str(best_path),
        "tuned_model_sha256": tuned_sha256,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    train(parse_args())
