"""Train the frozen Target-80 matchup-conditioned public value model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch
from torch import nn


EXPECTED_TREE_SHA256 = "e3897738ea62f6e231dc4d5e2256290408ad0c01ddbe96002d597252e1ec1fcf"
FOCAL_BOT = "HelbrosRocketSpidopsEarlyPadLethalV5"
SEED = 805001
EPOCHS = 40
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def tree_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{path.name}\t{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def split_for(filename: str) -> str:
    bucket = int(hashlib.sha256(filename.encode("utf-8")).hexdigest()[:8], 16) % 20
    if bucket <= 13:
        return "train"
    if bucket <= 16:
        return "validation"
    return "test"


def opponent_family(name: str) -> str:
    lowered = name.lower()
    if "marnie" in lowered:
        return "marnie"
    if "starmie" in lowered:
        return "starmie"
    if "alakazam" in lowered:
        return "alakazam"
    if "rocket" in lowered or "spidops" in lowered:
        return "rocket"
    return "other"


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metrics(records: list[dict], probabilities: list[float]) -> dict:
    labels = [int(record["label"]) for record in records]
    mean = sum(labels) / len(labels)
    brier = sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(labels)
    constant_brier = sum((mean - label) ** 2 for label in labels) / len(labels)
    return {
        "examples": len(labels),
        "games": len({record["game"] for record in records}),
        "win_rate": mean,
        "auc": auc(labels, probabilities),
        "brier": brier,
        "constant_brier": constant_brier,
        "accuracy": sum((probability >= 0.5) == bool(label) for probability, label in zip(probabilities, labels)) / len(labels),
    }


def load_records(dataset: Path, feature_module) -> tuple[dict[str, list[dict]], dict]:
    files = sorted(dataset.glob("*.jsonl"), key=lambda path: path.name)
    actual_hash = tree_digest(files)
    if len(files) != 1025 or actual_hash != EXPECTED_TREE_SHA256:
        raise ValueError(f"source drift: files={len(files)} tree={actual_hash}")
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    game_counts = Counter()
    outcome_counts = Counter()
    for path in files:
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if len(rows) < 3 or "meta" not in rows[0] or "result" not in rows[-1]:
            raise ValueError(f"malformed arena game: {path.name}")
        meta = rows[0]["meta"]
        players = meta["players"]
        if players.count(FOCAL_BOT) != 1:
            raise ValueError(f"missing focal bot: {path.name}")
        focal = players.index(FOCAL_BOT)
        opponent = players[1 - focal]
        opponent_deck = [int(card) for card in meta["decks"][1 - focal]]
        if len(opponent_deck) != 60:
            raise ValueError(f"opponent deck is not 60 cards: {path.name}")
        winner = int(rows[-1]["result"]["winner"])
        if winner not in (0, 1):
            raise ValueError(f"non-decisive source game: {path.name}")
        label = int(winner == focal)
        selected: dict[int, dict] = {}
        for row in rows[1:-1]:
            if int(row.get("p", -1)) != focal:
                continue
            observation = row.get("obs") or {}
            select = observation.get("select") or {}
            current = observation.get("current") or {}
            if select.get("type") != 0:
                continue
            turn = int(current.get("turn") or 0)
            selected.setdefault(turn, observation)
        if not selected:
            raise ValueError(f"no focal MAIN decisions: {path.name}")
        split = split_for(path.name)
        family = opponent_family(opponent)
        game_records = []
        for turn, observation in sorted(selected.items()):
            values = feature_module.extract_public_value_features(
                observation,
                root_player=focal,
                opponent_deck=opponent_deck,
            )
            game_records.append(
                {
                    "game": path.name,
                    "turn": turn,
                    "family": family,
                    "opponent": opponent,
                    "label": label,
                    "features": values,
                    "weight": 1.0 / len(selected),
                }
            )
        splits[split].extend(game_records)
        game_counts[split] += 1
        outcome_counts[(split, label)] += 1
    return splits, {
        "tree_sha256": actual_hash,
        "games": dict(game_counts),
        "outcomes": {f"{split}_{label}": count for (split, label), count in sorted(outcome_counts.items())},
        "examples": {split: len(records) for split, records in splits.items()},
    }


@torch.no_grad()
def probabilities(model, features, mean, std) -> list[float]:
    model.eval()
    logits = model((features - mean) / std)
    return torch.sigmoid(logits).cpu().tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    from soldier import public_value as value_support

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    splits, source = load_records(args.dataset.resolve(), value_support)
    dimensions = {len(record["features"]) for records in splits.values() for record in records}
    if len(dimensions) != 1:
        raise ValueError(f"feature dimensions changed: {dimensions}")
    feature_dim = dimensions.pop()

    tensors = {}
    for split, records in splits.items():
        tensors[split] = {
            "x": torch.tensor([record["features"] for record in records], dtype=torch.float32),
            "y": torch.tensor([record["label"] for record in records], dtype=torch.float32),
            "w": torch.tensor([record["weight"] for record in records], dtype=torch.float32),
        }
    train = tensors["train"]
    mean = train["x"].mean(dim=0)
    std = train["x"].std(dim=0, unbiased=False).clamp_min(1e-4)
    model = value_support.PublicValueNet(feature_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(SEED)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(train["x"].shape[0], generator=generator)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            x = (train["x"][index] - mean) / std
            y = train["y"][index]
            weights = train["w"][index]
            # Restore a mean-one scale while retaining equal total weight per game.
            weights = weights / weights.mean().clamp_min(1e-8)
            loss_rows = nn.functional.binary_cross_entropy_with_logits(
                model(x), y, reduction="none"
            )
            loss = (loss_rows * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float((loss_rows.detach() * weights).sum())
            total_weight += float(weights.sum())
        if epoch in (1, 10, 20, 30, 40):
            history.append({"epoch": epoch, "weighted_loss": total_loss / total_weight})

    split_metrics = {}
    split_probabilities = {}
    for split, records in splits.items():
        probs = probabilities(model, tensors[split]["x"], mean, std)
        split_probabilities[split] = probs
        overall = metrics(records, probs)
        early_pairs = [
            (record, probability)
            for record, probability in zip(records, probs)
            if record["turn"] <= 8
        ]
        early = metrics(
            [record for record, _ in early_pairs],
            [probability for _, probability in early_pairs],
        )
        families = {}
        grouped: dict[str, list[tuple[dict, float]]] = defaultdict(list)
        for record, probability in zip(records, probs):
            grouped[record["family"]].append((record, probability))
        for family, pairs in sorted(grouped.items()):
            families[family] = metrics(
                [record for record, _ in pairs],
                [probability for _, probability in pairs],
            )
        split_metrics[split] = {"overall": overall, "turn_lte_8": early, "families": families}

    major_family_failures = []
    for split in ("validation", "test"):
        for family, row in split_metrics[split]["families"].items():
            if row["examples"] >= 100 and row["auc"] is not None and row["auc"] < 0.52:
                major_family_failures.append({"split": split, "family": family, "auc": row["auc"]})
    gates = {
        "validation_auc": split_metrics["validation"]["overall"]["auc"] >= 0.62,
        "test_auc": split_metrics["test"]["overall"]["auc"] >= 0.62,
        "validation_early_auc": split_metrics["validation"]["turn_lte_8"]["auc"] >= 0.57,
        "test_early_auc": split_metrics["test"]["turn_lte_8"]["auc"] >= 0.57,
        "validation_brier": split_metrics["validation"]["overall"]["brier"] < split_metrics["validation"]["overall"]["constant_brier"],
        "test_brier": split_metrics["test"]["overall"]["brier"] < split_metrics["test"]["overall"]["constant_brier"],
        "major_family_auc": not major_family_failures,
    }

    metadata = {
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "source_tree_sha256": EXPECTED_TREE_SHA256,
    }
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    value_support.save_public_value(
        args.out_model, model, mean=mean, std=std, metadata=metadata
    )
    loaded = value_support.load_public_value(args.out_model)
    reload_ok = all(
        torch.equal(value.detach().cpu(), loaded.model.state_dict()[key])
        for key, value in model.state_dict().items()
    )
    reload_ok &= torch.equal(mean.detach().cpu(), loaded.mean)
    reload_ok &= torch.equal(std.detach().cpu(), loaded.std)
    for record, expected in list(zip(splits["test"], split_probabilities["test"]))[:32]:
        # Compare equal-shaped original/reloaded inference.  A batched GEMM and
        # a one-row GEMM need not round identically even with equal weights.
        values = torch.tensor(record["features"], dtype=torch.float32)
        with torch.no_grad():
            original = float(torch.sigmoid(model(((values - mean) / std).unsqueeze(0))[0]))
            actual = float(torch.sigmoid(loaded.model(((values - loaded.mean) / loaded.std).unsqueeze(0))[0]))
        reload_ok &= actual == original
    gates["serialization_reload"] = reload_ok
    admission = all(gates.values())

    report = {
        "schema": "rocket-target80-public-value-training/v2",
        "source": source,
        "feature_dim": feature_dim,
        "architecture": [feature_dim, 64, 32, 1],
        "training": metadata,
        "history": history,
        "metrics": split_metrics,
        "major_family_failures": major_family_failures,
        "gates": gates,
        "offline_admission": admission,
        "model_sha256": hashlib.sha256(args.out_model.read_bytes()).hexdigest(),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if admission else 2


if __name__ == "__main__":
    raise SystemExit(main())
