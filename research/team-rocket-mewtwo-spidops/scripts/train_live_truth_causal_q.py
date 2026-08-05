#!/usr/bin/env python3
"""Train a conservative action-value reranker from live-truth branches.

The frozen arena policy remains the blueprint.  A small Q head consumes its
contextual state/option tokens and predicts official terminal utility for only
the shortlist evaluated by ``collect_live_truth_causal_action_values.py``.
Complete source matches are split before fitting.  A validation-only margin
gate is selected, then opened once on an untouched test split.  No candidate is
declared deployable unless the clustered paired-improvement interval is above
zero on that test split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import monotonic
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from train_public_leader_v110_bc import Example, collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RawRoot:
    source: str
    game_key: str
    root_key: str
    observation: dict[str, Any]
    candidates: tuple[int, ...]
    utilities: tuple[float, ...]
    baseline: int


@dataclass(frozen=True)
class EncodedRoot:
    source: str
    game_key: str
    root_key: str
    global_token: torch.Tensor
    option_tokens: torch.Tensor
    base_logits: torch.Tensor
    candidates: torch.Tensor
    utilities: torch.Tensor
    baseline_position: int


class CausalQHead(nn.Module):
    def __init__(self, token_width: int, hidden: int = 192) -> None:
        super().__init__()
        self.token_width = int(token_width)
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.LayerNorm(2 * token_width + 1),
            nn.Linear(2 * token_width + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        global_token: torch.Tensor,
        option_tokens: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        count = option_tokens.shape[0]
        state = global_token.reshape(1, -1).expand(count, -1)
        joined = torch.cat((state, option_tokens, base_logits.reshape(-1, 1)), dim=-1)
        return self.network(joined).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_neural_resolver_safe_tie_v3"),
    )
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "live_truth_causal_q_v1"),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-roots", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--pairwise-coef", type=float, default=0.75)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=615_401)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=4_000)
    parser.add_argument(
        "--split-mode",
        choices=("fixed100", "random"),
        default="fixed100",
        help="Use game IDs 1-70/71-85/86-100 for the frozen run; random is smoke-only.",
    )
    return parser.parse_args()


def read_deck(path: Path) -> list[int]:
    deck = [int(value) for value in path.read_text(encoding="utf-8").replace(",", "\n").split()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(deck)}")
    return deck


def read_roots(paths: Sequence[Path]) -> tuple[list[RawRoot], Counter[str]]:
    roots = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for path in paths:
        source = path.resolve().name
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["rows"] += 1
            if row.get("schema") != "rocket-live-truth-causal-action/v1":
                counts["schema_rejected"] += 1
                continue
            if not row.get("fully_complete") or not isinstance(row.get("observation"), dict):
                counts["incomplete_rejected"] += 1
                continue
            candidates = tuple(int(index) for index in row.get("candidate_indices") or ())
            means = row.get("means") or {}
            baseline = int(row.get("baseline"))
            if len(candidates) < 2 or baseline not in candidates or any(str(index) not in means for index in candidates):
                counts["support_rejected"] += 1
                continue
            utilities = tuple(float(means[str(index)]) for index in candidates)
            if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in utilities):
                counts["utility_rejected"] += 1
                continue
            game_key = f"{source}:game={int(row['game_id'])}"
            root_key = f"{game_key}:root={int(row['root_index'])}"
            if root_key in seen:
                counts["duplicate_rejected"] += 1
                continue
            seen.add(root_key)
            roots.append(
                RawRoot(
                    source=source,
                    game_key=game_key,
                    root_key=root_key,
                    observation=row["observation"],
                    candidates=candidates,
                    utilities=utilities,
                    baseline=baseline,
                )
            )
    counts["accepted_roots"] = len(roots)
    counts["accepted_games"] = len({root.game_key for root in roots})
    return roots, counts


def split_games(
    roots: Sequence[RawRoot], seed: int, mode: str
) -> dict[str, set[str]]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for root in roots:
        by_source[root.source].add(root.game_key)
    splits = {"train": set(), "validation": set(), "test": set()}
    for source, games_set in sorted(by_source.items()):
        games = sorted(games_set)
        if mode == "fixed100":
            for game in games:
                game_id = int(game.rsplit("=", 1)[1])
                bucket = "train" if game_id <= 70 else "validation" if game_id <= 85 else "test"
                splits[bucket].add(game)
        else:
            random.Random(f"{seed}:{source}").shuffle(games)
            count = len(games)
            train_end = max(1, int(count * 0.70))
            validation_end = max(train_end + 1, int(count * 0.85))
            validation_end = min(validation_end, count)
            splits["train"].update(games[:train_end])
            splits["validation"].update(games[train_end:validation_end])
            splits["test"].update(games[validation_end:])
    all_games = set().union(*splits.values())
    expected = {root.game_key for root in roots}
    if all_games != expected or sum(len(values) for values in splits.values()) != len(expected):
        raise RuntimeError("complete-game split is not disjoint and exhaustive")
    if any(not values for values in splits.values()):
        raise RuntimeError("one or more complete-game splits are empty")
    return splits


def token_batch(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return model._tokens(
        batch["global"], batch["belief_global"], batch["hand_card"],
        batch["my_discard_card"], batch["opp_discard_card"], batch["stadium_card"],
        batch["my_unseen_card"], batch["opp_unseen_card"], batch["poke_feats"],
        batch["poke_card"], batch["opt_feats"], batch["opt_unseen"], batch["opt_card"],
        batch["opt_attack"], batch["opt_target"], batch["poke_mask"], batch["opt_mask"],
        batch.get("hand_mask"), batch.get("my_discard_mask"), batch.get("opp_discard_mask"),
        batch.get("my_unseen_mask"), batch.get("opp_unseen_mask"),
        kpc=batch.get("known_prized_card") if model.v4 else None,
        kpm=batch.get("known_prized_mask") if model.v4 else None,
        efc=batch.get("effect_card") if model.v4 else None,
        bpz=batch.get("belief_prized") if model.v4 else None,
        ctc=batch.get("context_card") if model.v5 else None,
        pwk=batch.get("poke_weak") if model.v5 else None,
    )


@torch.no_grad()
def encode_roots(
    raw_roots: Sequence[RawRoot],
    *,
    model: nn.Module,
    deck: Sequence[int],
    library: Mapping[str, Any],
    belief_class: Any,
    features_module: Any,
    batch_size: int = 64,
) -> tuple[list[EncodedRoot], Counter[str]]:
    prepared: list[tuple[RawRoot, Example]] = []
    counts: Counter[str] = Counter()
    for root in raw_roots:
        try:
            belief = belief_class(list(deck), opponent_deck_priors=library)
            feature = features_module.extract(root.observation, belief=belief, slot_link=True)
            if any(index >= feature["opt_card"].shape[0] for index in root.candidates):
                raise ValueError("candidate index exceeds extracted option support")
            prepared.append((root, Example(feature, root.baseline, root.game_key)))
        except Exception as error:
            counts[f"feature_error:{type(error).__name__}"] += 1
    encoded: list[EncodedRoot] = []
    reproduction = 0
    model.eval()
    for start in range(0, len(prepared), batch_size):
        selected = prepared[start : start + batch_size]
        batch = collate([example for _, example in selected], torch.device("cpu"))
        global_tokens, option_tokens = token_batch(model, batch)
        base_logits = model.logit_head(option_tokens).squeeze(-1)
        for row_index, (root, example) in enumerate(selected):
            option_count = int(example.feat["opt_card"].shape[0])
            logits = base_logits[row_index, :option_count].detach().cpu()
            reproduction += int(int(logits.argmax()) == root.baseline)
            candidates = torch.tensor(root.candidates, dtype=torch.long)
            encoded.append(
                EncodedRoot(
                    source=root.source,
                    game_key=root.game_key,
                    root_key=root.root_key,
                    global_token=global_tokens[row_index].detach().cpu(),
                    option_tokens=option_tokens[row_index, :option_count].detach().cpu(),
                    base_logits=logits,
                    candidates=candidates,
                    utilities=torch.tensor(root.utilities, dtype=torch.float32),
                    baseline_position=root.candidates.index(root.baseline),
                )
            )
    counts["encoded_roots"] = len(encoded)
    counts["fresh_belief_baseline_reproduced"] = reproduction
    return encoded, counts


def root_predictions(head: CausalQHead, root: EncodedRoot) -> torch.Tensor:
    return head(
        root.global_token,
        root.option_tokens[root.candidates],
        root.base_logits[root.candidates],
    )


def root_loss(head: CausalQHead, root: EncodedRoot, pairwise_coef: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = root_predictions(head, root)
    targets = root.utilities - root.utilities[root.baseline_position]
    mse = Fnn.mse_loss(prediction, targets)
    pair_losses = []
    for better in range(len(targets)):
        for worse in range(len(targets)):
            if float(targets[better] - targets[worse]) >= 0.5:
                pair_losses.append(Fnn.softplus(-(prediction[better] - prediction[worse])))
    pairwise = torch.stack(pair_losses).mean() if pair_losses else prediction.new_zeros(())
    return mse + pairwise_coef * pairwise, mse, pairwise


@torch.no_grad()
def prediction_rows(head: CausalQHead, roots: Sequence[EncodedRoot]) -> list[dict[str, Any]]:
    head.eval()
    rows = []
    for root in roots:
        prediction = root_predictions(head, root)
        targets = root.utilities - root.utilities[root.baseline_position]
        best_position = int(prediction.argmax())
        baseline_position = root.baseline_position
        rows.append(
            {
                "game": root.game_key,
                "root": root.root_key,
                "baseline_position": baseline_position,
                "best_position": best_position,
                "predicted_advantage": float(prediction[best_position] - prediction[baseline_position]),
                "baseline_utility": float(root.utilities[baseline_position]),
                "best_utility": float(root.utilities[best_position]),
                "mse": float(Fnn.mse_loss(prediction, targets)),
            }
        )
    return rows


def gate_metrics(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    deltas = []
    interventions = 0
    correct = 0
    for row in rows:
        use = (
            int(row["best_position"]) != int(row["baseline_position"])
            and float(row["predicted_advantage"]) >= threshold
        )
        delta = float(row["best_utility"] - row["baseline_utility"]) if use else 0.0
        deltas.append(delta)
        interventions += int(use)
        correct += int(use and delta > 0.0)
    return {
        "roots": len(rows),
        "threshold": float(threshold),
        "interventions": interventions,
        "intervention_rate": interventions / max(1, len(rows)),
        "intervention_precision": correct / max(1, interventions),
        "mean_paired_improvement": statistics.fmean(deltas) if deltas else 0.0,
        "mean_mse": statistics.fmean(float(row["mse"]) for row in rows) if rows else math.inf,
    }


def select_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    positive = sorted(
        {
            float(row["predicted_advantage"])
            for row in rows
            if int(row["best_position"]) != int(row["baseline_position"])
            and float(row["predicted_advantage"]) > 0.0
        }
    )
    candidates = {0.0}
    if positive:
        candidates.update(float(np.quantile(positive, value)) for value in np.linspace(0, 1, 41))
        candidates.add(max(positive) + 1e-6)
    table = [gate_metrics(rows, threshold) for threshold in sorted(candidates)]
    minimum = max(5, math.ceil(0.05 * len(rows)))
    eligible = [
        item for item in table
        if minimum <= item["interventions"] <= math.floor(0.40 * len(rows))
        and item["mean_paired_improvement"] > 0.0
        and item["intervention_precision"] >= 0.60
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item["mean_paired_improvement"],
                item["intervention_precision"],
                -item["intervention_rate"],
                item["threshold"],
            ),
        )
    else:
        maximum = max(positive, default=0.0)
        selected = gate_metrics(rows, maximum + 1e-6)
        table.append(selected)
    return float(selected["threshold"]), table


def clustered_interval(
    rows: Sequence[Mapping[str, Any]], threshold: float, *, samples: int, seed: int
) -> tuple[float, float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        use = (
            int(row["best_position"]) != int(row["baseline_position"])
            and float(row["predicted_advantage"]) >= threshold
        )
        grouped[str(row["game"])].append(
            float(row["best_utility"] - row["baseline_utility"]) if use else 0.0
        )
    games = sorted(grouped)
    observed_values = [value for game in games for value in grouped[game]]
    observed = statistics.fmean(observed_values) if observed_values else 0.0
    if not games or samples <= 0:
        return observed, observed, observed
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        chosen = [games[rng.randrange(len(games))] for _ in games]
        values = [value for game in chosen for value in grouped[game]]
        boot.append(statistics.fmean(values) if values else 0.0)
    boot.sort()
    lower = boot[max(0, int(0.025 * len(boot)) - 1)]
    upper = boot[min(len(boot) - 1, int(0.975 * len(boot)))]
    return observed, lower, upper


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    candidate = args.candidate.resolve()
    runtime = args.runtime.resolve()
    sys.path.insert(0, str(runtime))
    from soldier import features as feature_module  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy  # type: ignore

    data_paths = [path.resolve() for path in args.data]
    raw, read_counts = read_roots(data_paths)
    splits = split_games(raw, args.seed, args.split_mode)
    deck = read_deck(candidate / "deck.csv")
    library_path = candidate / "deck_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else {}
    model = load_policy(candidate / "model.pt").eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    encoded, encode_counts = encode_roots(
        raw,
        model=model,
        deck=deck,
        library=library,
        belief_class=Belief,
        features_module=feature_module,
    )
    data = {
        name: [root for root in encoded if root.game_key in games]
        for name, games in splits.items()
    }
    if any(not roots for roots in data.values()):
        raise RuntimeError("encoded split became empty")
    token_width = int(model.h)
    head = CausalQHead(token_width, args.hidden)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = math.inf
    best_epoch = 0
    patience = 0
    started = monotonic()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data["train"])))
        random.Random(args.seed + epoch).shuffle(order)
        head.train()
        totals = Counter()
        optimizer.zero_grad(set_to_none=True)
        for position, index in enumerate(order, start=1):
            loss, mse, pairwise = root_loss(head, data["train"][index], args.pairwise_coef)
            (loss / args.batch_roots).backward()
            totals["loss"] += float(loss.detach())
            totals["mse"] += float(mse.detach())
            totals["pairwise"] += float(pairwise.detach())
            if position % args.batch_roots == 0 or position == len(order):
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_rows = prediction_rows(head, data["validation"])
        validation_mse = statistics.fmean(row["mse"] for row in validation_rows)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(1, len(order)),
            "train_mse": totals["mse"] / max(1, len(order)),
            "train_pairwise": totals["pairwise"] / max(1, len(order)),
            "validation_mse": validation_mse,
            "elapsed_seconds": monotonic() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation_mse < best_score - 1e-6:
            best_score = validation_mse
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "schema": "rocket-live-truth-causal-q/v1",
                    "config": {"token_width": token_width, "hidden": args.hidden},
                    "state_dict": head.state_dict(),
                },
                output / "causal_q.pt",
            )
        else:
            patience += 1
            if patience >= args.patience:
                break

    artifact = torch.load(output / "causal_q.pt", map_location="cpu", weights_only=True)
    best = CausalQHead(**artifact["config"])
    best.load_state_dict(artifact["state_dict"])
    best.eval()
    rows_by_split = {name: prediction_rows(best, roots) for name, roots in data.items()}
    threshold, threshold_table = select_threshold(rows_by_split["validation"])
    metrics = {name: gate_metrics(rows, threshold) for name, rows in rows_by_split.items()}
    observed, lower, upper = clustered_interval(
        rows_by_split["test"], threshold, samples=args.bootstrap_samples, seed=args.seed + 99
    )
    test_gate = metrics["test"]
    admitted = bool(
        test_gate["interventions"] >= 5
        and test_gate["intervention_rate"] <= 0.40
        and test_gate["intervention_precision"] >= 0.60
        and observed > 0.0
        and lower > 0.0
    )
    summary = {
        "schema": "rocket-live-truth-causal-q-training/v1",
        "method": "frozen contextual blueprint tokens plus causal utility Q head",
        "deployment_authority": "admitted only by untouched clustered paired-improvement gate",
        "source": {
            "candidate": str(candidate),
            "candidate_model_sha256": sha(candidate / "model.pt"),
            "data": [str(path) for path in data_paths],
            "data_sha256": {str(path): sha(path) for path in data_paths},
        },
        "read_counts": dict(read_counts),
        "encode_counts": dict(encode_counts),
        "split_games": {name: sorted(games) for name, games in splits.items()},
        "split_roots": {name: len(roots) for name, roots in data.items()},
        "training": {
            "seed": args.seed,
            "best_epoch": best_epoch,
            "history": history,
            "pairwise_coef": args.pairwise_coef,
            "hidden": args.hidden,
        },
        "gate": {
            "threshold": threshold,
            "selection_table": threshold_table,
            "metrics": metrics,
            "test_clustered_95_interval": [lower, upper],
            "test_observed_mean": observed,
            "admitted": admitted,
        },
        "claims": {
            "official_descendant_execution": True,
            "realized_hidden_state_samples": True,
            "public_information_set_expected_values": False,
            "equilibrium_certificate": False,
            "safe_strategy_certificate": False,
            "arena_strength_improvement": False,
        },
        "artifact": str(output / "causal_q.pt"),
        "artifact_sha256": sha(output / "causal_q.pt"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "threshold": threshold,
                "validation": metrics["validation"],
                "test": metrics["test"],
                "test_clustered_95_interval": [lower, upper],
                "admitted": admitted,
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
