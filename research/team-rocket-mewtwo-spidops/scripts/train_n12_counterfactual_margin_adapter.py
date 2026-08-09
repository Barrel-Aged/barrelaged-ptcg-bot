#!/usr/bin/env python3
"""Train the preregistered N12 contextual counterfactual-margin adapter."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as Fnn


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
CORPUS = ROOT / "artifacts/team_rocket_recovery/n6r_exact_feature"
AUDIT = ROOT / "artifacts/team_rocket_recovery/n12_counterfactual_transfer_audit.json"
REPLAYS = ROOT / "work/n9_balanced_exact_replays"
PARITY = ROOT / "artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/exact_agent_parity.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n12_counterfactual_margin_adapter/training_v3_norm2"

MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
AUDIT_SHA = "9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313"
PARITY_SHA = "ebe70ea381a2f8eef38434ea8ca8ee2f08c867e03a497518520c17188758935e"
REPLAY_TREE_SHA = "f430676aa3c7786e969d35f0725c70015742adcef5d4e4d4408293e0655cd7ae"
PROTOCOL = "N12_NORM2_MARGIN_ADAPTER_PROTOCOL.md"
PREDECESSOR_PROTOCOL = "N12_COUNTERFACTUAL_MARGIN_ADAPTER_PROTOCOL.md"
BOT = "N9RocketControl"
GROUPS = (
    "AshMarnieGeneralV110_extracted",
    "HelbrosMarnieGeneralRecoveryV4_extracted",
)
THRESHOLDS = (12, 24, 36, 48)
PERMITTED_PREFIXES = ("encoder.layers.2.norm2.", "logit_head.")


sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(BASE))

from rocketbot.policy_feature_snapshot import deserialize_policy_feature  # noqa: E402
from train_belief_rollout_distillation_n6 import read_rows  # noqa: E402
from train_live_outcome_awr import (  # noqa: E402
    WeightedExample,
    collate_like_live_agent,
    deployment_end_indices,
    deployment_logits,
    extract_examples,
)
from train_n9_balanced_pcgrad_awr import (  # noqa: E402
    load_games,
    split_games,
    validate_parity,
)


@dataclass
class N12Example(WeightedExample):
    reference_probabilities: np.ndarray | None = None
    reference_top: int = -1
    reference_margin: float = 0.0
    teacher_best: int = -1
    correction: bool = False
    cell: str = ""
    cached_tokens: np.ndarray | None = None
    option_start: int = -1
    deployment_available: np.ndarray | None = None
    reference_relative_logits: np.ndarray | None = None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=BASE)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--replays", type=Path, default=REPLAYS)
    parser.add_argument("--parity-report", type=Path, default=PARITY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--anchor-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5.0e-7)
    parser.add_argument("--correction-margin", type=float, default=0.10)
    parser.add_argument("--retention-cap", type=float, default=0.25)
    parser.add_argument("--softplus-temperature", type=float, default=0.25)
    parser.add_argument("--n6-kl-coef", type=float, default=1.0)
    parser.add_argument("--n9-kl-coef", type=float, default=1.0)
    parser.add_argument("--n6-retention-coef", type=float, default=0.25)
    parser.add_argument("--n9-retention-coef", type=float, default=0.25)
    parser.add_argument("--displacement-coef", type=float, default=0.01)
    parser.add_argument("--gradient-cap", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=812101)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    locked = {
        "steps": 400,
        "anchor_batch_size": 128,
        "learning_rate": 5.0e-7,
        "correction_margin": 0.10,
        "retention_cap": 0.25,
        "softplus_temperature": 0.25,
        "n6_kl_coef": 1.0,
        "n9_kl_coef": 1.0,
        "n6_retention_coef": 0.25,
        "n9_retention_coef": 0.25,
        "displacement_coef": 0.01,
        "gradient_cap": 0.25,
        "seed": 812101,
        "threads": 4,
    }
    for key, expected in locked.items():
        if getattr(args, key) != expected:
            raise ValueError(f"N12 locked argument changed: {key}")


def repeated_order(length: int, target: int, seed: int) -> list[int]:
    if length <= 0 or target <= 0:
        raise ValueError("N12 repeated order requires positive sizes")
    order: list[int] = []
    cycle = 0
    while len(order) < target:
        values = list(range(length))
        random.Random(f"{seed}:{cycle}").shuffle(values)
        order.extend(values)
        cycle += 1
    return order[:target]


def correction_cell_weights(examples: Sequence[N12Example]) -> torch.Tensor:
    counts = Counter(example.cell for example in examples)
    if set(counts) != {
        f"{GROUPS[0]}:turn_2",
        f"{GROUPS[0]}:turn_4",
        f"{GROUPS[1]}:turn_2",
        f"{GROUPS[1]}:turn_4",
    }:
        raise ValueError(f"N12 correction cells changed: {dict(counts)}")
    weights = np.asarray(
        [1.0 / (len(counts) * counts[example.cell]) for example in examples],
        dtype=np.float32,
    )
    if not np.isclose(weights.sum(), 1.0, atol=1.0e-7):
        raise AssertionError("N12 correction weights do not sum to one")
    return torch.from_numpy(weights)


def configure_trainable(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    last_index = len(model.encoder.layers) - 1
    expected = (f"encoder.layers.{last_index}.norm2.", "logit_head.")
    if expected != PERMITTED_PREFIXES:
        raise ValueError(f"N12 architecture changed final layer index: {last_index}")
    output = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PERMITTED_PREFIXES))
        if parameter.requires_grad:
            output.append(parameter)
    if sum(parameter.numel() for parameter in output) != 1_153:
        raise ValueError("N12 trainable parameter count changed")
    return output


def changed_keys(reference: torch.nn.Module, model: torch.nn.Module) -> list[str]:
    before = reference.state_dict()
    return sorted(
        key for key, value in model.state_dict().items()
        if not torch.equal(value, before[key])
    )


def permitted_changes(keys: Iterable[str]) -> bool:
    materialized = list(keys)
    return bool(materialized) and all(key.startswith(PERMITTED_PREFIXES) for key in materialized)


def wrap_example(example: WeightedExample) -> N12Example:
    return N12Example(
        feat=example.feat,
        action=example.action,
        game=example.game,
        won=example.won,
        group=example.group,
        advantage=example.advantage,
        weight=example.weight,
        deployment_end_indices=example.deployment_end_indices,
    )


def _row_files(corpus: Path, prefix: str) -> list[Path]:
    return sorted(corpus.glob(f"{prefix}_*.jsonl"), key=lambda path: path.name)


def load_n6_examples(corpus: Path, audit_path: Path) -> tuple[list[N12Example], dict[str, Any]]:
    if digest(audit_path) != AUDIT_SHA:
        raise ValueError("N12 transfer audit changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("live_model_sha256") != MODEL_SHA or audit.get("deck_sha256") != DECK_SHA:
        raise ValueError("N12 audit substrate changed")
    paths = [*_row_files(corpus, "train"), *_row_files(corpus, "selection")]
    if len(paths) != 14:
        raise ValueError(f"N12 expected 14 N6R files, found {len(paths)}")
    rows, hashes = read_rows(paths)
    expected_hashes = audit.get("input_sha256") or {}
    for path in paths:
        if expected_hashes.get(str(path.resolve())) != digest(path):
            raise ValueError(f"N12 N6R input differs from transfer audit: {path}")
    estimates: dict[str, dict[str, Any]] = {}
    analysis_hashes = {}
    for name in ("train_analysis.json", "selection_analysis.json"):
        path = corpus / name
        analysis_hashes[str(path.resolve())] = digest(path)
        analysis = json.loads(path.read_text(encoding="utf-8"))
        for estimate in analysis["roots"]:
            fingerprint = str(estimate["observation_fingerprint"])
            if fingerprint in estimates:
                raise ValueError(f"duplicate N12 estimate: {fingerprint}")
            estimates[fingerprint] = estimate
    if len(rows) != 639 or len(estimates) != 639:
        raise ValueError(f"N12 root count changed: rows={len(rows)}, estimates={len(estimates)}")
    examples = []
    corrections = 0
    for index, row in enumerate(rows):
        fingerprint = str(row["observation_fingerprint"])
        estimate = estimates[fingerprint]
        admissible = bool(estimate["admissible_change"])
        best = int(estimate["best"])
        baseline = int(row["baseline"])
        if admissible and best == baseline:
            raise ValueError("N12 admissible correction equals baseline")
        feature = deserialize_policy_feature(row["policy_feature"])
        example = N12Example(
            feat=feature,
            action=baseline,
            game=f"n6r:{fingerprint}",
            won=False,
            group=str(row["opponent"]),
            deployment_end_indices=deployment_end_indices(row["observation"]),
            teacher_best=best,
            correction=admissible,
            cell=f"{row['opponent']}:turn_{int(row['turn'])}",
        )
        if admissible:
            corrections += 1
        examples.append(example)
    if corrections != 48:
        raise ValueError(f"N12 correction count changed: {corrections}")
    return examples, {
        "rows": len(rows),
        "corrections": corrections,
        "input_hashes": hashes,
        "analysis_hashes": analysis_hashes,
        "audit_sha256": digest(audit_path),
    }


def load_n9_examples(
    candidate: Path,
    replays: Path,
    parity_path: Path,
    deck: list[int],
    deck_library: Any,
    belief_class: Any,
    features_module: Any,
) -> tuple[list[N12Example], dict[str, Any]]:
    if digest(parity_path) != PARITY_SHA:
        raise ValueError("N12 N9 parity report changed")
    parity = validate_parity(parity_path, candidate, replays)
    games, counts = load_games(replays)
    splits = split_games(games)
    raw, extraction = extract_examples(
        splits["test"], BOT, deck, deck_library, belief_class, features_module
    )
    examples = [wrap_example(example) for example in raw]
    by_group = Counter(example.group for example in examples)
    expected = {GROUPS[0]: 3_456, GROUPS[1]: 3_297}
    if len(examples) != 6_753 or dict(by_group) != expected:
        raise ValueError(f"N12 N9 preservation support changed: {dict(by_group)}")
    return examples, {
        "replay_tree_sha256": REPLAY_TREE_SHA,
        "parity": parity,
        "all_games": len(games),
        "test_games": len(splits["test"]),
        "examples": len(examples),
        "examples_by_group": dict(by_group),
        "extraction": dict(extraction),
    }


@torch.no_grad()
def attach_reference(
    model: torch.nn.Module,
    examples: Sequence[N12Example],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Attach reference targets and cache the exact input to the trainable block."""

    model.eval()
    mismatches = 0
    final_layer = model.encoder.layers[-1]
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start:start + batch_size])
        batch, availability = collate_like_live_agent(selected, torch.device("cpu"))
        captured: list[torch.Tensor] = []
        hook = final_layer.norm2.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            logits, _ = model.forward_batch(batch)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError(f"N12 final-block cache hook count changed: {len(captured)}")
        prefinal = captured[0]
        logits = deployment_logits(logits, availability, selected)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        poke_width = int(batch["poke_mask"].shape[1])
        for row, (example, prediction) in enumerate(zip(selected, predictions.tolist())):
            width = int(example.feat["opt_feats"].shape[0])
            probability = probabilities[row, :width].cpu().numpy().astype(np.float32)
            example.reference_probabilities = probability
            example.reference_top = int(prediction)
            legal_logits = logits[row, :width].clone()
            legal_logits[example.reference_top] = -torch.inf
            runner_up = float(legal_logits.max())
            top = float(logits[row, example.reference_top])
            example.reference_margin = max(0.0, top - runner_up)
            relative = logits[row, :width] - logits[row, example.reference_top]
            example.reference_relative_logits = relative.cpu().numpy().astype(np.float32)
            real_poke = batch["poke_mask"][row].bool()
            global_token = prefinal[row, :1]
            poke_tokens = prefinal[row, 1:1 + poke_width][real_poke]
            option_tokens = prefinal[row, 1 + poke_width:1 + poke_width + width]
            tokens = torch.cat([global_token, poke_tokens, option_tokens], dim=0)
            example.cached_tokens = tokens.cpu().numpy().astype(np.float32)
            example.option_start = 1 + int(real_poke.sum())
            example.deployment_available = (
                availability[row, :width].bool().cpu().numpy().astype(np.bool_)
            )
            mismatches += int(example.reference_top != example.action)
    cache_top_mismatches = 0
    cache_max_relative_logit_error = 0.0
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start:start + batch_size])
        cached = cached_deployed(model, selected, torch.device("cpu"))
        predictions = cached.argmax(dim=1).tolist()
        for row, (example, prediction) in enumerate(zip(selected, predictions)):
            width = int(example.feat["opt_feats"].shape[0])
            relative = cached[row, :width] - cached[row, example.reference_top]
            expected = torch.as_tensor(example.reference_relative_logits)
            finite = torch.isfinite(expected) & (expected > -1.0e8)
            if bool(finite.any()):
                error = float((relative[finite] - expected[finite]).abs().max())
                cache_max_relative_logit_error = max(cache_max_relative_logit_error, error)
            cache_top_mismatches += int(prediction != example.reference_top)
    return {
        "examples": len(examples),
        "reference_action_mismatches": mismatches,
        "cache_top_mismatches": cache_top_mismatches,
        "cache_max_relative_logit_error": cache_max_relative_logit_error,
    }


def pad_reference(examples: Sequence[N12Example], width: int, device: torch.device) -> torch.Tensor:
    values = np.zeros((len(examples), width), dtype=np.float32)
    for row, example in enumerate(examples):
        if example.reference_probabilities is None:
            raise ValueError("N12 reference probabilities are absent")
        probability = example.reference_probabilities
        values[row, : len(probability)] = probability
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def full_forward_deployed(
    model: torch.nn.Module,
    examples: Sequence[N12Example],
    device: torch.device,
) -> torch.Tensor:
    selected = list(examples)
    batch, availability = collate_like_live_agent(selected, device)
    logits, _ = model.forward_batch(batch)
    return deployment_logits(logits, availability, selected)


def cached_deployed(
    model: torch.nn.Module,
    examples: Sequence[N12Example],
    device: torch.device,
) -> torch.Tensor:
    """Run the trainable final norm2 and head from post-attention cached tokens."""

    if not examples:
        raise ValueError("N12 cached forward received an empty batch")
    hidden = int(model.logit_head.in_features)
    sequence_width = max(
        len(example.cached_tokens) if example.cached_tokens is not None else 0
        for example in examples
    )
    option_width = max(int(example.feat["opt_feats"].shape[0]) for example in examples)
    tokens = torch.zeros(
        len(examples), sequence_width, hidden, dtype=torch.float32, device=device
    )
    padding = torch.ones(
        len(examples), sequence_width, dtype=torch.bool, device=device
    )
    for row, example in enumerate(examples):
        if example.cached_tokens is None or example.option_start < 0:
            raise ValueError("N12 cached final-block input is absent")
        values = torch.as_tensor(example.cached_tokens, dtype=torch.float32, device=device)
        tokens[row, : values.shape[0]] = values
        padding[row, : values.shape[0]] = False
    final_layer = model.encoder.layers[-1]
    # ``norm_first=True``: the norm2 input is the residual stream after the
    # frozen self-attention sublayer. The remaining frozen feed-forward block
    # is token-local, so removing ignored padding is exact up to float noise.
    encoded = tokens + final_layer._ff_block(final_layer.norm2(tokens))
    logits = torch.full(
        (len(examples), option_width), -1.0e9, dtype=torch.float32, device=device
    )
    for row, example in enumerate(examples):
        width = int(example.feat["opt_feats"].shape[0])
        option_tokens = encoded[row, example.option_start:example.option_start + width]
        row_logits = model.logit_head(option_tokens).squeeze(-1)
        if example.deployment_available is None:
            raise ValueError("N12 cached deployment mask is absent")
        available = torch.as_tensor(
            example.deployment_available, dtype=torch.bool, device=device
        )
        row_logits = row_logits.masked_fill(~available, -1.0e9)
        for index in example.deployment_end_indices:
            if 0 <= index < width:
                row_logits[index] = -1.0e9
        logits[row, :width] = row_logits
    return logits


def forward_kl(logits: torch.Tensor, examples: Sequence[N12Example]) -> torch.Tensor:
    reference = pad_reference(examples, logits.shape[1], logits.device)
    logp = torch.log_softmax(logits, dim=1)
    reference_log = torch.log(reference.clamp_min(1.0e-12))
    return (reference * (reference_log - logp)).sum(dim=1).mean()


def strongest_other(logits: torch.Tensor, winners: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked.scatter_(1, winners[:, None], -1.0e9)
    return masked.max(dim=1).values


def smooth_margin_loss(
    margins: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    values = Fnn.softplus((targets - margins) / temperature) * temperature
    if weights is None:
        return values.mean()
    normalized = weights / weights.sum().clamp_min(1.0e-12)
    return (values * normalized).sum()


def correction_loss(
    logits: torch.Tensor,
    examples: Sequence[N12Example],
    *,
    margin: float,
    temperature: float,
    weights: torch.Tensor,
) -> torch.Tensor:
    winners = torch.tensor(
        [example.teacher_best for example in examples], dtype=torch.long, device=logits.device
    )
    winner_logits = logits.gather(1, winners[:, None]).squeeze(1)
    margins = winner_logits - strongest_other(logits, winners)
    targets = torch.full_like(margins, margin)
    return smooth_margin_loss(
        margins, targets, temperature=temperature, weights=weights.to(logits.device)
    )


def retention_loss(
    logits: torch.Tensor,
    examples: Sequence[N12Example],
    *,
    cap: float,
    temperature: float,
) -> torch.Tensor:
    winners = torch.tensor(
        [example.reference_top for example in examples], dtype=torch.long, device=logits.device
    )
    winner_logits = logits.gather(1, winners[:, None]).squeeze(1)
    margins = winner_logits - strongest_other(logits, winners)
    targets = torch.tensor(
        [min(example.reference_margin, cap) for example in examples],
        dtype=logits.dtype,
        device=logits.device,
    )
    return smooth_margin_loss(margins, targets, temperature=temperature)


def relative_displacement(
    model: torch.nn.Module,
    reference_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    numerator = None
    denominator = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter - reference_state[name].to(parameter.device)
        value = (delta * delta).sum()
        numerator = value if numerator is None else numerator + value
        denominator += float((reference_state[name] ** 2).sum())
    if numerator is None or denominator <= 0.0:
        raise ValueError("N12 trainable displacement is undefined")
    return numerator / denominator


@torch.no_grad()
def correction_fit(model: torch.nn.Module, examples: Sequence[N12Example]) -> dict[str, Any]:
    model.eval()
    logits = cached_deployed(model, examples, torch.device("cpu"))
    predictions = logits.argmax(dim=1).tolist()
    by_cell: Counter[str] = Counter()
    for example, prediction in zip(examples, predictions):
        by_cell[example.cell] += int(prediction == example.teacher_best)
    fitted = sum(by_cell.values())
    return {"examples": len(examples), "fitted": fitted, "by_cell": dict(sorted(by_cell.items()))}


@torch.no_grad()
def preservation_metrics(
    model: torch.nn.Module,
    examples: Sequence[N12Example],
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    total = retained = 0
    kl_total = 0.0
    grouped: dict[str, Counter[str]] = {}
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start:start + batch_size])
        logits = cached_deployed(model, selected, torch.device("cpu"))
        reference = pad_reference(selected, logits.shape[1], logits.device)
        logp = torch.log_softmax(logits, dim=1)
        reference_log = torch.log(reference.clamp_min(1.0e-12))
        row_kl = (reference * (reference_log - logp)).sum(dim=1)
        predictions = logits.argmax(dim=1).tolist()
        for row, (example, prediction) in enumerate(zip(selected, predictions)):
            keep = int(prediction == example.reference_top)
            total += 1
            retained += keep
            kl_total += float(row_kl[row])
            group = grouped.setdefault(example.group, Counter())
            group["examples"] += 1
            group["retained"] += keep
    return {
        "examples": total,
        "retained": retained,
        "top1_retention": retained / max(1, total),
        "mean_forward_kl": kl_total / max(1, total),
        "by_group": {
            name: {
                **dict(counts),
                "top1_retention": counts["retained"] / max(1, counts["examples"]),
            }
            for name, counts in sorted(grouped.items())
        },
    }


def admission(
    threshold: int,
    fit: dict[str, Any],
    n6: dict[str, Any],
    n9: dict[str, Any],
    keys: list[str],
) -> dict[str, Any]:
    gates = {
        "permitted_state_keys_only": permitted_changes(keys),
        "named_correction_threshold": int(fit["fitted"]) >= threshold,
        "n6_top1_retention_ge_0_98": float(n6["top1_retention"]) >= 0.98,
        "n9_top1_retention_ge_0_99": float(n9["top1_retention"]) >= 0.99,
        "n9_each_group_retention_ge_0_985": all(
            float(row["top1_retention"]) >= 0.985 for row in n9["by_group"].values()
        ),
        "n6_mean_kl_le_0_05": float(n6["mean_forward_kl"]) <= 0.05,
        "n9_mean_kl_le_0_02": float(n9["mean_forward_kl"]) <= 0.02,
    }
    return {"passes": all(gates.values()), "gates": gates}


def backward_anchor_objective(
    model: torch.nn.Module,
    examples: Sequence[N12Example],
    *,
    kl_coefficient: float,
    retention_coefficient: float,
    retention_cap: float,
    temperature: float,
    microbatch_size: int = 32,
) -> tuple[float, float]:
    """Backpropagate one logical anchor mean in memory-bounded microbatches."""

    if not examples:
        raise ValueError("N12 anchor objective is empty")
    kl_total = retention_total = 0.0
    total = len(examples)
    for start in range(0, total, microbatch_size):
        selected = list(examples[start:start + microbatch_size])
        scale = len(selected) / total
        logits = cached_deployed(model, selected, torch.device("cpu"))
        kl = forward_kl(logits, selected)
        retention = retention_loss(
            logits,
            selected,
            cap=retention_cap,
            temperature=temperature,
        )
        objective = scale * (
            kl_coefficient * kl + retention_coefficient * retention
        )
        objective.backward()
        kl_total += scale * float(kl.detach())
        retention_total += scale * float(retention.detach())
    return kl_total, retention_total


def main() -> int:
    args = parse_args()
    validate_args(args)
    candidate = args.candidate.resolve()
    corpus = args.corpus.resolve()
    audit_path = args.audit.resolve()
    replays = args.replays.resolve()
    parity_path = args.parity_report.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N12 output: {output}")
    if digest(candidate / "model.pt") != MODEL_SHA or digest(candidate / "deck.csv") != DECK_SHA:
        raise ValueError("N12 frozen live substrate changed")
    if digest(ROOT / PROTOCOL) == "":  # pragma: no cover - explicit provenance read
        raise AssertionError("unreachable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    deck = [
        int(value)
        for value in (candidate / "deck.csv").read_text(encoding="utf-8").replace(",", "\n").split()
    ]
    library_path = candidate / "deck_library.json"
    deck_library = json.loads(library_path.read_text(encoding="utf-8"))

    reference = load_policy(candidate / "model.pt", map_location=device).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    n6_all, n6_provenance = load_n6_examples(corpus, audit_path)
    n9_all, n9_provenance = load_n9_examples(
        candidate, replays, parity_path, deck, deck_library, Belief, features
    )
    n6_reference = attach_reference(reference, n6_all)
    n9_reference = attach_reference(reference, n9_all)
    if n6_reference["reference_action_mismatches"] != 0:
        raise ValueError(f"N12 N6 exact parity failed: {n6_reference}")
    if n9_reference["reference_action_mismatches"] != 0:
        raise ValueError(f"N12 N9 exact parity failed: {n9_reference}")
    for name, result in (("N6", n6_reference), ("N9", n9_reference)):
        if result["cache_top_mismatches"] != 0:
            raise ValueError(f"N12 {name} cached top-action parity failed: {result}")
        if float(result["cache_max_relative_logit_error"]) > 2.0e-5:
            raise ValueError(f"N12 {name} cached logit parity failed: {result}")
    corrections = [example for example in n6_all if example.correction]
    n6_anchors = [example for example in n6_all if not example.correction]
    if len(corrections) != 48 or len(n6_anchors) != 591:
        raise ValueError("N12 correction/anchor split changed")
    cell_weights = correction_cell_weights(corrections).to(device)

    model = copy.deepcopy(reference)
    trainable = configure_trainable(model)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.0
    )
    reference_state = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    target_order_size = args.steps * args.anchor_batch_size
    n6_order = repeated_order(len(n6_anchors), target_order_size, args.seed + 101)
    n9_order = repeated_order(len(n9_all), target_order_size, args.seed + 202)

    output.mkdir(parents=True, exist_ok=False)
    checkpoint_steps: dict[int, int] = {}
    checkpoint_paths: dict[int, Path] = {}
    history = []
    started = monotonic()
    for step in range(1, args.steps + 1):
        begin = (step - 1) * args.anchor_batch_size
        end = step * args.anchor_batch_size
        n6_batch = [n6_anchors[index] for index in n6_order[begin:end]]
        n9_batch = [n9_all[index] for index in n9_order[begin:end]]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        correction_logits = cached_deployed(model, corrections, device)
        correction = correction_loss(
            correction_logits,
            corrections,
            margin=args.correction_margin,
            temperature=args.softplus_temperature,
            weights=cell_weights,
        )
        correction.backward()
        n6_kl, n6_retention = backward_anchor_objective(
            model,
            n6_batch,
            kl_coefficient=args.n6_kl_coef,
            retention_coefficient=args.n6_retention_coef,
            retention_cap=args.retention_cap,
            temperature=args.softplus_temperature,
        )
        n9_kl, n9_retention = backward_anchor_objective(
            model,
            n9_batch,
            kl_coefficient=args.n9_kl_coef,
            retention_coefficient=args.n9_retention_coef,
            retention_cap=args.retention_cap,
            temperature=args.softplus_temperature,
        )
        displacement = relative_displacement(model, reference_state)
        (args.displacement_coef * displacement).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_cap)
        optimizer.step()
        components = {
            "correction": float(correction.detach()),
            "n6_kl": n6_kl,
            "n9_kl": n9_kl,
            "n6_retention": n6_retention,
            "n9_retention": n9_retention,
            "displacement": float(displacement.detach()),
        }
        loss_value = (
            components["correction"]
            + args.n6_kl_coef * components["n6_kl"]
            + args.n9_kl_coef * components["n9_kl"]
            + args.n6_retention_coef * components["n6_retention"]
            + args.n9_retention_coef * components["n9_retention"]
            + args.displacement_coef * components["displacement"]
        )
        fit = correction_fit(model, corrections)
        record = {
            "step": step,
            "loss": loss_value,
            "gradient_norm_before_clip": float(gradient_norm),
            "correction_fit": fit,
            "components": components,
        }
        history.append(record)
        for threshold in THRESHOLDS:
            if threshold not in checkpoint_steps and int(fit["fitted"]) >= threshold:
                path = output / f"fit{threshold:02d}.pt"
                save_policy(model, path)
                checkpoint_steps[threshold] = step
                checkpoint_paths[threshold] = path
        if step == 1 or step % 10 == 0 or any(value == step for value in checkpoint_steps.values()):
            print(json.dumps(record, sort_keys=True), flush=True)

    evaluations: dict[str, Any] = {}
    seen_hashes: dict[str, str] = {}
    for threshold in THRESHOLDS:
        if threshold not in checkpoint_paths:
            continue
        path = checkpoint_paths[threshold]
        artifact_hash = digest(path)
        duplicate_of = seen_hashes.get(artifact_hash)
        if duplicate_of is None:
            seen_hashes[artifact_hash] = f"fit{threshold:02d}"
        checkpoint = load_policy(path, map_location=device).to(device).eval()
        keys = changed_keys(reference, checkpoint)
        fit = correction_fit(checkpoint, corrections)
        n6_metrics = preservation_metrics(checkpoint, n6_anchors)
        n9_metrics = preservation_metrics(checkpoint, n9_all)
        gate = admission(threshold, fit, n6_metrics, n9_metrics, keys)
        if duplicate_of is not None:
            gate["passes"] = False
            gate["gates"]["distinct_checkpoint"] = False
        else:
            gate["gates"]["distinct_checkpoint"] = True
        evaluations[f"fit{threshold:02d}"] = {
            "threshold": threshold,
            "step": checkpoint_steps[threshold],
            "path": str(path),
            "sha256": artifact_hash,
            "duplicate_of": duplicate_of,
            "changed_state_keys": keys,
            "correction_fit": fit,
            "n6_preservation": n6_metrics,
            "n9_preservation": n9_metrics,
            "admission": gate,
        }

    admitted = [
        name for name, evaluation in evaluations.items()
        if evaluation["admission"]["passes"]
    ]
    report = {
        "schema": "rocket-n12-counterfactual-margin-training/v3",
        "protocol": PROTOCOL,
        "protocol_sha256": digest(ROOT / PROTOCOL),
        "predecessor_protocol": PREDECESSOR_PROTOCOL,
        "predecessor_protocol_sha256": digest(ROOT / PREDECESSOR_PROTOCOL),
        "model_sha256": MODEL_SHA,
        "deck_sha256": DECK_SHA,
        "audit_sha256": AUDIT_SHA,
        "parity_sha256": PARITY_SHA,
        "replay_tree_sha256": REPLAY_TREE_SHA,
        "hyperparameters": {
            "steps": args.steps,
            "anchor_batch_size": args.anchor_batch_size,
            "learning_rate": args.learning_rate,
            "correction_margin": args.correction_margin,
            "retention_cap": args.retention_cap,
            "softplus_temperature": args.softplus_temperature,
            "n6_kl_coefficient": args.n6_kl_coef,
            "n9_kl_coefficient": args.n9_kl_coef,
            "n6_retention_coefficient": args.n6_retention_coef,
            "n9_retention_coefficient": args.n9_retention_coef,
            "displacement_coefficient": args.displacement_coef,
            "gradient_cap": args.gradient_cap,
            "optimizer_weight_decay": 0.0,
            "seed": args.seed,
            "threads": args.threads,
            "thresholds": list(THRESHOLDS),
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "permitted_prefixes": list(PERMITTED_PREFIXES),
        "n6_provenance": n6_provenance,
        "n9_provenance": n9_provenance,
        "reference_parity": {"n6": n6_reference, "n9": n9_reference},
        "history": history,
        "evaluations": evaluations,
        "admitted": admitted,
        "elapsed_seconds": monotonic() - started,
        "claims": {
            "development_panel_authority": bool(admitted),
            "deployment_authority": False,
            "strength_improvement": False,
        },
    }
    report_path = output / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "admitted": admitted}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
