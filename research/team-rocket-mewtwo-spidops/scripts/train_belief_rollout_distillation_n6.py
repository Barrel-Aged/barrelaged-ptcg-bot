#!/usr/bin/env python3
"""Distill N6 public-belief rollout means into the frozen neural blueprint."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from time import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as Fnn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rocketbot.n6_teacher import estimate_root, redistribution_coefficient  # noqa: E402
from rocketbot.policy_feature_snapshot import deserialize_policy_feature  # noqa: E402
from train_public_leader_v110_bc import Example, collate, configure_trainable  # noqa: E402


DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


@dataclass
class TeacherExample(Example):
    reference_probabilities: np.ndarray
    target_probabilities: np.ndarray
    baseline: int
    teacher_best: int
    admissible_change: bool
    confidence: float
    opponent: str


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
            / "scale_value_calibrated_v1"
        ),
    )
    parser.add_argument("--train-input", type=Path, action="append", required=True)
    parser.add_argument("--selection-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-temperature", type=float, required=True)
    parser.add_argument("--kl-coef", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=790301)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = []
    hashes = {}
    seen = set()
    for raw_path in paths:
        path = raw_path.resolve()
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") not in {
            "rocket-n6-belief-rollout-manifest/v1",
            "rocket-n6r-belief-rollout-manifest/v2",
        }:
            raise ValueError(f"{manifest_path}: unexpected N6 manifest schema")
        if not manifest.get("genuine_official_engine_descendant_transitions"):
            raise ValueError(f"{manifest_path}: descendant transitions are not official")
        if manifest.get("visualizer_or_realized_hidden_truth_read"):
            raise ValueError(f"{manifest_path}: realized hidden truth was read")
        if manifest.get("raw_hidden_allocations_serialized"):
            raise ValueError(f"{manifest_path}: raw hidden allocations were serialized")
        path_hash = sha256(path)
        if manifest.get("output_sha256") != path_hash:
            raise ValueError(f"{path}: content hash differs from its N6 manifest")
        hashes[str(path)] = path_hash
        hashes[str(manifest_path)] = sha256(manifest_path)
        file_rows = 0
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("schema") not in {
                "rocket-n6-belief-rollout-root/v1",
                "rocket-n6r-belief-rollout-root/v2",
            }:
                raise ValueError(f"{path}:{line_number}: unexpected N6 schema")
            if row.get("schema") == "rocket-n6r-belief-rollout-root/v2":
                if not row.get("policy_feature_baseline_reproduced"):
                    raise ValueError(
                        f"{path}:{line_number}: N6R baseline feature parity is absent"
                    )
                deserialize_policy_feature(row.get("policy_feature") or {})
            if not row.get("fully_complete"):
                raise ValueError(f"{path}:{line_number}: incomplete N6 root")
            if int(row.get("complete_pairs", -1)) != int(row.get("attempted_pairs", -2)):
                raise ValueError(f"{path}:{line_number}: incomplete paired rollouts")
            if row.get("errors"):
                raise ValueError(f"{path}:{line_number}: N6 root contains rollout errors")
            key = (str(path), int(row["game_id"]), int(row["root_index"]))
            if key in seen:
                raise ValueError(f"duplicate N6 root: {key}")
            seen.add(key)
            rows.append(row)
            file_rows += 1
        if file_rows != int((manifest.get("counts") or {}).get("roots", -1)):
            raise ValueError(f"{path}: row count differs from its N6 manifest")
    if not rows:
        raise ValueError("N6 input contains no roots")
    return rows, hashes


def apply_attack_guard(
    logits: torch.Tensor,
    observation: dict[str, Any],
) -> torch.Tensor:
    """Apply the deployment agent's attack/end legality guard to logits.

    The live actor masks END *after* the frozen network has evaluated the
    unmodified feature tensor.  Mutating ``opt_avail`` before inference is not
    equivalent: availability is itself a learned option feature and can move
    every action logit.  Keep the saved N6R feature byte-for-byte intact and
    reproduce the deployed post-logit operation instead.
    """

    options = (observation.get("select") or {}).get("option") or ()
    if not any(option.get("type") == 13 for option in options):
        return logits
    end_indices = [index for index, option in enumerate(options) if option.get("type") == 14]
    if not end_indices:
        return logits
    guarded = logits.clone()
    guarded[end_indices] = -torch.inf
    return guarded


def _softmax(values: Sequence[float], temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / temperature
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    return probabilities.astype(np.float32)


@torch.no_grad()
def encode_rows(
    rows: Sequence[dict[str, Any]],
    *,
    model: torch.nn.Module,
    deck: list[int],
    library: Any,
    belief_class: Any,
    features_module: Any,
    teacher_temperature: float,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[list[TeacherExample], dict[str, int]]:
    examples = []
    counts: Counter[str] = Counter()
    model.eval()
    for row_index, row in enumerate(rows):
        try:
            observation = row["observation"]
            if row.get("schema") == "rocket-n6r-belief-rollout-root/v2":
                feature = deserialize_policy_feature(row["policy_feature"])
            else:
                belief = belief_class(deck, opponent_deck_priors=library)
                feature = features_module.extract(
                    observation, belief=belief, slot_link=True
                )
            option_count = int(feature["opt_feats"].shape[0])
            reference_batch = collate(
                [Example(feature, int(row["baseline"]), f"encode={row_index}")],
                torch.device("cpu"),
            )
            logits, _ = model.forward_batch(reference_batch)
            logits = logits[0, :option_count]
            logits = apply_attack_guard(logits, observation)
            reference = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32)
            baseline = int(row["baseline"])
            reference_choice = int(logits.argmax())
            if reference_choice != baseline:
                counts["baseline_reproduction_mismatch"] += 1
                continue
            candidates = [int(value) for value in row["candidate_indices"]]
            if any(value < 0 or value >= reference.shape[0] for value in candidates):
                raise ValueError("candidate index exceeds option support")
            estimate = estimate_root(
                row,
                seed=seed + 100_003 * row_index,
                bootstrap_replicates=bootstrap_replicates,
            )
            means = [estimate.action(action).mean for action in candidates]
            confidence = redistribution_coefficient(means, estimate.complete_clusters)
            candidate_mass = float(reference[candidates].sum())
            redistributed = reference.copy()
            redistributed[candidates] = candidate_mass * _softmax(
                means,
                teacher_temperature,
            )
            target = (1.0 - confidence) * reference + confidence * redistributed
            target /= target.sum()
            examples.append(
                TeacherExample(
                    feat=feature,
                    action=estimate.best if estimate.admissible_change else baseline,
                    game=f"{row['opponent']}:game={row['game_id']}",
                    reference_probabilities=reference,
                    target_probabilities=target.astype(np.float32),
                    baseline=baseline,
                    teacher_best=estimate.best,
                    admissible_change=estimate.admissible_change,
                    confidence=confidence,
                    opponent=str(row["opponent"]),
                )
            )
            counts["encoded"] += 1
            counts["admissible_changes"] += int(estimate.admissible_change)
            counts["positive_confidence"] += int(confidence > 0.0)
        except Exception as error:
            counts[f"encode_error:{type(error).__name__}"] += 1
    if not examples:
        raise ValueError("N6 feature encoding produced no examples")
    return examples, dict(counts)


def pad_probabilities(
    examples: Sequence[TeacherExample],
    width: int,
    *,
    attribute: str,
    device: torch.device,
) -> torch.Tensor:
    array = np.zeros((len(examples), width), dtype=np.float32)
    for row, example in enumerate(examples):
        values = getattr(example, attribute)
        array[row, : len(values)] = values
    return torch.as_tensor(array, dtype=torch.float32, device=device)


@torch.no_grad()
def metrics(
    model: torch.nn.Module,
    examples: Sequence[TeacherExample],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals: Counter[str] = Counter()
    teacher_ce = 0.0
    reference_kl = 0.0
    confidence = 0.0
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start : start + batch_size])
        batch = collate(selected, device)
        logits, _ = model.forward_batch(batch)
        logits = logits.masked_fill(batch["opt_mask"] < 0.5, -1.0e9)
        logp = Fnn.log_softmax(logits, dim=1)
        target = pad_probabilities(
            selected,
            logits.shape[1],
            attribute="target_probabilities",
            device=device,
        )
        reference = pad_probabilities(
            selected,
            logits.shape[1],
            attribute="reference_probabilities",
            device=device,
        )
        teacher_ce += float((-(target * logp).sum(dim=1)).sum())
        reference_log = torch.log(reference.clamp(min=1.0e-12))
        reference_kl += float((reference * (reference_log - logp)).sum())
        predictions = logits.argmax(dim=1).tolist()
        for example, prediction in zip(selected, predictions):
            totals["count"] += 1
            totals["baseline_top1"] += int(prediction == example.baseline)
            totals["teacher_top1"] += int(prediction == example.teacher_best)
            if example.admissible_change:
                totals["admissible_count"] += 1
                totals["admissible_top1"] += int(prediction == example.teacher_best)
            confidence += example.confidence
    count = max(1, totals["count"])
    return {
        "examples": totals["count"],
        "teacher_cross_entropy": teacher_ce / count,
        "reference_kl": reference_kl / count,
        "baseline_top1": totals["baseline_top1"] / count,
        "teacher_top1": totals["teacher_top1"] / count,
        "admissible_examples": totals["admissible_count"],
        "admissible_top1": totals["admissible_top1"] / max(1, totals["admissible_count"]),
        "mean_confidence": confidence / count,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.teacher_temperature <= 0.0 or args.kl_coef < 0.0:
        raise ValueError("temperature must be positive and KL coefficient non-negative")
    for name in ("epochs", "batch_size", "threads", "bootstrap_replicates"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N6 training output: {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    candidate = args.candidate.resolve()
    if sha256(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("N6 frozen deck changed")
    if sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("N6 frozen model changed")
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy, save_policy  # type: ignore

    deck = [
        int(value)
        for value in (candidate / "deck.csv").read_text(encoding="utf-8").replace(",", "\n").split()
    ]
    library_path = candidate / "deck_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else {}
    train_rows, train_hashes = read_rows(args.train_input)
    selection_rows, selection_hashes = read_rows(args.selection_input)
    input_schemas = {
        str(row["schema"]) for row in [*train_rows, *selection_rows]
    }
    if len(input_schemas) != 1:
        raise ValueError(f"N6/N6R input schemas are mixed: {sorted(input_schemas)}")
    input_schema = next(iter(input_schemas))
    if input_schema == "rocket-n6r-belief-rollout-root/v2":
        if args.threads != 1:
            raise ValueError(
                "N6R exact-feature distillation requires --threads 1 to match "
                "the frozen collector inference path"
            )
        report_schema = "rocket-n6r-belief-rollout-distillation/v2"
        protocol = "N6R_DISTILLATION_PARITY_REPAIR.md"
    else:
        report_schema = "rocket-n6-belief-rollout-distillation/v1"
        protocol = "N6_BELIEF_ROLLOUT_DISTILLATION_PROTOCOL.md"
    reference = load_policy(candidate / "model.pt", map_location=device).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    train_examples, train_encoding = encode_rows(
        train_rows,
        model=reference,
        deck=deck,
        library=library,
        belief_class=Belief,
        features_module=features,
        teacher_temperature=args.teacher_temperature,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    selection_examples, selection_encoding = encode_rows(
        selection_rows,
        model=reference,
        deck=deck,
        library=library,
        belief_class=Belief,
        features_module=features,
        teacher_temperature=args.teacher_temperature,
        seed=args.seed + 1_000_003,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    if len(train_examples) != len(train_rows) or train_encoding.get("encoded") != len(train_rows):
        raise ValueError(
            "N6 training encoding was not lossless: "
            f"rows={len(train_rows)} examples={len(train_examples)} counts={train_encoding}"
        )
    if (
        len(selection_examples) != len(selection_rows)
        or selection_encoding.get("encoded") != len(selection_rows)
    ):
        raise ValueError(
            "N6 selection encoding was not lossless: "
            f"rows={len(selection_rows)} examples={len(selection_examples)} "
            f"counts={selection_encoding}"
        )
    train_fingerprints = {str(row["observation_fingerprint"]) for row in train_rows}
    selection_fingerprints = {str(row["observation_fingerprint"]) for row in selection_rows}
    overlap = train_fingerprints & selection_fingerprints
    if overlap:
        raise ValueError(f"N6 train/selection observation overlap: {len(overlap)} roots")
    model = copy.deepcopy(reference)
    trainable = configure_trainable(model, "last")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    base_metrics = {
        "train": metrics(reference, train_examples, batch_size=args.batch_size, device=device),
        "selection": metrics(reference, selection_examples, batch_size=args.batch_size, device=device),
    }
    history = []
    order = list(range(len(train_examples)))
    started = time()
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        loss_total = teacher_total = kl_total = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            selected = [train_examples[index] for index in order[start : start + args.batch_size]]
            batch = collate(selected, device)
            logits, _ = model.forward_batch(batch)
            logits = logits.masked_fill(batch["opt_mask"] < 0.5, -1.0e9)
            logp = Fnn.log_softmax(logits, dim=1)
            target = pad_probabilities(
                selected,
                logits.shape[1],
                attribute="target_probabilities",
                device=device,
            )
            reference_probabilities = pad_probabilities(
                selected,
                logits.shape[1],
                attribute="reference_probabilities",
                device=device,
            )
            teacher_loss = (
                -((target - reference_probabilities) * logp).sum(dim=1)
            ).mean()
            reference_log = torch.log(reference_probabilities.clamp(min=1.0e-12))
            raw_kl = (
                reference_probabilities * (reference_log - logp)
            ).sum(dim=1).mean()
            kl = torch.clamp(raw_kl - 1.0e-4, min=0.0)
            loss = teacher_loss + args.kl_coef * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if float(gradient_norm) > 1.0e-10:
                optimizer.step()
            loss_total += float(loss.detach())
            teacher_total += float(teacher_loss.detach())
            kl_total += float(raw_kl.detach())
            batches += 1
        record = {
            "epoch": epoch,
            "train_loss": loss_total / max(1, batches),
            "train_teacher_loss": teacher_total / max(1, batches),
            "train_reference_kl": kl_total / max(1, batches),
            "metrics": {
                "train": metrics(model, train_examples, batch_size=args.batch_size, device=device),
                "selection": metrics(model, selection_examples, batch_size=args.batch_size, device=device),
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "model.pt"
    save_policy(model, model_path)
    report = {
        "schema": report_schema,
        "protocol": protocol,
        "input_root_schema": input_schema,
        "initial_model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "train_inputs": train_hashes,
        "selection_inputs": selection_hashes,
        "hyperparameters": {
            "teacher_temperature": args.teacher_temperature,
            "kl_coefficient": args.kl_coef,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "threads": args.threads,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "trainable_scope": "final relational transformer block and action head",
        "objective": {
            "policy": "CE(teacher, student) - CE(reference, student)",
            "reference_kl_dead_zone": 1.0e-4,
            "deployment_guard": "post-logit END mask when ATTACK is legal",
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "encoding": {
            "train": train_encoding,
            "selection": selection_encoding,
        },
        "base_metrics": base_metrics,
        "history": history,
        "checkpoint_selection": "final epoch only",
        "artifact": str(model_path),
        "artifact_sha256": sha256(model_path),
        "official_engine_execution_in_training_loop": False,
        "teacher_source_official_descendants": True,
        "search_packaged_in_student": False,
        "deployment_authority": False,
        "elapsed_seconds": time() - started,
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "complete": True,
                "artifact": str(model_path),
                "sha256": report["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    train(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
