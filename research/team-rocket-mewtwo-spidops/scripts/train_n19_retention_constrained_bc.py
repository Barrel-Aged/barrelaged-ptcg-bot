#!/usr/bin/env python3
"""Run the sole preregistered N19 retention-constrained BC training job."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
from time import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "work/n18/blive"
AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n19_retention_teacher/"
    "N19_CORPUS_AUDIT.json"
)
AUDITOR = ROOT / "scripts/audit_n19_retention_teacher_corpus.py"
PROTOCOL = ROOT / "N19_RETENTION_CONSTRAINED_LEADER_BC_PROTOCOL.md"
OUTPUT = (
    ROOT
    / "artifacts/team_rocket_recovery/n19_retention_teacher/"
    "training_v1"
)

AUDIT_SHA = "d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed"
AUDITOR_SHA = "86f6e12bb96c16ebfa0980e27974bff881537cb8e750e9c5adc6a688bfda58c1"
PROTOCOL_SHA = "006fd34be9d3770ed8025c2db2a5c0f21c6aa5f2e4dbd4baa9cde50ad37e153e"
LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"

SEED = 819001
EPOCHS = 4
BATCH_SIZE = 96
LEARNING_RATE = 2.0e-6
WEIGHT_DECAY = 1.0e-4
LABEL_SMOOTHING = 0.01
RETENTION_KL_COEFFICIENT = 4.0
TEACHER_KL_COEFFICIENT = 0.25
SOURCE_ANCHOR_COEFFICIENT = 0.01
THREADS = 6
TRAINABLE_PARAMETERS = 1_184_257


sys.path.insert(0, str((ROOT / "scripts").resolve()))
from train_public_leader_v110_bc import (  # noqa: E402
    Example,
    collate,
    configure_trainable,
    evaluate,
    masked_cross_entropy,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_game(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["source_directory"]) / record["file"]
    if digest(path) != record["payload_sha256"]:
        raise ValueError(f"N19 selected replay changed: {path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    meta = rows[0]["meta"]
    result = next(row["result"] for row in rows if "result" in row)
    winner_raw = result.get("winner")
    winner = int(winner_raw) if winner_raw in (0, 1) else -1
    return {
        "payload_sha256": record["payload_sha256"],
        "path": path,
        "rows": rows,
        "players": list(meta["players"]),
        "decks": [[int(card) for card in deck] for deck in meta["decks"]],
        "winner": winner,
    }


@torch.no_grad()
def cache_base_probabilities(
    model: torch.nn.Module,
    examples: list[Example],
    device: torch.device,
) -> list[np.ndarray]:
    model.eval()
    result: list[np.ndarray] = []
    for start in range(0, len(examples), BATCH_SIZE):
        selected = examples[start : start + BATCH_SIZE]
        batch = collate(selected, device)
        logits, _ = model.forward_batch(batch)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        for row, example in enumerate(selected):
            option_count = int(example.feat["opt_feats"].shape[0])
            result.append(probabilities[row, :option_count].copy())
    if len(result) != len(examples):
        raise AssertionError("N19 base-probability cache changed length")
    return result


def padded_probabilities(
    probabilities: list[np.ndarray],
    indexes: list[int],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    values = np.zeros((len(indexes), width), dtype=np.float32)
    for row, index in enumerate(indexes):
        current = probabilities[index]
        values[row, : len(current)] = current
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def distribution_kl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_candidate = torch.log_softmax(logits, dim=1)
    target_log = torch.where(target > 0.0, torch.log(target.clamp_min(1.0e-30)), torch.zeros_like(target))
    terms = torch.where(
        target > 0.0,
        target * (target_log - log_candidate),
        torch.zeros_like(target),
    )
    return terms.sum(dim=1).mean()


@torch.no_grad()
def policy_retention(
    model: torch.nn.Module,
    examples: list[Example],
    base_probabilities: list[np.ndarray],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    retained = 0
    total = 0
    kl_total = 0.0
    for start in range(0, len(examples), BATCH_SIZE):
        selected = examples[start : start + BATCH_SIZE]
        indexes = list(range(start, start + len(selected)))
        batch = collate(selected, device)
        logits, _ = model.forward_batch(batch)
        target = padded_probabilities(
            base_probabilities, indexes, logits.shape[1], device
        )
        retained += int((logits.argmax(dim=1) == target.argmax(dim=1)).sum())
        kl_total += float(distribution_kl(logits, target)) * len(selected)
        total += len(selected)
    return {
        "examples": total,
        "top1_retention": retained / max(total, 1),
        "mean_kl": kl_total / max(total, 1),
    }


def state_comparison(
    source: torch.nn.Module, candidate: torch.nn.Module
) -> dict[str, Any]:
    source_state = source.state_dict()
    candidate_state = candidate.state_dict()
    if source_state.keys() != candidate_state.keys():
        raise ValueError("N19 candidate state keys changed")
    changed = []
    exact = []
    shape_errors = []
    nonfinite = []
    for key in source_state:
        left = source_state[key]
        right = candidate_state[key]
        if left.shape != right.shape:
            shape_errors.append(key)
            continue
        if not bool(torch.isfinite(right).all()):
            nonfinite.append(key)
        if torch.equal(left, right):
            exact.append(key)
        else:
            changed.append(key)
    return {
        "state_keys": len(source_state),
        "exact_tensors": exact,
        "changed_tensors": changed,
        "shape_errors": shape_errors,
        "nonfinite_tensors": nonfinite,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N19 training output: {OUTPUT}")
    for path, expected in (
        (AUDIT, AUDIT_SHA),
        (AUDITOR, AUDITOR_SHA),
        (PROTOCOL, PROTOCOL_SHA),
        (LIVE / "model.pt", LIVE_MODEL_SHA),
        (LIVE / "deck.csv", DECK_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"N19 frozen training input changed: {path}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    device = torch.device("cpu")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != "rocket-n19-retention-teacher-corpus-audit/v1"
        or audit.get("training_performed") is not False
        or audit.get("all_recorded_actions_legal") is not True
    ):
        raise ValueError("N19 corpus audit authority changed")

    sys.path.insert(0, str(LIVE.resolve()))
    from soldier import features  # type: ignore  # noqa: E402
    from soldier.belief import Belief  # type: ignore  # noqa: E402
    from soldier.net import load_policy, save_policy  # type: ignore  # noqa: E402
    from audit_n19_retention_teacher_corpus import extract_examples  # noqa: E402

    deck_library = json.loads((LIVE / "deck_library.json").read_text(encoding="utf-8"))
    selected_games: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {split: [] for split in ("train", "validation", "test")}
        for role in ("teacher", "retention")
    }
    for record in audit["game_manifest"]:
        selected_games[record["role"]][record["split"]].append(load_game(record))

    data: dict[str, dict[str, list[Example]]] = {
        role: {} for role in ("teacher", "retention")
    }
    extraction: dict[str, dict[str, Any]] = {
        role: {} for role in ("teacher", "retention")
    }
    for role in ("teacher", "retention"):
        for split in ("train", "validation", "test"):
            examples, counts = extract_examples(
                selected_games[role][split], role, deck_library, Belief, features
            )
            if dict(counts) != audit["extraction"][role][split]:
                raise ValueError(f"N19 extraction changed: {role}/{split}")
            data[role][split] = examples
            extraction[role][split] = dict(counts)

    source_model = load_policy(LIVE / "model.pt", map_location=device).cpu().eval()
    base_probabilities = {
        role: {
            split: cache_base_probabilities(source_model, data[role][split], device)
            for split in ("train", "validation", "test")
        }
        for role in ("teacher", "retention")
    }
    model = copy.deepcopy(source_model).train()
    trainable = configure_trainable(model, "last")
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != TRAINABLE_PARAMETERS:
        raise ValueError(f"N19 trainable parameter count changed: {trainable_count}")
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    source_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    teacher_train = data["teacher"]["train"]
    retention_train = data["retention"]["train"]
    teacher_base = base_probabilities["teacher"]["train"]
    retention_base = base_probabilities["retention"]["train"]
    steps_per_epoch = math.ceil(len(teacher_train) / BATCH_SIZE)
    history: list[dict[str, Any]] = []
    total_steps = 0
    started = time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        teacher_order = list(range(len(teacher_train)))
        retention_order = list(range(len(retention_train)))
        random.Random(SEED + epoch).shuffle(teacher_order)
        random.Random(SEED + 100 + epoch).shuffle(retention_order)
        sums = {
            "loss": 0.0,
            "teacher_cross_entropy": 0.0,
            "teacher_kl": 0.0,
            "retention_kl": 0.0,
            "source_anchor": 0.0,
        }
        seen = 0
        for step in range(steps_per_epoch):
            teacher_indexes = teacher_order[
                step * BATCH_SIZE : min((step + 1) * BATCH_SIZE, len(teacher_order))
            ]
            batch_count = len(teacher_indexes)
            retention_indexes = [
                retention_order[(step * BATCH_SIZE + offset) % len(retention_order)]
                for offset in range(batch_count)
            ]
            teacher_examples = [teacher_train[index] for index in teacher_indexes]
            retention_examples = [retention_train[index] for index in retention_indexes]
            teacher_batch = collate(teacher_examples, device)
            retention_batch = collate(retention_examples, device)
            teacher_labels = torch.as_tensor(
                [example.action for example in teacher_examples],
                dtype=torch.long,
                device=device,
            )
            teacher_logits, _ = model.forward_batch(teacher_batch)
            retention_logits, _ = model.forward_batch(retention_batch)
            teacher_target = padded_probabilities(
                teacher_base, teacher_indexes, teacher_logits.shape[1], device
            )
            retention_target = padded_probabilities(
                retention_base, retention_indexes, retention_logits.shape[1], device
            )
            teacher_cross_entropy = masked_cross_entropy(
                teacher_logits,
                teacher_labels,
                teacher_batch["opt_mask"],
                LABEL_SMOOTHING,
            )
            teacher_kl = distribution_kl(teacher_logits, teacher_target)
            retention_kl = distribution_kl(retention_logits, retention_target)
            source_anchor = torch.stack(
                [
                    (parameter - source_parameters[name]).square().mean()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                ]
            ).mean()
            loss = (
                teacher_cross_entropy
                + TEACHER_KL_COEFFICIENT * teacher_kl
                + RETENTION_KL_COEFFICIENT * retention_kl
                + SOURCE_ANCHOR_COEFFICIENT * source_anchor
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("N19 training produced a nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            values = {
                "loss": loss,
                "teacher_cross_entropy": teacher_cross_entropy,
                "teacher_kl": teacher_kl,
                "retention_kl": retention_kl,
                "source_anchor": source_anchor,
            }
            for key, value in values.items():
                sums[key] += float(value.detach()) * batch_count
            seen += batch_count
            total_steps += 1

        teacher_validation = evaluate(
            model, data["teacher"]["validation"], BATCH_SIZE, device
        )
        retention_validation = policy_retention(
            model,
            data["retention"]["validation"],
            base_probabilities["retention"]["validation"],
            device,
        )
        row = {
            "epoch": epoch,
            "steps": total_steps,
            **{key: value / max(seen, 1) for key, value in sums.items()},
            "teacher_validation": teacher_validation,
            "retention_validation": retention_validation,
            "elapsed_seconds": time() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    terminal_metrics = {
        role: {
            split: evaluate(model, data[role][split], BATCH_SIZE, device)
            for split in ("train", "validation", "test")
        }
        for role in ("teacher", "retention")
    }
    terminal_policy_retention = {
        split: policy_retention(
            model,
            data["retention"][split],
            base_probabilities["retention"][split],
            device,
        )
        for split in ("train", "validation", "test")
    }
    comparison = state_comparison(source_model, model)
    if set(comparison["changed_tensors"]) != set(trainable_names):
        raise ValueError("N19 changed tensor set differs from preregistration")
    if comparison["shape_errors"] or comparison["nonfinite_tensors"]:
        raise ValueError("N19 terminal state failed shape/finite checks")

    baseline = audit["baseline_metrics"]
    gates = {
        "serialization_source_valid": True,
        "only_14_allowed_tensors_changed": (
            len(comparison["changed_tensors"]) == 14
            and len(comparison["exact_tensors"]) == 45
        ),
        "teacher_validation_top1_plus_2pp": (
            terminal_metrics["teacher"]["validation"]["top1"]
            >= baseline["teacher"]["validation"]["live"]["top1"] + 0.02
        ),
        "teacher_validation_loss_lower": (
            terminal_metrics["teacher"]["validation"]["loss"]
            < baseline["teacher"]["validation"]["live"]["loss"]
        ),
        "teacher_test_top1_plus_1pp": (
            terminal_metrics["teacher"]["test"]["top1"]
            >= baseline["teacher"]["test"]["live"]["top1"] + 0.01
        ),
        "teacher_test_loss_lower": (
            terminal_metrics["teacher"]["test"]["loss"]
            < baseline["teacher"]["test"]["live"]["loss"]
        ),
        "fresh_retention_each_split_top1_ge_0_995": all(
            row["top1_retention"] >= 0.995
            for row in terminal_policy_retention.values()
        ),
        "fresh_retention_each_split_kl_le_0_01": all(
            row["mean_kl"] <= 0.01 for row in terminal_policy_retention.values()
        ),
        "recorded_retention_validation_within_half_pp": (
            terminal_metrics["retention"]["validation"]["top1"]
            >= baseline["retention"]["validation"]["live"]["top1"] - 0.005
        ),
        "recorded_retention_test_within_half_pp": (
            terminal_metrics["retention"]["test"]["top1"]
            >= baseline["retention"]["test"]["live"]["top1"] - 0.005
        ),
        "finite_terminal_model": not comparison["nonfinite_tensors"],
    }
    corpus_gates_pass = all(gates.values())

    copy_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
    shutil.copytree(LIVE, OUTPUT, ignore=copy_ignore)
    save_policy(model.eval(), OUTPUT / "model.pt")
    reloaded = load_policy(OUTPUT / "model.pt", map_location=device).cpu().eval()
    reloaded_comparison = state_comparison(model, reloaded)
    if reloaded_comparison["changed_tensors"] or reloaded_comparison["shape_errors"]:
        raise ValueError("N19 serialized terminal model changed on reload")
    terminal_sha = digest(OUTPUT / "model.pt")
    report = {
        "schema": "rocket-n19-retention-constrained-training/v1",
        "protocol": "rocket-n19-retention-constrained-leader-bc/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "corpus_audit_sha256": AUDIT_SHA,
        "corpus_auditor_sha256": AUDITOR_SHA,
        "source_model_sha256": LIVE_MODEL_SHA,
        "terminal_model_sha256": terminal_sha,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "hyperparameters": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size_per_objective": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "retention_kl_coefficient": RETENTION_KL_COEFFICIENT,
            "teacher_kl_coefficient": TEACHER_KL_COEFFICIENT,
            "source_anchor_coefficient": SOURCE_ANCHOR_COEFFICIENT,
            "gradient_norm_cap": 1.0,
            "threads": THREADS,
            "checkpoint_rule": "terminal_epoch_4_only",
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
        },
        "trainable_parameters": trainable_count,
        "trainable_tensors": trainable_names,
        "history": history,
        "baseline_metrics": baseline,
        "terminal_metrics": terminal_metrics,
        "terminal_policy_retention": terminal_policy_retention,
        "state_comparison": comparison,
        "corpus_gates": gates,
        "corpus_gates_pass": corpus_gates_pass,
        "n6_n9_anchor_audit_pending": True,
        "complete_offline_admission": False,
        "official_match_evidence": False,
        "development_panel_authority": False,
        "deployment_authority": False,
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "Terminal supervised training and offline corpus gates only. N6/N9 anchor audit is "
            "still required, and no official-match or deployment claim is authorized."
        ),
    }
    (OUTPUT / "N19_TRAINING_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    provenance = {
        "schema": "rocket-n19-training-provenance/v1",
        "training_script_sha256": digest(Path(__file__).resolve()),
        "training_report_sha256": digest(OUTPUT / "N19_TRAINING_REPORT.json"),
        "protocol_sha256": PROTOCOL_SHA,
        "corpus_audit_sha256": AUDIT_SHA,
        "source_model_sha256": LIVE_MODEL_SHA,
        "terminal_model_sha256": terminal_sha,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "corpus_gates_pass": corpus_gates_pass,
        "n6_n9_anchor_audit_pending": True,
        "development_panel_authority": False,
        "deployment_authority": False,
    }
    (OUTPUT / "N19_TRAINING_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "terminal_model_sha256": terminal_sha,
                "training_report_sha256": digest(OUTPUT / "N19_TRAINING_REPORT.json"),
                "corpus_gates_pass": corpus_gates_pass,
                "teacher_validation_top1": terminal_metrics["teacher"]["validation"]["top1"],
                "teacher_test_top1": terminal_metrics["teacher"]["test"]["top1"],
                "retention": terminal_policy_retention,
                "development_panel_authority": False,
                "deployment_authority": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
