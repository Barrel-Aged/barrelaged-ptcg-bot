#!/usr/bin/env python3
"""Run the sole N21 top-action-margin constrained BC training job."""

from __future__ import annotations

import copy
from collections import Counter
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
CORPUS_AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n19_retention_teacher/"
    "N19_CORPUS_AUDIT.json"
)
N19_RESULT = ROOT / "N19_RETENTION_CONSTRAINED_LEADER_BC_RESULT.md"
N20_REPORT = (
    ROOT
    / "artifacts/team_rocket_recovery/n20_trust_region_projection/"
    "projection_v1/N20_PROJECTION_REPORT.json"
)
N20_RESULT = ROOT / "N20_TRUST_REGION_PROJECTION_RESULT.md"
N17_TARGET_AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n17_replicated_targets/target_audit.json"
)
PROTOCOL = ROOT / "N21_TOP_ACTION_MARGIN_BC_PROTOCOL.md"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n21_top_action_margin/training_v1"

LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CORPUS_AUDIT_SHA = "d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed"
N19_RESULT_SHA = "cfb2560fca269d969a9490453db28b7f7a3bf6598e96864cfd886ecc1962f227"
N20_REPORT_SHA = "83f8094ecfacbeb6c73caaf899af2fb7cbbfc9779ce5f4ec9d77bb23218319e0"
N20_RESULT_SHA = "84fa85599e38cb50dc03265e8d090666467993ee34681dda62b46f82861c1d8c"
N17_TARGET_AUDIT_SHA = "c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c"
PROTOCOL_SHA = "2a18a5f8ad3242e5d4c13ff132187121563b20b27f384597b91dc25d607cf839"

N19_TRAINER = ROOT / "scripts/train_n19_retention_constrained_bc.py"
N20_EVALUATOR = ROOT / "scripts/evaluate_n20_trust_region_projection.py"
CORPUS_AUDITOR = ROOT / "scripts/audit_n19_retention_teacher_corpus.py"
PUBLIC_BC_SUPPORT = ROOT / "scripts/train_public_leader_v110_bc.py"
N12_SUPPORT = ROOT / "scripts/train_n12_counterfactual_margin_adapter.py"
LIVE_AWR_SUPPORT = ROOT / "scripts/train_live_outcome_awr.py"
N9_SUPPORT = ROOT / "scripts/train_n9_balanced_pcgrad_awr.py"
N6_SUPPORT = ROOT / "scripts/train_belief_rollout_distillation_n6.py"
N19_TRAINER_SHA = "3928abaf17457d05e02c7d9734f69f819705601051154bc78236a981868de7b7"
N20_EVALUATOR_SHA = "d612058c08f6c2dfbffd0f444300d95bd57e5d1d82c4718c881eab79ed0ac6f1"
CORPUS_AUDITOR_SHA = "86f6e12bb96c16ebfa0980e27974bff881537cb8e750e9c5adc6a688bfda58c1"
PUBLIC_BC_SUPPORT_SHA = "3dcd9bef8dcc1cf17bd0fb2360ef1e4d4d90ab8a9ec715d7ed4108515fb35a34"
N12_SUPPORT_SHA = "843c12e0d2a1684d53ed27423c6b6c5cfe2dc38109c57b089b3d98e38cf181bd"
LIVE_AWR_SUPPORT_SHA = "4f233d40e625c33711ab87438388f8ba68f65f2e103c563bf621c790c9a657e7"
N9_SUPPORT_SHA = "62182583002847e7ed0e18011050f35319ad5661b73f3fb2d83c2564286ed1ad"
N6_SUPPORT_SHA = "3c559b00c4d59c018295406e84dc78f2e19b5014b9174cb23c8db63790020a7f"

SEED = 821001
EPOCHS = 2
BATCH_SIZE = 96
LEARNING_RATE = 2.0e-6
WEIGHT_DECAY = 1.0e-4
LABEL_SMOOTHING = 0.01
TEACHER_KL_COEF = 0.25
FRESH_MARGIN_COEF = 4.0
FRESH_KL_COEF = 1.0
N6_MARGIN_COEF = 1.0
N6_KL_COEF = 0.25
N9_MARGIN_COEF = 1.0
N9_KL_COEF = 0.25
SOURCE_ANCHOR_COEF = 0.01
MARGIN_CAP = 0.25
MARGIN_TEMPERATURE = 0.10
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
from train_n19_retention_constrained_bc import (  # noqa: E402
    cache_base_probabilities,
    load_game,
    padded_probabilities,
    state_comparison,
)
from evaluate_n20_trust_region_projection import (  # noqa: E402
    combined_retention_metrics,
    full_preservation_metrics,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution_kl(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_candidate = torch.log_softmax(logits, dim=1)
    target_log = torch.where(
        target > 0.0,
        torch.log(target.clamp_min(1.0e-30)),
        torch.zeros_like(target),
    )
    return torch.where(
        target > 0.0,
        target * (target_log - log_candidate),
        torch.zeros_like(target),
    ).sum(dim=1).mean()


def probability_margin_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    winners = target.argmax(dim=1)
    winner_logits = logits.gather(1, winners[:, None]).squeeze(1)
    alternatives = logits.clone()
    alternatives.scatter_(1, winners[:, None], -1.0e9)
    candidate_margin = winner_logits - alternatives.max(dim=1).values

    target_log = torch.log(target.clamp_min(1.0e-30))
    target_winner = target_log.gather(1, winners[:, None]).squeeze(1)
    target_alternatives = target_log.clone()
    target_alternatives.scatter_(1, winners[:, None], -1.0e9)
    base_margin = target_winner - target_alternatives.max(dim=1).values
    protected = base_margin.clamp(min=0.0, max=MARGIN_CAP)
    return (
        F.softplus((protected - candidate_margin) / MARGIN_TEMPERATURE)
        * MARGIN_TEMPERATURE
    ).mean()


def cycle_indexes(order: list[int], step: int, count: int) -> list[int]:
    return [order[(step * BATCH_SIZE + offset) % len(order)] for offset in range(count)]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N21 output: {OUTPUT}")
    for path, expected in (
        (LIVE / "model.pt", LIVE_MODEL_SHA),
        (LIVE / "deck.csv", DECK_SHA),
        (CORPUS_AUDIT, CORPUS_AUDIT_SHA),
        (N19_RESULT, N19_RESULT_SHA),
        (N20_REPORT, N20_REPORT_SHA),
        (N20_RESULT, N20_RESULT_SHA),
        (N17_TARGET_AUDIT, N17_TARGET_AUDIT_SHA),
        (PROTOCOL, PROTOCOL_SHA),
        (N19_TRAINER, N19_TRAINER_SHA),
        (N20_EVALUATOR, N20_EVALUATOR_SHA),
        (CORPUS_AUDITOR, CORPUS_AUDITOR_SHA),
        (PUBLIC_BC_SUPPORT, PUBLIC_BC_SUPPORT_SHA),
        (N12_SUPPORT, N12_SUPPORT_SHA),
        (LIVE_AWR_SUPPORT, LIVE_AWR_SUPPORT_SHA),
        (N9_SUPPORT, N9_SUPPORT_SHA),
        (N6_SUPPORT, N6_SUPPORT_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"N21 frozen input changed: {path}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    device = torch.device("cpu")
    audit = json.loads(CORPUS_AUDIT.read_text(encoding="utf-8"))
    n20 = json.loads(N20_REPORT.read_text(encoding="utf-8"))
    if n20.get("selected_alpha") is not None or n20.get("offline_admission") is not False:
        raise ValueError("N21 requires the rejected N20 projection family")

    sys.path.insert(0, str(LIVE.resolve()))
    from soldier import features  # type: ignore  # noqa: E402
    from soldier.belief import Belief  # type: ignore  # noqa: E402
    from soldier.net import load_policy, save_policy  # type: ignore  # noqa: E402
    from audit_n19_retention_teacher_corpus import extract_examples  # noqa: E402
    import train_n12_counterfactual_margin_adapter as support  # noqa: E402

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
    for role in ("teacher", "retention"):
        for split in ("train", "validation", "test"):
            examples, counts = extract_examples(
                selected_games[role][split], role, deck_library, Belief, features
            )
            if dict(counts) != audit["extraction"][role][split]:
                raise ValueError(f"N21 N19 extraction changed: {role}/{split}")
            data[role][split] = examples

    source = load_policy(LIVE / "model.pt", map_location=device).cpu().eval()
    base_probabilities = {
        role: {
            split: cache_base_probabilities(source, data[role][split], device)
            for split in ("train", "validation", "test")
        }
        for role in ("teacher", "retention")
    }

    n6_examples, n6_provenance = support.load_n6_examples(
        support.CORPUS, support.AUDIT
    )
    n9_deck = [
        int(value)
        for value in (support.BASE / "deck.csv").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    n9_library = json.loads(
        (support.BASE / "deck_library.json").read_text(encoding="utf-8")
    )
    n9_examples, n9_provenance = support.load_n9_examples(
        support.BASE,
        support.REPLAYS,
        support.PARITY,
        n9_deck,
        n9_library,
        Belief,
        features,
    )
    n6_reference = support.attach_reference(source, n6_examples)
    n9_reference = support.attach_reference(source, n9_examples)
    target_audit = json.loads(N17_TARGET_AUDIT.read_text(encoding="utf-8"))
    excluded = {
        str(row["observation_fingerprint"]) for row in target_audit["targets"]
    }
    n6_anchors = [
        example
        for example in n6_examples
        if example.game.removeprefix("n6r:") not in excluded
    ]
    if len(n6_anchors) != 469 or len(n9_examples) != 6_753:
        raise ValueError("N21 historical anchor support changed")

    model = copy.deepcopy(source).train()
    trainable = configure_trainable(model, "last")
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != TRAINABLE_PARAMETERS:
        raise ValueError("N21 trainable parameter count changed")
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
    fresh_train = data["retention"]["train"]
    steps_per_epoch = math.ceil(len(teacher_train) / BATCH_SIZE)
    history: list[dict[str, Any]] = []
    total_steps = 0
    started = time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        teacher_order = list(range(len(teacher_train)))
        fresh_order = list(range(len(fresh_train)))
        n6_order = list(range(len(n6_anchors)))
        n9_order = list(range(len(n9_examples)))
        random.Random(SEED + epoch).shuffle(teacher_order)
        random.Random(SEED + 100 + epoch).shuffle(fresh_order)
        random.Random(SEED + 200 + epoch).shuffle(n6_order)
        random.Random(SEED + 300 + epoch).shuffle(n9_order)
        sums = Counter()
        seen = 0
        for step in range(steps_per_epoch):
            teacher_indexes = teacher_order[
                step * BATCH_SIZE : min((step + 1) * BATCH_SIZE, len(teacher_order))
            ]
            count = len(teacher_indexes)
            fresh_indexes = cycle_indexes(fresh_order, step, count)
            n6_indexes = cycle_indexes(n6_order, step, count)
            n9_indexes = cycle_indexes(n9_order, step, count)

            teacher_examples = [teacher_train[index] for index in teacher_indexes]
            fresh_examples = [fresh_train[index] for index in fresh_indexes]
            selected_n6 = [n6_anchors[index] for index in n6_indexes]
            selected_n9 = [n9_examples[index] for index in n9_indexes]
            teacher_batch = collate(teacher_examples, device)
            fresh_batch = collate(fresh_examples, device)
            teacher_labels = torch.as_tensor(
                [example.action for example in teacher_examples],
                dtype=torch.long,
                device=device,
            )
            teacher_logits, _ = model.forward_batch(teacher_batch)
            fresh_logits, _ = model.forward_batch(fresh_batch)
            n6_logits = support.full_forward_deployed(model, selected_n6, device)
            n9_logits = support.full_forward_deployed(model, selected_n9, device)
            teacher_target = padded_probabilities(
                base_probabilities["teacher"]["train"],
                teacher_indexes,
                teacher_logits.shape[1],
                device,
            )
            fresh_target = padded_probabilities(
                base_probabilities["retention"]["train"],
                fresh_indexes,
                fresh_logits.shape[1],
                device,
            )

            components = {
                "teacher_ce": masked_cross_entropy(
                    teacher_logits,
                    teacher_labels,
                    teacher_batch["opt_mask"],
                    LABEL_SMOOTHING,
                ),
                "teacher_kl": distribution_kl(teacher_logits, teacher_target),
                "fresh_margin": probability_margin_loss(fresh_logits, fresh_target),
                "fresh_kl": distribution_kl(fresh_logits, fresh_target),
                "n6_margin": support.retention_loss(
                    n6_logits,
                    selected_n6,
                    cap=MARGIN_CAP,
                    temperature=MARGIN_TEMPERATURE,
                ),
                "n6_kl": support.forward_kl(n6_logits, selected_n6),
                "n9_margin": support.retention_loss(
                    n9_logits,
                    selected_n9,
                    cap=MARGIN_CAP,
                    temperature=MARGIN_TEMPERATURE,
                ),
                "n9_kl": support.forward_kl(n9_logits, selected_n9),
                "source_anchor": torch.stack(
                    [
                        (parameter - source_parameters[name]).square().mean()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    ]
                ).mean(),
            }
            loss = (
                components["teacher_ce"]
                + TEACHER_KL_COEF * components["teacher_kl"]
                + FRESH_MARGIN_COEF * components["fresh_margin"]
                + FRESH_KL_COEF * components["fresh_kl"]
                + N6_MARGIN_COEF * components["n6_margin"]
                + N6_KL_COEF * components["n6_kl"]
                + N9_MARGIN_COEF * components["n9_margin"]
                + N9_KL_COEF * components["n9_kl"]
                + SOURCE_ANCHOR_COEF * components["source_anchor"]
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("N21 produced nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            sums["loss"] += float(loss.detach()) * count
            for name, value in components.items():
                sums[name] += float(value.detach()) * count
            seen += count
            total_steps += 1

        teacher_validation = evaluate(
            model, data["teacher"]["validation"], BATCH_SIZE, device
        )
        fresh_validation = combined_retention_metrics(
            model,
            data["retention"]["validation"],
            base_probabilities["retention"]["validation"],
            device,
        )
        row = {
            "epoch": epoch,
            "steps": total_steps,
            **{name: value / max(seen, 1) for name, value in sums.items()},
            "teacher_validation": teacher_validation,
            "fresh_validation": fresh_validation,
            "elapsed_seconds": time() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    teacher_metrics = {
        split: evaluate(model, data["teacher"][split], BATCH_SIZE, device)
        for split in ("train", "validation", "test")
    }
    fresh_retention = {
        split: combined_retention_metrics(
            model,
            data["retention"][split],
            base_probabilities["retention"][split],
            device,
        )
        for split in ("train", "validation", "test")
    }
    n6_retention = full_preservation_metrics(
        model, n6_anchors, support, device
    )
    n9_retention = full_preservation_metrics(
        model, n9_examples, support, device
    )
    comparison = state_comparison(source, model)
    if set(comparison["changed_tensors"]) != set(trainable_names):
        raise ValueError("N21 changed tensor set differs from lock")

    baseline = audit["baseline_metrics"]
    gates = {
        "state_valid": (
            not comparison["shape_errors"]
            and not comparison["nonfinite_tensors"]
            and len(comparison["changed_tensors"]) == 14
            and len(comparison["exact_tensors"]) == 45
        ),
        "teacher_validation_top1_plus_2pp": (
            teacher_metrics["validation"]["top1"]
            >= baseline["teacher"]["validation"]["live"]["top1"] + 0.02
        ),
        "teacher_validation_loss_lower": (
            teacher_metrics["validation"]["loss"]
            < baseline["teacher"]["validation"]["live"]["loss"]
        ),
        "teacher_test_top1_plus_1pp": (
            teacher_metrics["test"]["top1"]
            >= baseline["teacher"]["test"]["live"]["top1"] + 0.01
        ),
        "teacher_test_loss_lower": (
            teacher_metrics["test"]["loss"]
            < baseline["teacher"]["test"]["live"]["loss"]
        ),
        "fresh_each_split_top1_ge_0_995": all(
            row["base_top1_retention"] >= 0.995 for row in fresh_retention.values()
        ),
        "fresh_each_split_kl_le_0_01": all(
            row["mean_forward_kl"] <= 0.01 for row in fresh_retention.values()
        ),
        "recorded_validation_within_half_pp": (
            fresh_retention["validation"]["recorded_top1"]
            >= baseline["retention"]["validation"]["live"]["top1"] - 0.005
        ),
        "recorded_test_within_half_pp": (
            fresh_retention["test"]["recorded_top1"]
            >= baseline["retention"]["test"]["live"]["top1"] - 0.005
        ),
        "n6_top1_ge_0_99": n6_retention["top1_retention"] >= 0.99,
        "n6_kl_le_0_05": n6_retention["mean_forward_kl"] <= 0.05,
        "n9_top1_ge_0_99": n9_retention["top1_retention"] >= 0.99,
        "n9_each_group_ge_0_985": all(
            row["top1_retention"] >= 0.985
            for row in n9_retention["by_group"].values()
        ),
        "n9_kl_le_0_02": n9_retention["mean_forward_kl"] <= 0.02,
    }
    offline_admission = all(gates.values())

    copy_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
    shutil.copytree(LIVE, OUTPUT, ignore=copy_ignore)
    save_policy(model.eval(), OUTPUT / "model.pt")
    reloaded = load_policy(OUTPUT / "model.pt", map_location=device).cpu().eval()
    serialized = state_comparison(model, reloaded)
    if serialized["changed_tensors"] or serialized["shape_errors"]:
        raise ValueError("N21 serialized model changed on reload")
    terminal_sha = digest(OUTPUT / "model.pt")

    report = {
        "schema": "rocket-n21-top-action-margin-training/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "source_model_sha256": LIVE_MODEL_SHA,
        "terminal_model_sha256": terminal_sha,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "corpus_audit_sha256": CORPUS_AUDIT_SHA,
        "n19_result_sha256": N19_RESULT_SHA,
        "n20_report_sha256": N20_REPORT_SHA,
        "n20_result_sha256": N20_RESULT_SHA,
        "n17_target_audit_sha256": N17_TARGET_AUDIT_SHA,
        "hyperparameters": {
            "seed": SEED,
            "epochs": EPOCHS,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "batch_size_per_objective": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "teacher_kl_coefficient": TEACHER_KL_COEF,
            "fresh_margin_coefficient": FRESH_MARGIN_COEF,
            "fresh_kl_coefficient": FRESH_KL_COEF,
            "n6_margin_coefficient": N6_MARGIN_COEF,
            "n6_kl_coefficient": N6_KL_COEF,
            "n9_margin_coefficient": N9_MARGIN_COEF,
            "n9_kl_coefficient": N9_KL_COEF,
            "source_anchor_coefficient": SOURCE_ANCHOR_COEF,
            "margin_cap": MARGIN_CAP,
            "margin_temperature": MARGIN_TEMPERATURE,
            "gradient_norm_cap": 1.0,
            "threads": THREADS,
            "checkpoint_rule": "terminal_epoch_2_only",
        },
        "trainable_parameters": trainable_count,
        "trainable_tensors": trainable_names,
        "n6_provenance": n6_provenance,
        "n6_reference": n6_reference,
        "n6_unchanged_anchors": len(n6_anchors),
        "n9_provenance": n9_provenance,
        "n9_reference": n9_reference,
        "history": history,
        "baseline_metrics": baseline,
        "teacher_metrics": teacher_metrics,
        "fresh_retention": fresh_retention,
        "n6_retention": n6_retention,
        "n9_retention": n9_retention,
        "state_comparison": comparison,
        "gates": gates,
        "offline_admission": offline_admission,
        "official_match_evidence": False,
        "development_panel_authority": offline_admission,
        "deployment_authority": False,
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "Offline top-action-margin training only. Passage permits a separately locked "
            "official-engine development panel, never direct deployment."
        ),
    }
    (OUTPUT / "N21_TRAINING_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    provenance = {
        "schema": "rocket-n21-training-provenance/v1",
        "trainer_sha256": digest(Path(__file__).resolve()),
        "training_report_sha256": digest(OUTPUT / "N21_TRAINING_REPORT.json"),
        "protocol_sha256": PROTOCOL_SHA,
        "source_model_sha256": LIVE_MODEL_SHA,
        "terminal_model_sha256": terminal_sha,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "offline_admission": offline_admission,
        "development_panel_authority": offline_admission,
        "deployment_authority": False,
    }
    (OUTPUT / "N21_TRAINING_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "terminal_model_sha256": terminal_sha,
                "training_report_sha256": digest(OUTPUT / "N21_TRAINING_REPORT.json"),
                "teacher_validation_top1": teacher_metrics["validation"]["top1"],
                "teacher_test_top1": teacher_metrics["test"]["top1"],
                "fresh_retention": fresh_retention,
                "n6_retention": n6_retention,
                "n9_retention": n9_retention,
                "offline_admission": offline_admission,
                "development_panel_authority": offline_admission,
                "deployment_authority": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
