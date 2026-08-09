#!/usr/bin/env python3
"""Evaluate and select the sole preregistered N20 trust-region projection."""

from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "work/n18/blive"
TERMINAL = (
    ROOT
    / "artifacts/team_rocket_recovery/n19_retention_teacher/"
    "training_v1"
)
CORPUS_AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n19_retention_teacher/"
    "N19_CORPUS_AUDIT.json"
)
TRAINING_REPORT = TERMINAL / "N19_TRAINING_REPORT.json"
N19_RESULT = ROOT / "N19_RETENTION_CONSTRAINED_LEADER_BC_RESULT.md"
PROTOCOL = ROOT / "N20_TRUST_REGION_PROJECTION_PROTOCOL.md"
N17_TARGET_AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n17_replicated_targets/"
    "target_audit.json"
)
OUTPUT = (
    ROOT
    / "artifacts/team_rocket_recovery/n20_trust_region_projection/"
    "projection_v1"
)
N19_TRAINER = ROOT / "scripts/train_n19_retention_constrained_bc.py"
CORPUS_AUDITOR = ROOT / "scripts/audit_n19_retention_teacher_corpus.py"
PUBLIC_BC_SUPPORT = ROOT / "scripts/train_public_leader_v110_bc.py"
N12_SUPPORT = ROOT / "scripts/train_n12_counterfactual_margin_adapter.py"
LIVE_AWR_SUPPORT = ROOT / "scripts/train_live_outcome_awr.py"
N9_SUPPORT = ROOT / "scripts/train_n9_balanced_pcgrad_awr.py"
N6_SUPPORT = ROOT / "scripts/train_belief_rollout_distillation_n6.py"

LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
TERMINAL_MODEL_SHA = "b0f0e72f7eafbf2bd5e0e4eca1920398534ddef516b57750a3c1f58dbb7d467f"
CORPUS_AUDIT_SHA = "d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed"
TRAINING_REPORT_SHA = "ed88d68c63988e79b74e0c33966bcdfe219cabfb78d9b0fe19f1482df7ebbe43"
N19_RESULT_SHA = "cfb2560fca269d969a9490453db28b7f7a3bf6598e96864cfd886ecc1962f227"
PROTOCOL_SHA = "8f4628bf07f5a1abf21ba132bc9c7a7705031652a9d74560adbb509e1a9e952a"
N17_TARGET_AUDIT_SHA = "c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c"
N19_TRAINER_SHA = "3928abaf17457d05e02c7d9734f69f819705601051154bc78236a981868de7b7"
CORPUS_AUDITOR_SHA = "86f6e12bb96c16ebfa0980e27974bff881537cb8e750e9c5adc6a688bfda58c1"
PUBLIC_BC_SUPPORT_SHA = "3dcd9bef8dcc1cf17bd0fb2360ef1e4d4d90ab8a9ec715d7ed4108515fb35a34"
N12_SUPPORT_SHA = "843c12e0d2a1684d53ed27423c6b6c5cfe2dc38109c57b089b3d98e38cf181bd"
LIVE_AWR_SUPPORT_SHA = "4f233d40e625c33711ab87438388f8ba68f65f2e103c563bf621c790c9a657e7"
N9_SUPPORT_SHA = "62182583002847e7ed0e18011050f35319ad5661b73f3fb2d83c2564286ed1ad"
N6_SUPPORT_SHA = "3c559b00c4d59c018295406e84dc78f2e19b5014b9174cb23c8db63790020a7f"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"

ELIGIBLE_ALPHAS = (0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875)
DIAGNOSTIC_ALPHAS = (0.0, 1.0)
BATCH_SIZE = 96
THREADS = 6


sys.path.insert(0, str((ROOT / "scripts").resolve()))
from train_public_leader_v110_bc import Example, collate, evaluate  # noqa: E402
from train_n19_retention_constrained_bc import (  # noqa: E402
    cache_base_probabilities,
    load_game,
    padded_probabilities,
    state_comparison,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def combined_retention_metrics(
    model: torch.nn.Module,
    examples: list[Example],
    base_probabilities: list[np.ndarray],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    correct = retained = total = 0
    loss_total = kl_total = 0.0
    for start in range(0, len(examples), BATCH_SIZE):
        selected = examples[start : start + BATCH_SIZE]
        indexes = list(range(start, start + len(selected)))
        batch = collate(selected, device)
        labels = torch.as_tensor(
            [example.action for example in selected], dtype=torch.long, device=device
        )
        logits, _ = model.forward_batch(batch)
        target = padded_probabilities(
            base_probabilities, indexes, logits.shape[1], device
        )
        log_candidate = torch.log_softmax(logits, dim=1)
        target_log = torch.where(
            target > 0.0,
            torch.log(target.clamp_min(1.0e-30)),
            torch.zeros_like(target),
        )
        row_kl = torch.where(
            target > 0.0,
            target * (target_log - log_candidate),
            torch.zeros_like(target),
        ).sum(dim=1)
        loss_total += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=1) == labels).sum())
        retained += int((logits.argmax(dim=1) == target.argmax(dim=1)).sum())
        kl_total += float(row_kl.sum())
        total += len(selected)
    return {
        "examples": total,
        "recorded_loss": loss_total / max(total, 1),
        "recorded_top1": correct / max(total, 1),
        "base_top1_retention": retained / max(total, 1),
        "mean_forward_kl": kl_total / max(total, 1),
    }


@torch.no_grad()
def full_preservation_metrics(
    model: torch.nn.Module,
    examples: Sequence[Any],
    support: Any,
    device: torch.device,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    model.eval()
    total = retained = 0
    kl_total = 0.0
    grouped: dict[str, Counter[str]] = {}
    for start in range(0, len(examples), batch_size):
        selected = list(examples[start : start + batch_size])
        logits = support.full_forward_deployed(model, selected, device)
        reference = support.pad_reference(selected, logits.shape[1], device)
        logp = torch.log_softmax(logits, dim=1)
        reference_log = torch.log(reference.clamp_min(1.0e-30))
        row_kl = torch.where(
            reference > 0.0,
            reference * (reference_log - logp),
            torch.zeros_like(reference),
        ).sum(dim=1)
        predictions = logits.argmax(dim=1).tolist()
        for row, (example, prediction) in enumerate(zip(selected, predictions)):
            keep = int(prediction == example.reference_top)
            total += 1
            retained += keep
            kl_total += float(row_kl[row])
            group = grouped.setdefault(str(example.group), Counter())
            group["examples"] += 1
            group["retained"] += keep
    return {
        "examples": total,
        "retained": retained,
        "top1_retention": retained / max(total, 1),
        "mean_forward_kl": kl_total / max(total, 1),
        "by_group": {
            name: {
                "examples": int(row["examples"]),
                "retained": int(row["retained"]),
                "top1_retention": row["retained"] / max(1, row["examples"]),
            }
            for name, row in sorted(grouped.items())
        },
    }


def projected_model(
    source: torch.nn.Module,
    terminal: torch.nn.Module,
    alpha: float,
    changed: set[str],
) -> torch.nn.Module:
    candidate = copy.deepcopy(source)
    source_state = source.state_dict()
    terminal_state = terminal.state_dict()
    candidate_state = candidate.state_dict()
    for key in candidate_state:
        if key in changed:
            candidate_state[key] = source_state[key] + alpha * (
                terminal_state[key] - source_state[key]
            )
        else:
            candidate_state[key] = source_state[key].clone()
    candidate.load_state_dict(candidate_state, strict=True)
    return candidate.eval()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N20 projection output: {OUTPUT}")
    for path, expected in (
        (LIVE / "model.pt", LIVE_MODEL_SHA),
        (LIVE / "deck.csv", DECK_SHA),
        (TERMINAL / "model.pt", TERMINAL_MODEL_SHA),
        (CORPUS_AUDIT, CORPUS_AUDIT_SHA),
        (TRAINING_REPORT, TRAINING_REPORT_SHA),
        (N19_RESULT, N19_RESULT_SHA),
        (PROTOCOL, PROTOCOL_SHA),
        (N17_TARGET_AUDIT, N17_TARGET_AUDIT_SHA),
        (N19_TRAINER, N19_TRAINER_SHA),
        (CORPUS_AUDITOR, CORPUS_AUDITOR_SHA),
        (PUBLIC_BC_SUPPORT, PUBLIC_BC_SUPPORT_SHA),
        (N12_SUPPORT, N12_SUPPORT_SHA),
        (LIVE_AWR_SUPPORT, LIVE_AWR_SUPPORT_SHA),
        (N9_SUPPORT, N9_SUPPORT_SHA),
        (N6_SUPPORT, N6_SUPPORT_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"N20 frozen input changed: {path}")

    torch.set_num_threads(THREADS)
    torch.manual_seed(820001)
    np.random.seed(820001)
    device = torch.device("cpu")
    audit = json.loads(CORPUS_AUDIT.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    if training.get("corpus_gates_pass") is not False:
        raise ValueError("N20 requires the frozen rejected N19 terminal")

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
                raise ValueError(f"N20 N19 extraction changed: {role}/{split}")
            data[role][split] = examples

    source = load_policy(LIVE / "model.pt", map_location=device).cpu().eval()
    terminal = load_policy(TERMINAL / "model.pt", map_location=device).cpu().eval()
    comparison = state_comparison(source, terminal)
    changed = set(comparison["changed_tensors"])
    if len(changed) != 14 or len(comparison["exact_tensors"]) != 45:
        raise ValueError("N20 source-to-terminal direction changed")

    base_probabilities = {
        split: cache_base_probabilities(source, data["retention"][split], device)
        for split in ("train", "validation", "test")
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
    n17_targets = {
        str(row["observation_fingerprint"]) for row in target_audit["targets"]
    }
    if len(n17_targets) != 170:
        raise ValueError("N20 N17 target exclusion set changed")
    n6_anchors = [
        example
        for example in n6_examples
        if example.game.removeprefix("n6r:") not in n17_targets
    ]
    if len(n6_anchors) != 469:
        raise ValueError(f"N20 N6 unchanged-anchor count changed: {len(n6_anchors)}")

    baseline = audit["baseline_metrics"]
    grid: dict[str, Any] = {}
    passing: list[float] = []
    started = time()
    for alpha in (*DIAGNOSTIC_ALPHAS, *ELIGIBLE_ALPHAS):
        candidate = projected_model(source, terminal, alpha, changed)
        state = state_comparison(source, candidate)
        teacher_metrics = {
            split: evaluate(candidate, data["teacher"][split], BATCH_SIZE, device)
            for split in ("train", "validation", "test")
        }
        fresh_retention = {
            split: combined_retention_metrics(
                candidate,
                data["retention"][split],
                base_probabilities[split],
                device,
            )
            for split in ("train", "validation", "test")
        }
        n6 = full_preservation_metrics(candidate, n6_anchors, support, device)
        n9 = full_preservation_metrics(candidate, n9_examples, support, device)
        gates = {
            "state_valid": (
                not state["shape_errors"]
                and not state["nonfinite_tensors"]
                and set(state["changed_tensors"]).issubset(changed)
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
                row["base_top1_retention"] >= 0.995
                for row in fresh_retention.values()
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
            "n6_top1_ge_0_99": n6["top1_retention"] >= 0.99,
            "n6_kl_le_0_05": n6["mean_forward_kl"] <= 0.05,
            "n9_top1_ge_0_99": n9["top1_retention"] >= 0.99,
            "n9_each_group_ge_0_985": all(
                row["top1_retention"] >= 0.985 for row in n9["by_group"].values()
            ),
            "n9_kl_le_0_02": n9["mean_forward_kl"] <= 0.02,
        }
        eligible = alpha in ELIGIBLE_ALPHAS
        passes = eligible and all(gates.values())
        if passes:
            passing.append(alpha)
        grid[f"{alpha:.3f}"] = {
            "alpha": alpha,
            "eligible": eligible,
            "teacher_metrics": teacher_metrics,
            "fresh_retention": fresh_retention,
            "n6_retention": n6,
            "n9_retention": n9,
            "state_comparison": state,
            "gates": gates,
            "passes": passes,
        }
        print(
            json.dumps(
                {
                    "alpha": alpha,
                    "eligible": eligible,
                    "passes": passes,
                    "teacher_validation_top1": teacher_metrics["validation"]["top1"],
                    "fresh_min_top1_retention": min(
                        row["base_top1_retention"] for row in fresh_retention.values()
                    ),
                    "n6_top1_retention": n6["top1_retention"],
                    "n9_top1_retention": n9["top1_retention"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    selected_alpha = max(passing) if passing else None
    selected_model_sha = None
    if selected_alpha is not None:
        selected = projected_model(source, terminal, selected_alpha, changed)
        copy_ignore = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", ".pytest_cache"
        )
        shutil.copytree(LIVE, OUTPUT, ignore=copy_ignore)
        save_policy(selected, OUTPUT / "model.pt")
        reloaded = load_policy(OUTPUT / "model.pt", map_location=device).cpu().eval()
        serialized = state_comparison(selected, reloaded)
        if serialized["changed_tensors"] or serialized["shape_errors"]:
            raise ValueError("N20 selected model changed on serialization")
        selected_model_sha = digest(OUTPUT / "model.pt")
    else:
        OUTPUT.mkdir(parents=True)

    report = {
        "schema": "rocket-n20-trust-region-projection/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "source_model_sha256": LIVE_MODEL_SHA,
        "terminal_model_sha256": TERMINAL_MODEL_SHA,
        "corpus_audit_sha256": CORPUS_AUDIT_SHA,
        "n19_training_report_sha256": TRAINING_REPORT_SHA,
        "n19_result_sha256": N19_RESULT_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "eligible_alphas": list(ELIGIBLE_ALPHAS),
        "diagnostic_alphas": list(DIAGNOSTIC_ALPHAS),
        "direction_changed_tensors": sorted(changed),
        "n6_provenance": n6_provenance,
        "n6_reference": n6_reference,
        "n6_unchanged_anchors": len(n6_anchors),
        "n17_target_audit_sha256": N17_TARGET_AUDIT_SHA,
        "n9_provenance": n9_provenance,
        "n9_reference": n9_reference,
        "grid": grid,
        "passing_alphas": passing,
        "selected_alpha": selected_alpha,
        "selected_model_sha256": selected_model_sha,
        "offline_admission": selected_alpha is not None,
        "official_match_evidence": False,
        "development_panel_authority": selected_alpha is not None,
        "deployment_authority": False,
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "Offline fixed-grid trust-region projection only. A selected alpha permits a separately "
            "locked official-engine development panel, not deployment or a strength claim."
        ),
    }
    (OUTPUT / "N20_PROJECTION_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    provenance = {
        "schema": "rocket-n20-projection-provenance/v1",
        "projection_script_sha256": digest(Path(__file__).resolve()),
        "projection_report_sha256": digest(OUTPUT / "N20_PROJECTION_REPORT.json"),
        "protocol_sha256": PROTOCOL_SHA,
        "source_model_sha256": LIVE_MODEL_SHA,
        "terminal_model_sha256": TERMINAL_MODEL_SHA,
        "selected_alpha": selected_alpha,
        "selected_model_sha256": selected_model_sha,
        "development_panel_authority": selected_alpha is not None,
        "deployment_authority": False,
    }
    (OUTPUT / "N20_PROJECTION_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "projection_report_sha256": digest(OUTPUT / "N20_PROJECTION_REPORT.json"),
                "passing_alphas": passing,
                "selected_alpha": selected_alpha,
                "selected_model_sha256": selected_model_sha,
                "development_panel_authority": selected_alpha is not None,
                "deployment_authority": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
