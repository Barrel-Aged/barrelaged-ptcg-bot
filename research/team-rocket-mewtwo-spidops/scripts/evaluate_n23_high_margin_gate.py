#!/usr/bin/env python3
"""Evaluate the sole N23 held-out high-margin all-or-live gate."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from time import time
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "work/n18/blive"
CANDIDATE = ROOT / "artifacts/team_rocket_recovery/n21_top_action_margin/training_v1"
CORPUS_AUDIT = ROOT / "artifacts/team_rocket_recovery/n19_retention_teacher/N19_CORPUS_AUDIT.json"
N21_RESULT = ROOT / "N21_TOP_ACTION_MARGIN_BC_RESULT.md"
N22_AUDIT = ROOT / "artifacts/team_rocket_recovery/n22_disagreement_gate/TRAIN_ONLY_AUDIT.json"
N22_RESULT = ROOT / "N22_TRAIN_ONLY_DISAGREEMENT_AUDIT_RESULT.md"
N22_SCRIPT = ROOT / "scripts/audit_n22_disagreement_gate_feasibility.py"
N17_TARGET_AUDIT = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/target_audit.json"
PROTOCOL = ROOT / "N23_HELDOUT_HIGH_MARGIN_GATE_PROTOCOL.md"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n23_high_margin_gate/EVALUATION.json"

LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_MODEL_SHA = "7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CORPUS_AUDIT_SHA = "d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed"
N21_RESULT_SHA = "53683ec3d968e59d4343c5b7294aec00d46b2e8d0a49ed7a6d54d6f4890b1e98"
N22_AUDIT_SHA = "1dd3904da69914c322fe6d4b0f4337d1a4ed6146533519353431c9f70c999dd8"
N22_RESULT_SHA = "df2bbb6343999ad2db77ea3e49256a9cac10cad838d768e353b4787ac6833b5d"
N22_SCRIPT_SHA = "e5377d671fabc1141e28af05cea4b8ef7f418fad843822c5b0f61d988c1d8c56"
N17_TARGET_AUDIT_SHA = "c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c"
PROTOCOL_SHA = "b9eb93a9103a933b22e8f86c469af7ae39e06672e028643002c69ec295170171"

N19_TRAINER = ROOT / "scripts/train_n19_retention_constrained_bc.py"
CORPUS_AUDITOR = ROOT / "scripts/audit_n19_retention_teacher_corpus.py"
PUBLIC_BC_SUPPORT = ROOT / "scripts/train_public_leader_v110_bc.py"
N12_SUPPORT = ROOT / "scripts/train_n12_counterfactual_margin_adapter.py"
LIVE_AWR_SUPPORT = ROOT / "scripts/train_live_outcome_awr.py"
N9_SUPPORT = ROOT / "scripts/train_n9_balanced_pcgrad_awr.py"
N6_SUPPORT = ROOT / "scripts/train_belief_rollout_distillation_n6.py"
SUPPORT_HASHES = {
    N19_TRAINER: "3928abaf17457d05e02c7d9734f69f819705601051154bc78236a981868de7b7",
    CORPUS_AUDITOR: "86f6e12bb96c16ebfa0980e27974bff881537cb8e750e9c5adc6a688bfda58c1",
    PUBLIC_BC_SUPPORT: "3dcd9bef8dcc1cf17bd0fb2360ef1e4d4d90ab8a9ec715d7ed4108515fb35a34",
    N12_SUPPORT: "843c12e0d2a1684d53ed27423c6b6c5cfe2dc38109c57b089b3d98e38cf181bd",
    LIVE_AWR_SUPPORT: "4f233d40e625c33711ab87438388f8ba68f65f2e103c563bf621c790c9a657e7",
    N9_SUPPORT: "62182583002847e7ed0e18011050f35319ad5661b73f3fb2d83c2564286ed1ad",
    N6_SUPPORT: "3c559b00c4d59c018295406e84dc78f2e19b5014b9174cb23c8db63790020a7f",
}

THRESHOLD = 0.50
BATCH_SIZE = 96
THREADS = 6


sys.path.insert(0, str((ROOT / "scripts").resolve()))
from train_public_leader_v110_bc import Example, collate  # noqa: E402
from train_n19_retention_constrained_bc import load_game  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_available(logits: np.ndarray, available: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(available > 0.5)
    if not len(indices):
        raise ValueError("N23 decision has no available action")
    if not np.isfinite(logits[indices]).all():
        raise ValueError("N23 available logit is nonfinite")
    return indices


def top_and_margin(logits: np.ndarray, available: np.ndarray) -> tuple[int, float]:
    indices = finite_available(logits, available)
    order = indices[np.argsort(logits[indices])]
    top = int(order[-1])
    runner = float(logits[order[-2]]) if len(order) > 1 else float(logits[top] - 50.0)
    return top, min(float(logits[top] - runner), 50.0)


def negative_log_probability(logits: np.ndarray, available: np.ndarray, action: int) -> float:
    indices = finite_available(logits, available)
    if action not in set(indices.tolist()):
        raise ValueError("N23 recorded action is unavailable after deployment guard")
    maximum = float(np.max(logits[indices]))
    log_partition = maximum + math.log(float(np.exp(logits[indices] - maximum).sum()))
    return log_partition - float(logits[action])


def empty_metric() -> Counter[str]:
    return Counter()


def finish_metric(counts: Counter[str], loss: float, live_loss: float,
                  margins: list[float]) -> dict[str, Any]:
    examples = int(counts["examples"])
    evidence = counts["candidate_only"] + counts["live_only"]
    return {
        "examples": examples,
        "gated_loss": loss / max(examples, 1),
        "live_loss_reproduced": live_loss / max(examples, 1),
        "gated_top1": counts["gated_correct"] / max(examples, 1),
        "live_top1_reproduced": counts["live_correct"] / max(examples, 1),
        "live_top_retention": counts["retained"] / max(examples, 1),
        "replacements": int(counts["replacements"]),
        "replacement_rate": counts["replacements"] / max(examples, 1),
        "replacement_labels": {
            key: int(counts[key])
            for key in ("candidate_only", "live_only", "neither")
        },
        "candidate_precision_vs_live_only": (
            counts["candidate_only"] / evidence if evidence else None
        ),
        "replacement_margin": {
            "count": len(margins),
            "min": min(margins) if margins else None,
            "median": float(np.median(margins)) if margins else None,
            "max": max(margins) if margins else None,
        },
    }


@torch.no_grad()
def gated_n19_metrics(live: torch.nn.Module, candidate: torch.nn.Module,
                      examples: list[Example], device: torch.device) -> dict[str, Any]:
    counts = empty_metric()
    total_loss = live_loss = 0.0
    margins: list[float] = []
    for start in range(0, len(examples), BATCH_SIZE):
        selected = examples[start:start + BATCH_SIZE]
        batch = collate(selected, device)
        live_logits, _ = live.forward_batch(batch)
        candidate_logits, _ = candidate.forward_batch(batch)
        for row_index, example in enumerate(selected):
            width = int(example.feat["opt_feats"].shape[0])
            available = np.asarray(
                example.feat.get("opt_avail", np.ones(width, dtype=np.float32)),
                dtype=np.float64,
            )
            live_row = live_logits[row_index, :width].cpu().numpy().astype(np.float64)
            candidate_row = (
                candidate_logits[row_index, :width].cpu().numpy().astype(np.float64)
            )
            live_top, _ = top_and_margin(live_row, available)
            candidate_top, candidate_margin = top_and_margin(candidate_row, available)
            replace = live_top != candidate_top and candidate_margin >= THRESHOLD
            gated_row = candidate_row if replace else live_row
            gated_top = candidate_top if replace else live_top
            action = int(example.action)
            total_loss += negative_log_probability(gated_row, available, action)
            live_loss += negative_log_probability(live_row, available, action)
            counts["examples"] += 1
            counts["live_correct"] += int(live_top == action)
            counts["gated_correct"] += int(gated_top == action)
            counts["retained"] += int(gated_top == live_top)
            if replace:
                counts["replacements"] += 1
                margins.append(candidate_margin)
                if candidate_top == action:
                    counts["candidate_only"] += 1
                elif live_top == action:
                    counts["live_only"] += 1
                else:
                    counts["neither"] += 1
    return finish_metric(counts, total_loss, live_loss, margins)


@torch.no_grad()
def gated_preservation_metrics(candidate: torch.nn.Module, examples: Sequence[Any],
                               support: Any, device: torch.device) -> dict[str, Any]:
    counts = Counter()
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    margins: list[float] = []
    for start in range(0, len(examples), 128):
        selected = list(examples[start:start + 128])
        logits = support.full_forward_deployed(candidate, selected, device)
        for row_index, example in enumerate(selected):
            width = int(example.feat["opt_feats"].shape[0])
            available = np.asarray(example.deployment_available[:width], dtype=np.float64)
            candidate_row = logits[row_index, :width].cpu().numpy().astype(np.float64)
            candidate_top, candidate_margin = top_and_margin(candidate_row, available)
            live_top = int(example.reference_top)
            replace = candidate_top != live_top and candidate_margin >= THRESHOLD
            gated_top = candidate_top if replace else live_top
            group = str(example.group)
            counts["examples"] += 1
            grouped[group]["examples"] += 1
            keep = int(gated_top == live_top)
            counts["retained"] += keep
            grouped[group]["retained"] += keep
            if replace:
                counts["replacements"] += 1
                grouped[group]["replacements"] += 1
                margins.append(candidate_margin)
    return {
        "examples": int(counts["examples"]),
        "retained": int(counts["retained"]),
        "live_top_retention": counts["retained"] / max(counts["examples"], 1),
        "replacements": int(counts["replacements"]),
        "replacement_margin": {
            "count": len(margins),
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
        },
        "by_group": {
            group: {
                "examples": int(row["examples"]),
                "retained": int(row["retained"]),
                "live_top_retention": row["retained"] / max(row["examples"], 1),
                "replacements": int(row["replacements"]),
            }
            for group, row in sorted(grouped.items())
        },
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N23 evaluation: {OUTPUT}")
    frozen = {
        LIVE / "model.pt": LIVE_MODEL_SHA,
        LIVE / "deck.csv": DECK_SHA,
        CANDIDATE / "model.pt": CANDIDATE_MODEL_SHA,
        CORPUS_AUDIT: CORPUS_AUDIT_SHA,
        N21_RESULT: N21_RESULT_SHA,
        N22_AUDIT: N22_AUDIT_SHA,
        N22_RESULT: N22_RESULT_SHA,
        N22_SCRIPT: N22_SCRIPT_SHA,
        N17_TARGET_AUDIT: N17_TARGET_AUDIT_SHA,
        PROTOCOL: PROTOCOL_SHA,
        **SUPPORT_HASHES,
    }
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise ValueError(f"N23 frozen input changed: {path}")

    started = time()
    torch.set_num_threads(THREADS)
    torch.manual_seed(823001)
    device = torch.device("cpu")
    audit = json.loads(CORPUS_AUDIT.read_text(encoding="utf-8"))
    n22 = json.loads(N22_AUDIT.read_text(encoding="utf-8"))
    if n22.get("state_conditioned_gate_feasible") is not True:
        raise ValueError("N23 requires the passing frozen N22 feasibility audit")

    sys.path.insert(0, str(LIVE.resolve()))
    from soldier import features  # type: ignore  # noqa: E402
    from soldier.belief import Belief  # type: ignore  # noqa: E402
    from soldier.net import load_policy  # type: ignore  # noqa: E402
    from audit_n19_retention_teacher_corpus import extract_examples  # noqa: E402
    import train_n12_counterfactual_margin_adapter as support  # noqa: E402

    deck_library = json.loads((LIVE / "deck_library.json").read_text(encoding="utf-8"))
    live = load_policy(LIVE / "model.pt", map_location=device).cpu().eval()
    candidate = load_policy(CANDIDATE / "model.pt", map_location=device).cpu().eval()

    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {split: [] for split in ("train", "validation", "test")}
        for role in ("teacher", "retention")
    }
    for record in audit["game_manifest"]:
        records[record["role"]][record["split"]].append(record)
    n19_metrics: dict[str, dict[str, dict[str, Any]]] = {
        role: {} for role in ("teacher", "retention")
    }
    extraction: dict[str, dict[str, dict[str, int]]] = {
        role: {} for role in ("teacher", "retention")
    }
    for role in ("teacher", "retention"):
        for split in ("train", "validation", "test"):
            games = [load_game(record) for record in records[role][split]]
            examples, counts = extract_examples(
                games, role, deck_library, Belief, features
            )
            if dict(counts) != audit["extraction"][role][split]:
                raise ValueError(f"N23 extraction changed: {role}/{split}")
            extraction[role][split] = dict(counts)
            n19_metrics[role][split] = gated_n19_metrics(
                live, candidate, examples, device
            )

    teacher_train = n19_metrics["teacher"]["train"]
    retention_train = n19_metrics["retention"]["train"]
    train_reproduction = (
        teacher_train["replacements"] == 136
        and teacher_train["replacement_labels"]
        == {"candidate_only": 124, "live_only": 2, "neither": 10}
        and retention_train["replacements"] == 0
    )

    live_reproduction: dict[str, dict[str, bool]] = {
        role: {} for role in ("teacher", "retention")
    }
    for role in ("teacher", "retention"):
        for split in ("train", "validation", "test"):
            expected = audit["baseline_metrics"][role][split]["live"]
            actual = n19_metrics[role][split]
            live_reproduction[role][split] = (
                abs(actual["live_top1_reproduced"] - expected["top1"]) <= 1.0e-12
                and abs(actual["live_loss_reproduced"] - expected["loss"]) <= 1.0e-6
            )

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
    n6_reference = support.attach_reference(live, n6_examples)
    n9_reference = support.attach_reference(live, n9_examples)
    target_audit = json.loads(N17_TARGET_AUDIT.read_text(encoding="utf-8"))
    excluded = {str(row["observation_fingerprint"]) for row in target_audit["targets"]}
    n6_anchors = [
        example
        for example in n6_examples
        if example.game.removeprefix("n6r:") not in excluded
    ]
    if len(n6_anchors) != 469 or len(n9_examples) != 6_753:
        raise ValueError("N23 historical preservation support changed")
    n6_metrics = gated_preservation_metrics(
        candidate, n6_anchors, support, device
    )
    n9_metrics = gated_preservation_metrics(
        candidate, n9_examples, support, device
    )

    baseline = audit["baseline_metrics"]
    teacher_validation = n19_metrics["teacher"]["validation"]
    teacher_test = n19_metrics["teacher"]["test"]
    retention_validation = n19_metrics["retention"]["validation"]
    retention_test = n19_metrics["retention"]["test"]
    gates = {
        "integrity_and_live_reproduction": all(
            value for role in live_reproduction.values() for value in role.values()
        ),
        "train_threshold_reproduction": train_reproduction,
        "teacher_validation_top1_plus_half_pp": (
            teacher_validation["gated_top1"]
            >= baseline["teacher"]["validation"]["live"]["top1"] + 0.005
        ),
        "teacher_validation_loss_lower": (
            teacher_validation["gated_loss"]
            < baseline["teacher"]["validation"]["live"]["loss"]
        ),
        "teacher_test_top1_plus_half_pp": (
            teacher_test["gated_top1"]
            >= baseline["teacher"]["test"]["live"]["top1"] + 0.005
        ),
        "teacher_test_loss_lower": (
            teacher_test["gated_loss"]
            < baseline["teacher"]["test"]["live"]["loss"]
        ),
        "teacher_each_heldout_replacements_ge_10": (
            teacher_validation["replacements"] >= 10
            and teacher_test["replacements"] >= 10
        ),
        "teacher_each_heldout_precision_ge_0_80": all(
            metric["candidate_precision_vs_live_only"] is not None
            and metric["candidate_precision_vs_live_only"] >= 0.80
            for metric in (teacher_validation, teacher_test)
        ),
        "retention_each_heldout_live_top_ge_0_9975": all(
            metric["live_top_retention"] >= 0.9975
            for metric in (retention_validation, retention_test)
        ),
        "retention_each_recorded_top1_within_quarter_pp": (
            retention_validation["gated_top1"]
            >= baseline["retention"]["validation"]["live"]["top1"] - 0.0025
            and retention_test["gated_top1"]
            >= baseline["retention"]["test"]["live"]["top1"] - 0.0025
        ),
        "retention_heldout_replacements_ge_2": (
            retention_validation["replacements"]
            + retention_test["replacements"]
            >= 2
        ),
        "n6_exact_live_top": n6_metrics["live_top_retention"] == 1.0,
        "n9_exact_live_top": (
            n9_metrics["live_top_retention"] == 1.0
            and all(
                row["live_top_retention"] == 1.0
                for row in n9_metrics["by_group"].values()
            )
        ),
    }
    offline_admission = all(gates.values())

    report = {
        "schema": "rocket-n23-heldout-high-margin-gate-evaluation/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "evaluator_sha256": digest(Path(__file__).resolve()),
        "live_model_sha256": LIVE_MODEL_SHA,
        "candidate_model_sha256": CANDIDATE_MODEL_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "n22_audit_sha256": N22_AUDIT_SHA,
        "gate": {
            "candidate_margin_threshold": THRESHOLD,
            "replace_only_on_top_disagreement": True,
            "selected_distribution": "complete_candidate_or_complete_live_logit_row",
            "threshold_selection": "frozen_from_n22_train_only_table",
        },
        "extraction": extraction,
        "live_reproduction": live_reproduction,
        "train_threshold_reproduction": train_reproduction,
        "n19_metrics": n19_metrics,
        "n6_provenance": n6_provenance,
        "n6_reference": n6_reference,
        "n6_unchanged_anchors": len(n6_anchors),
        "n6_metrics": n6_metrics,
        "n9_provenance": n9_provenance,
        "n9_reference": n9_reference,
        "n9_metrics": n9_metrics,
        "gates": gates,
        "offline_admission": offline_admission,
        "development_panel_authority": offline_admission,
        "deployment_authority": False,
        "official_match_evidence": False,
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "Held-out offline action audit with recorded autoregressive prefixes. "
            "Passage permits only runtime parity and a separately locked official panel."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "output_sha256": digest(OUTPUT),
                "teacher_validation": teacher_validation,
                "teacher_test": teacher_test,
                "retention_validation": retention_validation,
                "retention_test": retention_test,
                "n6_metrics": n6_metrics,
                "n9_metrics": n9_metrics,
                "gates": gates,
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
