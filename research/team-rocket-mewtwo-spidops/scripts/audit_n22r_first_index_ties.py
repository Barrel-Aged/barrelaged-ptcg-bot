#!/usr/bin/env python3
"""Run the train-only N22/N23 first-index tie correction audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "N22_N23_TIE_BREAK_CORRECTION_AMENDMENT.md"
N23_EVALUATOR = ROOT / "scripts/evaluate_n23_high_margin_gate.py"
INVALID_N23 = ROOT / "artifacts/team_rocket_recovery/n23_high_margin_gate/EVALUATION.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n22r_tie_corrected/TRAIN_ONLY_CORRECTION.json"

AMENDMENT_SHA = "99afca5165703ad1b60882486cba67021a776a160025a5a70305535f5c4d59c7"
N23_EVALUATOR_SHA = "cf8b082e63549c15508515ae690b28ea1f29062223fc8a1df950557bcdaec8ae"
INVALID_N23_SHA = "873efb37a8652ceb03935af461883bb9abc50f1bb360a9199d295011559966fb"


sys.path.insert(0, str((ROOT / "scripts").resolve()))
import evaluate_n23_high_margin_gate as base  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_index_top_and_margin(
    logits: np.ndarray, available: np.ndarray
) -> tuple[int, float]:
    indices = base.finite_available(logits, available)
    top = int(indices[int(np.argmax(logits[indices]))])
    others = indices[indices != top]
    runner = float(np.max(logits[others])) if len(others) else float(logits[top] - 50.0)
    return top, min(float(logits[top] - runner), 50.0)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N22R correction output: {OUTPUT}")
    for path, expected in (
        (AMENDMENT, AMENDMENT_SHA),
        (N23_EVALUATOR, N23_EVALUATOR_SHA),
        (INVALID_N23, INVALID_N23_SHA),
        (base.LIVE / "model.pt", base.LIVE_MODEL_SHA),
        (base.LIVE / "deck.csv", base.DECK_SHA),
        (base.CANDIDATE / "model.pt", base.CANDIDATE_MODEL_SHA),
        (base.CORPUS_AUDIT, base.CORPUS_AUDIT_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"N22R frozen input changed: {path}")
    for path, expected in base.SUPPORT_HASHES.items():
        if digest(path) != expected:
            raise ValueError(f"N22R support changed: {path}")

    started = time()
    torch.set_num_threads(base.THREADS)
    torch.manual_seed(822101)
    device = torch.device("cpu")
    audit = json.loads(base.CORPUS_AUDIT.read_text(encoding="utf-8"))
    train_records: dict[str, list[dict[str, Any]]] = {
        "teacher": [],
        "retention": [],
    }
    for record in audit["game_manifest"]:
        if record["split"] == "train":
            train_records[record["role"]].append(record)
    if {role: len(rows) for role, rows in train_records.items()} != {
        "teacher": 96,
        "retention": 96,
    }:
        raise ValueError("N22R train-only game manifest changed")

    sys.path.insert(0, str(base.LIVE.resolve()))
    from soldier import features  # type: ignore  # noqa: E402
    from soldier.belief import Belief  # type: ignore  # noqa: E402
    from soldier.net import load_policy  # type: ignore  # noqa: E402
    from audit_n19_retention_teacher_corpus import extract_examples  # noqa: E402

    deck_library = json.loads(
        (base.LIVE / "deck_library.json").read_text(encoding="utf-8")
    )
    live = load_policy(base.LIVE / "model.pt", map_location=device).cpu().eval()
    candidate = load_policy(
        base.CANDIDATE / "model.pt", map_location=device
    ).cpu().eval()
    base.top_and_margin = first_index_top_and_margin

    metrics: dict[str, dict[str, Any]] = {}
    extraction: dict[str, dict[str, int]] = {}
    live_reproduction: dict[str, bool] = {}
    for role in ("teacher", "retention"):
        games = [base.load_game(record) for record in train_records[role]]
        examples, counts = extract_examples(
            games, role, deck_library, Belief, features
        )
        if dict(counts) != audit["extraction"][role]["train"]:
            raise ValueError(f"N22R extraction changed: {role}")
        extraction[role] = dict(counts)
        metric = base.gated_n19_metrics(live, candidate, examples, device)
        metrics[role] = metric
        expected = audit["baseline_metrics"][role]["train"]["live"]
        live_reproduction[role] = (
            abs(metric["live_top1_reproduced"] - expected["top1"]) <= 1.0e-12
            and abs(metric["live_loss_reproduced"] - expected["loss"]) <= 1.0e-6
        )

    gates = {
        "live_train_metrics_reproduce": all(live_reproduction.values()),
        "teacher_candidate_only_ge_100": (
            metrics["teacher"]["replacement_labels"]["candidate_only"] >= 100
        ),
        "teacher_live_only_le_5": (
            metrics["teacher"]["replacement_labels"]["live_only"] <= 5
        ),
        "retention_train_zero_replacements": metrics["retention"]["replacements"] == 0,
    }
    correction_admission = all(gates.values())
    report = {
        "schema": "rocket-n22r-first-index-tie-correction/v1",
        "amendment_sha256": AMENDMENT_SHA,
        "audit_script_sha256": digest(Path(__file__).resolve()),
        "invalid_n23_evaluation_sha256": INVALID_N23_SHA,
        "live_model_sha256": base.LIVE_MODEL_SHA,
        "candidate_model_sha256": base.CANDIDATE_MODEL_SHA,
        "exact_team_rocket_deck_sha256": base.DECK_SHA,
        "candidate_margin_threshold": base.THRESHOLD,
        "tie_rule": "smallest legal index attaining maximum logit",
        "split_boundary": {
            "n19_train_only": True,
            "validation_examples_loaded": 0,
            "test_examples_loaded": 0,
        },
        "extraction": extraction,
        "metrics": metrics,
        "live_reproduction": live_reproduction,
        "gates": gates,
        "correction_admission": correction_admission,
        "development_authority": False,
        "deployment_authority": False,
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "Train-only tie-semantics correction. It authorizes at most one corrected "
            "held-out rerun with the unchanged threshold and gates."
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
                "metrics": metrics,
                "gates": gates,
                "correction_admission": correction_admission,
                "development_authority": False,
                "deployment_authority": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
