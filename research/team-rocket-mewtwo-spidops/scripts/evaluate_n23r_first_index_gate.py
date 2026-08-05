#!/usr/bin/env python3
"""Run the sole corrected N23 held-out gate evaluation."""

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
COUNT_CORRECTION = ROOT / "N22R_MANIFEST_COUNT_CORRECTION.md"
CORRECTION_AUDIT = ROOT / "artifacts/team_rocket_recovery/n22r_tie_corrected/TRAIN_ONLY_CORRECTION.json"
CORRECTION_SCRIPT = ROOT / "scripts/audit_n22r_first_index_ties.py"
ORIGINAL_EVALUATOR = ROOT / "scripts/evaluate_n23_high_margin_gate.py"
INVALID_EVALUATION = ROOT / "artifacts/team_rocket_recovery/n23_high_margin_gate/EVALUATION.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n23r_first_index_gate/EVALUATION_CORRECTED.json"

AMENDMENT_SHA = "99afca5165703ad1b60882486cba67021a776a160025a5a70305535f5c4d59c7"
COUNT_CORRECTION_SHA = "541c60f751e602c7b69c7af92499758990b4f15850b78d7fa315e522d0fa5d05"
CORRECTION_AUDIT_SHA = "4a33b1226829e760231c48477f227c8c78429b7d4d9cbc41e90377f4357238e4"
CORRECTION_SCRIPT_SHA = "7ca495f3fc6f2ed3da4d394791fa2b317b276853fbd6a85df83002538bd00146"
ORIGINAL_EVALUATOR_SHA = "cf8b082e63549c15508515ae690b28ea1f29062223fc8a1df950557bcdaec8ae"
INVALID_EVALUATION_SHA = "873efb37a8652ceb03935af461883bb9abc50f1bb360a9199d295011559966fb"


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
        raise FileExistsError(f"refusing existing N23R output: {OUTPUT}")
    frozen = {
        AMENDMENT: AMENDMENT_SHA,
        COUNT_CORRECTION: COUNT_CORRECTION_SHA,
        CORRECTION_AUDIT: CORRECTION_AUDIT_SHA,
        CORRECTION_SCRIPT: CORRECTION_SCRIPT_SHA,
        ORIGINAL_EVALUATOR: ORIGINAL_EVALUATOR_SHA,
        INVALID_EVALUATION: INVALID_EVALUATION_SHA,
        base.LIVE / "model.pt": base.LIVE_MODEL_SHA,
        base.LIVE / "deck.csv": base.DECK_SHA,
        base.CANDIDATE / "model.pt": base.CANDIDATE_MODEL_SHA,
        base.CORPUS_AUDIT: base.CORPUS_AUDIT_SHA,
        base.N21_RESULT: base.N21_RESULT_SHA,
        base.N22_AUDIT: base.N22_AUDIT_SHA,
        base.N22_RESULT: base.N22_RESULT_SHA,
        base.N22_SCRIPT: base.N22_SCRIPT_SHA,
        base.N17_TARGET_AUDIT: base.N17_TARGET_AUDIT_SHA,
        base.PROTOCOL: base.PROTOCOL_SHA,
        **base.SUPPORT_HASHES,
    }
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise ValueError(f"N23R frozen input changed: {path}")
    correction = json.loads(CORRECTION_AUDIT.read_text(encoding="utf-8"))
    if correction.get("correction_admission") is not True:
        raise ValueError("N23R requires the admitted train-only correction")

    started = time()
    torch.set_num_threads(base.THREADS)
    torch.manual_seed(823101)
    device = torch.device("cpu")
    audit = json.loads(base.CORPUS_AUDIT.read_text(encoding="utf-8"))
    base.top_and_margin = first_index_top_and_margin

    sys.path.insert(0, str(base.LIVE.resolve()))
    from soldier import features  # type: ignore  # noqa: E402
    from soldier.belief import Belief  # type: ignore  # noqa: E402
    from soldier.net import load_policy  # type: ignore  # noqa: E402
    from audit_n19_retention_teacher_corpus import extract_examples  # noqa: E402
    import train_n12_counterfactual_margin_adapter as support  # noqa: E402

    deck_library = json.loads(
        (base.LIVE / "deck_library.json").read_text(encoding="utf-8")
    )
    live = load_policy(base.LIVE / "model.pt", map_location=device).cpu().eval()
    candidate = load_policy(
        base.CANDIDATE / "model.pt", map_location=device
    ).cpu().eval()

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
            games = [base.load_game(record) for record in records[role][split]]
            examples, counts = extract_examples(
                games, role, deck_library, Belief, features
            )
            if dict(counts) != audit["extraction"][role][split]:
                raise ValueError(f"N23R extraction changed: {role}/{split}")
            extraction[role][split] = dict(counts)
            n19_metrics[role][split] = base.gated_n19_metrics(
                live, candidate, examples, device
            )

    teacher_train = n19_metrics["teacher"]["train"]
    retention_train = n19_metrics["retention"]["train"]
    train_reproduction = (
        teacher_train["replacements"] == 136
        and teacher_train["replacement_labels"]
        == {"candidate_only": 124, "live_only": 3, "neither": 9}
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
    target_audit = json.loads(base.N17_TARGET_AUDIT.read_text(encoding="utf-8"))
    excluded = {str(row["observation_fingerprint"]) for row in target_audit["targets"]}
    n6_anchors = [
        example
        for example in n6_examples
        if example.game.removeprefix("n6r:") not in excluded
    ]
    if len(n6_anchors) != 469 or len(n9_examples) != 6_753:
        raise ValueError("N23R historical preservation support changed")
    n6_metrics = base.gated_preservation_metrics(
        candidate, n6_anchors, support, device
    )
    n9_metrics = base.gated_preservation_metrics(
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
        "schema": "rocket-n23r-first-index-gate-evaluation/v1",
        "governing_protocol_sha256": base.PROTOCOL_SHA,
        "correction_amendment_sha256": AMENDMENT_SHA,
        "evaluator_sha256": digest(Path(__file__).resolve()),
        "quarantined_evaluation_sha256": INVALID_EVALUATION_SHA,
        "train_correction_audit_sha256": CORRECTION_AUDIT_SHA,
        "live_model_sha256": base.LIVE_MODEL_SHA,
        "candidate_model_sha256": base.CANDIDATE_MODEL_SHA,
        "exact_team_rocket_deck_sha256": base.DECK_SHA,
        "gate": {
            "candidate_margin_threshold": base.THRESHOLD,
            "replace_only_on_top_disagreement": True,
            "tie_rule": "smallest legal index attaining maximum logit",
            "selected_distribution": "complete_candidate_or_complete_live_logit_row",
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
            "Corrected held-out offline action audit using deployed first-index ties. "
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
