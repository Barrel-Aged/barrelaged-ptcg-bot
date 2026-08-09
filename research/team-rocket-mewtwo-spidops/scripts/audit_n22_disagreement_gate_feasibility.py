#!/usr/bin/env python3
"""Audit N21/live disagreements using training-only evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "work/n18/blive"
CANDIDATE = (
    ROOT
    / "artifacts/team_rocket_recovery/n21_top_action_margin/training_v1"
)
CORPUS_AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n19_retention_teacher/N19_CORPUS_AUDIT.json"
)
N21_REPORT = CANDIDATE / "N21_TRAINING_REPORT.json"
N21_RESULT = ROOT / "N21_TOP_ACTION_MARGIN_BC_RESULT.md"
PROTOCOL = ROOT / "N22_TRAIN_ONLY_DISAGREEMENT_AUDIT_PROTOCOL.md"
OUTPUT = (
    ROOT
    / "artifacts/team_rocket_recovery/n22_disagreement_gate/"
    "TRAIN_ONLY_AUDIT.json"
)

LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_MODEL_SHA = "7e98ef2eb0933b1444183f8b9238344df3b17220af638a4b2045e5c6f5aeba2b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CORPUS_AUDIT_SHA = "d4c72d91e783f6fcbf1fe0ec7082d1df6c5d96e50f29f1820a61c5418cbb7aed"
N21_REPORT_SHA = "8f51c4ad18558ef7af48ca7a08fd9eb2e059a7e5f94290b0eb0a1f345357a0f6"
N21_RESULT_SHA = "53683ec3d968e59d4343c5b7294aec00d46b2e8d0a49ed7a6d54d6f4890b1e98"
PROTOCOL_SHA = "337d2a37dda170e0c856532a398bf7d26ad7be4e7d94a3adee31ece11edf22d3"

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

BATCH_SIZE = 96
THREADS = 6
N_OPTION_TYPE = 17
N_SELECT_TYPE = 11
N_SELECT_CONTEXT = 49
CANDIDATE_MARGIN_THRESHOLDS = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0)
LIVE_MARGIN_MAXIMUMS = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
MARGIN_DELTA_THRESHOLDS = (-0.50, 0.0, 0.25, 0.50, 1.0)


sys.path.insert(0, str((ROOT / "scripts").resolve()))
from train_public_leader_v110_bc import Example, collate  # noqa: E402
from train_n19_retention_constrained_bc import load_game  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "p10": None, "p25": None,
                "median": None, "p75": None, "p90": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(array.max()),
    }


def top_margin(logits: np.ndarray) -> tuple[int, float]:
    available = np.flatnonzero(np.isfinite(logits) & (logits > -1.0e8))
    if not len(available):
        raise ValueError("N22 scored a decision with no available action")
    order = available[np.argsort(logits[available])]
    top = int(order[-1])
    runner = float(logits[order[-2]]) if len(order) > 1 else float(logits[top] - 50.0)
    return top, min(float(logits[top] - runner), 50.0)


def public_keys(feat: dict[str, np.ndarray], live_top: int, candidate_top: int,
                rocket_cards: set[int]) -> dict[str, str]:
    glob = feat["global"]
    selection_type = int(np.argmax(glob[13:13 + N_SELECT_TYPE]))
    context_start = 13 + N_SELECT_TYPE
    selection_context = int(
        np.argmax(glob[context_start:context_start + N_SELECT_CONTEXT])
    )
    live_option_type = int(np.argmax(feat["opt_feats"][live_top, :N_OPTION_TYPE]))
    candidate_option_type = int(
        np.argmax(feat["opt_feats"][candidate_top, :N_OPTION_TYPE])
    )
    live_card = int(feat["opt_card"][live_top])
    candidate_card = int(feat["opt_card"][candidate_top])
    return {
        "selection_type": str(selection_type),
        "selection_context": str(selection_context),
        "option_type_pair": f"{live_option_type}->{candidate_option_type}",
        "candidate_card_in_rocket": str(candidate_card in rocket_cards).lower(),
        "live_card_in_rocket": str(live_card in rocket_cards).lower(),
        "card_overlap_pair": (
            f"{str(live_card in rocket_cards).lower()}->"
            f"{str(candidate_card in rocket_cards).lower()}"
        ),
    }


def make_row(*, role: str, group: str, feat: dict[str, np.ndarray], action: int,
             live_probabilities: np.ndarray, candidate_logits: np.ndarray,
             rocket_cards: set[int]) -> dict[str, Any]:
    live_top, live_margin = top_margin(np.log(live_probabilities.clip(1.0e-30)))
    candidate_top, candidate_margin = top_margin(candidate_logits)
    shifted = candidate_logits - float(np.max(candidate_logits))
    candidate_probabilities = np.exp(shifted)
    candidate_probabilities /= candidate_probabilities.sum()
    keys = public_keys(feat, live_top, candidate_top, rocket_cards)
    if live_top == candidate_top:
        label = "agreement"
    elif candidate_top == action and live_top != action:
        label = "candidate_only"
    elif live_top == action and candidate_top != action:
        label = "live_only"
    elif live_top != action and candidate_top != action:
        label = "neither"
    else:
        label = "both"
    return {
        "role": role,
        "group": group,
        "label": label,
        "action": int(action),
        "live_top": live_top,
        "candidate_top": candidate_top,
        "live_margin": live_margin,
        "candidate_margin": candidate_margin,
        "margin_delta": candidate_margin - live_margin,
        "recorded_probability_delta": (
            float(candidate_probabilities[action]) - float(live_probabilities[action])
        ),
        "keys": keys,
    }


@torch.no_grad()
def score_n19(role: str, examples: list[Example], live: torch.nn.Module,
              candidate: torch.nn.Module, rocket_cards: set[int],
              device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), BATCH_SIZE):
        selected = examples[start:start + BATCH_SIZE]
        batch = collate(selected, device)
        live_logits, _ = live.forward_batch(batch)
        candidate_logits, _ = candidate.forward_batch(batch)
        for index, example in enumerate(selected):
            width = int(example.feat["opt_feats"].shape[0])
            live_row = live_logits[index, :width].cpu().numpy().astype(np.float64)
            candidate_row = (
                candidate_logits[index, :width].cpu().numpy().astype(np.float64)
            )
            live_shifted = live_row - float(np.max(live_row))
            live_probabilities = np.exp(live_shifted)
            live_probabilities /= live_probabilities.sum()
            rows.append(
                make_row(
                    role=role,
                    group=role,
                    feat=example.feat,
                    action=int(example.action),
                    live_probabilities=live_probabilities,
                    candidate_logits=candidate_row,
                    rocket_cards=rocket_cards,
                )
            )
    return rows


@torch.no_grad()
def score_n9(examples: Sequence[Any], candidate: torch.nn.Module, support: Any,
             rocket_cards: set[int], device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), 128):
        selected = list(examples[start:start + 128])
        candidate_logits = support.full_forward_deployed(candidate, selected, device)
        for index, example in enumerate(selected):
            width = int(example.feat["opt_feats"].shape[0])
            reference = np.asarray(
                example.reference_probabilities[:width], dtype=np.float64
            )
            reference /= reference.sum()
            candidate_row = (
                candidate_logits[index, :width].cpu().numpy().astype(np.float64)
            )
            rows.append(
                make_row(
                    role="n9",
                    group=str(example.group),
                    feat=example.feat,
                    action=int(example.action),
                    live_probabilities=reference,
                    candidate_logits=candidate_row,
                    rocket_cards=rocket_cards,
                )
            )
    return rows


def label_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    disagreements = [row for row in rows if row["label"] != "agreement"]
    return {
        "examples": len(rows),
        "agreements": int(labels["agreement"]),
        "disagreements": len(disagreements),
        "disagreement_rate": len(disagreements) / max(len(rows), 1),
        "labels": {key: int(value) for key, value in sorted(labels.items())},
        "candidate_minus_live_recorded_top1": (
            (labels["candidate_only"] - labels["live_only"]) / max(len(rows), 1)
        ),
        "live_margin_on_disagreements": quantiles(
            row["live_margin"] for row in disagreements
        ),
        "candidate_margin_on_disagreements": quantiles(
            row["candidate_margin"] for row in disagreements
        ),
        "margin_delta_on_disagreements": quantiles(
            row["margin_delta"] for row in disagreements
        ),
        "recorded_probability_delta_on_disagreements": quantiles(
            row["recorded_probability_delta"] for row in disagreements
        ),
    }


def selected_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    evidence = labels["candidate_only"] + labels["live_only"]
    return {
        "examples": len(rows),
        "candidate_only": int(labels["candidate_only"]),
        "live_only": int(labels["live_only"]),
        "neither": int(labels["neither"]),
        "candidate_precision_vs_live_only": (
            labels["candidate_only"] / evidence if evidence else None
        ),
        "net_recorded_corrections": int(
            labels["candidate_only"] - labels["live_only"]
        ),
    }


def threshold_tables(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    disagreements = [row for row in rows if row["label"] != "agreement"]
    return {
        "candidate_margin_min": {
            f"{threshold:.2f}": selected_stats(
                [row for row in disagreements if row["candidate_margin"] >= threshold]
            )
            for threshold in CANDIDATE_MARGIN_THRESHOLDS
        },
        "live_margin_max": {
            f"{threshold:.2f}": selected_stats(
                [row for row in disagreements if row["live_margin"] <= threshold]
            )
            for threshold in LIVE_MARGIN_MAXIMUMS
        },
        "margin_delta_min": {
            f"{threshold:.2f}": selected_stats(
                [row for row in disagreements if row["margin_delta"] >= threshold]
            )
            for threshold in MARGIN_DELTA_THRESHOLDS
        },
    }


def bucket_tables(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    disagreements = [row for row in rows if row["label"] != "agreement"]
    families: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in disagreements:
        for family, key in row["keys"].items():
            families[family][key].append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for family, buckets in sorted(families.items()):
        values = []
        for key, selected in buckets.items():
            values.append({"key": key, **selected_stats(selected)})
        result[family] = sorted(
            values,
            key=lambda item: (
                -item["candidate_only"], -item["examples"], item["key"]
            ),
        )
    return result


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N22 audit: {OUTPUT}")
    frozen = {
        LIVE / "model.pt": LIVE_MODEL_SHA,
        LIVE / "deck.csv": DECK_SHA,
        CANDIDATE / "model.pt": CANDIDATE_MODEL_SHA,
        CORPUS_AUDIT: CORPUS_AUDIT_SHA,
        N21_REPORT: N21_REPORT_SHA,
        N21_RESULT: N21_RESULT_SHA,
        PROTOCOL: PROTOCOL_SHA,
        **SUPPORT_HASHES,
    }
    for path, expected in frozen.items():
        if digest(path) != expected:
            raise ValueError(f"N22 frozen input changed: {path}")

    started = time()
    torch.set_num_threads(THREADS)
    torch.manual_seed(822001)
    device = torch.device("cpu")
    audit = json.loads(CORPUS_AUDIT.read_text(encoding="utf-8"))
    n21 = json.loads(N21_REPORT.read_text(encoding="utf-8"))
    if n21.get("offline_admission") is not False:
        raise ValueError("N22 requires the rejected N21 terminal")

    train_records = [
        record for record in audit["game_manifest"] if record["split"] == "train"
    ]
    if any(record["split"] != "train" for record in train_records):
        raise ValueError("N22 split isolation failed")
    expected_records = {
        "teacher": audit["selection"]["teacher_split_counts_per_seat"]["train"] * 2,
        "retention": (
            audit["selection"]["retention_split_counts_per_seat_outcome"]["train"]
            * 4
        ),
    }
    records_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        records_by_role[record["role"]].append(record)
    if {role: len(rows) for role, rows in records_by_role.items()} != expected_records:
        raise ValueError("N22 training game manifest changed")

    sys.path.insert(0, str(LIVE.resolve()))
    from soldier import features  # type: ignore  # noqa: E402
    from soldier.belief import Belief  # type: ignore  # noqa: E402
    from soldier.net import load_policy  # type: ignore  # noqa: E402
    from audit_n19_retention_teacher_corpus import extract_examples  # noqa: E402
    import train_n12_counterfactual_margin_adapter as support  # noqa: E402

    deck_library = json.loads((LIVE / "deck_library.json").read_text(encoding="utf-8"))
    rocket_cards = {
        int(value)
        for value in (LIVE / "deck.csv").read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    live = load_policy(LIVE / "model.pt", map_location=device).cpu().eval()
    candidate = load_policy(CANDIDATE / "model.pt", map_location=device).cpu().eval()

    n19_rows: dict[str, list[dict[str, Any]]] = {}
    extraction: dict[str, dict[str, int]] = {}
    for role in ("teacher", "retention"):
        games = [load_game(record) for record in records_by_role[role]]
        examples, counts = extract_examples(
            games, role, deck_library, Belief, features
        )
        if dict(counts) != audit["extraction"][role]["train"]:
            raise ValueError(f"N22 N19 train extraction changed: {role}")
        extraction[role] = dict(counts)
        n19_rows[role] = score_n19(
            role, examples, live, candidate, rocket_cards, device
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
    n9_reference = support.attach_reference(live, n9_examples)
    if len(n9_examples) != 6_753:
        raise ValueError("N22 N9 support changed")
    n9_rows = score_n9(n9_examples, candidate, support, rocket_cards, device)

    all_rows = {
        "teacher_train": n19_rows["teacher"],
        "retention_train": n19_rows["retention"],
        "n9_historical": n9_rows,
    }
    summaries = {name: label_summary(rows) for name, rows in all_rows.items()}
    thresholds = {name: threshold_tables(rows) for name, rows in all_rows.items()}
    buckets = {name: bucket_tables(rows) for name, rows in all_rows.items()}

    exact_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in n19_rows["retention"] + n9_rows:
        if row["label"] == "agreement":
            continue
        for family, key in row["keys"].items():
            exact_counts[family][key] += 1
    eligible_buckets: list[dict[str, Any]] = []
    for family, values in buckets["teacher_train"].items():
        for value in values:
            precision = value["candidate_precision_vs_live_only"]
            overlap = int(exact_counts[family][value["key"]])
            if (
                value["candidate_only"] >= 25
                and precision is not None
                and precision >= 0.70
                and overlap > 0
            ):
                eligible_buckets.append(
                    {
                        "family": family,
                        **value,
                        "exact_rocket_disagreement_representation": overlap,
                    }
                )
    feasible = (
        summaries["teacher_train"]["labels"].get("candidate_only", 0) >= 100
        and bool(eligible_buckets)
    )
    n9_by_group = {
        group: label_summary([row for row in n9_rows if row["group"] == group])
        for group in sorted({row["group"] for row in n9_rows})
    }

    report = {
        "schema": "rocket-n22-train-only-disagreement-audit/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "audit_script_sha256": digest(Path(__file__).resolve()),
        "live_model_sha256": LIVE_MODEL_SHA,
        "candidate_model_sha256": CANDIDATE_MODEL_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "n19_corpus_audit_sha256": CORPUS_AUDIT_SHA,
        "n21_training_report_sha256": N21_REPORT_SHA,
        "n21_result_sha256": N21_RESULT_SHA,
        "split_boundary": {
            "n19_train_only": True,
            "n19_validation_examples_loaded": 0,
            "n19_test_examples_loaded": 0,
            "historical_n9_used_as_preservation_evidence": True,
        },
        "n19_train_game_counts": {
            role: len(rows) for role, rows in sorted(records_by_role.items())
        },
        "n19_extraction": extraction,
        "n9_provenance": n9_provenance,
        "n9_reference": n9_reference,
        "summaries": summaries,
        "n9_by_group": n9_by_group,
        "threshold_tables": thresholds,
        "bucket_tables": buckets,
        "eligible_public_buckets": eligible_buckets,
        "feasibility_gates": {
            "teacher_candidate_only_corrections_ge_100": (
                summaries["teacher_train"]["labels"].get("candidate_only", 0) >= 100
            ),
            "eligible_public_bucket_exists": bool(eligible_buckets),
        },
        "state_conditioned_gate_feasible": feasible,
        "development_authority": False,
        "deployment_authority": False,
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "Training-only disagreement feasibility audit. No held-out N19 state, "
            "official match, counterfactual value, strength, or deployment evidence."
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
                "summaries": summaries,
                "eligible_public_buckets": eligible_buckets,
                "state_conditioned_gate_feasible": feasible,
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
