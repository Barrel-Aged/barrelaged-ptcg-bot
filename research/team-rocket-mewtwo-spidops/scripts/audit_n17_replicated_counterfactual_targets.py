#!/usr/bin/env python3
"""Freeze the broader N17 replicated-half counterfactual target set."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/team_rocket_recovery/n6r_exact_feature"
TRAIN = CORPUS / "train_analysis.json"
SELECTION = CORPUS / "selection_analysis.json"
TRANSFER = ROOT / "artifacts/team_rocket_recovery/n12_counterfactual_transfer_audit.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/target_audit.json"

TRAIN_SHA = "a9b0dc1fd9414b63e327b35616455f7ff699dc69ea1833956add461aee8c3049"
SELECTION_SHA = "b13e8d804b466230d7019f86d018e9db1ef9d0de521ce71565804a0988c9acbd"
TRANSFER_SHA = "9ef9429934d7d94621406181f61c2a97d57b3409296235373f027efc9e8ed313"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualifies(root: dict[str, Any]) -> tuple[bool, bool, bool]:
    strict = bool(root["admissible_change"])
    best = int(root["best"])
    baseline = int(root["baseline"])
    halves = tuple(int(value) for value in root["half_best"])
    replicated = best != baseline and halves == (best, best)
    return strict or replicated, strict, replicated


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N17 target audit accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N17 target audit: {OUTPUT}")
    for path, expected in ((TRAIN, TRAIN_SHA), (SELECTION, SELECTION_SHA), (TRANSFER, TRANSFER_SHA)):
        if digest(path) != expected:
            raise ValueError(f"N17 frozen audit input changed: {path}")
    transfer = json.loads(TRANSFER.read_text(encoding="utf-8"))
    if transfer.get("deck_sha256") != DECK_SHA or transfer.get("live_model_sha256") != LIVE_MODEL_SHA:
        raise ValueError("N17 transfer substrate changed")

    roots = []
    for split, path in (("train", TRAIN), ("selection", SELECTION)):
        analysis = json.loads(path.read_text(encoding="utf-8"))
        if not analysis["claims"]["official_descendant_execution"]:
            raise ValueError("N17 source is not official-descendant evidence")
        for raw in analysis["roots"]:
            row = dict(raw)
            row["split"] = split
            roots.append(row)
    fingerprints = [str(root["observation_fingerprint"]) for root in roots]
    if len(roots) != 639 or len(set(fingerprints)) != 639:
        raise ValueError("N17 root count or uniqueness changed")

    targets = []
    strict_count = replicated_count = overlap = 0
    for root in roots:
        include, strict, replicated = qualifies(root)
        strict_count += int(strict)
        replicated_count += int(replicated)
        overlap += int(strict and replicated)
        if not include:
            continue
        best = int(root["best"])
        best_row = next(action for action in root["actions"] if int(action["action"]) == best)
        targets.append(
            {
                "observation_fingerprint": str(root["observation_fingerprint"]),
                "split": str(root["split"]),
                "source": str(root["source"]),
                "opponent": str(root["opponent"]),
                "turn": int(root["turn"]),
                "baseline": int(root["baseline"]),
                "teacher_best": best,
                "strict_admissible": strict,
                "replicated_half_best": replicated,
                "half_best": [int(value) for value in root["half_best"]],
                "half_paired_differences": [
                    float(value) for value in root["half_paired_differences"]
                ],
                "oracle_gain": float(root["oracle_gain"]),
                "best_paired_difference": float(best_row["paired_difference"]),
                "best_clustered_interval_95": [
                    float(value) for value in best_row["clustered_interval_95"]
                ],
                "complete_pairs": int(root["complete_pairs"]),
                "complete_clusters": int(root["complete_clusters"]),
            }
        )
    if strict_count != 48 or replicated_count != 160 or overlap != 38 or len(targets) != 170:
        raise ValueError(
            "N17 target counts changed: "
            f"strict={strict_count}, replicated={replicated_count}, overlap={overlap}, union={len(targets)}"
        )
    target_fingerprints = {row["observation_fingerprint"] for row in targets}
    transfer_strict = {
        str(row["fingerprint"])
        for row in transfer["actionable_corrections"]
    }
    strict_fingerprints = {
        str(root["observation_fingerprint"])
        for root in roots if bool(root["admissible_change"])
    }
    if strict_fingerprints != transfer_strict or not strict_fingerprints <= target_fingerprints:
        raise ValueError("N17 strict target identity changed from transfer audit")

    by_opponent = Counter(row["opponent"] for row in targets)
    by_turn = Counter(str(row["turn"]) for row in targets)
    by_cell = Counter(f"{row['opponent']}:turn_{row['turn']}" for row in targets)
    by_split = Counter(row["split"] for row in targets)
    report = {
        "schema": "rocket-n17-replicated-counterfactual-target-audit/v1",
        "criterion": (
            "strict_admissible_change OR "
            "(teacher_best != baseline AND half_best[0] == half_best[1] == teacher_best)"
        ),
        "input_sha256": {
            str(TRAIN.resolve()): TRAIN_SHA,
            str(SELECTION.resolve()): SELECTION_SHA,
            str(TRANSFER.resolve()): TRANSFER_SHA,
        },
        "deck_sha256": DECK_SHA,
        "live_model_sha256": LIVE_MODEL_SHA,
        "source_roots": len(roots),
        "strict_targets": strict_count,
        "replicated_half_targets": replicated_count,
        "strict_replicated_overlap": overlap,
        "union_targets": len(targets),
        "non_target_anchors": len(roots) - len(targets),
        "by_opponent": dict(sorted(by_opponent.items())),
        "by_turn": dict(sorted(by_turn.items())),
        "by_cell": dict(sorted(by_cell.items())),
        "by_split": dict(sorted(by_split.items())),
        "mean_oracle_gain": sum(row["oracle_gain"] for row in targets) / len(targets),
        "minimum_complete_pairs": min(row["complete_pairs"] for row in targets),
        "minimum_complete_clusters": min(row["complete_clusters"] for row in targets),
        "targets": sorted(targets, key=lambda row: row["observation_fingerprint"]),
        "claims": {
            "official_descendant_source": True,
            "hidden_completion_prior_official": False,
            "strength_improvement": False,
            "training_authority": False,
            "deployment_authority": False,
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": digest(OUTPUT),
                "targets": len(targets),
                "by_opponent": dict(by_opponent),
                "by_turn": dict(by_turn),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
