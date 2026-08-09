#!/usr/bin/env python3
"""Audit whether frozen N6R counterfactual roots transfer to the live policy."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
CORPUS = ROOT / "artifacts/team_rocket_recovery/n6r_exact_feature"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n12_counterfactual_transfer_audit.json"
MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N12 audit: {OUTPUT}")
    if digest(BASE / "model.pt") != MODEL_SHA or digest(BASE / "deck.csv") != DECK_SHA:
        raise ValueError("live N12 substrate changed")
    analyses = [
        json.loads((CORPUS / "train_analysis.json").read_text(encoding="utf-8")),
        json.loads((CORPUS / "selection_analysis.json").read_text(encoding="utf-8")),
    ]
    expected_inputs: dict[str, str] = {}
    estimates: dict[str, dict[str, Any]] = {}
    for analysis in analyses:
        expected_inputs.update({str(Path(path).resolve()): value for path, value in analysis["inputs"].items()})
        for root in analysis["roots"]:
            fingerprint = str(root["observation_fingerprint"])
            if fingerprint in estimates:
                raise ValueError(f"duplicate N6R root fingerprint: {fingerprint}")
            estimates[fingerprint] = root
    inputs = sorted(
        [*CORPUS.glob("train_*.jsonl"), *CORPUS.glob("selection_*.jsonl")],
        key=lambda path: path.name,
    )
    if len(inputs) != 14:
        raise ValueError(f"expected 14 N6R root files, found {len(inputs)}")
    rows: list[dict[str, Any]] = []
    input_hashes = {}
    for path in inputs:
        resolved = str(path.resolve())
        actual = digest(path)
        if expected_inputs.get(resolved) != actual:
            raise ValueError(f"N6R input changed: {path}")
        input_hashes[resolved] = actual
        rows.extend(read_jsonl(path))
    if len(rows) != 639 or len(estimates) != 639:
        raise ValueError(f"unexpected N6R root count: rows={len(rows)} estimates={len(estimates)}")

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(BASE))
    from rocketbot.policy_feature_snapshot import deserialize_policy_feature
    from soldier.net import load_policy  # type: ignore
    from train_belief_rollout_distillation_n6 import apply_attack_guard
    from train_public_leader_v110_bc import Example, collate

    torch.manual_seed(812001)
    torch.set_num_threads(1)
    model = load_policy(BASE / "model.pt", map_location="cpu").cpu().eval()
    totals: Counter[str] = Counter()
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_value_examples = []
    current_actionable = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            fingerprint = str(row["observation_fingerprint"])
            estimate = estimates[fingerprint]
            feature = deserialize_policy_feature(row["policy_feature"])
            option_count = int(feature["opt_feats"].shape[0])
            example = Example(feature, int(row["baseline"]), f"audit={index}")
            batch = collate([example], torch.device("cpu"))
            logits, _ = model.forward_batch(batch)
            logits = apply_attack_guard(logits[0, :option_count], row["observation"])
            current = int(logits.argmax())
            old_baseline = int(row["baseline"])
            candidates = [int(value) for value in row["candidate_indices"]]
            best = int(estimate["best"])
            admissible = bool(estimate["admissible_change"])
            opponent = str(row["opponent"])
            turn = int(row["turn"])
            groups = ("all", opponent, f"turn_{turn}", f"{opponent}:turn_{turn}")

            flags = {
                "roots": 1,
                "current_equals_old_baseline": int(current == old_baseline),
                "current_in_shortlist": int(current in candidates),
                "current_equals_teacher_best": int(current == best),
                "old_admissible_change": int(admissible),
                "admissible_current_already_best": int(admissible and current == best),
                "admissible_new_correction": int(admissible and current != best),
                "admissible_new_correction_covered": int(admissible and current != best and current in candidates),
            }
            for key, value in flags.items():
                totals[key] += value
                for group in groups:
                    grouped[group][key] += value
            if current not in candidates and len(unknown_value_examples) < 24:
                unknown_value_examples.append({
                    "fingerprint": fingerprint,
                    "opponent": opponent,
                    "turn": turn,
                    "current": current,
                    "old_baseline": old_baseline,
                    "shortlist": candidates,
                    "old_admissible_change": admissible,
                })
            if admissible and current != best and current in candidates:
                actions = {int(item["action"]): item for item in estimate["actions"]}
                gain = float(actions[best]["mean"] - actions[current]["mean"])
                current_actionable.append({
                    "fingerprint": fingerprint,
                    "opponent": opponent,
                    "turn": turn,
                    "current": current,
                    "best": best,
                    "estimated_gain": gain,
                    "best_lower_bound": float(actions[best]["clustered_interval_95"][0]),
                })

    def summarize(counter: Counter[str]) -> dict[str, Any]:
        roots = max(1, counter["roots"])
        return {
            **dict(counter),
            "current_old_baseline_agreement_rate": counter["current_equals_old_baseline"] / roots,
            "current_shortlist_coverage_rate": counter["current_in_shortlist"] / roots,
            "current_teacher_best_rate": counter["current_equals_teacher_best"] / roots,
            "old_admissible_change_rate": counter["old_admissible_change"] / roots,
        }

    report = {
        "schema": "rocket-n12-counterfactual-transfer-audit/v1",
        "live_model_sha256": MODEL_SHA,
        "deck_sha256": DECK_SHA,
        "input_sha256": input_hashes,
        "roots": len(rows),
        "official_descendant_execution_in_source": True,
        "source_policy_differs_from_live_policy": True,
        "aggregate": summarize(totals),
        "by_group": {name: summarize(counter) for name, counter in sorted(grouped.items())},
        "actionable_corrections": sorted(current_actionable, key=lambda item: (-item["estimated_gain"], item["fingerprint"])),
        "unknown_current_value_examples": unknown_value_examples,
        "claims": {
            "training_authority": False,
            "deployment_authority": False,
            "strength_improvement": False,
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "aggregate": report["aggregate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
