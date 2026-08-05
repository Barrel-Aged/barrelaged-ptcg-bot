#!/usr/bin/env python3
"""Freeze the control and preselected N9 epoch-4 arms for N10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
N9_ARMS = ROOT / "artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_arms/arms.json"
N9_ANALYSIS = ROOT / "artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_panel/analysis.json"
RUNTIME_PARITY = ROOT / "artifacts/team_rocket_recovery/n9_balanced_pcgrad_awr/development_arms/runtime_parity.json"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n10_epoch4_confirmation/arms.json"

N9_ARMS_SHA = "eb6efcd1a839ab494ccd6afd6a1e7854dec67bed0717ebaf6037c68f40824b23"
N9_ANALYSIS_SHA = "b0bc2e5f6e063476101e68281162fc6ff7f6d1c40c4065b583f6e20f5c4d4522"
RUNTIME_PARITY_SHA = "6c11d589ad7d7cb1235734a1ecf87f00f40159a31e2a90ece5f57900af04e535"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "bdafc1ce36f7e572554930a1a9b1beff541a351ece09f934d0416b4ce269a020"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    for path, expected in (
        (N9_ARMS, N9_ARMS_SHA),
        (N9_ANALYSIS, N9_ANALYSIS_SHA),
        (RUNTIME_PARITY, RUNTIME_PARITY_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"frozen N10 source changed: {path}")
    source = json.loads(N9_ARMS.read_text(encoding="utf-8"))
    source_by_name = {str(item["name"]): item for item in source["arms"]}
    selected = []
    for output_name, source_name, expected_sha in (
        ("control", "control", CONTROL_SHA),
        ("candidate", "epoch_04", CANDIDATE_SHA),
    ):
        item = dict(source_by_name[source_name])
        model_path = Path(str(item["path"])).resolve()
        if digest(model_path) != expected_sha or item.get("sha256") != expected_sha:
            raise ValueError(f"N10 model changed: {source_name}")
        item["name"] = output_name
        item["source_name"] = source_name
        selected.append(item)
    parity = json.loads(RUNTIME_PARITY.read_text(encoding="utf-8"))
    for name in ("control", "epoch_04"):
        row = parity["arms"][name]
        if int(row["mismatched_decisions"]) != 0 or int(row["matched_decisions"]) <= 0:
            raise ValueError(f"N10 source lacks exact runtime parity: {name}")
    report = {
        "schema": "rocket-n10-confirmation-arms/v1",
        "protocol": "rocket-n10-independent-epoch4-confirmation/v1",
        "deck_sha256": DECK_SHA,
        "initial_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "source_n9_arms_sha256": N9_ARMS_SHA,
        "source_n9_analysis_sha256": N9_ANALYSIS_SHA,
        "runtime_parity_sha256": RUNTIME_PARITY_SHA,
        "runtime_parity": {
            "control": parity["arms"]["control"],
            "candidate": parity["arms"]["epoch_04"],
        },
        "arms": selected,
        "confirmation_only": True,
        "deployment_authority": False,
    }
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N10 arms: {OUTPUT}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sha256": digest(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

