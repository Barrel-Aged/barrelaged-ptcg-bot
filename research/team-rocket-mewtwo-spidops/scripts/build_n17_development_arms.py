#!/usr/bin/env python3
"""Build the sole N17 candidate bundle and development arm manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
TRAINING = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/training_v1"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/development_arms"
PROTOCOL = ROOT / "N17_REPLICATED_TARGET_RESIDUAL_PROTOCOL.md"
OFFLINE_RESULT = ROOT / "N17_REPLICATED_TARGET_RESIDUAL_OFFLINE_RESULT.md"

DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_SHA = "381d6cb87a51fa1c8061f5e3ae441e446b0331bdd9c1f99e9118d4227e038772"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
REPORT_SHA = "50aaac5e3531699489133790bdcbf0d566aea7acec23775ddd5b91d144fc02dd"
TRAINING_PROVENANCE_SHA = "1b14b12eef429f5019816d905dbd9c3e4c8d0e1da31265af8832b8423fca5789"
OFFLINE_RESULT_SHA = "0ffc6b8ecc6872059835552ff497433b663f8b63a9ee51abefec39389206413c"
PROTOCOL_SHA = "0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    result = hashlib.sha256()
    files = sorted(
        (
            path for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix.lower() not in (".pyc", ".pyo")
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        result.update(len(relative).to_bytes(4, "big"))
        result.update(relative)
        result.update(path.read_bytes())
    return result.hexdigest()


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N17 arm builder accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N17 arms: {OUTPUT}")
    checks = {
        LIVE / "model.pt": CONTROL_SHA,
        LIVE / "deck.csv": DECK_SHA,
        TRAINING / "model.pt": CANDIDATE_SHA,
        TRAINING / "deck.csv": DECK_SHA,
        TRAINING / "soldier/net_v10.py": RUNTIME_SHA,
        TRAINING / "N17_TRAINING_REPORT.json": REPORT_SHA,
        TRAINING / "N17_TRAINING_PROVENANCE.json": TRAINING_PROVENANCE_SHA,
        PROTOCOL: PROTOCOL_SHA,
        OFFLINE_RESULT: OFFLINE_RESULT_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N17 frozen development-build input changed: {path}")
    report = json.loads((TRAINING / "N17_TRAINING_REPORT.json").read_text(encoding="utf-8"))
    gate = report["terminal_evaluation"]["admission"]
    if not gate["passes"] or not all(gate["gates"].values()):
        raise ValueError("N17 training lacks complete offline admission")
    if (
        report.get("terminal_model_sha256") != CANDIDATE_SHA
        or report.get("claims", {}).get("development_panel_authority") is not True
        or report.get("claims", {}).get("deployment_authority") is not False
    ):
        raise ValueError("N17 training authority boundary changed")

    bundle = OUTPUT / "candidate_bundle"
    shutil.copytree(
        TRAINING,
        bundle,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )
    provenance = {
        "schema": "rocket-n17-candidate-bundle/v1",
        "protocol": "rocket-n17-replicated-target-residual/v1",
        "training_report_sha256": REPORT_SHA,
        "training_provenance_sha256": TRAINING_PROVENANCE_SHA,
        "offline_result_sha256": OFFLINE_RESULT_SHA,
        "model_sha256": CANDIDATE_SHA,
        "deck_sha256": DECK_SHA,
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "offline_admission": True,
        "official_match_evidence": False,
        "deployment_authority": False,
    }
    (bundle / "N17_DEVELOPMENT_BUNDLE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if digest(bundle / "model.pt") != CANDIDATE_SHA:
        raise AssertionError("N17 bundled candidate model changed")

    manifest_data = {
        "schema": "rocket-n17-development-arms/v1",
        "protocol": "rocket-n17-replicated-target-residual/v1",
        "deck_sha256": DECK_SHA,
        "control_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "candidate_bundle": str(bundle.resolve()),
        "candidate_bundle_sha256": tree_digest(bundle),
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "training_report_sha256": REPORT_SHA,
        "arms": [
            {
                "name": "control",
                "path": str((LIVE / "model.pt").resolve()),
                "sha256": CONTROL_SHA,
            },
            {
                "name": "candidate",
                "path": str((bundle / "model.pt").resolve()),
                "sha256": CANDIDATE_SHA,
            },
        ],
        "development_only": True,
        "deployment_authority": False,
    }
    manifest = OUTPUT / "arms.json"
    manifest.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "arms": str(manifest),
                "arms_sha256": digest(manifest),
                "bundle": str(bundle),
                "bundle_sha256": tree_digest(bundle),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
