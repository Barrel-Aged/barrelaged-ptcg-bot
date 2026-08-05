#!/usr/bin/env python3
"""Build the sole N25 candidate bundle and frozen two-arm manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
BUILD = ROOT / "artifacts/team_rocket_recovery/n25_complementary_safe_composition/build_v1"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n25_complementary_safe_composition/development_arms"
PROTOCOL = ROOT / "N25_COMPLEMENTARY_SAFE_COMPOSITION_PROTOCOL.md"
CORRECTION = ROOT / "N25_SOURCE_BASE_CORRECTION.md"

DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
CANDIDATE_SHA = "e1d6567dfefee9c7b408b9438153e75e75f613864d7c0a2da021f7f5c63ebf04"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
REPORT_SHA = "b94d281479b4be1cb29b7352f27c3b91ec36f7503edd3f5b3a5ede556b7a81ce"
PROTOCOL_SHA = "82ba2a57736e396ecda5c2fdcda1a9577d68be25de88f4c2ddb4f098d86e78a0"
CORRECTION_SHA = "aadd14c675e44a641b9cb0022c111aa578e516206b4f1b556fc918ff5a3d9e33"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    result = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
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
        raise ValueError("N25 arm builder accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N25 arms: {OUTPUT}")
    checks = {
        LIVE / "model.pt": CONTROL_SHA,
        LIVE / "deck.csv": DECK_SHA,
        ZERO / "model.pt": ZERO_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": RUNTIME_SHA,
        BUILD / "model.pt": CANDIDATE_SHA,
        BUILD / "build_report.json": REPORT_SHA,
        PROTOCOL: PROTOCOL_SHA,
        CORRECTION: CORRECTION_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N25 frozen build input changed: {path}")
    report = json.loads((BUILD / "build_report.json").read_text(encoding="utf-8"))
    if (
        report.get("offline_admission") is not True
        or report.get("model_sha256") != CANDIDATE_SHA
        or report.get("source_base_correction_sha256") != CORRECTION_SHA
        or not all(report.get("gates", {}).values())
    ):
        raise ValueError("N25 candidate lacks exact offline admission")

    bundle = OUTPUT / "candidate_bundle"
    shutil.copytree(
        ZERO,
        bundle,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )
    shutil.copy2(BUILD / "model.pt", bundle / "model.pt")
    provenance = {
        "schema": "rocket-n25-candidate-bundle/v1",
        "protocol": "rocket-n25-complementary-safe-composition/v1",
        "source_build_report": str((BUILD / "build_report.json").resolve()),
        "build_report_sha256": REPORT_SHA,
        "source_base_correction_sha256": CORRECTION_SHA,
        "model_sha256": CANDIDATE_SHA,
        "deck_sha256": DECK_SHA,
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "offline_admission": True,
        "official_match_evidence": False,
        "deployment_authority": False,
    }
    (bundle / "N25_CANDIDATE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if digest(bundle / "model.pt") != CANDIDATE_SHA:
        raise AssertionError("N25 bundled candidate model changed")

    manifest_data = {
        "schema": "rocket-n25-development-arms/v1",
        "protocol": "rocket-n25-complementary-safe-composition/v1",
        "deck_sha256": DECK_SHA,
        "control_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "candidate_bundle": str(bundle.resolve()),
        "candidate_bundle_sha256": tree_digest(bundle),
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "build_report_sha256": REPORT_SHA,
        "source_base_correction_sha256": CORRECTION_SHA,
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
