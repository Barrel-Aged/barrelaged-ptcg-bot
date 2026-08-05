#!/usr/bin/env python3
"""Build the sole N16 candidate bundle and development arm manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
BUILD = ROOT / "artifacts/team_rocket_recovery/n16_minimum_scale_projection/build_v1"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n16_minimum_scale_projection/development_arms"
PROTOCOL = ROOT / "N16_MINIMUM_SCALE_RESIDUAL_PROTOCOL.md"

DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
CANDIDATE_SHA = "b4f35c7816097800b0c7265691f5f650bc9b467a4258612930d0fef07afcbcf7"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
REPORT_SHA = "cb8b8f970ab913362180f89a396d19adca7a300bf65be3ed64f4990d08ee525a"
PROTOCOL_SHA = "f7b74e00ba0f7250550bca21d4454bcf903169343e82d0dee60cab326d0ba190"
SCALE = 0.859655


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
        raise ValueError("N16 arm builder accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N16 arms: {OUTPUT}")
    checks = {
        LIVE / "model.pt": CONTROL_SHA,
        LIVE / "deck.csv": DECK_SHA,
        ZERO / "model.pt": ZERO_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": RUNTIME_SHA,
        BUILD / "model.pt": CANDIDATE_SHA,
        BUILD / "projection_report.json": REPORT_SHA,
        PROTOCOL: PROTOCOL_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N16 frozen build input changed: {path}")
    report = json.loads((BUILD / "projection_report.json").read_text(encoding="utf-8"))
    if not report["admission"]["passes"] or float(report["selected_scale"]) != SCALE:
        raise ValueError("N16 projection lacks exact offline admission")

    bundle = OUTPUT / "candidate_bundle"
    shutil.copytree(
        ZERO,
        bundle,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )
    shutil.copy2(BUILD / "model.pt", bundle / "model.pt")
    provenance = {
        "schema": "rocket-n16-candidate-bundle/v1",
        "protocol": "rocket-n16-minimum-scale-residual-projection/v1",
        "source_projection_report": str((BUILD / "projection_report.json").resolve()),
        "projection_report_sha256": REPORT_SHA,
        "model_sha256": CANDIDATE_SHA,
        "selected_scale": SCALE,
        "deck_sha256": DECK_SHA,
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "offline_admission": True,
        "official_match_evidence": False,
        "deployment_authority": False,
    }
    (bundle / "N16_CANDIDATE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if digest(bundle / "model.pt") != CANDIDATE_SHA:
        raise AssertionError("N16 bundled candidate model changed")

    manifest_data = {
        "schema": "rocket-n16-development-arms/v1",
        "protocol": "rocket-n16-minimum-scale-residual-projection/v1",
        "deck_sha256": DECK_SHA,
        "control_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "candidate_bundle": str(bundle.resolve()),
        "candidate_bundle_sha256": tree_digest(bundle),
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "projection_report_sha256": REPORT_SHA,
        "selected_scale": SCALE,
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
