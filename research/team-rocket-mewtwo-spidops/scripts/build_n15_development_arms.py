#!/usr/bin/env python3
"""Build the sole N15 candidate bundle and two-arm development manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
TRAINING = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/training_v1"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n15_terminal_consolidation/development_arms"
PROTOCOL = ROOT / "N15_TERMINAL_CONSOLIDATION_PROTOCOL.md"

DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CONTROL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ZERO_SHA = "0e4bbcc8f1bceff052f4f0921f34ac5448b26f551d2779e5dfbf7b88ca7d13eb"
CANDIDATE_SHA = "354a41f5d4146ae2b48b3935653a0e47f535150eebcd37e90aa866d98d0c5ee7"
RUNTIME_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
TRAINING_REPORT_SHA = "b508381af55d3f86654ab527622bec333696fa7ab37b05fb06f5b71933d92ce1"
PROTOCOL_SHA = "f6da7eb8f3bf6d8f64e9d5b3670c0cadee4fe171f4b7525f1569f07b1af32beb"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    result = hashlib.sha256()
    paths = sorted(
        (
            path for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix.lower() not in (".pyc", ".pyo")
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        result.update(len(relative).to_bytes(4, "big"))
        result.update(relative)
        result.update(path.read_bytes())
    return result.hexdigest()


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N15 arm builder accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N15 arms: {OUTPUT}")
    checks = {
        LIVE / "model.pt": CONTROL_SHA,
        LIVE / "deck.csv": DECK_SHA,
        ZERO / "model.pt": ZERO_SHA,
        ZERO / "deck.csv": DECK_SHA,
        ZERO / "soldier/net_v10.py": RUNTIME_SHA,
        TRAINING / "terminal2000.pt": CANDIDATE_SHA,
        TRAINING / "training_report.json": TRAINING_REPORT_SHA,
        PROTOCOL: PROTOCOL_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N15 frozen build input changed: {path}")
    training = json.loads((TRAINING / "training_report.json").read_text(encoding="utf-8"))
    if not training["terminal"]["admission"]["passes"]:
        raise ValueError("N15 terminal candidate lacks offline admission")

    bundle = OUTPUT / "candidate_bundle"
    shutil.copytree(
        ZERO,
        bundle,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )
    shutil.copy2(TRAINING / "terminal2000.pt", bundle / "model.pt")
    provenance = {
        "schema": "rocket-n15-candidate-bundle/v1",
        "protocol": "rocket-n15-prospective-terminal-consolidation/v1",
        "source_zero_bundle": str(ZERO.resolve()),
        "source_zero_model_sha256": ZERO_SHA,
        "source_terminal_model": str((TRAINING / "terminal2000.pt").resolve()),
        "model_sha256": CANDIDATE_SHA,
        "deck_sha256": DECK_SHA,
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "training_report_sha256": TRAINING_REPORT_SHA,
        "offline_admission": True,
        "official_match_evidence": False,
        "deployment_authority": False,
    }
    (bundle / "N15_CANDIDATE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if digest(bundle / "model.pt") != CANDIDATE_SHA:
        raise AssertionError("N15 bundled candidate model changed")

    arms = {
        "schema": "rocket-n15-development-arms/v1",
        "protocol": "rocket-n15-prospective-terminal-consolidation/v1",
        "deck_sha256": DECK_SHA,
        "control_model_sha256": CONTROL_SHA,
        "candidate_model_sha256": CANDIDATE_SHA,
        "candidate_bundle": str(bundle.resolve()),
        "candidate_bundle_sha256": tree_digest(bundle),
        "runtime_net_v10_sha256": RUNTIME_SHA,
        "training_report_sha256": TRAINING_REPORT_SHA,
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
    manifest.write_text(json.dumps(arms, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
