#!/usr/bin/env python3
"""Build a candidate directory only from a test-admitted causal artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MODEL_SHA = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
FROZEN_DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_value_calibrated_v1"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "randomized_full_match_ridge_v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_randomized_causal_ridge_v1"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    artifact_dir = args.artifact.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {output}")
    artifact_manifest = json.loads(
        (artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    test_report = json.loads((artifact_dir / "test_report.json").read_text(encoding="utf-8"))
    if artifact_manifest.get("schema") != "rocket-randomized-full-match-ridge-artifact/v1":
        raise ValueError("unsupported artifact manifest")
    if test_report.get("schema") != "rocket-randomized-full-match-ridge-test/v1":
        raise ValueError("unsupported test report")
    if test_report.get("admitted") is not True:
        raise ValueError("causal artifact did not pass its untouched randomized test")
    artifact_path = artifact_dir / "causal_ridge.npz"
    if sha256(artifact_path) != artifact_manifest.get("artifact_sha256"):
        raise ValueError("causal artifact hash mismatch")
    if sha256(base / "model.pt") != FROZEN_MODEL_SHA:
        raise ValueError("base model hash mismatch")
    if sha256(base / "deck.csv") != FROZEN_DECK_SHA:
        raise ValueError("base deck hash mismatch")
    shutil.copytree(base, output)
    shutil.copy2(artifact_path, output / "causal_ridge.npz")
    shutil.copy2(PROJECT_ROOT / "scripts" / "randomized_causal_main.py", output / "main.py")
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "randomized_causal_runtime.py",
        output / "soldier" / "randomized_causal_runtime.py",
    )
    runtime_config = {
        "schema": artifact_manifest["schema"],
        "artifact_sha256": artifact_manifest["artifact_sha256"],
        "confidence_multiplier": artifact_manifest["confidence_multiplier"],
        "margin": artifact_manifest["margin"],
        "penalty": artifact_manifest["penalty"],
        "eligible_turn": int(artifact_manifest.get("eligible_turn", 2)),
        "shortlist_size": int(artifact_manifest.get("shortlist_size", 4)),
        "validation_metrics": artifact_manifest["validation_metrics"],
        "test_metrics": test_report["metrics"],
        "claim_boundary": (
            "test-admitted randomized opponent overlay; fresh full-match and arena "
            "admission remain required"
        ),
    }
    (output / "causal_ridge_manifest.json").write_text(
        json.dumps(runtime_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_manifest = {
        "schema": "rocket-randomized-causal-candidate-build/v1",
        "base": str(base),
        "output": str(output),
        "base_model_sha256": sha256(output / "model.pt"),
        "base_deck_sha256": sha256(output / "deck.csv"),
        "causal_ridge_sha256": sha256(output / "causal_ridge.npz"),
        "runtime_sha256": sha256(output / "soldier" / "randomized_causal_runtime.py"),
        "main_sha256": sha256(output / "main.py"),
        "test_report_sha256": sha256(artifact_dir / "test_report.json"),
        "official_engine_modified": False,
        "legality_adapter_preserved": True,
    }
    (output / "candidate_build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(build_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
