#!/usr/bin/env python3
"""Stage an N5 arena candidate only after validation and regression both pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
)
DEFAULT_ARMS = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "validation_arms" / "arms.json"
)
DEFAULT_VALIDATION = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "validation_panel" / "analysis.json"
)
DEFAULT_REGRESSION = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "regression_panel" / "analysis.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "rocket_blueprint_n5_validated_v1"
)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
INITIAL_MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--validation-arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--validation-analysis", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--regression-analysis", type=Path, default=DEFAULT_REGRESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    arms_path = args.validation_arms.resolve()
    validation_path = args.validation_analysis.resolve()
    regression_path = args.regression_analysis.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N5 staged candidate: {output}")
    if sha256(base / "deck.csv") != DECK_SHA256 or sha256(base / "model.pt") != INITIAL_MODEL_SHA256:
        raise ValueError("N5 frozen base candidate changed")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    if validation.get("schema") != "rocket-blueprint-n5-validation-analysis/v1" or not validation.get("locked_validation_pass"):
        raise RuntimeError("N5 staging requires passing locked validation")
    if regression.get("schema") != "rocket-blueprint-n5-regression-analysis/v1" or not regression.get("broader_regression_pass"):
        raise RuntimeError("N5 staging requires passing broader regression")
    arms = json.loads(arms_path.read_text(encoding="utf-8"))
    if arms.get("schema") != "rocket-blueprint-n5-validation-arms/v1":
        raise ValueError("unexpected N5 validation arms")
    arms_hash = sha256(arms_path)
    if validation.get("arms_manifest_sha256") != arms_hash or regression.get("arms_manifest_sha256") != arms_hash:
        raise ValueError("N5 validation/regression reports do not bind the supplied frozen arms")
    if regression.get("validation_analysis_sha256") != sha256(validation_path):
        raise ValueError("N5 regression report does not bind the supplied validation analysis")
    candidate = next((item for item in arms.get("arms") or () if item.get("name") == "candidate"), None)
    if candidate is None:
        raise ValueError("N5 validation candidate is absent")
    model_source = Path(str(candidate["path"])).resolve()
    if sha256(model_source) != str(candidate["sha256"]):
        raise ValueError("N5 validation candidate model changed")

    shutil.copytree(base, output)
    shutil.copy2(model_source, output / "model.pt")
    provenance = {
        "schema": "rocket-blueprint-n5-promotion/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "deck_sha256": sha256(output / "deck.csv"),
        "initial_model_sha256": INITIAL_MODEL_SHA256,
        "promoted_model_sha256": sha256(output / "model.pt"),
        "selected_development_arm": arms["selected_development_arm"],
        "validation_arms_manifest": str(arms_path),
        "validation_arms_manifest_sha256": sha256(arms_path),
        "validation_analysis": str(validation_path),
        "validation_analysis_sha256": sha256(validation_path),
        "regression_analysis": str(regression_path),
        "regression_analysis_sha256": sha256(regression_path),
        "official_engine_validation_games": int(validation["rows"]),
        "official_engine_regression_games": int(regression["rows"]),
        "arena_claim": "none; candidate has not yet been submitted or rated",
    }
    (output / "N5_PROMOTION_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "provenance": provenance}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
