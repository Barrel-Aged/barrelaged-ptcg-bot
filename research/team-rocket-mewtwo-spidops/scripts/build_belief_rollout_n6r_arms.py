#!/usr/bin/env python3
"""Freeze the four preregistered N6R students plus control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
EXPECTED = {
    "arm_a": (0.35, 1.0),
    "arm_b": (0.50, 1.0),
    "arm_c": (0.75, 1.0),
    "arm_d": (0.50, 2.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "n6r_exact_feature" / "distillation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "team_rocket_recovery"
        / "n6r_exact_feature"
        / "development_arms"
        / "arms.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = BASE.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N6R arms manifest: {output}")
    if sha256(base / "deck.csv") != DECK_SHA256 or sha256(base / "model.pt") != MODEL_SHA256:
        raise ValueError("N6R frozen base changed")
    arms: list[dict[str, Any]] = [
        {
            "name": "control",
            "path": str((base / "model.pt").resolve()),
            "sha256": MODEL_SHA256,
            "teacher_temperature": None,
            "kl_coefficient": None,
        }
    ]
    common_train_inputs = common_selection_inputs = None
    for name, (temperature, kl_coefficient) in EXPECTED.items():
        directory = args.training_root.resolve() / name
        report_path = directory / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != "rocket-n6r-belief-rollout-distillation/v2":
            raise ValueError(f"unexpected N6R training report: {report_path}")
        if report.get("protocol") != "N6R_DISTILLATION_PARITY_REPAIR.md":
            raise ValueError(f"N6R protocol mismatch: {report_path}")
        hyper = report["hyperparameters"]
        expected_hyper = {
            "teacher_temperature": temperature,
            "kl_coefficient": kl_coefficient,
            "epochs": 4,
            "batch_size": 64,
            "learning_rate": 2.0e-5,
            "weight_decay": 1.0e-4,
            "seed": 792301,
            "threads": 1,
            "bootstrap_replicates": 2_000,
        }
        for key, expected in expected_hyper.items():
            if hyper.get(key) != expected:
                raise ValueError(f"{name} changed {key}: {hyper.get(key)} != {expected}")
        if report.get("initial_model_sha256") != MODEL_SHA256 or report.get("deck_sha256") != DECK_SHA256:
            raise ValueError(f"{name} frozen provenance changed")
        if report.get("checkpoint_selection") != "final epoch only" or report.get("search_packaged_in_student"):
            raise ValueError(f"{name} violates N6R student boundary")
        encoding = report.get("encoding") or {}
        for split in ("train", "selection"):
            counts = encoding.get(split) or {}
            if counts.get("baseline_reproduction_mismatch", 0) or any(
                str(key).startswith("encode_error:") for key in counts
            ):
                raise ValueError(f"{name} has non-lossless {split} encoding")
        model_path = Path(str(report["artifact"])).resolve()
        if model_path != (directory / "model.pt").resolve() or sha256(model_path) != report.get("artifact_sha256"):
            raise ValueError(f"{name} artifact mismatch")
        train_inputs = report.get("train_inputs")
        selection_inputs = report.get("selection_inputs")
        if common_train_inputs is None:
            common_train_inputs = train_inputs
            common_selection_inputs = selection_inputs
        elif train_inputs != common_train_inputs or selection_inputs != common_selection_inputs:
            raise ValueError("N6R arms did not use identical frozen inputs")
        arms.append(
            {
                "name": name,
                "path": str(model_path),
                "sha256": sha256(model_path),
                "teacher_temperature": temperature,
                "kl_coefficient": kl_coefficient,
                "training_report": str(report_path.resolve()),
                "training_report_sha256": sha256(report_path),
            }
        )
    manifest = {
        "schema": "rocket-n6r-development-arms/v2",
        "protocol": "N6R_EXACT_FEATURE_BELIEF_ROLLOUT_PROTOCOL.md",
        "distillation_execution_amendment": "N6R_DISTILLATION_PARITY_REPAIR.md",
        "deck_sha256": DECK_SHA256,
        "initial_model_sha256": MODEL_SHA256,
        "train_inputs": common_train_inputs,
        "selection_inputs": common_selection_inputs,
        "arms": arms,
        "order": ["control", "arm_a", "arm_b", "arm_c", "arm_d"],
        "development_only": True,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "arms": manifest["order"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
