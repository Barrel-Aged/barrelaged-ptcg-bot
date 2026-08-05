#!/usr/bin/env python3
"""Freeze the four mandatory N9 epoch arms plus immutable control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2" / "model.pt"
)
TRAINING = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n9_balanced_pcgrad_awr" / "training_v1"
)
REPORT = TRAINING / "training_report.json"
PARITY = TRAINING.parent / "exact_agent_parity.json"
DEFAULT_OUTPUT = TRAINING.parent / "development_arms" / "arms.json"
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
ORDER = ("control", "epoch_01", "epoch_02", "epoch_03", "epoch_04")
ALLOWED = ("logit_head.bias", "logit_head.weight")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"config", "state_dict"}:
        raise ValueError(f"unexpected policy payload: {path}")
    return payload


def main() -> int:
    output = DEFAULT_OUTPUT.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N9 arm manifest: {output}")
    if digest(CONTROL) != MODEL_SHA256:
        raise ValueError("N9 control model changed")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("schema") != "rocket-n9-balanced-pcgrad-awr-training/v1":
        raise ValueError("unexpected N9 training report")
    if report.get("protocol") != "rocket-balanced-exact-pcgrad-awr/v1":
        raise ValueError("N9 training protocol changed")
    if report.get("model_sha256") != MODEL_SHA256 or report.get("deck_sha256") != DECK_SHA256:
        raise ValueError("N9 training provenance changed")
    if report.get("checkpoint_selection") != "none from offline metrics; all four epochs are development arms":
        raise ValueError("N9 checkpoint policy changed")
    rows = list(report.get("epochs") or ())
    if [int(row["epoch"]) for row in rows] != [1, 2, 3, 4]:
        raise ValueError("N9 training report lacks four ordered epochs")
    control = load(CONTROL)
    arms = [
        {
            "name": "control",
            "epoch": 0,
            "path": str(CONTROL.resolve()),
            "sha256": MODEL_SHA256,
            "changed_tensors": [],
        }
    ]
    previous = control
    for epoch, row in enumerate(rows, start=1):
        path = (TRAINING / f"epoch_{epoch:02d}.pt").resolve()
        if digest(path) != row.get("model_sha256"):
            raise ValueError(f"N9 epoch hash changed: {epoch}")
        payload = load(path)
        if payload["config"] != control["config"]:
            raise ValueError(f"N9 epoch config changed: {epoch}")
        changed = sorted(
            key
            for key in control["state_dict"]
            if not torch.equal(control["state_dict"][key], payload["state_dict"][key])
        )
        if tuple(changed) != ALLOWED or not all(torch.isfinite(value).all() for value in payload["state_dict"].values()):
            raise ValueError(f"N9 epoch scope/finiteness failed: {epoch}/{changed}")
        if not any(
            not torch.equal(previous["state_dict"][key], payload["state_dict"][key])
            for key in control["state_dict"]
        ):
            raise ValueError(f"N9 epoch did not advance: {epoch}")
        arms.append(
            {
                "name": f"epoch_{epoch:02d}",
                "epoch": epoch,
                "path": str(path),
                "sha256": digest(path),
                "changed_tensors": changed,
            }
        )
        previous = payload
    if tuple(item["name"] for item in arms) != ORDER:
        raise ValueError("N9 arm order changed")
    manifest = {
        "schema": "rocket-n9-development-arms/v1",
        "protocol": "rocket-balanced-exact-pcgrad-awr/v1",
        "deck_sha256": DECK_SHA256,
        "initial_model_sha256": MODEL_SHA256,
        "training_report": str(REPORT.resolve()),
        "training_report_sha256": digest(REPORT),
        "parity_report": str(PARITY.resolve()),
        "parity_report_sha256": digest(PARITY),
        "builder_sha256": digest(Path(__file__).resolve()),
        "arms": arms,
        "development_only": True,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
