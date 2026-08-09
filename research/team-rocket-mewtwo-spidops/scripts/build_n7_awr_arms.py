#!/usr/bin/env python3
"""Freeze the four recovered N7 AWR epochs plus the exact live control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2"
)
TRAINING = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n7_marnie_second_awr" / "training_v1"
)
RECOVERY = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n7_marnie_second_awr" / "training_v1_recovery_manifest.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n7_marnie_second_awr" / "development_arms" / "arms.json"
)
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
RECOVERY_SHA256 = "6bc493471fcbb6a83e0ce8ed871d338df864330c0888760edb9b7d318c05a292"
AMENDMENT_SHA256 = "854472c4baf31d293e6ff2c20decfc68eaad8e40b6e86211a241580ba89b410d"
ORDER = ("control", "epoch_01", "epoch_02", "epoch_03", "epoch_04")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N7 arms manifest: {output}")
    if sha256(BASE / "model.pt") != MODEL_SHA256 or sha256(BASE / "deck.csv") != DECK_SHA256:
        raise ValueError("N7 frozen live control changed")
    amendment = PROJECT_ROOT / "N7_TRAINING_OUTPUT_RECOVERY_AMENDMENT.md"
    if sha256(amendment) != AMENDMENT_SHA256 or sha256(RECOVERY) != RECOVERY_SHA256:
        raise ValueError("N7 recovery evidence changed")
    recovery = json.loads(RECOVERY.read_text(encoding="utf-8"))
    integrity = recovery.get("integrity") or {}
    if not integrity or not all(bool(value) for value in integrity.values()):
        raise ValueError("N7 recovery integrity did not pass")
    arms: list[dict[str, Any]] = [
        {
            "name": "control",
            "epoch": 0,
            "path": str((BASE / "model.pt").resolve()),
            "sha256": MODEL_SHA256,
        }
    ]
    for artifact in recovery.get("artifacts") or ():
        epoch = int(artifact["epoch"])
        name = f"epoch_{epoch:02d}"
        path = Path(str(artifact["path"])).resolve()
        if path != (TRAINING / f"epoch_{epoch:02d}.pt").resolve():
            raise ValueError(f"N7 epoch path changed: {name}")
        digest = sha256(path)
        if digest != artifact.get("sha256"):
            raise ValueError(f"N7 epoch hash changed: {name}")
        arms.append(
            {
                "name": name,
                "epoch": epoch,
                "path": str(path),
                "sha256": digest,
                "reference_l2": float(artifact["reference_l2"]),
                "reference_max_abs": float(artifact["reference_max_abs"]),
            }
        )
    if tuple(item["name"] for item in arms) != ORDER:
        raise ValueError("N7 recovered arm order changed")
    manifest = {
        "schema": "rocket-n7-development-arms/v1",
        "protocol": "N7_ARENA_MARNIE_SECOND_AWR_PROTOCOL.md",
        "training_output_amendment": "N7_TRAINING_OUTPUT_RECOVERY_AMENDMENT.md",
        "base_candidate": str(BASE.resolve()),
        "deck_sha256": DECK_SHA256,
        "initial_model_sha256": MODEL_SHA256,
        "recovery_manifest": str(RECOVERY.resolve()),
        "recovery_manifest_sha256": RECOVERY_SHA256,
        "arms": arms,
        "order": list(ORDER),
        "development_only": True,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "arms": list(ORDER)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
