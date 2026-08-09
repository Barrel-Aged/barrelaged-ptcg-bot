#!/usr/bin/env python3
"""Build the preregistered N8B Recovery-minus-V110 logit-delta arms."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "scale_value_calibrated_v1" / "model.pt"
)
LIVE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2" / "model.pt"
)
V110 = PROJECT_ROOT / "work" / "arena_bundles" / "AshMarnieGeneralV110_extracted" / "model.pt"
RECOVERY = (
    PROJECT_ROOT / "work" / "arena_bundles"
    / "HelbrosMarnieGeneralRecoveryV4_extracted" / "model.pt"
)
DECK = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "scale_value_calibrated_v1" / "deck.csv"
)
PROTOCOL = PROJECT_ROOT / "N8B_RECOVERY_LOGIT_DELTA_PROTOCOL.md"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n8b_recovery_logit_delta" / "development_arms"
)
EXPECTED = {
    "control": "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff",
    "live": "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b",
    "v110": "552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b",
    "recovery": "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c",
    "deck": "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852",
}
SOURCE_CHANGED = ("logit_head.bias", "logit_head.weight")
ARMS = (
    ("control", 0.0),
    ("delta_050", 0.5),
    ("delta_100", 1.0),
    ("delta_150", 1.5),
    ("delta_200", 2.0),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"config", "state_dict"}:
        raise ValueError(f"unexpected policy payload: {path}")
    return payload


def changed(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> tuple[str, ...]:
    if set(a) != set(b):
        raise ValueError("state-dict key mismatch")
    result = []
    for key in sorted(a):
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            raise ValueError(f"state tensor schema mismatch: {key}")
        if not torch.equal(a[key], b[key]):
            result.append(key)
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = {"control": CONTROL, "live": LIVE, "v110": V110, "recovery": RECOVERY}
    for name, path in paths.items():
        actual = digest(path)
        if actual != EXPECTED[name]:
            raise ValueError(f"{name} source hash changed: {actual}")
    if digest(DECK) != EXPECTED["deck"]:
        raise ValueError("Rocket deck hash changed")
    output = args.output.resolve()
    manifest_path = output / "arms.json"
    if manifest_path.exists() or any((output / f"{name}.pt").exists() for name, _ in ARMS[1:]):
        raise FileExistsError(f"refusing existing N8B arms: {output}")
    protocol_sha = digest(PROTOCOL)
    payloads = {name: load(path) for name, path in paths.items()}
    reference_config = payloads["control"]["config"]
    if any(payload["config"] != reference_config for payload in payloads.values()):
        raise ValueError("N8B source model configurations differ")
    live_changes = changed(payloads["control"]["state_dict"], payloads["live"]["state_dict"])
    if not live_changes or any(not key.startswith("value.") for key in live_changes):
        raise ValueError(f"control/live action-policy equivalence failed: {live_changes}")
    source_changes = changed(payloads["v110"]["state_dict"], payloads["recovery"]["state_dict"])
    if source_changes != SOURCE_CHANGED:
        raise ValueError(f"Marnie source contrast is no longer two-tensor: {source_changes}")
    delta = {
        key: payloads["recovery"]["state_dict"][key].to(torch.float32)
        - payloads["v110"]["state_dict"][key].to(torch.float32)
        for key in SOURCE_CHANGED
    }
    output.mkdir(parents=True, exist_ok=True)
    arm_rows = []
    for name, alpha in ARMS:
        if alpha == 0.0:
            path = CONTROL.resolve()
        else:
            candidate = deepcopy(payloads["control"])
            state = candidate["state_dict"]
            for key in SOURCE_CHANGED:
                state[key] = (
                    state[key].to(torch.float32) + float(alpha) * delta[key]
                ).to(state[key].dtype)
            path = (output / f"{name}.pt").resolve()
            torch.save(candidate, path)
            reloaded = load(path)
            if changed(payloads["control"]["state_dict"], reloaded["state_dict"]) != SOURCE_CHANGED:
                raise ValueError(f"unexpected changed tensor in {name}")
            for key in SOURCE_CHANGED:
                expected = (
                    payloads["control"]["state_dict"][key].to(torch.float32)
                    + float(alpha) * delta[key]
                ).to(payloads["control"]["state_dict"][key].dtype)
                if not torch.equal(reloaded["state_dict"][key], expected):
                    raise ValueError(f"delta construction mismatch: {name}/{key}")
        arm_rows.append(
            {
                "name": name,
                "alpha": alpha,
                "path": str(path),
                "sha256": digest(path),
                "changed_tensors": [] if alpha == 0.0 else list(SOURCE_CHANGED),
            }
        )
    manifest = {
        "schema": "rocket-n8b-logit-delta-arms/v1",
        "protocol": "rocket-recovery-logit-delta-transfer/v1",
        "protocol_path": str(PROTOCOL.resolve()),
        "protocol_sha256": protocol_sha,
        "builder_sha256": digest(Path(__file__).resolve()),
        "deck_sha256": EXPECTED["deck"],
        "sources": {
            name: {"path": str(path.resolve()), "sha256": EXPECTED[name]}
            for name, path in paths.items()
        },
        "control_live_changed_tensors": list(live_changes),
        "source_changed_tensors": list(source_changes),
        "delta_norms": {
            key: {
                "l2": float(torch.linalg.vector_norm(value)),
                "max_abs": float(value.abs().max()),
            }
            for key, value in delta.items()
        },
        "arms": arm_rows,
        "development_only": True,
        "deployment_authority": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "sha256": digest(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
