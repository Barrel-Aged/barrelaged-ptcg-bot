#!/usr/bin/env python3
"""Build the exact-zero rank-32 N17 residual substrate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
N14_ZERO = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/zero_base"
PROTOCOL = ROOT / "N17_REPLICATED_TARGET_RESIDUAL_PROTOCOL.md"

LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
LIVE_NET_SHA = "130061f3a064c32b7f1496c2e11f472359686e9185dd42b7bf488b898f5b8762"
RESIDUAL_NET_SHA = "5a992eae3a33d4a1d100b73f6578923c43f5c5f520aeceea82ff525062dc0290"
PROTOCOL_SHA = "0a3348c41fd5fc821fbfda70280ee516d25c0425b45556a742afc56c9ef381b4"
TARGET_AUDIT_SHA = "c43ee98a05b6344a4a72e65892ea9445cc78155d303f85948bd7c467e28ec74c"
SEED = 817001
RANK = 32


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N17 zero-base builder accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N17 zero base: {OUTPUT}")
    checks = {
        LIVE / "model.pt": LIVE_MODEL_SHA,
        LIVE / "deck.csv": DECK_SHA,
        LIVE / "soldier/net_v10.py": LIVE_NET_SHA,
        N14_ZERO / "soldier/net_v10.py": RESIDUAL_NET_SHA,
        PROTOCOL: PROTOCOL_SHA,
        ROOT / "artifacts/team_rocket_recovery/n17_replicated_targets/target_audit.json": TARGET_AUDIT_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N17 frozen build input changed: {path}")
    shutil.copytree(
        LIVE,
        OUTPUT,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )
    shutil.copy2(N14_ZERO / "soldier/net_v10.py", OUTPUT / "soldier/net_v10.py")

    checkpoint = torch.load(LIVE / "model.pt", map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    if int(config.get("h", -1)) != 384 or int(config.get("mem", -1)) != 0:
        raise ValueError(f"N17 base architecture changed: {config}")
    config.update(adapter_rank=RANK, adapter_scale=1.0)
    state = dict(checkpoint["state_dict"])
    torch.manual_seed(SEED)
    adapter = torch.nn.Sequential(
        torch.nn.LayerNorm(384),
        torch.nn.Linear(384, RANK, bias=False),
        torch.nn.GELU(),
        torch.nn.Linear(RANK, 1, bias=False),
    )
    torch.nn.init.zeros_(adapter[-1].weight)
    for name, value in adapter.state_dict().items():
        state[f"logit_adapter.{name}"] = value.detach().cpu()
    torch.save({"config": config, "state_dict": state}, OUTPUT / "model.pt")
    provenance = {
        "schema": "rocket-n17-zero-residual-base/v1",
        "protocol": "rocket-n17-replicated-target-residual/v1",
        "source_directory": str(LIVE.resolve()),
        "source_model_sha256": LIVE_MODEL_SHA,
        "deck_sha256": DECK_SHA,
        "source_live_net_v10_sha256": LIVE_NET_SHA,
        "residual_net_v10_sha256": digest(OUTPUT / "soldier/net_v10.py"),
        "model_sha256": digest(OUTPUT / "model.pt"),
        "adapter_rank": RANK,
        "adapter_scale": 1.0,
        "adapter_seed": SEED,
        "adapter_parameters": sum(value.numel() for value in adapter.parameters()),
        "final_projection_exact_zero": bool(torch.count_nonzero(adapter[-1].weight) == 0),
        "target_audit_sha256": TARGET_AUDIT_SHA,
        "claims": {"strength_improvement": False, "deployment_authority": False},
    }
    (OUTPUT / "N17_ZERO_BASE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
