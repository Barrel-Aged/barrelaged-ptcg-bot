#!/usr/bin/env python3
"""Build the exact-zero N14 residual-capable candidate substrate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/team_rocket_recovery/candidates/leader_bc_scale_v2"
OUTPUT = ROOT / "artifacts/team_rocket_recovery/n14_contextual_residual/zero_base"
MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
NET_SHA = "130061f3a064c32b7f1496c2e11f472359686e9185dd42b7bf488b898f5b8762"
SEED = 814001
RANK = 16


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"N14 patch anchor count changed: {count} for {old[:60]!r}")
    return source.replace(old, new, 1)


def patched_source(source: str) -> str:
    source = replace_once(
        source,
        '                 aux_dim=F.PRIV_DIM, **_ignored):',
        '                 aux_dim=F.PRIV_DIM, adapter_rank=0, adapter_scale=1.0,\n'
        '                 **_ignored):',
    )
    source = replace_once(
        source,
        '                        value_detach=value_detach, aux_dim=aux_dim)',
        '                        value_detach=value_detach, aux_dim=aux_dim,\n'
        '                        adapter_rank=adapter_rank, adapter_scale=adapter_scale)',
    )
    source = replace_once(
        source,
        '        self.logit_head = nn.Linear(h + mem, 1) if mem else nn.Linear(h, 1)\n',
        '        self.logit_head = nn.Linear(h + mem, 1) if mem else nn.Linear(h, 1)\n'
        '        self.adapter_rank = int(adapter_rank)\n'
        '        self.adapter_scale = float(adapter_scale)\n'
        '        self.logit_adapter = (\n'
        '            nn.Sequential(\n'
        '                nn.LayerNorm(h),\n'
        '                nn.Linear(h, self.adapter_rank, bias=False),\n'
        '                nn.GELU(),\n'
        '                nn.Linear(self.adapter_rank, 1, bias=False),\n'
        '            )\n'
        '            if self.adapter_rank > 0 else None\n'
        '        )\n'
        '        if self.logit_adapter is not None:\n'
        '            nn.init.zeros_(self.logit_adapter[-1].weight)\n',
    )
    source = replace_once(
        source,
        '            logits = self.logit_head(otok).squeeze(-1).squeeze(0)         # [O]\n'
        '        if "opt_avail" in feat:',
        '            logits = self.logit_head(otok).squeeze(-1).squeeze(0)         # [O]\n'
        '        if self.logit_adapter is not None:\n'
        '            logits = logits + self.adapter_scale * self.logit_adapter(otok).squeeze(-1).squeeze(0)\n'
        '        if "opt_avail" in feat:',
    )
    source = replace_once(
        source,
        '            logits = self.logit_head(otok).squeeze(-1)                    # [B,O]\n'
        '        logits = logits.masked_fill(~b["opt_mask"].bool(), float("-inf"))',
        '            logits = self.logit_head(otok).squeeze(-1)                    # [B,O]\n'
        '        if self.logit_adapter is not None:\n'
        '            logits = logits + self.adapter_scale * self.logit_adapter(otok).squeeze(-1)\n'
        '        logits = logits.masked_fill(~b["opt_mask"].bool(), float("-inf"))',
    )
    return source


def copy_candidate(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N14 zero-base builder accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite N14 zero base: {OUTPUT}")
    if digest(BASE / "model.pt") != MODEL_SHA or digest(BASE / "deck.csv") != DECK_SHA:
        raise ValueError("N14 frozen live substrate changed")
    net_path = BASE / "soldier/net_v10.py"
    if digest(net_path) != NET_SHA:
        raise ValueError("N14 frozen net_v10 source changed")
    copy_candidate(BASE, OUTPUT)
    patched = patched_source(net_path.read_text(encoding="utf-8"))
    (OUTPUT / "soldier/net_v10.py").write_text(patched, encoding="utf-8")

    checkpoint = torch.load(BASE / "model.pt", map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    if int(config.get("h", -1)) != 384 or int(config.get("mem", -1)) != 0:
        raise ValueError(f"N14 base architecture changed: {config}")
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
        "schema": "rocket-n14-zero-residual-base/v1",
        "protocol": "N14_LOCALIZED_CONTEXTUAL_RESIDUAL_PROTOCOL.md",
        "source_directory": str(BASE.resolve()),
        "source_model_sha256": MODEL_SHA,
        "deck_sha256": DECK_SHA,
        "source_net_v10_sha256": NET_SHA,
        "patched_net_v10_sha256": digest(OUTPUT / "soldier/net_v10.py"),
        "model_sha256": digest(OUTPUT / "model.pt"),
        "adapter_rank": RANK,
        "adapter_scale": 1.0,
        "adapter_seed": SEED,
        "adapter_parameters": sum(value.numel() for value in adapter.parameters()),
        "final_projection_exact_zero": bool(torch.count_nonzero(adapter[-1].weight) == 0),
        "claims": {"strength_improvement": False, "deployment_authority": False},
    }
    (OUTPUT / "N14_ZERO_BASE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

