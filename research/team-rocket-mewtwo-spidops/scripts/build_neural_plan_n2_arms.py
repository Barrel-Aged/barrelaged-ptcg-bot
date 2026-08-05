#!/usr/bin/env python3
"""Build the preregistered N2 neural-plan ensemble artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "neural_plan_n1" / "development_arms"
)
OUTPUT = PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "neural_plan_n2" / "arms"
SEEDS = (781101, 781102, 781103, 781104)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
ARM_CONFIGS = (
    ("zero_control", "blueprint", 0.0, 4),
    ("mean_logits_025", "mean_logits", 0.25, 4),
    ("mean_logits_050", "mean_logits", 0.50, 4),
    ("mean_logits_100", "mean_logits", 1.00, 4),
    ("consensus_3of4", "consensus", 1.00, 3),
    ("consensus_4of4", "consensus", 1.00, 4),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(seed: int) -> tuple[dict[str, Any], Path]:
    path = SOURCE_ROOT / f"seed_{seed}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "rocket-neural-plan-residual/v1":
        raise ValueError(f"source schema mismatch: {path}")
    if int(payload.get("training_seed", -1)) != seed or int(payload.get("iteration", -1)) != 16:
        raise ValueError(f"source is not the locked final seed: {path}")
    if payload.get("deck_sha256") != DECK_SHA256:
        raise ValueError(f"source deck mismatch: {path}")
    if payload.get("base_model_sha256") != MODEL_SHA256:
        raise ValueError(f"source blueprint mismatch: {path}")
    return payload, path


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing nonempty N2 arm directory: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = [load_source(seed) for seed in SEEDS]
    sources = [item[0] for item in loaded]
    configs = [dict(source["config"]) for source in sources]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("N2 sources are not architecturally aligned")
    source_hashes = {
        f"seed_{seed}": sha256(path)
        for seed, (_, path) in zip(SEEDS, loaded)
    }
    arms = []
    for name, mode, scale, min_votes in ARM_CONFIGS:
        payload = {
            "schema": "rocket-neural-plan-ensemble/v1",
            "protocol": "rocket-neural-plan-ensemble/v1",
            "name": name,
            "mode": mode,
            "scale": scale,
            "min_votes": min_votes,
            "deck_sha256": DECK_SHA256,
            "base_model_sha256": MODEL_SHA256,
            "source_seeds": list(SEEDS),
            "source_sha256": source_hashes,
            "sources": sources,
            "construction": "locked_N2_independent_seed_ensemble",
        }
        path = OUTPUT / f"{name}.pt"
        torch.save(payload, path)
        arms.append({"name": name, "path": str(path), "sha256": sha256(path)})
    manifest = {
        "schema": "rocket-neural-plan-ensemble-arms/v1",
        "protocol": "rocket-neural-plan-ensemble/v1",
        "deck_sha256": DECK_SHA256,
        "base_model_sha256": MODEL_SHA256,
        "source_seeds": list(SEEDS),
        "source_sha256": source_hashes,
        "lock_document": "N2_NEURAL_PLAN_ENSEMBLE_PROTOCOL.md",
        "arms": arms,
    }
    (OUTPUT / "arms.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
