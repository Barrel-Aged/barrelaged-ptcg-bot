#!/usr/bin/env python3
"""Build the locked N1 development arms and aligned four-seed soup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "neural_plan_n1"
)
DEFAULT_OUTPUT = TRAINING_ROOT / "development_arms"
BASE_CANDIDATE = (
    PROJECT_ROOT
    / "artifacts"
    / "team_rocket_recovery"
    / "candidates"
    / "scale_value_calibrated_v1"
)
SEEDS = (781101, 781102, 781103, 781104)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_final(path: Path, seed: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "rocket-neural-plan-residual/v1":
        raise ValueError(f"unexpected residual schema: {path}")
    if int(payload.get("training_seed", -1)) != seed:
        raise ValueError(f"training seed mismatch: {path}")
    if int(payload.get("iteration", -1)) != 16:
        raise ValueError(f"N1 arm must be final iteration 16: {path}")
    if payload.get("deck_sha256") != DECK_SHA256:
        raise ValueError(f"deck hash mismatch: {path}")
    if payload.get("base_model_sha256") != MODEL_SHA256:
        raise ValueError(f"blueprint hash mismatch: {path}")
    return payload


def main() -> int:
    args = parse_args()
    training_root = args.training_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty arm output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE_CANDIDATE))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from neural_plan_residual_runtime import NeuralPlanResidual, residual_payload

    sources: dict[str, Path] = {}
    payloads: list[dict[str, Any]] = []
    for seed in SEEDS:
        source = training_root / f"seed_{seed}" / "neural_plan_residual.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        payload = load_final(source, seed)
        name = f"seed_{seed}"
        target = output / f"{name}.pt"
        torch.save(payload, target)
        sources[name] = target
        payloads.append(payload)

    config = dict(payloads[0]["config"])
    if any(dict(payload["config"]) != config for payload in payloads[1:]):
        raise ValueError("seed residual configurations are not aligned")
    state_names = tuple(payloads[0]["state_dict"])
    if any(tuple(payload["state_dict"]) != state_names for payload in payloads[1:]):
        raise ValueError("seed state dictionaries are not aligned")
    soup_state: dict[str, torch.Tensor] = {}
    for name in state_names:
        tensors = [payload["state_dict"][name] for payload in payloads]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"unaligned tensor shape: {name}")
        if not tensors[0].is_floating_point():
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"unaligned nonfloating tensor: {name}")
            soup_state[name] = tensors[0].clone()
        else:
            soup_state[name] = torch.stack(tensors, dim=0).mean(dim=0)
    soup_payload = dict(payloads[0])
    soup_payload.update(
        {
            "state_dict": soup_state,
            "training_seed": -1,
            "iteration": 16,
            "provenance": {
                "protocol": "rocket-neural-plan-residual/v1",
                "construction": "elementwise_equal_mean_of_aligned_final_state_dicts",
                "source_seeds": list(SEEDS),
                "source_sha256": {
                    f"seed_{seed}": sha256(
                        training_root / f"seed_{seed}" / "neural_plan_residual.pt"
                    )
                    for seed in SEEDS
                },
            },
        }
    )
    soup_path = output / "mean_all4.pt"
    torch.save(soup_payload, soup_path)
    sources["mean_all4"] = soup_path

    residual = NeuralPlanResidual(
        global_width=int(config["global_width"]),
        option_width=int(config["option_width"]),
        memory=int(config["memory"]),
        max_abs_bias=float(config["max_abs_bias"]),
    )
    zero_path = output / "zero_control.pt"
    torch.save(
        residual_payload(
            residual,
            base_model_sha256=MODEL_SHA256,
            deck_sha256=DECK_SHA256,
            training_seed=0,
            iteration=0,
            provenance={"construction": "fixed_exact_zero_initialization"},
        ),
        zero_path,
    )
    sources = {"zero_control": zero_path, **sources}
    manifest = {
        "schema": "rocket-neural-plan-development-arms/v1",
        "protocol": "rocket-neural-plan-residual/v1",
        "deck_sha256": DECK_SHA256,
        "base_model_sha256": MODEL_SHA256,
        "lock_document": "N1_NEURAL_PLAN_DEVELOPMENT_LOCK.md",
        "arms": [
            {"name": name, "path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        ],
        "allowed_development_arms": [
            "zero_control",
            "seed_781101",
            "seed_781102",
            "seed_781103",
            "seed_781104",
            "mean_all4",
        ],
    }
    (output / "arms.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
