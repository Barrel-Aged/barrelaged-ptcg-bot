#!/usr/bin/env python3
"""Fail-closed tensor audit for the four N7 checkpoints saved after pipe timeout."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

try:
    from train_public_leader_v110_bc import configure_trainable
except ModuleNotFoundError:  # pragma: no cover
    from scripts.train_public_leader_v110_bc import configure_trainable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2"
)
DEFAULT_TRAINING = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n7_marnie_second_awr" / "training_v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n7_marnie_second_awr" / "training_v1_recovery_manifest.json"
)
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
TRAINER_SHA256 = "4f233d40e625c33711ab87438388f8ba68f65f2e103c563bf621c790c9a657e7"
PROTOCOL_SHA256 = "f5aa03037ec3ed15d4347a95aa45cb1112b080ebd50f411092c144f110399365"
EXPECTED_EPOCH_SHA256 = {
    "epoch_01.pt": "e47df4642c513fe776d36d96904cac6346873741b54265a7669707d1e3221dc9",
    "epoch_02.pt": "1ac0797a97fe1494ffec337d96b2305e4797205ffb52e89b29eff250bc8f07f2",
    "epoch_03.pt": "b906d64e41f56554d088dee37f21a71561c6badcb6941b88c1dc29273d99ff2f",
    "epoch_04.pt": "aa079ee3a76803ac6c3024861f42bf2f8c4fbc9f41ccb8a3d480ceb8a6591192",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_is_finite(tensor: torch.Tensor) -> bool:
    return not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    training = args.training.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite recovery manifest: {output}")
    if sha256(candidate / "model.pt") != MODEL_SHA256:
        raise ValueError("N7 frozen source model changed")
    if sha256(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("N7 frozen deck changed")
    if sha256(PROJECT_ROOT / "scripts" / "train_live_outcome_awr.py") != TRAINER_SHA256:
        raise ValueError("N7 trainer changed after preregistration")
    if sha256(PROJECT_ROOT / "N7_ARENA_MARNIE_SECOND_AWR_PROTOCOL.md") != PROTOCOL_SHA256:
        raise ValueError("N7 protocol changed after preregistration")
    if (training / "model.pt").exists() or (training / "training_summary.json").exists():
        raise ValueError("recovery audit expected an interrupted epilogue, not final aliases")
    epoch_paths = sorted(training.glob("epoch_*.pt"))
    if [path.name for path in epoch_paths] != list(EXPECTED_EPOCH_SHA256):
        raise ValueError("N7 recovery epoch set changed")
    for path in epoch_paths:
        if sha256(path) != EXPECTED_EPOCH_SHA256[path.name]:
            raise ValueError(f"N7 epoch hash changed: {path.name}")

    sys.path.insert(0, str(candidate))
    from soldier.net import load_policy  # type: ignore

    reference = load_policy(candidate / "model.pt", map_location="cpu").eval()
    scope_probe = copy.deepcopy(reference)
    configure_trainable(scope_probe, "last")
    allowed_names = {
        name for name, parameter in scope_probe.named_parameters()
        if parameter.requires_grad
    }
    if not allowed_names:
        raise ValueError("N7 trainable scope is empty")
    reference_state = reference.state_dict()
    prior_state = reference_state
    artifacts: list[dict[str, Any]] = []
    prior_timestamp = 0.0
    for epoch, path in enumerate(epoch_paths, start=1):
        stat = path.stat()
        if stat.st_mtime <= prior_timestamp:
            raise ValueError("N7 checkpoint timestamps are not monotonic")
        prior_timestamp = stat.st_mtime
        model = load_policy(path, map_location="cpu").eval()
        if getattr(model, "cfg", None) != getattr(reference, "cfg", None):
            raise ValueError(f"N7 architecture config changed in {path.name}")
        state = model.state_dict()
        if state.keys() != reference_state.keys():
            raise ValueError(f"N7 state keys changed in {path.name}")
        changed = []
        changed_from_prior = []
        squared_l2 = 0.0
        max_abs = 0.0
        for name, tensor in state.items():
            if not tensor_is_finite(tensor):
                raise ValueError(f"non-finite tensor in {path.name}: {name}")
            reference_tensor = reference_state[name]
            if not torch.equal(tensor, reference_tensor):
                changed.append(name)
                if name not in allowed_names:
                    raise ValueError(
                        f"frozen parameter changed in {path.name}: {name}"
                    )
                difference = tensor.detach().float() - reference_tensor.detach().float()
                squared_l2 += float(torch.square(difference).sum())
                max_abs = max(max_abs, float(difference.abs().max()))
            if not torch.equal(tensor, prior_state[name]):
                changed_from_prior.append(name)
        if not changed or not changed_from_prior:
            raise ValueError(f"N7 checkpoint did not advance: {path.name}")
        artifacts.append(
            {
                "epoch": epoch,
                "path": str(path),
                "sha256": sha256(path),
                "bytes": stat.st_size,
                "modified_unix": stat.st_mtime,
                "changed_parameter_tensors": sorted(changed),
                "changed_from_prior_tensors": sorted(changed_from_prior),
                "reference_l2": squared_l2 ** 0.5,
                "reference_max_abs": max_abs,
                "all_tensors_finite": True,
                "only_preregistered_scope_changed": True,
            }
        )
        prior_state = state

    result = {
        "schema": "n7-training-output-recovery/v1",
        "development_only": True,
        "deployment_authority": False,
        "event": {
            "outer_shell_timeout_seconds": 3604,
            "trainer_pid": 17480,
            "observation": (
                "outer capture timed out; the same orphaned process continued and "
                "atomically saved all four epochs, then exited before final alias/summary"
            ),
            "unrecovered": ["stdout epoch metrics", "model.pt alias", "training_summary.json"],
        },
        "source": {
            "candidate": str(candidate),
            "model_sha256": MODEL_SHA256,
            "deck_sha256": DECK_SHA256,
            "trainer_sha256": TRAINER_SHA256,
            "protocol_sha256": PROTOCOL_SHA256,
        },
        "fixed_training": {
            "seed": 794101,
            "threads": 1,
            "mode": "last",
            "epochs": 4,
            "batch_size": 128,
            "learning_rate": 3.0e-6,
            "weight_decay": 1.0e-4,
            "temperature": 0.5,
            "kl_coefficient": 0.25,
        },
        "allowed_parameter_tensors": sorted(allowed_names),
        "artifacts": artifacts,
        "integrity": {
            "all_four_epochs_present": True,
            "hashes_locked": True,
            "timestamps_monotonic": True,
            "all_models_load": True,
            "all_tensors_finite": True,
            "only_preregistered_scope_changed": True,
            "every_epoch_advanced": True,
        },
        "recovery_decision": (
            "accept epoch_01.pt through epoch_04.pt as the four preregistered "
            "development arms; do not reconstruct or use offline checkpoint selection"
        ),
        "next_authority": "runtime parity and locked official-engine development gate only",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
