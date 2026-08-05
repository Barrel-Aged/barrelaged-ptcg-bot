#!/usr/bin/env python3
"""Build locked N5 greedy-policy development arms from final PPO checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates" / "scale_value_calibrated_v1"
)
TRAINING_ROOT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "training"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "development_arms"
)
SEEDS = (788101, 788102, 788103, 788104)
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
LOCKED_HYPERPARAMETERS = {
    "iterations": 32,
    "rollout_games": 64,
    "ppo_epochs": 2,
    "batch_size": 128,
    "head_learning_rate": 1.0e-4,
    "block_learning_rate": 2.0e-5,
    "weight_decay": 1.0e-5,
    "clip": 0.10,
    "gamma": 0.997,
    "gae_lambda": 0.95,
    "value_coef": 0.30,
    "entropy_coef": 0.001,
    "kl_coef": 0.20,
    "temperature": 0.35,
    "threads": 1,
    "timeout": 30.0,
    "max_steps": 2_000,
}
LOCKED_OPPONENT_HASHES = {
    "AshMarnieGeneralV110_extracted": {
        "deck_sha256": "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b",
        "model_sha256": "552f1b5ddfffe55ce0c6c0d499ddb9f209a8d1e8cd53a5c64da0fcf8e61fb12b",
    },
    "HelbrosMarnieGeneralRecoveryV4_extracted": {
        "deck_sha256": "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b",
        "model_sha256": "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c",
    },
}


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


def load_policy_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(("config", "state_dict")) - set(payload):
        raise ValueError(f"unexpected policy payload: {path}")
    return payload


def main() -> int:
    args = parse_args()
    training_root = args.training_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty arm output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if sha256(BASE_CANDIDATE / "deck.csv") != DECK_SHA256:
        raise ValueError("frozen N5 deck changed")
    if sha256(BASE_CANDIDATE / "model.pt") != MODEL_SHA256:
        raise ValueError("frozen N5 control model changed")

    source_paths: dict[str, Path] = {"control": (BASE_CANDIDATE / "model.pt").resolve()}
    payloads: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        run = training_root / f"seed_{seed}"
        report_path = run / "training_report.json"
        model_path = run / "model.pt"
        if not report_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"incomplete N5 seed: {run}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != "rocket-greedy-compatible-ppo-training/v1":
            raise ValueError(f"unexpected N5 report schema: {report_path}")
        if report.get("protocol") != "rocket-greedy-compatible-ppo/v1":
            raise ValueError(f"unexpected N5 protocol: {report_path}")
        if int(report.get("training_seed", -1)) != seed:
            raise ValueError(f"N5 seed mismatch: {report_path}")
        if int(report.get("total_training_games", -1)) != 2048:
            raise ValueError(f"N5 seed is not the locked 2048-game run: {report_path}")
        if len(report.get("iterations") or ()) != 32:
            raise ValueError(f"N5 seed is not final iteration 32: {report_path}")
        if report.get("deck_sha256") != DECK_SHA256 or report.get("initial_model_sha256") != MODEL_SHA256:
            raise ValueError(f"N5 frozen provenance mismatch: {report_path}")
        if report.get("checkpoint_selection") != "final iteration only":
            raise ValueError(f"N5 checkpoint selection changed: {report_path}")
        if not report.get("official_engine_execution") or report.get("hidden_information_used_by_actor_or_critic"):
            raise ValueError(f"N5 execution/information boundary changed: {report_path}")
        hyperparameters = report.get("hyperparameters") or {}
        for key, expected in LOCKED_HYPERPARAMETERS.items():
            actual = hyperparameters.get(key)
            if isinstance(expected, float):
                matches = actual is not None and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12)
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(f"N5 locked hyperparameter changed ({key}={actual!r}): {report_path}")
        if int(hyperparameters.get("seed", -1)) != seed:
            raise ValueError(f"N5 hyperparameter seed changed: {report_path}")
        if int(report.get("trainable_parameters", -1)) != 1_335_170 or int(report.get("total_parameters", -1)) != 4_298_205:
            raise ValueError(f"N5 trainable/model scope changed: {report_path}")
        if report.get("opponent_hashes") != LOCKED_OPPONENT_HASHES:
            raise ValueError(f"N5 opponent artifacts changed: {report_path}")
        for expected_iteration, record in enumerate(report.get("iterations") or (), start=1):
            if int(record.get("iteration", -1)) != expected_iteration:
                raise ValueError(f"N5 iteration sequence changed: {report_path}")
            if set((record.get("games") or {}).values()) != {32}:
                raise ValueError(f"N5 opponent balance changed: {report_path}")
            update = record.get("update") or {}
            if set(update) != {"policy_loss", "value_loss", "entropy", "kl", "clip_fraction"}:
                raise ValueError(f"N5 update telemetry changed: {report_path}")
            if not all(math.isfinite(float(value)) for value in update.values()):
                raise ValueError(f"N5 update became nonfinite: {report_path}")
            checkpoint = run / str(record.get("checkpoint"))
            if not checkpoint.exists() or sha256(checkpoint) != record.get("checkpoint_sha256"):
                raise ValueError(f"N5 checkpoint missing or changed: {checkpoint}")
        if sha256(model_path) != report.get("final_artifact_sha256"):
            raise ValueError(f"N5 final artifact hash mismatch: {model_path}")
        name = f"seed_{seed}"
        source_paths[name] = model_path.resolve()
        payloads.append(load_policy_payload(model_path))
        reports[name] = report

    config = payloads[0]["config"]
    state_names = tuple(payloads[0]["state_dict"])
    if any(payload["config"] != config for payload in payloads[1:]):
        raise ValueError("N5 seed model configurations are not aligned")
    if any(tuple(payload["state_dict"]) != state_names for payload in payloads[1:]):
        raise ValueError("N5 seed state dictionaries are not aligned")
    soup_state: dict[str, torch.Tensor] = {}
    for name in state_names:
        tensors = [payload["state_dict"][name] for payload in payloads]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"unaligned N5 tensor: {name}")
        if tensors[0].is_floating_point():
            soup_state[name] = torch.stack(tensors).mean(dim=0)
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"unaligned nonfloating N5 tensor: {name}")
            soup_state[name] = tensors[0].clone()
    soup_path = output / "mean_all4.pt"
    torch.save({"config": config, "state_dict": soup_state}, soup_path)
    source_paths["mean_all4"] = soup_path

    arms = [
        {"name": name, "path": str(path), "sha256": sha256(path)}
        for name, path in source_paths.items()
    ]
    manifest = {
        "schema": "rocket-blueprint-n5-development-arms/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "deck_sha256": DECK_SHA256,
        "initial_model_sha256": MODEL_SHA256,
        "lock_document": "N5_ROCKET_BLUEPRINT_FINETUNE_PROTOCOL.md",
        "checkpoint_selection": "final iteration only",
        "source_training_reports": {
            name: {
                "path": str((training_root / name / "training_report.json").resolve()),
                "sha256": sha256(training_root / name / "training_report.json"),
            }
            for name in reports
        },
        "arms": arms,
        "locked_order": ["control", *(f"seed_{seed}" for seed in SEEDS), "mean_all4"],
        "soup_construction": "elementwise mean of four aligned final state dictionaries",
    }
    manifest_path = output / "arms.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(manifest_path), "arms": arms}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
