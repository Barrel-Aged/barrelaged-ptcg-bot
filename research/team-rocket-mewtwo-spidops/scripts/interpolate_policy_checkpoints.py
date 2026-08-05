#!/usr/bin/env python3
"""Linearly interpolate compatible policy checkpoints with full provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = torch.load(args.base, map_location="cpu", weights_only=True)
    target = torch.load(args.target, map_location="cpu", weights_only=True)
    if base["config"] != target["config"]:
        raise ValueError("checkpoint configs differ")
    if base["state_dict"].keys() != target["state_dict"].keys():
        raise ValueError("checkpoint parameter keys differ")
    mixed: dict[str, torch.Tensor] = {}
    changed = 0
    maximum_delta = 0.0
    for key, base_value in base["state_dict"].items():
        target_value = target["state_dict"][key]
        if base_value.shape != target_value.shape:
            raise ValueError(f"shape mismatch for {key}")
        if torch.is_floating_point(base_value):
            value = torch.lerp(base_value, target_value, args.alpha)
            delta = float((target_value - base_value).abs().max())
            changed += int(delta > 0.0)
            maximum_delta = max(maximum_delta, delta)
        else:
            if not torch.equal(base_value, target_value):
                raise ValueError(f"non-floating tensor differs for {key}")
            value = base_value.clone()
        mixed[key] = value
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": base["config"], "state_dict": mixed}, args.out)
    report = {
        "schema_version": 1,
        "method": "linear_checkpoint_interpolation",
        "base": str(args.base.resolve()),
        "base_sha256": sha256(args.base),
        "target": str(args.target.resolve()),
        "target_sha256": sha256(args.target),
        "alpha": args.alpha,
        "changed_parameter_tensors": changed,
        "maximum_endpoint_tensor_delta": maximum_delta,
        "output": str(args.out.resolve()),
        "output_sha256": sha256(args.out),
    }
    report_path = args.report or args.out.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
