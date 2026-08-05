"""Clone a trained stop checkpoint with a different conservative threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one")
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint.get("config", {}).get("stop"):
        raise ValueError("source is not a stop-capable configured checkpoint")
    checkpoint["config"] = dict(checkpoint["config"])
    checkpoint["config"]["stop_threshold"] = float(args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    verified = torch.load(args.out, map_location="cpu", weights_only=True)
    if float(verified["config"]["stop_threshold"]) != float(args.threshold):
        raise AssertionError("saved threshold did not round-trip")
    print({"out": str(args.out.resolve()), "threshold": args.threshold})


if __name__ == "__main__":
    main()
