#!/usr/bin/env python3
"""Verify that restarted N10 Recovery shards reproduce archived prefixes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "artifacts/team_rocket_recovery/n10_epoch4_confirmation/panel"
ARCHIVE = PANEL / "interrupted_20260730_poweroff"
OUTPUT = PANEL / "resume_prefix_verification.json"
EXPECTED = {
    "recovery_01": {
        "lines": 551,
        "partial_sha256": "b34a83b9361dabcdde0666d8e5c8c1e6a32ea92aa1c89113b466646bccd305c0",
        "seed": 799201,
    },
    "recovery_02": {
        "lines": 558,
        "partial_sha256": "2628f1a59cd45470a02942b2c9d2605b3545ed36fc3a9169c0604d8c6dd3a9bc",
        "seed": 799202,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing resume verification: {OUTPUT}")
    results = {}
    for name, expected in EXPECTED.items():
        partial = ARCHIVE / f"{name}.partial.jsonl"
        resumed = PANEL / f"{name}.jsonl"
        manifest_path = PANEL / f"{name}.manifest.json"
        if digest(partial) != expected["partial_sha256"]:
            raise ValueError(f"archived prefix changed: {partial}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schedule_seed", -1)) != expected["seed"] or int(manifest.get("rows", -1)) != 800:
            raise ValueError(f"resumed manifest differs from lock: {manifest_path}")
        if digest(resumed) != manifest.get("output_sha256"):
            raise ValueError(f"resumed shard hash mismatch: {resumed}")
        partial_lines = partial.read_bytes().splitlines(keepends=True)
        resumed_lines = resumed.read_bytes().splitlines(keepends=True)
        if len(partial_lines) != expected["lines"] or len(resumed_lines) != 800:
            raise ValueError(f"unexpected line count for {name}")
        mismatch = next(
            (index for index, (old, new) in enumerate(zip(partial_lines, resumed_lines), start=1) if old != new),
            None,
        )
        if mismatch is not None:
            raise ValueError(f"resumed prefix mismatch for {name} at line {mismatch}")
        results[name] = {
            "archived_prefix_lines": len(partial_lines),
            "archived_prefix_sha256": digest(partial),
            "full_resumed_lines": len(resumed_lines),
            "full_resumed_sha256": digest(resumed),
            "exact_prefix_match": True,
            "schedule_seed": expected["seed"],
        }
    report = {
        "schema": "rocket-n10-resume-prefix-verification/v1",
        "protocol": "rocket-n10-independent-epoch4-confirmation/v1",
        "interruption_cause": "user_requested_host_poweroff",
        "outcomes_inspected_before_restart": False,
        "results": results,
        "all_exact": True,
        "deployment_authority": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "all_exact": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

