#!/usr/bin/env python3
"""Wait for monitor admission, then invoke the independent final audit once."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "work" / "arena_rank_goal_monitor" / "status.json"
WATCH_STATUS = ROOT / "work" / "arena_rank_goal_monitor" / "finalizer_status.json"
FINALIZER = ROOT / "scripts" / "finalize_arena_rank_goal.py"
FINAL_AUDIT = ROOT / "artifacts" / "team_rocket_rank1_300" / "final_audit" / "audit.json"


def write_status(value: dict[str, object]) -> None:
    WATCH_STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = WATCH_STATUS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(WATCH_STATUS)


def observed_at() -> str:
    return datetime.now(timezone.utc).isoformat()


def monitor_admitted() -> tuple[bool, dict[str, object] | None]:
    try:
        value = json.loads(STATUS.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False, None
    return value.get("goal_met") is True, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()
    if args.interval < 60:
        raise ValueError("watch interval must be at least 60 seconds")

    write_status(
        {
            "schema": "rocket-arena-rank1-300-finalizer-watch/v1",
            "observed_at_utc": observed_at(),
            "state": "watching",
            "pid": os.getpid(),
            "read_only_until_admission": True,
        }
    )
    while True:
        if FINAL_AUDIT.is_file():
            write_status(
                {
                    "schema": "rocket-arena-rank1-300-finalizer-watch/v1",
                    "observed_at_utc": observed_at(),
                    "state": "sealed",
                    "final_audit": str(FINAL_AUDIT.resolve()),
                }
            )
            return 0

        admitted, monitor = monitor_admitted()
        if not admitted:
            row = (monitor or {}).get("bot") if isinstance(monitor, dict) else None
            write_status(
                {
                    "schema": "rocket-arena-rank1-300-finalizer-watch/v1",
                    "observed_at_utc": observed_at(),
                    "state": "watching",
                    "monitor_row": row,
                    "read_only_until_admission": True,
                }
            )
            time.sleep(args.interval)
            continue

        completed = subprocess.run(
            [sys.executable, str(FINALIZER)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        attempt = {
            "schema": "rocket-arena-rank1-300-finalizer-watch/v1",
            "observed_at_utc": observed_at(),
            "state": "sealed" if completed.returncode == 0 else "retrying",
            "finalizer_exit_code": completed.returncode,
            "finalizer_stdout": completed.stdout,
            "finalizer_stderr": completed.stderr,
        }
        if completed.returncode == 0 and FINAL_AUDIT.is_file():
            attempt["final_audit"] = str(FINAL_AUDIT.resolve())
            write_status(attempt)
            return 0
        if completed.returncode == 3:
            # A fresh ratings call can legitimately disagree with the last
            # monitor snapshot. Keep waiting for a simultaneous live pass.
            write_status(attempt)
            time.sleep(args.interval)
            continue
        attempt["state"] = "fatal_error"
        write_status(attempt)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
