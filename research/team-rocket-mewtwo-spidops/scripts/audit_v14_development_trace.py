#!/usr/bin/env python3
"""Compare V14 official-engine traces with the exact ScaleBCv2 blueprint."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from audit_public_range_safe_router_v13 import load_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def reason(observation: dict, recorded: list[int], alternate: list[int]) -> str:
    selection = observation.get("select") or {}
    effect = selection.get("effect") or {}
    options = selection.get("option") or []
    if (
        int(selection.get("context") or -1) == 26
        and int(effect.get("id") or 0) == 431
        and recorded == []
        and alternate
    ):
        return "erasure_ball_base_ko"
    if len(recorded) == len(alternate) == 1:
        before = options[alternate[0]] if 0 <= alternate[0] < len(options) else {}
        after = options[recorded[0]] if 0 <= recorded[0] < len(options) else {}
        if int(before.get("attackId") or 0) == 610 and int(after.get("type") or -1) == 12:
            return "wobbuffet_exit"
    return "unexpected"


def main() -> int:
    args = parse_args()
    files = sorted(args.traces.resolve().glob("*_candidate.trace.jsonl"))
    if not files:
        raise FileNotFoundError("no candidate trace files")
    evaluator = load_evaluator(args.evaluator.resolve())
    worker_log_dir = args.output.with_suffix(".worker")
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    worker = evaluator.AgentWorker(
        "scale-control", args.alternate.resolve(), {}, worker_log_dir
    )
    counts: Counter[str] = Counter()
    rows = []
    try:
        for path in files:
            current_game = None
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("bot") != "ResourceSafeV14":
                    continue
                game = int(record.get("game") or 0)
                if game != current_game:
                    worker.act({"select": None}, 30.0)
                    current_game = game
                    counts["games"] += 1
                observation = record.get("obs") or {}
                recorded = list(record.get("action") or [])
                alternate = worker.act(observation, 30.0)
                counts["decisions"] += 1
                if recorded == alternate:
                    counts["agreements"] += 1
                    continue
                why = reason(observation, recorded, alternate)
                counts["disagreements"] += 1
                counts[f"reason:{why}"] += 1
                counts[f"file:{path.stem}:{why}"] += 1
                rows.append(
                    {
                        "file": path.name,
                        "game": game,
                        "turn": (observation.get("current") or {}).get("turn"),
                        "recorded": recorded,
                        "alternate": alternate,
                        "reason": why,
                    }
                )
    finally:
        worker.close()
    payload = {
        "schema": "v14-development-trace-audit/v1",
        "counts": dict(counts),
        "rows": rows,
        "passed": counts["reason:unexpected"] == 0,
        "claim_boundary": "Behavior audit on realized V14 games; not an off-policy outcome estimate.",
    }
    if not payload["passed"]:
        raise AssertionError(json.dumps(payload, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": payload["counts"], "passed": payload["passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
