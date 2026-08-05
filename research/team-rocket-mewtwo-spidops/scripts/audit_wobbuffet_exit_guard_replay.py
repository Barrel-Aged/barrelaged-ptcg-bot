#!/usr/bin/env python3
"""Audit the Wobbuffet exit guard against every frozen Kaggle observation."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any

import audit_kaggle_scalebcv2 as audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("wobbuffet_exit_guard_replay", path / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decisions(path: Path) -> tuple[int, list[tuple[dict[str, Any], list[int]]]]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    frames = audit.iter_frames(replay)
    seat, _mirror = audit.detect_focal_seat(audit.initial_decks(frames))
    selected: list[tuple[dict[str, Any], list[int]]] = []
    for frame in frames:
        observation = frame.get("obs") or {}
        if (observation.get("current") or {}).get("yourIndex") != seat:
            continue
        if not observation.get("select"):
            continue
        action = frame.get("selected")
        if not isinstance(action, list):
            action = [action] if isinstance(action, int) else []
        selected.append((observation, [int(index) for index in action]))
    return seat, selected


def main() -> int:
    args = parse_args()
    candidate = load_candidate(args.candidate.resolve())
    candidate._GUARD_ENABLED = True
    totals: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    for path in sorted(args.replays.resolve().glob("episode-*-replay.json")):
        _seat, rows = decisions(path)
        blueprint = candidate._fresh_agent()
        for decision_index, (observation, expected) in enumerate(rows):
            baseline = list(blueprint(observation))
            guarded = candidate._wobbuffet_exit_guard_action(observation, baseline)
            selection = observation.get("select") or {}
            options = selection.get("option") or []
            upper = min(int(selection.get("maxCount") or 0), len(options))
            equivalent = (
                upper == len(options)
                and len(expected) == upper
                and sorted(baseline) == sorted(expected)
            )
            totals["decisions"] += 1
            if baseline == expected:
                totals["blueprint_exact"] += 1
            elif equivalent:
                totals["blueprint_set_equivalent"] += 1
            else:
                raise RuntimeError(
                    f"blueprint mismatch in {path.name} decision {decision_index}: "
                    f"expected={expected}, baseline={baseline}"
                )
            if guarded == baseline:
                totals["unchanged"] += 1
                continue
            if len(baseline) != 1 or len(guarded) != 1:
                raise RuntimeError("guard changed a non-single-selection action")
            source = options[baseline[0]]
            target = options[guarded[0]]
            if int(source.get("attackId") or 0) != 610 or int(target.get("type") or -1) != 12:
                raise RuntimeError("guard changed an action outside the declared invariant")
            totals["changed"] += 1
            changes.append(
                {
                    "episode_id": int(path.name.split("-")[1]),
                    "decision_index": decision_index,
                    "turn": (observation.get("current") or {}).get("turn"),
                    "recorded": expected,
                    "blueprint": baseline,
                    "guarded": guarded,
                    "source_option": source,
                    "target_option": target,
                }
            )
        totals["games"] += 1

    payload = {
        "schema": "wobbuffet-exit-guard-kaggle-replay-audit/v1",
        "candidate": str(args.candidate.resolve()),
        "replays": str(args.replays.resolve()),
        "totals": dict(totals),
        "changes": changes,
        "evidence_boundary": (
            "This proves the guard's action-delta surface on recorded observations. "
            "It does not execute or score the alternative branch."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
