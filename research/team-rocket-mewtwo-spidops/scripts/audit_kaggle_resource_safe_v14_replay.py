#!/usr/bin/env python3
"""Replay every ScaleBCv2 Kaggle decision through resource-safe V14."""

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
    spec = importlib.util.spec_from_file_location("resource_safe_v14_replay", path / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decisions(path: Path) -> list[tuple[dict[str, Any], list[int]]]:
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
    return selected


def legal(action: list[int], selection: dict[str, Any]) -> bool:
    options = selection.get("option") or []
    minimum = int(selection.get("minCount") or 0)
    maximum = int(selection.get("maxCount") or 0)
    return bool(
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
    )


def main() -> int:
    args = parse_args()
    candidate = load_candidate(args.candidate.resolve())
    totals: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    for path in sorted(args.replays.resolve().glob("episode-*-replay.json")):
        blueprint = candidate._fresh_agent()
        for decision_index, (observation, expected) in enumerate(decisions(path)):
            baseline = list(blueprint(observation))
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

            after_erasure = candidate._erasure_ball_resource_guard(observation, baseline)
            guarded = candidate._wobbuffet_exit_guard(observation, after_erasure)
            if not legal(guarded, selection):
                raise RuntimeError(
                    f"illegal guarded action in {path.name} decision {decision_index}: {guarded}"
                )
            totals["legal"] += 1
            if guarded == baseline:
                totals["unchanged"] += 1
                continue
            if after_erasure != baseline:
                reason = "erasure_ball_base_ko"
                if guarded or int(selection.get("context") or -1) != 26:
                    raise RuntimeError("Erasure guard violated its declared surface")
                state = observation.get("current") or {}
                players = state.get("players") or []
                seat = state.get("yourIndex")
                if not isinstance(seat, int) or not 0 <= seat < len(players):
                    raise RuntimeError("Erasure guard lacks a valid acting player")
                bench = (players[seat] or {}).get("bench") or []
                for option in options:
                    bench_index = int(option.get("index") or 0)
                    energy_index = int(option.get("energyIndex") or 0)
                    if (
                        option.get("playerIndex") != seat
                        or not 0 <= bench_index < len(bench)
                        or not 0 <= energy_index < len(bench[bench_index].get("energyCards") or [])
                    ):
                        raise RuntimeError(
                            "Erasure guard reached an option that is not acting-player Bench Energy"
                        )
                totals["erasure_own_bench_invariant"] += 1
            else:
                reason = "wobbuffet_exit"
                if len(baseline) != 1 or len(guarded) != 1:
                    raise RuntimeError("Wobbuffet guard changed a non-single action")
                source = options[baseline[0]]
                target = options[guarded[0]]
                if int(source.get("attackId") or 0) != 610 or int(target.get("type") or -1) != 12:
                    raise RuntimeError("Wobbuffet guard violated its declared surface")
            totals["changed"] += 1
            totals[f"changed:{reason}"] += 1
            changes.append(
                {
                    "episode_id": int(path.name.split("-")[1]),
                    "decision_index": decision_index,
                    "turn": (observation.get("current") or {}).get("turn"),
                    "recorded": expected,
                    "blueprint": baseline,
                    "guarded": guarded,
                    "reason": reason,
                }
            )
        totals["games"] += 1

    payload = {
        "schema": "kaggle-resource-safe-v14-replay-audit/v1",
        "candidate": str(args.candidate.resolve()),
        "replays": str(args.replays.resolve()),
        "totals": dict(totals),
        "changes": changes,
        "passed": bool(
            totals["games"] == 20
            and totals["decisions"] == 1227
            and totals["legal"] == totals["decisions"]
            and totals["changed:erasure_ball_base_ko"] > 0
            and totals["erasure_own_bench_invariant"]
            == totals["changed:erasure_ball_base_ko"]
            and totals["changed:wobbuffet_exit"] == 2
        ),
        "evidence_boundary": (
            "Recorded-observation action audit only; altered branches are not replay outcomes."
        ),
    }
    if not payload["passed"]:
        raise AssertionError(json.dumps(payload, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "changes"}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
