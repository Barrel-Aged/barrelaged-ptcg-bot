#!/usr/bin/env python3
"""Measure a local agent's action agreement with a public Kaggle submission."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_kaggle_scalebcv2 import canonical_deck_bytes, initial_decks, iter_frames
from audit_public_range_safe_router_v13 import load_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def deck_ids(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][:60]


def legal(action: list[int], selection: dict[str, Any]) -> bool:
    options = selection.get("option") or []
    minimum = int(selection.get("minCount") or 0)
    maximum = int(selection.get("maxCount") or 0)
    return bool(
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
    )


def option_signature(option: Any) -> dict[str, Any]:
    if not isinstance(option, dict):
        return {}
    keys = (
        "type",
        "area",
        "index",
        "playerIndex",
        "cardId",
        "attackId",
        "abilityId",
        "inPlayArea",
        "inPlayIndex",
    )
    return {key: option.get(key) for key in keys if key in option}


def main() -> int:
    args = parse_args()
    replays = sorted(args.replays.resolve().glob("episode-*-replay.json"))
    if not replays:
        raise FileNotFoundError("no replay files")
    local_deck = deck_ids(args.agent.resolve() / "deck.csv")
    local_hash = hashlib.sha256(canonical_deck_bytes(local_deck)).hexdigest()
    evaluator = load_evaluator(args.evaluator.resolve())
    log_dir = args.output.with_suffix(".worker")
    log_dir.mkdir(parents=True, exist_ok=True)
    worker = evaluator.AgentWorker("clone-probe", args.agent.resolve(), {}, log_dir)
    totals: Counter[str] = Counter()
    mismatch_rows: list[dict[str, Any]] = []
    try:
        for replay_path in replays:
            data = json.loads(replay_path.read_text(encoding="utf-8"))
            frames = iter_frames(data)
            decks = initial_decks(frames)
            teams = list((data.get("info") or {}).get("TeamNames") or [])
            seats = [index for index, name in enumerate(teams) if name == args.team]
            if len(seats) != 1:
                raise ValueError(f"team {args.team!r} is not unique in {replay_path.name}: {teams}")
            seat = seats[0]
            observed_hash = hashlib.sha256(canonical_deck_bytes(decks[seat])).hexdigest()
            if observed_hash != local_hash:
                raise ValueError(
                    f"deck mismatch in {replay_path.name}: public={observed_hash}, local={local_hash}"
                )
            worker.act({"select": None}, 30.0)
            totals["games"] += 1
            for decision_index, frame in enumerate(frames):
                observation = frame.get("obs") or {}
                if (observation.get("current") or {}).get("yourIndex") != seat:
                    continue
                selection = observation.get("select")
                if not isinstance(selection, dict):
                    continue
                expected = frame.get("selected")
                if not isinstance(expected, list):
                    expected = [expected] if isinstance(expected, int) else []
                expected = [int(index) for index in expected]
                actual = list(worker.act(observation, 30.0))
                if not legal(actual, selection):
                    raise RuntimeError(f"local illegal action in {replay_path.name}: {actual}")
                context = int(selection.get("context") or 0)
                totals["decisions"] += 1
                totals[f"context:{context}:decisions"] += 1
                options = selection.get("option") or []
                full_set_equivalent = (
                    len(expected) == len(actual) == len(options)
                    and sorted(expected) == sorted(actual)
                )
                if expected == actual:
                    totals["exact"] += 1
                    totals[f"context:{context}:exact"] += 1
                elif full_set_equivalent:
                    totals["set_equivalent"] += 1
                    totals[f"context:{context}:set_equivalent"] += 1
                else:
                    totals["mismatch"] += 1
                    totals[f"context:{context}:mismatch"] += 1
                    if len(mismatch_rows) < 300:
                        mismatch_rows.append(
                            {
                                "episode_id": int((data.get("info") or {}).get("EpisodeId") or 0),
                                "decision_index": decision_index,
                                "turn": (observation.get("current") or {}).get("turn"),
                                "context": context,
                                "expected": expected,
                                "actual": actual,
                                "expected_options": [
                                    option_signature(options[index])
                                    for index in expected
                                    if 0 <= index < len(options)
                                ],
                                "actual_options": [
                                    option_signature(options[index])
                                    for index in actual
                                    if 0 <= index < len(options)
                                ],
                            }
                        )
    finally:
        worker.close()

    agreement = (totals["exact"] + totals["set_equivalent"]) / max(1, totals["decisions"])
    payload = {
        "schema": "kaggle-public-policy-clone-audit/v1",
        "team": args.team,
        "agent": str(args.agent.resolve()),
        "deck_sha256": local_hash,
        "counts": dict(totals),
        "agreement": agreement,
        "mismatch_sample": mismatch_rows,
        "claim_boundary": (
            "Recorded-observation behavioral agreement only; it does not reveal the public "
            "submission's source code or prove equal off-distribution strength."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "games": totals["games"],
                "decisions": totals["decisions"],
                "exact": totals["exact"],
                "set_equivalent": totals["set_equivalent"],
                "mismatch": totals["mismatch"],
                "agreement": agreement,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

