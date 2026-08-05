#!/usr/bin/env python3
"""Finalize a complete panel whose post-game sidecar step failed.

This never runs or changes games.  It validates every existing JSONL row,
recomputes bundle/output fingerprints, and writes the missing manifest once.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from collect_target80_deck_matched_bc_v1_panel import (
    bundle_fingerprint,
    read_deck,
    sha256,
)


GAME_COLLECTOR_SHA256 = (
    "266a1bcd02b27802722d809925dffb4519c4a5ee533db1eb19757d9de65c8537"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--games-per-arm", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat", choices=("first", "second"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    sidecar = output.with_suffix(".manifest.json")
    if not output.is_file():
        raise FileNotFoundError(output)
    if sidecar.exists():
        raise FileExistsError(f"refusing existing sidecar: {sidecar}")
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_rows = 2 * args.games_per_arm
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")
    expected_seat = 0 if args.seat == "first" else 1
    expected_indexes = list(range(1, expected_rows + 1))
    if [int(row["game_index"]) for row in rows] != expected_indexes:
        raise ValueError("game indexes are incomplete or out of order")
    for row in rows:
        if row.get("schema") != "target80-deck-matched-bc-panel-game/v1":
            raise ValueError("unexpected game schema")
        if row.get("arm") not in {"control", "candidate"}:
            raise ValueError("unexpected arm")
        if int(row.get("schedule_seed")) != args.seed:
            raise ValueError("schedule seed mismatch")
        if int(row.get("own_seat")) != expected_seat:
            raise ValueError("seat mismatch")
        if int(row.get("illegal_actions") or 0) != 0:
            raise ValueError("panel contains an illegal action")
        if int(row.get("exception_fallbacks") or 0) != 0:
            raise ValueError("panel contains an exception fallback")
        if int(row.get("result")) not in (0, 1, 2):
            raise ValueError("invalid terminal result")

    counts = Counter(str(row["arm"]) for row in rows)
    if counts != Counter({"control": args.games_per_arm, "candidate": args.games_per_arm}):
        raise ValueError(f"arm counts mismatch: {counts}")
    wins = Counter(str(row["arm"]) for row in rows if int(row["win"]) == 1)
    draws = Counter(str(row["arm"]) for row in rows if int(row["draw"]) == 1)

    control = args.control.resolve()
    candidate = args.candidate.resolve()
    opponent = args.opponent.resolve()
    control_deck = read_deck(control)
    if control_deck != read_deck(candidate):
        raise ValueError("control/candidate deck mismatch")
    if control_deck.count(432) != 0:
        raise ValueError("deck is not Wobbuffet-free")
    fingerprints = {
        "control": bundle_fingerprint(control),
        "candidate": bundle_fingerprint(candidate),
        "opponent": bundle_fingerprint(opponent),
    }
    manifest = {
        "schema": "target80-deck-matched-bc-panel/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_engine_execution": True,
        "packaged_worker_execution": True,
        "strict_legality": True,
        "hidden_information_used_by_actor": False,
        "visualizer_hidden_truth_used": False,
        "post_game_sidecar_recovery": True,
        "game_collector_sha256": GAME_COLLECTOR_SHA256,
        "finalizer_sha256": sha256(Path(__file__).resolve()),
        "arms": fingerprints,
        "shared_deck_sha256": fingerprints["control"]["deck_sha256"],
        "wobbuffet_count": 0,
        "opponent": str(opponent),
        "seed": args.seed,
        "seat": args.seat,
        "games_per_arm": args.games_per_arm,
        "rows": len(rows),
        "counts": dict(counts),
        "wins": {arm: int(wins[arm]) for arm in ("candidate", "control")},
        "draws": {arm: int(draws[arm]) for arm in ("candidate", "control")},
        "output": str(output),
        "output_sha256": sha256(output),
        "evaluator_sha256": sha256(args.evaluator.resolve()),
        "claim_boundary": "development panel only; no arena strength or deployment claim",
    }
    sidecar.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
