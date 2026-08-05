#!/usr/bin/env python3
"""Freeze a completed prefix of a causal JSONL shard with an audit manifest.

Collectors are sometimes stopped after a preregistered game boundary to avoid
spending compute on unused evidence.  This utility copies only complete records
from the requested inclusive game interval, verifies unique roots and complete
candidate rollout support, then records hashes for both source and frozen data.
It never changes the source collection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "rocket-live-truth-causal-action/v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-game", type=int, default=1)
    parser.add_argument("--last-game", type=int, default=30)
    return parser.parse_args()


def validate_row(row: dict[str, Any], line_number: int) -> tuple[int, int]:
    if row.get("schema") != SCHEMA:
        raise ValueError(f"line {line_number}: unsupported schema")
    game = int(row["game_id"])
    root = int(row["root_index"])
    if not row.get("fully_complete"):
        raise ValueError(f"line {line_number}: incomplete root {game}:{root}")
    candidates = [int(index) for index in row.get("candidate_indices") or ()]
    baseline = int(row["baseline"])
    samples = row.get("samples") or {}
    means = row.get("means") or {}
    terminal = row.get("terminal") or {}
    if len(candidates) < 2 or baseline not in candidates:
        raise ValueError(f"line {line_number}: invalid candidate support")
    rollout_counts = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in samples or key not in means or key not in terminal:
            raise ValueError(f"line {line_number}: missing branch {candidate}")
        branch_samples = samples[key]
        if not isinstance(branch_samples, list) or not branch_samples:
            raise ValueError(f"line {line_number}: empty branch {candidate}")
        if int(terminal[key]) != len(branch_samples):
            raise ValueError(f"line {line_number}: incomplete rollout {candidate}")
        rollout_counts.add(len(branch_samples))
    expected_rollouts = int(row.get("complete_rollout_sets") or 0)
    if rollout_counts != {expected_rollouts}:
        raise ValueError(f"line {line_number}: complete-rollout count mismatch")
    if len(rollout_counts) != 1:
        raise ValueError(f"line {line_number}: unequal rollout counts")
    return game, root


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if args.first_game < 1 or args.last_game < args.first_game:
        raise ValueError("invalid game interval")
    if source == output:
        raise ValueError("source and output must differ")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    roots_by_game: Counter[int] = Counter()
    rollout_counts: Counter[int] = Counter()
    source_rows = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        source_rows += 1
        row = json.loads(line)
        game, root = validate_row(row, line_number)
        if not args.first_game <= game <= args.last_game:
            continue
        key = (game, root)
        if key in seen:
            raise ValueError(f"line {line_number}: duplicate root {game}:{root}")
        seen.add(key)
        selected.append(row)
        roots_by_game[game] += 1
        first_candidate = str(int(row["candidate_indices"][0]))
        rollout_counts[len(row["samples"][first_candidate])] += 1
    expected_games = set(range(args.first_game, args.last_game + 1))
    observed_games = set(roots_by_game)
    if observed_games != expected_games:
        missing = sorted(expected_games - observed_games)
        extra = sorted(observed_games - expected_games)
        raise RuntimeError(f"game coverage mismatch; missing={missing}, extra={extra}")
    if not selected:
        raise RuntimeError("no rows selected")
    selected.sort(key=lambda row: (int(row["game_id"]), int(row["root_index"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema": "rocket-frozen-causal-shard/v1",
        "source": str(source),
        "source_sha256": digest(source),
        "source_rows": source_rows,
        "output": str(output),
        "output_sha256": digest(output),
        "first_game": args.first_game,
        "last_game": args.last_game,
        "games": len(observed_games),
        "roots": len(selected),
        "roots_per_game": dict(sorted(roots_by_game.items())),
        "roots_by_rollout_count": dict(sorted(rollout_counts.items())),
        "all_roots_fully_complete": True,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
