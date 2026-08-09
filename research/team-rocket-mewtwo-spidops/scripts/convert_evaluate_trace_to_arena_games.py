#!/usr/bin/env python3
"""Convert tools/evaluate.py trace output into arena-style per-game JSONL."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    traces: dict[int, list[dict]] = defaultdict(list)
    for row in jsonl(args.trace):
        traces[int(row["game"])].append(row)
    results = {int(row["game"]): row for row in jsonl(args.results)}
    if set(traces) != set(results):
        raise ValueError(
            f"game mismatch: trace-only={sorted(set(traces) - set(results))}, "
            f"result-only={sorted(set(results) - set(traces))}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for game in sorted(results):
        result = results[game]
        players = list(result["seat_to_agent"])
        winner_name = result.get("winner")
        winner = players.index(winner_name) if winner_name in players else None
        output_rows = [{"meta": {"players": players, "game": game}}]
        for row in sorted(traces[game], key=lambda value: int(value["step"])):
            output_rows.append(
                {
                    "p": int(row["seat"]),
                    "action": row.get("action") or [],
                    "obs": row["obs"],
                }
            )
        output_rows.append(
            {
                "result": {
                    "winner": winner,
                    "reason": result.get("reason"),
                    "steps": result.get("steps"),
                }
            }
        )
        path = args.out_dir / f"game_{game:04d}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, separators=(",", ":")) for row in output_rows)
            + "\n",
            encoding="utf-8",
        )
        manifest.append(
            {
                "game": game,
                "path": path.name,
                "players": players,
                "winner": winner,
                "decisions": len(output_rows) - 2,
            }
        )
    (args.out_dir / "manifest.json").write_text(
        json.dumps({"games": manifest}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"games": len(manifest), "out_dir": str(args.out_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
