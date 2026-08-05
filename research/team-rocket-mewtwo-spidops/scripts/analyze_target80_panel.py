#!/usr/bin/env python3
"""Validate and summarize a frozen Team Rocket target-80 development panel."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


EXPECTED_OPPONENTS = (
    "Marnie",
    "LotusControl",
    "SoldierRL45",
    "CrustleGalv71",
    "Alakazam93",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--games-per-cell", type=int, default=8)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tally(rows: list[dict], arm: str) -> dict[str, int | float]:
    wins = sum(row.get("winner") == arm for row in rows)
    draws = sum(row.get("winner") in (None, "draw") for row in rows)
    losses = len(rows) - wins - draws
    return {
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(rows) if rows else 0.0,
    }


def main() -> int:
    args = parse_args()
    arms = sorted(path for path in args.panel.iterdir() if path.is_dir())
    if not arms:
        raise ValueError(f"no arm directories under {args.panel}")

    report: dict[str, object] = {
        "panel": str(args.panel.resolve()),
        "games_per_cell": args.games_per_cell,
        "target": {"minimum_games": 200, "minimum_wins": 160},
        "arms": {},
    }
    for arm_path in arms:
        arm = arm_path.name
        pooled: list[dict] = []
        cells: dict[str, dict] = {}
        seats: dict[int, list[dict]] = defaultdict(list)
        for opponent in EXPECTED_OPPONENTS:
            path = arm_path / f"{opponent}_8.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_jsonl(path)
            if len(rows) != args.games_per_cell:
                raise ValueError(
                    f"{path}: expected {args.games_per_cell} games, got {len(rows)}"
                )
            bad = [row.get("reason") for row in rows if row.get("reason") != "terminal"]
            if bad:
                raise ValueError(f"{path}: nonterminal reasons: {bad}")
            for row in rows:
                seat_names = row.get("seat_to_agent") or []
                if arm not in seat_names:
                    raise ValueError(f"{path}: arm missing from seat map")
                seats[seat_names.index(arm)].append(row)
            cells[opponent] = tally(rows, arm)
            pooled.extend(rows)
        arm_report = tally(pooled, arm)
        arm_report["by_opponent"] = cells
        arm_report["by_seat"] = {
            str(seat): tally(rows, arm) for seat, rows in sorted(seats.items())
        }
        report["arms"][arm] = arm_report

    ranked = sorted(
        report["arms"].items(),
        key=lambda item: (item[1]["win_rate"], item[1]["wins"]),
        reverse=True,
    )
    report["ranking"] = [name for name, _ in ranked]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = [
        "# Team Rocket target-80 development panel",
        "",
        f"Panel: `{args.panel.resolve()}`",
        "",
        "All rows are official-engine terminal games. Win rate is wins divided by all games; draws are non-wins.",
        "",
        "| Arm | W-L-D | Win rate | Seat 0 | Seat 1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in ranked:
        seat0 = data["by_seat"]["0"]
        seat1 = data["by_seat"]["1"]
        markdown.append(
            f"| {name} | {data['wins']}-{data['losses']}-{data['draws']} | "
            f"{data['win_rate']:.1%} | {seat0['wins']}-{seat0['losses']}-{seat0['draws']} | "
            f"{seat1['wins']}-{seat1['losses']}-{seat1['draws']} |"
        )
    markdown.extend(["", "## Matchups", ""])
    for name, data in ranked:
        markdown.extend([f"### {name}", "", "| Opponent | W-L-D | Win rate |", "|---|---:|---:|"])
        for opponent in EXPECTED_OPPONENTS:
            cell = data["by_opponent"][opponent]
            markdown.append(
                f"| {opponent} | {cell['wins']}-{cell['losses']}-{cell['draws']} | "
                f"{cell['win_rate']:.1%} |"
            )
        markdown.append("")
    args.out.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
