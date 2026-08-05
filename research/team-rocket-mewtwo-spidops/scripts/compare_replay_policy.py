#!/usr/bin/env python3
"""Compare recorded arena decisions with an alternate bundle on the same states.

This is an observational diagnostic, not an off-policy value estimate.  It
identifies narrow prompt families where a candidate tactical spine disagrees
with the deployed policy; every proposed override must still pass fresh games
in the official engine.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any


DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--recorded-bot", required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class Worker:
    def __init__(self, bundle: Path, evaluator: Path) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(evaluator.resolve()),
             "--worker-agent-dir", str(bundle.resolve())],
            cwd=str(bundle.resolve()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.stderr: list[str] = []
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.strip():
                self.lines.put(line)
        self.lines.put(None)

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.stderr.append(line.rstrip())
            del self.stderr[:-100]

    def act(self, obs: dict[str, Any]) -> list[int]:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps({"obs": obs}, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.lines.get(timeout=30)
        if line is None:
            raise RuntimeError("alternate worker exited: " + "\n".join(self.stderr[-20:]))
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("traceback") or "alternate worker failed")
        return list(reply.get("action") or [])

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write('{"cmd":"stop"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


def selected_options(obs: dict[str, Any], action: list[int]) -> list[dict[str, Any]]:
    options = ((obs.get("select") or {}).get("option") or [])
    return [
        options[index]
        for index in action
        if isinstance(index, int) and 0 <= index < len(options)
    ]


def option_card_id(obs: dict[str, Any], option: dict[str, Any]) -> int:
    value = option.get("cardId")
    if isinstance(value, int):
        return value
    current = obs.get("current") or {}
    actor = int(current.get("yourIndex") or 0)
    players = current.get("players") or [{}, {}]
    mine = players[actor] or {}
    index = option.get("index")
    area = option.get("area")
    if not isinstance(index, int):
        return 0
    if area == 2 or option.get("type") in (7, 8, 9):
        hand = mine.get("hand") or []
        if 0 <= index < len(hand) and isinstance(hand[index], dict):
            return int(hand[index].get("id") or 0)
    return 0


def signature(obs: dict[str, Any], action: list[int]) -> str:
    values = []
    for option in selected_options(obs, action):
        values.append(
            ":".join(
                str(value)
                for value in (
                    option.get("type", -1),
                    option_card_id(obs, option),
                    option.get("attackId", 0),
                    option.get("inPlayArea", option.get("area", -1)),
                    option.get("inPlayIndex", option.get("index", -1)),
                )
            )
        )
    return ",".join(values) if values else "empty"


def family(name: str) -> str:
    lowered = name.lower()
    for token in ("marnie", "alakazam", "soldier", "starmie", "rocket", "dragapult"):
        if token in lowered:
            return token
    return "other"


def main() -> int:
    args = parse_args()
    files = sorted(args.replays.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no replay files under {args.replays}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker = Worker(args.alternate, args.evaluator)
    rows: list[dict[str, Any]] = []
    try:
        for game_number, path in enumerate(files, 1):
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            meta = records[0].get("meta") or {}
            players = list(meta.get("players") or [])
            if args.recorded_bot not in players:
                continue
            seat = players.index(args.recorded_bot)
            opponent = players[1 - seat]
            terminal = records[-1].get("result") or {}
            won = int(terminal.get("winner", -1)) == seat
            worker.act({"select": None})
            for decision_number, record in enumerate(records[1:-1]):
                if "obs" not in record or int(record.get("p", -1)) != seat:
                    continue
                obs = record["obs"]
                recorded = list(record.get("action") or [])
                alternate = worker.act(obs)
                selection = obs.get("select") or {}
                current = obs.get("current") or {}
                first = current.get("firstPlayer")
                item = {
                    "file": path.name,
                    "game_number": game_number,
                    "decision_number": decision_number,
                    "opponent": opponent,
                    "family": family(opponent),
                    "won": won,
                    "seat": seat,
                    "started_first": first == seat,
                    "turn": int(current.get("turn") or 0),
                    "context": int(selection.get("context", -1) or -1),
                    "selection_type": int(selection.get("type", -1) or -1),
                    "recorded": recorded,
                    "alternate": alternate,
                    "recorded_signature": signature(obs, recorded),
                    "alternate_signature": signature(obs, alternate),
                    "agreed": recorded == alternate,
                }
                if not item["agreed"]:
                    item["options"] = selection.get("option") or []
                rows.append(item)
            if game_number % 25 == 0:
                print(f"processed {game_number}/{len(files)} games", flush=True)
    finally:
        worker.close()

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    games = {row["file"] for row in rows}
    disagreements = [row for row in rows if not row["agreed"]]
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in disagreements:
        key = f"context={row['context']}:type={row['selection_type']}"
        buckets[key]["decisions"] += 1
        buckets[key]["wins"] += int(row["won"])
        buckets[key]["losses"] += int(not row["won"])
        buckets[key]["first"] += int(row["started_first"])
        buckets[key]["second"] += int(not row["started_first"])
    summary = {
        "games": len(games),
        "decisions": len(rows),
        "agreements": len(rows) - len(disagreements),
        "disagreements": len(disagreements),
        "agreement_rate": (len(rows) - len(disagreements)) / max(1, len(rows)),
        "disagreement_buckets": {
            key: dict(value)
            for key, value in sorted(
                buckets.items(), key=lambda item: (-item[1]["decisions"], item[0])
            )
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
