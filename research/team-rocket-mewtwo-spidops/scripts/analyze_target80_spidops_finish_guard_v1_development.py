#!/usr/bin/env python3
"""Audit the locked 200-game Spidops-finish direct development match."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "work" / "target80_spidops_finish_guard_v1_development"
INPUT = PANEL / "head_to_head_200.jsonl"
TRACE = PANEL / "head_to_head_200.trace.jsonl"
LOGS = PANEL / "head_to_head_200.logs"
OUTPUT = PANEL / "analysis.json"
CANDIDATE = ROOT / "work" / "target80_candidates" / "mimikyu_maxbelt_spidops_finish_guard_v1"
CONTROL = ROOT / "work" / "target80_candidates" / "mimikyu_maxbelt_neural_control_v1"

CANDIDATE_NAME = "finish_guard"
CONTROL_NAME = "control"
CANDIDATE_TREE = "952fc72d0cb2eb39c122b0f29e3aee3dac98c3cc17dbc97238c22a7ca3eae70c"
CONTROL_TREE = "30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f"
DECK_SHA256 = "4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea"
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)
EVALUATOR_SHA256 = "d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix.lower() not in (".pyc", ".pyo")
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("development analyzer accepts no overrides")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if (
        tree_sha256(CANDIDATE) != CANDIDATE_TREE
        or tree_sha256(CONTROL) != CONTROL_TREE
        or sha256(CANDIDATE / "deck.csv") != DECK_SHA256
        or sha256(CONTROL / "deck.csv") != DECK_SHA256
        or sha256(CANDIDATE / "model.pt") != MODEL_SHA256
        or sha256(CONTROL / "model.pt") != MODEL_SHA256
        or sha256(EVALUATOR) != EVALUATOR_SHA256
    ):
        raise ValueError("frozen candidate/control/evaluator identity changed")
    rows = load_jsonl(INPUT)
    trace = load_jsonl(TRACE)
    if len(rows) != 200 or [int(row["game"]) for row in rows] != list(range(1, 201)):
        raise ValueError("development game count/order changed")
    if any(
        sorted(row.get("seat_to_agent") or ()) != sorted((CANDIDATE_NAME, CONTROL_NAME))
        for row in rows
    ):
        raise ValueError("development arm identity changed")
    candidate_seats = [
        list(row["seat_to_agent"]).index(CANDIDATE_NAME)
        for row in rows
    ]
    if Counter(candidate_seats) != {0: 100, 1: 100}:
        raise ValueError("development sides are not exactly alternated")

    first_trace: dict[int, dict[str, Any]] = {}
    for item in trace:
        game = int(item["game"])
        current = (item.get("obs") or {}).get("current") or {}
        if int(current.get("firstPlayer", -1)) in (0, 1):
            first_trace.setdefault(game, item)
    if set(first_trace) != set(range(1, 201)):
        raise ValueError("one or more games lack a trace")
    candidate_started_first: dict[int, bool] = {}
    for game, item in first_trace.items():
        current = (item.get("obs") or {}).get("current") or {}
        first_player = int(current["firstPlayer"])
        players = list(item["players"])
        if first_player not in (0, 1) or len(players) != 2:
            raise ValueError(f"game {game}: invalid first-player trace")
        candidate_started_first[game] = players[first_player] == CANDIDATE_NAME

    telemetry = []
    log_errors = []
    candidate_logs = sorted(LOGS.glob(f"game_*/{CANDIDATE_NAME}.stderr.log"))
    if len(candidate_logs) != 200:
        raise ValueError("candidate worker log count changed")
    for path in candidate_logs:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"telemetry": "spidops_finish_energy_guard/v1"' in line:
                telemetry.append(json.loads(line))
            if "Traceback" in line or " agent_" in line.lower():
                log_errors.append(f"{path}:{line[:200]}")

    wins = Counter(str(row["winner"]) for row in rows)
    candidate_wins = int(wins[CANDIDATE_NAME])
    draws = int(wins["draw"])
    by_start: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "first" if candidate_started_first[int(row["game"])] else "second"
        by_start[key].append(row)
    start_metrics = {
        key: {
            "games": len(group),
            "wins": sum(row["winner"] == CANDIDATE_NAME for row in group),
            "win_rate": sum(row["winner"] == CANDIDATE_NAME for row in group)
            / len(group),
        }
        for key, group in sorted(by_start.items())
    }
    execution_errors = [
        row
        for row in rows
        if str(row.get("reason")) != "terminal"
        or int(row.get("result")) not in (0, 1, 2)
    ]
    gates = {
        "exact_200_games_100_each_side": len(rows) == 200
        and Counter(candidate_seats) == {0: 100, 1: 100},
        "candidate_at_least_110_wins": candidate_wins >= 110,
        "candidate_at_least_55pct": candidate_wins / len(rows) >= 0.55,
        "both_start_orders_at_least_50pct": min(
            value["win_rate"] for value in start_metrics.values()
        )
        >= 0.50,
        "at_least_five_guard_interventions": len(telemetry) >= 5,
        "zero_execution_or_log_errors": not execution_errors and not log_errors,
        "draws_are_nonwins": candidate_wins + wins[CONTROL_NAME] + draws == len(rows),
    }
    passes = all(gates.values())
    payload = {
        "schema": "target80-spidops-finish-guard-development-analysis/v1",
        "protocol": "TARGET80_SPIDOPS_FINISH_GUARD_V1_DEVELOPMENT_PROTOCOL.md",
        "input": str(INPUT.resolve()),
        "input_sha256": sha256(INPUT),
        "trace": str(TRACE.resolve()),
        "trace_sha256": sha256(TRACE),
        "candidate_tree_sha256": CANDIDATE_TREE,
        "control_tree_sha256": CONTROL_TREE,
        "evaluator_sha256": EVALUATOR_SHA256,
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "games": len(rows),
        "candidate_wins": candidate_wins,
        "control_wins": int(wins[CONTROL_NAME]),
        "draws": draws,
        "candidate_win_rate": candidate_wins / len(rows),
        "by_start_order": start_metrics,
        "guard_interventions": len(telemetry),
        "intervention_turn_counts": dict(Counter(str(item.get("turn")) for item in telemetry)),
        "execution_errors": len(execution_errors),
        "log_errors": log_errors,
        "gates": gates,
        "passes_development": passes,
        "authority": (
            "broad-opponent official-engine development only"
            if passes
            else "none; close Spidops finish guard V1"
        ),
        "deployment_authority": False,
        "strength_claim": False,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
