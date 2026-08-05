#!/usr/bin/env python3
"""Audit packaged N18 arms on a frozen sample of genuine Rocket observations."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "development_arms/arms.json"
)
AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "N18_NEW_LEADER_EVIDENCE_AUDIT.json"
)
REPLAYS = ROOT / "work/n18/am"
OUTPUT = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "development_arms/runtime_audit.json"
)
EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab\tools\evaluate.py"
)

BOT = "HelbrosRocketSpidopsScaleBCV2"
ARMS_SHA = "57ab74eaaf05aa2f4e36a09f1555aef4eadf1d84dd0415c59376de036bfbed18"
AUDIT_SHA = "f0441e0e602421c0ccbb57447e9b5857af442c93e7b6a02224cd4f09c24e9a0c"
CORPUS_SHA = "05fa90550b4720f44cb8b100c7d8b34a54c65957daca2d8c3df8444bbe719efe"
WORKER_SHA = "a8eb4808fd29109ebd851b9802b72245b3353b0d64f97384a1aebd1948b80b85"
EVALUATOR_SHA = "d676c7a385b73140d60cbce1eac65ee024c0dee8c7c6746e902e4388e67f5303"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
ORDER = (
    "control",
    "bridge_base",
    "live_residual",
    "leader_base",
    "leader_residual",
)
GAMES_PER_SEAT_OUTCOME = 3


sys.path.insert(0, str((ROOT / "scripts").resolve()))
from train_rank1_second_seat_ppo import OpponentWorker, legal, sha256  # noqa: E402
from collect_rocket_blueprint_n5_panel import tree_sha256  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_digest(root: Path) -> str:
    result = hashlib.sha256()
    for path in sorted(root.glob("*.jsonl")):
        result.update(path.name.encode("utf-8"))
        result.update(bytes.fromhex(digest(path)))
    return result.hexdigest()


def selected_games() -> list[dict[str, Any]]:
    strata: dict[tuple[int, str], list[dict[str, Any]]] = {
        (seat, outcome): [] for seat in (0, 1) for outcome in ("win", "loss")
    }
    for path in sorted(REPLAYS.glob("*.jsonl")):
        rows = load_rows(path)
        players = list(rows[0]["meta"]["players"])
        if BOT not in players:
            continue
        seat = players.index(BOT)
        result = next(row["result"] for row in rows if "result" in row)
        winner = result.get("winner")
        if winner not in (0, 1):
            continue
        outcome = "win" if int(winner) == seat else "loss"
        key = (seat, outcome)
        if len(strata[key]) < GAMES_PER_SEAT_OUTCOME:
            strata[key].append(
                {
                    "path": path,
                    "rows": rows,
                    "seat": seat,
                    "outcome": outcome,
                }
            )
    if any(len(items) != GAMES_PER_SEAT_OUTCOME for items in strata.values()):
        raise ValueError("N18 runtime sample lacks a complete seat/outcome stratum")
    return [item for key in sorted(strata) for item in strata[key]]


def action_sequence_digest(actions: list[tuple[str, int, list[int]]]) -> str:
    payload = json.dumps(actions, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N18 runtime audit: {OUTPUT}")
    checks = {
        ARMS: ARMS_SHA,
        AUDIT: AUDIT_SHA,
        ROOT / "scripts/train_rank1_second_seat_ppo.py": WORKER_SHA,
        EVALUATOR: EVALUATOR_SHA,
    }
    for path, expected in checks.items():
        if digest(path) != expected:
            raise ValueError(f"N18 runtime-audit input changed: {path}")
    if corpus_digest(REPLAYS) != CORPUS_SHA:
        raise ValueError("N18 audited leader replay corpus changed")

    manifest = json.loads(ARMS.read_text(encoding="utf-8"))
    if tuple(manifest.get("order") or ()) != ORDER:
        raise ValueError("N18 arm order changed")
    arms = {item["name"]: item for item in manifest.get("arms") or ()}
    if tuple(arms) != ORDER:
        raise ValueError("N18 arm manifest changed")
    for name, item in arms.items():
        bundle = Path(item["bundle"])
        if tree_sha256(bundle) != item["bundle_tree_sha256"]:
            raise ValueError(f"N18 arm tree changed: {name}")
        if sha256(bundle / "deck.csv") != DECK_SHA:
            raise ValueError(f"N18 arm deck changed: {name}")

    games = selected_games()
    sample = [
        {
            "file": item["path"].name,
            "file_sha256": digest(item["path"]),
            "seat": item["seat"],
            "outcome": item["outcome"],
        }
        for item in games
    ]
    control_actions: dict[tuple[str, int], list[int]] = {}
    results: dict[str, Any] = {}
    started = time()

    for name in ORDER:
        bundle = Path(arms[name]["bundle"])
        worker = OpponentWorker(bundle, EVALUATOR, 30.0)
        counts: Counter[str] = Counter()
        action_sequence: list[tuple[str, int, list[int]]] = []
        try:
            for game in games:
                worker.reset()
                for row_index, row in enumerate(game["rows"]):
                    if "obs" not in row or int(row.get("p", -1)) != game["seat"]:
                        continue
                    observation = row["obs"]
                    selection = observation.get("select") or {}
                    recorded = list(row.get("action") or [])
                    action = worker.act(observation)
                    if not legal(action, selection):
                        raise RuntimeError(
                            f"illegal packaged N18 action arm={name} file={game['path'].name} "
                            f"row={row_index}: {action}"
                        )
                    key = (game["path"].name, row_index)
                    counts["decisions"] += 1
                    counts["legal_decisions"] += 1
                    counts["recorded_ordered_mismatches"] += int(action != recorded)
                    counts["recorded_set_mismatches"] += int(
                        sorted(action) != sorted(recorded)
                    )
                    if name == "control":
                        control_actions[key] = action
                    else:
                        counts["control_ordered_mismatches"] += int(
                            action != control_actions[key]
                        )
                        counts["control_set_mismatches"] += int(
                            sorted(action) != sorted(control_actions[key])
                        )
                    action_sequence.append((key[0], key[1], action))
        finally:
            worker.close()
        results[name] = {
            "bundle_tree_sha256": arms[name]["bundle_tree_sha256"],
            "decisions": counts["decisions"],
            "legal_decisions": counts["legal_decisions"],
            "recorded_ordered_mismatches": counts["recorded_ordered_mismatches"],
            "recorded_set_mismatches": counts["recorded_set_mismatches"],
            "control_ordered_mismatches": (
                counts["control_ordered_mismatches"] if name != "control" else 0
            ),
            "control_set_mismatches": (
                counts["control_set_mismatches"] if name != "control" else 0
            ),
            "control_set_mismatch_rate": (
                counts["control_set_mismatches"] / counts["decisions"]
                if name != "control"
                else 0.0
            ),
            "action_sequence_sha256": action_sequence_digest(action_sequence),
        }

    decision_counts = {result["decisions"] for result in results.values()}
    if len(decision_counts) != 1:
        raise AssertionError("N18 runtime arms saw different decision counts")
    if any(result["decisions"] != result["legal_decisions"] for result in results.values()):
        raise AssertionError("N18 runtime audit encountered an illegal action")

    report = {
        "schema": "rocket-n18-packaged-runtime-audit/v1",
        "arms_manifest_sha256": ARMS_SHA,
        "evidence_audit_sha256": AUDIT_SHA,
        "replay_corpus_sha256": CORPUS_SHA,
        "evaluator_sha256": EVALUATOR_SHA,
        "worker_sha256": WORKER_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "sample_selection": {
            "rule": "first three sorted direct V177 games in each live-seat/live-outcome stratum",
            "games_per_seat_outcome": GAMES_PER_SEAT_OUTCOME,
            "games": sample,
        },
        "official_engine_observations": True,
        "packaged_worker_execution": True,
        "results": results,
        "elapsed_seconds": time() - started,
        "official_match_outcomes_collected": False,
        "development_authority": False,
        "deployment_authority": False,
        "claim_boundary": (
            "Every packaged arm is legal on the frozen official-engine sample. Historical replay "
            "disagreement is retained because the N10 audit already established restart-level "
            "nonreproducibility; it is not treated as a parity gate. Challenger disagreement with "
            "the same-run control is descriptive only, and no counterfactual strength or match "
            "outcome is inferred."
        ),
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "output_sha256": digest(OUTPUT),
                "games": len(games),
                "decisions_per_arm": next(iter(decision_counts)),
                "set_mismatches": {
                    name: result["control_set_mismatches"] for name, result in results.items()
                },
                "ordered_mismatches": {
                    name: result["control_ordered_mismatches"] for name, result in results.items()
                },
                "deployment_authority": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
