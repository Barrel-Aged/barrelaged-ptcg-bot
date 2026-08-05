#!/usr/bin/env python3
"""Freeze and audit a fresh-leader teacher / exact-Rocket retention corpus."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
AM = ROOT / "work/n18/am"
SZ = ROOT / "work/n18/sz"
LIVE = ROOT / "work/n18/blive"
LEADER = ROOT / "work/n18/bam"
N18_AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "N18_NEW_LEADER_EVIDENCE_AUDIT.json"
)
N18_RESULT = ROOT / "N18_CROSS_DECK_LEADER_TRANSFER_RESULT.md"
OUTPUT = (
    ROOT
    / "artifacts/team_rocket_recovery/n19_retention_teacher/"
    "N19_CORPUS_AUDIT.json"
)

STRATEGY_NAMES = {
    "AshMarnieGeneralV177MultiFailure",
    "SparkZoroarkArenaV2",
}
LIVE_NAME = "HelbrosRocketSpidopsScaleBCV2"
N18_AUDIT_SHA = "f0441e0e602421c0ccbb57447e9b5857af442c93e7b6a02224cd4f09c24e9a0c"
N18_RESULT_SHA = "77f02de76e3b7e7f840800cbae4a6580f7af5ffe1dc0ff43fab8452cb855aa15"
LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
LEADER_MODEL_SHA = "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
CORPUS_SHA = {
    str(AM.resolve()): "05fa90550b4720f44cb8b100c7d8b34a54c65957daca2d8c3df8444bbe719efe",
    str(SZ.resolve()): "77693230f0f04717083622a27732e54e52a588d58663768c5f746c3726f4c210",
}

TEACHER_PER_SEAT = 60
RETENTION_PER_SEAT_OUTCOME = 30
SPLIT_COUNTS_TEACHER = {"train": 48, "validation": 6, "test": 6}
SPLIT_COUNTS_RETENTION = {"train": 24, "validation": 3, "test": 3}


sys.path.insert(0, str((ROOT / "scripts").resolve()))
from train_public_leader_v110_bc import Example, evaluate  # noqa: E402


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


def canonical_games() -> list[dict[str, Any]]:
    by_payload: dict[str, Path] = {}
    for root in (AM, SZ):
        for path in sorted(root.glob("*.jsonl")):
            by_payload.setdefault(digest(path), path)
    games: list[dict[str, Any]] = []
    for payload_sha, path in sorted(by_payload.items()):
        rows = load_rows(path)
        meta = rows[0]["meta"]
        players = list(meta["players"])
        decks = [[int(card) for card in deck] for deck in meta["decks"]]
        result = next(row["result"] for row in rows if "result" in row)
        winner_raw = result.get("winner")
        winner = int(winner_raw) if winner_raw in (0, 1) else -1
        games.append(
            {
                "payload_sha256": payload_sha,
                "path": path,
                "rows": rows,
                "players": players,
                "decks": decks,
                "winner": winner,
            }
        )
    return games


def split_selected(items: list[dict[str, Any]], counts: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    expected = sum(counts.values())
    if len(items) < expected:
        raise ValueError(f"selection stratum has {len(items)} games, needs {expected}")
    selected = sorted(items, key=lambda item: item["payload_sha256"])[:expected]
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        count = counts[split]
        result[split] = selected[offset : offset + count]
        offset += count
    return result


def apply_attack_guard(feat: dict[str, np.ndarray], selection: dict[str, Any]) -> dict[str, np.ndarray]:
    options = list(selection.get("option") or [])
    if not any(int(option.get("type", -1)) == 13 for option in options):
        return feat
    end_indices = [
        index for index, option in enumerate(options) if int(option.get("type", -1)) == 14
    ]
    if not end_indices:
        return feat
    guarded = {key: value.copy() for key, value in feat.items()}
    available = guarded.get(
        "opt_avail", np.ones(len(options), dtype=np.float32)
    ).copy()
    available[end_indices] = 0.0
    guarded["opt_avail"] = available
    return guarded


def extract_examples(
    games: list[dict[str, Any]],
    role: str,
    deck_library: Any,
    belief_class: Any,
    features_module: Any,
) -> tuple[list[Example], Counter[str]]:
    examples: list[Example] = []
    counts: Counter[str] = Counter()
    for game in games:
        players = game["players"]
        if role == "teacher":
            seat = game["winner"]
            if seat not in (0, 1) or players[seat] not in STRATEGY_NAMES:
                raise ValueError("invalid N19 teacher game")
        elif role == "retention":
            if LIVE_NAME not in players:
                raise ValueError("invalid N19 retention game")
            seat = players.index(LIVE_NAME)
        else:
            raise ValueError(role)
        belief = belief_class(game["decks"][seat], opponent_deck_priors=deck_library)
        for row in game["rows"]:
            if "obs" not in row or int(row.get("p", -1)) != seat:
                continue
            observation = row["obs"]
            selection = observation.get("select") or {}
            options = list(selection.get("option") or [])
            action = row.get("action")
            if not isinstance(action, list):
                counts["invalid_nonlist"] += 1
                continue
            lower = int(selection.get("minCount") or 0)
            upper = min(int(selection.get("maxCount") or 0), len(options))
            legal = (
                len(action) == len(set(action))
                and lower <= len(action) <= upper
                and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
            )
            if not legal:
                counts["invalid_illegal"] += 1
                continue
            if len(options) <= 1:
                counts["trivial"] += 1
                continue
            if not action:
                counts["unsupported_empty"] += 1
                continue
            if len(action) != upper:
                counts["unsupported_early_stop"] += 1
                continue
            try:
                base = features_module.extract(observation, belief=belief, slot_link=True)
                base = apply_attack_guard(base, selection)
                selected: list[int] = []
                for target in action:
                    conditioned = features_module.mark_selection(base, selected, upper)
                    examples.append(
                        Example(conditioned, int(target), game["payload_sha256"])
                    )
                    selected.append(int(target))
                    counts["choice_targets"] += 1
                counts["supported_decisions"] += 1
            except Exception as error:
                counts[f"feature_error:{type(error).__name__}"] += 1
    counts["examples"] = len(examples)
    return examples, counts


def game_record(item: dict[str, Any], role: str, split: str) -> dict[str, Any]:
    if role == "teacher":
        seat = item["winner"]
        target = item["players"][seat]
        outcome = "strategy_win"
    else:
        seat = item["players"].index(LIVE_NAME)
        target = LIVE_NAME
        outcome = "win" if item["winner"] == seat else "loss"
    return {
        "payload_sha256": item["payload_sha256"],
        "file": item["path"].name,
        "source_directory": str(item["path"].parent.resolve()),
        "role": role,
        "split": split,
        "players": item["players"],
        "target_actor": target,
        "target_seat": seat,
        "outcome": outcome,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N19 corpus audit: {OUTPUT}")
    for path, expected in ((N18_AUDIT, N18_AUDIT_SHA), (N18_RESULT, N18_RESULT_SHA)):
        if digest(path) != expected:
            raise ValueError(f"N19 source evidence changed: {path}")
    if digest(LIVE / "model.pt") != LIVE_MODEL_SHA or digest(LIVE / "deck.csv") != DECK_SHA:
        raise ValueError("N19 live source changed")
    if digest(LEADER / "model.pt") != LEADER_MODEL_SHA:
        raise ValueError("N19 leader source changed")
    for corpus, expected in CORPUS_SHA.items():
        if corpus_digest(Path(corpus)) != expected:
            raise ValueError(f"N19 replay corpus changed: {corpus}")

    started = time()
    games = canonical_games()
    teacher_by_seat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    retention_by_stratum: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        players = game["players"]
        if (
            game["winner"] in (0, 1)
            and players[game["winner"]] in STRATEGY_NAMES
            and LIVE_NAME not in players
        ):
            teacher_by_seat[game["winner"]].append(game)
        if LIVE_NAME in players and game["winner"] in (0, 1):
            live_seat = players.index(LIVE_NAME)
            outcome = "win" if game["winner"] == live_seat else "loss"
            retention_by_stratum[(live_seat, outcome)].append(game)

    teacher_splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seat in (0, 1):
        selected = split_selected(teacher_by_seat[seat], SPLIT_COUNTS_TEACHER)
        for split, items in selected.items():
            teacher_splits[split].extend(items)
    retention_splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seat in (0, 1):
        for outcome in ("win", "loss"):
            selected = split_selected(
                retention_by_stratum[(seat, outcome)], SPLIT_COUNTS_RETENTION
            )
            for split, items in selected.items():
                retention_splits[split].extend(items)

    selected_hashes = [
        item["payload_sha256"]
        for split in ("train", "validation", "test")
        for item in teacher_splits[split] + retention_splits[split]
    ]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise ValueError("N19 teacher and retention selections overlap")

    sys.path.insert(0, str(LIVE.resolve()))
    from soldier import features  # type: ignore  # noqa: E402
    from soldier.belief import Belief  # type: ignore  # noqa: E402
    from soldier.net import load_policy  # type: ignore  # noqa: E402

    deck_library = json.loads((LIVE / "deck_library.json").read_text(encoding="utf-8"))
    models = {
        "live": load_policy(LIVE / "model.pt", map_location="cpu").cpu().eval(),
        "leader_base": load_policy(LEADER / "model.pt", map_location="cpu").cpu().eval(),
    }
    torch.set_num_threads(6)
    extraction: dict[str, Any] = {"teacher": {}, "retention": {}}
    metrics: dict[str, Any] = {"teacher": {}, "retention": {}}
    game_manifest: list[dict[str, Any]] = []
    for role, splits in (("teacher", teacher_splits), ("retention", retention_splits)):
        for split in ("train", "validation", "test"):
            examples, counts = extract_examples(
                splits[split], role, deck_library, Belief, features
            )
            if not examples or counts["invalid_illegal"] or counts["invalid_nonlist"]:
                raise ValueError(f"invalid N19 extraction: {role}/{split}: {counts}")
            extraction[role][split] = dict(counts)
            metrics[role][split] = {
                name: evaluate(model, examples, 96, torch.device("cpu"))
                for name, model in models.items()
            }
            game_manifest.extend(
                game_record(item, role, split) for item in splits[split]
            )

    teacher_opponents = Counter()
    for item in game_manifest:
        if item["role"] == "teacher":
            opponent = next(
                player for player in item["players"] if player != item["target_actor"]
            )
            teacher_opponents[opponent] += 1

    report = {
        "schema": "rocket-n19-retention-teacher-corpus-audit/v1",
        "purpose": "pretraining corpus and baseline audit only",
        "source_n18_audit_sha256": N18_AUDIT_SHA,
        "source_n18_result_sha256": N18_RESULT_SHA,
        "source_replay_corpora_sha256": CORPUS_SHA,
        "canonical_deduplicated_games": len(games),
        "selection": {
            "teacher_rule": (
                "first 60 payload hashes per winning strategy seat, excluding every game containing live Rocket"
            ),
            "retention_rule": (
                "first 30 payload hashes per live Rocket seat/outcome stratum"
            ),
            "teacher_split_counts_per_seat": SPLIT_COUNTS_TEACHER,
            "retention_split_counts_per_seat_outcome": SPLIT_COUNTS_RETENTION,
            "teacher_games": sum(len(items) for items in teacher_splits.values()),
            "retention_games": sum(len(items) for items in retention_splits.values()),
            "overlap_games": 0,
        },
        "teacher_opponents": dict(sorted(teacher_opponents.items())),
        "extraction": extraction,
        "baseline_metrics": metrics,
        "game_manifest": sorted(
            game_manifest,
            key=lambda item: (item["role"], item["split"], item["payload_sha256"]),
        ),
        "live_model_sha256": LIVE_MODEL_SHA,
        "leader_model_sha256": LEADER_MODEL_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "all_recorded_actions_legal": True,
        "hidden_information_beyond_actor_observation_used": False,
        "training_performed": False,
        "official_match_panel_performed": False,
        "development_authority": False,
        "deployment_authority": False,
        "elapsed_seconds": time() - started,
        "claim_boundary": (
            "Recorded official-engine actions and offline imitation metrics only. No training, "
            "counterfactual value, Team Rocket improvement, or deployment claim."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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
                "teacher_games": report["selection"]["teacher_games"],
                "retention_games": report["selection"]["retention_games"],
                "teacher_validation": metrics["teacher"]["validation"],
                "retention_validation": metrics["retention"]["validation"],
                "deployment_authority": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
