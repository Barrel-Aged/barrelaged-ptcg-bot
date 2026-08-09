#!/usr/bin/env python3
"""Batch-audit every N9 action against the exact packaged greedy semantics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from collect_n9_balanced_replays import BOT, LOCKED_SEEDS, replay_tree_digest
from train_live_outcome_awr import extract_examples, read_game, reference_parity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "candidates"
    / "leader_bc_scale_v2"
)
DEFAULT_REPLAYS = PROJECT_ROOT / "work" / "n9_balanced_exact_replays"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery"
    / "n9_balanced_pcgrad_awr" / "exact_agent_parity.json"
)
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
GROUPS = (
    "AshMarnieGeneralV110_extracted",
    "HelbrosMarnieGeneralRecoveryV4_extracted",
)


@dataclass(frozen=True)
class AuditGame:
    path: Path
    won: bool
    group: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def structural_audit(path: Path) -> tuple[AuditGame, Counter[str]]:
    rows = records(path)
    players = list((rows[0].get("meta") or {}).get("players") or ())
    if len(players) != 2 or players[1] != BOT or players[0] not in GROUPS:
        raise ValueError(f"N9 bot/opponent identity mismatch: {path}")
    terminal = rows[-1].get("result") or {}
    winner = terminal.get("winner")
    if winner not in (None, 0, 1):
        raise ValueError(f"N9 terminal winner malformed: {path}")
    counts: Counter[str] = Counter()
    first_player = None
    for row in rows[1:-1]:
        if int(row.get("p", -1)) != 1:
            continue
        observation = row.get("obs") or {}
        current = observation.get("current") or {}
        observed_first = current.get("firstPlayer")
        if observed_first in (0, 1):
            first_player = int(observed_first)
        selection = observation.get("select") or {}
        options = selection.get("option") or ()
        expected = list(row.get("action") or ())
        lower = min(max(0, int(selection.get("minCount") or 0)), len(options))
        upper = min(max(0, int(selection.get("maxCount") or 0)), len(options))
        legal = (
            len(expected) == len(set(expected))
            and lower <= len(expected) <= upper
            and all(isinstance(index, int) and 0 <= index < len(options) for index in expected)
        )
        if not legal or len(expected) != upper:
            raise ValueError(f"N9 recorded non-greedy-length/illegal action: {path}")
        counts["decisions"] += 1
        if len(options) <= 1:
            deterministic = [] if upper == 0 else [0]
            if expected != deterministic:
                raise ValueError(f"N9 trivial action mismatch: {path}")
            counts["trivial_decisions"] += 1
        elif upper == len(options):
            if sorted(expected) != list(range(upper)):
                raise ValueError(f"N9 forced-set action mismatch: {path}")
            counts["forced_set_decisions"] += 1
        else:
            counts["model_decisions"] += 1
            counts["model_action_steps"] += len(expected)
    if first_player != 0:
        raise ValueError(f"N9 replay lacks verified second-seat marker: {path}")
    counts["draws"] += int(winner is None)
    return AuditGame(path, winner == 1, players[0]), counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size != 256:
        raise ValueError("N9 parity batch size differs from lock")
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    replays = args.replays.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing N9 parity report: {output}")
    if digest(candidate / "model.pt") != MODEL_SHA256 or digest(candidate / "deck.csv") != DECK_SHA256:
        raise ValueError("N9 parity substrate changed")
    manifests = []
    for path in sorted(replays.glob("*.manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "rocket-n9-exact-replay-shard/v1":
            raise ValueError(f"unexpected N9 shard manifest: {path}")
        if int(payload.get("seed", -1)) not in LOCKED_SEEDS or int(payload.get("games", -1)) != 300:
            raise ValueError(f"N9 shard schedule mismatch: {path}")
        shard = replays / path.name.removesuffix(".manifest.json")
        paths = sorted(shard.glob("*.jsonl"))
        if len(paths) != 300 or replay_tree_digest(paths) != payload.get("replay_tree_sha256"):
            raise ValueError(f"N9 replay tree changed: {shard}")
        listed = {str(row["file"]): str(row["sha256"]) for row in payload.get("replays") or ()}
        if any(listed.get(item.name) != digest(item) for item in paths):
            raise ValueError(f"N9 replay file hash changed: {shard}")
        manifests.append({"path": str(path), "sha256": digest(path), **payload})
    if len(manifests) != 4 or {int(item["seed"]) for item in manifests} != LOCKED_SEEDS:
        raise ValueError("N9 does not have all four locked manifests")
    replay_paths = sorted(replays.rglob("*.jsonl"))
    if len(replay_paths) != 1200:
        raise ValueError(f"expected 1200 N9 replays, found {len(replay_paths)}")
    games = []
    totals: Counter[str] = Counter()
    for index, path in enumerate(replay_paths, start=1):
        game, counts = structural_audit(path)
        games.append(game)
        totals.update(counts)
        totals["games"] += 1
        if index == 1 or index % 100 == 0 or index == len(replay_paths):
            print(json.dumps({"structured": index, "target": len(replay_paths)}), flush=True)
    sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import load_policy  # type: ignore

    deck = [int(value) for value in (candidate / "deck.csv").read_text().split()]
    priors = json.loads((candidate / "deck_library.json").read_text(encoding="utf-8"))
    model = load_policy(candidate / "model.pt", map_location="cpu").cpu().eval()
    examples, extraction = extract_examples(games, BOT, deck, priors, Belief, features)
    if int(extraction.get("invalid", 0)) or int(extraction.get("unsupported_optional_or_early_stop", 0)):
        raise ValueError(f"N9 extraction found unsupported actions: {dict(extraction)}")
    if int(extraction.get("supported_decisions", -1)) != totals["model_decisions"]:
        raise ValueError("N9 structural/model decision counts disagree")
    if len(examples) != totals["model_action_steps"]:
        raise ValueError("N9 autoregressive action-step counts disagree")
    parity = reference_parity(model, examples, args.batch_size, torch.device("cpu"))
    if int(parity["mismatched"]):
        raise ValueError(f"N9 contains off-policy model choices: {parity}")
    totals["matched"] = totals["decisions"]
    totals["mismatched"] = 0
    totals["games_with_mismatch"] = 0
    report = {
        "schema": "rocket-n9-exact-agent-parity/v1",
        "protocol": "rocket-balanced-exact-pcgrad-awr/v1",
        "candidate": str(candidate),
        "reference_model_sha256": MODEL_SHA256,
        "deck_sha256": DECK_SHA256,
        "replays": str(replays),
        "recorded_bot": BOT,
        "replay_files": len(replay_paths),
        "combined_replay_tree_sha256": replay_tree_digest(replay_paths),
        "shard_manifests": manifests,
        "totals": dict(totals),
        "extraction": dict(extraction),
        "batched_reference_parity": parity,
        "equivalence_rules": {
            "trivial_or_zero_option": "canonical deterministic action checked directly",
            "forced_full_set": "complete option set checked directly; order is engine-equivalent",
            "other": "every autoregressive action step checked against frozen model logits",
        },
        "auditor_sha256": digest(Path(__file__).resolve()),
        "development_only": True,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "totals": dict(totals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
