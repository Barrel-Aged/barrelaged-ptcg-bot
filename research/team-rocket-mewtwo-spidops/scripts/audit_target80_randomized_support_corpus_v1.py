#!/usr/bin/env python3
"""Audit all 18 locked Target-80 shards without reporting outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_target80_randomized_support_v1 import audit as audit_shard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = PROJECT_ROOT / "work" / "target80_randomized_support_v1_corpus"
SCHEMA = "target80-randomized-support-corpus-audit/v1"
COLLECTOR_SHA256 = "d6e72dc79f3a92b49b718ae88e0685227f223d82641ce7aee10bc4bc53b09ece"
PROTOCOL_SHA256 = "044720b2c874fce7527a19d45877fa579fcc897db808af67de40876e9ef6af84"
CANDIDATE_TREE_SHA256 = "30509428ada5cb3b1948cc53f83f9093550c70138c140726a855c00fb10f389f"
CANDIDATE_DECK_SHA256 = "4781b32cf3f945cafb3bcbb8a6a8e892244fa74cba0f9e0f7e43f68719f983ea"
CANDIDATE_MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
CANDIDATE_MAIN_SHA256 = "80be7541439fa9655905607f73a9072c4eb09bfd1f21cad4b4b2cce91eccacf6"
SPLITS = ("train", "validation", "test")
OPPONENTS = ("marnie", "starmie", "alakazam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus_digest(manifests: list[Path], replay_hashes: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for path in sorted(manifests, key=lambda item: item.name):
        name = path.name.encode("utf-8")
        fingerprint = sha256(path).encode("ascii")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(fingerprint)
    for name_text, fingerprint_text in sorted(replay_hashes):
        name = name_text.encode("utf-8")
        fingerprint = fingerprint_text.encode("ascii")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(fingerprint)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    root = args.corpus.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing corpus audit: {output}")
    expected = {
        f"{split}_{opponent}_s{seat}.manifest.json"
        for split in SPLITS
        for opponent in OPPONENTS
        for seat in (0, 1)
    }
    actual = {path.name for path in root.glob("*.manifest.json")}
    if actual != expected:
        raise ValueError(
            f"locked manifest set differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    totals: Counter[str] = Counter()
    cells: dict[str, dict[str, int | str | bool]] = {}
    replay_names: set[str] = set()
    replay_hash_set: set[str] = set()
    replay_hashes: list[tuple[str, str]] = []
    shard_paths: list[Path] = []
    opponent_seat_exploration: Counter[str] = Counter()
    for split in SPLITS:
        for opponent in OPPONENTS:
            for seat in (0, 1):
                key = f"{split}_{opponent}_s{seat}"
                manifest_path = root / f"{key}.manifest.json"
                audited = audit_shard(manifest_path, False)
                manifest: dict[str, Any] = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                frozen = {
                    "collector_sha256": COLLECTOR_SHA256,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "candidate_bundle_sha256": CANDIDATE_TREE_SHA256,
                    "candidate_deck_sha256": CANDIDATE_DECK_SHA256,
                    "candidate_model_sha256": CANDIDATE_MODEL_SHA256,
                    "candidate_main_sha256": CANDIDATE_MAIN_SHA256,
                }
                for field, value in frozen.items():
                    if manifest.get(field) != value:
                        raise ValueError(f"{key}: frozen {field} mismatch")
                shard_paths.append(manifest_path)
                cell = {
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256(manifest_path),
                    "games": int(audited["games"]),
                    "decisions": int(audited["decisions"]),
                    "own_decisions": int(audited["own_decisions"]),
                    "eligible": int(audited["eligible"]),
                    "explored": int(audited["explored"]),
                    "guarded": int(audited["guarded"]),
                    "replay_tree_sha256": str(audited["replay_tree_sha256"]),
                    "audit_passed": True,
                }
                cells[key] = cell
                for field in ("games", "decisions", "own_decisions", "eligible", "explored", "guarded"):
                    totals[field] += int(cell[field])
                opponent_seat_exploration[f"{opponent}_s{seat}"] += int(cell["explored"])
                for replay in manifest.get("replays") or []:
                    name = f"{key}/{replay['file']}"
                    fingerprint = str(replay["sha256"])
                    if name in replay_names or fingerprint in replay_hash_set:
                        raise ValueError(f"duplicate replay name or digest: {name}")
                    replay_names.add(name)
                    replay_hash_set.add(fingerprint)
                    replay_hashes.append((name, fingerprint))
    if totals["games"] != 1_200:
        raise ValueError(f"expected 1,200 games, found {totals['games']}")
    if totals["eligible"] < 500 or totals["explored"] < 60:
        raise ValueError("global intervention-support gate failed")
    if any(value <= 0 for value in opponent_seat_exploration.values()):
        raise ValueError("an opponent/seat cell lacks realized interventions")
    report = {
        "schema": SCHEMA,
        "corpus": str(root),
        "corpus_sha256": corpus_digest(shard_paths, replay_hashes),
        "manifests": len(shard_paths),
        "manifest_set_exact": True,
        "games": totals["games"],
        "replays": len(replay_hashes),
        "decisions": totals["decisions"],
        "own_decisions": totals["own_decisions"],
        "hidden_boundary_audits": totals["own_decisions"],
        "eligible_decisions": totals["eligible"],
        "explored_decisions": totals["explored"],
        "guarded_decisions": totals["guarded"],
        "opponent_seat_exploration": dict(sorted(opponent_seat_exploration.items())),
        "cells": cells,
        "all_shard_audits_passed": True,
        "all_frozen_hashes_passed": True,
        "duplicate_names": 0,
        "duplicate_hashes": 0,
        "outcomes_reported": False,
        "data_quality_admitted": True,
        "development_only": True,
        "deployment_authority": False,
        "auditor_sha256": sha256(Path(__file__).resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

