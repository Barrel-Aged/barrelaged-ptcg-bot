#!/usr/bin/env python3
"""Audit the exact archive submitted as the recovery winner.

This binds the arena result to immutable bytes, checks the frozen V4 component
hashes, validates archive safety and deck integrity, executes the packaged
official-engine legality check, and captures a live leaderboard snapshot
without exposing arena credentials.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
from typing import Any


WINNER_BOT = "HelbrosMarnieGeneralRecoveryV4"
SOURCE_BOT = "AshMarnieGeneralV151 plus published seed-2 residual"
EXPECTED_ARCHIVE_BYTES = 18_820_579
EXPECTED_ARCHIVE_SHA256 = (
    "8798eca1ff3bfea1c7e656c1225275229476f9e1d69e59957a448b3dc35f85c5"
)
EXPECTED_KEY_HASHES = {
    "deck.csv": "8e431d4e9a52fdbeb5c507326ddf4d992896c720272a67d25b0f662afcd4dc2b",
    "history_residual.pt": "25ee55a36d35d771997de61e61f71395f882178668317b8b5ffee4ccd4ff181c",
    "history_residual_seed2.pt": "9947415b659ad6b43c42cbd1b7142e5249b2955b388c80fd0c01f2942cb0fe4a",
    "main.py": "d077ff28d8a267bf3f2c28fb26e0ae798dedeac5c1d9274627cc8c9e45243533",
    "model.pt": "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c",
    "soldier/history_residual_ensemble.py": "2c3f04bf0e2e09d7aae9ed3f4112c39813f90c399a931a1f4a3762e40d720396",
    "soldier/public_matchup_router.py": "c9d1433888b279a9534e87826e49b83cbd046ece44a208d60be1d64b98151e64",
}
EXPECTED_MEMBERS = {
    "cg/__init__.py", "cg/api.py", "cg/cg.dll", "cg/game.py", "cg/libcg.so",
    "cg/sim.py", "cg/utils.py", "deck.csv", "deck_library.json",
    "history_residual.pt", "history_residual_seed2.pt", "main.py", "model.pt",
    "soldier/__init__.py",
    "soldier/arch_priors.py", "soldier/belief.py", "soldier/cards.py",
    "soldier/decklists.py", "soldier/engine.py", "soldier/features.py",
    "soldier/history_residual.py", "soldier/history_residual_ensemble.py",
    "soldier/net.py", "soldier/net_v10.py", "soldier/public_belief.py",
    "soldier/public_matchup_router.py",
}
EXPECTED_DECK_COUNTS = {
    7: 10, 646: 4, 647: 3, 648: 3, 112: 4, 860: 2, 104: 2,
    1086: 4, 1152: 4, 1227: 4, 1219: 4, 1259: 4, 1079: 3,
    1097: 3, 1182: 2, 1161: 2, 1080: 1, 1231: 1,
}
DEFAULT_ARENA_LAB = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _arena_snapshot(arena_lab: Path) -> dict[str, Any]:
    sys.path.insert(0, str((arena_lab / "tools").resolve()))
    import arena  # type: ignore

    config = arena._load_json(arena.DEFAULT_CONFIG)
    ratings = arena._request(config, "GET", "/ratings")
    status = arena._request(config, "GET", "/status")
    ranked = sorted(
        ratings.items(), key=lambda item: item[1].get("rating", 0), reverse=True
    )
    row = ratings.get(WINNER_BOT)
    rank = next((index for index, (name, _) in enumerate(ranked, 1) if name == WINNER_BOT), None)
    return {
        "captured_bot": WINNER_BOT,
        "rank": rank,
        "entry": row,
        "leader": {"bot": ranked[0][0], **ranked[0][1]} if ranked else None,
        "top_five": [
            {"rank": index, "bot": name, **entry}
            for index, (name, entry) in enumerate(ranked[:5], 1)
        ],
        "active": WINNER_BOT in status.get("submissions", []),
        "arena_phase": status.get("phase"),
        "last_run": status.get("last_run"),
    }


def _runtime_probe(extracted: Path) -> dict[str, Any]:
    code = r'''import json
import main
from soldier.cards import validate_deck
deck = main.agent({"select": None})
ok, message = validate_deck(deck)
print(json.dumps({"deck": deck, "legal": ok, "message": message}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=extracted,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("packaged runtime probe emitted no JSON")
    return json.loads(lines[-1])


def _local_match_audit(paths: list[Path]) -> dict[str, Any]:
    games = 0
    terminal = 0
    bad_reasons: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            games += 1
            reason = str(row.get("reason"))
            if reason == "terminal":
                terminal += 1
            else:
                bad_reasons[reason] += 1
    return {
        "files": [str(path.resolve()) for path in paths if path.exists()],
        "games": games,
        "terminal_games": terminal,
        "nonterminal_reasons": dict(sorted(bad_reasons.items())),
        "zero_runtime_or_legality_failures": games > 0 and terminal == games,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.archive.read_bytes()
    archive_hash = _sha256(raw)
    checks: dict[str, bool] = {
        "archive_size_matches_frozen_v4": len(raw) == EXPECTED_ARCHIVE_BYTES,
        "archive_sha256_matches_frozen_v4": archive_hash == EXPECTED_ARCHIVE_SHA256,
    }

    payloads: dict[str, bytes] = {}
    unsafe: list[str] = []
    links: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name.replace("\\", "/") for member in members]
        for member, name in zip(members, names):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                unsafe.append(name)
            if member.issym() or member.islnk():
                links.append(name)
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(f"could not read {name}")
                payloads[name] = handle.read()

    file_names = set(payloads)
    checks.update(
        {
            "exact_member_set": file_names == EXPECTED_MEMBERS,
            "no_duplicate_members": len(names) == len(set(names)),
            "no_unsafe_paths": not unsafe,
            "no_links": not links,
        }
    )
    key_hashes = {name: _sha256(payloads[name]) for name in EXPECTED_KEY_HASHES}
    checks["key_hashes_match_frozen_v4"] = key_hashes == EXPECTED_KEY_HASHES

    deck = [int(line) for line in payloads["deck.csv"].decode("utf-8").splitlines() if line.strip()]
    deck_counts = Counter(deck)
    checks["deck_has_60_cards"] = len(deck) == 60
    checks["deck_counts_match_published"] = dict(deck_counts) == EXPECTED_DECK_COUNTS

    runtime = _runtime_probe(args.extracted.resolve())
    checks["runtime_returns_archived_deck"] = runtime.get("deck") == deck
    checks["official_engine_accepts_deck"] = runtime.get("legal") is True

    local_matches = _local_match_audit(args.local_match)
    checks["local_official_matches_clean"] = local_matches["zero_runtime_or_legality_failures"]
    arena = _arena_snapshot(args.arena_lab.resolve())
    checks["winner_is_active"] = arena["active"] is True
    checks["winner_is_rank_one"] = arena["rank"] == 1

    result = {
        "schema": "recovery-winner-audit/v2",
        "winner_bot": WINNER_BOT,
        "source_bot": SOURCE_BOT,
        "provenance": {
            "source": (
                "published Ash V151 bundle, published independent seed-2 residual, "
                "and project-local public-information router"
            ),
            "source_v151_bundle_sha256": (
                "744fc3511ac965903ae9635f896c3c8ce6ad9af6954dc8d0c1c6f85ece5409a4"
            ),
            "frozen_v4_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "frozen_v4_archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "policy_description": (
                "frozen neural Marnie blueprint plus synchronized seed-1 and "
                "two-seed recurrent policies; public-card matchup routing; "
                "attack-safe greedy legality guard"
            ),
            "submission_change": (
                "added output-space residual ensemble and a router based only on "
                "opponent cards visible in public zones"
            ),
        },
        "archive": {
            "path": str(args.archive.resolve()),
            "bytes": len(raw),
            "sha256": archive_hash,
            "members": len(payloads),
            "key_hashes": key_hashes,
            "unsafe_paths": unsafe,
            "links": links,
        },
        "deck": {
            "cards": len(deck),
            "counts_by_card_id": {str(key): value for key, value in sorted(deck_counts.items())},
            "engine_legal": runtime.get("legal"),
            "engine_message": runtime.get("message"),
        },
        "local_official_match_audit": local_matches,
        "arena_snapshot": arena,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arena-lab", type=Path, default=DEFAULT_ARENA_LAB)
    parser.add_argument("--local-match", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    result = audit(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
