#!/usr/bin/env python3
"""Seal the Team Rocket rank-one-at-300 result only after every gate passes.

The monitor is intentionally necessary but insufficient.  This finalizer also
queries the current arena ratings, audits the exact submitted archive, and runs
that archive through the bundled official ``cg`` engine.  It writes durable
evidence only after all checks succeed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARENA_LAB = Path(r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab")
ARENA_CLI = ARENA_LAB / "tools" / "arena.py"
EVALUATOR = ARENA_LAB / "tools" / "evaluate.py"
STATUS_PATH = ROOT / "work" / "arena_rank_goal_monitor" / "status.json"
FINAL_DIR = ROOT / "artifacts" / "team_rocket_rank1_300" / "final_audit"

BOT = "HelbrosRocketSpidopsScaleBCV2"
SECONDARY_BOT = "HelbrosRocketSpidopsScaleStopV3"
TARGET_GAMES = 300
MONITOR_SCHEMA = "rocket-arena-rank-goal-monitor/v2"
AUDIT_SCHEMA = "rocket-arena-rank1-300-final-audit/v1"
SEAL_SCHEMA = "rocket-arena-rank1-300-evidence-seal/v1"
DECK_SHA256 = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
MODEL_SHA256 = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
ARCHIVE_SHA256 = "27dff66b6d39d780e56a745c34039305ce5e76e0a17811253ed4315698f2c6f9"
SECONDARY_MODEL_SHA256 = "c6b808e3e50b5fffcfd44b59ad2a541c6661e8d22915b1cbb5ade5a57464d8f2"
SECONDARY_ARCHIVE_SHA256 = "366a3b4bf2e646cfb39754fc518183114f8e6b2dd6c0f2f7942946be4a8fcdae"
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024

EXPECTED_MEMBERS = frozenset(
    {
        ".src_mtime",
        "SCALE_TRAINING_PROVENANCE.json",
        "TRAINING_PROVENANCE.json",
        "cg/__init__.py",
        "cg/api.py",
        "cg/cg.dll",
        "cg/game.py",
        "cg/libcg-arm64.so",
        "cg/libcg.dylib",
        "cg/libcg.so",
        "cg/sim.py",
        "cg/utils.py",
        "deck.csv",
        "deck_library.json",
        "main.py",
        "model.pt",
        "soldier/__init__.py",
        "soldier/arch_priors.py",
        "soldier/belief.py",
        "soldier/cards.py",
        "soldier/decklists.py",
        "soldier/engine.py",
        "soldier/features.py",
        "soldier/net.py",
        "soldier/net_v10.py",
        "soldier/public_belief.py",
    }
)
SECONDARY_EXPECTED_MEMBERS = EXPECTED_MEMBERS | {"STOP_TRAINING_PROVENANCE.json"}


@dataclass(frozen=True)
class Target:
    name: str
    source_bundle: Path
    archive: Path
    model_sha256: str
    archive_sha256: str
    expected_members: frozenset[str]


TARGETS = {
    BOT: Target(
        name=BOT,
        source_bundle=(
            ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "leader_bc_scale_v2"
        ),
        archive=ROOT / "work" / "HelbrosRocketSpidopsScaleBCV2_20260728.tar.gz",
        model_sha256=MODEL_SHA256,
        archive_sha256=ARCHIVE_SHA256,
        expected_members=EXPECTED_MEMBERS,
    ),
    SECONDARY_BOT: Target(
        name=SECONDARY_BOT,
        source_bundle=ROOT / "work" / "scale_stop_archive_smoke_v3",
        archive=ROOT / "work" / "HelbrosRocketSpidopsScaleStopV3_20260801.tar.gz",
        model_sha256=SECONDARY_MODEL_SHA256,
        archive_sha256=SECONDARY_ARCHIVE_SHA256,
        expected_members=SECONDARY_EXPECTED_MEMBERS,
    ),
}

# Backward-compatible aliases used by focused tests and audit notes.
ARCHIVE = TARGETS[BOT].archive
SOURCE_BUNDLE = TARGETS[BOT].source_bundle

RATING_ROW = re.compile(
    r"^\s*(?P<rank>\d+)\s+(?P<name>.+?)\s+"
    r"(?P<rating>-?\d+(?:\.\d+)?)\s+"
    r"\((?P<w>\d+)-(?P<l>\d+)-(?P<d>\d+)\)\s*$"
)


class NotReadyError(RuntimeError):
    """The live goal has not yet met every dynamic admission condition."""


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_ratings(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = RATING_ROW.match(line)
        if match is None:
            continue
        values = match.groupdict()
        wins = int(values["w"])
        losses = int(values["l"])
        draws = int(values["d"])
        rows.append(
            {
                "rank": int(values["rank"]),
                "name": values["name"].strip(),
                "rating": float(values["rating"]),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "games": wins + losses + draws,
            }
        )
    if not rows:
        raise ValueError("arena ratings response contained no parsable rows")
    if len({int(row["rank"]) for row in rows}) != len(rows):
        raise ValueError("arena ratings response contains duplicate ranks")
    return rows


def _validate_record(row: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{source} rating row is missing")
    required = ("rank", "name", "rating", "wins", "losses", "draws", "games")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{source} rating row is missing fields: {missing}")
    normalized = {
        "rank": int(row["rank"]),
        "name": str(row["name"]),
        "rating": float(row["rating"]),
        "wins": int(row["wins"]),
        "losses": int(row["losses"]),
        "draws": int(row["draws"]),
        "games": int(row["games"]),
    }
    counted = normalized["wins"] + normalized["losses"] + normalized["draws"]
    if counted != normalized["games"]:
        raise ValueError(f"{source} game count does not equal W+L+D")
    return normalized


def require_qualifying_row(
    row: Any, *, source: str, target: Target = TARGETS[BOT]
) -> dict[str, Any]:
    normalized = _validate_record(row, source=source)
    if normalized["name"] != target.name:
        raise NotReadyError(f"{source} rank-one identity is not {target.name}")
    if normalized["rank"] != 1:
        raise NotReadyError(
            f"{source} has {target.name} at rank {normalized['rank']}, not rank 1"
        )
    if normalized["games"] < TARGET_GAMES:
        raise NotReadyError(
            f"{source} has only {normalized['games']} games for {target.name}; "
            f"{TARGET_GAMES} are required"
        )
    return normalized


def target_admitted_by_monitor(status: Any) -> Target:
    if not isinstance(status, dict):
        raise ValueError("monitor status must be a JSON object")
    goal = status.get("goal_bot")
    name = goal.get("name") if isinstance(goal, dict) else None
    if name not in TARGETS:
        raise NotReadyError("monitor has not admitted a frozen auditable identity")
    return TARGETS[str(name)]


def validate_monitor_status(
    status: Any, target: Target = TARGETS[BOT]
) -> dict[str, Any]:
    if not isinstance(status, dict):
        raise ValueError("monitor status must be a JSON object")
    if status.get("schema") != MONITOR_SCHEMA:
        raise ValueError("monitor status schema is not the frozen v2 schema")
    if status.get("read_only") is not True:
        raise ValueError("monitor status is not marked read-only")
    if int(status.get("target_games", -1)) != TARGET_GAMES:
        raise ValueError("monitor target game count changed")
    if status.get("qualifying_deck_sha256") != DECK_SHA256:
        raise ValueError("monitor qualifying deck hash changed")
    if status.get("goal_met") is not True:
        focal = status.get("bot") or {}
        games = focal.get("games", 0) if isinstance(focal, dict) else 0
        rank = focal.get("rank") if isinstance(focal, dict) else None
        raise NotReadyError(
            f"monitor has not admitted the goal (best exact bot rank={rank}, games={games})"
        )

    goal = status.get("goal_bot")
    if not isinstance(goal, dict):
        raise ValueError("goal_met is true but goal_bot is missing")
    if goal.get("name") != target.name:
        raise NotReadyError("the admitted monitor identity is not the frozen archive identity")
    if goal.get("present") is not True:
        raise NotReadyError("the admitted bot is not currently present on the ladder")
    if goal.get("deck_sha256") != DECK_SHA256:
        raise ValueError("admitted monitor bot deck hash changed")
    if goal.get("model_sha256") != target.model_sha256:
        raise ValueError("admitted monitor bot model hash changed")
    monitor_row = require_qualifying_row(
        goal.get("rating_row"), source="monitor", target=target
    )

    matching = [
        item
        for item in status.get("bots", [])
        if isinstance(item, dict) and item.get("name") == target.name
    ]
    if len(matching) != 1:
        raise ValueError("monitor must contain exactly one frozen bot identity")
    top = status.get("top")
    if not isinstance(top, list) or not top:
        raise ValueError("monitor top table is missing")
    top_row = require_qualifying_row(
        top[0], source="monitor top table", target=target
    )
    if top_row != monitor_row:
        raise ValueError("monitor goal row and rank-one table row disagree")
    return monitor_row


def query_current_ratings() -> tuple[list[dict[str, Any]], str]:
    completed = subprocess.run(
        [sys.executable, str(ARENA_CLI), "ratings"],
        cwd=ARENA_LAB,
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    return parse_ratings(completed.stdout), completed.stdout


def validate_current_ratings(
    rows: Iterable[dict[str, Any]], target: Target = TARGETS[BOT]
) -> dict[str, Any]:
    table = list(rows)
    matches = [row for row in table if str(row.get("name")) == target.name]
    if len(matches) != 1:
        raise NotReadyError(
            f"current arena ratings contain {len(matches)} exact {target.name} rows"
        )
    row = require_qualifying_row(
        matches[0], source="current authoritative arena", target=target
    )
    rank_one = [candidate for candidate in table if int(candidate.get("rank", -1)) == 1]
    if len(rank_one) != 1 or str(rank_one[0].get("name")) != target.name:
        raise NotReadyError("current authoritative rank-one row is not the frozen bot")
    return row


def validate_tar_members(
    members: list[tarfile.TarInfo], expected_members: frozenset[str]
) -> None:
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise ValueError("archive contains duplicate member names")
    actual = set(names)
    if actual != set(expected_members):
        missing = sorted(set(expected_members) - actual)
        extra = sorted(actual - set(expected_members))
        raise ValueError(f"archive member set changed: missing={missing}, extra={extra}")
    total_size = 0
    for member in members:
        path = PurePosixPath(member.name)
        unsafe = (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in member.name
            or re.match(r"^[A-Za-z]:", member.name) is not None
        )
        if unsafe:
            raise ValueError(f"unsafe archive path: {member.name!r}")
        if not member.isfile():
            raise ValueError(f"archive member is not a regular file: {member.name!r}")
        total_size += int(member.size)
    if total_size > MAX_UNCOMPRESSED_BYTES:
        raise ValueError(f"archive expands to {total_size} bytes, above the safety cap")


def audit_archive(target: Target = TARGETS[BOT]) -> dict[str, Any]:
    archive_path = target.archive.resolve()
    if digest(archive_path) != target.archive_sha256:
        raise ValueError("exact submitted archive SHA-256 changed")
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        validate_tar_members(members, target.expected_members)
        uncompressed_bytes = sum(int(member.size) for member in members)
        file_hashes: dict[str, str] = {}
        file_bytes: dict[str, bytes] = {}
        for member in members:
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read archive member: {member.name}")
            raw = source.read()
            if len(raw) != member.size:
                raise ValueError(f"archive member size mismatch: {member.name}")
            file_hashes[member.name] = digest_bytes(raw)
            if member.name in {"deck.csv", "model.pt"}:
                file_bytes[member.name] = raw

    if file_hashes["deck.csv"] != DECK_SHA256:
        raise ValueError("archive deck hash is not the frozen qualifying deck")
    if file_hashes["model.pt"] != target.model_sha256:
        raise ValueError("archive model hash is not the frozen submitted model")
    deck = [int(value) for value in file_bytes["deck.csv"].decode("utf-8").split()]
    if len(deck) != 60:
        raise ValueError(f"archive deck has {len(deck)} cards, not 60")
    return {
        "path": str(archive_path),
        "sha256": target.archive_sha256,
        "member_count": len(file_hashes),
        "members": sorted(file_hashes),
        "member_sha256": dict(sorted(file_hashes.items())),
        "uncompressed_bytes": uncompressed_bytes,
        "deck_cards": len(deck),
        "deck_sha256": file_hashes["deck.csv"],
        "model_sha256": file_hashes["model.pt"],
    }


def _safe_extract(target: Target, destination: Path) -> None:
    archive_path = target.archive
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        validate_tar_members(members, target.expected_members)
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not extract archive member: {member.name}")
            with target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def _query_packaged_deck(extracted: Path) -> list[int]:
    code = (
        "import json, main; "
        "print(json.dumps(list(main.agent({'select': None})), separators=(',', ':')))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=extracted,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    deck = json.loads(completed.stdout.strip().splitlines()[-1])
    if not isinstance(deck, list) or any(not isinstance(card, int) for card in deck):
        raise ValueError("packaged deck query did not return list[int]")
    return deck


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_official_engine_smoke(
    workspace: Path, target: Target = TARGETS[BOT]
) -> dict[str, Any]:
    extracted = workspace / "extracted_archive"
    extracted.mkdir(parents=True)
    _safe_extract(target, extracted)
    packaged_deck = _query_packaged_deck(extracted)
    csv_deck = [
        int(value)
        for value in (extracted / "deck.csv").read_text(encoding="utf-8").split()
    ]
    if packaged_deck != csv_deck or len(packaged_deck) != 60:
        raise ValueError("packaged deck query differs from the frozen deck.csv")

    output = workspace / "official_engine_smoke.jsonl"
    command = [
        sys.executable,
        str(EVALUATOR),
        "--agent-a",
        str(extracted),
        "--agent-b",
        str(extracted),
        "--agent-a-name",
        f"{target.name}-seat-a",
        "--agent-b-name",
        f"{target.name}-seat-b",
        "--sim-dir",
        str(extracted),
        "--games",
        "2",
        "--max-steps",
        "600",
        "--action-timeout-s",
        "30",
        "--out",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ARENA_LAB,
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    records = _read_json_lines(output)
    if len(records) != 2:
        raise ValueError(f"official-engine smoke wrote {len(records)} games, not 2")
    bad = [
        record
        for record in records
        if record.get("reason") != "terminal"
        or int(record.get("steps", 0)) <= 0
        or record.get("result") not in (0, 1, 2)
    ]
    if bad:
        raise ValueError(f"official-engine smoke did not terminate cleanly: {bad}")
    summary_path = output.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "engine": "official cg runtime extracted from the exact submitted archive",
        "network_arena_execution": False,
        "games": len(records),
        "seat_swapped": True,
        "packaged_deck_query_matches_csv": packaged_deck == csv_deck,
        "records": records,
        "summary": summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_sha256": digest_bytes(completed.stdout.encode("utf-8")),
        "stderr_sha256": digest_bytes(completed.stderr.encode("utf-8")),
    }


def _copy_smoke_artifacts(workspace: Path, destination: Path) -> None:
    for name in ("official_engine_smoke.jsonl", "official_engine_smoke.summary.json"):
        shutil.copy2(workspace / name, destination / name)
    logs = workspace / "official_engine_smoke.logs"
    if logs.is_dir():
        shutil.copytree(logs, destination / logs.name)


def build_seal_manifest(directory: Path) -> dict[str, Any]:
    """Fingerprint every evidence file except the manifest itself."""
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative == "seal_manifest.json":
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    if not files:
        raise ValueError("cannot seal an empty evidence directory")
    return {"schema": SEAL_SCHEMA, "files": files}


def validate_seal_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "seal_manifest.json"
    if not path.is_file():
        raise ValueError("final audit seal manifest is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SEAL_SCHEMA or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("final audit seal manifest is malformed")
    expected = manifest["files"]
    actual = build_seal_manifest(directory)["files"]
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"final audit evidence set changed: missing={missing}, extra={extra}")
    for name, expected_value in expected.items():
        if actual[name] != expected_value:
            raise ValueError(f"final audit evidence changed: {name}")
    return manifest


def _markdown(report: dict[str, Any]) -> str:
    row = report["authoritative_arena"]["row"]
    identity = report["identity"]
    return f"""# Team Rocket rank-one-at-300 final audit

Status: **QUALIFIED**  
Sealed: {report['finalized_at_utc']}

`{identity['name']}` was independently observed at current internal-arena rank **1** with
**{row['games']}** rated games ({row['wins']}-{row['losses']}-{row['draws']}) and
rating **{row['rating']}**. The game count belongs to this one exact identity;
no counts were combined across submissions.

## Frozen identity

- deck SHA-256: `{identity['deck_sha256']}`;
- model SHA-256: `{identity['model_sha256']}`;
- submitted archive SHA-256: `{report['archive']['sha256']}`;
- archive: exactly {report['archive']['member_count']} safe regular-file members;
- deck: exactly {report['archive']['deck_cards']} cards.

## Runtime gate

The exact archived package returned the same 60-card deck through its entry
point and completed {report['official_engine_smoke']['games']} seat-swapped
local games through the official `cg` engine with terminal results and no
agent, deck, timeout, or engine-legality error.

## Evidence boundary

The ratings call is genuine current network-arena evidence. The packaged
runtime check is genuine official-engine execution using the `cg` runtime from
the exact submitted archive, but it is a local legality/runtime smoke, not an
additional rated network-arena match or a strength estimate. No Kaggle action
was performed.

Machine-readable evidence is in `audit.json`; raw smoke records and worker logs
are stored beside it. `seal_manifest.json` fingerprints every evidence file.
"""


def _validate_source_bundle(target: Target = TARGETS[BOT]) -> dict[str, Any]:
    deck_hash = digest(target.source_bundle / "deck.csv")
    model_hash = digest(target.source_bundle / "model.pt")
    if deck_hash != DECK_SHA256 or model_hash != target.model_sha256:
        raise ValueError("the locally monitored source bundle changed")
    return {
        "path": str(target.source_bundle.resolve()),
        "deck_sha256": deck_hash,
        "model_sha256": model_hash,
    }


def _existing_seal() -> dict[str, Any] | None:
    path = FINAL_DIR / "audit.json"
    if not FINAL_DIR.exists():
        return None
    if not path.is_file() or not (FINAL_DIR / "report.md").is_file():
        raise ValueError("final audit directory exists but is incomplete")
    validate_seal_manifest(FINAL_DIR)
    report = json.loads(path.read_text(encoding="utf-8"))
    identity = report.get("identity", {})
    target = TARGETS.get(identity.get("name")) if isinstance(identity, dict) else None
    if target is None:
        raise ValueError("existing final audit identity is not a frozen target")
    if (
        report.get("schema") != AUDIT_SCHEMA
        or report.get("qualified") is not True
        or identity.get("deck_sha256") != DECK_SHA256
        or identity.get("model_sha256") != target.model_sha256
        or report.get("archive", {}).get("sha256") != target.archive_sha256
    ):
        raise ValueError("existing final audit seal does not match the frozen goal")
    require_qualifying_row(
        report.get("authoritative_arena", {}).get("row"),
        source="existing seal",
        target=target,
    )
    return report


def finalize() -> dict[str, Any]:
    existing = _existing_seal()
    if existing is not None:
        return existing

    status_raw = STATUS_PATH.read_bytes()
    status = json.loads(status_raw)
    target = target_admitted_by_monitor(status)
    monitor_row = validate_monitor_status(status, target)
    source_bundle = _validate_source_bundle(target)
    archive = audit_archive(target)

    rows_before, ratings_raw_before = query_current_ratings()
    observed_before = datetime.now(timezone.utc).isoformat()
    arena_row_before = validate_current_ratings(rows_before, target)

    work_root = ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rank1-finalizer-", dir=work_root) as temporary:
        workspace = Path(temporary)
        smoke = run_official_engine_smoke(workspace, target)
        rows_after, ratings_raw_after = query_current_ratings()
        finalized_at = datetime.now(timezone.utc).isoformat()
        arena_row = validate_current_ratings(rows_after, target)
        report: dict[str, Any] = {
            "schema": AUDIT_SCHEMA,
            "qualified": True,
            "finalized_at_utc": finalized_at,
            "goal": {
                "rank": 1,
                "minimum_rated_games": TARGET_GAMES,
                "single_identity_only": True,
                "active_presence_required": True,
            },
            "identity": {
                "name": target.name,
                "deck_sha256": DECK_SHA256,
                "model_sha256": target.model_sha256,
            },
            "monitor": {
                "schema": status["schema"],
                "path": str(STATUS_PATH.resolve()),
                "sha256": digest_bytes(status_raw),
                "observed_at_utc": status.get("observed_at_utc"),
                "row": monitor_row,
            },
            "authoritative_arena": {
                "command": [str(ARENA_CLI), "ratings"],
                "observed_at_utc": finalized_at,
                "row": arena_row,
                "pre_smoke": {
                    "observed_at_utc": observed_before,
                    "row": arena_row_before,
                    "raw_stdout": ratings_raw_before,
                    "raw_stdout_sha256": digest_bytes(
                        ratings_raw_before.encode("utf-8")
                    ),
                    "parsed_rows": rows_before,
                },
                "post_smoke": {
                    "observed_at_utc": finalized_at,
                    "row": arena_row,
                    "raw_stdout": ratings_raw_after,
                    "raw_stdout_sha256": digest_bytes(
                        ratings_raw_after.encode("utf-8")
                    ),
                    "parsed_rows": rows_after,
                },
            },
            "source_bundle": source_bundle,
            "archive": archive,
            "official_engine_smoke": smoke,
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": digest(Path(__file__).resolve()),
                "python": sys.version,
            },
            "claim_boundary": (
                "Current ratings are genuine internal-arena network evidence. "
                "The two-game packaged runtime gate is local official-cg-engine "
                "execution from the exact archive, not extra rated arena evidence. "
                "No Kaggle action or unverified performance claim is included."
            ),
        }

        parent = FINAL_DIR.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".final-audit-staging-", dir=parent))
        try:
            (staging / "audit.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (staging / "report.md").write_text(
                _markdown(report), encoding="utf-8", newline="\n"
            )
            (staging / "official_engine_smoke.stdout.txt").write_text(
                smoke["stdout"], encoding="utf-8", newline="\n"
            )
            (staging / "official_engine_smoke.stderr.txt").write_text(
                smoke["stderr"], encoding="utf-8", newline="\n"
            )
            (staging / "monitor_status.json").write_bytes(status_raw)
            (staging / "arena_ratings.pre_smoke.stdout.txt").write_text(
                ratings_raw_before, encoding="utf-8", newline="\n"
            )
            (staging / "arena_ratings.post_smoke.stdout.txt").write_text(
                ratings_raw_after, encoding="utf-8", newline="\n"
            )
            _copy_smoke_artifacts(workspace, staging)
            seal = build_seal_manifest(staging)
            (staging / "seal_manifest.json").write_text(
                json.dumps(seal, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            if FINAL_DIR.exists():
                raise FileExistsError("final audit appeared concurrently; refusing to overwrite")
            staging.rename(FINAL_DIR)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return report


def preflight(bot_name: str = BOT) -> dict[str, Any]:
    """Exercise static identity and runtime gates without claiming completion."""
    target = TARGETS[bot_name]
    source_bundle = _validate_source_bundle(target)
    archive = audit_archive(target)
    work_root = ROOT / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rank1-preflight-", dir=work_root) as temporary:
        smoke = run_official_engine_smoke(Path(temporary), target)
    return {
        "schema": "rocket-arena-rank1-300-preflight/v1",
        "passed": True,
        "completion_claim": False,
        "source_bundle": source_bundle,
        "archive": archive,
        "official_engine_smoke": smoke,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run static archive/runtime gates without checking or sealing rank",
    )
    parser.add_argument(
        "--bot",
        choices=tuple(TARGETS),
        default=BOT,
        help="frozen identity to exercise in preflight mode",
    )
    args = parser.parse_args()
    if not args.preflight and args.bot != BOT:
        parser.error("--bot only selects a static preflight; finalization follows monitor admission")
    try:
        report = preflight(args.bot) if args.preflight else finalize()
    except NotReadyError as exc:
        print(
            json.dumps(
                {
                    "schema": AUDIT_SCHEMA,
                    "qualified": False,
                    "ready": False,
                    "reason": str(exc),
                },
                indent=2,
            )
        )
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
