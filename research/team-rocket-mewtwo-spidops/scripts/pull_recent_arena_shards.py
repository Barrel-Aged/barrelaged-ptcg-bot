#!/usr/bin/env python3
"""Pull a bounded set of recent arena shards without requesting all history."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARENA_LAB = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab"
)
DEFAULT_BOT = "HelbrosRocketSpidopsScaleBCV2"


def recent_run_ids(status: dict[str, object], limit: int) -> list[str]:
    runs = status.get("runs")
    if not isinstance(runs, list):
        raise ValueError("arena status does not contain a run ledger")
    values = [
        str(item["run"])
        for item in runs
        if isinstance(item, dict) and item.get("run")
    ]
    return list(reversed(values[-limit:]))


def safe_extract_short_names(raw: bytes, destination: Path) -> list[dict[str, object]]:
    """Validate a shard archive and extract files under bounded local names."""
    extracted: list[dict[str, object]] = []
    total_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("arena returned an empty shard archive")
        if len({member.name for member in members}) != len(members):
            raise ValueError("arena shard contains duplicate member names")
        for index, member in enumerate(members):
            original = member.name.replace("\\", "/")
            path = PurePosixPath(original)
            if (
                path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
                or re.match(r"^[A-Za-z]:", original) is not None
                or not member.isfile()
            ):
                raise ValueError(f"unsafe arena shard member: {member.name!r}")
            total_bytes += int(member.size)
            if total_bytes > 256 * 1024 * 1024:
                raise ValueError("arena shard expands above the 256 MiB safety cap")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read arena shard member: {member.name!r}")
            data = source.read()
            if len(data) != member.size:
                raise ValueError(f"arena shard member size changed: {member.name!r}")
            suffix = ".jsonl" if original.lower().endswith(".jsonl") else ".bin"
            stored = f"game_{index:04d}{suffix}"
            (destination / stored).write_bytes(data)
            extracted.append(
                {
                    "original_name": original,
                    "stored_name": stored,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    (destination / "archive_members.json").write_text(
        json.dumps(extracted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", default=DEFAULT_BOT)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--successes", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=40)
    parser.add_argument("--arena-lab", type=Path, default=DEFAULT_ARENA_LAB)
    args = parser.parse_args()
    if args.successes < 1 or args.max_runs < args.successes:
        raise ValueError("require 1 <= successes <= max-runs")

    helper_dir = (args.arena_lab / "tools").resolve()
    sys.path.insert(0, str(helper_dir))
    import arena  # type: ignore  # Trusted authenticated workstation helper.

    config = arena._load_json(arena.DEFAULT_CONFIG)
    status = arena._request(config, "GET", "/status")
    if not isinstance(status, dict):
        raise TypeError("arena status is not a JSON object")

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    pulled: list[dict[str, object]] = []
    probed: list[str] = []
    encoded_bot = urllib.parse.quote(args.bot)
    for run_id in recent_run_ids(status, args.max_runs):
        probed.append(run_id)
        try:
            raw = arena._request(
                config,
                "GET",
                f"/data?bot={encoded_bot}&run={urllib.parse.quote(run_id)}",
                expect_json=False,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        if not isinstance(raw, bytes):
            raise TypeError(f"arena returned {type(raw).__name__}, expected bytes")
        run_dir = out_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing shard: {run_dir}")
        staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=out_root))
        extracted = safe_extract_short_names(raw, staging)
        (staging / "arena_shard.tar.gz").write_bytes(raw)
        archive_path = run_dir / "arena_shard.tar.gz"
        item = {
            "run": run_id,
            "archive": str(archive_path),
            "archive_bytes": len(raw),
            "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "extracted_files": len(extracted),
            "members": extracted,
        }
        (staging / "shard_manifest.json").write_text(
            json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(run_dir)
        pulled.append(item)
        print(json.dumps(item, sort_keys=True), flush=True)
        if len(pulled) >= args.successes:
            break

    result = {
        "schema": "rocket-recent-arena-shards/v1",
        "bot": args.bot,
        "server_time": status.get("server_time"),
        "requested_successes": args.successes,
        "max_runs": args.max_runs,
        "probed_runs": probed,
        "pulled": pulled,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if pulled else 2


if __name__ == "__main__":
    raise SystemExit(main())
