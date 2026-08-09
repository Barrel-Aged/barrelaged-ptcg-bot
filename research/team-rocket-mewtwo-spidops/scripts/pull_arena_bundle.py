#!/usr/bin/env python3
"""Download an auditable bot bundle from the internal arena.

The arena token remains owned by the existing workstation helper/config.  This
script never prints or copies it; it only uses the helper's authenticated GET
request and validates the returned tar archive before writing it under the
requested project path.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import urllib.parse


DEFAULT_ARENA_LAB = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs\ptcg_agent_lab"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", required=True, help="Exact internal-arena bot name")
    parser.add_argument("--out", required=True, type=Path, help="Destination .tar.gz")
    parser.add_argument(
        "--allow-single-nested-root",
        action="store_true",
        help=(
            "Accept a safe archive whose files all live below one top-level "
            "directory containing main.py"
        ),
    )
    parser.add_argument(
        "--arena-lab",
        type=Path,
        default=DEFAULT_ARENA_LAB,
        help="Existing lab containing tools/arena.py and configs/arena.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    helper_dir = (args.arena_lab / "tools").resolve()
    sys.path.insert(0, str(helper_dir))
    import arena  # type: ignore  # Imported from the trusted workstation helper.

    config = arena._load_json(arena.DEFAULT_CONFIG)
    raw = arena._request(
        config,
        "GET",
        "/bundles?bot=" + urllib.parse.quote(args.bot),
        expect_json=False,
    )
    if not isinstance(raw, bytes):
        raise TypeError(f"arena returned {type(raw).__name__}, expected bytes")

    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("arena returned an empty archive")
        normalized_names = []
        for member in members:
            normalized = PurePosixPath(member.name.replace("\\", "/"))
            if normalized.is_absolute() or ".." in normalized.parts:
                raise ValueError(f"unsafe archive member: {member.name!r}")
            normalized_names.append(normalized)
        package_root = "."
        if PurePosixPath("main.py") not in normalized_names:
            main_candidates = [
                name for name in normalized_names if name.name == "main.py"
            ]
            if not args.allow_single_nested_root or len(main_candidates) != 1:
                raise ValueError("bundle does not contain root main.py")
            candidate = main_candidates[0]
            if len(candidate.parts) != 2:
                raise ValueError("bundle main.py is not below one top-level directory")
            package_root = candidate.parts[0]
            if any(not name.parts or name.parts[0] != package_root for name in normalized_names):
                raise ValueError("bundle contains members outside its package root")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    report = {
        "bot": args.bot,
        "bytes": len(raw),
        "members": len(members),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "package_root": package_root,
        "out": str(args.out.resolve()),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
