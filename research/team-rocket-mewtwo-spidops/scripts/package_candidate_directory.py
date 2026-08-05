#!/usr/bin/env python3
"""Create a deterministic arena archive from a vetted candidate directory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def included_files(source: Path) -> list[Path]:
    files = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(source).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.is_dir():
        raise ValueError(f"candidate directory does not exist: {source}")
    files = included_files(source)
    names = {path.relative_to(source).as_posix() for path in files}
    required = {"main.py", "deck.csv"}
    missing = sorted(required - names)
    if not any("/" not in name and name.endswith(".pt") for name in names):
        missing.append("<top-level .pt checkpoint>")
    if missing:
        raise ValueError(f"candidate is missing required files: {missing}")

    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                relative = path.relative_to(source).as_posix()
                raw = path.read_bytes()
                info = tarfile.TarInfo(relative)
                info.size = len(raw)
                info.mode = 0o755 if relative == "main.py" else 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(raw))

    raw_archive = output.getvalue()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw_archive)
    print(
        json.dumps(
            {
                "archive": str(args.out.resolve()),
                "bytes": len(raw_archive),
                "files": len(files),
                "sha256": hashlib.sha256(raw_archive).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
