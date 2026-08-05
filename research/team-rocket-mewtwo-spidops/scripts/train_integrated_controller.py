"""Fit and export the Phase 6AN controller under the frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.integrated_training import train_integrated_controller  # noqa: E402


DEFAULT_ARCHIVED_REPLAYS = Path(
    r"C:\Users\jhos4\PokemonDocs\_local_archive_2026-07-26\evidence"
    r"\rocket_spidops_mewtwo_20260719\arena_v3_refresh"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived-replays", type=Path, default=DEFAULT_ARCHIVED_REPLAYS)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "integrated_controller" / "integrated_planner.json",
    )
    args = parser.parse_args()
    report = train_integrated_controller(
        project_root=PROJECT_ROOT,
        archived_replays=args.archived_replays,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
