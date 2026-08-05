#!/usr/bin/env python3
"""Freeze the sole N5 development winner and control for locked validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "development_panel" / "analysis.json"
)
DEFAULT_ARMS = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "development_arms" / "arms.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "team_rocket_recovery" / "rocket_blueprint_n5" / "validation_arms"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_path = args.analysis.resolve()
    arms_path = args.arms.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty N5 validation freeze: {output}")
    output.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("schema") != "rocket-blueprint-n5-development-analysis/v1":
        raise ValueError("unexpected N5 development analysis")
    advances = analysis.get("advances_exactly_one_or_none") or []
    if len(advances) != 1:
        raise RuntimeError(f"N5 validation requires exactly one development winner, got {advances}")
    selected = str(advances[0])
    result = next((row for row in analysis.get("results") or () if row.get("arm") == selected), None)
    if not result or not result.get("development_gate_pass"):
        raise ValueError("selected N5 arm did not pass the locked development gate")
    arms_manifest = json.loads(arms_path.read_text(encoding="utf-8"))
    if arms_manifest.get("schema") != "rocket-blueprint-n5-development-arms/v1":
        raise ValueError("unexpected N5 development arm manifest")
    if analysis.get("arms_manifest_sha256") != sha256(arms_path):
        raise ValueError("N5 development analysis does not bind the supplied arms manifest")
    indexed = {str(item["name"]): item for item in arms_manifest.get("arms") or ()}
    if "control" not in indexed or selected not in indexed:
        raise ValueError("N5 freeze sources are absent")

    frozen = []
    for target_name, source_name in (("control", "control"), ("candidate", selected)):
        item = indexed[source_name]
        source = Path(str(item["path"])).resolve()
        if sha256(source) != str(item["sha256"]):
            raise ValueError(f"N5 freeze source changed: {source}")
        target = output / f"{target_name}.pt"
        shutil.copy2(source, target)
        frozen.append(
            {
                "name": target_name,
                "path": str(target),
                "sha256": sha256(target),
                "source_arm": source_name,
                "source_path": str(source),
                "source_sha256": str(item["sha256"]),
            }
        )
    manifest = {
        "schema": "rocket-blueprint-n5-validation-arms/v1",
        "protocol": "rocket-greedy-compatible-ppo/v1",
        "deck_sha256": arms_manifest["deck_sha256"],
        "initial_model_sha256": arms_manifest["initial_model_sha256"],
        "selection": "sole highest-ranked arm passing all preregistered development gates",
        "selected_development_arm": selected,
        "development_analysis": str(analysis_path),
        "development_analysis_sha256": sha256(analysis_path),
        "development_arms_manifest": str(arms_path),
        "development_arms_manifest_sha256": sha256(arms_path),
        "arms": frozen,
        "validation_schedule": {
            "opponents": 2,
            "shards_per_opponent": 2,
            "games_per_arm_per_shard": 375,
            "total_games": 3000,
        },
    }
    manifest_path = output / "arms.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(manifest_path), "selected": selected, "arms": frozen}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
