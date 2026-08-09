#!/usr/bin/env python3
"""Build the frozen N18 cross-deck transfer arms without running games."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "work/n18/blive"
LEADER = ROOT / "work/n18/bam"
AUDIT = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "N18_NEW_LEADER_EVIDENCE_AUDIT.json"
)
AUDIT_DOC = ROOT / "N18_NEW_LEADER_EVIDENCE_AUDIT.md"
OUTPUT = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "development_arms"
)

LIVE_TREE_SHA = "11988ad1e5ab0b81d993ad54bf32b1a903caf89b6a4040aacf57e0f1d6a8480a"
LEADER_TREE_SHA = "1a78564bb3304fc4c968e14cb01245eba2c232fe32594c0cfdb563a01a76e828"
AUDIT_SHA = "f0441e0e602421c0ccbb57447e9b5857af442c93e7b6a02224cd4f09c24e9a0c"
AUDIT_DOC_SHA = "f6b4d74c007d3dd5cb438eecdd4a1560aef64b311d0fe5bcbb1b9c4a177ee084"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
LIVE_MODEL_SHA = "3c969193bfaa7fecf79b6fa220d0d20000379685585080f5916b01d724bf041b"
LEADER_MODEL_SHA = "6f93f728c3f7cf5993d2f635183d48a530de1143257944316b2782d067885a0c"
RESIDUAL_SHA = "5df9bd6e9f2d733092b0653765b4d1d3f082a488cfe68fe31f23c94f3aa43f4f"
LIVE_MAIN_SHA = "8c87db1a011acf3cc5a614159d1ac6b6bf56aa7286f0f8a7148bb8e2aa20a6d1"
LEADER_MAIN_SHA = "0d90801416496e194f1d4dd46ea41d34a7e1f5fcc35bac84cabd095227857e6b"

ORDER = (
    "control",
    "bridge_base",
    "live_residual",
    "leader_base",
    "leader_residual",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_digest(root: Path) -> str:
    result = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.relative_to(root).parts
            and path.suffix.lower() not in (".pyc", ".pyo")
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        result.update(len(relative).to_bytes(4, "big"))
        result.update(relative)
        result.update(path.read_bytes())
    return result.hexdigest()


def copy_source(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
    )


def configure_arm(name: str, destination: Path) -> dict[str, Any]:
    if name == "control":
        copy_source(LIVE, destination)
        runtime = "live"
        residual = False
        model_source = "live"
    else:
        copy_source(LEADER, destination)
        shutil.copy2(LIVE / "deck.csv", destination / "deck.csv")
        shutil.copy2(LIVE / "deck_library.json", destination / "deck_library.json")
        runtime = "leader"
        residual = name in ("live_residual", "leader_residual")
        model_source = "live" if name in ("bridge_base", "live_residual") else "leader"
        if model_source == "live":
            shutil.copy2(LIVE / "model.pt", destination / "model.pt")
        if not residual:
            (destination / "history_residual.pt").unlink()

    expected_model = LIVE_MODEL_SHA if model_source == "live" else LEADER_MODEL_SHA
    expected_main = LIVE_MAIN_SHA if runtime == "live" else LEADER_MAIN_SHA
    if digest(destination / "deck.csv") != DECK_SHA:
        raise ValueError(f"{name} does not contain the exact Team Rocket deck")
    if len((destination / "deck.csv").read_text(encoding="utf-8").splitlines()) != 60:
        raise ValueError(f"{name} deck is not 60 cards")
    if digest(destination / "model.pt") != expected_model:
        raise ValueError(f"{name} model source changed")
    if digest(destination / "main.py") != expected_main:
        raise ValueError(f"{name} runtime entry point changed")
    residual_path = destination / "history_residual.pt"
    if residual_path.exists() != residual:
        raise ValueError(f"{name} residual presence changed")
    if residual and digest(residual_path) != RESIDUAL_SHA:
        raise ValueError(f"{name} residual changed")

    provenance = {
        "schema": "rocket-n18-cross-deck-arm/v1",
        "arm": name,
        "source_live_tree_sha256": LIVE_TREE_SHA,
        "source_leader_tree_sha256": LEADER_TREE_SHA,
        "evidence_audit_sha256": AUDIT_SHA,
        "deck_sha256": DECK_SHA,
        "model_source": model_source,
        "model_sha256": expected_model,
        "runtime_source": runtime,
        "main_sha256": expected_main,
        "history_residual_present": residual,
        "history_residual_sha256": RESIDUAL_SHA if residual else None,
        "official_match_evidence": False,
        "development_authority": False,
        "deployment_authority": False,
    }
    (destination / "N18_ARM_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return provenance


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError("N18 arm builder accepts no arguments")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N18 output: {OUTPUT}")
    checks = {
        LIVE: LIVE_TREE_SHA,
        LEADER: LEADER_TREE_SHA,
    }
    for path, expected in checks.items():
        actual = tree_digest(path)
        if actual != expected:
            raise ValueError(f"N18 source tree changed: {path}: {actual}")
    for path, expected in ((AUDIT, AUDIT_SHA), (AUDIT_DOC, AUDIT_DOC_SHA)):
        if digest(path) != expected:
            raise ValueError(f"N18 audit input changed: {path}")

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    facts = audit.get("admissible_facts") or {}
    if (
        audit.get("schema") != "rocket-n18-new-leader-evidence-audit/v1"
        or not facts.get("all_recorded_leader_actions_are_legal_for_row_local_observations")
        or not facts.get("leader_model_architecture_matches_live")
        or facts.get("leader_bundle_can_be_used_as_a_direct_rocket_candidate_without_official_match_testing")
        is not False
    ):
        raise ValueError("N18 evidence audit does not authorize arm construction")

    OUTPUT.mkdir(parents=True)
    arms: list[dict[str, Any]] = []
    for name in ORDER:
        bundle = OUTPUT / name
        provenance = configure_arm(name, bundle)
        arms.append(
            {
                "name": name,
                "bundle": str(bundle.resolve()),
                "bundle_tree_sha256": tree_digest(bundle),
                "deck_sha256": DECK_SHA,
                "model_sha256": provenance["model_sha256"],
                "main_sha256": provenance["main_sha256"],
                "history_residual_sha256": provenance["history_residual_sha256"],
                "development_eligible": name != "control",
                "deployment_authority": False,
            }
        )

    manifest = {
        "schema": "rocket-n18-cross-deck-arms/v1",
        "evidence_audit_sha256": AUDIT_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "order": list(ORDER),
        "arms": arms,
        "official_match_panel_performed": False,
        "development_authority": False,
        "deployment_authority": False,
    }
    manifest_path = OUTPUT / "arms.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": digest(manifest_path),
                "arms": {
                    item["name"]: item["bundle_tree_sha256"] for item in arms
                },
                "deployment_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
