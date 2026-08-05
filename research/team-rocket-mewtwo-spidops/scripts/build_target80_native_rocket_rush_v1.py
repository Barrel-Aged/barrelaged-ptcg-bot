#!/usr/bin/env python3
"""Build the deck-matched native Rocket Rush mechanism candidate."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "target80_candidates" / "hancang_wobbfree_scale_control_v1"
MODEL_SOURCE = (
    ROOT
    / "work"
    / "target80_exact_deck_elite_bc_v1r_seed806051"
    / "iteration_03.pt"
)
MAIN_SOURCE = ROOT / "scripts" / "target80_native_rocket_rush_main_v1.py"
TARGET = (
    ROOT
    / "work"
    / "target80_candidates"
    / "hancang_native_rocket_rush_route_v1r"
)

EXPECTED_DECK_SHA256 = "0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8"
EXPECTED_MODEL_SHA256 = "b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if TARGET.exists():
        raise FileExistsError(f"refusing existing target: {TARGET}")
    if sha256(SOURCE / "deck.csv") != EXPECTED_DECK_SHA256:
        raise RuntimeError("exact Hancang deck identity changed")
    if sha256(MODEL_SOURCE) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("selected exact-deck model identity changed")

    shutil.copytree(
        SOURCE,
        TARGET,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    shutil.copy2(MODEL_SOURCE, TARGET / "model.pt")
    shutil.copy2(MAIN_SOURCE, TARGET / "main.py")
    deck = [int(value) for value in (TARGET / "deck.csv").read_text().split()]
    if len(deck) != 60 or deck.count(432) != 0:
        raise RuntimeError("candidate is not an exact 60-card zero-Wobbuffet deck")
    payload = {
        "schema": "target80-native-rocket-rush-v1r/v1",
        "role": "public-state mechanism probe and trajectory teacher only",
        "source_bundle": str(SOURCE.resolve()),
        "source_model": str(MODEL_SOURCE.resolve()),
        "deck_count": len(deck),
        "deck_counts": {
            str(card_id): count
            for card_id, count in sorted(Counter(deck).items())
        },
        "deck_sha256": sha256(TARGET / "deck.csv"),
        "model_sha256": sha256(TARGET / "model.pt"),
        "main_sha256": sha256(TARGET / "main.py"),
        "wobbuffet_count": deck.count(432),
        "claim_boundary": (
            "official-engine mechanism development only; heuristic wrapper is "
            "not eligible for arena submission"
        ),
    }
    (TARGET / "NATIVE_ROCKET_RUSH_V1_PROVENANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
