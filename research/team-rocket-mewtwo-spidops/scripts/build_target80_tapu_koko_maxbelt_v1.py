#!/usr/bin/env python3
"""Build the corrected Pikachu ex/Crispin/Sparkling Crystal V1R probe arm."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "target80_candidates" / "hancang_wobbfree_scale_control_v1"
MODEL_SOURCE = ROOT / "work" / "target80_exact_deck_elite_bc_v1r_seed806051" / "iteration_03.pt"
MAIN_SOURCE = ROOT / "scripts" / "target80_tapu_koko_maxbelt_main_v1.py"
CONTROL = ROOT / "work" / "target80_candidates" / "pikachu_crispin_crystal_deck_control_v1"
ROUTED = ROOT / "work" / "target80_candidates" / "pikachu_crispin_crystal_route_v1r"

EXPECTED_MODEL_SHA256 = "b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f"

DECK_COUNTS = {
    1: 4, 4: 4, 15: 4,
    210: 2, 400: 4, 401: 4, 414: 1, 431: 2, 434: 1,
    1086: 1, 1094: 2, 1097: 2, 1121: 4, 1123: 2, 1134: 4,
    1152: 4, 1165: 1, 1175: 2, 1198: 4, 1218: 1, 1219: 2,
    1220: 2, 1227: 2, 1257: 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(path: Path, role: str) -> None:
    deck = [int(value) for value in (path / "deck.csv").read_text().split()]
    payload = {
        "schema": "target80-pikachu-crispin-crystal-v1r/v1",
        "role": role,
        "source_bundle": str(SOURCE.resolve()),
        "source_model": str(MODEL_SOURCE.resolve()),
        "deck_count": len(deck),
        "deck_counts": {str(k): v for k, v in sorted(Counter(deck).items())},
        "deck_sha256": sha256(path / "deck.csv"),
        "main_sha256": sha256(path / "main.py"),
        "model_sha256": sha256(path / "model.pt"),
        "wobbuffet_count": deck.count(432),
        "pikachu_ex_count": deck.count(210),
        "sparkling_crystal_count": deck.count(1165),
        "claim_boundary": "structural mechanism development only; no deployment authority",
    }
    (path / "PIKACHU_CRYSTAL_V1_PROVENANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    if ROUTED.exists():
        raise FileExistsError(f"refusing existing target: {ROUTED}")
    if sum(DECK_COUNTS.values()) != 60 or DECK_COUNTS.get(432, 0):
        raise RuntimeError("candidate must be an exact 60-card zero-Wobbuffet deck")
    if sha256(MODEL_SOURCE) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("selected exact-deck model identity changed")

    deck = [card_id for card_id, count in sorted(DECK_COUNTS.items()) for _ in range(count)]
    if not CONTROL.exists():
        shutil.copytree(SOURCE, CONTROL, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        shutil.copy2(MODEL_SOURCE, CONTROL / "model.pt")
        (CONTROL / "deck.csv").write_text(
            "\n".join(str(card_id) for card_id in deck) + "\n", encoding="utf-8"
        )
        write_provenance(CONTROL, "Pikachu ex/Sparkling Crystal deck with neural control")
    control_deck = [int(value) for value in (CONTROL / "deck.csv").read_text().split()]
    if control_deck != deck or sha256(CONTROL / "model.pt") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("existing deck-control identity changed")

    shutil.copytree(CONTROL, ROUTED, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(MAIN_SOURCE, ROUTED / "main.py")
    write_provenance(
        ROUTED,
        "corrected Pikachu ex/Crispin/Petrel/Sparkling Crystal public Starmie route",
    )

    print(json.dumps({
        path.name: json.loads((path / "PIKACHU_CRYSTAL_V1_PROVENANCE.json").read_text())
        for path in (CONTROL, ROUTED)
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
