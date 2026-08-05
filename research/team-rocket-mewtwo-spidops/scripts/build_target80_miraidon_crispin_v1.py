#!/usr/bin/env python3
"""Build immutable research bundles for the Miraidon/Crispin mechanism probe."""
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
CONTROL = ROOT / "work" / "target80_candidates" / "hancang_elitebc_iter3_control_v1"
DECK_CONTROL = ROOT / "work" / "target80_candidates" / "miraidon_crispin_deck_control_v1"
ROUTED = ROOT / "work" / "target80_candidates" / "miraidon_crispin_route_v1"
ROUTED_MAIN = ROOT / "scripts" / "target80_miraidon_crispin_main_v1.py"


DECK_COUNTS = {
    1: 6,       # Basic Grass
    4: 3,       # Basic Lightning
    15: 4,      # Team Rocket's Energy
    400: 4,     # Team Rocket's Tarountula
    401: 4,     # Team Rocket's Spidops
    414: 1,     # Team Rocket's Articuno
    431: 2,     # Team Rocket's Mewtwo ex
    434: 1,     # Team Rocket's Mimikyu
    957: 2,     # Miraidon ex
    1086: 1,    # Buddy-Buddy Poffin
    1094: 2,    # Bug Catching Set
    1097: 2,    # Night Stretcher
    1121: 4,    # Ultra Ball
    1134: 4,    # Team Rocket's Transceiver
    1152: 3,    # Poke Pad
    1159: 1,    # Hero's Cape
    1175: 2,    # Brave Bangle
    1198: 3,    # Crispin
    1216: 2,    # Team Rocket's Ariana
    1218: 1,    # Team Rocket's Giovanni
    1219: 1,    # Team Rocket's Petrel
    1220: 2,    # Team Rocket's Proton
    1227: 3,    # Lillie's Determination
    1257: 2,    # Team Rocket's Factory
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(path: Path, role: str) -> None:
    deck = [
        int(value)
        for value in path.joinpath("deck.csv").read_text(encoding="utf-8").split()
    ]
    payload = {
        "schema": "target80-miraidon-crispin-v1/v1",
        "role": role,
        "source_bundle": str(SOURCE.resolve()),
        "source_model": str(MODEL_SOURCE.resolve()),
        "source_model_sha256": sha256(MODEL_SOURCE),
        "deck_count": len(deck),
        "deck_counts": {str(key): value for key, value in sorted(Counter(deck).items())},
        "deck_sha256": sha256(path / "deck.csv"),
        "main_sha256": sha256(path / "main.py"),
        "model_sha256": sha256(path / "model.pt"),
        "wobbuffet_count": deck.count(432),
        "miraidon_count": deck.count(957),
        "crispin_count": deck.count(1198),
        "lightning_energy_count": deck.count(4),
        "claim_boundary": "structural development mechanism only; not deployment evidence",
    }
    (path / "MIRAIDON_CRISPIN_V1_PROVENANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    for path in (CONTROL, DECK_CONTROL, ROUTED):
        if path.exists():
            raise FileExistsError(f"refusing existing target: {path}")
    if sum(DECK_COUNTS.values()) != 60:
        raise RuntimeError("candidate deck does not total 60 cards")
    if DECK_COUNTS.get(432, 0) != 0:
        raise RuntimeError("candidate violates zero-Wobbuffet constraint")
    if sha256(MODEL_SOURCE) != "b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f":
        raise RuntimeError("V1R iteration-3 model identity changed")

    shutil.copytree(SOURCE, CONTROL, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(MODEL_SOURCE, CONTROL / "model.pt")
    write_provenance(CONTROL, "exact-deck learned Hancang control")

    shutil.copytree(CONTROL, DECK_CONTROL, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    deck = [card for card, count in sorted(DECK_COUNTS.items()) for _ in range(count)]
    (DECK_CONTROL / "deck.csv").write_text(
        "\n".join(str(card) for card in deck) + "\n", encoding="utf-8"
    )
    write_provenance(DECK_CONTROL, "Miraidon/Crispin deck with neural control runtime")

    shutil.copytree(DECK_CONTROL, ROUTED, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(ROUTED_MAIN, ROUTED / "main.py")
    write_provenance(ROUTED, "Miraidon/Crispin deck with public Starmie route")

    result = {
        path.name: json.loads(
            path.joinpath("MIRAIDON_CRISPIN_V1_PROVENANCE.json").read_text(
                encoding="utf-8"
            )
        )
        for path in (CONTROL, DECK_CONTROL, ROUTED)
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
