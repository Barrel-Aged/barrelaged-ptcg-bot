#!/usr/bin/env python3
"""Build immutable zero-Wobbuffet Lightning-package mechanism bundles."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "target80_candidates" / "hancang_elitebc_iter3_control_v1"
GALV_CONTROL = ROOT / "work" / "target80_candidates" / "joltik_galvantula_deck_control_v1"
GALV_ROUTE = ROOT / "work" / "target80_candidates" / "joltik_galvantula_route_v1"
ZAPDOS_CONTROL = ROOT / "work" / "target80_candidates" / "rocket_zapdos_deck_control_v1"
ZAPDOS_ROUTE = ROOT / "work" / "target80_candidates" / "rocket_zapdos_route_v1"
GALV_MAIN = ROOT / "scripts" / "target80_joltik_galvantula_main_v1.py"
ZAPDOS_MAIN = ROOT / "scripts" / "target80_rocket_zapdos_main_v1.py"

EXPECTED_MODEL_SHA256 = "b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f"
WOBBUFFET = 432


GALV_COUNTS = {
    1: 6,       # Basic Grass Energy
    4: 4,       # Basic Lightning Energy
    15: 4,      # Team Rocket's Energy
    160: 2,     # Joltik (Jolting Charge)
    161: 2,     # Galvantula ex
    400: 4,     # Team Rocket's Tarountula
    401: 4,     # Team Rocket's Spidops
    414: 1,     # Team Rocket's Articuno
    431: 2,     # Team Rocket's Mewtwo ex
    434: 1,     # Team Rocket's Mimikyu
    1086: 2,    # Buddy-Buddy Poffin
    1094: 2,    # Bug Catching Set
    1097: 2,    # Night Stretcher
    1121: 4,    # Ultra Ball
    1123: 2,    # Switch
    1134: 3,    # Team Rocket's Transceiver
    1152: 2,    # Poke Pad
    1165: 1,    # Sparkling Crystal
    1175: 1,    # Brave Bangle
    1216: 2,    # Team Rocket's Ariana
    1219: 1,    # Team Rocket's Petrel
    1220: 2,    # Team Rocket's Proton
    1225: 2,    # Hilda
    1227: 3,    # Lillie's Determination
    1257: 1,    # Team Rocket's Factory
}


ZAPDOS_COUNTS = {
    1: 6,       # Basic Grass Energy
    4: 3,       # Basic Lightning Energy
    15: 4,      # Team Rocket's Energy
    400: 4,     # Team Rocket's Tarountula
    401: 4,     # Team Rocket's Spidops
    414: 1,     # Team Rocket's Articuno
    425: 2,     # Team Rocket's Zapdos
    431: 2,     # Team Rocket's Mewtwo ex
    434: 1,     # Team Rocket's Mimikyu
    1086: 1,    # Buddy-Buddy Poffin
    1094: 2,    # Bug Catching Set
    1097: 2,    # Night Stretcher
    1121: 1,    # Ultra Ball
    1123: 2,    # Switch
    1134: 4,    # Team Rocket's Transceiver
    1152: 3,    # Poke Pad
    1158: 1,    # Maximum Belt
    1175: 1,    # Brave Bangle
    1198: 3,    # Crispin
    1216: 4,    # Team Rocket's Ariana
    1218: 1,    # Team Rocket's Giovanni
    1219: 2,    # Team Rocket's Petrel
    1220: 2,    # Team Rocket's Proton
    1227: 2,    # Lillie's Determination
    1257: 2,    # Team Rocket's Factory
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deck_from_counts(counts: dict[int, int]) -> list[int]:
    if sum(counts.values()) != 60:
        raise RuntimeError(f"deck totals {sum(counts.values())}, expected 60")
    if counts.get(WOBBUFFET, 0):
        raise RuntimeError("candidate violates zero-Wobbuffet constraint")
    return [card_id for card_id, count in sorted(counts.items()) for _ in range(count)]


def write_provenance(path: Path, *, package: str, role: str) -> None:
    deck = [int(value) for value in (path / "deck.csv").read_text(encoding="utf-8").split()]
    payload = {
        "schema": "target80-lightning-package-screen-v1/v1",
        "package": package,
        "role": role,
        "source_bundle": str(SOURCE.resolve()),
        "source_model_sha256": sha256(SOURCE / "model.pt"),
        "deck_count": len(deck),
        "deck_counts": {str(key): value for key, value in sorted(Counter(deck).items())},
        "deck_sha256": sha256(path / "deck.csv"),
        "main_sha256": sha256(path / "main.py"),
        "model_sha256": sha256(path / "model.pt"),
        "wobbuffet_count": deck.count(WOBBUFFET),
        "claim_boundary": "official-engine mechanism research only; not deployment evidence",
    }
    (path / "LIGHTNING_PACKAGE_V1_PROVENANCE.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_pair(
    *,
    counts: dict[int, int],
    control: Path,
    route: Path,
    route_main: Path,
    package: str,
) -> None:
    deck = deck_from_counts(counts)
    shutil.copytree(SOURCE, control, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    (control / "deck.csv").write_text(
        "\n".join(str(card_id) for card_id in deck) + "\n", encoding="utf-8"
    )
    write_provenance(control, package=package, role="deck-identical neural control")

    shutil.copytree(control, route, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    shutil.copy2(route_main, route / "main.py")
    write_provenance(route, package=package, role="deterministic mechanism route")


def main() -> int:
    targets = (GALV_CONTROL, GALV_ROUTE, ZAPDOS_CONTROL, ZAPDOS_ROUTE)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing existing immutable targets: {existing}")
    if sha256(SOURCE / "model.pt") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("frozen source model identity changed")
    if not GALV_MAIN.is_file() or not ZAPDOS_MAIN.is_file():
        raise FileNotFoundError("route main source is missing")

    build_pair(
        counts=GALV_COUNTS,
        control=GALV_CONTROL,
        route=GALV_ROUTE,
        route_main=GALV_MAIN,
        package="joltik-galvantula",
    )
    build_pair(
        counts=ZAPDOS_COUNTS,
        control=ZAPDOS_CONTROL,
        route=ZAPDOS_ROUTE,
        route_main=ZAPDOS_MAIN,
        package="rocket-zapdos",
    )

    report = {
        path.name: json.loads(
            (path / "LIGHTNING_PACKAGE_V1_PROVENANCE.json").read_text(encoding="utf-8")
        )
        for path in targets
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
