#!/usr/bin/env python3
"""Build immutable V1R mechanism routes from the locked V1 deck controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SPECS = (
    (
        "joltik-galvantula",
        ROOT / "work" / "target80_candidates" / "joltik_galvantula_deck_control_v1",
        ROOT / "work" / "target80_candidates" / "joltik_galvantula_route_v1r",
        ROOT / "scripts" / "target80_joltik_galvantula_main_v1.py",
        "b7b649ba120a17680ed62c5910539363ee66a0fe86ff68e779492bef632a63bc",
    ),
    (
        "rocket-zapdos",
        ROOT / "work" / "target80_candidates" / "rocket_zapdos_deck_control_v1",
        ROOT / "work" / "target80_candidates" / "rocket_zapdos_route_v1r",
        ROOT / "scripts" / "target80_rocket_zapdos_main_v1.py",
        "b4d4e87fd490c404dc1457048f13c04fb1ba9b43fce0231a7092bf1b25f342bd",
    ),
)
EXPECTED_MODEL_SHA256 = "b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    for _, control, route, main_source, expected_deck in SPECS:
        if route.exists():
            raise FileExistsError(f"refusing existing immutable target: {route}")
        if sha256(control / "deck.csv") != expected_deck:
            raise RuntimeError(f"deck control identity changed: {control}")
        if sha256(control / "model.pt") != EXPECTED_MODEL_SHA256:
            raise RuntimeError(f"model identity changed: {control}")
        if not main_source.is_file():
            raise FileNotFoundError(main_source)

    result = {}
    for package, control, route, main_source, expected_deck in SPECS:
        shutil.copytree(control, route, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        shutil.copy2(main_source, route / "main.py")
        provenance = json.loads(
            (route / "LIGHTNING_PACKAGE_V1_PROVENANCE.json").read_text(encoding="utf-8")
        )
        provenance.update(
            {
                "schema": "target80-lightning-package-screen-v1r/v1",
                "package": package,
                "role": "corrected deterministic mechanism route",
                "deck_sha256": expected_deck,
                "main_sha256": sha256(route / "main.py"),
                "model_sha256": sha256(route / "model.pt"),
                "repair_scope": (
                    "recognize official type-9 evolution actions for Galvantula; "
                    "preserve Switch while Active Zapdos is the routed attacker"
                ),
            }
        )
        (route / "LIGHTNING_PACKAGE_V1_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result[route.name] = provenance
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
