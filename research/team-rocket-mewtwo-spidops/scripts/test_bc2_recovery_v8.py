"""Deterministic integrity and Articuno-reservation checks for V8."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "work" / "target80_candidates" / "bc2_recovery_murkrow_articuno_v8"
V5 = ROOT / "work" / "target80_candidates" / "earlypad_bangle_lethal_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_main():
    spec = importlib.util.spec_from_file_location("bc2_recovery_v8_main", CANDIDATE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load V8 main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _card(card_id: int) -> dict:
    return {"id": card_id, "hp": 100}


def _observation(*, hand_ids: list[int], own_bench: list[int], opponent_id: int = 121) -> dict:
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 7, "index": index} for index in range(len(hand_ids))]
            + [{"type": 14}],
        },
        "current": {
            "turn": 3,
            "yourIndex": 0,
            "players": [
                {
                    "active": [_card(431)],
                    "bench": [_card(card_id) for card_id in own_bench],
                    "hand": [_card(card_id) for card_id in hand_ids],
                    "discard": [],
                },
                {
                    "active": [_card(opponent_id)],
                    "bench": [],
                    "hand": [],
                    "discard": [],
                },
            ],
        },
    }


def main() -> None:
    deck = [int(row) for row in (CANDIDATE / "deck.csv").read_text(encoding="utf-8").splitlines()]
    counts = Counter(deck)
    assert len(deck) == 60
    assert counts[432] == 0
    assert counts[463] == 3
    assert _sha256(CANDIDATE / "model.pt") == _sha256(V5 / "model.pt")

    candidate = _load_main()
    forced, forbidden = candidate._articuno_bench_plan(
        _observation(hand_ids=[414, 463], own_bench=[400, 400, 463, 463])
    )
    assert forced == [0]
    assert forbidden == frozenset()

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(hand_ids=[463, 431, 400], own_bench=[400, 400, 463, 463])
    )
    assert forced is None
    assert forbidden == frozenset({0, 1})

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(hand_ids=[463], own_bench=[414, 400, 463, 463])
    )
    assert forced is None
    assert forbidden == frozenset()

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(hand_ids=[414], own_bench=[400, 400, 463, 463], opponent_id=1031)
    )
    assert forced is None
    assert forbidden == frozenset()

    print("BC2 recovery V8 checks passed (deck/model + 4 Articuno cases).")


if __name__ == "__main__":
    main()
