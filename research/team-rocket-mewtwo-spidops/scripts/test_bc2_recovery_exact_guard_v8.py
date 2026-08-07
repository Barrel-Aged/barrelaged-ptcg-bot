"""Deterministic integrity and guard tests for the final V8 repair."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "work" / "target80_candidates" / "bc2_recovery_exact_guard_v8"
V5 = ROOT / "work" / "target80_candidates" / "earlypad_bangle_lethal_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_main():
    spec = importlib.util.spec_from_file_location("bc2_recovery_exact_v8_main", CANDIDATE / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load final V8 main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _card(card_id: int, *, energies: list[int] | None = None) -> dict:
    return {"id": card_id, "hp": 100, "energies": energies or [], "energyCards": []}


def _articuno_observation(*, hand_ids: list[int], own_bench: list[int]) -> dict:
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
                {"active": [_card(121)], "bench": [], "hand": [], "discard": []},
            ],
        },
    }


def _wobbuffet_observation() -> dict:
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 13, "attackId": 610},
                {"type": 12},
                {"type": 14},
            ],
        },
        "current": {
            "turn": 8,
            "yourIndex": 0,
            "players": [
                {
                    "active": [_card(432, energies=[5, 0, 0])],
                    "bench": [
                        _card(401, energies=[1, 0]),
                        _card(400),
                        _card(463),
                    ],
                    "hand": [],
                    "discard": [],
                },
                {"active": [_card(756)], "bench": [], "hand": [], "discard": []},
            ],
        },
    }


def main() -> None:
    deck = [int(row) for row in (CANDIDATE / "deck.csv").read_text(encoding="utf-8").splitlines()]
    counts = Counter(deck)
    assert len(deck) == 60
    assert counts[432] == 1
    assert counts[463] == 2
    assert _sha256(CANDIDATE / "deck.csv") == _sha256(V5 / "deck.csv")
    assert _sha256(CANDIDATE / "model.pt") == _sha256(V5 / "model.pt")

    candidate = _load_main()
    forced, forbidden = candidate._articuno_bench_plan(
        _articuno_observation(hand_ids=[414, 432], own_bench=[400, 400, 463, 463])
    )
    assert forced == [0]
    assert forbidden == frozenset()

    forced, forbidden = candidate._articuno_bench_plan(
        _articuno_observation(hand_ids=[432, 463, 431], own_bench=[400, 400, 463, 463])
    )
    assert forced is None
    assert forbidden == frozenset({0, 1, 2})

    observation = _wobbuffet_observation()
    assert candidate._wobbuffet_exit_guard_action(observation, [0]) == [1]
    observation["select"]["option"][0]["attackId"] = 609
    assert candidate._wobbuffet_exit_guard_action(observation, [0]) == [0]
    observation["select"]["option"][0]["attackId"] = 610
    observation["current"]["players"][0]["bench"][0]["energies"] = [1]
    assert candidate._wobbuffet_exit_guard_action(observation, [0]) == [0]

    print("Final BC2 recovery V8 checks passed (exact substrate + 5 guard cases).")


if __name__ == "__main__":
    main()
