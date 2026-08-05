"""Deterministic checks for the final Articuno bench-reservation guard."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = (
    ROOT
    / "work"
    / "target80_candidates"
    / "hancang_wobbfree_elite_articuno_v7"
)


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "hancang_wobbfree_elite_articuno_v7_main",
        CANDIDATE / "main.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load final candidate main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _card(card_id: int) -> dict:
    return {"id": card_id, "hp": 100}


def _observation(
    *,
    hand_ids: list[int],
    opponent_in_play: list[int] | None = None,
    opponent_discard: list[int] | None = None,
    own_active: int = 431,
    own_bench: list[int] | None = None,
) -> dict:
    bench_ids = own_bench if own_bench is not None else [400, 400, 400, 434]
    options = [
        {"type": 7, "index": index}
        for index in range(len(hand_ids))
    ]
    options.append({"type": 14})
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": options,
        },
        "current": {
            "turn": 3,
            "yourIndex": 0,
            "players": [
                {
                    "active": [_card(own_active)],
                    "bench": [_card(card_id) for card_id in bench_ids],
                    "hand": [_card(card_id) for card_id in hand_ids],
                    "discard": [],
                },
                {
                    "active": [
                        _card(card_id)
                        for card_id in (opponent_in_play or [])[:1]
                    ],
                    "bench": [
                        _card(card_id)
                        for card_id in (opponent_in_play or [])[1:]
                    ],
                    "hand": [],
                    "discard": [
                        _card(card_id) for card_id in (opponent_discard or [])
                    ],
                },
            ],
        },
    }


def main() -> None:
    candidate = _load_main()

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(hand_ids=[414, 434], opponent_in_play=[121])
    )
    assert forced == [0]
    assert forbidden == frozenset()

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(
            hand_ids=[431, 434, 400],
            opponent_in_play=[121],
            own_bench=[400, 400, 400, 434],
        )
    )
    assert forced is None
    assert forbidden == frozenset({0, 1, 2})

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(hand_ids=[414, 434], opponent_in_play=[1031])
    )
    assert forced is None
    assert forbidden == frozenset()

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(
            hand_ids=[431, 434],
            opponent_in_play=[121],
            own_bench=[414, 400, 400, 434],
        )
    )
    assert forced is None
    assert forbidden == frozenset()

    forced, forbidden = candidate._articuno_bench_plan(
        _observation(hand_ids=[414], opponent_discard=[743])
    )
    assert forced == [0]
    assert forbidden == frozenset()

    assert candidate._fallback_allowed_action(
        {"option": [{"type": 7}, {"type": 13}, {"type": 14}]},
        [0, 1, 2],
    ) == [1]
    print("Articuno bench-reservation checks passed (6/6).")


if __name__ == "__main__":
    main()
