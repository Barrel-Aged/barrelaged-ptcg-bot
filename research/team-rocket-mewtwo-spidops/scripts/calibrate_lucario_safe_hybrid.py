"""Calibrate V3's confidence gate on V1 validation games, then audit frozen test."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Row:
    game: str
    context: int
    confidence: float | None
    actual: tuple[int, ...]
    neural: tuple[int, ...]
    baseline: tuple[int, ...]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--focal-team", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aligned_decisions(path: Path, focal_team: str) -> Iterator[tuple[dict[str, Any], list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    teams = list((payload.get("info") or {}).get("TeamNames") or [])
    seats = [index for index, team in enumerate(teams) if team == focal_team]
    if len(seats) != 1:
        return
    focal_seat = seats[0]
    previous: list[dict[str, Any] | None] = [None, None]
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for seat, agent in enumerate(step[:2]):
            if not isinstance(agent, dict):
                continue
            observation = agent.get("observation")
            action = agent.get("action")
            decision_observation = previous[seat]
            if (
                seat == focal_seat
                and isinstance(decision_observation, dict)
                and decision_observation.get("select") is not None
                and isinstance(action, list)
            ):
                yield decision_observation, action
            if isinstance(observation, dict):
                previous[seat] = observation


def legal(action: list[int], selection: dict[str, Any]) -> bool:
    option_count = len(selection.get("option") or [])
    lower = int(selection.get("minCount") or 0)
    upper = min(int(selection.get("maxCount") or 0), option_count)
    return (
        len(action) == len(set(action))
        and lower <= len(action) <= upper
        and all(isinstance(index, int) and 0 <= index < option_count for index in action)
    )


def collect(
    paths: list[Path],
    focal_team: str,
    hybrid_class,
    net,
    deck: list[int],
    deck_library,
) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        hybrid = hybrid_class(net, deck, threshold=0.0, opponent_deck_priors=deck_library)
        for observation, action in aligned_decisions(path, focal_team):
            selection = observation["select"]
            proposed = hybrid(observation)
            trace = hybrid.last_trace
            option_count = len(selection.get("option") or [])
            upper = min(int(selection.get("maxCount") or 0), option_count)
            if (
                option_count <= 1
                or upper <= 0
                or not legal(action, selection)
                or len(action) != upper
            ):
                continue
            if not legal(proposed, selection):
                raise RuntimeError(f"Hybrid emitted an illegal action in {path.name}")
            rows.append(
                Row(
                    game=path.name,
                    context=int(selection.get("context") or 0),
                    confidence=(
                        float(trace["confidence"])
                        if trace.get("confidence") is not None
                        else None
                    ),
                    actual=tuple(action),
                    neural=tuple(trace["neural_action"]),
                    baseline=tuple(trace["baseline_action"]),
                )
            )
    return rows


def metrics(rows: list[Row], threshold: float) -> dict[str, Any]:
    chosen: list[tuple[int, ...]] = []
    routes = 0
    disagreements_routed = 0
    for row in rows:
        route = row.confidence is None or row.confidence < threshold
        action = row.baseline if route else row.neural
        chosen.append(action)
        routes += int(route)
        disagreements_routed += int(route and row.baseline != row.neural)
    count = len(rows)
    exact = sum(action == row.actual for action, row in zip(chosen, rows))
    first = sum(
        bool(action and row.actual and action[0] == row.actual[0])
        for action, row in zip(chosen, rows)
    )
    return {
        "rows": count,
        "exact": exact,
        "exact_accuracy": exact / max(count, 1),
        "first_action": first,
        "first_action_accuracy": first / max(count, 1),
        "baseline_routes": routes,
        "baseline_route_rate": routes / max(count, 1),
        "routed_disagreements": disagreements_routed,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    candidate = args.candidate.resolve()
    replays = args.replays.resolve()
    summary = json.loads(args.split_summary.resolve().read_text(encoding="utf-8"))
    split_games = summary["split_games"]
    by_name = {path.name: path for path in replays.glob("episode-*-replay.json")}
    validation_paths = [by_name[name] for name in split_games["validation"]]
    test_paths = [by_name[name] for name in split_games["test"]]

    sys.path.insert(0, str(candidate))
    hybrid_module = importlib.import_module("hybrid_policy")
    net_module = importlib.import_module("soldier.net")
    net = net_module.load_policy(candidate / "model.pt").eval()
    deck = [
        int(line)
        for line in (candidate / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:60]
    deck_library_path = candidate / "deck_library.json"
    deck_library = (
        json.loads(deck_library_path.read_text(encoding="utf-8"))
        if deck_library_path.exists()
        else None
    )

    validation = collect(
        validation_paths,
        args.focal_team,
        hybrid_module.ConfidenceSafeHybridAgent,
        net,
        deck,
        deck_library,
    )
    cut_points = sorted(
        {0.0, 1.0000001}
        | {
            math.nextafter(row.confidence, math.inf)
            for row in validation
            if row.confidence is not None
        }
    )
    scored = [(threshold, metrics(validation, threshold)) for threshold in cut_points]
    selected_threshold, selected_validation = max(
        scored,
        key=lambda item: (
            item[1]["exact_accuracy"],
            -item[1]["baseline_routes"],
            -item[0],
        ),
    )

    test = collect(
        test_paths,
        args.focal_team,
        hybrid_module.ConfidenceSafeHybridAgent,
        net,
        deck,
        deck_library,
    )
    neural_validation = metrics(validation, 0.0)
    neural_test = metrics(test, 0.0)
    selected_test = metrics(test, selected_threshold)
    admitted = (
        selected_validation["exact_accuracy"] > neural_validation["exact_accuracy"]
        and selected_test["exact_accuracy"] >= neural_test["exact_accuracy"]
        and selected_test["first_action_accuracy"] >= neural_test["first_action_accuracy"]
        and selected_validation["routed_disagreements"] > 0
    )
    result = {
        "schema_version": 1,
        "candidate": str(candidate),
        "model_sha256": sha256(candidate / "model.pt"),
        "focal_team": args.focal_team,
        "split_summary": str(args.split_summary.resolve()),
        "validation_games": [path.name for path in validation_paths],
        "test_games": [path.name for path in test_paths],
        "selected_threshold": selected_threshold,
        "neural_validation": neural_validation,
        "selected_validation": selected_validation,
        "neural_test": neural_test,
        "selected_test": selected_test,
        "offline_admitted": admitted,
        "selection_rule": "max validation exact; tie fewer baseline routes; tie lower threshold",
        "validation_rows": [asdict(row) for row in validation],
        "test_rows": [asdict(row) for row in test],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("_rows")}, indent=2))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
