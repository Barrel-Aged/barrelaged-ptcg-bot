#!/usr/bin/env python3
"""Require exact safe-path action parity on real public observation streams."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_public_leader_v110_bc import matches_family, matches_seat_role  # noqa: E402
from train_rank1_second_seat_ppo import OpponentWorker, legal, sha256  # noqa: E402


DEFAULT_EVALUATOR = Path(
    r"C:\Users\jhos4\Documents\Codex\2026-06-20\you\outputs"
    r"\ptcg_agent_lab\tools\evaluate.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--max-games", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--family",
        choices=("all", "marnie", "starmie", "other"),
        default="marnie",
        help="Select replay streams by publicly visible opponent family.",
    )
    parser.add_argument(
        "--seat",
        choices=("all", "first", "second"),
        default="second",
        help="Select the focal Rocket player's starting seat.",
    )
    return parser.parse_args()


def matches_requested_family(path: Path, family: str) -> bool:
    if family == "other":
        return not matches_family(path, "marnie") and not matches_family(
            path, "starmie"
        )
    return matches_family(path, family)


def focal_observations(path: Path) -> list[dict[str, Any]]:
    """Recover the winning focal actor's true chronological decision inputs."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    rewards = payload.get("rewards") or []
    if len(rewards) != 2 or rewards[0] == rewards[1]:
        raise ValueError(f"{path.name}: expected a decisive focal-win replay")
    actor = int(rewards[1] > rewards[0])
    previous: list[dict[str, Any] | None] = [None, None]
    observations: list[dict[str, Any]] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for seat, agent in enumerate(step[:2]):
            if not isinstance(agent, dict):
                continue
            decision_input = previous[seat]
            action = agent.get("action")
            if (
                seat == actor
                and isinstance(decision_input, dict)
                and isinstance(decision_input.get("select"), dict)
                and isinstance(action, list)
            ):
                observations.append(decision_input)
            observation = agent.get("observation")
            if isinstance(observation, dict):
                previous[seat] = observation
    return observations


def option_context(observation: dict[str, Any]) -> dict[str, Any]:
    selection = observation.get("select") or {}
    current = observation.get("current") or {}
    return {
        "selection_type": selection.get("type"),
        "effect_id": (selection.get("effect") or {}).get("id"),
        "option_count": len(selection.get("option") or []),
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "first_player": current.get("firstPlayer"),
        "your_index": current.get("yourIndex"),
    }


def main() -> int:
    args = parse_args()
    control_path = args.control.resolve()
    candidate_path = args.candidate.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = [
        path
        for path in sorted(args.replays.resolve().glob("episode_*.json"))
        if matches_requested_family(path, args.family)
        and matches_seat_role(path, args.seat)
    ][: args.max_games]
    if len(paths) != args.max_games:
        raise ValueError(f"requested {args.max_games} games, found {len(paths)}")

    control = OpponentWorker(control_path, args.evaluator.resolve(), args.timeout)
    candidate = OpponentWorker(candidate_path, args.evaluator.resolve(), args.timeout)
    comparisons = 0
    illegal_control = 0
    illegal_candidate = 0
    mismatches: list[dict[str, Any]] = []
    per_game: list[dict[str, Any]] = []
    try:
        for path in paths:
            control.reset()
            candidate.reset()
            observations = focal_observations(path)
            game_mismatches = 0
            for decision_index, observation in enumerate(observations, start=1):
                control_action = control.act(observation)
                candidate_action = candidate.act(observation)
                selection = observation["select"]
                control_legal = legal(control_action, selection)
                candidate_legal = legal(candidate_action, selection)
                illegal_control += int(not control_legal)
                illegal_candidate += int(not candidate_legal)
                comparisons += 1
                if control_action != candidate_action:
                    game_mismatches += 1
                    if len(mismatches) < 50:
                        mismatches.append(
                            {
                                "game": path.name,
                                "decision": decision_index,
                                "control": control_action,
                                "candidate": candidate_action,
                                **option_context(observation),
                            }
                        )
            per_game.append(
                {
                    "game": path.name,
                    "decisions": len(observations),
                    "mismatches": game_mismatches,
                }
            )
    finally:
        control.close()
        candidate.close()

    passed = not mismatches and not illegal_control and not illegal_candidate
    payload = {
        "schema": "target80-safe-overlay-parity/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "control": str(control_path),
        "candidate": str(candidate_path),
        "control_main_sha256": sha256(control_path / "main.py"),
        "candidate_main_sha256": sha256(candidate_path / "main.py"),
        "control_model_sha256": sha256(control_path / "model.pt"),
        "candidate_base_model_sha256": sha256(candidate_path / "model.pt"),
        "candidate_overlay_model_sha256": sha256(
            candidate_path / "marnie_first_model.pt"
        ),
        "replays": str(args.replays.resolve()),
        "selection": {
            "opponent_family": args.family,
            "focal_seat": args.seat,
            "family_evidence": "publicly visible cards only",
        },
        "games": len(paths),
        "decision_comparisons": comparisons,
        "action_mismatches": len(mismatches),
        "illegal_control_actions": illegal_control,
        "illegal_candidate_actions": illegal_candidate,
        "exact_parity_pass": passed,
        "mismatch_examples": mismatches,
        "per_game": per_game,
        "public_observations_only": True,
        "deployment_authority": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "games": len(paths),
                "decisions": comparisons,
                "mismatches": len(mismatches),
                "illegal_control": illegal_control,
                "illegal_candidate": illegal_candidate,
                "passed": passed,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
