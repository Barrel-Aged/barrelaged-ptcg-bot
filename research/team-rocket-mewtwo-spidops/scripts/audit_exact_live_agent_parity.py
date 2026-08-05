#!/usr/bin/env python3
"""Compare archived actions with the exact submitted NNAgent execution path."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--recorded-bot", required=True)
    parser.add_argument("--family", default="marnie")
    parser.add_argument("--started", choices=("any", "first", "second"), default="second")
    parser.add_argument("--slice", choices=("first", "middle", "last", "all"), default="all")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--trace-logits", action="store_true")
    parser.add_argument("--include-draws", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def opponent_family(name: str) -> str:
    lowered = name.lower()
    for token in (
        "marnie", "alakazam", "soldier", "starmie", "rocket", "dragapult",
        "lucario", "archaludon", "trevenant", "garchomp", "crustle",
    ):
        if token in lowered:
            return token
    return "other"


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def eligible(
    path: Path,
    recorded_bot: str,
    family: str,
    started: str,
) -> tuple[int, list[dict[str, Any]], bool] | None:
    records = read_records(path)
    players = list((records[0].get("meta") or {}).get("players") or [])
    if recorded_bot not in players:
        return None
    seat = players.index(recorded_bot)
    if opponent_family(players[1 - seat]) != family:
        return None
    decisions = [
        record for record in records[1:-1]
        if "obs" in record and int(record.get("p", -1)) == seat
    ]
    first_player = next(
        (
            (record.get("obs") or {}).get("current", {}).get("firstPlayer")
            for record in decisions
            if (record.get("obs") or {}).get("current", {}).get("firstPlayer") in (0, 1)
        ),
        -1,
    )
    started_first = first_player == seat
    if started == "first" and not started_first:
        return None
    if started == "second" and started_first:
        return None
    terminal = records[-1].get("result") or {}
    return seat, decisions, terminal.get("winner") is None


def select_slice(paths: list[Path], name: str, limit: int) -> list[Path]:
    if name == "all" or limit <= 0 or limit >= len(paths):
        return paths
    if name == "first":
        return paths[:limit]
    if name == "last":
        return paths[-limit:]
    start = max(0, (len(paths) - limit) // 2)
    return paths[start:start + limit]


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    torch.set_num_threads(1)
    candidate = args.candidate.resolve()
    sys.path.insert(0, str(candidate))
    from soldier import features  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import (  # type: ignore
        NNAgent,
        attack_safe_logits,
        feat_to_tensors,
        load_policy,
    )

    deck = [int(value) for value in (candidate / "deck.csv").read_text().split()]
    library_path = candidate / "deck_library.json"
    library = json.loads(library_path.read_text()) if library_path.exists() else None
    model = load_policy(candidate / "model.pt").eval()
    eligible_games: list[tuple[Path, int, list[dict[str, Any]]]] = []
    rejected_draws = 0
    for path in sorted(args.replays.resolve().glob("*.jsonl")):
        selected = eligible(
            path,
            args.recorded_bot,
            args.family,
            args.started,
        )
        if selected is not None:
            if selected[2] and not args.include_draws:
                rejected_draws += 1
                continue
            eligible_games.append((path, selected[0], selected[1]))
    selected_paths = set(
        select_slice([item[0] for item in eligible_games], args.slice, args.limit)
    )
    selected_games = [item for item in eligible_games if item[0] in selected_paths]

    totals: Counter[str] = Counter()
    mismatch_details = []
    per_game = []
    for path, _seat, decisions in selected_games:
        agent = NNAgent(
            model,
            greedy=True,
            deck_ids=deck,
            forbid_end_when_attack=True,
            opponent_deck_priors=library,
        )
        trace_belief = (
            Belief(deck, opponent_deck_priors=library) if args.trace_logits else None
        )
        game_matched = game_mismatched = game_equivalent = 0
        for decision_index, record in enumerate(decisions):
            observation = record["obs"]
            expected = list(record.get("action") or [])
            selection = observation.get("select") or {}
            options = selection.get("option") or []
            upper = min(int(selection.get("maxCount") or 0), len(options))
            traced = []
            trace_logits = []
            if trace_belief is not None:
                base = features.extract(
                    observation,
                    belief=trace_belief,
                    slot_link=getattr(model, "slot_link", False),
                )
                for _ in range(upper):
                    conditioned = (
                        base
                        if upper == 1
                        else features.mark_selection(base, traced, upper)
                    )
                    logits, _ = model(feat_to_tensors(conditioned))
                    if upper == 1:
                        logits = attack_safe_logits(logits, options)
                    choice = int(logits.argmax())
                    traced.append(choice)
                    trace_logits.append(logits.detach().cpu())
            predicted = list(agent(observation))
            if trace_belief is not None and traced != predicted:
                raise RuntimeError(
                    f"trace diverged from exact NNAgent in {path.name}: "
                    f"trace={traced} agent={predicted}"
                )
            totals["decisions"] += 1
            forced_set_equivalent = (
                upper == len(options)
                and len(expected) == upper
                and sorted(predicted) == sorted(expected)
            )
            if predicted == expected or forced_set_equivalent:
                totals["matched"] += 1
                game_matched += 1
                if forced_set_equivalent and predicted != expected:
                    totals["equivalent_set_order"] += 1
                    game_equivalent += 1
                continue
            totals["mismatched"] += 1
            game_mismatched += 1
            if len(mismatch_details) < 30:
                gap = None
                top_margin = None
                if len(expected) == 1 and trace_logits:
                    values = trace_logits[0]
                    gap = float(values[predicted[0]] - values[expected[0]])
                    finite = values[torch.isfinite(values)]
                    if finite.numel() >= 2:
                        top_two = torch.topk(finite, 2).values
                        top_margin = float(top_two[0] - top_two[1])
                mismatch_details.append(
                    {
                        "file": path.name,
                        "decision_index": decision_index,
                        "expected": expected,
                        "predicted": predicted,
                        "min_count": selection.get("minCount"),
                        "max_count": selection.get("maxCount"),
                        "option_types": [
                            option.get("type") for option in selection.get("option") or []
                        ],
                        "expected_options": [
                            (selection.get("option") or [])[index]
                            for index in expected
                            if 0 <= index < len(selection.get("option") or [])
                        ],
                        "predicted_options": [
                            (selection.get("option") or [])[index]
                            for index in predicted
                            if 0 <= index < len(selection.get("option") or [])
                        ],
                        "predicted_minus_expected_logit": gap,
                        "top_two_logit_margin": top_margin,
                    }
                )
        totals["games"] += 1
        totals["games_with_mismatch"] += int(game_mismatched > 0)
        if not args.summary_only or game_mismatched:
            per_game.append(
                {
                    "file": path.name,
                    "decisions": game_matched + game_mismatched,
                    "matched": game_matched,
                    "mismatched": game_mismatched,
                    "equivalent_set_order": game_equivalent,
                }
            )
    result = {
        "schema": "exact-live-agent-replay-parity/v2",
        "candidate": str(candidate),
        "reference_model_sha256": hashlib.sha256(
            (candidate / "model.pt").read_bytes()
        ).hexdigest(),
        "deck_sha256": hashlib.sha256(
            (candidate / "deck.csv").read_bytes()
        ).hexdigest(),
        "replays": str(args.replays.resolve()),
        "filters": {
            "recorded_bot": args.recorded_bot,
            "family": args.family,
            "started": args.started,
            "include_draws": args.include_draws,
        },
        "rejected_draws": rejected_draws,
        "eligible_games": len(eligible_games),
        "slice": args.slice,
        "requested_limit": args.limit,
        "totals": dict(totals),
        "first_selected": selected_games[0][0].name if selected_games else None,
        "last_selected": selected_games[-1][0].name if selected_games else None,
        "per_game": per_game,
        "mismatch_details": mismatch_details,
    }
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite parity report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if totals["mismatched"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
