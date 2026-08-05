#!/usr/bin/env python3
"""Re-execute the exact ScaleBCV2 policy on its downloaded Kaggle observations.

This is retrospective and read-only.  It proves whether a replayed choice came
from the packaged neural policy and records logits for the narrow Wobbuffet
decision class.  It does not branch the official engine or claim a causal game
outcome for an unplayed alternative.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import torch

import audit_kaggle_scalebcv2 as audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def option_label(
    option: dict[str, Any],
    state: dict[str, Any],
    seat: int,
    names: dict[int, str],
) -> str:
    option_type = int(option.get("type") if option.get("type") is not None else -1)
    if option_type == audit.OPTION_ATTACK:
        attack_id = int(option.get("attackId") or 0)
        return audit.ATTACK_NAMES.get(attack_id, f"Attack {attack_id}")
    if option_type == audit.OPTION_RETREAT:
        return "Retreat"
    if option_type == audit.OPTION_END:
        return "End turn"
    if option_type in (audit.OPTION_PLAY, audit.OPTION_ATTACH, audit.OPTION_EVOLVE):
        card_id = audit.option_card_id(option, state, seat)
        action = {
            audit.OPTION_PLAY: "Play",
            audit.OPTION_ATTACH: "Attach",
            audit.OPTION_EVOLVE: "Evolve",
        }[option_type]
        target_id = audit.option_target_id(option, state, seat)
        suffix = f" -> {names.get(target_id, target_id)}" if target_id else ""
        return f"{action} {names.get(card_id, card_id)}{suffix}"
    return f"type={option_type} {json.dumps(option, sort_keys=True)}"


def focal_decisions(path: Path) -> tuple[int, bool, list[tuple[dict[str, Any], list[int]]]]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    frames = audit.iter_frames(replay)
    seat, mirror = audit.detect_focal_seat(audit.initial_decks(frames))
    decisions: list[tuple[dict[str, Any], list[int]]] = []
    for frame in frames:
        observation = frame.get("obs") or {}
        state = observation.get("current") or {}
        if state.get("yourIndex") != seat or not observation.get("select"):
            continue
        selected = frame.get("selected")
        if not isinstance(selected, list):
            selected = [selected] if isinstance(selected, int) else []
        decisions.append((observation, [int(index) for index in selected]))
    return seat, mirror, decisions


def main() -> int:
    args = parse_args()
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

    names = audit.load_card_names(args.cards)
    deck_path = candidate / "deck.csv"
    deck = [int(value) for value in deck_path.read_text(encoding="utf-8").split()][:60]
    deck_hash = sha256(audit.canonical_deck_bytes(deck)).hexdigest()
    if deck_hash != audit.FOCAL_DECK_SHA256:
        raise ValueError(f"candidate deck hash is {deck_hash}, not frozen ScaleBCV2")
    library_path = candidate / "deck_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else None
    model_path = candidate / "model.pt"
    model = load_policy(model_path).eval()

    totals: Counter[str] = Counter()
    per_game: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    wobbuffet_decisions: list[dict[str, Any]] = []

    for path in sorted(args.replays.resolve().glob("episode-*-replay.json")):
        seat, mirror, decisions = focal_decisions(path)
        agent = NNAgent(
            model,
            greedy=True,
            deck_ids=deck,
            forbid_end_when_attack=True,
            opponent_deck_priors=library,
        )
        trace_belief = Belief(deck, opponent_deck_priors=library)
        game_counts: Counter[str] = Counter()
        for decision_index, (observation, expected) in enumerate(decisions):
            selection = observation.get("select") or {}
            options = selection.get("option") or []
            upper = min(int(selection.get("maxCount") or 0), len(options))
            base = features.extract(
                observation,
                belief=trace_belief,
                slot_link=getattr(model, "slot_link", False),
            )
            traced: list[int] = []
            trace_logits: list[torch.Tensor] = []
            for _ in range(upper):
                conditioned = base if upper == 1 else features.mark_selection(base, traced, upper)
                logits, _ = model(feat_to_tensors(conditioned))
                if upper == 1:
                    logits = attack_safe_logits(logits, options)
                choice = int(torch.argmax(logits))
                traced.append(choice)
                trace_logits.append(logits.detach().cpu())
            predicted = list(agent(observation))
            if traced != predicted:
                raise RuntimeError(
                    f"manual trace diverged in {path.name} decision {decision_index}: "
                    f"trace={traced}, agent={predicted}"
                )

            totals["decisions"] += 1
            game_counts["decisions"] += 1
            equivalent = (
                upper == len(options)
                and len(expected) == upper
                and sorted(predicted) == sorted(expected)
            )
            if predicted == expected:
                totals["exact_matches"] += 1
                game_counts["exact_matches"] += 1
            elif equivalent:
                totals["set_equivalent_order"] += 1
                game_counts["set_equivalent_order"] += 1
            else:
                totals["mismatches"] += 1
                game_counts["mismatches"] += 1
                mismatches.append(
                    {
                        "episode_file": path.name,
                        "decision_index": decision_index,
                        "turn": (observation.get("current") or {}).get("turn"),
                        "expected": expected,
                        "predicted": predicted,
                    }
                )

            state = observation.get("current") or {}
            players = state.get("players") or []
            mine = players[seat] if seat < len(players) else {}
            active = ((mine.get("active") or [{}])[0] or {})
            chosen = [options[index] for index in expected if 0 <= index < len(options)]
            chose_wobb_attack = (
                int(active.get("id") or 0) == audit.WOBBUFFET
                and any(int(option.get("type") or -1) == audit.OPTION_ATTACK for option in chosen)
            )
            if chose_wobb_attack and upper == 1 and trace_logits:
                logits = trace_logits[0]
                probabilities = torch.softmax(logits, dim=0)
                ready = audit.ready_attacker_ids(mine)
                wobbuffet_decisions.append(
                    {
                        "episode_id": int(path.name.split("-")[1]),
                        "decision_index": decision_index,
                        "turn": state.get("turn"),
                        "active_energy_cards": [
                            int(card.get("id") or 0) for card in audit.energy_cards(active)
                        ],
                        "ready_attacker_names": [names.get(card_id, str(card_id)) for card_id in ready],
                        "ready_spidops": audit.SPIDOPS in ready,
                        "options": [
                            {
                                "index": index,
                                "label": option_label(option, state, seat, names),
                                "type": option.get("type"),
                                "logit": float(logits[index]),
                                "probability": float(probabilities[index]),
                                "recorded": index in expected,
                            }
                            for index, option in enumerate(options)
                        ],
                    }
                )

        per_game.append(
            {
                "episode_id": int(path.name.split("-")[1]),
                "seat": seat,
                "validation_mirror": mirror,
                **dict(game_counts),
            }
        )
        totals["games"] += 1

    payload = {
        "schema": "scalebcv2-kaggle-policy-introspection/v1",
        "candidate": str(candidate),
        "replays": str(args.replays.resolve()),
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        "deck_sha256": deck_hash,
        "model_config": getattr(model, "cfg", None),
        "totals": dict(totals),
        "per_game": per_game,
        "mismatches": mismatches,
        "wobbuffet_attack_decisions": wobbuffet_decisions,
        "evidence_boundary": (
            "Logits reproduce recorded choices. Alternative options were not executed in the official "
            "engine, so this artifact does not identify their counterfactual terminal outcomes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"totals": payload["totals"], "wobbuffet": wobbuffet_decisions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
