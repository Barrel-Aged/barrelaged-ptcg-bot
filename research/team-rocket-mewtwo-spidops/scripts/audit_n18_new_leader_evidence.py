#!/usr/bin/env python3
"""Audit fresh public-arena evidence for the preregistered N18 decision.

This is a read-only evidence audit.  It verifies that replay actions are legal
for their row-local official-engine observations, fingerprints the current
leader bundles, measures corpus overlap, and checks neural architecture
compatibility with the unchanged live Team Rocket policy.  It does not train,
select a candidate, or authorize deployment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPORA = {
    "AshMarnieGeneralV177MultiFailure": ROOT / "work/n18/am",
    "SparkZoroarkArenaV2": ROOT / "work/n18/sz",
}
DEFAULT_BUNDLES = {
    "AshMarnieGeneralV177MultiFailure": ROOT / "work/n18/bam",
    "SparkZoroarkArenaV2": ROOT / "work/n18/bsz",
}
DEFAULT_ARCHIVES = {
    "AshMarnieGeneralV177MultiFailure": (
        ROOT
        / "artifacts/team_rocket_recovery/n18_new_leader_evidence/bundles/"
        "AshMarnieGeneralV177MultiFailure.tar.gz"
    ),
    "SparkZoroarkArenaV2": (
        ROOT
        / "artifacts/team_rocket_recovery/n18_new_leader_evidence/bundles/"
        "SparkZoroarkArenaV2.tar.gz"
    ),
}
DEFAULT_LIVE = ROOT / "work/n18/blive"
DEFAULT_LIVE_ARCHIVE = ROOT / "work/n18/live.tar.gz"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "N18_NEW_LEADER_EVIDENCE_AUDIT.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deck_ids(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_deck_digest(cards: list[int]) -> str:
    payload = "".join(f"{card}\n" for card in cards).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def outcome_label(winner: int, seat: int) -> str:
    if winner == seat:
        return "win"
    if winner in (0, 1):
        return "loss"
    return "draw"


def audit_corpus(bot: str, replay_dir: Path, expected_deck: list[int]) -> tuple[dict[str, Any], set[str]]:
    paths = sorted(replay_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no replay files for {bot}: {replay_dir}")

    totals: Counter[str] = Counter()
    seats: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    selected_types: Counter[str] = Counter()
    opponents: dict[str, Counter[str]] = defaultdict(Counter)
    file_hashes: set[str] = set()
    file_hash_sequence: list[tuple[str, str]] = []
    run_ids: Counter[str] = Counter()
    deck_sequences: Counter[str] = Counter()

    for path in paths:
        rows = load_rows(path)
        if len(rows) < 2 or "meta" not in rows[0]:
            raise ValueError(f"invalid arena replay header: {path}")
        meta = rows[0]["meta"]
        players = list(meta.get("players") or [])
        decks = list(meta.get("decks") or [])
        if players.count(bot) != 1 or len(players) != 2 or len(decks) != 2:
            raise ValueError(f"unexpected players/decks in {path.name}")
        seat = players.index(bot)
        opponent = players[1 - seat]
        observed_deck = [int(card) for card in decks[seat]]
        if observed_deck != expected_deck:
            raise ValueError(f"bundle/replay deck mismatch for {bot}: {path.name}")
        deck_sequences[canonical_deck_digest(observed_deck)] += 1
        seats[str(seat)] += 1
        run_ids[path.name.split("_g", 1)[0]] += 1

        result_rows = [row["result"] for row in rows if "result" in row]
        if len(result_rows) != 1:
            raise ValueError(f"expected exactly one result row: {path.name}")
        result = result_rows[0]
        winner_raw = result.get("winner")
        winner = int(winner_raw) if winner_raw in (0, 1) else -1
        label = outcome_label(winner, seat)
        outcomes[label] += 1
        opponents[opponent][label] += 1
        opponents[opponent]["games"] += 1
        totals["official_result_steps"] += int(result.get("steps") or 0)

        for row in rows[1:]:
            if "obs" not in row or int(row.get("p", -1)) != seat:
                continue
            observation = row["obs"]
            selection = observation.get("select") or {}
            options = list(selection.get("option") or [])
            action = row.get("action")
            if not isinstance(action, list):
                raise ValueError(f"non-list action in {path.name}")
            lower = int(selection.get("minCount") or 0)
            upper = min(int(selection.get("maxCount") or 0), len(options))
            legal = (
                len(action) == len(set(action))
                and lower <= len(action) <= upper
                and all(isinstance(index, int) and 0 <= index < len(options) for index in action)
            )
            totals["decisions"] += 1
            totals["legal_decisions"] += int(legal)
            totals["choice_targets"] += len(action)
            totals["nontrivial_decisions"] += int(len(options) > 1)
            totals["full_upper_bound_decisions"] += int(len(action) == upper)
            totals["optional_empty_decisions"] += int(lower == 0 and not action)
            totals[f"{label}_decisions"] += 1
            totals[f"{label}_choice_targets"] += len(action)
            for index in action:
                if legal:
                    selected_types[str(int(options[index].get("type", -1)))] += 1
        if totals["legal_decisions"] != totals["decisions"]:
            raise ValueError(f"illegal recorded decision detected by {path.name}")

        file_hash = digest(path)
        if file_hash in file_hashes:
            raise ValueError(f"duplicate replay payload inside {bot} corpus: {path.name}")
        file_hashes.add(file_hash)
        file_hash_sequence.append((path.name, file_hash))

    corpus_hasher = hashlib.sha256()
    for name, file_hash in file_hash_sequence:
        corpus_hasher.update(name.encode("utf-8"))
        corpus_hasher.update(bytes.fromhex(file_hash))
    opponent_rows = {
        name: dict(sorted(counts.items()))
        for name, counts in sorted(opponents.items())
    }
    return (
        {
            "bot": bot,
            "replay_directory": str(replay_dir.resolve()),
            "games": len(paths),
            "outcomes": dict(sorted(outcomes.items())),
            "seats": dict(sorted(seats.items())),
            "run_count": len(run_ids),
            "run_game_counts": dict(sorted(run_ids.items())),
            "opponent_count": len(opponents),
            "opponents": opponent_rows,
            "recorded_actions_are_row_local": True,
            "official_engine_action_legality": {
                "decisions": totals["decisions"],
                "legal_decisions": totals["legal_decisions"],
                "choice_targets": totals["choice_targets"],
                "nontrivial_decisions": totals["nontrivial_decisions"],
                "full_upper_bound_decisions": totals["full_upper_bound_decisions"],
                "optional_empty_decisions": totals["optional_empty_decisions"],
                "win_decisions": totals["win_decisions"],
                "win_choice_targets": totals["win_choice_targets"],
                "loss_decisions": totals["loss_decisions"],
                "loss_choice_targets": totals["loss_choice_targets"],
                "draw_decisions": totals["draw_decisions"],
                "draw_choice_targets": totals["draw_choice_targets"],
            },
            "selected_option_types": dict(sorted(selected_types.items())),
            "total_official_result_steps": totals["official_result_steps"],
            "deck_sequence_sha256": next(iter(deck_sequences)),
            "deck_sequence_count": sum(deck_sequences.values()),
            "corpus_sha256": corpus_hasher.hexdigest(),
            "file_payload_sha256_count": len(file_hashes),
        },
        file_hashes,
    )


def tensor_parameter_count(state: dict[str, Any]) -> int:
    return sum(int(value.numel()) for value in state.values() if isinstance(value, torch.Tensor))


def model_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise TypeError(f"unexpected model payload: {path}")
    return payload


def compare_models(reference: Path, candidate: Path) -> dict[str, Any]:
    left = model_payload(reference)
    right = model_payload(candidate)
    left_state = left["state_dict"]
    right_state = right["state_dict"]
    shared = sorted(set(left_state) & set(right_state))
    shape_mismatches = [
        key for key in shared if tuple(left_state[key].shape) != tuple(right_state[key].shape)
    ]
    exact_equal = [
        key
        for key in shared
        if not shape_mismatches
        and torch.equal(left_state[key], right_state[key])
    ]
    changed_parameters = sum(
        int(right_state[key].numel())
        for key in shared
        if key not in exact_equal and key not in shape_mismatches
    )
    return {
        "reference": str(reference.resolve()),
        "candidate": str(candidate.resolve()),
        "reference_sha256": digest(reference),
        "candidate_sha256": digest(candidate),
        "config_equal": left.get("config") == right.get("config"),
        "config": right.get("config"),
        "reference_state_keys": len(left_state),
        "candidate_state_keys": len(right_state),
        "missing_candidate_keys": sorted(set(left_state) - set(right_state)),
        "extra_candidate_keys": sorted(set(right_state) - set(left_state)),
        "shape_mismatches": shape_mismatches,
        "exact_equal_tensors": len(exact_equal),
        "changed_tensors": len(shared) - len(exact_equal) - len(shape_mismatches),
        "candidate_parameters": tensor_parameter_count(right_state),
        "changed_parameters": changed_parameters,
        "architecture_compatible": (
            left.get("config") == right.get("config")
            and set(left_state) == set(right_state)
            and not shape_mismatches
        ),
    }


def audit_history_residual(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise TypeError(f"unexpected history-residual payload: {path}")
    projection = payload.get("projection")
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "schema": payload.get("schema"),
        "global_width": int(payload.get("global_width") or 0),
        "option_width": int(payload.get("option_width") or 0),
        "memory": int(payload.get("memory") or 0),
        "projection_shape": list(projection.shape) if isinstance(projection, torch.Tensor) else None,
        "state_keys": len(payload["state_dict"]),
        "parameters": tensor_parameter_count(payload["state_dict"]),
        "declared_base": payload.get("base"),
        "card_belief_checkpoint": payload.get("card_belief_checkpoint"),
        "card_belief_representation": payload.get("card_belief_representation"),
    }


def bundle_audit(bot: str, bundle: Path, archive: Path) -> dict[str, Any]:
    cards = deck_ids(bundle / "deck.csv")
    return {
        "bot": bot,
        "bundle_directory": str(bundle.resolve()),
        "archive": str(archive.resolve()),
        "archive_sha256": digest(archive),
        "archive_bytes": archive.stat().st_size,
        "main_sha256": digest(bundle / "main.py"),
        "model_sha256": digest(bundle / "model.pt"),
        "deck_sha256": digest(bundle / "deck.csv"),
        "deck_sequence_sha256": canonical_deck_digest(cards),
        "deck_count": len(cards),
        "deck_distinct_cards": len(set(cards)),
        "deck_card_counts": {str(card): count for card, count in sorted(Counter(cards).items())},
        "history_residual": audit_history_residual(bundle / "history_residual.pt"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    live_cards = deck_ids(DEFAULT_LIVE / "deck.csv")
    live = {
        "bundle_directory": str(DEFAULT_LIVE.resolve()),
        "archive": str(DEFAULT_LIVE_ARCHIVE.resolve()),
        "archive_sha256": digest(DEFAULT_LIVE_ARCHIVE),
        "model_sha256": digest(DEFAULT_LIVE / "model.pt"),
        "deck_sha256": digest(DEFAULT_LIVE / "deck.csv"),
        "deck_sequence_sha256": canonical_deck_digest(live_cards),
        "deck_count": len(live_cards),
        "deck_distinct_cards": len(set(live_cards)),
    }
    if len(live_cards) != 60:
        raise ValueError("live Team Rocket deck is not exactly 60 cards")

    bundles = {
        bot: bundle_audit(bot, DEFAULT_BUNDLES[bot], DEFAULT_ARCHIVES[bot])
        for bot in DEFAULT_BUNDLES
    }
    corpora: dict[str, Any] = {}
    hashes: dict[str, set[str]] = {}
    for bot, replay_dir in DEFAULT_CORPORA.items():
        report, file_hashes = audit_corpus(
            bot, replay_dir, deck_ids(DEFAULT_BUNDLES[bot] / "deck.csv")
        )
        corpora[bot] = report
        hashes[bot] = file_hashes

    bots = list(DEFAULT_CORPORA)
    overlap = hashes[bots[0]] & hashes[bots[1]]
    union = hashes[bots[0]] | hashes[bots[1]]
    top_models_identical = (
        bundles[bots[0]]["model_sha256"] == bundles[bots[1]]["model_sha256"]
    )
    top_residuals_identical = (
        bundles[bots[0]]["history_residual"]["sha256"]
        == bundles[bots[1]]["history_residual"]["sha256"]
    )
    top_decks_identical = (
        bundles[bots[0]]["deck_sequence_sha256"]
        == bundles[bots[1]]["deck_sequence_sha256"]
    )
    comparisons = {
        bot: compare_models(DEFAULT_LIVE / "model.pt", DEFAULT_BUNDLES[bot] / "model.pt")
        for bot in DEFAULT_BUNDLES
    }

    report = {
        "schema": "rocket-n18-new-leader-evidence-audit/v1",
        "purpose": "fresh evidence audit before N18 preregistration",
        "live_control": live,
        "leader_bundles": bundles,
        "leader_corpora": corpora,
        "cross_corpus": {
            "file_payload_overlap": len(overlap),
            "deduplicated_union_games": len(union),
            "top_models_identical": top_models_identical,
            "top_history_residuals_identical": top_residuals_identical,
            "top_decks_identical": top_decks_identical,
            "independent_strategy_count": 1 if (
                top_models_identical and top_residuals_identical and top_decks_identical
            ) else 2,
        },
        "live_to_leader_model_compatibility": comparisons,
        "admissible_facts": {
            "replays_are_genuine_internal_arena_official_engine_execution": True,
            "all_recorded_leader_actions_are_legal_for_row_local_observations": all(
                corpus["official_engine_action_legality"]["decisions"]
                == corpus["official_engine_action_legality"]["legal_decisions"]
                for corpus in corpora.values()
            ),
            "leader_model_architecture_matches_live": all(
                comparison["architecture_compatible"] for comparison in comparisons.values()
            ),
            "leader_deck_is_not_team_rocket": all(
                bundle["deck_sequence_sha256"] != live["deck_sequence_sha256"]
                for bundle in bundles.values()
            ),
            "leader_bundle_can_be_used_as_a_direct_rocket_candidate_without_official_match_testing": False,
            "leader_replays_can_authorize_deployment_without_official_match_testing": False,
        },
        "claim_boundary": (
            "This audit proves provenance, legality, corpus contents, duplication, and architecture "
            "compatibility only.  It does not establish cross-deck transfer quality, Team Rocket "
            "strength, causal improvement, confirmation, or deployment authority."
        ),
        "training_performed": False,
        "official_match_panel_performed": False,
        "deployment_authority": False,
    }
    if not report["admissible_facts"]["all_recorded_leader_actions_are_legal_for_row_local_observations"]:
        raise AssertionError("N18 leader corpus contains an illegal recorded action")
    if not top_models_identical or not top_residuals_identical:
        raise AssertionError("expected the audited #1/#2 bundles to share the same strategy")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": digest(args.output),
                "corpus_games": {bot: corpus["games"] for bot, corpus in corpora.items()},
                "deduplicated_union_games": len(union),
                "independent_strategy_count": report["cross_corpus"]["independent_strategy_count"],
                "deployment_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
