"""Audit and play the extracted arena archive against the official engine."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import statistics
import sys
import tarfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_integrated_controller import run_match  # noqa: E402
from rocketbot.official_evaluation import PriorityBaseline  # noqa: E402


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
    unsafe = [
        name
        for name in names
        if Path(name).is_absolute() or ".." in Path(name).parts
    ]
    if unsafe:
        raise AssertionError(f"unsafe archive paths: {unsafe}")
    return names


def smoke(archive_path: Path, extracted: Path, output: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve()
    extracted = extracted.resolve()
    members = _safe_members(archive_path)
    manifest = json.loads(
        (extracted / "submission_manifest.json").read_text(encoding="utf-8")
    )
    hash_mismatches = {
        name: {"expected": expected, "actual": _sha256(extracted / name)}
        for name, expected in manifest["files"].items()
        if not (extracted / name).is_file()
        or _sha256(extracted / name) != expected
    }
    if hash_mismatches:
        raise AssertionError(f"manifest hash mismatches: {hash_mismatches}")

    extracted_text = str(extracted)
    if extracted_text not in sys.path:
        sys.path.insert(0, extracted_text)
    packaged_main = importlib.import_module("main")
    deck_query = list(packaged_main.agent({"select": None}))
    packaged_deck = list(packaged_main.DECK.card_ids)
    csv_deck = [
        int(value)
        for value in (extracted / "deck.csv").read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if not deck_query == packaged_deck == csv_deck or len(csv_deck) != 60:
        raise AssertionError("packaged deck query, resolved deck, and deck.csv differ")
    summary = packaged_main.BOT.summary()
    if summary["strategy_provider"] != "frozen_public_belief_neural_blueprint":
        raise AssertionError("packaged strategy provider is not the neural blueprint")
    if summary["heuristic_final_controller"] or summary["sampled_rollout_final_controller"]:
        raise AssertionError("a prohibited legacy controller is marked final")

    from cg.game import battle_finish, battle_select, battle_start  # type: ignore[import-not-found]

    records: list[dict[str, Any]] = []
    for index in range(12):
        style = ("tempo", "attack_rush", "control")[index % 3]
        record = run_match(
            battle_start=battle_start,
            battle_select=battle_select,
            battle_finish=battle_finish,
            match_id=f"archive-smoke-{index:02d}",
            focal_label="packaged_integrated_controller",
            focal_agent=packaged_main.BOT,
            focal_deck=packaged_deck,
            opponent_label=f"priority_baseline_{style}",
            opponent_agent=PriorityBaseline(packaged_deck, style=style),
            opponent_deck=packaged_deck,
            focal_seat=index % 2,
            opponent_archetype="rocket_mewtwo",
            opponent_intent=None,
            capture=False,
            replay_directory=extracted / "unused_replays",
        )
        records.append(record)

    final_summary = packaged_main.BOT.summary()
    passed = bool(
        not hash_mismatches
        and all(record["engine_error"] is None for record in records)
        and sum(int(record["runner_illegal_fallbacks"]) for record in records) == 0
        and int(final_summary["exception_fallback_count"]) == 0
        and int(final_summary["decision_count"]) > 0
        and bool(final_summary["hidden_resource_hypotheses"])
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "archive_member_count": len(members),
        "manifest_file_hashes_verified": len(manifest["files"]),
        "manifest_hash_mismatches": hash_mismatches,
        "deck_count": len(packaged_deck),
        "deck_query_matches_csv": deck_query == csv_deck,
        "deck_resolution": manifest["deck_resolution"],
        "model_fingerprint": summary["model_fingerprint"],
        "strategy_provider": summary["strategy_provider"],
        "heuristic_final_controller": summary["heuristic_final_controller"],
        "sampled_rollout_final_controller": summary[
            "sampled_rollout_final_controller"
        ],
        "official_engine_games": len(records),
        "wins": sum(bool(record["focal_win"]) for record in records),
        "runner_illegal_fallbacks": sum(
            int(record["runner_illegal_fallbacks"]) for record in records
        ),
        "engine_errors": sum(record["engine_error"] is not None for record in records),
        "mean_latency_ms": statistics.fmean(
            float(record["mean_latency_ms"]) for record in records
        ),
        "max_latency_ms": max(float(record["max_latency_ms"]) for record in records),
        "final_controller_exception_fallbacks": final_summary[
            "exception_fallback_count"
        ],
        "final_hidden_hypothesis_count": len(
            final_summary["hidden_resource_hypotheses"]
        ),
        "tactical_teacher_deferences_last_game": final_summary[
            "tactical_teacher_deference_count"
        ],
        "claim_boundary": (
            "deterministic extracted-archive smoke only; strength is measured "
            "separately by the internal arena"
        ),
    }
    if not passed:
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "submissions"
            / "HelbrosMewtwoSpidopsGTWTV1.tar.gz"
        ),
    )
    parser.add_argument(
        "--extracted",
        type=Path,
        default=PROJECT_ROOT / "work" / "final_archive_audit_20260727_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "submissions"
            / "HelbrosMewtwoSpidopsGTWTV1_smoke_report.json"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(smoke(args.archive, args.extracted, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
