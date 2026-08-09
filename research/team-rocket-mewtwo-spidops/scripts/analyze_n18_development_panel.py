#!/usr/bin/env python3
"""Analyze the complete locked N18 development panel exactly once."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PANEL = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "development_panel"
)
OUTPUT = PANEL / "analysis.json"
ARMS = (
    ROOT
    / "artifacts/team_rocket_recovery/n18_new_leader_evidence/"
    "development_arms/arms.json"
)
RUNTIME_AUDIT = ARMS.parent / "runtime_audit.json"
PROTOCOL = ROOT / "N18_CROSS_DECK_LEADER_TRANSFER_PROTOCOL.md"

ARMS_SHA = "57ab74eaaf05aa2f4e36a09f1555aef4eadf1d84dd0415c59376de036bfbed18"
RUNTIME_AUDIT_SHA = "93c2e64764108ee78415b13ce3de3bf6e0c7bc8256b44804a002081e802f0b93"
PROTOCOL_SHA = "b64bd56fb0c1593f2310b636723c83c7c08596f5dbb63c3d9e848015b3c25805"
COLLECTOR_SHA = "2891a7c6ec14e7c6de68f755b66e5ede606bcf302ac23e01254b4bcc02844625"
DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"

MATCH_ARMS = ("control", "live_residual", "leader_base", "leader_residual")
CHALLENGERS = ("live_residual", "leader_base", "leader_residual")
TIE_PRIORITY = ("leader_residual", "leader_base", "live_residual")
GAMES_PER_ARM_CELL = 120
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 818001
BONFERRONI_LOWER = 0.05 / (2 * len(CHALLENGERS))
BONFERRONI_UPPER = 1.0 - BONFERRONI_LOWER

SHARDS = {
    "leader_seat0.jsonl": {
        "opponent": "leader",
        "opponent_bundle_name": "bam",
        "opponent_tree_sha256": "1a78564bb3304fc4c968e14cb01245eba2c232fe32594c0cfdb563a01a76e828",
        "seed": 818101,
        "seat": 0,
    },
    "leader_seat1.jsonl": {
        "opponent": "leader",
        "opponent_bundle_name": "bam",
        "opponent_tree_sha256": "1a78564bb3304fc4c968e14cb01245eba2c232fe32594c0cfdb563a01a76e828",
        "seed": 818102,
        "seat": 1,
    },
    "v110_seat0.jsonl": {
        "opponent": "v110",
        "opponent_bundle_name": "AshMarnieGeneralV110_extracted",
        "opponent_tree_sha256": "3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b",
        "seed": 818201,
        "seat": 0,
    },
    "v110_seat1.jsonl": {
        "opponent": "v110",
        "opponent_bundle_name": "AshMarnieGeneralV110_extracted",
        "opponent_tree_sha256": "3bb5e40bb1583e31470eee263ed615c9207427c7b5d2199e80fbe3571e2cbb2b",
        "seed": 818202,
        "seat": 1,
    },
    "recovery_seat0.jsonl": {
        "opponent": "recovery",
        "opponent_bundle_name": "HelbrosMarnieGeneralRecoveryV4_extracted",
        "opponent_tree_sha256": "caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4",
        "seed": 818301,
        "seat": 0,
    },
    "recovery_seat1.jsonl": {
        "opponent": "recovery",
        "opponent_bundle_name": "HelbrosMarnieGeneralRecoveryV4_extracted",
        "opponent_tree_sha256": "caf361fd297ab63b24338650b6e396a296b961bb6ba61d2519799a90dff738f4",
        "seed": 818302,
        "seat": 1,
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def interval(values: np.ndarray, lower: float, upper: float) -> list[float]:
    result = np.quantile(values, [lower, upper])
    return [float(result[0]), float(result[1])]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing N18 analysis: {OUTPUT}")
    for path, expected in (
        (ARMS, ARMS_SHA),
        (RUNTIME_AUDIT, RUNTIME_AUDIT_SHA),
        (PROTOCOL, PROTOCOL_SHA),
        (ROOT / "scripts/collect_n18_development_panel.py", COLLECTOR_SHA),
    ):
        if digest(path) != expected:
            raise ValueError(f"N18 analyzer input changed: {path}")

    arms_manifest = json.loads(ARMS.read_text(encoding="utf-8"))
    arm_entries = {item["name"]: item for item in arms_manifest.get("arms") or ()}
    if any(name not in arm_entries for name in MATCH_ARMS):
        raise ValueError("N18 analyzer arm manifest changed")

    cells: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    draws: Counter[tuple[str, int, str]] = Counter()
    total_rows = 0
    correctness_errors: list[str] = []
    shard_reports: dict[str, Any] = {}

    for filename, expected in SHARDS.items():
        path = PANEL / filename
        sidecar = path.with_suffix(".manifest.json")
        if not path.exists() or not sidecar.exists():
            raise FileNotFoundError(f"N18 panel is incomplete: {path}")
        rows = read_jsonl(path)
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        if digest(path) != manifest.get("output_sha256"):
            correctness_errors.append(f"output hash mismatch: {filename}")
        expected_rows = GAMES_PER_ARM_CELL * len(MATCH_ARMS)
        if len(rows) != expected_rows or int(manifest.get("rows") or -1) != expected_rows:
            correctness_errors.append(f"row count mismatch: {filename}")
        if (
            manifest.get("schema") != "rocket-n18-development-shard/v1"
            or manifest.get("protocol") != "rocket-n18-cross-deck-leader-transfer/v1"
            or manifest.get("collector_sha256") != COLLECTOR_SHA
            or manifest.get("protocol_sha256") != PROTOCOL_SHA
            or manifest.get("arms_manifest_sha256") != ARMS_SHA
            or manifest.get("runtime_audit_sha256") != RUNTIME_AUDIT_SHA
            or manifest.get("exact_team_rocket_deck_sha256") != DECK_SHA
            or manifest.get("official_engine_execution") is not True
            or manifest.get("packaged_worker_execution") is not True
            or manifest.get("strict_legality") is not True
            or manifest.get("hidden_information_used_by_actor") is not False
            or manifest.get("opponent_tree_sha256") != expected["opponent_tree_sha256"]
            or Path(str(manifest.get("opponent"))).name != expected["opponent_bundle_name"]
            or int(manifest.get("schedule_seed") or -1) != expected["seed"]
            or int(manifest.get("own_seat") if manifest.get("own_seat") is not None else -1)
            != expected["seat"]
        ):
            correctness_errors.append(f"manifest mismatch: {filename}")

        counts: Counter[str] = Counter()
        wins: Counter[str] = Counter()
        shard_draws: Counter[str] = Counter()
        for row in rows:
            arm = str(row.get("arm"))
            if arm not in MATCH_ARMS:
                correctness_errors.append(f"unknown arm in {filename}: {arm}")
                continue
            result = int(row.get("result") if row.get("result") is not None else -1)
            own_seat = int(row.get("own_seat") if row.get("own_seat") is not None else -1)
            valid = (
                row.get("schema") == "rocket-n18-development-match/v1"
                and row.get("protocol") == "rocket-n18-cross-deck-leader-transfer/v1"
                and int(row.get("schedule_seed") or -1) == expected["seed"]
                and own_seat == expected["seat"]
                and int(row.get("first_player") if row.get("first_player") is not None else -1) == 0
                and bool(row.get("started_first")) == (own_seat == 0)
                and bool(row.get("started_second")) == (own_seat == 1)
                and result in (0, 1, 2)
                and int(row.get("win") or 0) == int(result == own_seat)
                and int(row.get("draw") or 0) == int(result == 2)
                and int(row.get("illegal_actions") or 0) == 0
                and int(row.get("exception_fallbacks") or 0) == 0
                and row.get("exact_team_rocket_deck_sha256") == DECK_SHA
                and row.get("arm_bundle_tree_sha256")
                == arm_entries[arm]["bundle_tree_sha256"]
                and row.get("arm_model_sha256") == arm_entries[arm]["model_sha256"]
                and row.get("arm_main_sha256") == arm_entries[arm]["main_sha256"]
                and row.get("arm_history_residual_sha256")
                == arm_entries[arm]["history_residual_sha256"]
                and row.get("opponent_tree_sha256") == expected["opponent_tree_sha256"]
                and row.get("opponent") == expected["opponent_bundle_name"]
            )
            if not valid:
                correctness_errors.append(
                    f"row correctness mismatch: {filename}:{row.get('game_index')}"
                )
            win = int(row.get("win") or 0)
            draw = int(row.get("draw") or 0)
            key = (expected["opponent"], expected["seat"], arm)
            cells[key].append(win)
            draws[key] += draw
            counts[arm] += 1
            wins[arm] += win
            shard_draws[arm] += draw
            total_rows += 1
        if any(counts[name] != GAMES_PER_ARM_CELL for name in MATCH_ARMS):
            correctness_errors.append(f"unbalanced arm count: {filename}")
        shard_reports[filename] = {
            "output_sha256": digest(path),
            "manifest_sha256": digest(sidecar),
            "opponent": expected["opponent"],
            "own_seat": expected["seat"],
            "schedule_seed": expected["seed"],
            "counts": dict(counts),
            "wins": dict(wins),
            "draws": dict(shard_draws),
        }

    expected_total = len(SHARDS) * len(MATCH_ARMS) * GAMES_PER_ARM_CELL
    if total_rows != expected_total:
        correctness_errors.append("total row count mismatch")
    for opponent in ("leader", "v110", "recovery"):
        for seat in (0, 1):
            for arm in MATCH_ARMS:
                if len(cells[(opponent, seat, arm)]) != GAMES_PER_ARM_CELL:
                    correctness_errors.append(
                        f"cell count mismatch: {opponent}/seat{seat}/{arm}"
                    )

    correctness_passes = not correctness_errors
    if not correctness_passes:
        raise AssertionError(json.dumps(correctness_errors[:30], indent=2))

    cell_rates: dict[str, Any] = {}
    for opponent in ("leader", "v110", "recovery"):
        for seat in (0, 1):
            label = f"{opponent}/seat{seat}"
            cell_rates[label] = {
                arm: {
                    "games": len(cells[(opponent, seat, arm)]),
                    "wins": int(sum(cells[(opponent, seat, arm)])),
                    "draws": int(draws[(opponent, seat, arm)]),
                    "win_rate": float(np.mean(cells[(opponent, seat, arm)])),
                }
                for arm in MATCH_ARMS
            }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_means: dict[tuple[str, int, str], np.ndarray] = {}
    for opponent in ("leader", "v110", "recovery"):
        for seat in (0, 1):
            for arm in MATCH_ARMS:
                values = cells[(opponent, seat, arm)]
                probability = float(np.mean(values))
                bootstrap_means[(opponent, seat, arm)] = rng.binomial(
                    len(values), probability, size=BOOTSTRAP_REPLICATES
                ) / len(values)

    candidate_reports: dict[str, Any] = {}
    passing: list[str] = []
    for candidate in CHALLENGERS:
        cell_diffs: dict[str, float] = {}
        replicate_diffs: list[np.ndarray] = []
        for opponent in ("leader", "v110", "recovery"):
            for seat in (0, 1):
                control_rate = float(np.mean(cells[(opponent, seat, "control")]))
                candidate_rate = float(np.mean(cells[(opponent, seat, candidate)]))
                cell_diffs[f"{opponent}/seat{seat}"] = candidate_rate - control_rate
                replicate_diffs.append(
                    bootstrap_means[(opponent, seat, candidate)]
                    - bootstrap_means[(opponent, seat, "control")]
                )
        aggregate_replicates = np.mean(np.stack(replicate_diffs, axis=0), axis=0)
        aggregate = float(np.mean(list(cell_diffs.values())))
        opponent_diffs = {
            opponent: float(
                np.mean([cell_diffs[f"{opponent}/seat0"], cell_diffs[f"{opponent}/seat1"]])
            )
            for opponent in ("leader", "v110", "recovery")
        }
        seat_diffs = {
            str(seat): float(
                np.mean(
                    [
                        cell_diffs[f"{opponent}/seat{seat}"]
                        for opponent in ("leader", "v110", "recovery")
                    ]
                )
            )
            for seat in (0, 1)
        }
        ordinary = interval(aggregate_replicates, 0.025, 0.975)
        simultaneous = interval(
            aggregate_replicates, BONFERRONI_LOWER, BONFERRONI_UPPER
        )
        gates = {
            "correctness": correctness_passes,
            "aggregate_at_least_plus_2pp": aggregate >= 0.02,
            "positive_every_opponent": all(value > 0.0 for value in opponent_diffs.values()),
            "positive_both_seats": all(value > 0.0 for value in seat_diffs.values()),
            "no_cell_at_or_below_minus_5pp": min(cell_diffs.values()) > -0.05,
            "simultaneous_lower_above_minus_half_pp": simultaneous[0] > -0.005,
        }
        passes = all(gates.values())
        if passes:
            passing.append(candidate)
        candidate_reports[candidate] = {
            "aggregate_improvement": aggregate,
            "aggregate_improvement_pp": aggregate * 100.0,
            "ordinary_95_interval": ordinary,
            "ordinary_95_interval_pp": [value * 100.0 for value in ordinary],
            "bonferroni_simultaneous_95_interval": simultaneous,
            "bonferroni_simultaneous_95_interval_pp": [
                value * 100.0 for value in simultaneous
            ],
            "cell_improvements": cell_diffs,
            "cell_improvements_pp": {key: value * 100.0 for key, value in cell_diffs.items()},
            "opponent_improvements": opponent_diffs,
            "opponent_improvements_pp": {
                key: value * 100.0 for key, value in opponent_diffs.items()
            },
            "seat_improvements": seat_diffs,
            "seat_improvements_pp": {key: value * 100.0 for key, value in seat_diffs.items()},
            "gates": gates,
            "passes": passes,
        }

    selected: str | None = None
    if passing:
        priority = {name: index for index, name in enumerate(TIE_PRIORITY)}
        selected = sorted(
            passing,
            key=lambda name: (
                -candidate_reports[name]["aggregate_improvement"],
                priority[name],
            ),
        )[0]

    report = {
        "schema": "rocket-n18-development-analysis/v1",
        "protocol": "rocket-n18-cross-deck-leader-transfer/v1",
        "protocol_sha256": PROTOCOL_SHA,
        "collector_sha256": COLLECTOR_SHA,
        "analyzer_sha256": digest(Path(__file__).resolve()),
        "arms_manifest_sha256": ARMS_SHA,
        "runtime_audit_sha256": RUNTIME_AUDIT_SHA,
        "exact_team_rocket_deck_sha256": DECK_SHA,
        "official_engine_execution": True,
        "packaged_worker_execution": True,
        "strict_correctness": correctness_passes,
        "correctness_errors": correctness_errors,
        "total_games": total_rows,
        "games_per_arm": total_rows // len(MATCH_ARMS),
        "games_per_arm_cell": GAMES_PER_ARM_CELL,
        "shards": shard_reports,
        "cell_results": cell_rates,
        "bootstrap": {
            "method": "equal-cell stratified nonparametric Bernoulli bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ordinary_quantiles": [0.025, 0.975],
            "bonferroni_family_size": len(CHALLENGERS),
            "bonferroni_quantiles": [BONFERRONI_LOWER, BONFERRONI_UPPER],
        },
        "candidate_results": candidate_reports,
        "passing_candidates": passing,
        "selection_tie_priority": list(TIE_PRIORITY),
        "selected_candidate": selected,
        "development_gate_passes": selected is not None,
        "fresh_confirmation_authority": selected is not None,
        "deployment_authority": False,
        "claim_boundary": (
            "Development evidence only. A selected candidate may enter the preregistered fresh "
            "confirmation; it may not be submitted or described as stronger without confirmation "
            "and broad regression."
        ),
    }
    OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "output_sha256": digest(OUTPUT),
                "total_games": total_rows,
                "passing_candidates": passing,
                "selected_candidate": selected,
                "deployment_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
