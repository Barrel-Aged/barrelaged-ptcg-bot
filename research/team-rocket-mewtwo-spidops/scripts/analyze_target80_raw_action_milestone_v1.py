#!/usr/bin/env python3
"""Audit the locked Target-80 raw-action milestone V1 pilot gates."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OWN = {
    "own_deck_sha256": "0bb258484e5638f3c51ac45def50fc2a1056feb20d27bd3842266514c8e109d8",
    "own_model_sha256": "b4a361f42900629df28da9029e5193753344f71bc8806748b971cb551bfb674f",
    "own_main_sha256": "3072e57ea576443365368926145645954854c9a340ef3ca5e729180a4ec4d341",
}
CELLS = (
    ("01_marnie_first", "AshV178MarnieFullBest_20260801_extracted", "first", 806101, "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"),
    ("02_marnie_second", "AshV178MarnieFullBest_20260801_extracted", "second", 806102, "92b92bac9f9163ecff933b3dc39294d2cc154c8684f3c8497877661419ebc59d"),
    ("03_mist_first", "SparkStarmieMistFlatV20_20260801_extracted", "first", 806201, "64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901"),
    ("04_mist_second", "SparkStarmieMistFlatV20_20260801_extracted", "second", 806202, "64ef6e2c0f5fadc12f96426aa80a07e1402501ed00d1778a0badd25d171e4901"),
    ("05_fros_first", "SparkStarmieFroslassCFV13_extracted", "first", 806301, "60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61"),
    ("06_fros_second", "SparkStarmieFroslassCFV13_extracted", "second", 806302, "60e91bf519988fd2efd14b8c9806bf03febf8baa2caab5250eab1da65b674d61"),
    ("07_rocket_first", "SparkRocketSpidopsV1_extracted", "first", 806401, "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"),
    ("08_rocket_second", "SparkRocketSpidopsV1_extracted", "second", 806402, "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"),
    ("09_purerl_first", "SoldierPureRLV45_extracted", "first", 806501, "80798a3cae4e541da01acaa9a58c493f3d5010571e4dbabb10bf7dc701c94281"),
    ("10_purerl_second", "SoldierPureRLV45_extracted", "second", 806502, "80798a3cae4e541da01acaa9a58c493f3d5010571e4dbabb10bf7dc701c94281"),
    ("11_alakazam_first", "VioletAlakazamMKV93_extracted", "first", 806601, "4e56a20249fad14097d06aaee2a241f355b90ef9bef19307f95b47798d0a2a3d"),
    ("12_alakazam_second", "VioletAlakazamMKV93_extracted", "second", 806602, "4e56a20249fad14097d06aaee2a241f355b90ef9bef19307f95b47798d0a2a3d"),
    ("13_crustle_first", "SparkTrueCrustleCrispinGalvV71_extracted", "first", 806701, "599393da0203019c8cc3b2192c54f91ea5213d8c6e9631070df14fce1944d79c"),
    ("14_crustle_second", "SparkTrueCrustleCrispinGalvV71_extracted", "second", 806702, "599393da0203019c8cc3b2192c54f91ea5213d8c6e9631070df14fce1944d79c"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "work" / "target80_raw_action_milestone_v1_pilot",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    dependencies: dict[str, set[Any]] = defaultdict(set)
    cell_counts: dict[str, int] = {}
    fingerprints: list[str] = []

    for stem, opponent, seat, seed, opponent_deck_sha in CELLS:
        data_path = input_dir / f"{stem}.jsonl"
        manifest_path = input_dir / f"{stem}.manifest.json"
        if not data_path.is_file() or not manifest_path.is_file():
            errors.append(f"missing cell artifacts: {stem}")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        cell_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line]
        cell_counts[stem] = len(cell_rows)
        if len(cell_rows) != 2:
            errors.append(f"{stem}: expected 2 roots, got {len(cell_rows)}")
        if sha256(data_path) != manifest.get("output_sha256"):
            errors.append(f"{stem}: output digest mismatch")
        for key, expected in EXPECTED_OWN.items():
            if manifest.get(key) != expected:
                errors.append(f"{stem}: {key} mismatch")
        if manifest.get("opponent_deck_sha256") != opponent_deck_sha:
            errors.append(f"{stem}: opponent deck mismatch")
        params = manifest.get("parameters") or {}
        if int(params.get("seed") or -1) != seed or params.get("seat") != seat:
            errors.append(f"{stem}: seed/seat mismatch")
        if Path(str(manifest.get("opponent"))).name != opponent:
            errors.append(f"{stem}: opponent identity mismatch")
        for key in ("collector_sha256", "v1_dependency_sha256", "v2_dependency_sha256"):
            dependencies[key].add(manifest.get(key))
        for row in cell_rows:
            rows.append(row)
            fingerprints.append(str(row.get("observation_fingerprint")))
            if row.get("opponent") != opponent:
                errors.append(f"{stem}: row opponent mismatch")
            if bool(row.get("started_first")) != (seat == "first"):
                errors.append(f"{stem}: row start-order mismatch")
            if not row.get("guarded_baseline_reproduced"):
                errors.append(f"{stem}: guarded baseline parity failed")
            if not row.get("fully_complete") or int(row.get("complete_pairs") or 0) != 16:
                errors.append(f"{stem}: incomplete paired samples")
            if int(row.get("attempted_pairs") or 0) != 16:
                errors.append(f"{stem}: attempted pair count mismatch")
            if row.get("errors"):
                errors.append(f"{stem}: collector errors present")
            candidate_count = len(row.get("candidate_indices") or ())
            if candidate_count < 2 or candidate_count > 3:
                errors.append(f"{stem}: invalid candidate count {candidate_count}")
            for particle in row.get("particle_manifests") or ():
                if particle.get("raw_card_ids_persisted") is not False:
                    errors.append(f"{stem}: raw hidden allocation persistence flag")

    for key, values in dependencies.items():
        if len(values) != 1 or None in values:
            errors.append(f"dependency identity unstable: {key}")
    if len(rows) != 28:
        errors.append(f"expected 28 total roots, got {len(rows)}")
    if len(set(fingerprints)) != len(fingerprints):
        errors.append("duplicate root observation fingerprint")

    stable = [row for row in rows if row.get("stable_label")]
    stable_changes = [row for row in stable if row.get("changed")]
    changed_root_deltas = [float(row["best_mean_paired_delta"]) for row in stable_changes]
    underlying_changed_deltas = [
        float(value)
        for row in stable_changes
        for value in (row.get("paired_deltas") or {}).get(str(row.get("best")), ())
    ]
    stable_by_start: dict[str, list[dict[str, Any]]] = {
        "first": [row for row in stable if row.get("started_first")],
        "second": [row for row in stable if not row.get("started_first")],
    }
    stable_changes_by_start = {
        name: sum(bool(row.get("changed")) for row in selected)
        for name, selected in stable_by_start.items()
    }
    start_advantage = {
        name: mean(float(row["best_mean_paired_delta"]) for row in selected)
        for name, selected in stable_by_start.items()
    }
    opponent_advantage: dict[str, float | None] = {}
    for opponent in sorted({cell[1] for cell in CELLS}):
        selected = [row for row in stable if row.get("opponent") == opponent]
        opponent_advantage[opponent] = mean(
            float(row["best_mean_paired_delta"]) for row in selected
        )
    nonnegative_opponents = sum(
        value is not None and value >= -1e-12 for value in opponent_advantage.values()
    )
    changed_median = statistics.median(changed_root_deltas) if changed_root_deltas else None
    changed_mean = mean(changed_root_deltas)
    negative_fraction = (
        sum(value < -1e-9 for value in underlying_changed_deltas)
        / len(underlying_changed_deltas)
        if underlying_changed_deltas
        else None
    )

    starmie_names = (
        "SparkStarmieMistFlatV20_20260801_extracted",
        "SparkStarmieFroslassCFV13_extracted",
    )
    gates = {
        "complete_28_balanced_roots": len(rows) == 28 and all(value == 2 for value in cell_counts.values()),
        "integrity_and_complete_pairs": not errors,
        "stable_labels_at_least_21": len(stable) >= 21,
        "stable_change_count_4_to_16_both_starts": (
            4 <= len(stable_changes) <= 16
            and stable_changes_by_start["first"] >= 1
            and stable_changes_by_start["second"] >= 1
        ),
        "stable_changed_advantage_and_tail": (
            changed_mean is not None
            and changed_mean >= 0.06
            and changed_median is not None
            and changed_median > 0.0
            and negative_fraction is not None
            and negative_fraction <= 0.25
        ),
        "both_start_advantage_at_least_003": all(
            value is not None and value >= 0.03 for value in start_advantage.values()
        ),
        "five_opponents_nonnegative_and_starmie_safe": (
            nonnegative_opponents >= 5
            and all(
                opponent_advantage[name] is not None and opponent_advantage[name] >= -0.02
                for name in starmie_names
            )
        ),
    }
    result = {
        "schema": "target80-raw-action-milestone-analysis/v1",
        "protocol": "TARGET80_RAW_ACTION_MILESTONE_POLICY_V1_PROTOCOL.md",
        "input_dir": str(input_dir),
        "input_file_sha256": {
            path.name: sha256(path) for path in sorted(input_dir.glob("*.jsonl"))
        },
        "manifests": len(manifests),
        "cell_counts": cell_counts,
        "roots": len(rows),
        "unique_root_fingerprints": len(set(fingerprints)),
        "stable_roots": len(stable),
        "stable_changes": len(stable_changes),
        "stable_changes_by_start": stable_changes_by_start,
        "stable_changed_mean_advantage": changed_mean,
        "stable_changed_median_advantage": changed_median,
        "stable_changed_underlying_negative_fraction": negative_fraction,
        "stable_advantage_by_start": start_advantage,
        "stable_advantage_by_opponent": opponent_advantage,
        "nonnegative_opponents": nonnegative_opponents,
        "errors": errors,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
