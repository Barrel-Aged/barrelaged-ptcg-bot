"""Prove that the packaged planner is behaviorally identical to the evaluated one.

The exploitation experiment runs after the main official-engine evaluation.  A
rejected candidate adds an evidence record to the JSON artifact but leaves every
runtime-bearing field unchanged.  This check removes precisely that evidence
record, recomputes the canonical fingerprint, and requires it to match the model
fingerprint recorded by the 240-game evaluation.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "integrated_controller" / "integrated_planner.json"
)
EVALUATION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "integrated_evaluation"
    / "integrated_evaluation_report.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "integrated_evaluation"
    / "deployed_artifact_binding_report.json"
)


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    deployed = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    deployed_fingerprint = str(deployed.pop("fingerprint"))
    exploitation = deployed.get("evidence", {}).pop(
        "opponent_exploitation_overlay", None
    )
    reconstructed_evaluated_fingerprint = _fingerprint(deployed)
    evaluated_fingerprint = str(evaluation["model_fingerprint"])
    exploitation_rejected = bool(
        isinstance(exploitation, dict)
        and not exploitation.get("admitted", True)
        and exploitation.get("safe_baseline_preserved", False)
    )
    gameplay_calibration_safe = bool(
        deployed["calibration"]["opponent_exploitation_admitted"] is False
        and float(deployed["calibration"]["exploitation_weight"]) == 0.0
    )
    passed = bool(
        reconstructed_evaluated_fingerprint == evaluated_fingerprint
        and exploitation_rejected
        and gameplay_calibration_safe
    )
    report = {
        "schema_version": 1,
        "passed": passed,
        "binding_method": (
            "remove only evidence.opponent_exploitation_overlay and recompute "
            "the canonical artifact fingerprint"
        ),
        "deployed_model_fingerprint": deployed_fingerprint,
        "deployed_model_sha256": sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "evaluated_model_fingerprint": evaluated_fingerprint,
        "reconstructed_evaluated_fingerprint": reconstructed_evaluated_fingerprint,
        "fingerprints_match": reconstructed_evaluated_fingerprint
        == evaluated_fingerprint,
        "only_post_evaluation_change": "rejected exploitation evidence record",
        "exploitation_candidate_admitted": bool(
            exploitation.get("admitted", False)
            if isinstance(exploitation, dict)
            else True
        ),
        "deployed_exploitation_weight": deployed["calibration"][
            "exploitation_weight"
        ],
        "gameplay_bearing_fields_unchanged": passed,
        "evaluation_game_count": evaluation["game_count"],
        "official_engine_execution": evaluation["official_engine_execution"],
    }
    if not passed:
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
