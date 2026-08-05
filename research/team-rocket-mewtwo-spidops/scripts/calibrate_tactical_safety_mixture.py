"""Calibrate and admit the Phase 6AQ low-confidence tactical provider."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.competition import (  # noqa: E402
    KAGGLE_WATCHTOWER_SUBSTITUTION,
    SelectionContract,
    resolve_official_deck,
)
from rocketbot.integrated_planner import (  # noqa: E402
    FrozenStrategicPlanner,
    IntegratedBeliefPlannerAgent,
)
from rocketbot.integrated_training import _file_split, _fingerprint  # noqa: E402


ARCHIVED_REPLAYS = Path(
    r"C:\Users\jhos4\PokemonDocs\_local_archive_2026-07-26\evidence"
    r"\rocket_spidops_mewtwo_20260719\arena_v3_refresh"
)


@dataclass(frozen=True, slots=True)
class Decision:
    split: str
    confidence: float
    neural_correct: bool
    nontrivial: bool


def _target_seat(meta: dict[str, Any], target_name: str | None) -> int:
    if target_name is None:
        return int(meta["target_seat"])
    return list(meta["players"]).index(target_name)


def load_decisions(model: Path) -> tuple[list[Decision], dict[str, Any]]:
    deck = resolve_official_deck(
        substitutions=KAGGLE_WATCHTOWER_SUBSTITUTION,
        fail_on_unresolved=True,
    )
    decisions: list[Decision] = []
    invalid_teacher = 0
    file_counts: dict[str, int] = {}
    for source, directory, target_name in (
        ("compatibility_teacher", PROJECT_ROOT / "work" / "integrated_teacher_replays", None),
        ("archived_prov3", ARCHIVED_REPLAYS, "HelbrosRocketSpidopsProV3"),
    ):
        for path in sorted(directory.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            meta = rows[0].get("meta", {})
            match_id = str(meta.get("match_id", path.stem))
            split = _file_split(source, match_id)
            if split not in {"validation", "test"}:
                continue
            try:
                target_seat = _target_seat(meta, target_name)
            except ValueError:
                continue
            file_counts[f"{source}:{split}"] = file_counts.get(f"{source}:{split}", 0) + 1
            agent = IntegratedBeliefPlannerAgent(deck, model, telemetry_limit=4_096)
            for row in rows[1:]:
                if "obs" not in row or int(row.get("p", -1)) != target_seat:
                    continue
                observation = row["obs"]
                contract = SelectionContract.from_observation(observation)
                try:
                    teacher = list(contract.validate(row.get("action", ())))
                except Exception:
                    invalid_teacher += 1
                    continue
                before = len(agent.memory.telemetry)
                neural = agent.choose(observation)
                if len(agent.memory.telemetry) > before:
                    telemetry = agent.memory.telemetry[-1]
                    neural = list(telemetry["neural_selection"])
                    confidence = float(telemetry["tactical_policy_minimum_prefix_confidence"])
                else:
                    confidence = 0.0
                decisions.append(
                    Decision(
                        split=split,
                        confidence=confidence,
                        neural_correct=neural == teacher,
                        nontrivial=contract.option_count > 1,
                    )
                )
    return decisions, {
        "file_counts": file_counts,
        "invalid_teacher_sequences": invalid_teacher,
        "decision_counts": {
            split: sum(decision.split == split for decision in decisions)
            for split in ("validation", "test")
        },
    }


def metrics(rows: list[Decision], threshold: float) -> dict[str, Any]:
    correct = [decision.neural_correct or decision.confidence < threshold for decision in rows]
    nontrivial = [decision for decision in rows if decision.nontrivial]
    nontrivial_correct = [
        decision.neural_correct or decision.confidence < threshold for decision in nontrivial
    ]
    deferred = [decision.confidence < threshold for decision in rows]
    nontrivial_deferred = [decision.confidence < threshold for decision in nontrivial]
    return {
        "count": len(rows),
        "exact_sequence_accuracy": sum(correct) / max(len(correct), 1),
        "nontrivial_count": len(nontrivial),
        "nontrivial_exact_sequence_accuracy": sum(nontrivial_correct) / max(len(nontrivial_correct), 1),
        "deference_rate": sum(deferred) / max(len(deferred), 1),
        "nontrivial_neural_control_rate": 1.0 - sum(nontrivial_deferred) / max(len(nontrivial_deferred), 1),
        "neural_only_accuracy": sum(decision.neural_correct for decision in rows) / max(len(rows), 1),
    }


def main() -> None:
    model = PROJECT_ROOT / "artifacts" / "integrated_controller" / "integrated_planner.json"
    decisions, provenance = load_decisions(model)
    validation = [decision for decision in decisions if decision.split == "validation"]
    test = [decision for decision in decisions if decision.split == "test"]
    candidates = sorted({0.0, *(decision.confidence + 1e-12 for decision in validation)})
    eligible: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        result = metrics(validation, threshold)
        if (
            result["exact_sequence_accuracy"] >= 0.95
            and result["nontrivial_neural_control_rate"] >= 0.50
        ):
            eligible.append((threshold, result))
    if not eligible:
        raise RuntimeError("no Phase 6AQ threshold satisfies validation gates")
    threshold, validation_metrics = min(eligible, key=lambda item: item[0])
    test_metrics = metrics(test, threshold)
    admitted = bool(
        provenance["invalid_teacher_sequences"] == 0
        and test_metrics["exact_sequence_accuracy"] >= 0.93
        and test_metrics["nontrivial_exact_sequence_accuracy"] >= 0.90
        and test_metrics["nontrivial_neural_control_rate"] >= 0.50
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "PHASE6AQ_TACTICAL_SAFETY_MIXTURE_PROTOCOL.md",
        "threshold": threshold,
        "validation": validation_metrics,
        "test": test_metrics,
        "provenance": provenance,
        "admitted": admitted,
        "provider_boundary": "low-confidence tactical sequence provider; not strategic blueprint",
    }
    report["fingerprint"] = _fingerprint(report)
    artifact = json.loads(model.read_text(encoding="utf-8"))
    artifact.pop("fingerprint", None)
    artifact["calibration"] = dict(artifact["calibration"])
    artifact["calibration"]["tactical_teacher_deference_threshold"] = threshold
    artifact["calibration"]["tactical_teacher_deference_admitted"] = admitted
    artifact["evidence"] = dict(artifact["evidence"])
    artifact["evidence"]["tactical_safety_mixture"] = report
    artifact["fingerprint"] = _fingerprint(artifact)
    model.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    FrozenStrategicPlanner(json.loads(model.read_text(encoding="utf-8")))
    report["model_fingerprint"] = artifact["fingerprint"]
    report["model_sha256"] = sha256(model.read_bytes()).hexdigest()
    report_path = model.with_name("tactical_safety_mixture_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not admitted:
        raise RuntimeError("Phase 6AQ tactical safety mixture failed test admission")


if __name__ == "__main__":
    main()
