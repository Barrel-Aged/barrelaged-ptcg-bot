"""Phase 6AO recovery training for the selective plan-search gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .integrated_planner import (
    PLAN_NAMES,
    SEARCH_GATE_FEATURE_NAMES,
    FrozenStrategicPlanner,
    search_gate_feature_items,
)
from .integrated_training import _canonical, _fingerprint, _linear, _normalizer, _normalize_row


PHASE6AO_SEED = 61_501
PHASE6AO_HIDDEN = 16
PHASE6AO_STEPS = 2_000


@dataclass(frozen=True, slots=True)
class GateRoot:
    root_id: str
    features: Mapping[str, float]
    previous_plan: tuple[float, ...]
    q_values: tuple[float, ...]


def _feature_map(value_root: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in ("public", "private", "history"):
        result.update(
            {str(name): float(value) for name, value in value_root["features"][group]}
        )
    return result


def _blueprint(value_root: Mapping[str, Any]) -> tuple[float, ...]:
    raw = {
        str(name).removeprefix("plan:"): float(value)
        for name, value in value_root["features"]["blueprint_probabilities"]
    }
    return tuple(raw[name] for name in PLAN_NAMES)


def _q_values(value_roots: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    returns: dict[str, list[float]] = {name: [] for name in PLAN_NAMES}
    for value_root in value_roots:
        for surface in value_root["surfaces"]:
            for action in surface["action_values"]:
                name = str(action["action_id"]).removeprefix("plan:")
                returns[name].extend(float(value) for value in action["terminal_returns"])
    return tuple(statistics.fmean(returns[name]) for name in PLAN_NAMES)


def load_development(path: Path) -> tuple[list[GateRoot], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        GateRoot(
            root_id=str(root["root_id"]),
            features=_feature_map(root),
            previous_plan=_blueprint(root),
            q_values=_q_values((root,)),
        )
        for root in raw["roots"]
    ]
    return rows, str(raw["fingerprint"])


def load_panel(path: Path) -> tuple[list[GateRoot], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[GateRoot] = []
    for root in raw["roots"]:
        value_roots = tuple(batch["value_root"] for batch in root["batches"])
        rows.append(
            GateRoot(
                root_id=str(root["root_id"]),
                features=_feature_map(value_roots[0]),
                previous_plan=_blueprint(value_roots[0]),
                q_values=_q_values(value_roots),
            )
        )
    return rows, str(raw["fingerprint"])


def gate_rows(
    roots: Sequence[GateRoot], planner: FrozenStrategicPlanner
) -> tuple[list[dict[str, float]], list[float]]:
    features: list[dict[str, float]] = []
    benefits: list[float] = []
    for root in roots:
        evaluation = planner.evaluate(root.features, root.previous_plan, allow_safe_resolve=False)
        baseline = sum(
            probability * value
            for probability, value in zip(
                evaluation.blueprint_plan_probabilities, root.q_values, strict=True
            )
        )
        benefit = max(root.q_values) - baseline
        features.append(
            search_gate_feature_items(
                root.features,
                original_gate_probability=evaluation.search_gate_probability,
                uncertainty=evaluation.uncertainty,
                archetype_probabilities=evaluation.archetype_probabilities,
                intent_probabilities=evaluation.intent_probabilities,
                plan_probabilities=evaluation.blueprint_plan_probabilities,
                plan_values=evaluation.plan_values,
                strategic_values=evaluation.strategic_values,
            )
        )
        benefits.append(benefit)
    return features, benefits


class SearchGateNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(len(SEARCH_GATE_FEATURE_NAMES), PHASE6AO_HIDDEN)
        self.head = nn.Linear(PHASE6AO_HIDDEN, 1)

    def forward(self, values: Tensor) -> Tensor:
        return self.head(torch.tanh(self.hidden(values))).squeeze(-1)


def _score(model: SearchGateNetwork, rows: Sequence[Mapping[str, float]], normalizer: Mapping[str, Any]) -> list[float]:
    values = torch.tensor([_normalize_row(row, normalizer) for row in rows], dtype=torch.float32)
    with torch.no_grad():
        return [float(value) for value in torch.sigmoid(model(values))]


def _threshold(
    probabilities: Sequence[float], benefits: Sequence[float]
) -> tuple[float, dict[str, float | int]]:
    candidates: list[tuple[float, float, float]] = []
    for threshold in sorted(set(float(value) for value in probabilities)):
        gated = [benefit for probability, benefit in zip(probabilities, benefits, strict=True) if probability >= threshold]
        routine = [benefit for probability, benefit in zip(probabilities, benefits, strict=True) if probability < threshold]
        fraction = len(gated) / len(benefits)
        if gated and routine and 0.20 <= fraction <= 0.60:
            candidates.append(
                (
                    statistics.fmean(gated) - statistics.fmean(routine),
                    threshold,
                    fraction,
                )
            )
    if not candidates:
        raise RuntimeError("no eligible Phase 6AO calibration threshold")
    separation, threshold, fraction = max(candidates, key=lambda item: (item[0], item[1]))
    return threshold, {
        "count": len(benefits),
        "trigger_fraction": fraction,
        "mean_available_improvement": statistics.fmean(benefits),
        "gated_mean_available_improvement": statistics.fmean(
            benefit for probability, benefit in zip(probabilities, benefits, strict=True) if probability >= threshold
        ),
        "routine_mean_available_improvement": statistics.fmean(
            benefit for probability, benefit in zip(probabilities, benefits, strict=True) if probability < threshold
        ),
        "separation": separation,
    }


def _evaluate(
    probabilities: Sequence[float], benefits: Sequence[float], threshold: float
) -> dict[str, float | int | bool]:
    gated = [benefit for probability, benefit in zip(probabilities, benefits, strict=True) if probability >= threshold]
    routine = [benefit for probability, benefit in zip(probabilities, benefits, strict=True) if probability < threshold]
    overall = statistics.fmean(benefits)
    gated_mean = statistics.fmean(gated) if gated else 0.0
    routine_mean = statistics.fmean(routine) if routine else 0.0
    fraction = len(gated) / len(benefits)
    passed = bool(
        gated
        and routine
        and 0.15 <= fraction <= 0.70
        and gated_mean - routine_mean >= 0.02
        and gated_mean > overall
    )
    return {
        "count": len(benefits),
        "gated_count": len(gated),
        "routine_count": len(routine),
        "trigger_fraction": fraction,
        "mean_available_improvement": overall,
        "gated_mean_available_improvement": gated_mean,
        "routine_mean_available_improvement": routine_mean,
        "separation": gated_mean - routine_mean,
        "passed": passed,
    }


def train_phase6ao_gate(
    *,
    base_model_path: Path,
    development_path: Path,
    calibration_path: Path,
    validation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    base_artifact = json.loads(base_model_path.read_text(encoding="utf-8"))
    planner = FrozenStrategicPlanner(base_artifact)
    development, development_fingerprint = load_development(development_path)
    calibration, calibration_fingerprint = load_panel(calibration_path)
    validation, validation_fingerprint = load_panel(validation_path)
    root_sets = [set(row.root_id for row in rows) for rows in (development, calibration, validation)]
    if any(root_sets[left].intersection(root_sets[right]) for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Phase 6AO root partitions overlap")
    development_rows, development_benefits = gate_rows(development, planner)
    calibration_rows, calibration_benefits = gate_rows(calibration, planner)
    validation_rows, validation_benefits = gate_rows(validation, planner)
    normalizer = _normalizer(development_rows, SEARCH_GATE_FEATURE_NAMES)
    x = torch.tensor([_normalize_row(row, normalizer) for row in development_rows], dtype=torch.float32)
    benefits = torch.tensor(development_benefits, dtype=torch.float32)
    target = (benefits / max(float(benefits.max()), 1e-6)).clamp(0.0, 1.0)
    torch.manual_seed(PHASE6AO_SEED)
    model = SearchGateNetwork()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=2e-4)
    for _ in range(PHASE6AO_STEPS):
        optimizer.zero_grad()
        logits = model(x)
        point_loss = F.binary_cross_entropy_with_logits(logits, target)
        differences = benefits[:, None] - benefits[None, :]
        eligible = differences.abs() >= 0.02
        signs = differences.sign()
        logit_differences = logits[:, None] - logits[None, :]
        ranking_loss = F.softplus(-signs[eligible] * logit_differences[eligible]).mean()
        loss = point_loss + 0.8 * ranking_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    calibration_probabilities = _score(model, calibration_rows, normalizer)
    threshold, calibration_metrics = _threshold(calibration_probabilities, calibration_benefits)
    validation_probabilities = _score(model, validation_rows, normalizer)
    validation_metrics = _evaluate(validation_probabilities, validation_benefits, threshold)
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "PHASE6AO_SELECTIVE_SEARCH_GATE_PROTOCOL.md",
        "seed": PHASE6AO_SEED,
        "development_steps": PHASE6AO_STEPS,
        "development_count": len(development),
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "root_partitions_disjoint": True,
        "source_fingerprints": {
            "development": development_fingerprint,
            "calibration": calibration_fingerprint,
            "validation": validation_fingerprint,
        },
        "threshold": threshold,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "admitted": bool(validation_metrics["passed"]),
        "claim_boundary": "fixed-panel selective concentration; not a general search theorem",
    }
    report["fingerprint"] = _fingerprint(report)
    candidate = dict(base_artifact)
    candidate.pop("fingerprint", None)
    candidate["layers"] = dict(candidate["layers"])
    candidate["layers"]["search_gate_feature_hidden"] = _linear(model.hidden)
    candidate["layers"]["search_gate_feature_head"] = _linear(model.head)
    candidate["normalizers"] = dict(candidate["normalizers"])
    candidate["normalizers"]["search_gate"] = normalizer
    candidate["calibration"] = dict(candidate["calibration"])
    candidate["calibration"]["search_gate_threshold"] = threshold
    candidate["calibration"]["selective_search_gate_admitted"] = bool(report["admitted"])
    candidate["evidence"] = dict(candidate["evidence"])
    candidate["evidence"]["selective_search_gate"] = report
    candidate["fingerprint"] = _fingerprint(candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    FrozenStrategicPlanner(json.loads(output_path.read_text(encoding="utf-8")))
    report["candidate_model_fingerprint"] = candidate["fingerprint"]
    report["candidate_model_sha256"] = sha256(output_path.read_bytes()).hexdigest()
    report_path = output_path.with_name("phase6ao_selective_search_gate_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not report["admitted"]:
        raise RuntimeError("Phase 6AO selective-search admission failed")
    return report


__all__ = ["GateRoot", "SearchGateNetwork", "train_phase6ao_gate"]
