"""Phase 6AP low-capacity monotone selective-search gate."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from .integrated_gate_training import (
    _evaluate,
    _threshold,
    gate_rows,
    load_development,
    load_panel,
)
from .integrated_planner import SEARCH_GATE_FEATURE_NAMES, FrozenStrategicPlanner
from .integrated_training import _fingerprint, _normalizer, _normalize_row


PHASE6AP_SEED = 61_502
PHASE6AP_STEPS = 3_000
PHASE6AP_FEATURES = (
    "uncertainty",
    "archetype_entropy",
    "strategic_disagreement",
    "turn",
)


def _probabilities(
    rows: Sequence[Mapping[str, float]],
    normalizer: Mapping[str, Any],
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> list[float]:
    values = torch.tensor(
        [[_normalize_row(row, normalizer)[SEARCH_GATE_FEATURE_NAMES.index(name)] for name in PHASE6AP_FEATURES] for row in rows],
        dtype=torch.float32,
    )
    with torch.no_grad():
        return [float(value) for value in torch.sigmoid(values @ weights + bias)]


def train_phase6ap_gate(
    *,
    base_model_path: Path,
    development_path: Path,
    additional_development_path: Path,
    selection_path: Path,
    validation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    base = json.loads(base_model_path.read_text(encoding="utf-8"))
    planner = FrozenStrategicPlanner(base)
    development_a, fingerprint_a = load_development(development_path)
    development_b, fingerprint_b = load_panel(additional_development_path)
    selection, fingerprint_selection = load_panel(selection_path)
    validation, fingerprint_validation = load_panel(validation_path)
    partitions = (development_a, development_b, selection, validation)
    root_sets = [set(row.root_id for row in rows) for rows in partitions]
    if any(
        root_sets[left].intersection(root_sets[right])
        for left in range(len(root_sets))
        for right in range(left + 1, len(root_sets))
    ):
        raise RuntimeError("Phase 6AP root partitions overlap")
    development = list(development_a) + list(development_b)
    development_rows, development_benefits = gate_rows(development, planner)
    selection_rows, selection_benefits = gate_rows(selection, planner)
    validation_rows, validation_benefits = gate_rows(validation, planner)
    normalizer = _normalizer(development_rows, SEARCH_GATE_FEATURE_NAMES)
    x = torch.tensor(
        [[_normalize_row(row, normalizer)[SEARCH_GATE_FEATURE_NAMES.index(name)] for name in PHASE6AP_FEATURES] for row in development_rows],
        dtype=torch.float32,
    )
    benefits = torch.tensor(development_benefits, dtype=torch.float32)
    targets = (benefits / max(float(benefits.max()), 1e-6)).clamp(0.0, 1.0)
    torch.manual_seed(PHASE6AP_SEED)
    raw_weights = torch.nn.Parameter(torch.zeros(len(PHASE6AP_FEATURES)))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam((raw_weights, bias), lr=0.02)
    signs = torch.tensor((1.0, 1.0, 1.0, -1.0))
    for _ in range(PHASE6AP_STEPS):
        optimizer.zero_grad()
        weights = F.softplus(raw_weights) * signs
        logits = x @ weights + bias
        point_loss = F.binary_cross_entropy_with_logits(logits, targets)
        differences = benefits[:, None] - benefits[None, :]
        eligible = differences.abs() >= 0.02
        logit_differences = logits[:, None] - logits[None, :]
        ranking_loss = F.softplus(
            -differences.sign()[eligible] * logit_differences[eligible]
        ).mean()
        loss = point_loss + 0.5 * ranking_loss + 0.02 * (weights * weights).sum()
        loss.backward()
        optimizer.step()
    weights = (F.softplus(raw_weights) * signs).detach()
    bias_value = bias.detach()
    selection_probabilities = _probabilities(
        selection_rows, normalizer, weights, bias_value
    )
    threshold, selection_metrics = _threshold(
        selection_probabilities, selection_benefits
    )
    validation_probabilities = _probabilities(
        validation_rows, normalizer, weights, bias_value
    )
    validation_metrics = _evaluate(
        validation_probabilities, validation_benefits, threshold
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "PHASE6AP_MONOTONE_SELECTIVE_GATE_PROTOCOL.md",
        "seed": PHASE6AP_SEED,
        "steps": PHASE6AP_STEPS,
        "feature_names": list(PHASE6AP_FEATURES),
        "feature_weights": [float(value) for value in weights],
        "bias": float(bias_value),
        "development_count": len(development),
        "selection_count": len(selection),
        "validation_count": len(validation),
        "root_partitions_disjoint": True,
        "source_fingerprints": {
            "development_phase6ah": fingerprint_a,
            "development_phase6al_calibration": fingerprint_b,
            "selection_phase6al_validation": fingerprint_selection,
            "validation_phase6am": fingerprint_validation,
        },
        "threshold": threshold,
        "selection_metrics": selection_metrics,
        "validation_metrics": validation_metrics,
        "admitted": bool(validation_metrics["passed"]),
        "claim_boundary": "fresh fixed-panel selective concentration; not a general theorem",
    }
    report["fingerprint"] = _fingerprint(report)
    candidate = dict(base)
    candidate.pop("fingerprint", None)
    candidate["layers"] = dict(candidate["layers"])
    direct_weights = [0.0 for _ in SEARCH_GATE_FEATURE_NAMES]
    for name, value in zip(PHASE6AP_FEATURES, weights, strict=True):
        direct_weights[SEARCH_GATE_FEATURE_NAMES.index(name)] = float(value)
    candidate["layers"].pop("search_gate_feature_hidden", None)
    candidate["layers"].pop("search_gate_feature_head", None)
    candidate["layers"]["search_gate_direct_head"] = {
        "weight": [direct_weights],
        "bias": [float(bias_value)],
    }
    candidate["normalizers"] = dict(candidate["normalizers"])
    candidate["normalizers"]["search_gate"] = normalizer
    candidate["calibration"] = dict(candidate["calibration"])
    candidate["calibration"]["search_gate_threshold"] = threshold
    candidate["calibration"]["selective_search_gate_admitted"] = bool(report["admitted"])
    candidate["evidence"] = dict(candidate["evidence"])
    candidate["evidence"]["selective_search_gate"] = report
    candidate["fingerprint"] = _fingerprint(candidate)
    output_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    FrozenStrategicPlanner(json.loads(output_path.read_text(encoding="utf-8")))
    report["candidate_model_fingerprint"] = candidate["fingerprint"]
    report["candidate_model_sha256"] = sha256(output_path.read_bytes()).hexdigest()
    report_path = output_path.with_name("phase6ap_monotone_selective_gate_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not report["admitted"]:
        raise RuntimeError("Phase 6AP monotone selective-search admission failed")
    return report


__all__ = ["train_phase6ap_gate"]
