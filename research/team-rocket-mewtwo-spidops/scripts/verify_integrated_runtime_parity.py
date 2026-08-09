"""Verify PyTorch training math and dependency-free runtime math agree."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rocketbot.integrated_planner import (  # noqa: E402
    ACTION_FEATURE_NAMES,
    ARCHETYPE_NAMES,
    INTENT_NAMES,
    PLAN_NAMES,
    PREFIX_FEATURE_NAMES,
    RESOURCE_NAMES,
    STRATEGIC_AXES,
    FrozenStrategicPlanner,
    PlannerMemory,
    action_feature_rows,
    planner_feature_items,
    prefix_feature_items,
    search_gate_feature_items,
    token_feature_rows,
)


def _norm(row: Mapping[str, float], spec: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [
            (float(row.get(str(name), 0.0)) - float(mean)) / max(float(scale), 1e-8)
            for name, mean, scale in zip(spec["names"], spec["means"], spec["scales"], strict=True)
        ],
        dtype=torch.float64,
    )


def _linear(value: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    weight = torch.tensor(spec["weight"], dtype=torch.float64)
    bias = torch.tensor(spec["bias"], dtype=torch.float64)
    return weight @ value + bias


def _max_error(left: Sequence[float], right: torch.Tensor) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def verify(model_path: Path, replay_path: Path) -> dict[str, Any]:
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    runtime = FrozenStrategicPlanner(artifact)
    with replay_path.open(encoding="utf-8") as handle:
        observation = next(row["obs"] for row in map(json.loads, handle) if "obs" in row)
    memory = PlannerMemory()
    features = planner_feature_items(observation, memory)
    previous = [0.2] * len(PLAN_NAMES)
    actual = runtime.evaluate(features, previous, allow_safe_resolve=False)
    layers = artifact["layers"]
    calibration = artifact["calibration"]
    hidden = torch.tanh(_linear(torch.cat((_norm(features, artifact["normalizers"]["planner"]), torch.tensor(previous, dtype=torch.float64))), layers["planner_hidden_1"]))
    hidden = torch.tanh(_linear(hidden, layers["planner_hidden_2"]))
    archetype_logits = _linear(hidden, layers["archetype_head"])
    intent_logits = _linear(hidden, layers["intent_head"])
    resource_logits = _linear(hidden, layers["resource_head"])
    archetype = torch.softmax(archetype_logits / calibration["archetype_temperature"], 0)
    intent = torch.softmax(intent_logits / calibration["intent_temperature"], 0)
    resources = torch.sigmoid(resource_logits)
    plan_logits = _linear(hidden, layers["plan_policy_head"])
    if "belief_plan_bridge" in layers:
        bridge_input = torch.cat(
            (torch.softmax(archetype_logits, 0), torch.softmax(intent_logits, 0), resources)
        )
        plan_logits = plan_logits + _linear(bridge_input, layers["belief_plan_bridge"])
    plan = torch.softmax(plan_logits / calibration["plan_temperature"], 0)
    q_values = torch.tanh(_linear(hidden, layers["plan_value_head"]))
    strategic = torch.tanh(_linear(hidden, layers["strategic_head"]))
    meta = torch.softmax(_linear(hidden, layers["meta_head"]), 0)
    aggregate = float((strategic * meta).sum())
    uncertainty = max(0.0, float(_linear(hidden, layers["uncertainty_head"])[0]))
    original_gate = float(torch.sigmoid(_linear(hidden, layers["search_gate_head"])[0]))
    gate_features = search_gate_feature_items(
        features,
        original_gate_probability=original_gate,
        uncertainty=uncertainty,
        archetype_probabilities=tuple(float(value) for value in archetype),
        intent_probabilities=tuple(float(value) for value in intent),
        plan_probabilities=tuple(float(value) for value in plan),
        plan_values=tuple(float(value) for value in q_values),
        strategic_values=tuple(float(value) for value in strategic),
    )
    normalized_gate = _norm(gate_features, artifact["normalizers"]["search_gate"])
    if "search_gate_direct_head" in layers:
        gate = float(torch.sigmoid(_linear(normalized_gate, layers["search_gate_direct_head"])[0]))
    else:
        gate_hidden = torch.tanh(
            _linear(normalized_gate, layers["search_gate_feature_hidden"])
        )
        gate = float(torch.sigmoid(_linear(gate_hidden, layers["search_gate_feature_head"])[0]))
    planner_errors = {
        "archetype": _max_error(actual.archetype_probabilities, archetype),
        "intent": _max_error(actual.intent_probabilities, intent),
        "resources": _max_error(actual.resource_probabilities, resources),
        "plan": _max_error(actual.blueprint_plan_probabilities, plan),
        "plan_values": _max_error(actual.plan_values, q_values),
        "strategic": _max_error(actual.strategic_values, strategic),
        "meta": _max_error(actual.strategic_weights, meta),
        "aggregate": abs(actual.aggregate_value - aggregate),
        "uncertainty": abs(actual.uncertainty - uncertainty),
        "search_gate": abs(actual.search_gate_probability - gate),
    }

    baseline_scores = [float(index * 30 - 20) for index in range(len(observation["select"]["option"]))]
    indices, action_rows = action_feature_rows(
        observation,
        baseline_scores,
        actual.final_plan_probabilities,
        actual.strategic_weights,
        prefix=(),
        remaining=tuple(range(len(baseline_scores))),
        include_stop=False,
    )
    token_rows = token_feature_rows(observation)
    prefix = prefix_feature_items(observation, ())
    actual_scores = runtime.action_scores(features, token_rows, action_rows, prefix, actual)
    state_inputs = torch.cat(
        (
            _norm(features, artifact["normalizers"]["planner"]),
            archetype,
            intent,
            resources,
            plan,
            strategic,
            meta,
            torch.tensor([aggregate, uncertainty], dtype=torch.float64),
        )
    )
    state_hidden = torch.tanh(_linear(state_inputs, layers["action_state_hidden"]))
    token_hidden = torch.stack(
        [torch.tanh(_linear(_norm(row, artifact["normalizers"]["token"]), layers["token_hidden"])) for row in token_rows]
    )
    queries = torch.stack([_linear(row, layers["token_query"]) for row in token_hidden])
    keys = torch.stack([_linear(row, layers["token_key"]) for row in token_hidden])
    values = torch.stack([_linear(row, layers["token_value"]) for row in token_hidden])
    contexts = torch.softmax(queries @ keys.T / math.sqrt(queries.shape[-1]), -1) @ values
    pooled = contexts.mean(0)
    state_query = _linear(state_hidden, layers["token_pool_query"])
    attended = torch.softmax(keys @ state_query / math.sqrt(keys.shape[-1]), 0) @ values
    token_pool = torch.tanh(_linear(torch.cat((pooled, attended)), layers["token_output"]))
    prefix_values = torch.tensor([float(prefix.get(name, 0.0)) for name in PREFIX_FEATURE_NAMES], dtype=torch.float64)
    query = torch.tanh(_linear(torch.cat((state_hidden, token_pool, prefix_values)), layers["action_query"]))
    exploit_query = torch.tanh(_linear(torch.cat((state_hidden, intent)), layers["exploit_query"]))
    action_scores: list[float] = []
    baseline_index = ACTION_FEATURE_NAMES.index("baseline_score")
    plan_indices = [ACTION_FEATURE_NAMES.index(f"plan_alignment_{name}") for name in PLAN_NAMES]
    axis_indices = [ACTION_FEATURE_NAMES.index(f"axis_alignment_{name}") for name in STRATEGIC_AXES]
    for row in action_rows:
        normalized = _norm(row, artifact["normalizers"]["action"])
        action_hidden = torch.tanh(_linear(normalized, layers["action_hidden"]))
        base = float((query * action_hidden).sum() / math.sqrt(len(query)))
        bias = float(_linear(action_hidden, layers["action_bias"])[0])
        exploit = float((exploit_query * action_hidden).sum() / math.sqrt(len(exploit_query)))
        exploit_weight = calibration["exploitation_weight"] if calibration["opponent_exploitation_admitted"] else 0.0
        tactical_prior = calibration["tactical_prior_weight"] * float(normalized[baseline_index])
        plan_alignment = calibration["plan_alignment_weight"] * sum(float(normalized[index]) for index in plan_indices)
        axis_alignment = calibration["axis_alignment_weight"] * sum(float(normalized[index]) for index in axis_indices)
        action_scores.append(
            base + bias + exploit_weight * exploit + tactical_prior + plan_alignment + axis_alignment
        )
    action_error = max(abs(left - right) for left, right in zip(actual_scores, action_scores, strict=True))
    maximum = max((*planner_errors.values(), action_error))
    if maximum > 1e-6:
        raise AssertionError(f"runtime parity exceeded tolerance: {maximum}")
    return {
        "schema_version": 1,
        "model_fingerprint": runtime.fingerprint,
        "replay": replay_path.name,
        "candidate_count": len(indices),
        "planner_max_absolute_errors": planner_errors,
        "action_max_absolute_error": action_error,
        "maximum_absolute_error": maximum,
        "tolerance": 1e-6,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "integrated_controller" / "integrated_planner.json",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=next((PROJECT_ROOT / "work" / "integrated_teacher_replays").glob("*.jsonl")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "integrated_controller" / "runtime_parity_report.json",
    )
    args = parser.parse_args()
    report = verify(args.model, args.replay)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
