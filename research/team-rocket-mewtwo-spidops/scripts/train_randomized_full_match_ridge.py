#!/usr/bin/env python3
"""Fit the frozen low-capacity causal overlay on randomized full matches.

Only training and validation paths are accepted. The untouched test split is
intentionally unsupported here and is opened by a separate audit command after
this script freezes an admissible artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import monotonic
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from train_live_truth_causal_q import token_batch
from train_public_leader_v110_bc import Example, collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MODEL_SHA = "cb62f347280fde7f8efbbcecb879c6cdda2bad6c610b05754a089011950e50ff"
FROZEN_DECK_SHA = "256b54736e54ff0062f2310f08acdc5a43696975bd960f87503009375c13b852"
RIDGE_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
CONFIDENCE_GRID = (0.0, 0.5, 1.0)
FIXED_MARGINS = (0.0, 0.02, 0.05, 0.10)
QUANTILES = tuple(value / 10.0 for value in range(1, 10))


@dataclass(frozen=True)
class RawTrial:
    source: str
    game_key: str
    opponent: str
    observation: dict[str, Any]
    candidates: tuple[int, ...]
    assigned_position: int
    propensity: float
    utility: float
    baseline: int


@dataclass(frozen=True)
class EncodedTrial:
    source: str
    game_key: str
    opponent: str
    candidates: tuple[int, ...]
    assigned_position: int
    propensity: float
    utility: float
    global_token: np.ndarray
    option_tokens: np.ndarray
    logits: np.ndarray
    state_value: float


@dataclass(frozen=True)
class Projection:
    global_mean: np.ndarray
    global_components: np.ndarray
    option_mean: np.ndarray
    option_components: np.ndarray
    difference_mean: np.ndarray
    difference_components: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_value_calibrated_v1"
        ),
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "candidates"
            / "scale_neural_resolver_safe_tie_v3"
        ),
    )
    parser.add_argument("--train-data", type=Path, action="append", required=True)
    parser.add_argument("--validation-data", type=Path, action="append", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "team_rocket_recovery"
            / "randomized_full_match_ridge_v1"
        ),
    )
    parser.add_argument("--global-components", type=int, default=32)
    parser.add_argument("--option-components", type=int, default=64)
    parser.add_argument("--difference-components", type=int, default=64)
    parser.add_argument("--ensemble-members", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=731_901)
    parser.add_argument(
        "--protocol",
        default="RANDOMIZED_FULL_MATCH_CAUSAL_RECOVERY_PROTOCOL.md",
    )
    parser.add_argument("--expected-turn", type=int, default=2)
    parser.add_argument("--expected-shortlist-size", type=int, default=0)
    parser.add_argument("--minimum-policy-matches", type=int, default=20)
    parser.add_argument("--minimum-opponent-dr", type=float, default=-0.05)
    parser.add_argument("--test-minimum-policy-matches", type=int, default=30)
    parser.add_argument(
        "--require-positive-ips",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [int(value) for value in path.read_text(encoding="utf-8").replace(",", "\n").split()]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, found {len(deck)}")
    return deck


def audit_manifest(
    path: Path, expected_split: str, expected_protocol: str | None = None
) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "rocket-randomized-full-match-manifest/v1":
        raise ValueError(f"{manifest_path}: unsupported manifest schema")
    if manifest.get("split") != expected_split:
        raise ValueError(f"{manifest_path}: expected split {expected_split}")
    if expected_protocol is not None and manifest.get("protocol") != expected_protocol:
        raise ValueError(f"{manifest_path}: protocol mismatch")
    if manifest.get("output_sha256") != sha256(path):
        raise ValueError(f"{manifest_path}: data hash mismatch")
    if manifest.get("own_model_sha256") != FROZEN_MODEL_SHA:
        raise ValueError(f"{manifest_path}: blueprint model hash mismatch")
    if manifest.get("own_deck_sha256") != FROZEN_DECK_SHA:
        raise ValueError(f"{manifest_path}: Team Rocket deck hash mismatch")
    forbidden = (
        manifest.get("visualizer_truth_used"),
        manifest.get("hidden_truth_serialized"),
        manifest.get("search_branch_used"),
    )
    if any(forbidden) or not manifest.get("genuine_official_engine_complete_matches"):
        raise ValueError(f"{manifest_path}: official/public claim audit failed")
    return manifest


def read_trials(
    paths: Sequence[Path],
    expected_split: str,
    *,
    allow_test: bool = False,
    expected_turn: int = 2,
    expected_shortlist_size: int = 0,
    expected_protocol: str | None = None,
) -> tuple[list[RawTrial], Counter[str]]:
    if expected_split == "test" and not allow_test:
        raise ValueError("training command is forbidden from opening the test split")
    trials: list[RawTrial] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.resolve()
        manifest = audit_manifest(path, expected_split, expected_protocol)
        counts["manifests"] += 1
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["rows"] += 1
            if row.get("schema") != "rocket-randomized-full-match-action/v1":
                raise ValueError(f"{path}:{line_number}: unsupported row schema")
            if row.get("split") != expected_split:
                raise ValueError(f"{path}:{line_number}: split mismatch")
            if row.get("started_first") or row.get("own_seat") != 1 or row.get("first_player") != 0:
                raise ValueError(f"{path}:{line_number}: Team Rocket was not genuinely second")
            if int(row.get("turn") or -1) != int(expected_turn):
                raise ValueError(
                    f"{path}:{line_number}: randomized root was not turn {expected_turn}"
                )
            candidates = tuple(int(value) for value in row.get("candidate_indices") or ())
            baseline = int(row.get("baseline_action_index"))
            assigned_position = int(row.get("assigned_position"))
            assigned_index = int(row.get("assigned_action_index"))
            propensity = float(row.get("propensity"))
            utility = float(row.get("terminal_utility"))
            if not 2 <= len(candidates) <= 4 or candidates[0] != baseline:
                raise ValueError(f"{path}:{line_number}: malformed shortlist")
            if expected_shortlist_size and len(candidates) != expected_shortlist_size:
                raise ValueError(
                    f"{path}:{line_number}: expected shortlist size "
                    f"{expected_shortlist_size}, found {len(candidates)}"
                )
            if len(set(candidates)) != len(candidates):
                raise ValueError(f"{path}:{line_number}: duplicate candidate index")
            if not 0 <= assigned_position < len(candidates):
                raise ValueError(f"{path}:{line_number}: assignment outside shortlist")
            if candidates[assigned_position] != assigned_index:
                raise ValueError(f"{path}:{line_number}: assignment/index mismatch")
            if not math.isclose(propensity, 1.0 / len(candidates), abs_tol=1e-12):
                raise ValueError(f"{path}:{line_number}: invalid propensity")
            if utility not in (-1.0, 0.0, 1.0):
                raise ValueError(f"{path}:{line_number}: invalid terminal utility")
            if not isinstance(row.get("observation"), dict):
                raise ValueError(f"{path}:{line_number}: missing public observation")
            game_key = f"{path.name}:game={int(row['game_id'])}"
            if game_key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate game key")
            seen.add(game_key)
            trials.append(
                RawTrial(
                    source=path.name,
                    game_key=game_key,
                    opponent=str(row.get("opponent") or manifest.get("opponent") or path.name),
                    observation=row["observation"],
                    candidates=candidates,
                    assigned_position=assigned_position,
                    propensity=propensity,
                    utility=utility,
                    baseline=baseline,
                )
            )
    counts["accepted_trials"] = len(trials)
    counts["opponents"] = len({trial.opponent for trial in trials})
    return trials, counts


@torch.no_grad()
def encode_trials(
    raw: Sequence[RawTrial],
    *,
    model: Any,
    deck: Sequence[int],
    library: Mapping[str, Any],
    belief_class: Any,
    features_module: Any,
    attack_safe_logits: Any,
    batch_size: int,
) -> tuple[list[EncodedTrial], Counter[str]]:
    prepared: list[tuple[RawTrial, Example]] = []
    counts: Counter[str] = Counter()
    for trial in raw:
        try:
            belief = belief_class(list(deck), opponent_deck_priors=library)
            feature = features_module.extract(trial.observation, belief=belief, slot_link=True)
            support = int(feature["opt_card"].shape[0])
            if any(index < 0 or index >= support for index in trial.candidates):
                raise ValueError("shortlist exceeds extracted legal support")
            prepared.append((trial, Example(feature, trial.baseline, trial.game_key)))
        except Exception as error:
            counts[f"feature_error:{type(error).__name__}"] += 1
    encoded: list[EncodedTrial] = []
    for start in range(0, len(prepared), batch_size):
        selected = prepared[start : start + batch_size]
        batch = collate([example for _, example in selected], torch.device("cpu"))
        global_tokens, option_tokens = token_batch(model, batch)
        raw_logits, state_values = model.forward_batch(batch)
        for row_index, (trial, example) in enumerate(selected):
            option_count = int(example.feat["opt_card"].shape[0])
            logits = raw_logits[row_index, :option_count].detach().cpu()
            logits = attack_safe_logits(logits, trial.observation["select"]["option"])
            counts["baseline_reproduced"] += int(int(torch.argmax(logits)) == trial.baseline)
            candidate_tensor = torch.tensor(trial.candidates, dtype=torch.long)
            encoded.append(
                EncodedTrial(
                    source=trial.source,
                    game_key=trial.game_key,
                    opponent=trial.opponent,
                    candidates=trial.candidates,
                    assigned_position=trial.assigned_position,
                    propensity=trial.propensity,
                    utility=trial.utility,
                    global_token=global_tokens[row_index].detach().cpu().numpy().astype(np.float32),
                    option_tokens=option_tokens[row_index, :option_count][candidate_tensor]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    logits=logits[candidate_tensor].numpy().astype(np.float32),
                    state_value=float(state_values[row_index]),
                )
            )
    counts["encoded_trials"] = len(encoded)
    if len(encoded) != len(raw):
        raise RuntimeError(f"encoded only {len(encoded)} of {len(raw)} public trials")
    return encoded, counts


def fit_pca(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    centered = values - mean
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    width = min(int(width), right.shape[0], right.shape[1])
    return mean.astype(np.float32), right[:width].astype(np.float32)


def fit_projection(
    trials: Sequence[EncodedTrial],
    global_width: int,
    option_width: int,
    difference_width: int,
) -> Projection:
    globals_ = np.stack([trial.global_token for trial in trials])
    options = np.concatenate([trial.option_tokens for trial in trials], axis=0)
    differences = np.concatenate(
        [trial.option_tokens - trial.option_tokens[0:1] for trial in trials], axis=0
    )
    global_mean, global_components = fit_pca(globals_, global_width)
    option_mean, option_components = fit_pca(options, option_width)
    difference_mean, difference_components = fit_pca(differences, difference_width)
    return Projection(
        global_mean,
        global_components,
        option_mean,
        option_components,
        difference_mean,
        difference_components,
    )


def project(values: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - mean) @ components.T


def raw_candidate_features(trial: EncodedTrial, projection: Projection) -> np.ndarray:
    count = len(trial.candidates)
    global_features = project(
        trial.global_token.reshape(1, -1),
        projection.global_mean,
        projection.global_components,
    )
    global_features = np.repeat(global_features, count, axis=0)
    option_features = project(
        trial.option_tokens,
        projection.option_mean,
        projection.option_components,
    )
    difference_features = project(
        trial.option_tokens - trial.option_tokens[0:1],
        projection.difference_mean,
        projection.difference_components,
    )
    finite_logits = np.nan_to_num(trial.logits.astype(np.float32), neginf=-50.0, posinf=50.0)
    logit_scale = max(1e-4, float(finite_logits.std()))
    normalized_logits = (finite_logits - float(finite_logits.mean())) / logit_scale
    logit_delta = finite_logits - finite_logits[0]
    positions = np.zeros((count, 4), dtype=np.float32)
    positions[np.arange(count), np.arange(count)] = 1.0
    return np.concatenate(
        (
            global_features,
            np.full((count, 1), trial.state_value, dtype=np.float32),
            option_features,
            difference_features,
            normalized_logits.reshape(-1, 1),
            logit_delta.reshape(-1, 1),
            positions,
            np.full((count, 1), count / 4.0, dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)


def build_feature_sets(
    trials: Sequence[EncodedTrial],
    projection: Projection,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    raw = [raw_candidate_features(trial, projection) for trial in trials]
    if mean is None or scale is None:
        assigned = np.stack(
            [features[trial.assigned_position] for trial, features in zip(trials, raw)]
        )
        mean = assigned.mean(axis=0).astype(np.float32)
        scale = assigned.std(axis=0).astype(np.float32)
        scale[scale < 1e-5] = 1.0
    standardized = [((features - mean) / scale).astype(np.float32) for features in raw]
    return standardized, mean, scale


def solve_ridge(x: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.concatenate((np.ones((len(x), 1), dtype=np.float64), x), axis=1)
    gram = design.T @ design
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * float(penalty)
    regularizer[0, 0] = 0.0
    rhs = design.T @ y
    try:
        weights = np.linalg.solve(gram + regularizer, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(gram + regularizer, rhs, rcond=None)[0]
    return weights.astype(np.float32)


def stratified_bootstrap_indices(
    trials: Sequence[EncodedTrial], member: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_007 * member)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, trial in enumerate(trials):
        groups[trial.opponent].append(index)
    result: list[int] = []
    for opponent in sorted(groups):
        values = np.asarray(groups[opponent], dtype=np.int64)
        result.extend(values[rng.integers(0, len(values), size=len(values))].tolist())
    return np.asarray(result, dtype=np.int64)


def fit_ensemble(
    trials: Sequence[EncodedTrial],
    features: Sequence[np.ndarray],
    penalty: float,
    members: int,
    seed: int,
) -> np.ndarray:
    assigned_x = np.stack(
        [matrix[trial.assigned_position] for trial, matrix in zip(trials, features)]
    )
    outcomes = np.asarray([trial.utility for trial in trials], dtype=np.float32)
    weights = []
    for member in range(members):
        indices = stratified_bootstrap_indices(trials, member, seed)
        weights.append(solve_ridge(assigned_x[indices], outcomes[indices], penalty))
    return np.stack(weights)


def predict_ensemble(weights: np.ndarray, features: Sequence[np.ndarray]) -> list[np.ndarray]:
    result = []
    for matrix in features:
        design = np.concatenate((np.ones((len(matrix), 1), dtype=np.float32), matrix), axis=1)
        prediction = weights @ design.T
        result.append(np.clip(prediction, -1.0, 1.0).astype(np.float32))
    return result


def policy_positions(
    predictions: Sequence[np.ndarray], confidence: float, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    best_scores = []
    for values in predictions:
        if values.shape[1] < 2:
            positions.append(0)
            best_scores.append(float("-inf"))
            continue
        advantages = values[:, 1:] - values[:, 0:1]
        mean = advantages.mean(axis=0)
        std = advantages.std(axis=0, ddof=1) if advantages.shape[0] > 1 else np.zeros_like(mean)
        lower = mean - float(confidence) * std
        local = int(np.argmax(lower))
        score = float(lower[local])
        positions.append(local + 1 if score >= float(margin) else 0)
        best_scores.append(score)
    return np.asarray(positions, dtype=np.int64), np.asarray(best_scores, dtype=np.float32)


def bootstrap_weight_matrix(
    trials: Sequence[EncodedTrial], samples: int, seed: int
) -> np.ndarray:
    opponents = sorted({trial.opponent for trial in trials})
    weights = np.zeros((samples, len(trials)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for opponent in opponents:
        indices = np.asarray(
            [index for index, trial in enumerate(trials) if trial.opponent == opponent],
            dtype=np.int64,
        )
        counts = rng.multinomial(
            len(indices),
            np.full(len(indices), 1.0 / len(indices)),
            size=samples,
        ).astype(np.float32)
        weights[:, indices] = counts
    return weights / max(1, len(trials))


def policy_metrics(
    trials: Sequence[EncodedTrial],
    predictions: Sequence[np.ndarray],
    policy: np.ndarray,
    bootstrap_weights: np.ndarray,
) -> dict[str, Any]:
    dr_difference = []
    ips_difference = []
    assigned_matches = 0
    opponent_values: dict[str, list[float]] = defaultdict(list)
    for index, (trial, member_predictions) in enumerate(zip(trials, predictions)):
        q = member_predictions.mean(axis=0)
        action = int(trial.assigned_position)
        selected = int(policy[index])
        propensity = float(trial.propensity)
        outcome = float(trial.utility)
        dr_selected = float(q[selected]) + (outcome - float(q[action])) / propensity * int(action == selected)
        dr_baseline = float(q[0]) + (outcome - float(q[action])) / propensity * int(action == 0)
        ips_selected = outcome / propensity * int(action == selected)
        ips_baseline = outcome / propensity * int(action == 0)
        dr_delta = dr_selected - dr_baseline
        dr_difference.append(dr_delta)
        ips_difference.append(ips_selected - ips_baseline)
        opponent_values[trial.opponent].append(dr_delta)
        assigned_matches += int(action == selected)
    dr = np.asarray(dr_difference, dtype=np.float32)
    ips = np.asarray(ips_difference, dtype=np.float32)
    boot = np.sort(bootstrap_weights @ dr)
    lower = float(np.quantile(boot, 0.025))
    upper = float(np.quantile(boot, 0.975))
    return {
        "games": len(trials),
        "dr_improvement": float(dr.mean()),
        "dr_interval_95": [lower, upper],
        "ips_improvement": float(ips.mean()),
        "interventions": int(np.sum(policy != 0)),
        "intervention_rate": float(np.mean(policy != 0)),
        "assigned_policy_matches": int(assigned_matches),
        "opponent_dr_improvement": {
            opponent: float(np.mean(values)) for opponent, values in sorted(opponent_values.items())
        },
    }


def admissible_validation(
    metrics: Mapping[str, Any],
    *,
    minimum_policy_matches: int,
    minimum_opponent_dr: float,
    require_positive_ips: bool,
) -> bool:
    return bool(
        float(metrics["dr_improvement"]) > 0.0
        and float(metrics["dr_interval_95"][0]) > 0.0
        and (not require_positive_ips or float(metrics["ips_improvement"]) > 0.0)
        and int(metrics["assigned_policy_matches"]) >= minimum_policy_matches
        and 0.05 <= float(metrics["intervention_rate"]) <= 0.60
        and all(
            float(value) >= minimum_opponent_dr
            for value in (metrics.get("opponent_dr_improvement") or {}).values()
        )
    )


def feature_config(projection: Projection) -> dict[str, int]:
    return {
        "global_components": int(projection.global_components.shape[0]),
        "option_components": int(projection.option_components.shape[0]),
        "difference_components": int(projection.difference_components.shape[0]),
    }


def main() -> int:
    args = parse_args()
    if args.ensemble_members < 2 or args.bootstrap_samples <= 0:
        raise ValueError("ensemble-members must be at least 2 and bootstrap-samples positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    candidate = args.candidate.resolve()
    runtime = args.runtime.resolve()
    if sha256(candidate / "model.pt") != FROZEN_MODEL_SHA:
        raise ValueError("candidate model is not the frozen blueprint")
    if sha256(candidate / "deck.csv") != FROZEN_DECK_SHA:
        raise ValueError("candidate deck is not the frozen qualifying Team Rocket deck")
    train_raw, train_read_counts = read_trials(
        args.train_data,
        "train",
        expected_turn=args.expected_turn,
        expected_shortlist_size=args.expected_shortlist_size,
        expected_protocol=args.protocol,
    )
    validation_raw, validation_read_counts = read_trials(
        args.validation_data,
        "validation",
        expected_turn=args.expected_turn,
        expected_shortlist_size=args.expected_shortlist_size,
        expected_protocol=args.protocol,
    )
    if {trial.game_key for trial in train_raw} & {trial.game_key for trial in validation_raw}:
        raise RuntimeError("training and validation game keys overlap")
    if len({trial.opponent for trial in train_raw}) < 2 or len({trial.opponent for trial in validation_raw}) < 2:
        raise RuntimeError("both frozen opponents must appear in training and validation")
    sys.path.insert(0, str(runtime))
    from soldier import features as features_module  # type: ignore
    from soldier.belief import Belief  # type: ignore
    from soldier.net import attack_safe_logits, load_policy  # type: ignore

    deck = read_deck(candidate / "deck.csv")
    library_path = candidate / "deck_library.json"
    library = json.loads(library_path.read_text(encoding="utf-8")) if library_path.exists() else {}
    model = load_policy(candidate / "model.pt").eval()
    started = monotonic()
    train, train_encode_counts = encode_trials(
        train_raw,
        model=model,
        deck=deck,
        library=library,
        belief_class=Belief,
        features_module=features_module,
        attack_safe_logits=attack_safe_logits,
        batch_size=args.batch_size,
    )
    validation, validation_encode_counts = encode_trials(
        validation_raw,
        model=model,
        deck=deck,
        library=library,
        belief_class=Belief,
        features_module=features_module,
        attack_safe_logits=attack_safe_logits,
        batch_size=args.batch_size,
    )
    projection = fit_projection(
        train,
        args.global_components,
        args.option_components,
        args.difference_components,
    )
    train_features, feature_mean, feature_scale = build_feature_sets(train, projection)
    validation_features, _, _ = build_feature_sets(
        validation, projection, feature_mean, feature_scale
    )
    bootstrap_weights = bootstrap_weight_matrix(
        validation, args.bootstrap_samples, args.seed + 500_003
    )
    selection_table: list[dict[str, Any]] = []
    fitted: dict[float, np.ndarray] = {}
    for penalty in RIDGE_GRID:
        weights = fit_ensemble(
            train,
            train_features,
            penalty,
            args.ensemble_members,
            args.seed + 901,
        )
        fitted[penalty] = weights
        predictions = predict_ensemble(weights, validation_features)
        for confidence in CONFIDENCE_GRID:
            _, scores = policy_positions(predictions, confidence, -math.inf)
            positive = scores[np.isfinite(scores) & (scores > 0)]
            margins = set(FIXED_MARGINS)
            if len(positive):
                margins.update(float(np.quantile(positive, q)) for q in QUANTILES)
            for margin in sorted(margins):
                policy, _ = policy_positions(predictions, confidence, margin)
                metrics = policy_metrics(validation, predictions, policy, bootstrap_weights)
                selection_table.append(
                    {
                        "penalty": penalty,
                        "confidence_multiplier": confidence,
                        "margin": margin,
                        "admissible": admissible_validation(
                            metrics,
                            minimum_policy_matches=args.minimum_policy_matches,
                            minimum_opponent_dr=args.minimum_opponent_dr,
                            require_positive_ips=args.require_positive_ips,
                        ),
                        **metrics,
                    }
                )
    eligible = [row for row in selection_table if row["admissible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["dr_interval_95"][0]),
                float(row["dr_improvement"]),
                float(row["ips_improvement"]),
                -float(row["intervention_rate"]),
                -float(row["penalty"]),
            ),
        )
        if eligible
        else None
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "rocket-randomized-full-match-ridge-training/v1",
        "protocol": args.protocol,
        "admissible": selected is not None,
        "selected": selected,
        "selection_table": selection_table,
        "ridge_grid": list(RIDGE_GRID),
        "confidence_grid": list(CONFIDENCE_GRID),
        "fixed_margins": list(FIXED_MARGINS),
        "quantiles": list(QUANTILES),
        "train_read_counts": dict(train_read_counts),
        "validation_read_counts": dict(validation_read_counts),
        "train_encode_counts": dict(train_encode_counts),
        "validation_encode_counts": dict(validation_encode_counts),
        "feature_config": feature_config(projection),
        "feature_width": int(feature_mean.shape[0]),
        "expected_turn": args.expected_turn,
        "expected_shortlist_size": args.expected_shortlist_size,
        "minimum_policy_matches": args.minimum_policy_matches,
        "minimum_opponent_dr": args.minimum_opponent_dr,
        "require_positive_ips": args.require_positive_ips,
        "candidate": str(candidate),
        "candidate_model_sha256": sha256(candidate / "model.pt"),
        "candidate_deck_sha256": sha256(candidate / "deck.csv"),
        "train_data": [str(path.resolve()) for path in args.train_data],
        "validation_data": [str(path.resolve()) for path in args.validation_data],
        "train_data_sha256": {str(path.resolve()): sha256(path.resolve()) for path in args.train_data},
        "validation_data_sha256": {
            str(path.resolve()): sha256(path.resolve()) for path in args.validation_data
        },
        "test_data_opened": False,
        "elapsed_seconds": monotonic() - started,
        "claim_boundary": (
            "training/validation randomized official-match evidence only; "
            "not test admission or arena strength"
        ),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if selected is None:
        print(json.dumps({"admissible": False, "report": str(output / "training_report.json")}, sort_keys=True))
        return 2
    penalty = float(selected["penalty"])
    np.savez_compressed(
        output / "causal_ridge.npz",
        global_mean=projection.global_mean,
        global_components=projection.global_components,
        option_mean=projection.option_mean,
        option_components=projection.option_components,
        difference_mean=projection.difference_mean,
        difference_components=projection.difference_components,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=fitted[penalty],
    )
    manifest = {
        "schema": "rocket-randomized-full-match-ridge-artifact/v1",
        "protocol": args.protocol,
        "artifact": str((output / "causal_ridge.npz").resolve()),
        "artifact_sha256": sha256(output / "causal_ridge.npz"),
        "candidate_model_sha256": FROZEN_MODEL_SHA,
        "candidate_deck_sha256": FROZEN_DECK_SHA,
        "feature_config": feature_config(projection),
        "feature_width": int(feature_mean.shape[0]),
        "ensemble_members": args.ensemble_members,
        "eligible_turn": args.expected_turn,
        "shortlist_size": args.expected_shortlist_size or 4,
        "test_minimum_policy_matches": args.test_minimum_policy_matches,
        "penalty": penalty,
        "confidence_multiplier": float(selected["confidence_multiplier"]),
        "margin": float(selected["margin"]),
        "validation_metrics": selected,
        "test_data_opened": False,
        "training_report_sha256": sha256(output / "training_report.json"),
        "claim_boundary": "frozen from training and validation; untouched test is still sealed",
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "admissible": True,
                "artifact": str(output / "causal_ridge.npz"),
                "selected": selected,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
