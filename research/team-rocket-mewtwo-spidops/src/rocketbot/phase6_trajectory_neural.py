"""Deterministic Phase 6I logged-trajectory neural-bootstrap evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .official_trajectory import OfficialTrajectoryDataset
from .trajectory_neural import (
    DEFAULT_TRAJECTORY_NEURAL_SEED,
    TrajectoryBootstrapArtifact,
    TrajectoryBootstrapTrainingRun,
    TrajectorySplitMetrics,
    restore_logged_trajectory_bootstrap,
    train_logged_trajectory_bootstrap,
)


PHASE6_TRAJECTORY_NEURAL_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class TrajectoryBootstrapRoundtripEvidence:
    artifact_fingerprint: str
    artifact_serialization_exact: bool
    maximum_action_probability_delta: float
    maximum_outcome_value_delta: float
    maximum_range_probability_delta: float
    maximum_next_public_feature_delta: float
    legal_support_exact: bool

    def __post_init__(self) -> None:
        _validate_digest(self.artifact_fingerprint, "artifact_fingerprint")
        for field_name in (
            "maximum_action_probability_delta",
            "maximum_outcome_value_delta",
            "maximum_range_probability_delta",
            "maximum_next_public_feature_delta",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Phase6TrajectoryNeuralEvidence:
    benchmark_id: str
    status: str
    seed: int
    trajectory_dataset_fingerprint: str
    source_public_replay_fingerprint: str
    artifact_fingerprint: str
    train_steps: int
    calibration_steps: int
    validation_steps: int
    train_matches: int
    calibration_matches: int
    validation_matches: int
    split_match_overlap_count: int
    public_feature_width: int
    private_feature_width: int
    action_feature_width: int
    candidate_count: int
    initial_training_loss: float
    final_training_loss: float
    gradient_parameter_count: int
    required_head_gradients_present: bool
    normalizers_fit_train_only: bool
    policy_temperature: float
    range_temperature: float
    metrics: tuple[TrajectorySplitMetrics, ...]
    roundtrip: TrajectoryBootstrapRoundtripEvidence
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6I benchmark identity/status is required")
        for field_name in (
            "trajectory_dataset_fingerprint",
            "source_public_replay_fingerprint",
            "artifact_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        for field_name in (
            "train_steps",
            "calibration_steps",
            "validation_steps",
            "train_matches",
            "calibration_matches",
            "validation_matches",
            "public_feature_width",
            "private_feature_width",
            "action_feature_width",
            "candidate_count",
            "gradient_parameter_count",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.split_match_overlap_count < 0:
            raise ValueError("split overlap count cannot be negative")
        if not math.isfinite(self.initial_training_loss) or not math.isfinite(
            self.final_training_loss
        ):
            raise ValueError("Phase 6I training losses must be finite")
        if self.final_training_loss >= self.initial_training_loss:
            raise ValueError("Phase 6I training loss must decrease")
        if tuple(metric.split for metric in self.metrics) != (
            "train",
            "calibration",
            "validation",
        ):
            raise ValueError("Phase 6I metrics must cover ordered splits")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6I claim names must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_TRAJECTORY_NEURAL_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "seed": self.seed,
            "trajectory_dataset_fingerprint": self.trajectory_dataset_fingerprint,
            "source_public_replay_fingerprint": self.source_public_replay_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
            "split_steps": {
                "train": self.train_steps,
                "calibration": self.calibration_steps,
                "validation": self.validation_steps,
            },
            "split_matches": {
                "train": self.train_matches,
                "calibration": self.calibration_matches,
                "validation": self.validation_matches,
            },
            "split_match_overlap_count": self.split_match_overlap_count,
            "feature_widths": {
                "public": self.public_feature_width,
                "private": self.private_feature_width,
                "action": self.action_feature_width,
            },
            "candidate_count": self.candidate_count,
            "training": {
                "initial_loss": self.initial_training_loss,
                "final_loss": self.final_training_loss,
                "gradient_parameter_count": self.gradient_parameter_count,
                "required_head_gradients_present": self.required_head_gradients_present,
                "normalizers_fit_train_only": self.normalizers_fit_train_only,
                "policy_temperature": self.policy_temperature,
                "range_temperature": self.range_temperature,
            },
            "metrics": {
                metric.split: asdict(metric) for metric in self.metrics
            },
            "roundtrip": asdict(self.roundtrip),
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        metric_rows = "\n".join(
            (
                f"| {metric.split} | {metric.policy_nll:.6f} "
                f"({metric.uniform_policy_nll:.6f}) | "
                f"{metric.policy_top1_accuracy:.3%} | "
                f"{metric.outcome_mse:.6f} ({metric.constant_outcome_mse:.6f}) | "
                f"{metric.range_nll:.6f} ({metric.uniform_range_nll:.6f}) | "
                f"{metric.range_top1_accuracy:.3%} | "
                f"{metric.transition_mse:.6f} ({metric.identity_transition_mse:.6f}) |"
            )
            for metric in self.metrics
        )
        claim_rows = "\n".join(
            f"| {name} | {value} |" for name, value in self.claims
        )
        return f"""# Phase 6I logged-trajectory neural bootstrap

Benchmark: `{self.benchmark_id}`  
Evidence fingerprint: `{self.fingerprint}`  
Artifact fingerprint: `{self.artifact_fingerprint}`  
Status: **{self.status}**

This is deterministic multi-task pretraining on the non-branchable Phase 6H
corpus. The action head imitates the preserved logged heuristic behavior. It is
not the final strategy, an equilibrium blueprint, counterfactual-value
training, safe re-solving, or official-engine execution.

| Split | Policy NLL (uniform) | Policy top-1 | Outcome MSE (constant) | Range NLL (uniform) | Range top-1 | Next-public MSE (identity) |
|---|---:|---:|---:|---:|---:|---:|
{metric_rows}

The 12 episodes / 6 matches form a small engineering corpus. These diagnostics
were inspected while developing the scaffold, so none is a pristine final test
or a performance claim. They validate wiring, gradients, split isolation, and
checkpoint restoration only.

## Integrity

| Check | Value |
|---|---:|
| Train / calibration / validation steps | {self.train_steps} / {self.calibration_steps} / {self.validation_steps} |
| Train / calibration / validation matches | {self.train_matches} / {self.calibration_matches} / {self.validation_matches} |
| Split match overlaps | {self.split_match_overlap_count} |
| Public / private / action widths | {self.public_feature_width} / {self.private_feature_width} / {self.action_feature_width} |
| Initial / final train loss | {self.initial_training_loss:.6f} / {self.final_training_loss:.6f} |
| Non-zero-gradient parameters | {self.gradient_parameter_count} |
| Policy / range calibration temperature | {self.policy_temperature:.2f} / {self.range_temperature:.2f} |
| Exact serialized restore | {str(self.roundtrip.artifact_serialization_exact).lower()} |
| Exact legal support after restore | {str(self.roundtrip.legal_support_exact).lower()} |
| Maximum restored prediction delta | {max(self.roundtrip.maximum_action_probability_delta, self.roundtrip.maximum_outcome_value_delta, self.roundtrip.maximum_range_probability_delta, self.roundtrip.maximum_next_public_feature_delta):.3e} |

## Claim boundary

| Claim | Status |
|---|---|
{claim_rows}
"""


@dataclass(slots=True)
class Phase6TrajectoryNeuralArtifacts:
    evidence: Phase6TrajectoryNeuralEvidence
    training_run: TrajectoryBootstrapTrainingRun

    @property
    def artifact(self) -> TrajectoryBootstrapArtifact:
        return self.training_run.artifact


def _load_trajectory_dataset(path: Path) -> OfficialTrajectoryDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    dataset = OfficialTrajectoryDataset.from_dict(raw)
    expected = raw.get("fingerprint")
    if expected is not None and str(expected) != dataset.fingerprint:
        raise ValueError("trajectory dataset artifact fingerprint is stale or corrupt")
    return dataset


def _split_overlap(artifact: TrajectoryBootstrapArtifact) -> int:
    train = set(artifact.train_match_ids)
    calibration = set(artifact.calibration_match_ids)
    validation = set(artifact.validation_match_ids)
    return len(train & calibration) + len(train & validation) + len(calibration & validation)


def _roundtrip(
    dataset: OfficialTrajectoryDataset,
    run: TrajectoryBootstrapTrainingRun,
) -> TrajectoryBootstrapRoundtripEvidence:
    encoded = run.artifact.to_json()
    restored_artifact = TrajectoryBootstrapArtifact.from_dict(json.loads(encoded))
    restored = restore_logged_trajectory_bootstrap(restored_artifact)
    maxima = {
        "action": 0.0,
        "outcome": 0.0,
        "range": 0.0,
        "next_public": 0.0,
    }
    legal_support_exact = True
    # Probe every split and both perspectives without turning the artifact into
    # an official execution path.
    for step in dataset.steps:
        original = run.provider.predict_step(step)
        roundtripped = restored.predict_step(step)
        legal_ids = tuple(action.action_id for action in step.information_set.legal_actions)
        original_ids = tuple(name for name, _ in original.action_probabilities)
        restored_ids = tuple(name for name, _ in roundtripped.action_probabilities)
        legal_support_exact &= original_ids == legal_ids == restored_ids
        maxima["action"] = max(
            maxima["action"],
            *(abs(left[1] - right[1]) for left, right in zip(
                original.action_probabilities,
                roundtripped.action_probabilities,
                strict=True,
            )),
        )
        maxima["outcome"] = max(
            maxima["outcome"],
            abs(original.outcome_value - roundtripped.outcome_value),
        )
        maxima["range"] = max(
            maxima["range"],
            *(abs(left[1] - right[1]) for left, right in zip(
                original.opponent_range,
                roundtripped.opponent_range,
                strict=True,
            )),
        )
        maxima["next_public"] = max(
            maxima["next_public"],
            *(abs(left[1] - right[1]) for left, right in zip(
                original.next_public_features,
                roundtripped.next_public_features,
                strict=True,
            )),
        )
    return TrajectoryBootstrapRoundtripEvidence(
        artifact_fingerprint=restored_artifact.fingerprint,
        artifact_serialization_exact=restored_artifact.fingerprint == run.artifact.fingerprint,
        maximum_action_probability_delta=maxima["action"],
        maximum_outcome_value_delta=maxima["outcome"],
        maximum_range_probability_delta=maxima["range"],
        maximum_next_public_feature_delta=maxima["next_public"],
        legal_support_exact=legal_support_exact,
    )


def run_phase6_trajectory_neural_bootstrap(
    *,
    project_root: str | Path | None = None,
    seed: int = DEFAULT_TRAJECTORY_NEURAL_SEED,
    epochs: int = 10,
    hidden_dim: int = 32,
) -> Phase6TrajectoryNeuralArtifacts:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    dataset = _load_trajectory_dataset(
        root
        / "artifacts"
        / "phase6_official_trajectory"
        / "official_trajectory_dataset.json"
    )
    run = train_logged_trajectory_bootstrap(
        dataset,
        seed=seed,
        epochs=epochs,
        hidden_dim=hidden_dim,
    )
    artifact = run.artifact
    roundtrip = _roundtrip(dataset, run)
    metrics = run.report.metrics
    metric_by_split = {metric.split: metric for metric in metrics}
    gradient_names = set(run.report.gradient_parameter_names)
    required_prefixes = (
        "policy_query.",
        "value_head.",
        "range_head.",
        "next_public_delta_head.",
    )
    required_head_gradients = all(
        any(name.startswith(prefix) for name in gradient_names)
        for prefix in required_prefixes
    )
    normalizers_fit_train_only = all(
        normalizer.fit_split == "train"
        for normalizer in (
            artifact.public_normalizer,
            artifact.private_normalizer,
            artifact.action_normalizer,
        )
    )
    overlap = _split_overlap(artifact)
    gates = (
        dataset.summary["steps"] == 610,
        metric_by_split["train"].steps == 385,
        metric_by_split["calibration"].steps == 86,
        metric_by_split["validation"].steps == 139,
        overlap == 0,
        normalizers_fit_train_only,
        required_head_gradients,
        run.report.final_training_loss < run.report.initial_training_loss,
        roundtrip.artifact_serialization_exact,
        roundtrip.legal_support_exact,
        max(
            roundtrip.maximum_action_probability_delta,
            roundtrip.maximum_outcome_value_delta,
            roundtrip.maximum_range_probability_delta,
            roundtrip.maximum_next_public_feature_delta,
        )
        == 0.0,
        not artifact.equilibrium_trained,
        not artifact.deployable_blueprint,
        not artifact.counterfactual_value_provider,
        not artifact.model_official_engine_executed,
    )
    status = (
        "phase6i_logged_neural_bootstrap_validated_partial"
        if all(gates)
        else "failed"
    )
    claims = (
        ("source_transitions", "genuine_official_engine_replays"),
        ("training_boundary", "derived_nonbranchable_abstraction"),
        ("behavior_policy_target", "preserved_heuristic_imitation_not_final_strategy"),
        ("public_range_head", "supervised_six_candidate_engineering_baseline"),
        ("transition_head", "observed_next_public_features_only"),
        ("checkpoint_restore", "validated_exact_predictions_and_legal_support"),
        ("branchable_information_set_game", "not_implemented"),
        ("full_game_equilibrium_blueprint", "not_trained"),
        ("counterfactual_value_bounds", "not_available_full_game"),
        ("official_safe_resolve_pipeline", "not_deployed"),
        ("official_exploitation_overlay", "disabled"),
        ("kaggle_submission", "not_attempted"),
        ("performance_claim", "none_small_development_corpus"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6TrajectoryNeuralEvidence(
        benchmark_id="phase6i-logged-trajectory-neural-bootstrap-v1",
        status=status,
        seed=seed,
        trajectory_dataset_fingerprint=dataset.fingerprint,
        source_public_replay_fingerprint=dataset.source_public_replay_fingerprint,
        artifact_fingerprint=artifact.fingerprint,
        train_steps=metric_by_split["train"].steps,
        calibration_steps=metric_by_split["calibration"].steps,
        validation_steps=metric_by_split["validation"].steps,
        train_matches=len(artifact.train_match_ids),
        calibration_matches=len(artifact.calibration_match_ids),
        validation_matches=len(artifact.validation_match_ids),
        split_match_overlap_count=overlap,
        public_feature_width=len(artifact.public_normalizer.names),
        private_feature_width=len(artifact.private_normalizer.names),
        action_feature_width=len(artifact.action_normalizer.names),
        candidate_count=len(artifact.candidate_ids),
        initial_training_loss=run.report.initial_training_loss,
        final_training_loss=run.report.final_training_loss,
        gradient_parameter_count=len(run.report.gradient_parameter_names),
        required_head_gradients_present=required_head_gradients,
        normalizers_fit_train_only=normalizers_fit_train_only,
        policy_temperature=run.report.policy_temperature,
        range_temperature=run.report.range_temperature,
        metrics=metrics,
        roundtrip=roundtrip,
        claims=claims,
    )
    if evidence.status == "failed":
        raise RuntimeError("Phase 6I trajectory neural bootstrap failed a declared gate")
    return Phase6TrajectoryNeuralArtifacts(evidence=evidence, training_run=run)


def main() -> None:
    print(run_phase6_trajectory_neural_bootstrap().evidence.to_json())


if __name__ == "__main__":
    main()

