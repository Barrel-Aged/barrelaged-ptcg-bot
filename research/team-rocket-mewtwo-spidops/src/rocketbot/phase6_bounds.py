"""Deterministic Phase 6D counterfactual-bound evidence assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from .counterfactual_calibration import (
    BoundCalibrationConfig,
    CalibratedNeuralCounterfactualBoundProvider,
    CounterfactualBoundCalibrationArtifact,
    CounterfactualBoundContext,
    ExactCounterfactualSafetyAudit,
    audit_calibrated_counterfactual_bounds,
    calibrate_neural_counterfactual_bounds,
)
from .exact_game import RocketPrizeDuel
from .neural_blueprint import (
    NeuralBlueprintArtifact,
    NeuralDuelBlueprintProvider,
    restore_neural_duel_blueprint,
)


PHASE6_BOUND_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class Phase6BoundEvidence:
    benchmark_id: str
    status: str
    game_fingerprint: str
    neural_artifact_fingerprint: str
    phase6c_evidence_fingerprint: str
    calibration_fingerprint: str
    bound_provider_fingerprint: str
    calibration_examples: int
    validation_examples: int
    conformal_radius: float
    validation_coverage: float
    base_half_width: float
    widened_half_width: float
    widening_delta: float
    audit: ExactCounterfactualSafetyAudit
    claims: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.status:
            raise ValueError("Phase 6D benchmark identity/status must be non-empty")
        for field_name in (
            "game_fingerprint",
            "neural_artifact_fingerprint",
            "phase6c_evidence_fingerprint",
            "calibration_fingerprint",
            "bound_provider_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if self.calibration_examples <= 0 or self.validation_examples <= 0:
            raise ValueError("Phase 6D example counts must be positive")
        if not 0 <= self.validation_coverage <= 1:
            raise ValueError("Phase 6D validation coverage must be in [0, 1]")
        if min(self.conformal_radius, self.base_half_width, self.widened_half_width, self.widening_delta) < 0:
            raise ValueError("Phase 6D widths must be non-negative")
        if self.widened_half_width + 1e-12 < self.base_half_width:
            raise ValueError("uncertainty widening cannot narrow an interval")
        if len(dict(self.claims)) != len(self.claims):
            raise ValueError("Phase 6D claim names must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_BOUND_SCHEMA_VERSION,
            "benchmark_id": self.benchmark_id,
            "status": self.status,
            "game_fingerprint": self.game_fingerprint,
            "neural_artifact_fingerprint": self.neural_artifact_fingerprint,
            "phase6c_evidence_fingerprint": self.phase6c_evidence_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "bound_provider_fingerprint": self.bound_provider_fingerprint,
            "calibration_examples": self.calibration_examples,
            "validation_examples": self.validation_examples,
            "conformal_radius": self.conformal_radius,
            "validation_coverage": self.validation_coverage,
            "base_half_width": self.base_half_width,
            "widened_half_width": self.widened_half_width,
            "widening_delta": self.widening_delta,
            "audit": {
                **asdict(self.audit),
                "passes_exact_gate": self.audit.passes_exact_gate,
            },
            "claims": dict(self.claims),
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        claims = "\n".join(
            f"| {name} | {value} |" for name, value in self.claims
        )
        return f"""# Phase 6D counterfactual-bound evidence

Benchmark: `{self.benchmark_id}`  
Evidence fingerprint: `{self.fingerprint}`  
Status: **{self.status}**

The neural opponent counterfactual-value residuals are calibrated on one exact-
game information-set split and audited on a disjoint calibration-heldout split.
The split was not held out from neural training. All results below are for the
non-official `RocketPrizeDuel` microgame.

| Metric | Value |
|---|---:|
| Calibration examples | {self.calibration_examples} |
| Validation examples | {self.validation_examples} |
| Symmetric conformal radius | {self.conformal_radius:.10f} |
| Validation coverage | {self.validation_coverage:.3f} |
| Base half-width | {self.base_half_width:.10f} |
| Widened half-width | {self.widened_half_width:.10f} |
| Widening delta | {self.widening_delta:.10f} |
| Validation information sets resolved | {self.audit.validation_information_sets} |
| Safe certificates audited | {self.audit.resolver_certificates} |
| False safe certificates | {self.audit.false_safe_certificates} |
| Maximum exact interval violation | {self.audit.maximum_true_interval_violation:.3e} |
| Maximum certified-bound violation | {self.audit.maximum_certified_bound_violation:.3e} |
| Maximum true baseline regression | {self.audit.maximum_true_baseline_regression:.10f} |
| Regression beyond declared uncertainty | {self.audit.maximum_regression_beyond_declared_uncertainty:.3e} |

High OOD score, excessive incomplete-search fraction, and a confidence request
above the calibrated level each return a blueprint fallback instead of bounds.

## Claim boundary

| Claim | Status |
|---|---|
{claims}
"""


@dataclass(frozen=True, slots=True)
class Phase6BoundArtifacts:
    evidence: Phase6BoundEvidence
    calibration: CounterfactualBoundCalibrationArtifact
    blueprint: NeuralDuelBlueprintProvider


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return raw


def run_phase6_counterfactual_bounds(
    *,
    project_root: Path | None = None,
    selected_checkpoint_path: Path | None = None,
    calibration_config: BoundCalibrationConfig | None = None,
) -> Phase6BoundArtifacts:
    root = project_root or Path(__file__).resolve().parents[2]
    checkpoint_path = selected_checkpoint_path or (
        root
        / "artifacts"
        / "phase6_neural_blueprint"
        / "selected_neural_blueprint.json"
    )
    phase6c_path = (
        root / "artifacts" / "phase6_neural_blueprint" / "phase6c_evidence.json"
    )
    checkpoint = NeuralBlueprintArtifact.from_dict(_read_json(checkpoint_path))
    phase6c = _read_json(phase6c_path)
    phase6c_evidence_fingerprint = str(phase6c["fingerprint"])
    _validate_digest(phase6c_evidence_fingerprint, "phase6c_evidence_fingerprint")
    selected = phase6c.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Phase 6C evidence lacks selected-checkpoint metadata")
    if str(selected["artifact_fingerprint"]) != checkpoint.fingerprint:
        raise ValueError("Phase 6C evidence and selected checkpoint disagree")
    game = RocketPrizeDuel()
    blueprint = restore_neural_duel_blueprint(game, checkpoint)
    calibration = calibrate_neural_counterfactual_bounds(
        game,
        blueprint,
        calibration_config,
    )
    if not calibration.passes_exact_gate:
        raise RuntimeError("Phase 6D calibration failed its exact held-out gate")
    bound_provider = CalibratedNeuralCounterfactualBoundProvider(
        game,
        blueprint,
        calibration,
    )
    audit = audit_calibrated_counterfactual_bounds(
        game,
        blueprint,
        calibration,
    )
    if not audit.passes_exact_gate:
        raise RuntimeError("Phase 6D exact false-safety audit failed")
    validation_id = next(
        item.information_set_id
        for item in calibration.examples
        if item.split == "validation"
    )
    information_set = next(
        item for item in game.information_sets() if item.info_set_id == validation_id
    )
    state = game.states_for_information_set(validation_id)[0]
    actor = state.acting_player
    if actor is None:
        raise RuntimeError("Phase 6D representative validation state has no actor")
    belief = game.public_belief(state, player=actor)
    base = bound_provider.decide_bounds(information_set, belief)
    widened = bound_provider.decide_bounds(
        information_set,
        belief,
        context=CounterfactualBoundContext(
            abstraction_error_bound=0.01,
            ood_score=0.20,
            incomplete_search_fraction=0.25,
            required_confidence=calibration.config.target_coverage,
        ),
    )
    if base.bounds is None or widened.bounds is None:
        raise RuntimeError("Phase 6D in-gate widening unexpectedly fell back")
    if widened.half_width <= base.half_width:
        raise RuntimeError("Phase 6D uncertainty context did not widen bounds")
    claims = (
        ("counterfactual_value_intervals", "validated_exact_microgame_only"),
        ("calibration_heldout_coverage", "validated_1.0_on_12_targets"),
        ("false_safety_certificates", "zero_on_6_exact_resolve_roots"),
        ("uncertainty_widening", "validated_componentwise"),
        ("ood_confidence_interruption_gates", "validated_blueprint_fallback"),
        ("replay_derived_value_calibration", "not_implemented"),
        ("full_game_counterfactual_bounds", "not_calibrated"),
        ("official_engine_safe_resolve", "not_implemented"),
        ("phase6_complete", "false"),
    )
    evidence = Phase6BoundEvidence(
        benchmark_id="phase6d-neural-exact-bounds-v1",
        status="phase6d_exact_counterfactual_bounds_validated",
        game_fingerprint=game.fingerprint,
        neural_artifact_fingerprint=checkpoint.fingerprint,
        phase6c_evidence_fingerprint=phase6c_evidence_fingerprint,
        calibration_fingerprint=calibration.fingerprint,
        bound_provider_fingerprint=bound_provider.fingerprint,
        calibration_examples=calibration.metrics.calibration_examples,
        validation_examples=calibration.metrics.validation_examples,
        conformal_radius=calibration.metrics.conformal_radius,
        validation_coverage=calibration.metrics.validation_coverage,
        base_half_width=base.half_width,
        widened_half_width=widened.half_width,
        widening_delta=widened.half_width - base.half_width,
        audit=audit,
        claims=claims,
    )
    return Phase6BoundArtifacts(
        evidence=evidence,
        calibration=calibration,
        blueprint=blueprint,
    )


def main() -> None:
    artifacts = run_phase6_counterfactual_bounds()
    print(artifacts.evidence.to_json())


if __name__ == "__main__":
    main()
