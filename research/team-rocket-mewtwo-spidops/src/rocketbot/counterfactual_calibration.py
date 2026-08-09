"""Conservative counterfactual-value calibration for Phase 6D.

The implementation in this module is deliberately scoped to the exactly
enumerable :class:`~rocketbot.exact_game.RocketPrizeDuel`.  It calibrates neural
value residuals on one deterministic information-set split and audits coverage
on a disjoint split.  The split is held out from *interval calibration*, not
from the neural blueprint's training, and the evidence reports that distinction.

Nothing here is an official-engine safety certificate.  The widening and gate
contracts are intended to survive later full-game work, while exact enumeration
lets us detect false certificates on the validation microgame today.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .exact_game import (
    ExactGridSafeResolver,
    ExactTabularBlueprintProvider,
    RocketPrizeDuel,
    TabularStrategyProfile,
    evaluate_profile_from_state,
)
from .game_theory import (
    BlueprintStrategyProvider,
    CounterfactualValueBounds,
    HypothesisValueBound,
    InformationSet,
    PublicBeliefState,
    ResolveBudget,
    SafeResolveRequest,
)
from .neural_blueprint import NeuralDuelBlueprintProvider, provider_profile


COUNTERFACTUAL_CALIBRATION_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _finite_nonnegative(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class BoundCalibrationConfig:
    """Predeclared split, coverage, widening, and fail-closed gates."""

    target_coverage: float = 0.90
    split_salt: str = "phase6d-counterfactual-bounds-v1"
    holdout_modulus: int = 3
    holdout_bucket: int = 0
    minimum_calibration_examples: int = 8
    minimum_validation_examples: int = 4
    minimum_validation_coverage: float = 0.90
    numerical_margin: float = 1e-9
    maximum_ood_score: float = 0.50
    maximum_incomplete_search_fraction: float = 0.50
    ood_margin_scale: float = 0.05
    incomplete_search_margin_scale: float = 0.10

    def __post_init__(self) -> None:
        for field_name in (
            "target_coverage",
            "minimum_validation_coverage",
            "maximum_ood_score",
            "maximum_incomplete_search_fraction",
        ):
            value = float(getattr(self, field_name))
            if not 0 < value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and in (0, 1]")
        if not self.split_salt:
            raise ValueError("split_salt must be non-empty")
        if self.holdout_modulus < 2:
            raise ValueError("holdout_modulus must be at least two")
        if not 0 <= self.holdout_bucket < self.holdout_modulus:
            raise ValueError("holdout_bucket must be inside the modulus")
        if self.minimum_calibration_examples <= 0 or self.minimum_validation_examples <= 0:
            raise ValueError("minimum calibration/validation examples must be positive")
        for field_name in (
            "numerical_margin",
            "ood_margin_scale",
            "incomplete_search_margin_scale",
        ):
            _finite_nonnegative(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class CounterfactualCalibrationExample:
    information_set_id: str
    hypothesis_id: str
    opponent_player: int
    split: str
    predicted_value: float
    exact_value: float
    absolute_error: float

    def __post_init__(self) -> None:
        if not self.information_set_id or not self.hypothesis_id:
            raise ValueError("calibration example identities must be non-empty")
        if self.opponent_player not in (0, 1):
            raise ValueError("opponent_player must be zero or one")
        if self.split not in {"calibration", "validation"}:
            raise ValueError("calibration split must be calibration or validation")
        for field_name in ("predicted_value", "exact_value", "absolute_error"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.absolute_error < 0:
            raise ValueError("absolute_error cannot be negative")
        if not math.isclose(
            self.absolute_error,
            abs(self.predicted_value - self.exact_value),
            abs_tol=1e-12,
        ):
            raise ValueError("absolute_error must match predicted/exact values")


@dataclass(frozen=True, slots=True)
class BoundCalibrationMetrics:
    calibration_examples: int
    validation_examples: int
    conformal_rank: int
    conformal_radius: float
    calibration_coverage: float
    validation_coverage: float
    maximum_calibration_error: float
    maximum_validation_error: float
    maximum_validation_miss: float

    def __post_init__(self) -> None:
        if self.calibration_examples <= 0 or self.validation_examples <= 0:
            raise ValueError("calibration metric counts must be positive")
        if not 1 <= self.conformal_rank <= self.calibration_examples:
            raise ValueError("conformal rank must index the calibration examples")
        for field_name in (
            "conformal_radius",
            "maximum_calibration_error",
            "maximum_validation_error",
            "maximum_validation_miss",
        ):
            _finite_nonnegative(getattr(self, field_name), field_name)
        for field_name in ("calibration_coverage", "validation_coverage"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class CounterfactualBoundCalibrationArtifact:
    blueprint_fingerprint: str
    game_fingerprint: str
    deck_fingerprint: str
    profile_fingerprint: str
    split_fingerprint: str
    config: BoundCalibrationConfig
    examples: tuple[CounterfactualCalibrationExample, ...]
    metrics: BoundCalibrationMetrics
    scope_notes: str

    def __post_init__(self) -> None:
        for field_name in (
            "blueprint_fingerprint",
            "game_fingerprint",
            "deck_fingerprint",
            "profile_fingerprint",
            "split_fingerprint",
        ):
            _validate_digest(getattr(self, field_name), field_name)
        if not self.examples or not self.scope_notes:
            raise ValueError("calibration examples and scope notes are required")
        keys = [
            (item.information_set_id, item.hypothesis_id) for item in self.examples
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration examples must be unique")
        calibration = sum(item.split == "calibration" for item in self.examples)
        validation = sum(item.split == "validation" for item in self.examples)
        if calibration != self.metrics.calibration_examples:
            raise ValueError("calibration example count mismatch")
        if validation != self.metrics.validation_examples:
            raise ValueError("validation example count mismatch")

    @property
    def passes_exact_gate(self) -> bool:
        return (
            self.metrics.calibration_examples
            >= self.config.minimum_calibration_examples
            and self.metrics.validation_examples
            >= self.config.minimum_validation_examples
            and self.metrics.validation_coverage
            >= self.config.minimum_validation_coverage
            and self.metrics.maximum_validation_miss <= 1e-12
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": COUNTERFACTUAL_CALIBRATION_SCHEMA_VERSION,
            "blueprint_fingerprint": self.blueprint_fingerprint,
            "game_fingerprint": self.game_fingerprint,
            "deck_fingerprint": self.deck_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "config": asdict(self.config),
            "examples": [asdict(item) for item in self.examples],
            "metrics": asdict(self.metrics),
            "passes_exact_gate": self.passes_exact_gate,
            "scope_notes": self.scope_notes,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any]
    ) -> "CounterfactualBoundCalibrationArtifact":
        if int(raw.get("schema_version", -1)) != COUNTERFACTUAL_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported counterfactual calibration schema")
        artifact = cls(
            blueprint_fingerprint=str(raw["blueprint_fingerprint"]),
            game_fingerprint=str(raw["game_fingerprint"]),
            deck_fingerprint=str(raw["deck_fingerprint"]),
            profile_fingerprint=str(raw["profile_fingerprint"]),
            split_fingerprint=str(raw["split_fingerprint"]),
            config=BoundCalibrationConfig(**dict(raw["config"])),
            examples=tuple(
                CounterfactualCalibrationExample(**dict(item))
                for item in raw["examples"]
            ),
            metrics=BoundCalibrationMetrics(**dict(raw["metrics"])),
            scope_notes=str(raw["scope_notes"]),
        )
        supplied = raw.get("fingerprint")
        if supplied is not None and str(supplied) != artifact.fingerprint:
            raise ValueError("counterfactual calibration fingerprint mismatch")
        if "passes_exact_gate" in raw and bool(raw["passes_exact_gate"]) != artifact.passes_exact_gate:
            raise ValueError("counterfactual calibration gate status mismatch")
        return artifact


def _split_for_information_set(
    information_set_id: str,
    config: BoundCalibrationConfig,
) -> str:
    digest = sha256(
        f"{config.split_salt}|{information_set_id}".encode("utf-8")
    ).hexdigest()
    bucket = int(digest[:16], 16) % config.holdout_modulus
    return "validation" if bucket == config.holdout_bucket else "calibration"


def _exact_profile_provider(
    game: RocketPrizeDuel,
    blueprint: NeuralDuelBlueprintProvider,
) -> tuple[TabularStrategyProfile, ExactTabularBlueprintProvider]:
    profile = provider_profile(
        game,
        blueprint,
        label="neural_blueprint_exact_value_audit",
    )
    provider = ExactTabularBlueprintProvider(
        game,
        profile,
        deck_fingerprint=blueprint.checkpoint.deck_fingerprint,
        algorithm="exact_continuation_value_audit_of_neural_profile",
        training_steps=max(1, blueprint.checkpoint.training_steps),
    )
    return profile, provider


def calibrate_neural_counterfactual_bounds(
    game: RocketPrizeDuel,
    blueprint: NeuralDuelBlueprintProvider,
    config: BoundCalibrationConfig | None = None,
) -> CounterfactualBoundCalibrationArtifact:
    """Calibrate symmetric residual intervals and audit a disjoint split."""

    config = config or BoundCalibrationConfig()
    if blueprint.checkpoint.game_fingerprint != game.fingerprint:
        raise ValueError("blueprint/game fingerprint mismatch")
    if not blueprint.checkpoint.trained:
        raise ValueError("counterfactual calibration requires a trained blueprint")
    profile, truth_provider = _exact_profile_provider(game, blueprint)
    examples: list[CounterfactualCalibrationExample] = []
    split_assignments: dict[str, str] = {}
    for information_set in game.information_sets():
        states = game.states_for_information_set(information_set.info_set_id)
        actor = states[0].acting_player
        if actor is None:
            raise ValueError("calibration root must belong to a player")
        belief = game.public_belief(states[0], player=actor)
        opponent = 1 - actor
        predicted = dict(
            blueprint.counterfactual_values(
                information_set,
                belief,
                player=opponent,
            ).values
        )
        exact = dict(
            truth_provider.counterfactual_values(
                information_set,
                belief,
                player=opponent,
            ).values
        )
        if set(predicted) != set(exact):
            raise ValueError("predicted and exact hypothesis supports differ")
        split = _split_for_information_set(information_set.info_set_id, config)
        split_assignments[information_set.info_set_id] = split
        for hypothesis_id in sorted(exact):
            error = abs(predicted[hypothesis_id] - exact[hypothesis_id])
            examples.append(
                CounterfactualCalibrationExample(
                    information_set_id=information_set.info_set_id,
                    hypothesis_id=hypothesis_id,
                    opponent_player=opponent,
                    split=split,
                    predicted_value=predicted[hypothesis_id],
                    exact_value=exact[hypothesis_id],
                    absolute_error=error,
                )
            )
    examples.sort(key=lambda item: (item.information_set_id, item.hypothesis_id))
    calibration_errors = sorted(
        item.absolute_error for item in examples if item.split == "calibration"
    )
    validation_errors = [
        item.absolute_error for item in examples if item.split == "validation"
    ]
    if len(calibration_errors) < config.minimum_calibration_examples:
        raise ValueError("deterministic split has too few calibration examples")
    if len(validation_errors) < config.minimum_validation_examples:
        raise ValueError("deterministic split has too few validation examples")
    conformal_rank = min(
        len(calibration_errors),
        max(1, math.ceil((len(calibration_errors) + 1) * config.target_coverage)),
    )
    radius = calibration_errors[conformal_rank - 1] + config.numerical_margin
    calibration_coverage = sum(error <= radius for error in calibration_errors) / len(
        calibration_errors
    )
    validation_coverage = sum(error <= radius for error in validation_errors) / len(
        validation_errors
    )
    maximum_validation_miss = max(
        (max(0.0, error - radius) for error in validation_errors),
        default=0.0,
    )
    metrics = BoundCalibrationMetrics(
        calibration_examples=len(calibration_errors),
        validation_examples=len(validation_errors),
        conformal_rank=conformal_rank,
        conformal_radius=radius,
        calibration_coverage=calibration_coverage,
        validation_coverage=validation_coverage,
        maximum_calibration_error=max(calibration_errors),
        maximum_validation_error=max(validation_errors),
        maximum_validation_miss=maximum_validation_miss,
    )
    return CounterfactualBoundCalibrationArtifact(
        blueprint_fingerprint=blueprint.checkpoint.model_fingerprint,
        game_fingerprint=game.fingerprint,
        deck_fingerprint=blueprint.checkpoint.deck_fingerprint,
        profile_fingerprint=profile.fingerprint,
        split_fingerprint=_fingerprint(split_assignments),
        config=config,
        examples=tuple(examples),
        metrics=metrics,
        scope_notes=(
            "Residual intervals are calibrated and validated on disjoint "
            "information-set splits of RocketPrizeDuel. The validation split "
            "is held out from interval calibration, not from neural training. "
            "This is abstract microgame evidence, never official-engine safety."
        ),
    )


@dataclass(frozen=True, slots=True)
class CounterfactualBoundContext:
    """Runtime uncertainty that must widen a bound or trip a fallback gate."""

    abstraction_error_bound: float = 0.0
    ood_score: float = 0.0
    incomplete_search_fraction: float = 0.0
    required_confidence: float = 0.90

    def __post_init__(self) -> None:
        _finite_nonnegative(self.abstraction_error_bound, "abstraction_error_bound")
        for field_name in ("ood_score", "incomplete_search_fraction", "required_confidence"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class CalibratedBoundDecision:
    bounds: CounterfactualValueBounds | None
    fallback_to_blueprint: bool
    reason: str
    half_width: float
    widening_components: tuple[tuple[str, float], ...]
    calibration_fingerprint: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("bound decision reason must be non-empty")
        _finite_nonnegative(self.half_width, "half_width")
        _validate_digest(self.calibration_fingerprint, "calibration_fingerprint")
        if self.fallback_to_blueprint != (self.bounds is None):
            raise ValueError("fallback decisions must omit bounds and accepted decisions need bounds")
        names = [name for name, _ in self.widening_components]
        if len(names) != len(set(names)):
            raise ValueError("widening component names must be unique")
        for name, value in self.widening_components:
            if not name:
                raise ValueError("widening components must be named")
            _finite_nonnegative(value, f"widening component {name}")


class BoundGateFailure(RuntimeError):
    """Raised when the protocol-only ``bounds`` method cannot pass safety gates."""


class CalibratedNeuralCounterfactualBoundProvider:
    """Exact-game neural interval provider with explicit widen-or-fallback gates."""

    def __init__(
        self,
        game: RocketPrizeDuel,
        blueprint: NeuralDuelBlueprintProvider,
        calibration: CounterfactualBoundCalibrationArtifact,
    ) -> None:
        if calibration.game_fingerprint != game.fingerprint:
            raise ValueError("calibration/game fingerprint mismatch")
        if calibration.blueprint_fingerprint != blueprint.checkpoint.model_fingerprint:
            raise ValueError("calibration/blueprint fingerprint mismatch")
        if calibration.deck_fingerprint != blueprint.checkpoint.deck_fingerprint:
            raise ValueError("calibration/deck fingerprint mismatch")
        self.game = game
        self.blueprint = blueprint
        self.calibration = calibration

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "provider": "calibrated_neural_counterfactual_bounds",
                "game": self.game.fingerprint,
                "blueprint": self.blueprint.checkpoint.model_fingerprint,
                "calibration": self.calibration.fingerprint,
            }
        )

    def _fallback(self, reason: str) -> CalibratedBoundDecision:
        return CalibratedBoundDecision(
            bounds=None,
            fallback_to_blueprint=True,
            reason=reason,
            half_width=0.0,
            widening_components=(),
            calibration_fingerprint=self.calibration.fingerprint,
        )

    def decide_bounds(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        blueprint: BlueprintStrategyProvider | None = None,
        *,
        context: CounterfactualBoundContext | None = None,
    ) -> CalibratedBoundDecision:
        supplied_blueprint = blueprint or self.blueprint
        supplied_checkpoint = getattr(supplied_blueprint, "checkpoint", None)
        supplied_fingerprint = getattr(supplied_checkpoint, "model_fingerprint", None)
        if supplied_fingerprint != self.blueprint.checkpoint.model_fingerprint:
            return self._fallback("blueprint fingerprint differs from calibration")
        if not self.calibration.passes_exact_gate:
            return self._fallback("calibration failed the held-out exact gate")
        context = context or CounterfactualBoundContext(
            required_confidence=self.calibration.config.target_coverage
        )
        config = self.calibration.config
        if context.required_confidence > config.target_coverage + 1e-12:
            return self._fallback("requested confidence exceeds calibrated confidence")
        if context.ood_score > config.maximum_ood_score:
            return self._fallback("out-of-distribution score exceeds calibrated gate")
        if (
            context.incomplete_search_fraction
            > config.maximum_incomplete_search_fraction
        ):
            return self._fallback("incomplete-search fraction exceeds calibrated gate")
        states = self.game.states_for_information_set(information_set.info_set_id)
        actor = states[0].acting_player
        if actor is None or any(state.acting_player != actor for state in states):
            return self._fallback("information set does not have one acting player")
        opponent = 1 - actor
        strategy = supplied_blueprint.strategy(information_set, belief)
        values = supplied_blueprint.counterfactual_values(
            information_set,
            belief,
            player=opponent,
        )
        components = (
            ("calibrated_residual", self.calibration.metrics.conformal_radius),
            ("abstraction_error", context.abstraction_error_bound),
            ("out_of_distribution", context.ood_score * config.ood_margin_scale),
            (
                "incomplete_search",
                context.incomplete_search_fraction
                * config.incomplete_search_margin_scale,
            ),
        )
        half_width = sum(value for _, value in components)
        bounds = CounterfactualValueBounds(
            opponent_player=opponent,
            bounds=tuple(
                HypothesisValueBound(
                    hypothesis_id=hypothesis_id,
                    lower=value - half_width,
                    upper=value + half_width,
                    baseline_value=value,
                )
                for hypothesis_id, value in values.values
            ),
            confidence=config.target_coverage,
            baseline_strategy_fingerprint=strategy.fingerprint,
            source=(
                "phase6d calibrated neural residual bound; exact abstract "
                f"microgame only; calibration={self.calibration.fingerprint}"
            ),
        )
        return CalibratedBoundDecision(
            bounds=bounds,
            fallback_to_blueprint=False,
            reason="all calibration, confidence, OOD, and completion gates passed",
            half_width=half_width,
            widening_components=components,
            calibration_fingerprint=self.calibration.fingerprint,
        )

    def bounds(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        blueprint: BlueprintStrategyProvider,
    ) -> CounterfactualValueBounds:
        decision = self.decide_bounds(information_set, belief, blueprint)
        if decision.bounds is None:
            raise BoundGateFailure(decision.reason)
        return decision.bounds


@dataclass(frozen=True, slots=True)
class ExactCounterfactualSafetyAudit:
    validation_information_sets: int
    validation_hypotheses: int
    baseline_interval_coverage: float
    resolver_certificates: int
    false_safe_certificates: int
    fallback_resolves: int
    maximum_true_interval_violation: float
    maximum_certified_bound_violation: float
    maximum_true_baseline_regression: float
    maximum_regression_beyond_declared_uncertainty: float
    ood_gate_fallback: bool
    incomplete_search_gate_fallback: bool
    confidence_gate_fallback: bool
    audit_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "validation_information_sets",
            "validation_hypotheses",
            "resolver_certificates",
            "false_safe_certificates",
            "fallback_resolves",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if not 0 <= self.baseline_interval_coverage <= 1:
            raise ValueError("baseline_interval_coverage must be in [0, 1]")
        for field_name in (
            "maximum_true_interval_violation",
            "maximum_certified_bound_violation",
            "maximum_true_baseline_regression",
            "maximum_regression_beyond_declared_uncertainty",
        ):
            _finite_nonnegative(getattr(self, field_name), field_name)
        _validate_digest(self.audit_fingerprint, "audit_fingerprint")

    @property
    def passes_exact_gate(self) -> bool:
        return (
            self.validation_information_sets > 0
            and self.validation_hypotheses > 0
            and self.baseline_interval_coverage >= 0.90
            and self.false_safe_certificates == 0
            and self.maximum_true_interval_violation <= 1e-12
            and self.maximum_certified_bound_violation <= 1e-12
            and self.maximum_regression_beyond_declared_uncertainty <= 1e-12
            and self.ood_gate_fallback
            and self.incomplete_search_gate_fallback
            and self.confidence_gate_fallback
        )


def audit_calibrated_counterfactual_bounds(
    game: RocketPrizeDuel,
    blueprint: NeuralDuelBlueprintProvider,
    calibration: CounterfactualBoundCalibrationArtifact,
    *,
    grid_steps: int = 100,
) -> ExactCounterfactualSafetyAudit:
    """Audit interval truth coverage and resolver certificates by enumeration."""

    if grid_steps <= 0:
        raise ValueError("grid_steps must be positive")
    bound_provider = CalibratedNeuralCounterfactualBoundProvider(
        game,
        blueprint,
        calibration,
    )
    profile, truth_provider = _exact_profile_provider(game, blueprint)
    resolver = ExactGridSafeResolver(profile, grid_steps=grid_steps)
    validation_ids = sorted(
        {
            item.information_set_id
            for item in calibration.examples
            if item.split == "validation"
        }
    )
    covered = 0
    hypothesis_count = 0
    false_safe = 0
    fallback_resolves = 0
    max_interval_violation = 0.0
    max_certified_violation = 0.0
    max_baseline_regression = 0.0
    max_excess_regression = 0.0
    resolver_certificates = 0
    trace: list[dict[str, Any]] = []
    representative: tuple[InformationSet, PublicBeliefState] | None = None
    for identifier in validation_ids:
        information_set = next(
            item for item in game.information_sets() if item.info_set_id == identifier
        )
        states = game.states_for_information_set(identifier)
        actor = states[0].acting_player
        if actor is None:
            raise ValueError("validation information set has no acting player")
        opponent = 1 - actor
        belief = game.public_belief(states[0], player=actor)
        representative = representative or (information_set, belief)
        decision = bound_provider.decide_bounds(information_set, belief)
        if decision.bounds is None:
            raise RuntimeError("in-distribution exact validation unexpectedly fell back")
        supplied = {item.hypothesis_id: item for item in decision.bounds.bounds}
        exact_baseline = dict(
            truth_provider.counterfactual_values(
                information_set,
                belief,
                player=opponent,
            ).values
        )
        for hypothesis_id, exact_value in exact_baseline.items():
            bound = supplied[hypothesis_id]
            violation = max(0.0, bound.lower - exact_value, exact_value - bound.upper)
            max_interval_violation = max(max_interval_violation, violation)
            covered += violation <= 1e-12
            hypothesis_count += 1
        baseline = blueprint.strategy(information_set, belief)
        request = SafeResolveRequest(
            information_set=information_set,
            belief=belief,
            baseline_strategy=baseline,
            opponent_value_bounds=decision.bounds,
            budget=ResolveBudget(
                iterations=grid_steps + 1,
                max_nodes=(grid_steps + 1) * max(1, len(states)),
                time_budget_ms=1_000.0,
                safety_tolerance=0.0,
            ),
        )
        result = resolver.resolve(game, request)
        resolver_certificates += 1
        fallback_resolves += result.fallback_to_blueprint
        candidate_profile = profile.with_strategy(
            identifier,
            result.strategy,
            label="phase6d_exact_certificate_audit",
        )
        state_by_hypothesis = {}
        for state in states:
            card = state.cards[opponent]
            if card is None:
                raise ValueError("validation state lacks opponent card")
            state_by_hypothesis[f"opponent:{card.label}"] = state
        root_false = False
        root_max_violation = 0.0
        root_max_regression = 0.0
        root_max_excess = 0.0
        for hypothesis_id, state in state_by_hypothesis.items():
            candidate_value = evaluate_profile_from_state(
                game,
                candidate_profile,
                state,
                player=opponent,
            )
            upper = supplied[hypothesis_id].upper
            certified_violation = max(0.0, candidate_value - upper)
            baseline_value = exact_baseline[hypothesis_id]
            regression = max(0.0, candidate_value - baseline_value)
            uncertainty_allowance = max(0.0, upper - baseline_value)
            excess = max(0.0, regression - uncertainty_allowance)
            root_max_violation = max(root_max_violation, certified_violation)
            root_max_regression = max(root_max_regression, regression)
            root_max_excess = max(root_max_excess, excess)
            root_false = root_false or certified_violation > 1e-12
        if result.certificate.safe and root_false:
            false_safe += 1
        max_certified_violation = max(max_certified_violation, root_max_violation)
        max_baseline_regression = max(max_baseline_regression, root_max_regression)
        max_excess_regression = max(max_excess_regression, root_max_excess)
        trace.append(
            {
                "information_set": identifier,
                "bounds": decision.bounds.fingerprint,
                "strategy": result.strategy.fingerprint,
                "safe": result.certificate.safe,
                "fallback": result.fallback_to_blueprint,
                "max_truth_violation": root_max_violation,
                "max_baseline_regression": root_max_regression,
                "max_excess_regression": root_max_excess,
            }
        )
    if representative is None:
        raise RuntimeError("calibration contains no validation information sets")
    information_set, belief = representative
    config = calibration.config
    ood_fallback = bound_provider.decide_bounds(
        information_set,
        belief,
        context=CounterfactualBoundContext(
            ood_score=min(1.0, config.maximum_ood_score + 0.01),
            required_confidence=config.target_coverage,
        ),
    ).fallback_to_blueprint
    incomplete_fallback = bound_provider.decide_bounds(
        information_set,
        belief,
        context=CounterfactualBoundContext(
            incomplete_search_fraction=min(
                1.0, config.maximum_incomplete_search_fraction + 0.01
            ),
            required_confidence=config.target_coverage,
        ),
    ).fallback_to_blueprint
    confidence_fallback = bound_provider.decide_bounds(
        information_set,
        belief,
        context=CounterfactualBoundContext(
            required_confidence=min(1.0, config.target_coverage + 0.01)
        ),
    ).fallback_to_blueprint
    audit_payload = {
        "schema": COUNTERFACTUAL_CALIBRATION_SCHEMA_VERSION,
        "game": game.fingerprint,
        "blueprint": blueprint.checkpoint.model_fingerprint,
        "calibration": calibration.fingerprint,
        "grid_steps": grid_steps,
        "trace": trace,
        "ood_fallback": ood_fallback,
        "incomplete_fallback": incomplete_fallback,
        "confidence_fallback": confidence_fallback,
    }
    return ExactCounterfactualSafetyAudit(
        validation_information_sets=len(validation_ids),
        validation_hypotheses=hypothesis_count,
        baseline_interval_coverage=covered / hypothesis_count,
        resolver_certificates=resolver_certificates,
        false_safe_certificates=false_safe,
        fallback_resolves=fallback_resolves,
        maximum_true_interval_violation=max_interval_violation,
        maximum_certified_bound_violation=max_certified_violation,
        maximum_true_baseline_regression=max_baseline_regression,
        maximum_regression_beyond_declared_uncertainty=max_excess_regression,
        ood_gate_fallback=ood_fallback,
        incomplete_search_gate_fallback=incomplete_fallback,
        confidence_gate_fallback=confidence_fallback,
        audit_fingerprint=_fingerprint(audit_payload),
    )
