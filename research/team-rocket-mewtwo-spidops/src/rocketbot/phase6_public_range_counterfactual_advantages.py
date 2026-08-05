"""Phase 6AM public-range normalized counterfactual-advantage audit.

The official engine supplies complete matches, retained snapshots, native
branches, and descendants.  The candidate range, hidden allocations, five-plan
boundary, continuation blueprint, counterfactual normalization, conformal
bounds, and every admission decision are research abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

from .beliefs import OPPONENT_INTENTS
from .counterfactual_reach import CommonChanceLawReachRatio
from .game_theory import (
    InformationSetActor,
    OpponentRange,
    PublicBeliefState,
)
from .native_blueprint_features import native_observer_public_feature_items
from .official_bridge import sanitize_official_public_state
from .phase6_branch_conditioned_values import (
    BranchConditionedRoot,
    build_branch_value_examples_from_roots,
    _build_branch_conditioned_root,
)
from .phase6_expected_value_surfaces import (
    PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
    PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
    _CompleteMatchCapture,
    _ExpectedValueSources,
    _load_expected_value_sources,
    _root_public_history,
)
from .phase6_high_precision_confirmation import (
    HighPrecisionModelReference,
    HighPrecisionPanel,
)
from .phase6_official_range_calibration import (
    CandidateParticleSummary,
    OfficialPublicRangeModel,
    _candidate_particle_summaries,
    _uniform_candidate_range,
)


PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SCHEMA_VERSION = 1
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_SEED = 6_060
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_RANGE_SEED = 6_061
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BOOTSTRAP_SEED = 6_062
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BOOTSTRAP_RESAMPLES = 10_000
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS = (6_060, 1_006_060)
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_ROOTS = 36
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_RETURNS_PER_ACTION = 32
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_COVERAGE = 0.975
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_RANK = 352
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_FROZEN_RADIUS = 0.555019619275336
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_COVERAGE = 0.95
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_SLICE_COVERAGE = 0.80
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_MEAN_WIDTH = 1.25
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_TARGET_SE = 0.20
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_RANGE_TOP1 = 0.95
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_RANGE_ECE = 0.15
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_CANDIDATE_TOP1 = 5.0 / 6.0
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_RANGE_MODEL_FINGERPRINT = (
    "f4574ce0b5facec0ab2e6e2c08063d575c7f92112a8c109d04e5a81665ec04ca"
)
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_PLAN_FINGERPRINT = (
    "b416597d44ad31329acf46bb59a517ffa09a722e4e4e7cf14c79176a38c343d9"
)
PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_TACTICAL_FINGERPRINT = (
    "e9cf893ccf28eeee656af9791de389196d7a869d9bdae090e0e15b7dc5f83232"
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Phase 6AM {name} is not a SHA-256 digest")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Phase 6AM {name} is nonfinite")
    return result


def _write_artifact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _load_range_model(project_root: Path) -> OfficialPublicRangeModel:
    path = (
        project_root
        / "artifacts"
        / "phase6_official_range_calibration"
        / "official_public_range_model.json"
    )
    model = OfficialPublicRangeModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    if model.fingerprint != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_RANGE_MODEL_FINGERPRINT:
        raise ValueError("Phase 6AM source range model fingerprint drifted")
    return model


def _load_phase6al_panels(project_root: Path) -> tuple[HighPrecisionPanel, HighPrecisionPanel]:
    directory = project_root / "artifacts" / "phase6_high_precision_confirmation"
    calibration = HighPrecisionPanel.from_dict(
        json.loads((directory / "calibration_panel.json").read_text(encoding="utf-8"))
    )
    validation = HighPrecisionPanel.from_dict(
        json.loads((directory / "validation_panel.json").read_text(encoding="utf-8"))
    )
    if calibration.role != "calibration" or validation.role != "validation":
        raise ValueError("Phase 6AM source Phase 6AL panel roles drifted")
    return calibration, validation


@dataclass(frozen=True, slots=True)
class CounterfactualAdvantageTarget:
    action_id: str
    point_center: float
    empirical_advantage: float
    target_standard_error: float
    lower: float
    upper: float
    covered: bool

    def __post_init__(self) -> None:
        for name in (
            "point_center",
            "empirical_advantage",
            "target_standard_error",
            "lower",
            "upper",
        ):
            _finite(getattr(self, name), name)
        if (
            not self.action_id
            or self.target_standard_error < 0
            or not -2 <= self.point_center <= 2
            or not -2 <= self.empirical_advantage <= 2
            or not -2 <= self.lower <= self.upper <= 2
        ):
            raise ValueError("Phase 6AM advantage target is invalid")
        empirical_lower = max(
            -2.0,
            self.empirical_advantage - 1.96 * self.target_standard_error,
        )
        empirical_upper = min(
            2.0,
            self.empirical_advantage + 1.96 * self.target_standard_error,
        )
        expected = self.lower <= empirical_lower and empirical_upper <= self.upper
        if self.covered != expected:
            raise ValueError("Phase 6AM advantage coverage flag is inconsistent")

    @property
    def width(self) -> float:
        return self.upper - self.lower


def _advantage_targets(
    batches: Sequence[BranchConditionedRoot],
    candidate_ids: Sequence[str],
    *,
    radius: float,
) -> tuple[CounterfactualAdvantageTarget, ...]:
    examples = build_branch_value_examples_from_roots(
        batches,
        candidate_ids,
        expected_count=len(batches),
    )
    first = examples[0]
    if any(
        item.root_features != first.root_features
        or tuple(action.action_id for action in item.actions)
        != tuple(action.action_id for action in first.actions)
        for item in examples[1:]
    ):
        raise ValueError("Phase 6AM replay batches are not root/action aligned")
    probabilities = dict(batches[0].value_root.blueprint_probabilities)
    if any(dict(item.value_root.blueprint_probabilities) != probabilities for item in batches):
        raise ValueError("Phase 6AM blueprint changed across replay batches")
    returns = {
        action.action_id: tuple(
            value
            for example in examples
            for value in next(
                item for item in example.actions if item.action_id == action.action_id
            ).terminal_returns
        )
        for action in first.actions
    }
    count = len(next(iter(returns.values())))
    if count not in (16, PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_RETURNS_PER_ACTION):
        raise ValueError("Phase 6AM return count is unsupported")
    baseline_returns = tuple(
        sum(probabilities[action_id] * values[index] for action_id, values in returns.items())
        for index in range(count)
    )
    branch_values = {
        action.action_id: statistics.mean(
            next(
                item for item in example.actions if item.action_id == action.action_id
            ).branch_heuristic_value
            for example in examples
        )
        for action in first.actions
    }
    branch_baseline = sum(
        probabilities[action_id] * value for action_id, value in branch_values.items()
    )
    targets = []
    for action in first.actions:
        advantages = tuple(
            value - baseline
            for value, baseline in zip(
                returns[action.action_id],
                baseline_returns,
                strict=True,
            )
        )
        empirical = statistics.mean(advantages)
        standard_error = statistics.stdev(advantages) / math.sqrt(len(advantages))
        center = branch_values[action.action_id] - branch_baseline
        lower = max(-2.0, center - radius)
        upper = min(2.0, center + radius)
        empirical_lower = max(-2.0, empirical - 1.96 * standard_error)
        empirical_upper = min(2.0, empirical + 1.96 * standard_error)
        targets.append(
            CounterfactualAdvantageTarget(
                action_id=action.action_id,
                point_center=center,
                empirical_advantage=empirical,
                target_standard_error=standard_error,
                lower=lower,
                upper=upper,
                covered=lower <= empirical_lower and empirical_upper <= upper,
            )
        )
    return tuple(targets)


def _development_scores(
    panels: Sequence[HighPrecisionPanel],
) -> tuple[float, ...]:
    scores = []
    for panel in panels:
        for root in panel.roots:
            for target in _advantage_targets(root.batches, panel.candidate_ids, radius=0.0):
                scores.append(
                    abs(target.point_center - target.empirical_advantage)
                    + 1.96 * target.target_standard_error
                )
    return tuple(scores)


@dataclass(frozen=True, slots=True)
class PublicRangeCounterfactualReference:
    source_phase6al_calibration_panel_fingerprint: str
    source_phase6al_validation_panel_fingerprint: str
    source_range_model_fingerprint: str
    source_plan_fingerprint: str
    source_tactical_fingerprint: str
    development_root_ids: tuple[str, ...]
    calibration_score_count: int
    calibration_coverage: float
    calibration_rank: int
    advantage_radius: float
    prior_roots_declared_development: bool = True
    validation_roots_collected_before_freeze: int = 0
    absolute_official_chance_reach_available: bool = False
    safe_resolver_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_phase6al_calibration_panel_fingerprint",
            "source_phase6al_validation_panel_fingerprint",
            "source_range_model_fingerprint",
            "source_plan_fingerprint",
            "source_tactical_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        if (
            self.source_range_model_fingerprint
            != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_RANGE_MODEL_FINGERPRINT
            or self.source_plan_fingerprint
            != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_PLAN_FINGERPRINT
            or self.source_tactical_fingerprint
            != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_TACTICAL_FINGERPRINT
            or len(self.development_root_ids) != 360
            or len(set(self.development_root_ids)) != 360
            or self.calibration_score_count != 360
            or not math.isclose(
                self.calibration_coverage,
                PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_COVERAGE,
                abs_tol=1e-12,
            )
            or self.calibration_rank != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_RANK
            or not math.isclose(
                self.advantage_radius,
                PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_FROZEN_RADIUS,
                abs_tol=1e-12,
            )
            or not self.prior_roots_declared_development
            or self.validation_roots_collected_before_freeze != 0
            or self.absolute_official_chance_reach_available
            or self.safe_resolver_enabled
            or self.official_strategy_control_enabled
        ):
            raise ValueError("Phase 6AM frozen reference contract drifted")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SCHEMA_VERSION,
            "sources": {
                "phase6al_calibration_panel_fingerprint": self.source_phase6al_calibration_panel_fingerprint,
                "phase6al_validation_panel_fingerprint": self.source_phase6al_validation_panel_fingerprint,
                "range_model_fingerprint": self.source_range_model_fingerprint,
                "plan_fingerprint": self.source_plan_fingerprint,
                "tactical_fingerprint": self.source_tactical_fingerprint,
            },
            "development_root_ids": list(self.development_root_ids),
            "calibration": {
                "score_count": self.calibration_score_count,
                "coverage": self.calibration_coverage,
                "rank": self.calibration_rank,
                "advantage_radius": self.advantage_radius,
            },
            "claims": {
                "prior_roots_declared_development": self.prior_roots_declared_development,
                "validation_roots_collected_before_freeze": self.validation_roots_collected_before_freeze,
                "absolute_official_chance_reach_available": self.absolute_official_chance_reach_available,
                "safe_resolver_enabled": self.safe_resolver_enabled,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRangeCounterfactualReference":
        sources = raw["sources"]
        calibration = raw["calibration"]
        claims = raw["claims"]
        result = cls(
            source_phase6al_calibration_panel_fingerprint=str(
                sources["phase6al_calibration_panel_fingerprint"]
            ),
            source_phase6al_validation_panel_fingerprint=str(
                sources["phase6al_validation_panel_fingerprint"]
            ),
            source_range_model_fingerprint=str(sources["range_model_fingerprint"]),
            source_plan_fingerprint=str(sources["plan_fingerprint"]),
            source_tactical_fingerprint=str(sources["tactical_fingerprint"]),
            development_root_ids=tuple(str(value) for value in raw["development_root_ids"]),
            calibration_score_count=int(calibration["score_count"]),
            calibration_coverage=float(calibration["coverage"]),
            calibration_rank=int(calibration["rank"]),
            advantage_radius=float(calibration["advantage_radius"]),
            prior_roots_declared_development=bool(claims["prior_roots_declared_development"]),
            validation_roots_collected_before_freeze=int(
                claims["validation_roots_collected_before_freeze"]
            ),
            absolute_official_chance_reach_available=bool(
                claims["absolute_official_chance_reach_available"]
            ),
            safe_resolver_enabled=bool(claims["safe_resolver_enabled"]),
            official_strategy_control_enabled=bool(claims["official_strategy_control_enabled"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AM reference fingerprint mismatch")
        return result


def prepare_public_range_counterfactual_reference(
    project_root: str | Path | None = None,
) -> PublicRangeCounterfactualReference:
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    calibration, validation = _load_phase6al_panels(root)
    prior_reference = HighPrecisionModelReference.from_dict(
        json.loads(
            (
                root
                / "artifacts"
                / "phase6_high_precision_confirmation"
                / "high_precision_model_reference.json"
            ).read_text(encoding="utf-8")
        )
    )
    development_root_ids = tuple(
        sorted(
            prior_reference.development_root_ids
            + tuple(item.root_id for panel in (calibration, validation) for item in panel.roots)
        )
    )
    scores = sorted(_development_scores((calibration, validation)))
    rank = math.ceil((len(scores) + 1) * PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_COVERAGE)
    radius = scores[min(len(scores), rank) - 1]
    if rank != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_RANK or not math.isclose(
        radius,
        PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_FROZEN_RADIUS,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Phase 6AM preregistered calibration no longer reproduces")
    sources = _load_expected_value_sources(root)
    if (
        sources.plan_artifact.fingerprint
        != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_PLAN_FINGERPRINT
        or sources.tactical_artifact.fingerprint
        != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_TACTICAL_FINGERPRINT
    ):
        raise ValueError("Phase 6AM blueprint source fingerprint drifted")
    _load_range_model(root)
    return PublicRangeCounterfactualReference(
        source_phase6al_calibration_panel_fingerprint=calibration.fingerprint,
        source_phase6al_validation_panel_fingerprint=validation.fingerprint,
        source_range_model_fingerprint=PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SOURCE_RANGE_MODEL_FINGERPRINT,
        source_plan_fingerprint=sources.plan_artifact.fingerprint,
        source_tactical_fingerprint=sources.tactical_artifact.fingerprint,
        development_root_ids=development_root_ids,
        calibration_score_count=len(scores),
        calibration_coverage=PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_CALIBRATION_COVERAGE,
        calibration_rank=rank,
        advantage_radius=radius,
    )


@dataclass(frozen=True, slots=True)
class PublicRangeCounterfactualRoot:
    true_candidate_id_supervision_only: str
    true_intent_id_supervision_only: str
    public_belief: PublicBeliefState
    candidate_summaries: tuple[CandidateParticleSummary, ...]
    common_chance_ratio: CommonChanceLawReachRatio
    batch_seeds: tuple[int, ...]
    batches: tuple[BranchConditionedRoot, ...]
    candidate_truth_selected_for_range_input: bool = False
    raw_private_card_ids_persisted: int = 0

    def __post_init__(self) -> None:
        if (
            self.true_candidate_id_supervision_only
            not in {item.candidate_id for item in self.candidate_summaries}
            or self.true_intent_id_supervision_only not in {item.value for item in OPPONENT_INTENTS}
            or self.batch_seeds != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS
            or len(self.batches) != 2
            or self.candidate_truth_selected_for_range_input
            or self.raw_private_card_ids_persisted != 0
        ):
            raise ValueError("Phase 6AM root identity or leakage contract drifted")
        if self.common_chance_ratio.target_player != 0 or not math.isclose(
            self.common_chance_ratio.counterfactual_likelihood_ratio,
            1.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Phase 6AM common-chance reach contract drifted")
        if self.common_chance_ratio.absolute_chance_reach_available:
            raise ValueError("Phase 6AM invented absolute chance reach")
        identifiers = tuple(item.candidate_id for item in self.candidate_summaries)
        range_ids = tuple(
            item.hypothesis_id.removeprefix("official-range:")
            for item in self.public_belief.opponent_range.hypotheses
        )
        if identifiers != range_ids or any(item.particle_count != 32 for item in self.candidate_summaries):
            raise ValueError("Phase 6AM range candidate support drifted")
        first = self.batches[0].value_root
        if any(
            batch.value_root.public_state_fingerprint != first.public_state_fingerprint
            or batch.value_root.public_history_fingerprint != first.public_history_fingerprint
            or batch.value_root.private_observation_fingerprint
            != first.private_observation_fingerprint
            or batch.value_root.blueprint_probabilities != first.blueprint_probabilities
            for batch in self.batches[1:]
        ):
            raise ValueError("Phase 6AM replay batches do not share one retained root")
        if (
            self.public_belief.public_state_fingerprint != first.public_state_fingerprint
            or self.public_belief.public_history_fingerprint != first.public_history_fingerprint
            or self.public_belief.acting_player is not InformationSetActor.SELF
            or sum(
                batch.value_root.terminal_rollout_count for batch in self.batches
            )
            != 160
        ):
            raise ValueError("Phase 6AM public belief/replay accounting drifted")

    @property
    def top_candidate_id(self) -> str:
        return min(
            self.public_belief.opponent_range.hypotheses,
            key=lambda item: (-item.weight, item.hypothesis_id),
        ).hypothesis_id.removeprefix("official-range:")

    @property
    def true_candidate_probability(self) -> float:
        return next(
            item.weight
            for item in self.public_belief.opponent_range.hypotheses
            if item.hypothesis_id
            == f"official-range:{self.true_candidate_id_supervision_only}"
        )

    @property
    def root_id(self) -> str:
        return _fingerprint(
            {
                "belief": self.public_belief.fingerprint,
                "reach": self.common_chance_ratio.fingerprint,
                "batches": [item.root_id for item in self.batches],
            }
        )

    @property
    def match_id(self) -> str:
        return self.batches[0].value_root.match_id

    def targets(self, radius: float) -> tuple[CounterfactualAdvantageTarget, ...]:
        return _advantage_targets(
            self.batches,
            tuple(item.candidate_id for item in self.candidate_summaries),
            radius=radius,
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SCHEMA_VERSION,
            "root_id": self.root_id,
            "supervision_only": {
                "candidate_id": self.true_candidate_id_supervision_only,
                "intent_id": self.true_intent_id_supervision_only,
            },
            # Normalize tuple-valued nested range resources to their exact JSON
            # representation so artifact -> object -> artifact is byte-shape stable.
            "public_belief": json.loads(_canonical(self.public_belief.to_dict())),
            "candidate_summaries": [item.to_dict() for item in self.candidate_summaries],
            "common_chance_ratio": self.common_chance_ratio.to_dict(),
            "batch_seeds": list(self.batch_seeds),
            "batches": [item.to_dict() for item in self.batches],
            "claims": {
                "one_complete_official_match_executed": True,
                "retained_snapshot_replayed_per_batch": True,
                "descendant_transitions_official_engine_executed": True,
                "candidate_truth_selected_for_range_input": self.candidate_truth_selected_for_range_input,
                "raw_private_card_ids_persisted": self.raw_private_card_ids_persisted,
                "range_model_official_engine_executed": False,
                "absolute_official_chance_reach_available": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.root_id
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRangeCounterfactualRoot":
        claims = raw["claims"]
        if not all(
            bool(claims[name])
            for name in (
                "one_complete_official_match_executed",
                "retained_snapshot_replayed_per_batch",
                "descendant_transitions_official_engine_executed",
            )
        ) or bool(claims["range_model_official_engine_executed"]):
            raise ValueError("Phase 6AM execution provenance changed")
        if bool(claims["absolute_official_chance_reach_available"]):
            raise ValueError("Phase 6AM invented chance reach")
        supervision = raw["supervision_only"]
        result = cls(
            true_candidate_id_supervision_only=str(supervision["candidate_id"]),
            true_intent_id_supervision_only=str(supervision["intent_id"]),
            public_belief=PublicBeliefState.from_dict(raw["public_belief"]),
            candidate_summaries=tuple(
                CandidateParticleSummary.from_dict(item)
                for item in raw["candidate_summaries"]
            ),
            common_chance_ratio=CommonChanceLawReachRatio.from_dict(
                raw["common_chance_ratio"]
            ),
            batch_seeds=tuple(int(value) for value in raw["batch_seeds"]),
            batches=tuple(BranchConditionedRoot.from_dict(item) for item in raw["batches"]),
            candidate_truth_selected_for_range_input=bool(
                claims["candidate_truth_selected_for_range_input"]
            ),
            raw_private_card_ids_persisted=int(claims["raw_private_card_ids_persisted"]),
        )
        if raw.get("root_id") not in (None, result.root_id) or raw.get("fingerprint") not in (
            None,
            result.root_id,
        ):
            raise ValueError("Phase 6AM root fingerprint mismatch")
        return result


def build_public_range_counterfactual_root(
    capture: _CompleteMatchCapture,
    *,
    sources: _ExpectedValueSources,
    range_model: OfficialPublicRangeModel,
    true_candidate: Any,
    intent: Any,
    match_id: str,
    reference: PublicRangeCounterfactualReference,
) -> PublicRangeCounterfactualRoot:
    observation = capture.snapshot.observation
    candidate_ids = tuple(item.archetype for item in sources.candidates)
    public_features = native_observer_public_feature_items(
        observation,
        _uniform_candidate_range(candidate_ids),
        observer_player=0,
    )
    summaries = _candidate_particle_summaries(
        observation,
        target=sources.target,
        candidates=sources.candidates,
        particles_per_candidate=32,
        root_token=match_id,
        style="action_first",
        seed=PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_RANGE_SEED,
    )
    opponent_range = range_model.opponent_range(public_features, summaries)
    belief = PublicBeliefState(
        public_state_fingerprint=_fingerprint(sanitize_official_public_state(observation)),
        public_history_fingerprint=_root_public_history(observation),
        turn=capture.snapshot.selected_root_turn,
        acting_player=InformationSetActor.SELF,
        opponent_range=opponent_range,
        public_features=public_features,
        updater_fingerprint=_fingerprint(
            {
                "model": range_model.fingerprint,
                "root": match_id,
                "orientation": "player_0",
                "candidate_truth_input": False,
            }
        ),
    )
    top_candidate_id = min(
        opponent_range.hypotheses,
        key=lambda item: (-item.weight, item.hypothesis_id),
    ).hypothesis_id.removeprefix("official-range:")
    candidate_by_id = {item.archetype: item for item in sources.candidates}
    predicted_candidate = candidate_by_id[top_candidate_id]
    batches = tuple(
        _build_branch_conditioned_root(
            capture,
            sources=sources,
            candidate=predicted_candidate,
            intent=intent,
            policy_variant="action_first",
            split="validation",
            replica=0,
            match_id=match_id,
            seed=seed,
            maximum_rollout_transitions=PHASE6_EXPECTED_VALUE_MAXIMUM_ROLLOUT_TRANSITIONS,
            maximum_tactical_transitions=PHASE6_EXPECTED_VALUE_MAXIMUM_TACTICAL_TRANSITIONS,
            experiment_label="Phase 6AM",
        )
        for seed in PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS
    )
    chance_law_fingerprint = _fingerprint(
        {
            "public_state": belief.public_state_fingerprint,
            "prefix_chain": capture.snapshot.prefix_chain_fingerprint,
            "target_and_sampling_chance_law": "same_opaque_official_root_law",
        }
    )
    result = PublicRangeCounterfactualRoot(
        true_candidate_id_supervision_only=true_candidate.archetype,
        true_intent_id_supervision_only=intent.value,
        public_belief=belief,
        candidate_summaries=summaries,
        common_chance_ratio=CommonChanceLawReachRatio(
            target_player=0,
            chance_law_fingerprint=chance_law_fingerprint,
            player_0_target_reach=1.0,
            player_1_target_reach=1.0,
            player_0_sampling_reach=1.0,
            player_1_sampling_reach=1.0,
        ),
        batch_seeds=PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS,
        batches=batches,
    )
    if result.root_id in set(reference.development_root_ids):
        raise ValueError("Phase 6AM root overlaps the development ledger")
    return result


@dataclass(frozen=True, slots=True)
class PublicRangeCounterfactualPanel:
    source_reference_fingerprint: str
    validation_seed: int
    match_namespace: str
    candidate_ids: tuple[str, ...]
    intent_ids: tuple[str, ...]
    roots: tuple[PublicRangeCounterfactualRoot, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.source_reference_fingerprint, "source_reference_fingerprint")
        expected = tuple(
            (candidate, intent)
            for candidate in self.candidate_ids
            for intent in self.intent_ids
        )
        actual = tuple(
            (item.true_candidate_id_supervision_only, item.true_intent_id_supervision_only)
            for item in self.roots
        )
        if (
            self.validation_seed != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_SEED
            or self.match_namespace != "phase6am-validation"
            or len(self.roots) != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_ROOTS
            or actual != expected
            or len({item.root_id for item in self.roots}) != len(self.roots)
        ):
            raise ValueError("Phase 6AM validation panel grid drifted")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    @property
    def official_match_count(self) -> int:
        return len(self.roots)

    @property
    def branch_observation_count(self) -> int:
        return sum(sum(len(batch.branch_observations) for batch in root.batches) for root in self.roots)

    @property
    def terminal_rollout_count(self) -> int:
        return sum(
            sum(batch.value_root.terminal_rollout_count for batch in root.batches)
            for root in self.roots
        )

    @property
    def native_transition_count(self) -> int:
        return sum(
            sum(
                surface.native_transition_count
                for batch in root.batches
                for surface in batch.value_root.surfaces
            )
            for root in self.roots
        )

    @property
    def live_selection_count(self) -> int:
        return sum(root.batches[0].value_root.match_selection_count for root in self.roots)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SCHEMA_VERSION,
            "source_reference_fingerprint": self.source_reference_fingerprint,
            "protocol": {
                "validation_seed": self.validation_seed,
                "match_namespace": self.match_namespace,
                "candidate_ids": list(self.candidate_ids),
                "intent_ids": list(self.intent_ids),
                "batch_seeds": list(PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS),
            },
            "roots": [item.to_dict() for item in self.roots],
            "counts": {
                "official_matches": self.official_match_count,
                "branch_observations": self.branch_observation_count,
                "terminal_rollouts": self.terminal_rollout_count,
                "native_transitions": self.native_transition_count,
                "live_selections": self.live_selection_count,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRangeCounterfactualPanel":
        protocol = raw["protocol"]
        if tuple(int(value) for value in protocol["batch_seeds"]) != (
            PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BATCH_SEEDS
        ):
            raise ValueError("Phase 6AM panel batch seeds drifted")
        result = cls(
            source_reference_fingerprint=str(raw["source_reference_fingerprint"]),
            validation_seed=int(protocol["validation_seed"]),
            match_namespace=str(protocol["match_namespace"]),
            candidate_ids=tuple(str(value) for value in protocol["candidate_ids"]),
            intent_ids=tuple(str(value) for value in protocol["intent_ids"]),
            roots=tuple(PublicRangeCounterfactualRoot.from_dict(item) for item in raw["roots"]),
        )
        expected_counts = {
            "official_matches": result.official_match_count,
            "branch_observations": result.branch_observation_count,
            "terminal_rollouts": result.terminal_rollout_count,
            "native_transitions": result.native_transition_count,
            "live_selections": result.live_selection_count,
        }
        if dict(raw["counts"]) != expected_counts:
            raise ValueError("Phase 6AM panel accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AM panel fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CounterfactualSliceMetric:
    dimension: str
    slice_id: str
    row_count: int
    coverage: float

    def __post_init__(self) -> None:
        if self.dimension not in {"action", "candidate", "intent"} or not self.slice_id:
            raise ValueError("Phase 6AM slice identity is invalid")
        if self.row_count <= 0 or not 0 <= self.coverage <= 1:
            raise ValueError("Phase 6AM slice metric is invalid")


def _range_ece(roots: Sequence[PublicRangeCounterfactualRoot]) -> float:
    bins = 10
    total = len(roots)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            root
            for root in roots
            if lower <= max(item.weight for item in root.public_belief.opponent_range.hypotheses)
            < upper
            or (
                index == bins - 1
                and math.isclose(
                    max(item.weight for item in root.public_belief.opponent_range.hypotheses),
                    1.0,
                    abs_tol=1e-12,
                )
            )
        ]
        if not members:
            continue
        confidence = statistics.mean(
            max(item.weight for item in root.public_belief.opponent_range.hypotheses)
            for root in members
        )
        accuracy = statistics.mean(
            root.top_candidate_id == root.true_candidate_id_supervision_only
            for root in members
        )
        error += len(members) / total * abs(confidence - accuracy)
    return error


def _coverage_bootstrap_lower(
    root_coverages: Sequence[float],
) -> float:
    rng = random.Random(PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BOOTSTRAP_SEED)
    values = []
    materialized = tuple(root_coverages)
    for _ in range(PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_BOOTSTRAP_RESAMPLES):
        values.append(
            statistics.mean(rng.choice(materialized) for _ in materialized) - 0.90
        )
    values.sort()
    return values[max(0, math.ceil(0.10 * len(values)) - 1)]


@dataclass(frozen=True, slots=True)
class PublicRangeCounterfactualEvaluation:
    source_reference_fingerprint: str
    validation_panel_fingerprint: str
    validation_root_ids: tuple[str, ...]
    overall_coverage: float
    mean_interval_width: float
    minimum_slice_coverage: float
    maximum_target_standard_error: float
    candidate_top1_accuracy: float
    candidate_ece: float
    minimum_candidate_top1_accuracy: float
    coverage_bootstrap_lower_bound: float
    slice_metrics: tuple[CounterfactualSliceMetric, ...]
    precision_gate_passed: bool
    coverage_gate_passed: bool
    slice_gate_passed: bool
    width_gate_passed: bool
    bootstrap_gate_passed: bool
    range_gate_passed: bool
    reach_gate_passed: bool
    collection_holdout_gate_passed: bool
    normalized_counterfactual_advantage_bounds_admitted: bool
    absolute_official_counterfactual_values_available: bool = False
    equilibrium_convergence_validated: bool = False
    safe_resolver_enabled: bool = False
    opponent_exploitation_enabled: bool = False
    official_strategy_control_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.source_reference_fingerprint, "source_reference_fingerprint")
        _validate_digest(self.validation_panel_fingerprint, "validation_panel_fingerprint")
        if len(self.validation_root_ids) != 36 or len(set(self.validation_root_ids)) != 36:
            raise ValueError("Phase 6AM evaluation root ledger is invalid")
        for name in (
            "overall_coverage",
            "mean_interval_width",
            "minimum_slice_coverage",
            "maximum_target_standard_error",
            "candidate_top1_accuracy",
            "candidate_ece",
            "minimum_candidate_top1_accuracy",
            "coverage_bootstrap_lower_bound",
        ):
            _finite(getattr(self, name), name)
        if (
            not 0 <= self.overall_coverage <= 1
            or not 0 <= self.mean_interval_width <= 4
            or not 0 <= self.minimum_slice_coverage <= 1
            or not 0 <= self.maximum_target_standard_error
            or not 0 <= self.candidate_top1_accuracy <= 1
            or not 0 <= self.candidate_ece <= 1
            or not 0 <= self.minimum_candidate_top1_accuracy <= 1
            or not -0.90 <= self.coverage_bootstrap_lower_bound <= 0.10
        ):
            raise ValueError("Phase 6AM evaluation metric is outside its support")
        dimension_counts = {
            dimension: sum(item.dimension == dimension for item in self.slice_metrics)
            for dimension in ("action", "candidate", "intent")
        }
        if dimension_counts != {"action": 5, "candidate": 6, "intent": 6}:
            raise ValueError("Phase 6AM evaluation slice grid drifted")
        if len({(item.dimension, item.slice_id) for item in self.slice_metrics}) != 17:
            raise ValueError("Phase 6AM evaluation slices are duplicated")
        if not math.isclose(
            self.minimum_slice_coverage,
            min(item.coverage for item in self.slice_metrics),
            abs_tol=1e-12,
        ):
            raise ValueError("Phase 6AM minimum slice coverage is inconsistent")
        expected_gate_flags = {
            "precision_gate_passed": (
                self.maximum_target_standard_error
                <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_TARGET_SE
            ),
            "coverage_gate_passed": (
                self.overall_coverage
                >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_COVERAGE
            ),
            "slice_gate_passed": (
                self.minimum_slice_coverage
                >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_SLICE_COVERAGE
            ),
            "width_gate_passed": (
                self.mean_interval_width
                <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_MEAN_WIDTH
            ),
            "bootstrap_gate_passed": self.coverage_bootstrap_lower_bound > 0,
            "range_gate_passed": (
                self.candidate_top1_accuracy
                >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_RANGE_TOP1
                and self.candidate_ece
                <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_RANGE_ECE
                and self.minimum_candidate_top1_accuracy
                >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_CANDIDATE_TOP1
            ),
        }
        if any(
            getattr(self, name) != expected_value
            for name, expected_value in expected_gate_flags.items()
        ):
            raise ValueError("Phase 6AM evaluation gate is inconsistent with its metric")
        expected = (
            self.precision_gate_passed
            and self.coverage_gate_passed
            and self.slice_gate_passed
            and self.width_gate_passed
            and self.bootstrap_gate_passed
            and self.range_gate_passed
            and self.reach_gate_passed
            and self.collection_holdout_gate_passed
        )
        if self.normalized_counterfactual_advantage_bounds_admitted != expected:
            raise ValueError("Phase 6AM admission flag is inconsistent")
        if any(
            (
                self.absolute_official_counterfactual_values_available,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AM promoted a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SCHEMA_VERSION,
            "source_reference_fingerprint": self.source_reference_fingerprint,
            "validation_panel_fingerprint": self.validation_panel_fingerprint,
            "validation_root_ids": list(self.validation_root_ids),
            "metrics": {
                "overall_coverage": float(self.overall_coverage),
                "mean_interval_width": float(self.mean_interval_width),
                "minimum_slice_coverage": float(self.minimum_slice_coverage),
                "maximum_target_standard_error": float(
                    self.maximum_target_standard_error
                ),
                "candidate_top1_accuracy": float(self.candidate_top1_accuracy),
                "candidate_ece": float(self.candidate_ece),
                "minimum_candidate_top1_accuracy": float(
                    self.minimum_candidate_top1_accuracy
                ),
                "coverage_bootstrap_lower_bound": float(
                    self.coverage_bootstrap_lower_bound
                ),
            },
            "slice_metrics": [
                {
                    "dimension": item.dimension,
                    "slice_id": item.slice_id,
                    "row_count": item.row_count,
                    "coverage": float(item.coverage),
                }
                for item in self.slice_metrics
            ],
            "gates": {
                "precision_gate_passed": self.precision_gate_passed,
                "coverage_gate_passed": self.coverage_gate_passed,
                "slice_gate_passed": self.slice_gate_passed,
                "width_gate_passed": self.width_gate_passed,
                "bootstrap_gate_passed": self.bootstrap_gate_passed,
                "range_gate_passed": self.range_gate_passed,
                "reach_gate_passed": self.reach_gate_passed,
                "collection_holdout_gate_passed": self.collection_holdout_gate_passed,
                "normalized_counterfactual_advantage_bounds_admitted": self.normalized_counterfactual_advantage_bounds_admitted,
                "absolute_official_counterfactual_values_available": self.absolute_official_counterfactual_values_available,
                "equilibrium_convergence_validated": self.equilibrium_convergence_validated,
                "safe_resolver_enabled": self.safe_resolver_enabled,
                "opponent_exploitation_enabled": self.opponent_exploitation_enabled,
                "official_strategy_control_enabled": self.official_strategy_control_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicRangeCounterfactualEvaluation":
        metrics = raw["metrics"]
        gates = raw["gates"]
        result = cls(
            source_reference_fingerprint=str(raw["source_reference_fingerprint"]),
            validation_panel_fingerprint=str(raw["validation_panel_fingerprint"]),
            validation_root_ids=tuple(str(value) for value in raw["validation_root_ids"]),
            overall_coverage=float(metrics["overall_coverage"]),
            mean_interval_width=float(metrics["mean_interval_width"]),
            minimum_slice_coverage=float(metrics["minimum_slice_coverage"]),
            maximum_target_standard_error=float(metrics["maximum_target_standard_error"]),
            candidate_top1_accuracy=float(metrics["candidate_top1_accuracy"]),
            candidate_ece=float(metrics["candidate_ece"]),
            minimum_candidate_top1_accuracy=float(metrics["minimum_candidate_top1_accuracy"]),
            coverage_bootstrap_lower_bound=float(metrics["coverage_bootstrap_lower_bound"]),
            slice_metrics=tuple(
                CounterfactualSliceMetric(
                    dimension=str(item["dimension"]),
                    slice_id=str(item["slice_id"]),
                    row_count=int(item["row_count"]),
                    coverage=float(item["coverage"]),
                )
                for item in raw["slice_metrics"]
            ),
            precision_gate_passed=bool(gates["precision_gate_passed"]),
            coverage_gate_passed=bool(gates["coverage_gate_passed"]),
            slice_gate_passed=bool(gates["slice_gate_passed"]),
            width_gate_passed=bool(gates["width_gate_passed"]),
            bootstrap_gate_passed=bool(gates["bootstrap_gate_passed"]),
            range_gate_passed=bool(gates["range_gate_passed"]),
            reach_gate_passed=bool(gates["reach_gate_passed"]),
            collection_holdout_gate_passed=bool(gates["collection_holdout_gate_passed"]),
            normalized_counterfactual_advantage_bounds_admitted=bool(
                gates["normalized_counterfactual_advantage_bounds_admitted"]
            ),
            absolute_official_counterfactual_values_available=bool(
                gates["absolute_official_counterfactual_values_available"]
            ),
            equilibrium_convergence_validated=bool(gates["equilibrium_convergence_validated"]),
            safe_resolver_enabled=bool(gates["safe_resolver_enabled"]),
            opponent_exploitation_enabled=bool(gates["opponent_exploitation_enabled"]),
            official_strategy_control_enabled=bool(gates["official_strategy_control_enabled"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AM evaluation fingerprint mismatch")
        return result


def evaluate_public_range_counterfactual_advantages(
    panel: PublicRangeCounterfactualPanel,
    reference: PublicRangeCounterfactualReference,
) -> PublicRangeCounterfactualEvaluation:
    if panel.source_reference_fingerprint != reference.fingerprint:
        raise ValueError("Phase 6AM panel/reference binding failed")
    if set(item.root_id for item in panel.roots) & set(reference.development_root_ids):
        raise ValueError("Phase 6AM validation reused a development root")
    rows = []
    for root in panel.roots:
        for target in root.targets(reference.advantage_radius):
            rows.append((root, target))
    coverage = statistics.mean(target.covered for _, target in rows)
    width = statistics.mean(target.width for _, target in rows)
    maximum_se = max(target.target_standard_error for _, target in rows)
    slices = []
    dimensions = (
        (
            "action",
            tuple(
                item.action_id
                for item in panel.roots[0].targets(reference.advantage_radius)
            ),
            lambda root, target: target.action_id,
        ),
        (
            "candidate",
            panel.candidate_ids,
            lambda root, target: root.true_candidate_id_supervision_only,
        ),
        (
            "intent",
            panel.intent_ids,
            lambda root, target: root.true_intent_id_supervision_only,
        ),
    )
    for dimension, support, selector in dimensions:
        for slice_id in support:
            selected = [(root, target) for root, target in rows if selector(root, target) == slice_id]
            slices.append(
                CounterfactualSliceMetric(
                    dimension=dimension,
                    slice_id=slice_id,
                    row_count=len(selected),
                    coverage=statistics.mean(target.covered for _, target in selected),
                )
            )
    minimum_slice = min(item.coverage for item in slices)
    top1 = statistics.mean(
        item.top_candidate_id == item.true_candidate_id_supervision_only
        for item in panel.roots
    )
    candidate_top1 = []
    for candidate in panel.candidate_ids:
        selected = [item for item in panel.roots if item.true_candidate_id_supervision_only == candidate]
        candidate_top1.append(
            statistics.mean(item.top_candidate_id == candidate for item in selected)
        )
    ece = _range_ece(panel.roots)
    root_coverages = tuple(
        statistics.mean(target.covered for target in root.targets(reference.advantage_radius))
        for root in panel.roots
    )
    bootstrap_lower = _coverage_bootstrap_lower(root_coverages)
    precision_gate = maximum_se <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_TARGET_SE
    coverage_gate = coverage >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_COVERAGE
    slice_gate = minimum_slice >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_SLICE_COVERAGE
    width_gate = width <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_MEAN_WIDTH
    bootstrap_gate = bootstrap_lower > 0
    range_gate = (
        top1 >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_RANGE_TOP1
        and ece <= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MAXIMUM_RANGE_ECE
        and min(candidate_top1)
        >= PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_MINIMUM_CANDIDATE_TOP1
    )
    reach_gate = all(
        math.isclose(
            item.common_chance_ratio.counterfactual_likelihood_ratio,
            1.0,
            abs_tol=1e-12,
        )
        and not item.common_chance_ratio.absolute_chance_reach_available
        for item in panel.roots
    )
    holdout_gate = (
        len(set(item.root_id for item in panel.roots)) == 36
        and not set(item.root_id for item in panel.roots).intersection(
            reference.development_root_ids
        )
    )
    admitted = all(
        (
            precision_gate,
            coverage_gate,
            slice_gate,
            width_gate,
            bootstrap_gate,
            range_gate,
            reach_gate,
            holdout_gate,
        )
    )
    return PublicRangeCounterfactualEvaluation(
        source_reference_fingerprint=reference.fingerprint,
        validation_panel_fingerprint=panel.fingerprint,
        validation_root_ids=tuple(item.root_id for item in panel.roots),
        overall_coverage=coverage,
        mean_interval_width=width,
        minimum_slice_coverage=minimum_slice,
        maximum_target_standard_error=maximum_se,
        candidate_top1_accuracy=top1,
        candidate_ece=ece,
        minimum_candidate_top1_accuracy=min(candidate_top1),
        coverage_bootstrap_lower_bound=bootstrap_lower,
        slice_metrics=tuple(slices),
        precision_gate_passed=precision_gate,
        coverage_gate_passed=coverage_gate,
        slice_gate_passed=slice_gate,
        width_gate_passed=width_gate,
        bootstrap_gate_passed=bootstrap_gate,
        range_gate_passed=range_gate,
        reach_gate_passed=reach_gate,
        collection_holdout_gate_passed=holdout_gate,
        normalized_counterfactual_advantage_bounds_admitted=admitted,
    )


@dataclass(frozen=True, slots=True)
class HypothesisActionAdvantageBound:
    hypothesis_id: str
    action_id: str
    lower: float
    upper: float
    point_center: float
    calibrated: bool

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.action_id:
            raise ValueError("Phase 6AM hypothesis/action bound identity is empty")
        for name in ("lower", "upper", "point_center"):
            _finite(getattr(self, name), name)
        if not -2 <= self.lower <= self.point_center <= self.upper <= 2:
            raise ValueError("Phase 6AM hypothesis/action bound is invalid")
        if not self.calibrated and (
            self.lower != -2.0 or self.upper != 2.0 or self.point_center != 0.0
        ):
            raise ValueError("Phase 6AM unsupported hypothesis bound is not vacuous")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "action_id": self.action_id,
            "lower": self.lower,
            "upper": self.upper,
            "point_center": self.point_center,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "HypothesisActionAdvantageBound":
        return cls(
            hypothesis_id=str(raw["hypothesis_id"]),
            action_id=str(raw["action_id"]),
            lower=float(raw["lower"]),
            upper=float(raw["upper"]),
            point_center=float(raw["point_center"]),
            calibrated=bool(raw["calibrated"]),
        )


@dataclass(frozen=True, slots=True)
class PublicRangeCounterfactualAdvantageBounds:
    public_belief_fingerprint: str
    baseline_strategy_fingerprint: str
    confidence: float
    bounds: tuple[HypothesisActionAdvantageBound, ...]
    absolute_chance_reach_available: bool = False
    safe_resolver_enabled: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.public_belief_fingerprint, "public_belief_fingerprint")
        _validate_digest(self.baseline_strategy_fingerprint, "baseline_strategy_fingerprint")
        if not math.isclose(
            self.confidence,
            PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_COVERAGE,
            abs_tol=1e-12,
        ) or not self.bounds:
            raise ValueError("Phase 6AM bound confidence/support is invalid")
        identities = {(item.hypothesis_id, item.action_id) for item in self.bounds}
        if len(identities) != len(self.bounds):
            raise ValueError("Phase 6AM hypothesis/action bounds are duplicated")
        hypotheses = {item.hypothesis_id for item in self.bounds}
        actions = {item.action_id for item in self.bounds}
        calibrated_hypotheses = {
            item.hypothesis_id for item in self.bounds if item.calibrated
        }
        if (
            len(self.bounds) != 30
            or len(hypotheses) != 6
            or len(actions) != 5
            or len(calibrated_hypotheses) != 1
            or sum(item.calibrated for item in self.bounds) != 5
        ):
            raise ValueError("Phase 6AM hypothesis/action bound grid drifted")
        if self.absolute_chance_reach_available or self.safe_resolver_enabled:
            raise ValueError("Phase 6AM bound provider promoted a downstream claim")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SCHEMA_VERSION,
            "public_belief_fingerprint": self.public_belief_fingerprint,
            "baseline_strategy_fingerprint": self.baseline_strategy_fingerprint,
            "confidence": self.confidence,
            "bounds": [item.to_dict() for item in self.bounds],
            "claims": {
                "absolute_chance_reach_available": self.absolute_chance_reach_available,
                "safe_resolver_enabled": self.safe_resolver_enabled,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
    ) -> "PublicRangeCounterfactualAdvantageBounds":
        claims = raw["claims"]
        result = cls(
            public_belief_fingerprint=str(raw["public_belief_fingerprint"]),
            baseline_strategy_fingerprint=str(raw["baseline_strategy_fingerprint"]),
            confidence=float(raw["confidence"]),
            bounds=tuple(
                HypothesisActionAdvantageBound.from_dict(item)
                for item in raw["bounds"]
            ),
            absolute_chance_reach_available=bool(
                claims["absolute_chance_reach_available"]
            ),
            safe_resolver_enabled=bool(claims["safe_resolver_enabled"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AM advantage-bound fingerprint mismatch")
        return result


def counterfactual_advantage_bounds_for_root(
    root: PublicRangeCounterfactualRoot,
    reference: PublicRangeCounterfactualReference,
) -> PublicRangeCounterfactualAdvantageBounds:
    targets = root.targets(reference.advantage_radius)
    top_hypothesis = f"official-range:{root.top_candidate_id}"
    result = []
    for hypothesis in root.public_belief.opponent_range.hypotheses:
        for target in targets:
            calibrated = hypothesis.hypothesis_id == top_hypothesis
            result.append(
                HypothesisActionAdvantageBound(
                    hypothesis_id=hypothesis.hypothesis_id,
                    action_id=target.action_id,
                    lower=target.lower if calibrated else -2.0,
                    upper=target.upper if calibrated else 2.0,
                    point_center=target.point_center if calibrated else 0.0,
                    calibrated=calibrated,
                )
            )
    baseline_fingerprint = _fingerprint(root.batches[0].value_root.blueprint_probabilities)
    return PublicRangeCounterfactualAdvantageBounds(
        public_belief_fingerprint=root.public_belief.fingerprint,
        baseline_strategy_fingerprint=baseline_fingerprint,
        confidence=PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_COVERAGE,
        bounds=tuple(result),
    )


@dataclass(frozen=True, slots=True)
class Phase6PublicRangeCounterfactualEvidence:
    reference_fingerprint: str
    panel_fingerprint: str
    evaluation_fingerprint: str
    status: str
    official_match_count: int
    branch_observation_count: int
    terminal_rollout_count: int
    native_transition_count: int
    live_selection_count: int
    overall_coverage: float
    mean_interval_width: float
    minimum_slice_coverage: float
    maximum_target_standard_error: float
    candidate_top1_accuracy: float
    candidate_ece: float
    coverage_bootstrap_lower_bound: float
    normalized_counterfactual_advantage_bounds_admitted: bool
    absolute_official_counterfactual_values_available: bool
    equilibrium_convergence_validated: bool
    safe_resolver_enabled: bool
    opponent_exploitation_enabled: bool
    official_strategy_control_enabled: bool

    def __post_init__(self) -> None:
        for name in (
            "reference_fingerprint",
            "panel_fingerprint",
            "evaluation_fingerprint",
        ):
            _validate_digest(getattr(self, name), name)
        for name in (
            "overall_coverage",
            "mean_interval_width",
            "minimum_slice_coverage",
            "maximum_target_standard_error",
            "candidate_top1_accuracy",
            "candidate_ece",
            "coverage_bootstrap_lower_bound",
        ):
            _finite(getattr(self, name), name)
        expected_status = (
            "normalized_counterfactual_advantage_bounds_admitted_shadow_only"
            if self.normalized_counterfactual_advantage_bounds_admitted
            else "normalized_counterfactual_advantage_bounds_rejected_fail_closed"
        )
        if (
            self.status != expected_status
            or self.official_match_count != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_ROOTS
            or self.branch_observation_count != 720
            or self.terminal_rollout_count != 5_760
            or self.native_transition_count < self.terminal_rollout_count
            or self.live_selection_count <= 0
            or not 0 <= self.overall_coverage <= 1
            or not 0 <= self.mean_interval_width <= 4
            or not 0 <= self.minimum_slice_coverage <= 1
            or self.maximum_target_standard_error < 0
            or not 0 <= self.candidate_top1_accuracy <= 1
            or not 0 <= self.candidate_ece <= 1
            or not -0.90 <= self.coverage_bootstrap_lower_bound <= 0.10
        ):
            raise ValueError("Phase 6AM evidence contract drifted")
        if any(
            (
                self.absolute_official_counterfactual_values_available,
                self.equilibrium_convergence_validated,
                self.safe_resolver_enabled,
                self.opponent_exploitation_enabled,
                self.official_strategy_control_enabled,
            )
        ):
            raise ValueError("Phase 6AM evidence promoted a downstream capability")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_SCHEMA_VERSION,
            "reference_fingerprint": self.reference_fingerprint,
            "panel_fingerprint": self.panel_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "status": self.status,
            "official_match_count": self.official_match_count,
            "branch_observation_count": self.branch_observation_count,
            "terminal_rollout_count": self.terminal_rollout_count,
            "native_transition_count": self.native_transition_count,
            "live_selection_count": self.live_selection_count,
            "overall_coverage": float(self.overall_coverage),
            "mean_interval_width": float(self.mean_interval_width),
            "minimum_slice_coverage": float(self.minimum_slice_coverage),
            "maximum_target_standard_error": float(
                self.maximum_target_standard_error
            ),
            "candidate_top1_accuracy": float(self.candidate_top1_accuracy),
            "candidate_ece": float(self.candidate_ece),
            "coverage_bootstrap_lower_bound": float(
                self.coverage_bootstrap_lower_bound
            ),
            "normalized_counterfactual_advantage_bounds_admitted": self.normalized_counterfactual_advantage_bounds_admitted,
            "absolute_official_counterfactual_values_available": self.absolute_official_counterfactual_values_available,
            "equilibrium_convergence_validated": self.equilibrium_convergence_validated,
            "safe_resolver_enabled": self.safe_resolver_enabled,
            "opponent_exploitation_enabled": self.opponent_exploitation_enabled,
            "official_strategy_control_enabled": self.official_strategy_control_enabled,
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    def to_markdown(self) -> str:
        return f"""# Phase 6AM public-range counterfactual-advantage evidence

Status: `{self.status}`

| Metric | Result |
|---|---:|
| Fresh official matches | {self.official_match_count} |
| Immediate native branches | {self.branch_observation_count} |
| Terminal native continuations | {self.terminal_rollout_count} |
| Maximum target SE | `{self.maximum_target_standard_error:.6f}` |
| Advantage coverage / worst slice | `{self.overall_coverage:.6f}` / `{self.minimum_slice_coverage:.6f}` |
| Mean interval width | `{self.mean_interval_width:.6f}` |
| Candidate top-1 / ECE | `{self.candidate_top1_accuracy:.6f}` / `{self.candidate_ece:.6f}` |
| Coverage bootstrap lower bound over 0.90 | `{self.coverage_bootstrap_lower_bound:.6f}` |
| Normalized counterfactual-advantage bounds admitted | **{str(self.normalized_counterfactual_advantage_bounds_admitted).lower()}** |
| Absolute values / equilibrium / resolver / exploitation / control | **false / false / false / false / false** |

Matches, legal selections, retained snapshots, native branches, and descendants
are genuine official-engine execution. The range, allocation law, five-plan
game, blueprint, branch heuristic, normalization, bootstrap, and intervals are
research abstractions. No Kaggle submission or strength claim is made.
"""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Phase6PublicRangeCounterfactualEvidence":
        result = cls(
            reference_fingerprint=str(raw["reference_fingerprint"]),
            panel_fingerprint=str(raw["panel_fingerprint"]),
            evaluation_fingerprint=str(raw["evaluation_fingerprint"]),
            status=str(raw["status"]),
            official_match_count=int(raw["official_match_count"]),
            branch_observation_count=int(raw["branch_observation_count"]),
            terminal_rollout_count=int(raw["terminal_rollout_count"]),
            native_transition_count=int(raw["native_transition_count"]),
            live_selection_count=int(raw["live_selection_count"]),
            overall_coverage=float(raw["overall_coverage"]),
            mean_interval_width=float(raw["mean_interval_width"]),
            minimum_slice_coverage=float(raw["minimum_slice_coverage"]),
            maximum_target_standard_error=float(raw["maximum_target_standard_error"]),
            candidate_top1_accuracy=float(raw["candidate_top1_accuracy"]),
            candidate_ece=float(raw["candidate_ece"]),
            coverage_bootstrap_lower_bound=float(raw["coverage_bootstrap_lower_bound"]),
            normalized_counterfactual_advantage_bounds_admitted=bool(
                raw["normalized_counterfactual_advantage_bounds_admitted"]
            ),
            absolute_official_counterfactual_values_available=bool(
                raw["absolute_official_counterfactual_values_available"]
            ),
            equilibrium_convergence_validated=bool(
                raw["equilibrium_convergence_validated"]
            ),
            safe_resolver_enabled=bool(raw["safe_resolver_enabled"]),
            opponent_exploitation_enabled=bool(raw["opponent_exploitation_enabled"]),
            official_strategy_control_enabled=bool(raw["official_strategy_control_enabled"]),
        )
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("Phase 6AM evidence fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class Phase6PublicRangeCounterfactualArtifacts:
    reference: PublicRangeCounterfactualReference
    panel: PublicRangeCounterfactualPanel
    evaluation: PublicRangeCounterfactualEvaluation
    evidence: Phase6PublicRangeCounterfactualEvidence


def run_phase6_public_range_counterfactual_advantages(
    *,
    reference: PublicRangeCounterfactualReference,
    panel: PublicRangeCounterfactualPanel,
) -> Phase6PublicRangeCounterfactualArtifacts:
    evaluation = evaluate_public_range_counterfactual_advantages(panel, reference)
    status = (
        "normalized_counterfactual_advantage_bounds_admitted_shadow_only"
        if evaluation.normalized_counterfactual_advantage_bounds_admitted
        else "normalized_counterfactual_advantage_bounds_rejected_fail_closed"
    )
    evidence = Phase6PublicRangeCounterfactualEvidence(
        reference_fingerprint=reference.fingerprint,
        panel_fingerprint=panel.fingerprint,
        evaluation_fingerprint=evaluation.fingerprint,
        status=status,
        official_match_count=panel.official_match_count,
        branch_observation_count=panel.branch_observation_count,
        terminal_rollout_count=panel.terminal_rollout_count,
        native_transition_count=panel.native_transition_count,
        live_selection_count=panel.live_selection_count,
        overall_coverage=evaluation.overall_coverage,
        mean_interval_width=evaluation.mean_interval_width,
        minimum_slice_coverage=evaluation.minimum_slice_coverage,
        maximum_target_standard_error=evaluation.maximum_target_standard_error,
        candidate_top1_accuracy=evaluation.candidate_top1_accuracy,
        candidate_ece=evaluation.candidate_ece,
        coverage_bootstrap_lower_bound=evaluation.coverage_bootstrap_lower_bound,
        normalized_counterfactual_advantage_bounds_admitted=(
            evaluation.normalized_counterfactual_advantage_bounds_admitted
        ),
        absolute_official_counterfactual_values_available=False,
        equilibrium_convergence_validated=False,
        safe_resolver_enabled=False,
        opponent_exploitation_enabled=False,
        official_strategy_control_enabled=False,
    )
    return Phase6PublicRangeCounterfactualArtifacts(reference, panel, evaluation, evidence)


def _load_root_shards(directory: Path) -> tuple[PublicRangeCounterfactualRoot, ...]:
    if not directory.exists():
        return ()
    return tuple(
        PublicRangeCounterfactualRoot.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in sorted(directory.glob("*.json"))
    )


def main() -> None:
    import argparse

    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare-reference", "evaluate"), required=True)
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=project_root / "artifacts" / "phase6_public_range_counterfactual_advantages",
    )
    parser.add_argument(
        "--validation-shard-directory",
        type=Path,
        default=project_root / "work" / "phase6am_validation_shards",
    )
    args = parser.parse_args()
    target = args.artifact_directory.resolve()
    reference_path = target / "counterfactual_advantage_reference.json"
    panel_path = target / "validation_panel.json"
    if args.stage == "prepare-reference":
        if _load_root_shards(args.validation_shard_directory) or panel_path.exists():
            raise RuntimeError("Phase 6AM reference must be frozen before collection")
        reference = prepare_public_range_counterfactual_reference(project_root)
        _write_artifact(reference_path, reference.to_json() + "\n")
        print(
            json.dumps(
                {
                    "status": "phase6am_reference_frozen_before_collection",
                    "reference_fingerprint": reference.fingerprint,
                    "development_root_count": len(reference.development_root_ids),
                    "calibration_score_count": reference.calibration_score_count,
                    "advantage_radius": reference.advantage_radius,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not reference_path.is_file():
        raise FileNotFoundError("Phase 6AM frozen reference is missing")
    reference = PublicRangeCounterfactualReference.from_dict(
        json.loads(reference_path.read_text(encoding="utf-8"))
    )
    shards = _load_root_shards(args.validation_shard_directory)
    if len(shards) != PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_ROOTS:
        raise RuntimeError("Phase 6AM requires exactly 36 validation roots")
    sources = _load_expected_value_sources(project_root)
    candidate_ids = tuple(item.archetype for item in sources.candidates)
    intent_ids = tuple(item.value for item in OPPONENT_INTENTS)
    by_key = {
        (item.true_candidate_id_supervision_only, item.true_intent_id_supervision_only): item
        for item in shards
    }
    expected = tuple((candidate, intent) for candidate in candidate_ids for intent in intent_ids)
    if len(by_key) != len(shards) or set(by_key) != set(expected):
        raise ValueError("Phase 6AM shard grid is incomplete or duplicated")
    panel = PublicRangeCounterfactualPanel(
        source_reference_fingerprint=reference.fingerprint,
        validation_seed=PHASE6_PUBLIC_RANGE_COUNTERFACTUAL_VALIDATION_SEED,
        match_namespace="phase6am-validation",
        candidate_ids=candidate_ids,
        intent_ids=intent_ids,
        roots=tuple(by_key[key] for key in expected),
    )
    artifacts = run_phase6_public_range_counterfactual_advantages(
        reference=reference,
        panel=panel,
    )
    _write_artifact(panel_path, panel.to_json() + "\n")
    _write_artifact(
        target / "counterfactual_advantage_evaluation.json",
        artifacts.evaluation.to_json() + "\n",
    )
    _write_artifact(target / "phase6am_evidence.json", artifacts.evidence.to_json() + "\n")
    _write_artifact(target / "phase6am_evidence.md", artifacts.evidence.to_markdown())
    print(artifacts.evidence.to_json())


if __name__ == "__main__":
    main()
