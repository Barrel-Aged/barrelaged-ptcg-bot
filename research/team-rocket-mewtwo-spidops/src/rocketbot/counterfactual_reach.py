"""Reach accounting and exact expectation audits for counterfactual targets.

Counterfactual values exclude the acting player's reach to an information set.
This module makes that convention executable and auditable.  The generic reach
trace stores only hashed choice identities and probabilities; it can therefore
be attached to native research data without persisting private card IDs.

The exact audit uses :class:`~rocketbot.exact_game.RocketPrizeDuel`.  Passing it
proves the estimator algebra on that enumerable abstraction only.  It does not
make a native hidden-state prior official, calibrate a full-game value model,
or authorize safe resolving.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
from typing import Any, Mapping

from .exact_game import (
    DuelState,
    RocketPrizeDuel,
    TabularStrategyProfile,
    evaluate_profile_from_state,
)
from .game_theory import InformationSetActor


COUNTERFACTUAL_REACH_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _probability(value: float, name: str, *, positive: bool) -> float:
    result = float(value)
    lower_ok = result > 0 if positive else result >= 0
    if not lower_ok or result > 1 or not math.isfinite(result):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a {qualifier}probability")
    return result


def counterfactual_importance_weight(
    *,
    target_player: int,
    chance_reach: float,
    player_0_reach: float,
    player_1_reach: float,
    sampling_reach: float,
) -> float:
    """Return ``chance * opponent reach / sampling reach``.

    The target player's own pre-information-set reach is deliberately absent
    from the numerator.  This is the importance ratio whose expectation gives
    an unnormalised counterfactual value or reach mass.
    """

    if target_player not in (0, 1):
        raise ValueError("counterfactual target player must be 0 or 1")
    chance = _probability(chance_reach, "chance_reach", positive=False)
    reach_0 = _probability(player_0_reach, "player_0_reach", positive=False)
    reach_1 = _probability(player_1_reach, "player_1_reach", positive=False)
    sample = _probability(sampling_reach, "sampling_reach", positive=True)
    opponent_reach = reach_1 if target_player == 0 else reach_0
    result = chance * opponent_reach / sample
    if result < 0 or not math.isfinite(result):
        raise ValueError("counterfactual importance weight is invalid")
    return result


def common_chance_law_counterfactual_ratio(
    *,
    target_player: int,
    player_0_target_reach: float,
    player_1_target_reach: float,
    player_0_sampling_reach: float,
    player_1_sampling_reach: float,
) -> float:
    """Return the counterfactual likelihood ratio after common chance cancels.

    This function is valid only when target and sampling use the same chance
    law pointwise.  It does not calculate that law's absolute probability.  If
    ``q_c == pi_c``, then

    ``pi_c * pi_-i / (q_c * q_0 * q_1)``

    reduces to ``pi_-i / (q_0 * q_1)``.  The result can weight an on-policy
    sampled contribution without pretending that an opaque engine disclosed
    the absolute chance reach.  It is not itself a counterfactual value, reach
    mass, calibration result, or safety bound.
    """

    if target_player not in (0, 1):
        raise ValueError("counterfactual target player must be 0 or 1")
    target_0 = _probability(
        player_0_target_reach,
        "player_0_target_reach",
        positive=False,
    )
    target_1 = _probability(
        player_1_target_reach,
        "player_1_target_reach",
        positive=False,
    )
    sample_0 = _probability(
        player_0_sampling_reach,
        "player_0_sampling_reach",
        positive=True,
    )
    sample_1 = _probability(
        player_1_sampling_reach,
        "player_1_sampling_reach",
        positive=True,
    )
    opponent_target = target_1 if target_player == 0 else target_0
    result = opponent_target / (sample_0 * sample_1)
    if result < 0 or not math.isfinite(result):
        raise ValueError("common-chance counterfactual ratio is invalid")
    return result


@dataclass(frozen=True, slots=True)
class CommonChanceLawReachRatio:
    """Auditable symbolic cancellation for an opaque shared chance law."""

    target_player: int
    chance_law_fingerprint: str
    player_0_target_reach: float
    player_1_target_reach: float
    player_0_sampling_reach: float
    player_1_sampling_reach: float
    target_and_sampling_chance_law_identical: bool = True
    absolute_chance_reach_available: bool = False

    def __post_init__(self) -> None:
        if self.target_player not in (0, 1):
            raise ValueError("common-chance ratio target player must be 0 or 1")
        _validate_digest(self.chance_law_fingerprint, "chance_law_fingerprint")
        _probability(
            self.player_0_target_reach,
            "player_0_target_reach",
            positive=False,
        )
        _probability(
            self.player_1_target_reach,
            "player_1_target_reach",
            positive=False,
        )
        _probability(
            self.player_0_sampling_reach,
            "player_0_sampling_reach",
            positive=True,
        )
        _probability(
            self.player_1_sampling_reach,
            "player_1_sampling_reach",
            positive=True,
        )
        if not self.target_and_sampling_chance_law_identical:
            raise ValueError("common-chance cancellation requires identical laws")
        if self.absolute_chance_reach_available:
            raise ValueError("opaque common-chance ratio cannot claim absolute reach")

    @property
    def counterfactual_likelihood_ratio(self) -> float:
        return common_chance_law_counterfactual_ratio(
            target_player=self.target_player,
            player_0_target_reach=self.player_0_target_reach,
            player_1_target_reach=self.player_1_target_reach,
            player_0_sampling_reach=self.player_0_sampling_reach,
            player_1_sampling_reach=self.player_1_sampling_reach,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": COUNTERFACTUAL_REACH_SCHEMA_VERSION,
            "target_player": self.target_player,
            "chance_law_fingerprint": self.chance_law_fingerprint,
            "target_reach": {
                "player_0": self.player_0_target_reach,
                "player_1": self.player_1_target_reach,
            },
            "sampling_reach": {
                "player_0": self.player_0_sampling_reach,
                "player_1": self.player_1_sampling_reach,
            },
            "derived": {
                "counterfactual_likelihood_ratio": (
                    self.counterfactual_likelihood_ratio
                ),
                "absolute_chance_reach": None,
            },
            "claims": {
                "target_and_sampling_chance_law_identical": (
                    self.target_and_sampling_chance_law_identical
                ),
                "absolute_chance_reach_available": (
                    self.absolute_chance_reach_available
                ),
                "target_player_self_reach_excluded": True,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CommonChanceLawReachRatio":
        target = raw["target_reach"]
        sampling = raw["sampling_reach"]
        derived = raw["derived"]
        claims = raw["claims"]
        if derived["absolute_chance_reach"] is not None:
            raise ValueError("common-chance record invented absolute reach")
        if not bool(claims["target_player_self_reach_excluded"]):
            raise ValueError("common-chance record included target-player self reach")
        result = cls(
            target_player=int(raw["target_player"]),
            chance_law_fingerprint=str(raw["chance_law_fingerprint"]),
            player_0_target_reach=float(target["player_0"]),
            player_1_target_reach=float(target["player_1"]),
            player_0_sampling_reach=float(sampling["player_0"]),
            player_1_sampling_reach=float(sampling["player_1"]),
            target_and_sampling_chance_law_identical=bool(
                claims["target_and_sampling_chance_law_identical"]
            ),
            absolute_chance_reach_available=bool(
                claims["absolute_chance_reach_available"]
            ),
        )
        if not math.isclose(
            float(derived["counterfactual_likelihood_ratio"]),
            result.counterfactual_likelihood_ratio,
            abs_tol=1e-12,
        ):
            raise ValueError("common-chance derived ratio mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("common-chance ratio fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class ReachStep:
    """One redacted chance or player choice on a sampled prefix."""

    step_index: int
    actor: str
    selected_identifier_fingerprint: str
    target_probability: float
    sampling_probability: float

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("reach step index cannot be negative")
        if self.actor not in {"chance", "player_0", "player_1"}:
            raise ValueError("reach step actor is invalid")
        _validate_digest(
            self.selected_identifier_fingerprint,
            "selected_identifier_fingerprint",
        )
        _probability(self.target_probability, "target_probability", positive=False)
        _probability(self.sampling_probability, "sampling_probability", positive=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReachStep":
        return cls(
            step_index=int(raw["step_index"]),
            actor=str(raw["actor"]),
            selected_identifier_fingerprint=str(
                raw["selected_identifier_fingerprint"]
            ),
            target_probability=float(raw["target_probability"]),
            sampling_probability=float(raw["sampling_probability"]),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualReachTrace:
    """Root-relative reach ledger ending before the target player's decision."""

    target_player: int
    root_state_fingerprint: str
    decision_state_fingerprint: str
    steps: tuple[ReachStep, ...]
    target_and_sampling_strategy_match: bool
    source_descendants_official_engine_executed: bool
    upstream_official_reach_complete: bool = False
    raw_choice_identifiers_persisted: int = 0

    def __post_init__(self) -> None:
        if self.target_player not in (0, 1):
            raise ValueError("reach trace target player must be 0 or 1")
        _validate_digest(self.root_state_fingerprint, "root_state_fingerprint")
        _validate_digest(self.decision_state_fingerprint, "decision_state_fingerprint")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("reach trace step indices must be contiguous")
        target_actor = f"player_{self.target_player}"
        if any(step.actor == target_actor for step in self.steps):
            raise ValueError(
                "first-decision reach trace cannot contain a target-player action"
            )
        observed_match = all(
            math.isclose(
                step.target_probability,
                step.sampling_probability,
                abs_tol=1e-12,
            )
            for step in self.steps
        )
        if self.target_and_sampling_strategy_match != observed_match:
            raise ValueError("reach trace target/sampling match flag is inconsistent")
        if self.raw_choice_identifiers_persisted != 0:
            raise ValueError("reach trace must not persist raw choice identifiers")
        if (
            self.upstream_official_reach_complete
            and not self.source_descendants_official_engine_executed
        ):
            raise ValueError(
                "an abstract trace cannot claim complete official upstream reach"
            )
        if self.sampling_reach <= 0:
            raise ValueError("reach trace has zero sampling reach")

    @property
    def chance_reach(self) -> float:
        return math.prod(
            step.target_probability for step in self.steps if step.actor == "chance"
        )

    def player_reach(self, player: int) -> float:
        if player not in (0, 1):
            raise ValueError("reach player must be 0 or 1")
        return math.prod(
            step.target_probability
            for step in self.steps
            if step.actor == f"player_{player}"
        )

    @property
    def player_0_reach(self) -> float:
        return self.player_reach(0)

    @property
    def player_1_reach(self) -> float:
        return self.player_reach(1)

    @property
    def sampling_reach(self) -> float:
        return math.prod(step.sampling_probability for step in self.steps)

    @property
    def counterfactual_reach(self) -> float:
        opponent_reach = (
            self.player_1_reach if self.target_player == 0 else self.player_0_reach
        )
        return self.chance_reach * opponent_reach

    @property
    def importance_weight(self) -> float:
        return counterfactual_importance_weight(
            target_player=self.target_player,
            chance_reach=self.chance_reach,
            player_0_reach=self.player_0_reach,
            player_1_reach=self.player_1_reach,
            sampling_reach=self.sampling_reach,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": COUNTERFACTUAL_REACH_SCHEMA_VERSION,
            "target_player": self.target_player,
            "root_state_fingerprint": self.root_state_fingerprint,
            "decision_state_fingerprint": self.decision_state_fingerprint,
            "steps": [step.to_dict() for step in self.steps],
            "derived_reach": {
                "chance": self.chance_reach,
                "player_0": self.player_0_reach,
                "player_1": self.player_1_reach,
                "sampling": self.sampling_reach,
                "counterfactual": self.counterfactual_reach,
                "importance_weight": self.importance_weight,
            },
            "claims": {
                "first_target_player_decision": True,
                "target_and_sampling_strategy_match": (
                    self.target_and_sampling_strategy_match
                ),
                "source_descendants_official_engine_executed": (
                    self.source_descendants_official_engine_executed
                ),
                "root_relative": True,
                "upstream_official_reach_complete": (
                    self.upstream_official_reach_complete
                ),
                "raw_choice_identifiers_persisted": (
                    self.raw_choice_identifiers_persisted
                ),
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CounterfactualReachTrace":
        claims = raw["claims"]
        result = cls(
            target_player=int(raw["target_player"]),
            root_state_fingerprint=str(raw["root_state_fingerprint"]),
            decision_state_fingerprint=str(raw["decision_state_fingerprint"]),
            steps=tuple(ReachStep.from_dict(item) for item in raw["steps"]),
            target_and_sampling_strategy_match=bool(
                claims["target_and_sampling_strategy_match"]
            ),
            source_descendants_official_engine_executed=bool(
                claims["source_descendants_official_engine_executed"]
            ),
            upstream_official_reach_complete=bool(
                claims["upstream_official_reach_complete"]
            ),
            raw_choice_identifiers_persisted=int(
                claims["raw_choice_identifiers_persisted"]
            ),
        )
        if not bool(claims["first_target_player_decision"]):
            raise ValueError("reach trace lost first-decision semantics")
        if not bool(claims["root_relative"]):
            raise ValueError("reach trace lost root-relative semantics")
        derived = raw["derived_reach"]
        expected = {
            "chance": result.chance_reach,
            "player_0": result.player_0_reach,
            "player_1": result.player_1_reach,
            "sampling": result.sampling_reach,
            "counterfactual": result.counterfactual_reach,
            "importance_weight": result.importance_weight,
        }
        if any(
            not math.isclose(float(derived[name]), value, abs_tol=1e-12)
            for name, value in expected.items()
        ):
            raise ValueError("reach trace derived accounting mismatch")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("counterfactual reach trace fingerprint mismatch")
        return result


def _sample_identifier(
    probabilities: tuple[tuple[str, float], ...],
    rng: random.Random,
) -> str:
    if not probabilities:
        raise ValueError("cannot sample an empty reach distribution")
    total = sum(value for _, value in probabilities)
    if not math.isclose(total, 1.0, abs_tol=1e-8):
        raise ValueError("reach sampling distribution must sum to one")
    threshold = rng.random()
    cumulative = 0.0
    for identifier, probability in probabilities:
        cumulative += probability
        if threshold <= cumulative:
            return identifier
    return probabilities[-1][0]


def sample_first_player_decision_reach(
    game: Any,
    target_provider: Any,
    *,
    target_player: int,
    rng: random.Random,
    maximum_transitions: int,
    redaction_salt: str,
    behavior_provider: Any | None = None,
    source_descendants_official_engine_executed: bool,
) -> tuple[Any, CounterfactualReachTrace]:
    """Sample to the first target-player decision and retain a redacted ledger."""

    if target_player not in (0, 1) or maximum_transitions <= 0 or not redaction_salt:
        raise ValueError("first-decision reach configuration is invalid")
    behavior = behavior_provider or target_provider
    root = game.root_state()
    state = root
    steps: list[ReachStep] = []
    while not state.terminal:
        player = game.player_to_act(state)
        if state.actor is not InformationSetActor.CHANCE and player == target_player:
            trace = CounterfactualReachTrace(
                target_player=target_player,
                root_state_fingerprint=state_digest(root.state_id),
                decision_state_fingerprint=state_digest(state.state_id),
                steps=tuple(steps),
                target_and_sampling_strategy_match=all(
                    math.isclose(
                        step.target_probability,
                        step.sampling_probability,
                        abs_tol=1e-12,
                    )
                    for step in steps
                ),
                source_descendants_official_engine_executed=(
                    source_descendants_official_engine_executed
                ),
            )
            return state, trace
        if len(steps) >= maximum_transitions:
            raise RuntimeError("first-decision reach exceeded its transition budget")
        if state.actor is InformationSetActor.CHANCE:
            outcomes = game.chance_outcomes(state)
            probabilities = tuple(
                (outcome.outcome_id, float(outcome.probability)) for outcome in outcomes
            )
            selected_id = _sample_identifier(probabilities, rng)
            selected = next(
                outcome for outcome in outcomes if outcome.outcome_id == selected_id
            )
            probability = float(selected.probability)
            steps.append(
                ReachStep(
                    step_index=len(steps),
                    actor="chance",
                    selected_identifier_fingerprint=_fingerprint(
                        {
                            "salt": redaction_salt,
                            "step": len(steps),
                            "actor": "chance",
                            "selected": selected_id,
                        }
                    ),
                    target_probability=probability,
                    sampling_probability=probability,
                )
            )
            state = game.transition(state, None, chance_outcome=selected)
            continue
        if player not in (0, 1):
            raise RuntimeError("sampled reach encountered a decision without a player")
        information_set = game.information_set(state, player=player)
        belief = game.public_belief(state, player=player)
        target = target_provider.strategy(information_set, belief)
        target.validate_support(information_set)
        sampled = behavior.strategy(information_set, belief)
        sampled.validate_support(information_set)
        probabilities = tuple(
            (action.action_id, sampled.probability(action.action_id))
            for action in information_set.legal_actions
        )
        selected_id = _sample_identifier(probabilities, rng)
        selected = next(
            action
            for action in information_set.legal_actions
            if action.action_id == selected_id
        )
        steps.append(
            ReachStep(
                step_index=len(steps),
                actor=f"player_{player}",
                selected_identifier_fingerprint=_fingerprint(
                    {
                        "salt": redaction_salt,
                        "step": len(steps),
                        "actor": f"player_{player}",
                        "selected": selected_id,
                    }
                ),
                target_probability=target.probability(selected_id),
                sampling_probability=sampled.probability(selected_id),
            )
        )
        state = game.transition(state, selected)
    raise RuntimeError("sampled reach became terminal before the target decision")


def state_digest(value: str) -> str:
    """Preserve an existing digest or hash an opaque non-digest state ID."""

    candidate = str(value)
    if len(candidate) == 64 and all(
        character in "0123456789abcdef" for character in candidate
    ):
        return candidate
    return _fingerprint({"opaque_state_id": candidate})


@dataclass(frozen=True, slots=True)
class CounterfactualReachAuditRow:
    player: int
    information_set_id: str
    action_id: str
    state_count: int
    exact_counterfactual_reach_mass: float
    enumerated_importance_reach_mass: float
    exact_unnormalized_action_value: float
    enumerated_importance_action_value: float
    exact_conditional_action_value: float
    enumerated_conditional_action_value: float
    maximum_importance_weight: float
    absolute_mass_error: float
    absolute_value_error: float
    absolute_conditional_error: float

    def __post_init__(self) -> None:
        if self.player not in (0, 1) or self.state_count <= 0:
            raise ValueError("counterfactual reach audit row identity is invalid")
        if not self.information_set_id or not self.action_id:
            raise ValueError("counterfactual reach audit row labels are missing")
        for field_name in (
            "exact_counterfactual_reach_mass",
            "enumerated_importance_reach_mass",
            "maximum_importance_weight",
            "absolute_mass_error",
            "absolute_value_error",
            "absolute_conditional_error",
        ):
            value = float(getattr(self, field_name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and non-negative")
        for field_name in (
            "exact_unnormalized_action_value",
            "enumerated_importance_action_value",
            "exact_conditional_action_value",
            "enumerated_conditional_action_value",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite")


@dataclass(frozen=True, slots=True)
class CounterfactualReachExpectationAudit:
    game_fingerprint: str
    profile_fingerprint: str
    rows: tuple[CounterfactualReachAuditRow, ...]
    maximum_absolute_error: float
    maximum_importance_weight: float
    exact_expectation_passed: bool
    official_engine_executed: bool = False
    full_game_native_counterfactual_values_calibrated: bool = False

    def __post_init__(self) -> None:
        _validate_digest(self.game_fingerprint, "game_fingerprint")
        _validate_digest(self.profile_fingerprint, "profile_fingerprint")
        if not self.rows:
            raise ValueError("counterfactual reach audit must contain rows")
        expected_error = max(
            max(
                row.absolute_mass_error,
                row.absolute_value_error,
                row.absolute_conditional_error,
            )
            for row in self.rows
        )
        expected_weight = max(row.maximum_importance_weight for row in self.rows)
        if not math.isclose(self.maximum_absolute_error, expected_error, abs_tol=1e-15):
            raise ValueError("counterfactual reach audit maximum error is inconsistent")
        if not math.isclose(
            self.maximum_importance_weight,
            expected_weight,
            abs_tol=1e-15,
        ):
            raise ValueError("counterfactual reach audit maximum weight is inconsistent")
        if self.exact_expectation_passed != (expected_error <= 1e-12):
            raise ValueError("counterfactual reach audit pass flag is inconsistent")
        if self.official_engine_executed or self.full_game_native_counterfactual_values_calibrated:
            raise ValueError("exact microgame audit cannot promote native claims")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": COUNTERFACTUAL_REACH_SCHEMA_VERSION,
            "game_fingerprint": self.game_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
            "rows": [asdict(row) for row in self.rows],
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_importance_weight": self.maximum_importance_weight,
            "gates": {"exact_expectation": self.exact_expectation_passed},
            "claims": {
                "counterfactual_reach_excludes_target_player": True,
                "importance_expectation_exhaustively_enumerated": True,
                "official_engine_executed": self.official_engine_executed,
                "full_game_native_counterfactual_values_calibrated": (
                    self.full_game_native_counterfactual_values_calibrated
                ),
                "pokemon_strength": False,
            },
        }
        if include_fingerprint:
            result["fingerprint"] = self.fingerprint
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
    ) -> "CounterfactualReachExpectationAudit":
        claims = raw["claims"]
        result = cls(
            game_fingerprint=str(raw["game_fingerprint"]),
            profile_fingerprint=str(raw["profile_fingerprint"]),
            rows=tuple(CounterfactualReachAuditRow(**item) for item in raw["rows"]),
            maximum_absolute_error=float(raw["maximum_absolute_error"]),
            maximum_importance_weight=float(raw["maximum_importance_weight"]),
            exact_expectation_passed=bool(raw["gates"]["exact_expectation"]),
            official_engine_executed=bool(claims["official_engine_executed"]),
            full_game_native_counterfactual_values_calibrated=bool(
                claims["full_game_native_counterfactual_values_calibrated"]
            ),
        )
        if not bool(claims["counterfactual_reach_excludes_target_player"]):
            raise ValueError("counterfactual reach exclusion claim changed")
        if not bool(claims["importance_expectation_exhaustively_enumerated"]):
            raise ValueError("counterfactual reach enumeration claim changed")
        if raw.get("fingerprint") not in (None, result.fingerprint):
            raise ValueError("counterfactual reach audit fingerprint mismatch")
        return result


def _state_reaches(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> dict[str, tuple[float, float, float]]:
    reaches: dict[str, tuple[float, float, float]] = {}

    def visit(
        state: DuelState,
        reach_0: float,
        reach_1: float,
        chance_reach: float,
    ) -> None:
        current = (reach_0, reach_1, chance_reach)
        previous = reaches.get(state.state_id)
        if previous is not None:
            if any(
                not math.isclose(left, right, abs_tol=1e-12)
                for left, right in zip(previous, current, strict=True)
            ):
                raise RuntimeError("exact state has inconsistent unique-history reaches")
            return
        reaches[state.state_id] = current
        if state.terminal:
            return
        if state.cards == (None, None):
            for outcome in game.chance_outcomes(state):
                visit(
                    game.transition(state, None, chance_outcome=outcome),
                    reach_0,
                    reach_1,
                    chance_reach * outcome.probability,
                )
            return
        player = game.player_to_act(state)
        assert player is not None
        information_set = game.information_set(state, player=player)
        strategy = profile.strategy(information_set)
        for action in information_set.legal_actions:
            probability = strategy.probability(action.action_id)
            if probability <= 0:
                raise ValueError("counterfactual reach audit requires full-support profile")
            visit(
                game.transition(state, action),
                reach_0 * probability if player == 0 else reach_0,
                reach_1 * probability if player == 1 else reach_1,
                chance_reach,
            )

    visit(game.root_state(), 1.0, 1.0, 1.0)
    return reaches


def audit_counterfactual_reach_expectation(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> CounterfactualReachExpectationAudit:
    """Exhaust every information-set state and verify the reach estimator."""

    reaches = _state_reaches(game, profile)
    rows: list[CounterfactualReachAuditRow] = []
    for information_set in game.information_sets():
        states = game.states_for_information_set(information_set.info_set_id)
        player = game.player_to_act(states[0])
        assert player is not None
        for action in information_set.legal_actions:
            exact_mass = 0.0
            estimated_mass = 0.0
            exact_value = 0.0
            estimated_value = 0.0
            maximum_weight = 0.0
            for state in states:
                reach_0, reach_1, chance_reach = reaches[state.state_id]
                sampling_reach = chance_reach * reach_0 * reach_1
                counterfactual_reach = chance_reach * (
                    reach_1 if player == 0 else reach_0
                )
                weight = counterfactual_importance_weight(
                    target_player=player,
                    chance_reach=chance_reach,
                    player_0_reach=reach_0,
                    player_1_reach=reach_1,
                    sampling_reach=sampling_reach,
                )
                continuation = evaluate_profile_from_state(
                    game,
                    profile,
                    game.transition(state, action),
                    player=player,
                )
                exact_mass += counterfactual_reach
                exact_value += counterfactual_reach * continuation
                estimated_mass += sampling_reach * weight
                estimated_value += sampling_reach * weight * continuation
                maximum_weight = max(maximum_weight, weight)
            if exact_mass <= 0 or estimated_mass <= 0:
                raise RuntimeError("counterfactual reach audit found zero information-set mass")
            exact_conditional = exact_value / exact_mass
            estimated_conditional = estimated_value / estimated_mass
            rows.append(
                CounterfactualReachAuditRow(
                    player=player,
                    information_set_id=information_set.info_set_id,
                    action_id=action.action_id,
                    state_count=len(states),
                    exact_counterfactual_reach_mass=exact_mass,
                    enumerated_importance_reach_mass=estimated_mass,
                    exact_unnormalized_action_value=exact_value,
                    enumerated_importance_action_value=estimated_value,
                    exact_conditional_action_value=exact_conditional,
                    enumerated_conditional_action_value=estimated_conditional,
                    maximum_importance_weight=maximum_weight,
                    absolute_mass_error=abs(exact_mass - estimated_mass),
                    absolute_value_error=abs(exact_value - estimated_value),
                    absolute_conditional_error=abs(
                        exact_conditional - estimated_conditional
                    ),
                )
            )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (row.player, row.information_set_id, row.action_id),
        )
    )
    maximum_error = max(
        max(
            row.absolute_mass_error,
            row.absolute_value_error,
            row.absolute_conditional_error,
        )
        for row in ordered
    )
    return CounterfactualReachExpectationAudit(
        game_fingerprint=game.fingerprint,
        profile_fingerprint=profile.fingerprint,
        rows=ordered,
        maximum_absolute_error=maximum_error,
        maximum_importance_weight=max(
            row.maximum_importance_weight for row in ordered
        ),
        exact_expectation_passed=maximum_error <= 1e-12,
    )
