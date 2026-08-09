"""Interfaces for the neural game-theoretic planning pivot.

Nothing in this module claims to solve the official Pokémon TCG game.  It
defines the information-set contracts required to learn a blueprint strategy,
resolve subgames with counterfactual-value safety bounds, and add bounded
opponent exploitation.  The official engine integration lives in
``competition.py``/``official_evaluation.py``; bridging it into these abstract
objects requires validated adapters. A one-step shadow adapter lives in
``official_bridge.py``. ``official_search_game.py`` now exposes branchable
native-engine descendants, but its hidden-zone completion prior remains a
non-official research abstraction. Phase 6K trains a small sampled-regret
candidate on those descendants, but no equilibrium-validated, safe, or
deployable full-game strategy is implied.

The only concrete strategies here are deliberately named stubs or guards:
``UniformBlueprintStub`` and ``BlueprintFallbackResolver``.  They make shape,
serialization, and safety behavior executable without being misrepresented as
a trained game-theoretic agent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Protocol, Sequence


GAME_THEORY_SCHEMA_VERSION = 1


def _fingerprint(payload: Any) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _finite(value: float, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


class GameBoundaryKind(str, Enum):
    """Whether an object is official execution or a research abstraction."""

    OFFICIAL_ENGINE = "official_engine"
    ABSTRACT_INFORMATION_SET_GAME = "abstract_information_set_game"


class InformationSetActor(str, Enum):
    SELF = "self"
    OPPONENT = "opponent"
    CHANCE = "chance"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class GameBoundaryProvenance:
    name: str
    kind: GameBoundaryKind
    schema_version: int
    ruleset_fingerprint: str
    official_engine_executed: bool
    fidelity_notes: str

    def __post_init__(self) -> None:
        if not self.name or not self.fidelity_notes:
            raise ValueError("game boundary provenance must be described")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        _validate_digest(self.ruleset_fingerprint, "ruleset_fingerprint")
        if self.official_engine_executed and self.kind is not GameBoundaryKind.OFFICIAL_ENGINE:
            raise ValueError("an abstract game cannot claim official-engine execution")


@dataclass(frozen=True, slots=True)
class RangeHypothesis:
    """One opponent private-state/plan particle, never simulator truth."""

    hypothesis_id: str
    weight: float
    hidden_resource_counts: tuple[tuple[str, int], ...]
    intent_id: str
    style_id: str
    source: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.intent_id or not self.style_id or not self.source:
            raise ValueError("range hypothesis identity fields must be non-empty")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("range hypothesis weight must be finite and non-negative")
        names = [name for name, _ in self.hidden_resource_counts]
        if len(names) != len(set(names)):
            raise ValueError("hidden resource names must be unique")
        if any(not name or count < 0 for name, count in self.hidden_resource_counts):
            raise ValueError("hidden resource counts must be named and non-negative")

    def with_weight(self, weight: float) -> "RangeHypothesis":
        return RangeHypothesis(
            hypothesis_id=self.hypothesis_id,
            weight=weight,
            hidden_resource_counts=self.hidden_resource_counts,
            intent_id=self.intent_id,
            style_id=self.style_id,
            source=self.source,
        )


@dataclass(frozen=True, slots=True)
class OpponentRange:
    """Normalized public belief over opponent private states and plans."""

    hypotheses: tuple[RangeHypothesis, ...]

    def __post_init__(self) -> None:
        if not self.hypotheses:
            raise ValueError("opponent range cannot be empty")
        identifiers = [item.hypothesis_id for item in self.hypotheses]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("opponent range hypothesis IDs must be unique")
        if not math.isclose(sum(item.weight for item in self.hypotheses), 1.0, abs_tol=1e-8):
            raise ValueError("opponent range weights must sum to one")

    @classmethod
    def normalized(cls, hypotheses: Sequence[RangeHypothesis]) -> "OpponentRange":
        materialized = tuple(hypotheses)
        total = sum(item.weight for item in materialized)
        if not materialized or total <= 0 or not math.isfinite(total):
            raise ValueError("opponent range requires positive finite total weight")
        return cls(tuple(item.with_weight(item.weight / total) for item in materialized))

    @property
    def entropy(self) -> float:
        return -sum(
            item.weight * math.log(max(item.weight, 1e-12)) for item in self.hypotheses
        )

    @property
    def effective_sample_size(self) -> float:
        return 1.0 / sum(item.weight * item.weight for item in self.hypotheses)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def top_k(self, count: int) -> "OpponentRange":
        if count <= 0:
            raise ValueError("top_k count must be positive")
        selected = sorted(
            self.hypotheses, key=lambda item: (-item.weight, item.hypothesis_id)
        )[:count]
        return OpponentRange.normalized(selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GAME_THEORY_SCHEMA_VERSION,
            "hypotheses": [asdict(item) for item in self.hypotheses],
            "entropy": self.entropy,
            "effective_sample_size": self.effective_sample_size,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OpponentRange":
        # Serialized ranges are already normalized by construction.  Reapplying
        # normalization here can perturb otherwise valid floating-point weights
        # and break an enclosing public-belief fingerprint after a JSON
        # round-trip.  The constructor still enforces a unit total, so malformed
        # payloads fail closed without silently changing their evidence bytes.
        return cls(
            tuple(
                RangeHypothesis(
                    hypothesis_id=str(item["hypothesis_id"]),
                    weight=float(item["weight"]),
                    hidden_resource_counts=tuple(
                        (str(name), int(count))
                        for name, count in item.get("hidden_resource_counts", ())
                    ),
                    intent_id=str(item["intent_id"]),
                    style_id=str(item["style_id"]),
                    source=str(item["source"]),
                )
                for item in raw["hypotheses"]
            )
        )


@dataclass(frozen=True, slots=True)
class PublicBeliefState:
    """Common/public state plus a range—not the opponent's true hidden cards."""

    public_state_fingerprint: str
    public_history_fingerprint: str
    turn: int
    acting_player: InformationSetActor
    opponent_range: OpponentRange
    public_features: tuple[tuple[str, float], ...]
    updater_fingerprint: str

    def __post_init__(self) -> None:
        _validate_digest(self.public_state_fingerprint, "public_state_fingerprint")
        _validate_digest(self.public_history_fingerprint, "public_history_fingerprint")
        _validate_digest(self.updater_fingerprint, "updater_fingerprint")
        if self.turn < 0:
            raise ValueError("turn cannot be negative")
        names = [name for name, _ in self.public_features]
        if len(names) != len(set(names)):
            raise ValueError("public feature names must be unique")
        for name, value in self.public_features:
            if not name:
                raise ValueError("public feature names must be non-empty")
            _finite(value, f"public feature {name}")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GAME_THEORY_SCHEMA_VERSION,
            "public_state_fingerprint": self.public_state_fingerprint,
            "public_history_fingerprint": self.public_history_fingerprint,
            "turn": self.turn,
            "acting_player": self.acting_player.value,
            "opponent_range": self.opponent_range.to_dict(),
            "public_features": dict(self.public_features),
            "updater_fingerprint": self.updater_fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PublicBeliefState":
        features = raw.get("public_features", {})
        feature_items = features.items() if isinstance(features, Mapping) else features
        return cls(
            public_state_fingerprint=str(raw["public_state_fingerprint"]),
            public_history_fingerprint=str(raw["public_history_fingerprint"]),
            turn=int(raw["turn"]),
            acting_player=InformationSetActor(str(raw["acting_player"])),
            opponent_range=OpponentRange.from_dict(raw["opponent_range"]),
            public_features=tuple((str(name), float(value)) for name, value in feature_items),
            updater_fingerprint=str(raw["updater_fingerprint"]),
        )


class PublicBeliefUpdater(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def update(
        self,
        previous: PublicBeliefState,
        *,
        public_action: Mapping[str, Any],
        next_public_observation: Mapping[str, Any],
    ) -> PublicBeliefState: ...


@dataclass(frozen=True, slots=True)
class AbstractAction:
    action_id: str
    label: str
    official_option_indices: tuple[int, ...]
    features: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.action_id or not self.label:
            raise ValueError("abstract actions require an ID and label")
        if any(index < 0 for index in self.official_option_indices):
            raise ValueError("official option indices cannot be negative")
        if len(set(self.official_option_indices)) != len(self.official_option_indices):
            raise ValueError("official option indices cannot repeat")
        for name, value in self.features:
            if not name:
                raise ValueError("action feature names must be non-empty")
            _finite(value, f"action feature {name}")


@dataclass(frozen=True, slots=True)
class InformationSet:
    info_set_id: str
    public_belief_fingerprint: str
    actor: InformationSetActor
    legal_actions: tuple[AbstractAction, ...]
    depth: int
    terminal: bool = False
    private_observation_fingerprint: str | None = None
    private_features: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.info_set_id:
            raise ValueError("information set ID must be non-empty")
        _validate_digest(self.public_belief_fingerprint, "public_belief_fingerprint")
        if self.depth < 0:
            raise ValueError("information-set depth cannot be negative")
        if self.private_observation_fingerprint is not None:
            _validate_digest(
                self.private_observation_fingerprint,
                "private_observation_fingerprint",
            )
        private_names = [name for name, _ in self.private_features]
        if len(private_names) != len(set(private_names)):
            raise ValueError("private feature names must be unique")
        for name, value in self.private_features:
            if not name:
                raise ValueError("private feature names must be non-empty")
            _finite(value, f"private feature {name}")
        identifiers = [action.action_id for action in self.legal_actions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("legal abstract action IDs must be unique")
        if self.terminal:
            if self.actor is not InformationSetActor.TERMINAL or self.legal_actions:
                raise ValueError("terminal information sets cannot have legal actions")
        elif self.actor is InformationSetActor.TERMINAL:
            raise ValueError("nonterminal information sets cannot use the terminal actor")
        elif self.actor is InformationSetActor.CHANCE:
            if self.legal_actions:
                raise ValueError("chance nodes use chance outcomes, not player actions")
        elif not self.legal_actions:
            raise ValueError("player information sets require legal actions")


@dataclass(frozen=True, slots=True)
class ChanceOutcome:
    outcome_id: str
    probability: float
    public_label: str

    def __post_init__(self) -> None:
        if not self.outcome_id or not self.public_label:
            raise ValueError("chance outcomes require an ID and public label")
        if not 0 <= self.probability <= 1 or not math.isfinite(self.probability):
            raise ValueError("chance probability must be finite and in [0, 1]")


class AbstractGameState(Protocol):
    """Opaque solver state; private truth here must never enter policy features."""

    @property
    def state_id(self) -> str: ...

    @property
    def actor(self) -> InformationSetActor: ...

    @property
    def depth(self) -> int: ...

    @property
    def terminal(self) -> bool: ...


class AbstractInformationSetGame(Protocol):
    """Extensive-form interface for training/solving; never the native engine.

    Solvers traverse opaque states containing sampled/enumerated private truth,
    while policies receive only the corresponding ``InformationSet`` and
    ``PublicBeliefState``. Information sets are not assumed to be Markov states.
    """

    @property
    def provenance(self) -> GameBoundaryProvenance: ...

    @property
    def fingerprint(self) -> str: ...

    def root_state(self) -> AbstractGameState: ...

    def player_to_act(self, state: AbstractGameState) -> int | None:
        """Return player 0/1 at decisions and ``None`` at chance/terminal nodes.

        Keeping this query on the game boundary lets generic regret traversals
        avoid interpreting solver-private state fields.  In particular, a
        native state may contain opaque hidden-completion truth that must never
        become a policy input.
        """
        ...

    def information_set(
        self,
        state: AbstractGameState,
        *,
        player: int,
    ) -> InformationSet: ...

    def public_belief(
        self,
        state: AbstractGameState,
        *,
        player: int,
    ) -> PublicBeliefState: ...

    def legal_actions(self, state: AbstractGameState) -> tuple[AbstractAction, ...]: ...

    def transition(
        self,
        state: AbstractGameState,
        action: AbstractAction | None,
        *,
        chance_outcome: ChanceOutcome | None = None,
    ) -> AbstractGameState: ...

    def chance_outcomes(
        self, state: AbstractGameState
    ) -> tuple[ChanceOutcome, ...]: ...

    def terminal_utility(self, state: AbstractGameState, player: int) -> float: ...


class OfficialInformationSetBridge(Protocol):
    """Adapter contract between official observations and an abstract game."""

    @property
    def provenance(self) -> GameBoundaryProvenance: ...

    def public_belief_from_official_observation(
        self, observation: Mapping[str, Any]
    ) -> PublicBeliefState: ...

    def information_set_from_official_observation(
        self,
        observation: Mapping[str, Any],
        belief: PublicBeliefState,
    ) -> InformationSet: ...

    def official_selection(
        self,
        strategy: "ActionDistribution",
        observation: Mapping[str, Any],
    ) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class ActionDistribution:
    probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("action distribution cannot be empty")
        identifiers = [action_id for action_id, _ in self.probabilities]
        if any(not action_id for action_id in identifiers):
            raise ValueError("action IDs must be non-empty")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("action IDs must be unique")
        if any(probability < 0 or not math.isfinite(probability) for _, probability in self.probabilities):
            raise ValueError("action probabilities must be finite and non-negative")
        if not math.isclose(sum(value for _, value in self.probabilities), 1.0, abs_tol=1e-8):
            raise ValueError("action probabilities must sum to one")

    @classmethod
    def normalized(cls, probabilities: Mapping[str, float]) -> "ActionDistribution":
        if not probabilities:
            raise ValueError("cannot normalize an empty action distribution")
        total = sum(float(value) for value in probabilities.values())
        if total <= 0 or not math.isfinite(total):
            raise ValueError("action distribution needs positive finite mass")
        return cls(
            tuple(
                sorted(
                    (str(action_id), float(value) / total)
                    for action_id, value in probabilities.items()
                )
            )
        )

    def as_dict(self) -> dict[str, float]:
        return dict(self.probabilities)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.probabilities)

    @property
    def argmax(self) -> str:
        return min(self.probabilities, key=lambda item: (-item[1], item[0]))[0]

    def probability(self, action_id: str) -> float:
        return dict(self.probabilities).get(action_id, 0.0)

    def validate_support(self, information_set: InformationSet) -> None:
        legal = {action.action_id for action in information_set.legal_actions}
        supplied = {action_id for action_id, _ in self.probabilities}
        if supplied != legal:
            raise ValueError(
                f"strategy support must exactly match legal actions: "
                f"missing={legal - supplied}, extra={supplied - legal}"
            )


@dataclass(frozen=True, slots=True)
class CounterfactualValues:
    player: int
    values: tuple[tuple[str, float], ...]
    source_fingerprint: str

    def __post_init__(self) -> None:
        if self.player not in (0, 1):
            raise ValueError("counterfactual-value player must be 0 or 1")
        _validate_digest(self.source_fingerprint, "source_fingerprint")
        identifiers = [identifier for identifier, _ in self.values]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("counterfactual-value hypothesis IDs must be unique")
        for identifier, value in self.values:
            if not identifier:
                raise ValueError("counterfactual-value IDs must be non-empty")
            _finite(value, "counterfactual value")


@dataclass(frozen=True, slots=True)
class BlueprintCheckpoint:
    provider_name: str
    algorithm: str
    model_fingerprint: str
    game_fingerprint: str
    deck_fingerprint: str
    training_steps: int
    trained: bool
    evaluation_notes: str

    def __post_init__(self) -> None:
        if not self.provider_name or not self.algorithm or not self.evaluation_notes:
            raise ValueError("blueprint checkpoint metadata must be described")
        for field_name in ("model_fingerprint", "game_fingerprint", "deck_fingerprint"):
            _validate_digest(getattr(self, field_name), field_name)
        if self.training_steps < 0:
            raise ValueError("training_steps cannot be negative")
        if self.trained and self.training_steps == 0:
            raise ValueError("a trained blueprint must record positive training steps")


class BlueprintStrategyProvider(Protocol):
    @property
    def checkpoint(self) -> BlueprintCheckpoint: ...

    def strategy(
        self, information_set: InformationSet, belief: PublicBeliefState
    ) -> ActionDistribution: ...

    def counterfactual_values(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        player: int,
    ) -> CounterfactualValues: ...


class NeuralBlueprintStrategyProvider(BlueprintStrategyProvider, Protocol):
    """Marker protocol for a trained neural blueprint; no model is fabricated here."""

    def encode_public_belief(self, belief: PublicBeliefState) -> Sequence[float]: ...


class UniformBlueprintStub:
    """Executable shape stub. It is explicitly untrained and never a final policy."""

    def __init__(self, *, game_fingerprint: str, deck_fingerprint: str) -> None:
        self._checkpoint = BlueprintCheckpoint(
            provider_name="uniform_blueprint_stub",
            algorithm="uniform_shape_stub",
            model_fingerprint=_fingerprint({"name": "uniform_blueprint_stub", "schema": 1}),
            game_fingerprint=game_fingerprint,
            deck_fingerprint=deck_fingerprint,
            training_steps=0,
            trained=False,
            evaluation_notes="Interface validation only; no strategic-strength claim.",
        )

    @property
    def checkpoint(self) -> BlueprintCheckpoint:
        return self._checkpoint

    def strategy(
        self, information_set: InformationSet, belief: PublicBeliefState
    ) -> ActionDistribution:
        del belief
        if information_set.terminal:
            raise ValueError("terminal information sets have no strategy")
        probability = 1.0 / len(information_set.legal_actions)
        return ActionDistribution(
            tuple((action.action_id, probability) for action in information_set.legal_actions)
        )

    def counterfactual_values(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        player: int,
    ) -> CounterfactualValues:
        del information_set
        return CounterfactualValues(
            player=player,
            values=tuple(
                (hypothesis.hypothesis_id, 0.0)
                for hypothesis in belief.opponent_range.hypotheses
            ),
            source_fingerprint=self.checkpoint.model_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class HypothesisValueBound:
    hypothesis_id: str
    lower: float
    upper: float
    baseline_value: float

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("value-bound hypothesis ID must be non-empty")
        lower = _finite(self.lower, "lower bound")
        upper = _finite(self.upper, "upper bound")
        baseline = _finite(self.baseline_value, "baseline value")
        if lower > baseline or baseline > upper:
            raise ValueError("bounds must contain the baseline counterfactual value")


@dataclass(frozen=True, slots=True)
class CounterfactualValueBounds:
    opponent_player: int
    bounds: tuple[HypothesisValueBound, ...]
    confidence: float
    baseline_strategy_fingerprint: str
    source: str

    def __post_init__(self) -> None:
        if self.opponent_player not in (0, 1):
            raise ValueError("opponent_player must be 0 or 1")
        if not self.bounds or not self.source:
            raise ValueError("counterfactual bounds and source are required")
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("bound confidence must be finite and in [0, 1]")
        _validate_digest(
            self.baseline_strategy_fingerprint, "baseline_strategy_fingerprint"
        )
        identifiers = [bound.hypothesis_id for bound in self.bounds]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("counterfactual-bound IDs must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))

    @property
    def max_width(self) -> float:
        return max(bound.upper - bound.lower for bound in self.bounds)


class CounterfactualValueBoundProvider(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def bounds(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        blueprint: BlueprintStrategyProvider,
    ) -> CounterfactualValueBounds: ...


@dataclass(frozen=True, slots=True)
class ResolveBudget:
    iterations: int
    max_nodes: int
    time_budget_ms: float
    safety_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.max_nodes <= 0 or self.time_budget_ms <= 0:
            raise ValueError("resolve budgets must be positive")
        if self.safety_tolerance < 0 or not math.isfinite(self.safety_tolerance):
            raise ValueError("safety tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SafeResolveRequest:
    information_set: InformationSet
    belief: PublicBeliefState
    baseline_strategy: ActionDistribution
    opponent_value_bounds: CounterfactualValueBounds
    budget: ResolveBudget

    def __post_init__(self) -> None:
        self.baseline_strategy.validate_support(self.information_set)
        if (
            self.opponent_value_bounds.baseline_strategy_fingerprint
            != self.baseline_strategy.fingerprint
        ):
            raise ValueError(
                "counterfactual bounds must be calibrated to the supplied baseline strategy"
            )
        range_ids = {
            hypothesis.hypothesis_id for hypothesis in self.belief.opponent_range.hypotheses
        }
        bound_ids = {
            bound.hypothesis_id for bound in self.opponent_value_bounds.bounds
        }
        if range_ids != bound_ids:
            raise ValueError("counterfactual bounds must cover every range hypothesis")


@dataclass(frozen=True, slots=True)
class SafetyCertificate:
    safe: bool
    max_counterfactual_value_violation: float
    tolerance: float
    confidence: float
    bound_fingerprint: str
    method: str

    def __post_init__(self) -> None:
        violation = _finite(
            self.max_counterfactual_value_violation,
            "max_counterfactual_value_violation",
        )
        if violation < 0 or self.tolerance < 0:
            raise ValueError("safety violation and tolerance cannot be negative")
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("safety confidence must be finite and in [0, 1]")
        _validate_digest(self.bound_fingerprint, "bound_fingerprint")
        if not self.method:
            raise ValueError("safety certificate method must be described")
        if self.safe != (violation <= self.tolerance):
            raise ValueError("safety flag must agree with measured violation and tolerance")


@dataclass(frozen=True, slots=True)
class ResolvedStrategy:
    strategy: ActionDistribution
    baseline_strategy: ActionDistribution
    certificate: SafetyCertificate
    iterations: int
    expanded_nodes: int
    elapsed_ms: float
    fallback_to_blueprint: bool
    resolver_name: str

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.expanded_nodes < 0 or self.elapsed_ms < 0:
            raise ValueError("resolver accounting cannot be negative")
        if not self.resolver_name:
            raise ValueError("resolver_name must be non-empty")
        if self.fallback_to_blueprint and self.strategy != self.baseline_strategy:
            raise ValueError("blueprint fallback must return the baseline strategy exactly")


class SafeSubgameResolver(Protocol):
    @property
    def name(self) -> str: ...

    def resolve(
        self,
        game: AbstractInformationSetGame,
        request: SafeResolveRequest,
    ) -> ResolvedStrategy: ...


class BlueprintFallbackResolver:
    """Safe no-op resolver used until a real continual resolver is validated."""

    name = "blueprint_fallback_resolver"

    def resolve(
        self,
        game: AbstractInformationSetGame,
        request: SafeResolveRequest,
    ) -> ResolvedStrategy:
        del game
        certificate = SafetyCertificate(
            safe=True,
            max_counterfactual_value_violation=0.0,
            tolerance=request.budget.safety_tolerance,
            confidence=request.opponent_value_bounds.confidence,
            bound_fingerprint=request.opponent_value_bounds.fingerprint,
            method="identity_to_blueprint_stub",
        )
        return ResolvedStrategy(
            strategy=request.baseline_strategy,
            baseline_strategy=request.baseline_strategy,
            certificate=certificate,
            iterations=0,
            expanded_nodes=0,
            elapsed_ms=0.0,
            fallback_to_blueprint=True,
            resolver_name=self.name,
        )


@dataclass(frozen=True, slots=True)
class OpponentModelSignal:
    exploitative_strategy: ActionDistribution
    confidence: float
    effective_samples: float
    calibration_error: float
    model_fingerprint: str

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1 or not math.isfinite(self.confidence):
            raise ValueError("opponent-model confidence must be finite and in [0, 1]")
        if self.effective_samples < 0 or not math.isfinite(self.effective_samples):
            raise ValueError("effective_samples must be finite and non-negative")
        if not 0 <= self.calibration_error <= 1 or not math.isfinite(self.calibration_error):
            raise ValueError("calibration_error must be finite and in [0, 1]")
        _validate_digest(self.model_fingerprint, "model_fingerprint")


@dataclass(frozen=True, slots=True)
class ExploitationConstraint:
    max_kl_from_safe_strategy: float = 0.05
    max_blend: float = 0.25
    min_baseline_mass_fraction: float = 0.75
    min_model_confidence: float = 0.60
    min_effective_samples: float = 25.0
    max_calibration_error: float = 0.20
    require_safe_resolve: bool = True

    def __post_init__(self) -> None:
        if self.max_kl_from_safe_strategy < 0 or not math.isfinite(
            self.max_kl_from_safe_strategy
        ):
            raise ValueError("KL limit must be finite and non-negative")
        for field_name in (
            "max_blend",
            "min_baseline_mass_fraction",
            "min_model_confidence",
            "max_calibration_error",
        ):
            value = getattr(self, field_name)
            if not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
        if self.min_effective_samples < 0 or not math.isfinite(
            self.min_effective_samples
        ):
            raise ValueError("min_effective_samples must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class ExploitationDecision:
    strategy: ActionDistribution
    safe_baseline: ActionDistribution
    blend: float
    kl_from_safe_strategy: float
    used_opponent_model: bool
    reason: str
    final_safety_certificate: SafetyCertificate | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.blend <= 1 or self.kl_from_safe_strategy < 0:
            raise ValueError("exploitation blend/KL must be non-negative and bounded")
        if not self.reason:
            raise ValueError("exploitation decision reason must be non-empty")
        if (
            self.final_safety_certificate is not None
            and not self.final_safety_certificate.safe
        ):
            raise ValueError("a final exploitation certificate must certify safety")


class OpponentExploitationOverlay(Protocol):
    def apply(
        self,
        *,
        safe_strategy: ActionDistribution,
        safety_certificate: SafetyCertificate,
        opponent_model: OpponentModelSignal,
    ) -> ExploitationDecision: ...


class BaselineConstrainedExploitationOverlay:
    """KL- and support-constrained mixture around a certified safe strategy."""

    def __init__(self, constraint: ExploitationConstraint | None = None) -> None:
        self.constraint = constraint or ExploitationConstraint()

    def apply(
        self,
        *,
        safe_strategy: ActionDistribution,
        safety_certificate: SafetyCertificate,
        opponent_model: OpponentModelSignal,
    ) -> ExploitationDecision:
        if self.constraint.require_safe_resolve and not safety_certificate.safe:
            return self._baseline(safe_strategy, "unsafe resolve certificate")
        if opponent_model.confidence < self.constraint.min_model_confidence:
            return self._baseline(safe_strategy, "opponent model below confidence gate")
        if opponent_model.effective_samples < self.constraint.min_effective_samples:
            return self._baseline(safe_strategy, "opponent model below sample-size gate")
        if opponent_model.calibration_error > self.constraint.max_calibration_error:
            return self._baseline(safe_strategy, "opponent model above calibration-error gate")
        baseline = safe_strategy.as_dict()
        exploit = opponent_model.exploitative_strategy.as_dict()
        if set(exploit) != set(baseline):
            return self._baseline(safe_strategy, "opponent model action support mismatch")
        # Safe baseline zero-probability actions cannot be introduced by the
        # exploitation layer without a separate safety proof.
        if any(exploit[action_id] > 0 and baseline[action_id] == 0 for action_id in baseline):
            return self._baseline(safe_strategy, "opponent model adds zero-support action")

        maximum_alpha = min(
            self.constraint.max_blend
            * opponent_model.confidence
            * (1.0 - opponent_model.calibration_error),
            1.0 - self.constraint.min_baseline_mass_fraction,
        )
        low, high = 0.0, max(0.0, maximum_alpha)
        for _ in range(48):
            middle = (low + high) / 2.0
            candidate = self._mix(baseline, exploit, middle)
            if self._kl(candidate, baseline) <= self.constraint.max_kl_from_safe_strategy:
                low = middle
            else:
                high = middle
        blend = low
        candidate = self._mix(baseline, exploit, blend)
        strategy = ActionDistribution.normalized(candidate)
        divergence = self._kl(strategy.as_dict(), baseline)
        used = blend > 1e-9
        return ExploitationDecision(
            strategy=strategy,
            safe_baseline=safe_strategy,
            blend=blend,
            kl_from_safe_strategy=divergence,
            used_opponent_model=used,
            reason=(
                "confidence/KL/support constrained mixture"
                if used
                else "constraints reduced exploitation to baseline"
            ),
        )

    @staticmethod
    def _mix(
        baseline: Mapping[str, float], exploit: Mapping[str, float], alpha: float
    ) -> dict[str, float]:
        return {
            action_id: (1.0 - alpha) * probability + alpha * exploit[action_id]
            for action_id, probability in baseline.items()
        }

    @staticmethod
    def _kl(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> float:
        value = 0.0
        for action_id, probability in candidate.items():
            if probability <= 0:
                continue
            reference = baseline[action_id]
            if reference <= 0:
                return math.inf
            value += probability * math.log(probability / reference)
        return max(0.0, value)

    @staticmethod
    def _baseline(strategy: ActionDistribution, reason: str) -> ExploitationDecision:
        return ExploitationDecision(
            strategy=strategy,
            safe_baseline=strategy,
            blend=0.0,
            kl_from_safe_strategy=0.0,
            used_opponent_model=False,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class GameTheoreticPlanningRequest:
    information_set: InformationSet
    belief: PublicBeliefState
    resolve_budget: ResolveBudget
    opponent_value_bounds: CounterfactualValueBounds
    opponent_model: OpponentModelSignal | None = None


@dataclass(frozen=True, slots=True)
class GameTheoreticPlanningDecision:
    blueprint_strategy: ActionDistribution
    resolved: ResolvedStrategy
    final_strategy: ActionDistribution
    exploitation: ExploitationDecision | None
    boundary_kind: GameBoundaryKind
    official_engine_executed: bool
    final_strategy_safety_status: str
    planner_status: str

    def __post_init__(self) -> None:
        if self.official_engine_executed and self.boundary_kind is not GameBoundaryKind.OFFICIAL_ENGINE:
            raise ValueError("abstract planning cannot claim official-engine execution")
        if not self.final_strategy_safety_status or not self.planner_status:
            raise ValueError("planner and final-strategy safety status must be non-empty")


class NeuralGameTheoreticPlanner(Protocol):
    """Target planner contract; requires a trained neural blueprint in production."""

    def decide(
        self,
        game: AbstractInformationSetGame,
        request: GameTheoreticPlanningRequest,
    ) -> GameTheoreticPlanningDecision: ...


class SafePlanningPipeline:
    """Wires blueprint, safe resolver, and exploitation overlay without pretending training."""

    def __init__(
        self,
        *,
        blueprint: BlueprintStrategyProvider,
        resolver: SafeSubgameResolver,
        exploitation_overlay: OpponentExploitationOverlay | None = None,
    ) -> None:
        self.blueprint = blueprint
        self.resolver = resolver
        self.exploitation_overlay = exploitation_overlay

    def decide(
        self,
        game: AbstractInformationSetGame,
        request: GameTheoreticPlanningRequest,
    ) -> GameTheoreticPlanningDecision:
        if game.provenance.kind is not GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME:
            raise ValueError(
                "SafePlanningPipeline operates on the abstract information-set game; "
                "official execution requires a separately validated bridge"
            )
        if self.blueprint.checkpoint.game_fingerprint != game.fingerprint:
            raise ValueError(
                "blueprint checkpoint does not match the abstract game fingerprint"
            )
        blueprint_strategy = self.blueprint.strategy(
            request.information_set, request.belief
        )
        blueprint_strategy.validate_support(request.information_set)
        resolved = self.resolver.resolve(
            game,
            SafeResolveRequest(
                information_set=request.information_set,
                belief=request.belief,
                baseline_strategy=blueprint_strategy,
                opponent_value_bounds=request.opponent_value_bounds,
                budget=request.resolve_budget,
            ),
        )
        resolved.strategy.validate_support(request.information_set)
        exploitation: ExploitationDecision | None = None
        final_strategy = resolved.strategy
        if request.opponent_model is not None and self.exploitation_overlay is not None:
            exploitation = self.exploitation_overlay.apply(
                safe_strategy=resolved.strategy,
                safety_certificate=resolved.certificate,
                opponent_model=request.opponent_model,
            )
            exploitation.strategy.validate_support(request.information_set)
            final_strategy = exploitation.strategy
        if exploitation is None or not exploitation.used_opponent_model:
            final_safety_status = "inherits_resolve_certificate"
        elif exploitation.final_safety_certificate is not None:
            final_safety_status = "final_overlay_strategy_certified"
        else:
            final_safety_status = "engineering_guard_only_not_formally_certified"
        trained = self.blueprint.checkpoint.trained
        return GameTheoreticPlanningDecision(
            blueprint_strategy=blueprint_strategy,
            resolved=resolved,
            final_strategy=final_strategy,
            exploitation=exploitation,
            boundary_kind=game.provenance.kind,
            official_engine_executed=False,
            final_strategy_safety_status=final_safety_status,
            planner_status=(
                "abstract_trained_blueprint_pipeline"
                if trained
                else "abstract_interface_stub_untrained"
            ),
        )
