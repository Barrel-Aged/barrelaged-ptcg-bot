"""Depth-limited information-set game with explicit leaf-value provenance.

The wrapper is intentionally an abstract game even when its source transitions
come from the licensed native engine. A cutoff value is not an official
terminal result. This separation is essential before using bounded horizons to
control outcome-sampling variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Protocol

from .exact_game import RocketPrizeDuel, TabularStrategyProfile, evaluate_profile_from_state
from .game_theory import (
    AbstractAction,
    AbstractGameState,
    AbstractInformationSetGame,
    ChanceOutcome,
    GameBoundaryKind,
    GameBoundaryProvenance,
    InformationSet,
    InformationSetActor,
    PublicBeliefState,
)


DEPTH_LIMITED_GAME_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LeafValueEstimate:
    player: int
    value: float
    lower: float
    upper: float
    confidence: float
    source: str
    calibrated: bool
    official_terminal_result: bool = False

    def __post_init__(self) -> None:
        if self.player not in (0, 1) or not self.source:
            raise ValueError("leaf value identity is invalid")
        for name in ("value", "lower", "upper", "confidence"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"leaf {name} must be finite")
        if not self.lower <= self.value <= self.upper:
            raise ValueError("leaf interval must contain its point estimate")
        if not 0 <= self.confidence <= 1:
            raise ValueError("leaf confidence must be in [0, 1]")
        if self.official_terminal_result:
            raise ValueError("a cutoff leaf value cannot claim official terminal status")
        if self.calibrated and self.confidence <= 0:
            raise ValueError("a calibrated leaf value needs positive confidence")


class LeafValueProvider(Protocol):
    @property
    def fingerprint(self) -> str: ...

    @property
    def calibrated(self) -> bool: ...

    def estimate(
        self,
        game: AbstractInformationSetGame,
        state: AbstractGameState,
        *,
        player: int,
    ) -> LeafValueEstimate: ...


@dataclass(frozen=True, slots=True)
class DepthLimitedGameState:
    game_fingerprint: str
    source_state: AbstractGameState
    root_source_depth: int
    maximum_source_depth_delta: int

    def __post_init__(self) -> None:
        if self.root_source_depth < 0 or self.maximum_source_depth_delta <= 0:
            raise ValueError("depth-limited state bounds are invalid")
        if self.source_state.depth < self.root_source_depth:
            raise ValueError("depth-limited source state precedes its root")

    @property
    def cutoff(self) -> bool:
        return (
            not self.source_state.terminal
            and self.source_state.actor is not InformationSetActor.CHANCE
            and self.source_state.depth - self.root_source_depth
            >= self.maximum_source_depth_delta
        )

    @property
    def state_id(self) -> str:
        return _fingerprint(
            {
                "game": self.game_fingerprint,
                "source": self.source_state.state_id,
                "root_depth": self.root_source_depth,
                "maximum_delta": self.maximum_source_depth_delta,
                "cutoff": self.cutoff,
            }
        )

    @property
    def actor(self) -> InformationSetActor:
        if self.terminal:
            return InformationSetActor.TERMINAL
        return self.source_state.actor

    @property
    def depth(self) -> int:
        return self.source_state.depth - self.root_source_depth

    @property
    def terminal(self) -> bool:
        return self.source_state.terminal or self.cutoff


class DepthLimitedInformationSetGame:
    """Abstract cutoff wrapper around any information-set game boundary."""

    def __init__(
        self,
        source_game: AbstractInformationSetGame,
        leaf_provider: LeafValueProvider,
        *,
        maximum_source_depth_delta: int,
        root_state: AbstractGameState | None = None,
    ) -> None:
        if maximum_source_depth_delta <= 0:
            raise ValueError("depth-limited horizon must be positive")
        source_root = root_state if root_state is not None else source_game.root_state()
        if source_root.terminal:
            raise ValueError("depth-limited wrapper requires a nonterminal root")
        self.source_game = source_game
        self.leaf_provider = leaf_provider
        self.maximum_source_depth_delta = int(maximum_source_depth_delta)
        self.source_root = source_root
        self._fingerprint = _fingerprint(
            {
                "schema": DEPTH_LIMITED_GAME_SCHEMA_VERSION,
                "source_game": source_game.fingerprint,
                "source_root": source_root.state_id,
                "leaf_provider": leaf_provider.fingerprint,
                "maximum_source_depth_delta": self.maximum_source_depth_delta,
            }
        )
        self._provenance = GameBoundaryProvenance(
            name="depth_limited_information_set_game",
            kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            schema_version=DEPTH_LIMITED_GAME_SCHEMA_VERSION,
            ruleset_fingerprint=source_game.provenance.ruleset_fingerprint,
            official_engine_executed=False,
            fidelity_notes=(
                "Source transitions may be genuine, but any cutoff utility is a "
                "leaf-provider abstraction and never an official terminal result."
            ),
        )
        self._leaf_estimates: list[LeafValueEstimate] = []

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def provenance(self) -> GameBoundaryProvenance:
        return self._provenance

    @property
    def source_official_engine_executed(self) -> bool:
        return False

    @property
    def source_descendant_official_engine_executed(self) -> bool:
        return bool(getattr(self.source_game, "source_official_engine_executed", False))

    @property
    def hidden_completion_prior_official(self) -> bool:
        return bool(getattr(self.source_game, "hidden_completion_prior_official", False))

    @property
    def leaf_value_calibrated(self) -> bool:
        return bool(self.leaf_provider.calibrated)

    @property
    def leaf_estimates(self) -> tuple[LeafValueEstimate, ...]:
        return tuple(self._leaf_estimates)

    def _wrap(self, state: AbstractGameState) -> DepthLimitedGameState:
        return DepthLimitedGameState(
            self.fingerprint,
            state,
            self.source_root.depth,
            self.maximum_source_depth_delta,
        )

    def _validate(self, state: DepthLimitedGameState) -> None:
        if state.game_fingerprint != self.fingerprint:
            raise ValueError("depth-limited state belongs to another game")

    def root_state(self) -> DepthLimitedGameState:
        return self._wrap(self.source_root)

    def player_to_act(self, state: DepthLimitedGameState) -> int | None:
        self._validate(state)
        if state.terminal:
            return None
        return self.source_game.player_to_act(state.source_state)

    def information_set(
        self, state: DepthLimitedGameState, *, player: int
    ) -> InformationSet:
        self._validate(state)
        if state.terminal:
            raise ValueError("cutoff/terminal states have no information set")
        return self.source_game.information_set(state.source_state, player=player)

    def public_belief(
        self, state: DepthLimitedGameState, *, player: int
    ) -> PublicBeliefState:
        self._validate(state)
        if state.terminal:
            raise ValueError("cutoff/terminal states have no public belief")
        return self.source_game.public_belief(state.source_state, player=player)

    def legal_actions(
        self, state: DepthLimitedGameState
    ) -> tuple[AbstractAction, ...]:
        self._validate(state)
        if state.terminal:
            return ()
        return self.source_game.legal_actions(state.source_state)

    def chance_outcomes(
        self, state: DepthLimitedGameState
    ) -> tuple[ChanceOutcome, ...]:
        self._validate(state)
        if state.terminal:
            return ()
        return self.source_game.chance_outcomes(state.source_state)

    def transition(
        self,
        state: DepthLimitedGameState,
        action: AbstractAction | None,
        *,
        chance_outcome: ChanceOutcome | None = None,
    ) -> DepthLimitedGameState:
        self._validate(state)
        if state.terminal:
            raise ValueError("cannot transition a cutoff/terminal state")
        return self._wrap(
            self.source_game.transition(
                state.source_state,
                action,
                chance_outcome=chance_outcome,
            )
        )

    def terminal_utility(self, state: DepthLimitedGameState, player: int) -> float:
        self._validate(state)
        if not state.terminal:
            raise ValueError("terminal utility requires a terminal/cutoff state")
        if state.source_state.terminal:
            return self.source_game.terminal_utility(state.source_state, player)
        estimate = self.leaf_provider.estimate(
            self.source_game, state.source_state, player=player
        )
        self._leaf_estimates.append(estimate)
        return estimate.value


class ExactProfileLeafValueProvider:
    """Exact continuation value for the enumerable RocketPrizeDuel only."""

    def __init__(self, game: RocketPrizeDuel, profile: TabularStrategyProfile) -> None:
        self.game = game
        self.profile = profile
        self._fingerprint = _fingerprint(
            {
                "provider": "exact_profile_leaf_value",
                "game": game.fingerprint,
                "profile": profile.fingerprint,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def calibrated(self) -> bool:
        return True

    def estimate(
        self,
        game: AbstractInformationSetGame,
        state: AbstractGameState,
        *,
        player: int,
    ) -> LeafValueEstimate:
        if game is not self.game:
            raise ValueError("exact leaf provider received another game")
        value = evaluate_profile_from_state(
            self.game, self.profile, state, player=player  # type: ignore[arg-type]
        )
        return LeafValueEstimate(
            player=player,
            value=value,
            lower=value,
            upper=value,
            confidence=1.0,
            source="exact enumeration under the supplied RocketPrizeDuel profile",
            calibrated=True,
        )
