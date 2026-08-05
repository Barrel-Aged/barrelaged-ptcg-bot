"""Exactly enumerable imperfect-information validation game for Phase 6B.

``RocketPrizeDuel`` is a deliberately tiny, zero-sum Kuhn-style endgame. Three
private tactical strengths are themed as Mimikyu, Spidops, and Mewtwo; each
player receives a distinct hidden strength and then chooses whether to hold,
push, concede, or contest. The game is not Pokemon TCG and never claims official
engine fidelity. Its purpose is to make information sets, chance, counterfactual
values, exploitability, CFR convergence, and bounded local re-solving exactly
auditable before any full-game neural scaling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
import math
from time import perf_counter
from typing import Any, Iterable, Mapping

from .game_theory import (
    AbstractAction,
    ActionDistribution,
    BlueprintCheckpoint,
    ChanceOutcome,
    CounterfactualValueBounds,
    CounterfactualValues,
    GameBoundaryKind,
    GameBoundaryProvenance,
    HypothesisValueBound,
    InformationSet,
    InformationSetActor,
    OpponentRange,
    PublicBeliefState,
    RangeHypothesis,
    ResolvedStrategy,
    SafeResolveRequest,
    SafetyCertificate,
)


EXACT_GAME_SCHEMA_VERSION = 1
ANALYTIC_GAME_VALUE_PLAYER_0 = -1.0 / 18.0


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_digest(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


class DuelCard(IntEnum):
    MIMIKYU = 0
    SPIDOPS = 1
    MEWTWO = 2

    @property
    def label(self) -> str:
        return self.name.lower()


HOLD = "hold"
PUSH = "push"
CONCEDE = "concede"
CONTEST = "contest"

_VALID_HISTORIES = frozenset(
    {
        (),
        (HOLD,),
        (PUSH,),
        (HOLD, HOLD),
        (HOLD, PUSH),
        (PUSH, CONCEDE),
        (PUSH, CONTEST),
        (HOLD, PUSH, CONCEDE),
        (HOLD, PUSH, CONTEST),
    }
)
_TERMINAL_HISTORIES = frozenset(
    {
        (HOLD, HOLD),
        (PUSH, CONCEDE),
        (PUSH, CONTEST),
        (HOLD, PUSH, CONCEDE),
        (HOLD, PUSH, CONTEST),
    }
)


@lru_cache(maxsize=4)
def _action(action_id: str) -> AbstractAction:
    labels = {
        HOLD: "Hold resources",
        PUSH: "Push the prize race",
        CONCEDE: "Concede the exchange",
        CONTEST: "Contest the exchange",
    }
    return AbstractAction(
        action_id=action_id,
        label=labels[action_id],
        official_option_indices=(),
        features=(("aggression", 1.0 if action_id in {PUSH, CONTEST} else 0.0),),
    )


@dataclass(frozen=True, slots=True)
class DuelState:
    cards: tuple[DuelCard | None, DuelCard | None]
    history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.cards) != 2 or self.history not in _VALID_HISTORIES:
            raise ValueError("invalid duel cards/history")
        undealt = self.cards == (None, None)
        if undealt and self.history:
            raise ValueError("undealt root cannot have action history")
        if not undealt:
            if any(card is None for card in self.cards) or self.cards[0] == self.cards[1]:
                raise ValueError("dealt duel requires two distinct private cards")

    @property
    def state_id(self) -> str:
        # Solver-only identity. It must never be used as a policy feature.
        return _fingerprint(
            {
                "cards": [None if card is None else int(card) for card in self.cards],
                "history": self.history,
            }
        )

    @property
    def depth(self) -> int:
        return len(self.history) + (0 if self.cards == (None, None) else 1)

    @property
    def terminal(self) -> bool:
        return self.history in _TERMINAL_HISTORIES

    @property
    def acting_player(self) -> int | None:
        if self.cards == (None, None) or self.terminal:
            return None
        if self.history in ((), (HOLD, PUSH)):
            return 0
        if self.history in ((HOLD,), (PUSH,)):
            return 1
        raise ValueError("invalid nonterminal duel history")

    @property
    def actor(self) -> InformationSetActor:
        if self.cards == (None, None):
            return InformationSetActor.CHANCE
        if self.terminal:
            return InformationSetActor.TERMINAL
        return (
            InformationSetActor.SELF
            if self.acting_player == 0
            else InformationSetActor.OPPONENT
        )


@dataclass(frozen=True, slots=True)
class TabularStrategyProfile:
    strategies: tuple[tuple[str, ActionDistribution], ...]
    label: str

    def __post_init__(self) -> None:
        if not self.strategies or not self.label:
            raise ValueError("tabular profile requires strategies and a label")
        identifiers = [identifier for identifier, _ in self.strategies]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("tabular strategy information-set IDs must be unique")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def strategy(self, information_set: InformationSet | str) -> ActionDistribution:
        identifier = (
            information_set.info_set_id
            if isinstance(information_set, InformationSet)
            else information_set
        )
        strategies = dict(self.strategies)
        if identifier not in strategies:
            raise KeyError(f"profile has no strategy for {identifier}")
        result = strategies[identifier]
        if isinstance(information_set, InformationSet):
            result.validate_support(information_set)
        return result

    def with_strategy(
        self,
        information_set_id: str,
        strategy: ActionDistribution,
        *,
        label: str | None = None,
    ) -> "TabularStrategyProfile":
        values = dict(self.strategies)
        if information_set_id not in values:
            raise KeyError(f"unknown information set {information_set_id}")
        values[information_set_id] = strategy
        return TabularStrategyProfile(
            strategies=tuple(sorted(values.items())),
            label=label or self.label,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXACT_GAME_SCHEMA_VERSION,
            "label": self.label,
            "strategies": {
                identifier: distribution.as_dict()
                for identifier, distribution in self.strategies
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TabularStrategyProfile":
        return cls(
            strategies=tuple(
                sorted(
                    (
                        str(identifier),
                        ActionDistribution.normalized(
                            {str(action): float(value) for action, value in strategy.items()}
                        ),
                    )
                    for identifier, strategy in raw["strategies"].items()
                )
            ),
            label=str(raw["label"]),
        )


@dataclass(frozen=True, slots=True)
class ExactProfileMetrics:
    expected_value_player_0: float
    best_response_value_player_0: float
    best_response_value_player_1: float
    nash_conv: float
    exploitability: float
    profile_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "expected_value_player_0",
            "best_response_value_player_0",
            "best_response_value_player_1",
            "nash_conv",
            "exploitability",
        ):
            if not math.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if self.nash_conv < -1e-10 or self.exploitability < -1e-10:
            raise ValueError("NashConv/exploitability cannot be negative")
        _validate_digest(self.profile_fingerprint, "profile_fingerprint")


@dataclass(frozen=True, slots=True)
class CFRTrainingResult:
    iterations: int
    profile: TabularStrategyProfile
    metrics: ExactProfileMetrics
    maximum_positive_regret: float
    solver_fingerprint: str

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.maximum_positive_regret < 0:
            raise ValueError("CFR result iterations/regret is invalid")
        _validate_digest(self.solver_fingerprint, "solver_fingerprint")
        if self.metrics.profile_fingerprint != self.profile.fingerprint:
            raise ValueError("CFR metrics must describe the returned profile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXACT_GAME_SCHEMA_VERSION,
            "iterations": self.iterations,
            "profile": self.profile.to_dict(),
            "metrics": asdict(self.metrics),
            "maximum_positive_regret": self.maximum_positive_regret,
            "solver_fingerprint": self.solver_fingerprint,
        }


class RocketPrizeDuel:
    """Enumerable Kuhn-style game; explicitly abstract and non-official."""

    RULES = {
        "schema": EXACT_GAME_SCHEMA_VERSION,
        "cards": [card.label for card in DuelCard],
        "actions": [HOLD, PUSH, CONCEDE, CONTEST],
        "showdown_stakes": {"hold_hold": 1.0, "contest": 2.0},
        "concede_utility_to_pusher": 1.0,
        "zero_sum": True,
        "official_fidelity": False,
    }

    def __init__(self) -> None:
        self._ruleset_fingerprint = _fingerprint(self.RULES)
        self._fingerprint = _fingerprint(
            {"name": "rocket_prize_duel", "rules": self._ruleset_fingerprint}
        )
        self._provenance = GameBoundaryProvenance(
            name="rocket_prize_duel_exact_validation_game",
            kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            schema_version=EXACT_GAME_SCHEMA_VERSION,
            ruleset_fingerprint=self._ruleset_fingerprint,
            official_engine_executed=False,
            fidelity_notes=(
                "Exactly enumerable Kuhn-style validation game; Pokemon-themed "
                "labels only, no official card-rule fidelity claim."
            ),
        )
        self._information_sets: dict[str, InformationSet] | None = None
        self._states_by_information_set: dict[str, tuple[DuelState, ...]] | None = None
        self._information_set_cache: dict[str, InformationSet] = {}

    @property
    def provenance(self) -> GameBoundaryProvenance:
        return self._provenance

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def root_state(self) -> DuelState:
        return DuelState((None, None), ())

    def player_to_act(self, state: DuelState) -> int | None:
        return state.acting_player

    def chance_outcomes(self, state: DuelState) -> tuple[ChanceOutcome, ...]:
        if state.cards != (None, None):
            return ()
        deals = tuple((first, second) for first in DuelCard for second in DuelCard if first != second)
        probability = 1.0 / len(deals)
        return tuple(
            ChanceOutcome(
                outcome_id=f"{first.label}|{second.label}",
                probability=probability,
                public_label="private_tactical_strength_deal",
            )
            for first, second in deals
        )

    def legal_actions(self, state: DuelState) -> tuple[AbstractAction, ...]:
        if state.cards == (None, None) or state.terminal:
            return ()
        if state.history in ((), (HOLD,)):
            return (_action(HOLD), _action(PUSH))
        if state.history in ((PUSH,), (HOLD, PUSH)):
            return (_action(CONCEDE), _action(CONTEST))
        raise ValueError("invalid nonterminal history")

    def transition(
        self,
        state: DuelState,
        action: AbstractAction | None,
        *,
        chance_outcome: ChanceOutcome | None = None,
    ) -> DuelState:
        if state.cards == (None, None):
            if action is not None or chance_outcome is None:
                raise ValueError("root transition requires exactly one chance outcome")
            valid = {outcome.outcome_id for outcome in self.chance_outcomes(state)}
            if chance_outcome.outcome_id not in valid:
                raise ValueError("unknown private deal")
            first_label, second_label = chance_outcome.outcome_id.split("|")
            by_label = {card.label: card for card in DuelCard}
            return DuelState((by_label[first_label], by_label[second_label]), ())
        if state.terminal or chance_outcome is not None or action is None:
            raise ValueError("player transition requires one legal action")
        legal = {candidate.action_id for candidate in self.legal_actions(state)}
        if action.action_id not in legal:
            raise ValueError(f"illegal duel action {action.action_id}")
        return DuelState(state.cards, state.history + (action.action_id,))

    def information_set(self, state: DuelState, *, player: int) -> InformationSet:
        if player not in (0, 1) or state.acting_player != player:
            raise ValueError("information set is available only to the acting player")
        own_card = state.cards[player]
        if own_card is None:
            raise ValueError("acting player must have a private card")
        history_label = "root" if not state.history else "-".join(state.history)
        identifier = f"p{player}:{own_card.label}:{history_label}"
        cached = self._information_set_cache.get(identifier)
        if cached is not None:
            return cached
        belief = self.public_belief(state, player=player)
        result = InformationSet(
            info_set_id=identifier,
            public_belief_fingerprint=belief.fingerprint,
            actor=InformationSetActor.SELF,
            legal_actions=self.legal_actions(state),
            depth=state.depth,
            private_observation_fingerprint=_fingerprint(
                {"player": player, "own_private_card": own_card.label}
            ),
        )
        self._information_set_cache[identifier] = result
        return result

    def public_belief(self, state: DuelState, *, player: int) -> PublicBeliefState:
        if player not in (0, 1) or state.cards[player] is None:
            raise ValueError("public belief requires a dealt player perspective")
        own_card = state.cards[player]
        assert own_card is not None
        possible = tuple(card for card in DuelCard if card != own_card)
        opponent_range = OpponentRange.normalized(
            tuple(
                RangeHypothesis(
                    hypothesis_id=f"opponent:{card.label}",
                    weight=1.0,
                    hidden_resource_counts=(("private_strength", int(card) + 1),),
                    intent_id=("push" if card is DuelCard.MEWTWO else "mixed"),
                    style_id="equilibrium_unknown",
                    source="exact_card-removal public belief",
                )
                for card in possible
            )
        )
        public_history = _fingerprint({"history": state.history})
        return PublicBeliefState(
            public_state_fingerprint=_fingerprint(
                {
                    "history": state.history,
                    "terminal": state.terminal,
                    "pot_commitment": self._pot_commitment(state.history),
                }
            ),
            public_history_fingerprint=public_history,
            turn=len(state.history),
            acting_player=InformationSetActor.SELF,
            opponent_range=opponent_range,
            public_features=(
                ("pot_commitment", self._pot_commitment(state.history)),
                ("push_count", float(state.history.count(PUSH))),
            ),
            updater_fingerprint=_fingerprint(
                {"game": self.fingerprint, "updater": "exact_card_removal"}
            ),
        )

    @staticmethod
    def _pot_commitment(history: tuple[str, ...]) -> float:
        return 2.0 if CONTEST in history else 1.0 if PUSH in history else 0.0

    def terminal_utility(self, state: DuelState, player: int) -> float:
        if not state.terminal or player not in (0, 1):
            raise ValueError("terminal utility requires a terminal state/player")
        if state.history == (PUSH, CONCEDE):
            value_player_0 = 1.0
        elif state.history == (HOLD, PUSH, CONCEDE):
            value_player_0 = -1.0
        else:
            assert state.cards[0] is not None and state.cards[1] is not None
            stake = 2.0 if CONTEST in state.history else 1.0
            value_player_0 = stake if state.cards[0] > state.cards[1] else -stake
        return value_player_0 if player == 0 else -value_player_0

    def all_states(self) -> tuple[DuelState, ...]:
        result: dict[str, DuelState] = {}

        def visit(state: DuelState) -> None:
            result[state.state_id] = state
            if state.terminal:
                return
            if state.cards == (None, None):
                for outcome in self.chance_outcomes(state):
                    visit(self.transition(state, None, chance_outcome=outcome))
                return
            for action in self.legal_actions(state):
                visit(self.transition(state, action))

        visit(self.root_state())
        return tuple(sorted(result.values(), key=lambda item: (item.depth, item.state_id)))

    def information_sets(self) -> tuple[InformationSet, ...]:
        self._ensure_information_sets()
        assert self._information_sets is not None
        return tuple(self._information_sets[key] for key in sorted(self._information_sets))

    def states_for_information_set(self, identifier: str) -> tuple[DuelState, ...]:
        self._ensure_information_sets()
        assert self._states_by_information_set is not None
        if identifier not in self._states_by_information_set:
            raise KeyError(f"unknown information set {identifier}")
        return self._states_by_information_set[identifier]

    def _ensure_information_sets(self) -> None:
        if self._information_sets is not None:
            return
        info_sets: dict[str, InformationSet] = {}
        states: dict[str, list[DuelState]] = {}
        for state in self.all_states():
            player = state.acting_player
            if player is None:
                continue
            info_set = self.information_set(state, player=player)
            existing = info_sets.get(info_set.info_set_id)
            if existing is not None and existing != info_set:
                raise ValueError("states in one information set disagree on public contract")
            info_sets[info_set.info_set_id] = info_set
            states.setdefault(info_set.info_set_id, []).append(state)
        self._information_sets = info_sets
        self._states_by_information_set = {
            key: tuple(sorted(value, key=lambda item: item.state_id))
            for key, value in states.items()
        }


def uniform_profile(game: RocketPrizeDuel) -> TabularStrategyProfile:
    return TabularStrategyProfile(
        strategies=tuple(
            (
                information_set.info_set_id,
                ActionDistribution(
                    tuple(
                        (action.action_id, 1.0 / len(information_set.legal_actions))
                        for action in information_set.legal_actions
                    )
                ),
            )
            for information_set in game.information_sets()
        ),
        label="uniform_validation_baseline",
    )


def analytic_equilibrium_profile(
    game: RocketPrizeDuel,
    *,
    alpha: float = 1.0 / 3.0,
) -> TabularStrategyProfile:
    """Return one exact Kuhn equilibrium from the alpha-parameterized family."""

    if not 0.0 <= alpha <= 1.0 / 3.0:
        raise ValueError("Kuhn equilibrium alpha must be in [0, 1/3]")
    strategies: dict[str, ActionDistribution] = {}
    for info_set in game.information_sets():
        _, card_label, history = info_set.info_set_id.split(":", 2)
        card = {value.label: value for value in DuelCard}[card_label]
        if history == "root":
            push = (
                alpha
                if card is DuelCard.MIMIKYU
                else 0.0
                if card is DuelCard.SPIDOPS
                else 3.0 * alpha
            )
            probabilities = {HOLD: 1.0 - push, PUSH: push}
        elif history == HOLD:
            push = (
                1.0 / 3.0
                if card is DuelCard.MIMIKYU
                else 0.0
                if card is DuelCard.SPIDOPS
                else 1.0
            )
            probabilities = {HOLD: 1.0 - push, PUSH: push}
        elif history == PUSH:
            contest = (
                0.0
                if card is DuelCard.MIMIKYU
                else 1.0 / 3.0
                if card is DuelCard.SPIDOPS
                else 1.0
            )
            probabilities = {CONCEDE: 1.0 - contest, CONTEST: contest}
        elif history == f"{HOLD}-{PUSH}":
            contest = (
                0.0
                if card is DuelCard.MIMIKYU
                else alpha + 1.0 / 3.0
                if card is DuelCard.SPIDOPS
                else 1.0
            )
            probabilities = {CONCEDE: 1.0 - contest, CONTEST: contest}
        else:
            raise ValueError(f"unexpected information-set history {history}")
        strategies[info_set.info_set_id] = ActionDistribution.normalized(probabilities)
    return TabularStrategyProfile(
        strategies=tuple(sorted(strategies.items())),
        label=f"analytic_equilibrium_alpha_{alpha:.12f}",
    )


def evaluate_profile_from_state(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
    state: DuelState,
    *,
    player: int = 0,
) -> float:
    if state.terminal:
        return game.terminal_utility(state, player)
    if state.cards == (None, None):
        return sum(
            outcome.probability
            * evaluate_profile_from_state(
                game,
                profile,
                game.transition(state, None, chance_outcome=outcome),
                player=player,
            )
            for outcome in game.chance_outcomes(state)
        )
    acting = state.acting_player
    assert acting is not None
    information_set = game.information_set(state, player=acting)
    strategy = profile.strategy(information_set)
    action_by_id = {action.action_id: action for action in information_set.legal_actions}
    return sum(
        probability
        * evaluate_profile_from_state(
            game,
            profile,
            game.transition(state, action_by_id[action_id]),
            player=player,
        )
        for action_id, probability in strategy.probabilities
    )


def evaluate_profile(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> float:
    return evaluate_profile_from_state(game, profile, game.root_state(), player=0)


def _player_information_sets(
    game: RocketPrizeDuel,
    player: int,
) -> tuple[InformationSet, ...]:
    return tuple(
        information_set
        for information_set in game.information_sets()
        if information_set.info_set_id.startswith(f"p{player}:")
    )


def exact_best_response_value(
    game: RocketPrizeDuel,
    opponent_profile: TabularStrategyProfile,
    *,
    player: int,
) -> float:
    """Enumerate all 2^6 pure responses, respecting information-set consistency."""

    if player not in (0, 1):
        raise ValueError("best-response player must be 0 or 1")
    information_sets = _player_information_sets(game, player)
    best = -math.inf
    for selected_indices in product(*(range(len(item.legal_actions)) for item in information_sets)):
        candidate = opponent_profile
        for information_set, selected in zip(
            information_sets, selected_indices, strict=True
        ):
            chosen = information_set.legal_actions[selected].action_id
            candidate = candidate.with_strategy(
                information_set.info_set_id,
                ActionDistribution(
                    tuple(
                        (action.action_id, 1.0 if action.action_id == chosen else 0.0)
                        for action in information_set.legal_actions
                    )
                ),
                label=f"best_response_player_{player}",
            )
        own_value = evaluate_profile_from_state(
            game, candidate, game.root_state(), player=player
        )
        best = max(best, own_value)
    return best


def exact_profile_metrics(
    game: RocketPrizeDuel,
    profile: TabularStrategyProfile,
) -> ExactProfileMetrics:
    value = evaluate_profile(game, profile)
    best_0 = exact_best_response_value(game, profile, player=0)
    best_1 = exact_best_response_value(game, profile, player=1)
    nash_conv = max(0.0, best_0 + best_1)
    return ExactProfileMetrics(
        expected_value_player_0=value,
        best_response_value_player_0=best_0,
        best_response_value_player_1=best_1,
        nash_conv=nash_conv,
        exploitability=nash_conv / 2.0,
        profile_fingerprint=profile.fingerprint,
    )


class TabularCFRSolver:
    """Deterministic full-tree vanilla CFR reference implementation."""

    def __init__(self, game: RocketPrizeDuel) -> None:
        self.game = game
        self.information_sets = {
            info.info_set_id: info for info in game.information_sets()
        }
        self.regrets = {
            identifier: {action.action_id: 0.0 for action in info.legal_actions}
            for identifier, info in self.information_sets.items()
        }
        self.strategy_sums = {
            identifier: {action.action_id: 0.0 for action in info.legal_actions}
            for identifier, info in self.information_sets.items()
        }

    def train(self, iterations: int) -> CFRTrainingResult:
        if iterations <= 0:
            raise ValueError("CFR iterations must be positive")
        for _ in range(iterations):
            self._cfr(
                self.game.root_state(),
                reach_player_0=1.0,
                reach_player_1=1.0,
                chance_reach=1.0,
            )
        profile = self.average_profile(label=f"vanilla_cfr_{iterations}_iterations")
        maximum_regret = max(
            max(0.0, value)
            for regrets in self.regrets.values()
            for value in regrets.values()
        )
        solver_fingerprint = _fingerprint(
            {
                "algorithm": "full_tree_vanilla_cfr",
                "game": self.game.fingerprint,
                "iterations": iterations,
                "regrets": self.regrets,
                "strategy_sums": self.strategy_sums,
            }
        )
        return CFRTrainingResult(
            iterations=iterations,
            profile=profile,
            metrics=exact_profile_metrics(self.game, profile),
            maximum_positive_regret=maximum_regret,
            solver_fingerprint=solver_fingerprint,
        )

    def _strategy(self, identifier: str) -> dict[str, float]:
        positive = {
            action: max(0.0, regret)
            for action, regret in self.regrets[identifier].items()
        }
        total = sum(positive.values())
        if total > 0:
            return {action: value / total for action, value in positive.items()}
        probability = 1.0 / len(positive)
        return {action: probability for action in positive}

    def _cfr(
        self,
        state: DuelState,
        *,
        reach_player_0: float,
        reach_player_1: float,
        chance_reach: float,
    ) -> float:
        if state.terminal:
            return self.game.terminal_utility(state, 0)
        if state.cards == (None, None):
            value = 0.0
            for outcome in self.game.chance_outcomes(state):
                value += outcome.probability * self._cfr(
                    self.game.transition(state, None, chance_outcome=outcome),
                    reach_player_0=reach_player_0,
                    reach_player_1=reach_player_1,
                    chance_reach=chance_reach * outcome.probability,
                )
            return value
        player = state.acting_player
        assert player is not None
        info_set = self.game.information_set(state, player=player)
        identifier = info_set.info_set_id
        strategy = self._strategy(identifier)
        own_reach = reach_player_0 if player == 0 else reach_player_1
        for action_id, probability in strategy.items():
            self.strategy_sums[identifier][action_id] += (
                chance_reach * own_reach * probability
            )
        actions = {action.action_id: action for action in info_set.legal_actions}
        values: dict[str, float] = {}
        for action_id, probability in strategy.items():
            values[action_id] = self._cfr(
                self.game.transition(state, actions[action_id]),
                reach_player_0=(
                    reach_player_0 * probability if player == 0 else reach_player_0
                ),
                reach_player_1=(
                    reach_player_1 * probability if player == 1 else reach_player_1
                ),
                chance_reach=chance_reach,
            )
        node_value = sum(strategy[action] * value for action, value in values.items())
        counterfactual_reach = chance_reach * (
            reach_player_1 if player == 0 else reach_player_0
        )
        for action_id, action_value in values.items():
            regret = (
                action_value - node_value if player == 0 else node_value - action_value
            )
            self.regrets[identifier][action_id] += counterfactual_reach * regret
        return node_value

    def average_profile(self, *, label: str = "cfr_average") -> TabularStrategyProfile:
        strategies: list[tuple[str, ActionDistribution]] = []
        for identifier in sorted(self.information_sets):
            totals = self.strategy_sums[identifier]
            total = sum(totals.values())
            probabilities = (
                {action: value / total for action, value in totals.items()}
                if total > 0
                else {action: 1.0 / len(totals) for action in totals}
            )
            strategies.append((identifier, ActionDistribution.normalized(probabilities)))
        return TabularStrategyProfile(tuple(strategies), label)

    def normalized_regret_targets(self) -> dict[str, dict[str, float]]:
        """Return scale-normalized cumulative regrets for neural approximation."""

        result: dict[str, dict[str, float]] = {}
        for identifier, regrets in self.regrets.items():
            scale = max(1e-12, max(abs(value) for value in regrets.values()))
            result[identifier] = {
                action: value / scale for action, value in regrets.items()
            }
        return result


class ExactTabularBlueprintProvider:
    """Solved tabular provider for the microgame only; it is not a neural model."""

    def __init__(
        self,
        game: RocketPrizeDuel,
        profile: TabularStrategyProfile,
        *,
        deck_fingerprint: str,
        algorithm: str,
        training_steps: int,
    ) -> None:
        _validate_digest(deck_fingerprint, "deck_fingerprint")
        if training_steps <= 0:
            raise ValueError("solved tabular provider requires positive training steps")
        self.game = game
        self.profile = profile
        self._checkpoint = BlueprintCheckpoint(
            provider_name="exact_tabular_validation_blueprint",
            algorithm=algorithm,
            model_fingerprint=profile.fingerprint,
            game_fingerprint=game.fingerprint,
            deck_fingerprint=deck_fingerprint,
            training_steps=training_steps,
            trained=True,
            evaluation_notes=(
                "Solved/evaluated only on RocketPrizeDuel; tabular, non-neural, "
                "abstract, and not transferable evidence for official Pokemon."
            ),
        )

    @property
    def checkpoint(self) -> BlueprintCheckpoint:
        return self._checkpoint

    def strategy(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
    ) -> ActionDistribution:
        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("information set and public belief fingerprints disagree")
        return self.profile.strategy(information_set)

    def counterfactual_values(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        *,
        player: int,
    ) -> CounterfactualValues:
        states = self.game.states_for_information_set(information_set.info_set_id)
        state_by_hypothesis: dict[str, DuelState] = {}
        acting_player = states[0].acting_player
        assert acting_player is not None
        for state in states:
            opponent_card = state.cards[1 - acting_player]
            assert opponent_card is not None
            state_by_hypothesis[f"opponent:{opponent_card.label}"] = state
        values: list[tuple[str, float]] = []
        for hypothesis in belief.opponent_range.hypotheses:
            if hypothesis.hypothesis_id not in state_by_hypothesis:
                raise ValueError("belief hypothesis is not represented in exact states")
            values.append(
                (
                    hypothesis.hypothesis_id,
                    evaluate_profile_from_state(
                        self.game,
                        self.profile,
                        state_by_hypothesis[hypothesis.hypothesis_id],
                        player=player,
                    ),
                )
            )
        return CounterfactualValues(
            player=player,
            values=tuple(values),
            source_fingerprint=self.profile.fingerprint,
        )


def exact_counterfactual_bounds(
    provider: ExactTabularBlueprintProvider,
    information_set: InformationSet,
    belief: PublicBeliefState,
    *,
    opponent_player: int,
    half_width: float = 0.0,
) -> CounterfactualValueBounds:
    if half_width < 0 or not math.isfinite(half_width):
        raise ValueError("counterfactual half-width must be finite and non-negative")
    baseline = provider.strategy(information_set, belief)
    values = provider.counterfactual_values(
        information_set, belief, player=opponent_player
    )
    return CounterfactualValueBounds(
        opponent_player=opponent_player,
        bounds=tuple(
            HypothesisValueBound(
                hypothesis_id=identifier,
                lower=value - half_width,
                upper=value + half_width,
                baseline_value=value,
            )
            for identifier, value in values.values
        ),
        confidence=1.0,
        baseline_strategy_fingerprint=baseline.fingerprint,
        source="exact RocketPrizeDuel conditional enumeration",
    )


class ExactGridSafeResolver:
    """Exact microgame local resolver with per-hypothesis opponent upper bounds."""

    name = "exact_grid_counterfactual_bounded_microgame_resolver"

    def __init__(
        self,
        base_profile: TabularStrategyProfile,
        *,
        grid_steps: int = 100,
    ) -> None:
        if grid_steps <= 0:
            raise ValueError("resolver grid_steps must be positive")
        self.base_profile = base_profile
        self.grid_steps = grid_steps

    def resolve(
        self,
        game: RocketPrizeDuel,
        request: SafeResolveRequest,
    ) -> ResolvedStrategy:
        if not isinstance(game, RocketPrizeDuel):
            raise TypeError("exact grid resolver supports RocketPrizeDuel only")
        baseline = self.base_profile.strategy(request.information_set)
        if baseline != request.baseline_strategy:
            raise ValueError("resolver base profile does not match request baseline")
        states = game.states_for_information_set(request.information_set.info_set_id)
        actor = states[0].acting_player
        if actor is None or any(state.acting_player != actor for state in states):
            raise ValueError("resolver root must be one player's information set")
        opponent = 1 - actor
        if request.opponent_value_bounds.opponent_player != opponent:
            raise ValueError("counterfactual bounds target the wrong opponent")
        actions = request.information_set.legal_actions
        if len(actions) != 2:
            raise ValueError("exact grid resolver currently requires two actions")
        state_by_hypothesis: dict[str, DuelState] = {}
        for state in states:
            card = state.cards[opponent]
            assert card is not None
            state_by_hypothesis[f"opponent:{card.label}"] = state
        weights = {
            item.hypothesis_id: item.weight
            for item in request.belief.opponent_range.hypotheses
        }
        bounds = {
            item.hypothesis_id: item for item in request.opponent_value_bounds.bounds
        }
        candidate_probabilities = {index / self.grid_steps for index in range(self.grid_steps + 1)}
        candidate_probabilities.add(baseline.probability(actions[0].action_id))
        start = perf_counter()
        best: tuple[float, float, ActionDistribution, float] | None = None
        evaluated_nodes = 0
        for first_probability in sorted(candidate_probabilities):
            strategy = ActionDistribution(
                (
                    (actions[0].action_id, first_probability),
                    (actions[1].action_id, 1.0 - first_probability),
                )
            )
            candidate = self.base_profile.with_strategy(
                request.information_set.info_set_id,
                strategy,
                label="exact_grid_resolve_candidate",
            )
            maximum_violation = 0.0
            actor_value = 0.0
            for hypothesis_id, state in state_by_hypothesis.items():
                evaluated_nodes += 1
                opponent_value = evaluate_profile_from_state(
                    game, candidate, state, player=opponent
                )
                maximum_violation = max(
                    maximum_violation,
                    opponent_value - bounds[hypothesis_id].upper,
                )
                actor_value += weights[hypothesis_id] * evaluate_profile_from_state(
                    game, candidate, state, player=actor
                )
            maximum_violation = max(0.0, maximum_violation)
            if maximum_violation > request.budget.safety_tolerance + 1e-12:
                continue
            distance = sum(
                abs(strategy.probability(action.action_id) - baseline.probability(action.action_id))
                for action in actions
            )
            score = (actor_value, -distance, strategy, maximum_violation)
            if best is None or score[:2] > best[:2]:
                best = score
        if best is None:
            raise RuntimeError("no safe candidate found; baseline bound invariant is broken")
        actor_value, _, strategy, maximum_violation = best
        del actor_value
        if sum(
            abs(
                strategy.probability(action.action_id)
                - baseline.probability(action.action_id)
            )
            for action in actions
        ) <= 1e-12:
            strategy = baseline
        elapsed_ms = (perf_counter() - start) * 1000.0
        certificate = SafetyCertificate(
            safe=maximum_violation <= request.budget.safety_tolerance,
            max_counterfactual_value_violation=maximum_violation,
            tolerance=request.budget.safety_tolerance,
            confidence=request.opponent_value_bounds.confidence,
            bound_fingerprint=request.opponent_value_bounds.fingerprint,
            method="exact_per_hypothesis_conditional_upper_bound_enumeration",
        )
        return ResolvedStrategy(
            strategy=strategy,
            baseline_strategy=baseline,
            certificate=certificate,
            iterations=len(candidate_probabilities),
            expanded_nodes=evaluated_nodes,
            elapsed_ms=elapsed_ms,
            fallback_to_blueprint=strategy == baseline,
            resolver_name=self.name,
        )


def profile_l1_distance(
    first: TabularStrategyProfile,
    second: TabularStrategyProfile,
) -> float:
    first_values = dict(first.strategies)
    second_values = dict(second.strategies)
    if set(first_values) != set(second_values):
        raise ValueError("profile supports differ")
    return sum(
        abs(first_values[identifier].probability(action_id) - probability)
        for identifier, distribution in second_values.items()
        for action_id, probability in distribution.probabilities
    )


def information_set_action_count(game: RocketPrizeDuel) -> int:
    return sum(len(item.legal_actions) for item in game.information_sets())


def terminal_state_count(game: RocketPrizeDuel) -> int:
    return sum(state.terminal for state in game.all_states())


def deal_count(game: RocketPrizeDuel) -> int:
    return len(game.chance_outcomes(game.root_state()))
