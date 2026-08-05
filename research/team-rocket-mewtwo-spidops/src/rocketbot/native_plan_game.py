"""Turn-level strategic-plan abstraction over genuine native descendants.

The licensed engine asks for many procedural selections inside one turn.  CFR
over every such prompt creates long horizons, huge importance weights, and a
strategy whose unit of choice is an option index rather than a game plan.  This
module supplies a deliberately explicit abstraction: at the first ``MAIN``
prompt of a turn, the acting player selects one of the project's five persistent
strategic plans.  A fixed tactical executor then maps that plan to legal native
selections until the turn changes, a chance node is encountered, or the game
terminates.

Every descendant selection is executed by the official native search API, but
the plan actions, tactical executor, hidden completion, and turn-level game are
research abstractions.  A plan action is not a single official option and must
never be described as an official engine action.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from .competition import OptionType, SelectContext, StrategicPlan
from .game_theory import (
    AbstractAction,
    ActionDistribution,
    ChanceOutcome,
    GameBoundaryKind,
    GameBoundaryProvenance,
    InformationSet,
    InformationSetActor,
    PublicBeliefState,
)
from .native_blueprint_features import NativeBlueprintFeatureBridge
from .native_regret_blueprint import NativeRegretBlueprintProvider
from .official_search_game import OfficialNativeSearchGame, OfficialSearchGameState


NATIVE_PLAN_GAME_SCHEMA_VERSION = 1
NATIVE_PLAN_ACTION_FEATURE_NAMES = tuple(plan.value for plan in StrategicPlan)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fingerprint(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _integer(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _selection(state: OfficialSearchGameState) -> Mapping[str, Any]:
    value = state.observation.get("select")
    if not isinstance(value, Mapping):
        raise ValueError("native plan state lacks a selection")
    return value


def _is_main(state: OfficialSearchGameState) -> bool:
    if state.terminal or state.actor is InformationSetActor.CHANCE:
        return False
    return _integer(_selection(state).get("context")) == int(SelectContext.MAIN)


def _plan_action(plan: StrategicPlan) -> AbstractAction:
    return AbstractAction(
        action_id=f"plan:{plan.value}",
        label=f"Commit turn to {plan.value.replace('_', ' ')}",
        official_option_indices=(),
        features=tuple(
            (name, 1.0 if name == plan.value else 0.0)
            for name in NATIVE_PLAN_ACTION_FEATURE_NAMES
        ),
    )


NATIVE_PLAN_ACTIONS = tuple(_plan_action(plan) for plan in StrategicPlan)
_PLAN_BY_ACTION = {
    action.action_id: plan
    for action, plan in zip(NATIVE_PLAN_ACTIONS, StrategicPlan, strict=True)
}


def heuristic_plan_strategy(
    information_set: InformationSet,
    belief: PublicBeliefState,
) -> ActionDistribution:
    """Full-support turn-plan behavior policy used only for data collection."""

    if information_set.public_belief_fingerprint != belief.fingerprint:
        raise ValueError("heuristic plan strategy received mismatched belief")
    if tuple(action.action_id for action in information_set.legal_actions) != tuple(
        action.action_id for action in NATIVE_PLAN_ACTIONS
    ):
        raise ValueError("heuristic plan strategy requires the five plan actions")
    features = dict(belief.public_features)
    own_board = features.get("own_board_count", 0.0)
    own_hp = features.get("own_board_hp", 0.0)
    opponent_hp = features.get("opponent_board_hp", 0.0)
    own_prizes = features.get("own_prize_count", 6.0)
    turn = int(features.get("turn", 0.0))
    if own_prizes <= 2.0:
        preferred = StrategicPlan.PRIZE_PUSH
    elif own_board < 3.0 or turn <= 2:
        preferred = StrategicPlan.BOARD_SETUP
    elif own_hp + 60.0 < opponent_hp:
        preferred = StrategicPlan.RECOVERY
    elif turn % 3 == 0:
        preferred = StrategicPlan.ENERGY_ENGINE
    else:
        preferred = StrategicPlan.PRIZE_PUSH
    values = {
        action.action_id: (
            0.84 if _PLAN_BY_ACTION[action.action_id] is preferred else 0.04
        )
        for action in information_set.legal_actions
    }
    result = ActionDistribution(tuple(values.items()))
    result.validate_support(information_set)
    return result


_PLAN_TYPE_BIASES: Mapping[StrategicPlan, Mapping[int, float]] = {
    StrategicPlan.BOARD_SETUP: {
        int(OptionType.PLAY): 2.4,
        int(OptionType.EVOLVE): 3.0,
        int(OptionType.ABILITY): 1.5,
        int(OptionType.ATTACH): 0.8,
        int(OptionType.ATTACK): -1.0,
        int(OptionType.END): -2.0,
    },
    StrategicPlan.ENERGY_ENGINE: {
        int(OptionType.ATTACH): 3.2,
        int(OptionType.PLAY): 1.2,
        int(OptionType.ABILITY): 0.8,
        int(OptionType.END): -2.0,
    },
    StrategicPlan.PRIZE_PUSH: {
        int(OptionType.ATTACK): 4.0,
        int(OptionType.ABILITY): 0.8,
        int(OptionType.PLAY): 0.4,
        int(OptionType.RETREAT): -0.8,
        int(OptionType.END): -1.0,
    },
    StrategicPlan.RECOVERY: {
        int(OptionType.PLAY): 2.6,
        int(OptionType.RETREAT): 2.0,
        int(OptionType.ABILITY): 1.2,
        int(OptionType.ATTACK): -0.3,
    },
    StrategicPlan.CONTROL: {
        int(OptionType.ABILITY): 2.8,
        int(OptionType.PLAY): 1.8,
        int(OptionType.RETREAT): 0.8,
        int(OptionType.ATTACK): 0.5,
    },
}


class NativePlanMacroBudgetExceeded(RuntimeError):
    """Raised when the tactical executor cannot reach the next boundary."""


@dataclass(frozen=True, slots=True)
class NativePlanGameState:
    game_fingerprint: str
    native_state: OfficialSearchGameState
    macro_depth: int
    pending_plan: StrategicPlan | None
    plan_player: int | None
    tactical_transitions: int

    def __post_init__(self) -> None:
        if self.macro_depth < 0 or self.tactical_transitions < 0:
            raise ValueError("native plan state depth/accounting is invalid")
        if (self.pending_plan is None) != (self.plan_player is None):
            raise ValueError("native plan state pending-plan metadata disagrees")
        if self.plan_player not in (None, 0, 1):
            raise ValueError("native plan state player is invalid")
        if self.native_state.terminal and self.pending_plan is not None:
            raise ValueError("terminal native plan state cannot retain a plan")
        if (
            not self.native_state.terminal
            and self.native_state.actor is not InformationSetActor.CHANCE
            and not _is_main(self.native_state)
        ):
            raise ValueError("native plan state must stop at chance/main/terminal")

    @property
    def state_id(self) -> str:
        return _fingerprint(
            {
                "game": self.game_fingerprint,
                "native": self.native_state.state_id,
                "macro_depth": self.macro_depth,
                "pending_plan": (
                    None if self.pending_plan is None else self.pending_plan.value
                ),
                "plan_player": self.plan_player,
            }
        )

    @property
    def actor(self) -> InformationSetActor:
        if self.native_state.terminal:
            return InformationSetActor.TERMINAL
        if self.native_state.actor is InformationSetActor.CHANCE:
            return InformationSetActor.CHANCE
        player = self.native_state.player_index
        if player not in (0, 1):
            raise ValueError("native plan decision lacks player")
        return InformationSetActor.SELF if player == 0 else InformationSetActor.OPPONENT

    @property
    def depth(self) -> int:
        return self.macro_depth

    @property
    def terminal(self) -> bool:
        return self.native_state.terminal


class OfficialNativePlanGame:
    """Turn-level plan game backed by genuine native search transitions."""

    def __init__(
        self,
        native_game: OfficialNativeSearchGame,
        scenario_root: OfficialSearchGameState,
        *,
        bridge: NativeBlueprintFeatureBridge,
        tactical_provider: NativeRegretBlueprintProvider,
        maximum_tactical_transitions: int = 512,
    ) -> None:
        if maximum_tactical_transitions <= 0:
            raise ValueError("native plan tactical budget must be positive")
        if scenario_root.depth <= 0 or scenario_root.terminal:
            raise ValueError("native plan game requires a selected nonterminal completion")
        self.native_game = native_game
        self.bridge = bridge
        self.tactical_provider = tactical_provider
        self.maximum_tactical_transitions = int(maximum_tactical_transitions)
        self._fingerprint = _fingerprint(
            {
                "schema": NATIVE_PLAN_GAME_SCHEMA_VERSION,
                "native_game": native_game.fingerprint,
                "scenario_hypothesis": scenario_root.hypothesis_fingerprint,
                "feature_bridge": bridge.fingerprint,
                "tactical_provider": tactical_provider.artifact.fingerprint,
                "plans": [action.action_id for action in NATIVE_PLAN_ACTIONS],
                "biases": {
                    plan.value: dict(values)
                    for plan, values in _PLAN_TYPE_BIASES.items()
                },
                "maximum_tactical_transitions": self.maximum_tactical_transitions,
            }
        )
        self._provenance = GameBoundaryProvenance(
            name="official_native_turn_plan_abstraction",
            kind=GameBoundaryKind.ABSTRACT_INFORMATION_SET_GAME,
            schema_version=NATIVE_PLAN_GAME_SCHEMA_VERSION,
            ruleset_fingerprint=native_game.provenance.ruleset_fingerprint,
            official_engine_executed=False,
            fidelity_notes=(
                "Plan actions and tactical auto-resolution are research abstractions; "
                "each descendant selection itself executes the licensed native search API."
            ),
        )
        self._macro_transition_count = 0
        self._tactical_transition_count = 0
        self._acting_value_by_information_set: dict[str, float] = {}
        self._root = self._advance(
            scenario_root,
            macro_depth=0,
            pending_plan=None,
            plan_player=None,
            tactical_transitions=0,
        )

    @property
    def provenance(self) -> GameBoundaryProvenance:
        return self._provenance

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def source_official_engine_executed(self) -> bool:
        return True

    @property
    def hidden_completion_prior_official(self) -> bool:
        return False

    @property
    def macro_transition_count(self) -> int:
        return self._macro_transition_count

    @property
    def tactical_transition_count(self) -> int:
        return self._tactical_transition_count

    def root_state(self) -> NativePlanGameState:
        return self._root

    def player_to_act(self, state: NativePlanGameState) -> int | None:
        self._validate_state(state)
        if state.terminal or state.actor is InformationSetActor.CHANCE:
            return None
        return self.native_game.player_to_act(state.native_state)

    def public_belief(
        self,
        state: NativePlanGameState,
        *,
        player: int,
    ) -> PublicBeliefState:
        self._validate_decision(state, player)
        return self.bridge.view(self.native_game, state.native_state).public_belief

    def information_set(
        self,
        state: NativePlanGameState,
        *,
        player: int,
    ) -> InformationSet:
        self._validate_decision(state, player)
        view = self.bridge.view(self.native_game, state.native_state)
        info_set_id = _fingerprint(
                {
                    "schema": NATIVE_PLAN_GAME_SCHEMA_VERSION,
                    "native_information_set": view.information_set.info_set_id,
                    "plan_support": [action.action_id for action in NATIVE_PLAN_ACTIONS],
                }
            )
        prediction = self.tactical_provider.predict(
            view.information_set,
            view.public_belief,
        )
        self._acting_value_by_information_set[info_set_id] = prediction.state_value
        return InformationSet(
            info_set_id=info_set_id,
            public_belief_fingerprint=view.public_belief.fingerprint,
            actor=state.actor,
            legal_actions=NATIVE_PLAN_ACTIONS,
            depth=state.depth,
            terminal=False,
            private_observation_fingerprint=(
                view.information_set.private_observation_fingerprint
            ),
            private_features=view.information_set.private_features,
        )

    def plan_value_baseline(
        self,
        information_set: InformationSet,
        belief: PublicBeliefState,
        value_player: int,
        acting_player: int,
    ) -> dict[str, float]:
        """Pre-action scalar bootstrap expanded over plan actions.

        The scalar comes from the Phase 6K tactical candidate and is transformed
        to the traverser's utility perspective.  It is a variance-reduction
        baseline only, never a calibrated counterfactual value.
        """

        if information_set.public_belief_fingerprint != belief.fingerprint:
            raise ValueError("native plan baseline received mismatched belief")
        if value_player not in (0, 1) or acting_player not in (0, 1):
            raise ValueError("native plan baseline players are invalid")
        try:
            acting_value = self._acting_value_by_information_set[
                information_set.info_set_id
            ]
        except KeyError as exc:
            raise ValueError("native plan baseline has no registered information set") from exc
        value = acting_value if value_player == acting_player else -acting_value
        return {action.action_id: value for action in information_set.legal_actions}

    def legal_actions(
        self,
        state: NativePlanGameState,
    ) -> tuple[AbstractAction, ...]:
        self._validate_state(state)
        if state.terminal or state.actor is InformationSetActor.CHANCE:
            return ()
        return NATIVE_PLAN_ACTIONS

    def chance_outcomes(
        self,
        state: NativePlanGameState,
    ) -> tuple[ChanceOutcome, ...]:
        self._validate_state(state)
        if state.actor is not InformationSetActor.CHANCE:
            return ()
        return self.native_game.chance_outcomes(state.native_state)

    def transition(
        self,
        state: NativePlanGameState,
        action: AbstractAction | None,
        *,
        chance_outcome: ChanceOutcome | None = None,
    ) -> NativePlanGameState:
        self._validate_state(state)
        if state.terminal:
            raise ValueError("cannot transition a terminal native plan state")
        self._macro_transition_count += 1
        if state.actor is InformationSetActor.CHANCE:
            if action is not None or chance_outcome is None:
                raise ValueError("native plan chance transition requires only an outcome")
            child = self.native_game.transition(
                state.native_state,
                None,
                chance_outcome=chance_outcome,
            )
            return self._advance(
                child,
                macro_depth=state.macro_depth + 1,
                pending_plan=state.pending_plan,
                plan_player=state.plan_player,
                tactical_transitions=state.tactical_transitions,
            )
        if action is None or chance_outcome is not None:
            raise ValueError("native plan decision requires one plan action")
        plan = _PLAN_BY_ACTION.get(action.action_id)
        if plan is None or action != _plan_action(plan):
            raise ValueError("unknown or tampered native plan action")
        player = self.native_game.player_to_act(state.native_state)
        if player not in (0, 1):
            raise ValueError("native plan decision lacks acting player")
        return self._advance(
            state.native_state,
            macro_depth=state.macro_depth + 1,
            pending_plan=plan,
            plan_player=player,
            tactical_transitions=state.tactical_transitions,
        )

    def terminal_utility(self, state: NativePlanGameState, player: int) -> float:
        self._validate_state(state)
        return self.native_game.terminal_utility(state.native_state, player)

    def _validate_state(self, state: NativePlanGameState) -> None:
        if state.game_fingerprint != self.fingerprint:
            raise ValueError("native plan state belongs to another game")

    def _validate_decision(self, state: NativePlanGameState, player: int) -> None:
        self._validate_state(state)
        if player not in (0, 1) or self.player_to_act(state) != player:
            raise ValueError("native plan information view has wrong player")

    def _raw_action(
        self,
        state: OfficialSearchGameState,
        action_id: str,
        official_indices: tuple[int, ...],
    ) -> AbstractAction:
        matches = [
            action
            for action in self.native_game.legal_actions(state)
            if action.action_id == action_id
            and action.official_option_indices == official_indices
        ]
        if len(matches) != 1:
            raise RuntimeError("native tactical action did not map exactly once")
        return matches[0]

    def _option_types(
        self,
        state: OfficialSearchGameState,
        indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        options = _selection(state).get("option", ())
        if not isinstance(options, list):
            return ()
        result = []
        for index in indices:
            if 0 <= index < len(options) and isinstance(options[index], Mapping):
                result.append(_integer(options[index].get("type")))
        return tuple(result)

    def _tactical_action(
        self,
        state: OfficialSearchGameState,
        plan: StrategicPlan | None,
    ) -> AbstractAction:
        view = self.bridge.view(self.native_game, state)
        distribution = self.tactical_provider.strategy(
            view.information_set,
            view.public_belief,
        )
        main = _is_main(state)
        scores = []
        for enriched in view.information_set.legal_actions:
            probability = distribution.probability(enriched.action_id)
            score = math.log(max(probability, 1e-15))
            if plan is not None and main:
                biases = _PLAN_TYPE_BIASES[plan]
                types = self._option_types(state, enriched.official_option_indices)
                if not types and not enriched.official_option_indices:
                    score -= 0.3
                score += sum(biases.get(value, 0.0) for value in types)
            scores.append((score, enriched.action_id, enriched))
        _, _, selected = max(scores, key=lambda item: (item[0], item[1]))
        return self._raw_action(
            state,
            selected.action_id,
            selected.official_option_indices,
        )

    def _advance(
        self,
        native_state: OfficialSearchGameState,
        *,
        macro_depth: int,
        pending_plan: StrategicPlan | None,
        plan_player: int | None,
        tactical_transitions: int,
    ) -> NativePlanGameState:
        consumed = 0
        while True:
            if native_state.terminal:
                return NativePlanGameState(
                    self.fingerprint,
                    native_state,
                    macro_depth,
                    None,
                    None,
                    tactical_transitions + consumed,
                )
            if native_state.actor is InformationSetActor.CHANCE:
                return NativePlanGameState(
                    self.fingerprint,
                    native_state,
                    macro_depth,
                    pending_plan,
                    plan_player,
                    tactical_transitions + consumed,
                )
            player = self.native_game.player_to_act(native_state)
            if player not in (0, 1):
                raise RuntimeError("native tactical state lacks player")
            if pending_plan is not None and player != plan_player:
                pending_plan = None
                plan_player = None
            if pending_plan is None and _is_main(native_state):
                return NativePlanGameState(
                    self.fingerprint,
                    native_state,
                    macro_depth,
                    None,
                    None,
                    tactical_transitions + consumed,
                )
            if consumed >= self.maximum_tactical_transitions:
                raise NativePlanMacroBudgetExceeded(
                    "tactical executor did not reach chance/turn/terminal boundary"
                )
            selected = self._tactical_action(native_state, pending_plan)
            native_state = self.native_game.transition(native_state, selected)
            consumed += 1
            self._tactical_transition_count += 1
